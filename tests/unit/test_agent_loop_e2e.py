"""End-to-end agent-loop integration tests.

Full chain — real ``Session``, real ``ContextManager``, real ``ToolRouter``,
real ``submission_loop`` / ``Handlers`` from ``agent.core`` — with only the
network boundary stubbed: ``litellm.acompletion`` is monkeypatched with
scripted ``ModelResponse`` objects, and the ToolRouter's OpenAPI-spec fetch
plus MLflow tracing init are no-op'd so everything runs offline with no
Databricks workspace.

Covers:
  1. user input → LLM requests an approval-gated tool → approval granted →
     tool executes → LLM final answer → turn completes (via submission_loop).
  2. approval REJECTED → loop continues gracefully to a final answer, no stall.
  3. LLM emits malformed tool-call JSON mid-loop → loop self-corrects
     (error tool-result fed back, next iteration retries) instead of stalling.
  4. pending-approval tool call with malformed JSON → exec_approval refuses
     to execute it, sanitises the args, and the loop continues (post-patch).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from litellm import ModelResponse

from agent.config import Config
from agent.core import agent_loop
from agent.core.agent_loop import Handlers, submission_loop
from agent.core.session import OpType, Session
from agent.core.tools import ToolRouter, ToolSpec

# ── scripted LLM responses ──────────────────────────────────────────────


def _text_response(content: str) -> ModelResponse:
    return ModelResponse(**{
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    })


def _tool_call_response(
    name: str, arguments: Any, call_id: str = "call_1",
) -> ModelResponse:
    """``arguments`` may be a dict (serialized) or a raw — possibly
    malformed — string passed through verbatim."""
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ModelResponse(**{
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": raw},
                }],
            },
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    })


def _script_llm(monkeypatch, responses: list[ModelResponse]) -> list[list]:
    """Stub ``litellm.acompletion`` as imported by agent_loop. Returns the
    list of message snapshots, one per LLM call, for history assertions."""
    calls: list[list] = []

    async def fake_acompletion(*, messages, **kwargs):
        calls.append(list(messages))
        if len(calls) > len(responses):
            raise AssertionError(
                f"LLM called {len(calls)} times but only "
                f"{len(responses)} responses were scripted"
            )
        return responses[len(calls) - 1]

    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    return calls


# ── real router with stub handlers at the workspace boundary ────────────


def _make_router(monkeypatch) -> tuple[ToolRouter, list]:
    """Real ToolRouter (all builtin specs registered) with the two tools the
    tests drive re-pointed at offline stub handlers. ``hf_to_uc`` is
    approval-gated by ``_needs_approval``; ``echo_stub`` is not."""
    executed: list = []

    async def stub_ingest(args, session=None, tool_call_id=None):
        executed.append(("hf_to_uc", args, tool_call_id))
        return "ingested ok", True

    async def stub_echo(args, session=None, tool_call_id=None):
        executed.append(("echo_stub", args, tool_call_id))
        return f"echo: {args.get('text', '')}", True

    router = ToolRouter({})
    router.register_tool(ToolSpec(
        name="hf_to_uc",
        description="stub ingest (approval-gated by name)",
        parameters={"type": "object", "properties": {}},
        handler=stub_ingest,
    ))
    router.register_tool(ToolSpec(
        name="echo_stub",
        description="stub echo (no approval needed)",
        parameters={"type": "object", "properties": {}},
        handler=stub_echo,
    ))

    # Network boundary: the OpenAPI spec fetch inside __aenter__.
    async def _no_openapi():
        return None

    monkeypatch.setattr(router, "register_openapi_tool", _no_openapi)
    return router, executed


def _make_config() -> Config:
    return Config.model_validate(
        {"model_name": "openai/test", "save_sessions": False}
    )


def _drain(event_queue: asyncio.Queue) -> list:
    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())
    return events


def _types(events) -> list[str]:
    return [e.event_type for e in events]


async def _collect_until(event_queue: asyncio.Queue, event_type: str, timeout=15):
    """Collect events until ``event_type`` arrives. Bounded wait per event
    so a stalled loop fails the test instead of hanging it."""
    events = []
    while True:
        event = await asyncio.wait_for(event_queue.get(), timeout=timeout)
        events.append(event)
        if event.event_type == event_type:
            return events


# ── submission plumbing (mirrors agent/main.py + backend/session_manager) ──


@dataclass
class _Operation:
    op_type: OpType
    data: dict | None = None


@dataclass
class _Submission:
    operation: _Operation
    id: str = "sub_test"


def _quiet_tracing(monkeypatch):
    monkeypatch.setattr(agent_loop.tracing, "init_tracing", lambda *_: False)


# ── 1. full chain: input → approval-gated tool → approve → final answer ──


@pytest.mark.asyncio
async def test_full_chain_tool_approval_granted(monkeypatch):
    _quiet_tracing(monkeypatch)
    calls = _script_llm(monkeypatch, [
        _tool_call_response("hf_to_uc", {"operation": "ingest_dataset", "repo_id": "squad"}),
        _text_response("Ingestion complete."),
    ])
    router, executed = _make_router(monkeypatch)

    submission_queue: asyncio.Queue = asyncio.Queue()
    event_queue: asyncio.Queue = asyncio.Queue()
    holder: list = [None]
    loop_task = asyncio.create_task(submission_loop(
        submission_queue, event_queue,
        config=_make_config(), tool_router=router,
        session_holder=holder, stream=False,
    ))

    try:
        # Turn 1: user input → tool call needing approval → loop pauses.
        await submission_queue.put(
            _Submission(_Operation(OpType.USER_INPUT, {"text": "ingest squad into UC"}))
        )
        pre_approval = await _collect_until(event_queue, "approval_required")
        types = _types(pre_approval)
        assert "ready" in types
        assert "processing" in types
        assert "turn_complete" not in types  # paused, not finished
        assert executed == []  # nothing ran before approval

        approval_event = pre_approval[-1]
        tools_data = approval_event.data["tools"]
        assert [t["tool"] for t in tools_data] == ["hf_to_uc"]
        tool_call_id = tools_data[0]["tool_call_id"]

        session = holder[0]
        assert session is not None
        assert session.pending_approval is not None

        # Turn 2: approve → tool executes → LLM wraps up → turn completes.
        await submission_queue.put(_Submission(_Operation(
            OpType.EXEC_APPROVAL,
            {"approvals": [{"tool_call_id": tool_call_id, "approved": True}]},
        )))
        post_approval = await _collect_until(event_queue, "turn_complete")
        types = _types(post_approval)

        states = [
            e.data["state"] for e in post_approval
            if e.event_type == "tool_state_change"
        ]
        assert "approved" in states
        assert "running" in states

        outputs = [e for e in post_approval if e.event_type == "tool_output"]
        assert len(outputs) == 1
        assert outputs[0].data["success"] is True
        assert outputs[0].data["output"] == "ingested ok"
        assert outputs[0].data["tool_call_id"] == tool_call_id

        assert executed == [
            ("hf_to_uc", {"operation": "ingest_dataset", "repo_id": "squad"}, tool_call_id)
        ]
        assert session.pending_approval is None

        finals = [e for e in post_approval if e.event_type == "assistant_message"]
        assert finals and finals[-1].data["content"] == "Ingestion complete."
        assert types[-1] == "turn_complete"

        # The second LLM call saw the tool result in history.
        assert len(calls) == 2
        tool_msgs = [m for m in calls[1] if getattr(m, "role", None) == "tool"]
        assert any(m.content == "ingested ok" for m in tool_msgs)

        # Shutdown ends the loop cleanly.
        await submission_queue.put(_Submission(_Operation(OpType.SHUTDOWN)))
        await asyncio.wait_for(loop_task, timeout=15)
        assert "shutdown" in _types(_drain(event_queue))
    finally:
        if not loop_task.done():
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)


# ── 2. approval rejected → loop continues, no stall ─────────────────────


@pytest.mark.asyncio
async def test_approval_rejected_loop_continues(monkeypatch):
    _quiet_tracing(monkeypatch)
    calls = _script_llm(monkeypatch, [
        _tool_call_response("hf_to_uc", {"operation": "ingest_dataset", "repo_id": "squad"}),
        _text_response("Understood — skipping the ingestion."),
    ])
    router, executed = _make_router(monkeypatch)

    event_queue: asyncio.Queue = asyncio.Queue()
    session = Session(
        event_queue, config=_make_config(), tool_router=router, stream=False,
    )

    result = await asyncio.wait_for(
        Handlers.run_agent(session, "ingest squad"), timeout=15,
    )
    assert result is None  # paused on approval
    assert session.pending_approval is not None
    tool_call_id = session.pending_approval["tool_calls"][0].id
    _drain(event_queue)

    await asyncio.wait_for(
        Handlers.exec_approval(session, [{
            "tool_call_id": tool_call_id,
            "approved": False,
            "feedback": "too expensive",
        }]),
        timeout=15,
    )

    events = _drain(event_queue)
    types = _types(events)

    assert executed == []  # rejected tool never ran

    states = [
        e.data["state"] for e in events if e.event_type == "tool_state_change"
    ]
    assert "rejected" in states

    outputs = [e for e in events if e.event_type == "tool_output"]
    assert len(outputs) == 1
    assert outputs[0].data["success"] is False
    assert "cancelled by user" in outputs[0].data["output"]
    assert "too expensive" in outputs[0].data["output"]

    # The loop continued: a follow-up LLM call produced a final answer and
    # the turn completed — no stall.
    finals = [e for e in events if e.event_type == "assistant_message"]
    assert finals and finals[-1].data["content"] == "Understood — skipping the ingestion."
    assert "turn_complete" in types
    assert session.pending_approval is None

    assert len(calls) == 2
    rejection_msgs = [
        m for m in calls[1]
        if getattr(m, "role", None) == "tool" and "cancelled by user" in (m.content or "")
    ]
    assert rejection_msgs, "rejection tool-result must be replayed to the LLM"


# ── 3. malformed tool-call JSON mid-loop → self-correct, no stall ────────


@pytest.mark.asyncio
async def test_malformed_tool_json_self_corrects(monkeypatch):
    _quiet_tracing(monkeypatch)
    calls = _script_llm(monkeypatch, [
        _tool_call_response("echo_stub", '{"text": "hi', call_id="call_bad"),  # truncated JSON
        _tool_call_response("echo_stub", {"text": "hi again"}, call_id="call_ok"),
        _text_response("Echoed successfully."),
    ])
    router, executed = _make_router(monkeypatch)

    event_queue: asyncio.Queue = asyncio.Queue()
    session = Session(
        event_queue, config=_make_config(), tool_router=router, stream=False,
    )

    result = await asyncio.wait_for(
        Handlers.run_agent(session, "echo hi"), timeout=15,
    )

    events = _drain(event_queue)
    outputs = [e for e in events if e.event_type == "tool_output"]
    assert len(outputs) == 2

    # First attempt: malformed JSON surfaced as a failed tool result —
    # not executed, not a crash.
    assert outputs[0].data["tool_call_id"] == "call_bad"
    assert outputs[0].data["success"] is False
    assert "malformed JSON" in outputs[0].data["output"]

    # Second attempt (the self-correction) executed for real.
    assert outputs[1].data["tool_call_id"] == "call_ok"
    assert outputs[1].data["success"] is True
    assert executed == [("echo_stub", {"text": "hi again"}, "call_ok")]

    # Loop reached a final answer and completed the turn.
    assert result == "Echoed successfully."
    assert "turn_complete" in _types(events)

    # The corrective error tool-result was replayed to the LLM on call 2.
    assert len(calls) == 3
    error_msgs = [
        m for m in calls[1]
        if getattr(m, "role", None) == "tool"
        and "malformed JSON" in (m.content or "")
    ]
    assert error_msgs, "LLM must see the malformed-JSON error to self-correct"


# ── 4. malformed JSON on a pending-approval tool → rejected, not executed ──


@pytest.mark.asyncio
async def test_exec_approval_rejects_malformed_tool_json(monkeypatch):
    _quiet_tracing(monkeypatch)
    calls = _script_llm(monkeypatch, [
        _tool_call_response("hf_to_uc", {"operation": "ingest_dataset"}),
        _text_response("Recovered after the bad tool call."),
    ])
    router, executed = _make_router(monkeypatch)

    event_queue: asyncio.Queue = asyncio.Queue()
    session = Session(
        event_queue, config=_make_config(), tool_router=router, stream=False,
    )

    await asyncio.wait_for(Handlers.run_agent(session, "ingest squad"), timeout=15)
    assert session.pending_approval is not None
    tc = session.pending_approval["tool_calls"][0]
    # Corrupt the stored arguments (models the pre-patch stall scenario:
    # unparseable tool-call JSON sitting in pending_approval).
    tc.function.arguments = '{"operation": '
    _drain(event_queue)

    await asyncio.wait_for(
        Handlers.exec_approval(
            session, [{"tool_call_id": tc.id, "approved": True}],
        ),
        timeout=15,
    )

    events = _drain(event_queue)

    # Post-patch behavior: NOT executed even though the user approved.
    assert executed == []
    # Args sanitised so the poisoned string can't break the next LLM request.
    assert tc.function.arguments == "{}"

    outputs = [e for e in events if e.event_type == "tool_output"]
    assert len(outputs) == 1
    assert outputs[0].data["success"] is False
    assert "malformed JSON" in outputs[0].data["output"]

    # Loop continued to a clean turn end instead of stalling.
    finals = [e for e in events if e.event_type == "assistant_message"]
    assert finals and finals[-1].data["content"] == "Recovered after the bad tool call."
    assert "turn_complete" in _types(events)

    # The error tool-result reached the LLM so it can re-issue the call.
    assert len(calls) == 2
    error_msgs = [
        m for m in calls[1]
        if getattr(m, "role", None) == "tool"
        and "malformed JSON" in (m.content or "")
    ]
    assert error_msgs
