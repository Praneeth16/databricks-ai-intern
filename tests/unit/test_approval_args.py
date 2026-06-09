"""Tests for ``Handlers.exec_approval`` handling of malformed tool-call JSON.

A tool call whose arguments fail ``json.loads`` must not stall the loop:
the call is sanitised in place (its arguments live inside the assistant
message already in context) and a tool-error result tells the model its
tool-call JSON was invalid so the loop self-corrects next iteration.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.core.agent_loop import Handlers
from agent.core.session import Session


class _FakeContext:
    def __init__(self):
        self.items = []

    def add_message(self, message, token_count=None):
        self.items.append(message)


def _make_session() -> Session:
    return Session(asyncio.Queue(), context_manager=_FakeContext())


def _tool_call(tc_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=tc_id, function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.mark.asyncio
async def test_exec_approval_malformed_args_appends_session_tool_error():
    session = _make_session()
    bad_tc = _tool_call("tc-bad", "databricks_jobs", '{"script": ')
    session.pending_approval = {"tool_calls": [bad_tc]}

    with patch.object(Handlers, "run_agent", new=AsyncMock()) as run_agent:
        await Handlers.exec_approval(
            session, [{"tool_call_id": "tc-bad", "approved": True}],
        )

    # Sanitised in place so the assistant message in context no longer
    # carries an unparseable arguments string.
    assert bad_tc.function.arguments == "{}"

    # A tool-error result was appended telling the model to re-issue.
    msgs = session.context_manager.items
    assert len(msgs) == 1
    assert msgs[0].role == "tool"
    assert msgs[0].tool_call_id == "tc-bad"
    assert "malformed JSON" in msgs[0].content
    assert "NOT executed" in msgs[0].content

    # The loop continues so the model can self-correct next iteration.
    run_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_exec_approval_malformed_args_do_not_block_valid_session_calls():
    session = _make_session()
    bad_tc = _tool_call("tc-bad", "databricks_jobs", "{not json")
    good_tc = _tool_call("tc-good", "databricks_jobs", '{"kind": "script"}')
    session.pending_approval = {"tool_calls": [bad_tc, good_tc]}
    session.tool_router = SimpleNamespace(
        call_tool=AsyncMock(return_value=("submitted", True)),
    )

    approvals = [
        {"tool_call_id": "tc-bad", "approved": True},
        {"tool_call_id": "tc-good", "approved": True},
    ]
    with patch.object(Handlers, "run_agent", new=AsyncMock()):
        await Handlers.exec_approval(session, approvals)

    # Only the valid call executed.
    session.tool_router.call_tool.assert_awaited_once()
    _, kwargs = session.tool_router.call_tool.await_args
    assert kwargs.get("tool_call_id") == "tc-good"

    # Both calls got a tool result: error for the bad, output for the good.
    by_id = {m.tool_call_id: m.content for m in session.context_manager.items}
    assert "malformed JSON" in by_id["tc-bad"]
    assert by_id["tc-good"] == "submitted"
