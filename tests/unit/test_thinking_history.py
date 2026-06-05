"""Regression tests for thinking-state replay across tool turns.

Anthropic extended-thinking models reject the next request with
"Invalid signature in thinking block" when ``thinking_blocks`` from the
prior assistant turn is dropped during context-history rebuild. Same
shape applies to Databricks FMAPI Claude — the gateway proxies the
Anthropic served-model schema, so input messages may carry the same
metadata.

These tests cover:
1. ``_extract_thinking_state`` reads the reasoning fields off a litellm
   ``Message``.
2. ``_assistant_message_from_result`` preserves them on the reconstructed
   history message — gated by provider prefix.
3. The streaming path rebuilds the thinking state via
   ``stream_chunk_builder`` for accepted providers, skips it otherwise.
"""

from types import SimpleNamespace

import pytest
from litellm import ChatCompletionMessageToolCall, Message

from agent.core import agent_loop
from agent.core.agent_loop import (
    LLMResult,
    _assistant_message_from_result,
    _call_llm_streaming,
    _extract_thinking_state,
    _should_replay_thinking_state,
)


def test_extract_thinking_state_from_litellm_message():
    message = Message(
        role="assistant",
        content="working",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
        reasoning_content="reasoned",
    )

    thinking_blocks, reasoning_content = _extract_thinking_state(message)

    assert thinking_blocks == [{"type": "thinking", "thinking": "reasoned"}]
    assert reasoning_content == "reasoned"


def test_assistant_message_from_result_preserves_thinking_with_tool_calls():
    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function={"name": "bash", "arguments": '{"command": "date"}'},
    )
    result = LLMResult(
        content=None,
        tool_calls_acc={},
        token_count=12,
        finish_reason="tool_calls",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
        reasoning_content="reasoned",
    )

    message = _assistant_message_from_result(
        result,
        model_name="anthropic/claude-opus-4-6",
        tool_calls=[tool_call],
    )

    assert message.tool_calls == [tool_call]
    assert message.thinking_blocks == [{"type": "thinking", "thinking": "reasoned"}]
    assert message.reasoning_content == "reasoned"


def test_assistant_message_from_result_preserves_thinking_for_databricks_fmapi():
    """Databricks FMAPI proxies Claude with the same served-model schema as
    direct Anthropic, so ``databricks/`` model ids must follow the same
    replay path. This is the local extension to the upstream HF#143 gate.
    """
    result = LLMResult(
        content=None,
        tool_calls_acc={},
        token_count=12,
        finish_reason="tool_calls",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
        reasoning_content="reasoned",
    )

    message = _assistant_message_from_result(
        result,
        model_name="databricks/databricks-claude-opus-4-7",
    )

    assert message.thinking_blocks == [{"type": "thinking", "thinking": "reasoned"}]
    assert message.reasoning_content == "reasoned"


def test_assistant_message_from_result_strips_non_anthropic_reasoning_content():
    result = LLMResult(
        content=None,
        tool_calls_acc={},
        token_count=12,
        finish_reason="tool_calls",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
        reasoning_content="reasoned",
    )

    message = _assistant_message_from_result(
        result,
        model_name="openai/Qwen/Qwen3-Next-80B-A3B-Instruct",
    )

    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


def test_assistant_message_from_result_omits_absent_thinking_fields():
    result = LLMResult(
        content="done",
        tool_calls_acc={},
        token_count=12,
        finish_reason="stop",
    )

    message = _assistant_message_from_result(
        result,
        model_name="anthropic/claude-opus-4-6",
    )

    assert message.content == "done"
    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


def test_should_replay_thinking_state_gate():
    """Gate accepts Anthropic + Databricks-Claude; rejects everything else."""
    assert _should_replay_thinking_state("anthropic/claude-opus-4-6") is True
    # Databricks FMAPI Claude endpoints — name carries "claude".
    assert _should_replay_thinking_state("databricks/databricks-claude-opus-4-7") is True
    assert _should_replay_thinking_state("databricks/databricks-claude-3-7-sonnet") is True
    # Databricks FMAPI non-Claude endpoints — must NOT receive Anthropic
    # replay metadata. Codex review on the staged diff caught this:
    # broad ``databricks/`` gate would have shipped a P1 regression.
    assert _should_replay_thinking_state("databricks/databricks-meta-llama-3-1-70b-instruct") is False
    assert _should_replay_thinking_state("databricks/databricks-dbrx-instruct") is False
    assert _should_replay_thinking_state("databricks/databricks-mixtral-8x7b-instruct") is False
    # Other providers.
    assert _should_replay_thinking_state("openai/gpt-5") is False
    assert _should_replay_thinking_state("bedrock/anthropic.claude") is False
    assert _should_replay_thinking_state("huggingface/Qwen3") is False
    assert _should_replay_thinking_state(None) is False
    assert _should_replay_thinking_state("") is False


def test_assistant_message_from_result_strips_thinking_for_non_claude_databricks():
    """Codex P1 regression test: a Databricks endpoint serving DBRX or
    Llama through the same ``databricks/`` prefix must not get
    Anthropic-shape ``thinking_blocks`` replayed on the next request —
    non-Claude FMAPI schemas reject the field and tool use breaks after
    the first turn.
    """
    result = LLMResult(
        content=None,
        tool_calls_acc={},
        token_count=12,
        finish_reason="tool_calls",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
        reasoning_content="reasoned",
    )

    message = _assistant_message_from_result(
        result,
        model_name="databricks/databricks-meta-llama-3-1-70b-instruct",
    )

    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


@pytest.mark.asyncio
async def test_streaming_call_rebuilds_anthropic_thinking_state(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=3))

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    def fake_chunk_builder(chunks, **_kwargs):
        assert len(chunks) == 2
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=Message(
                        role="assistant",
                        content="done",
                        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
                        reasoning_content="reasoned",
                    )
                )
            ]
        )

    async def fake_record_llm_call(*_args, **_kwargs):
        return {}

    events = []
    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="anthropic/claude-opus-4-6"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    monkeypatch.setattr(agent_loop, "stream_chunk_builder", fake_chunk_builder)
    monkeypatch.setattr(agent_loop, "with_prompt_caching", lambda m, t, _model: (m, t))
    monkeypatch.setattr(agent_loop.telemetry, "record_llm_call", fake_record_llm_call)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "anthropic/claude-opus-4-6"},
    )

    assert result.content == "done"
    assert result.thinking_blocks == [{"type": "thinking", "thinking": "reasoned"}]
    assert result.reasoning_content == "reasoned"


@pytest.mark.asyncio
async def test_streaming_call_rebuilds_databricks_thinking_state(monkeypatch):
    """Same code path must fire for ``databricks/`` prefix — the local
    extension to the upstream HF#143 gate.
    """
    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    invoked = {"value": False}

    def fake_chunk_builder(chunks, **_kwargs):
        invoked["value"] = True
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=Message(
                        role="assistant",
                        content="ok",
                        thinking_blocks=[{"type": "thinking", "thinking": "fmapi"}],
                        reasoning_content="fmapi",
                    )
                )
            ]
        )

    async def fake_record_llm_call(*_args, **_kwargs):
        return {}

    async def send_event(event):
        pass

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="databricks/databricks-claude-opus-4-7"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    monkeypatch.setattr(agent_loop, "stream_chunk_builder", fake_chunk_builder)
    monkeypatch.setattr(agent_loop, "with_prompt_caching", lambda m, t, _model: (m, t))
    monkeypatch.setattr(agent_loop.telemetry, "record_llm_call", fake_record_llm_call)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "databricks/databricks-claude-opus-4-7"},
    )

    assert invoked["value"], "stream_chunk_builder must run for databricks/ models"
    assert result.thinking_blocks == [{"type": "thinking", "thinking": "fmapi"}]
    assert result.reasoning_content == "fmapi"


@pytest.mark.asyncio
async def test_streaming_call_skips_chunk_rebuild_for_non_anthropic(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    def fail_chunk_builder(*_args, **_kwargs):
        raise AssertionError("stream_chunk_builder should not run")

    async def fake_record_llm_call(*_args, **_kwargs):
        return {}

    async def send_event(event):
        pass

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="openai/Qwen/Qwen3"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    monkeypatch.setattr(agent_loop, "stream_chunk_builder", fail_chunk_builder)
    monkeypatch.setattr(agent_loop, "with_prompt_caching", lambda m, t, _model: (m, t))
    monkeypatch.setattr(agent_loop.telemetry, "record_llm_call", fake_record_llm_call)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/Qwen/Qwen3"},
    )

    assert result.content == "done"
    assert result.thinking_blocks is None
    assert result.reasoning_content is None
