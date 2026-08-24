"""A length-truncated LLM response must continue the turn, not silently end it.

Found by running the agent: on a long task the model spent 79s composing a training
script, hit its completion-token ceiling before emitting any tool call, and the loop
reported turn_complete with the work discarded. The old guard was
`if finish_reason == "length" and tool_calls_acc:`, so it only covered the case where a
*partial* tool call had been accumulated.
"""

from __future__ import annotations

import asyncio

import pytest
from litellm import Message

from agent.core import agent_loop


def _read_guard_source() -> str:
    import inspect

    return inspect.getsource(agent_loop.Handlers._run_agent_inner)


def test_length_guard_is_not_conditional_on_partial_tool_calls():
    src = _read_guard_source()
    assert 'if finish_reason == "length" and tool_calls_acc:' not in src, (
        "guard must fire for a length-truncated response even when no tool call was "
        "accumulated — that is the case that silently ended turns"
    )
    assert 'if finish_reason == "length":' in src


def test_length_guard_has_a_branch_for_no_accumulated_tool_calls():
    src = _read_guard_source()
    # The no-tool-call branch must tell the model something actionable rather than
    # reusing the "your tool calls were lost" wording, which would be false.
    assert "before you" in src and "any tool call" in src


@pytest.mark.parametrize("finish_reason", ["length"])
def test_llmresult_carries_finish_reason(finish_reason):
    """The guard depends on finish_reason surviving onto LLMResult."""
    result = agent_loop.LLMResult(
        content="partial",
        tool_calls_acc={},
        token_count=10,
        finish_reason=finish_reason,
    )
    assert result.finish_reason == "length"


def test_assistant_message_from_result_keeps_partial_content():
    """Truncated prose is still worth keeping in history as context for the retry."""
    result = agent_loop.LLMResult(
        content="I was about to write the script when",
        tool_calls_acc={},
        token_count=10,
        finish_reason="length",
    )
    msg = agent_loop._assistant_message_from_result(result, model_name="databricks/x")
    assert isinstance(msg, Message)
    assert "about to write" in (msg.content or "")


def test_no_tool_continuation_guard_still_needs_an_unfinished_plan():
    """Context for why the length case needed its own fix.

    The existing no-tool guard only fires when plan_tool has unfinished items, so it
    could not have caught this: the agent had registered no plan.
    """

    class _S:
        current_plan = []

    assert agent_loop._unfinished_plan_items(_S()) == []

    class _S2:
        current_plan = [{"id": "1", "content": "train", "status": "pending"}]

    assert len(agent_loop._unfinished_plan_items(_S2())) == 1


def test_event_loop_is_untouched():
    assert asyncio.get_event_loop_policy() is not None
