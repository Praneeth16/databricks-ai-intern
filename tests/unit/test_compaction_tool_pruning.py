"""Compaction phase 1: prune stale tool-result bodies before paying to summarize.

`_truncate_oversized` only fires on single messages above `_MAX_TOKENS_PER_MESSAGE`
(50k tokens). A long run accumulates many *medium* tool results — a hundred 2k-token
ones trip nothing there yet consume 200k tokens. Pruning those costs no LLM call, where
summarization costs one, so it runs first.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from litellm import Message

from agent.context_manager.manager import (
    _TOOL_RESULT_PRUNE_KEEP_CHARS,
    ContextManager,
)


def _make_cm(*, untouched_messages: int = 4) -> ContextManager:
    cm = ContextManager.__new__(ContextManager)
    cm.system_prompt = "system"
    cm.model_max_tokens = 100_000
    cm.compact_size = 1_000
    cm.running_context_usage = 0
    cm.untouched_messages = untouched_messages
    cm.items = [Message(role="system", content="system")]
    cm.on_message_added = None
    return cm


def _tool_msg(content: str, call_id: str = "call_1") -> Message:
    return Message(role="tool", content=content, tool_call_id=call_id, name="bash")


BIG = "D" * 20_000
MODEL = "anthropic/claude-opus-4-6"


def test_prunes_stale_tool_results():
    cm = _make_cm(untouched_messages=2)
    cm.items += [
        Message(role="user", content="task"),
        _tool_msg(BIG, "c1"),
        _tool_msg(BIG, "c2"),
        Message(role="assistant", content="recent"),
        Message(role="user", content="recent"),
    ]
    with patch.object(cm, "_recompute_usage"):
        pruned = cm._prune_stale_tool_results(MODEL)
    assert pruned == 2
    assert len(cm.items[2].content) < len(BIG)
    assert "elided during compaction" in cm.items[2].content


def test_keeps_a_readable_head_of_each_pruned_result():
    cm = _make_cm(untouched_messages=1)
    cm.items += [Message(role="user", content="task"), _tool_msg("HEADMARKER" + BIG), Message(role="user", content="r")]
    with patch.object(cm, "_recompute_usage"):
        cm._prune_stale_tool_results(MODEL)
    assert cm.items[2].content.startswith("HEADMARKER")
    assert len(cm.items[2].content) < _TOOL_RESULT_PRUNE_KEEP_CHARS + 200


def test_does_not_prune_the_untouched_recent_tail():
    """Recent tool output is what the agent is actively reasoning about."""
    cm = _make_cm(untouched_messages=3)
    cm.items += [
        Message(role="user", content="task"),
        _tool_msg(BIG, "old"),
        _tool_msg(BIG, "recent1"),
        Message(role="assistant", content="a"),
        Message(role="user", content="b"),
    ]
    with patch.object(cm, "_recompute_usage"):
        pruned = cm._prune_stale_tool_results(MODEL)
    assert pruned == 1
    assert cm.items[3].content == BIG, "recent tool result must survive untouched"


def test_leaves_short_tool_results_alone():
    cm = _make_cm(untouched_messages=1)
    cm.items += [Message(role="user", content="task"), _tool_msg("tiny"), Message(role="user", content="r")]
    with patch.object(cm, "_recompute_usage"):
        pruned = cm._prune_stale_tool_results(MODEL)
    assert pruned == 0
    assert cm.items[2].content == "tiny"


def test_only_tool_messages_are_pruned():
    cm = _make_cm(untouched_messages=1)
    cm.items += [
        Message(role="user", content="task"),
        Message(role="assistant", content=BIG),
        Message(role="user", content=BIG),
        Message(role="user", content="r"),
    ]
    with patch.object(cm, "_recompute_usage"):
        pruned = cm._prune_stale_tool_results(MODEL)
    assert pruned == 0
    assert cm.items[2].content == BIG


def test_preserves_tool_call_id_so_the_next_request_stays_valid():
    """Dropping tool_call_id orphans the assistant's tool_call and the API 400s."""
    cm = _make_cm(untouched_messages=1)
    cm.items += [Message(role="user", content="task"), _tool_msg(BIG, "call_abc"), Message(role="user", content="r")]
    with patch.object(cm, "_recompute_usage"):
        cm._prune_stale_tool_results(MODEL)
    assert cm.items[2].tool_call_id == "call_abc"
    assert cm.items[2].name == "bash"
    assert cm.items[2].role == "tool"


def test_no_system_message_is_ever_touched():
    cm = _make_cm(untouched_messages=0)
    cm.items = [Message(role="system", content=BIG), _tool_msg(BIG)]
    with patch.object(cm, "_recompute_usage"):
        cm._prune_stale_tool_results(MODEL)
    assert cm.items[0].content == BIG


def test_handles_history_shorter_than_the_untouched_window():
    cm = _make_cm(untouched_messages=10)
    cm.items += [Message(role="user", content="task"), _tool_msg(BIG)]
    with patch.object(cm, "_recompute_usage"):
        assert cm._prune_stale_tool_results(MODEL) == 0


def test_recomputes_usage_only_when_something_was_pruned():
    cm = _make_cm(untouched_messages=1)
    cm.items += [Message(role="user", content="task"), _tool_msg("tiny"), Message(role="user", content="r")]
    with patch.object(cm, "_recompute_usage") as recompute:
        cm._prune_stale_tool_results(MODEL)
        recompute.assert_not_called()

    cm.items[2] = _tool_msg(BIG)
    with patch.object(cm, "_recompute_usage") as recompute:
        cm._prune_stale_tool_results(MODEL)
        recompute.assert_called_once()


@pytest.mark.asyncio
async def test_compact_skips_summarization_when_pruning_is_enough():
    """The saving is a whole LLM call, so assert the summarizer is never invoked."""
    cm = _make_cm(untouched_messages=2)
    cm.items += [
        Message(role="user", content="task"),
        _tool_msg(BIG, "c1"),
        Message(role="assistant", content="a"),
        Message(role="user", content="b"),
    ]
    cm.running_context_usage = 99_000  # over threshold

    def shrink_after_prune(_model):
        cm.running_context_usage = 1_000  # pruning got us under

    with patch.object(cm, "_recompute_usage", side_effect=shrink_after_prune), \
         patch("agent.context_manager.manager.summarize_messages") as summarize:
        await cm.compact(MODEL)
        summarize.assert_not_called()
    # History spine is intact — nothing was summarized away.
    assert [m.role for m in cm.items] == ["system", "user", "tool", "assistant", "user"]


@pytest.mark.asyncio
async def test_compact_still_summarizes_when_pruning_is_not_enough():
    cm = _make_cm(untouched_messages=2)
    # compact() walks back to a user message to anchor the recent window, so there must
    # be a user turn after the task prompt or `messages_to_summarize` comes out empty
    # and compact() takes its nothing-to-summarize branch instead.
    cm.items += [
        Message(role="user", content="task"),
        _tool_msg(BIG, "c1"),
        Message(role="assistant", content="mid"),
        Message(role="user", content="follow-up"),
        Message(role="assistant", content="a"),
        Message(role="user", content="b"),
    ]
    cm.running_context_usage = 99_000  # stays over even after pruning

    async def fake_summarize(*_args, **_kwargs):
        return "SUMMARY", 10

    with patch.object(cm, "_recompute_usage"), \
         patch.object(cm, "_truncate_oversized", side_effect=lambda m, _mn: m), \
         patch.object(cm, "_aggressive_reduce"), \
         patch("agent.context_manager.manager.summarize_messages", side_effect=fake_summarize) as summarize:
        try:
            await cm.compact(MODEL)
        except Exception:
            pass  # terminal CompactionFailedError is fine; we only assert the call
        summarize.assert_called_once()
