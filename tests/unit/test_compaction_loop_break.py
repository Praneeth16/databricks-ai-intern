"""Regression tests for the 2026-05-03 infinite-compaction-loop bug.

Pod logs from prod-114 showed sessions stuck retrying compaction every
few seconds because a single oversized tool output in the untouched tail
kept the post-compact context above the 90% threshold:

    Context compacted: 200001 -> 215566 tokens
    Context compacted: 215566 -> 215572 tokens
    ContextWindowExceededError — forcing compaction
    ... (continues for 5+ minutes)

These tests cover three fixes:

1. ``_truncate_oversized`` replaces oversized message content with a
   placeholder and preserves all extended-thinking metadata fields.
2. ``compact()`` raises ``CompactionFailedError`` when the post-compact
   context is still over threshold.
3. ``_compact_and_notify`` catches the error, sets ``session.is_running
   = False``, and emits a ``session_terminated`` event so callers can
   exit the agent loop.

The P0 caught by PR #213 review (loop didn't actually exit on
``is_running = False``) would have been caught by an end-to-end
behavioral test of #3 — that gap is closed by the
``test_compact_and_notify_terminates_session`` case below.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm import Message

from agent.context_manager.manager import (
    CompactionFailedError,
    ContextManager,
    _MAX_TOKENS_PER_MESSAGE,
)


# ── helpers ────────────────────────────────────────────────────────────


def _make_cm(
    *,
    model_max_tokens: int = 100_000,
    compact_size: int = 1_000,
    untouched_messages: int = 5,
) -> ContextManager:
    cm = ContextManager.__new__(ContextManager)
    cm.system_prompt = "system"
    cm.model_max_tokens = model_max_tokens
    cm.compact_size = compact_size
    cm.running_context_usage = 0
    cm.untouched_messages = untouched_messages
    cm.items = [Message(role="system", content="system")]
    cm.on_message_added = None
    return cm


def _msg(role: str, content: str | None = "x", **extra) -> Message:
    return Message(role=role, content=content, **extra)


# ── _truncate_oversized ────────────────────────────────────────────────


def test_truncate_oversized_skips_messages_below_threshold():
    cm = _make_cm()
    msgs = [_msg("user", "small content")]
    with patch("litellm.token_counter", return_value=100):
        out = cm._truncate_oversized(msgs, "anthropic/claude-opus-4-6")
    assert out == msgs  # unchanged


def test_truncate_oversized_replaces_content_above_threshold():
    cm = _make_cm()
    big = "x" * (_MAX_TOKENS_PER_MESSAGE * 5)
    msgs = [_msg("user", big)]
    # token_counter returns the simulated big size for any message in this test
    with patch("litellm.token_counter", return_value=_MAX_TOKENS_PER_MESSAGE * 2):
        out = cm._truncate_oversized(msgs, "anthropic/claude-opus-4-6")
    assert len(out) == 1
    assert out[0].content != big
    assert "[truncated for compaction" in out[0].content
    assert str(_MAX_TOKENS_PER_MESSAGE * 2) in out[0].content


def test_truncate_oversized_preserves_thinking_blocks():
    """Anthropic extended-thinking models reject the next request with
    ``Invalid signature in thinking block`` if a prior assistant message
    drops thinking_blocks. Truncation must keep this metadata.
    """
    cm = _make_cm()
    big = "x" * (_MAX_TOKENS_PER_MESSAGE * 5)
    thinking = [{"type": "thinking", "thinking": "...", "signature": "abc123"}]
    msg = Message(role="assistant", content=big)
    msg.thinking_blocks = thinking
    msg.reasoning_content = "deep thought"
    with patch("litellm.token_counter", return_value=_MAX_TOKENS_PER_MESSAGE * 2):
        out = cm._truncate_oversized([msg], "anthropic/claude-opus-4-6")
    assert getattr(out[0], "thinking_blocks", None) == thinking
    assert getattr(out[0], "reasoning_content", None) == "deep thought"


def test_truncate_oversized_never_touches_system_message():
    """The system prompt is the agent's instructions — must never be truncated.

    Caught by the integration smoke test on PR #213: when items has fewer than
    ``untouched_messages`` entries, the slice math in ``compact()`` can let
    ``items[0]`` (the system message) leak into the ``recent_messages`` list
    that gets passed to ``_truncate_oversized``. The function must guard
    explicitly against this.
    """
    cm = _make_cm()
    huge_system = "x" * (_MAX_TOKENS_PER_MESSAGE * 5)
    msgs = [_msg("system", huge_system)]
    with patch("litellm.token_counter", return_value=_MAX_TOKENS_PER_MESSAGE * 2):
        out = cm._truncate_oversized(msgs, "anthropic/claude-opus-4-6")
    assert out[0].content == huge_system, "system message must never be truncated"


def test_truncate_oversized_resilient_to_token_counter_failure():
    """token_counter occasionally raises on edge-case content. A blip there
    must NOT drop the message — better to leave it and let compaction
    handle it (or fail with CompactionFailedError) than to lose data.
    """
    cm = _make_cm()
    msgs = [_msg("user", "anything")]
    with patch("litellm.token_counter", side_effect=Exception("counter blew up")):
        out = cm._truncate_oversized(msgs, "anthropic/claude-opus-4-6")
    assert out == msgs


# ── compact() raises CompactionFailedError ─────────────────────────────


@pytest.mark.asyncio
async def test_compact_raises_when_post_compact_still_over_threshold():
    """The whole point of the new behavior: don't loop on a useless
    compaction call. Raise so the caller can terminate the session.
    """
    cm = _make_cm(model_max_tokens=100_000)
    # Build a context that's "over threshold" from the start
    cm.items = [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        Message(role="assistant", content="x" * 1000),
        Message(role="user", content="follow-up 1"),
        Message(role="assistant", content="reply 1"),
        Message(role="user", content="follow-up 2"),
        Message(role="assistant", content="reply 2"),
    ]
    cm.running_context_usage = 95_000  # over threshold (90% of 100k = 90k)

    # Mock summarize_messages to return a tiny summary; mock _recompute_usage
    # to keep the running_context_usage above threshold so compact() raises.
    async def fake_summarize(*args, **kwargs):
        return ("summary", 10)

    def fake_recompute(self, model_name):
        # Simulate post-compact still over threshold
        self.running_context_usage = 95_000

    with (
        patch("agent.context_manager.manager.summarize_messages", side_effect=fake_summarize),
        patch.object(ContextManager, "_recompute_usage", fake_recompute),
        # Avoid token_counter calls in _truncate_oversized
        patch("litellm.token_counter", return_value=100),
    ):
        with pytest.raises(CompactionFailedError):
            await cm.compact(
                model_name="anthropic/claude-opus-4-6",
                tool_specs=None,
                hf_token=None,
                session=None,
            )


@pytest.mark.asyncio
async def test_compact_does_not_duplicate_system_when_idx_is_zero():
    """Regression for the second P0 caught by bot review on PR #213.

    When ``len(items) == untouched_messages`` (the canonical 5-message
    early-compaction case: system + user-task + giant-tool-output +
    user-followup + assistant-reply), ``idx`` initialises to 0 and the
    walk-back ``while idx > 1`` loop is a no-op. Without an explicit
    clamp ``if idx < 1: idx = 1``, ``recent_messages = items[0:]``
    starts at the system message, and the rebuild duplicates system +
    first-user. Anthropic API rejects two system messages.
    """
    cm = _make_cm(model_max_tokens=100_000, untouched_messages=5)
    cm.items = [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        Message(role="assistant", content="ok"),  # would be the only
                                                   # message_to_summarize but the
                                                   # idx bug pulls it into recent
        Message(role="user", content="followup"),
        Message(role="assistant", content="reply"),
    ]  # exactly 5 = untouched_messages, so idx initialises to 0
    cm.running_context_usage = 95_000

    async def fake_summarize(*args, **kwargs):
        return ("summary", 10)

    def fake_recompute(self, model_name):
        self.running_context_usage = 5_000

    with (
        patch("agent.context_manager.manager.summarize_messages", side_effect=fake_summarize),
        patch.object(ContextManager, "_recompute_usage", fake_recompute),
        patch("litellm.token_counter", return_value=100),
    ):
        await cm.compact(
            model_name="anthropic/claude-opus-4-6",
            tool_specs=None,
            hf_token=None,
            session=None,
        )

    # Critical assertion: only ONE system message in items
    system_count = sum(1 for m in cm.items if m.role == "system")
    assert system_count == 1, (
        f"Expected exactly 1 system message, found {system_count}. "
        f"Roles: {[m.role for m in cm.items]}"
    )
    # And the first-user "task" message must also appear exactly once.
    # Bot review on PR #213 caught a follow-up bug: clamping idx=1
    # excludes the system but still overlaps with first_user_idx (also 1),
    # so first_user_msg ends up in BOTH head and recent_messages →
    # duplicate user message → Anthropic 400 (two consecutive user roles).
    task_count = sum(
        1 for m in cm.items
        if m.role == "user" and (m.content or "") == "task"
    )
    assert task_count == 1, (
        f"Expected exactly 1 'task' user message, found {task_count}. "
        f"Roles+content: {[(m.role, (m.content or '')[:20]) for m in cm.items]}"
    )
    # Defense in depth: no two consecutive same-role messages (Anthropic
    # API contract). System counts separately.
    non_system = [m for m in cm.items if m.role != "system"]
    for i in range(1, len(non_system)):
        assert non_system[i].role != non_system[i-1].role, (
            f"Two consecutive {non_system[i].role} messages at non-system "
            f"position {i-1},{i} — Anthropic API rejects this. "
            f"Roles: {[m.role for m in cm.items]}"
        )


@pytest.mark.asyncio
async def test_compact_succeeds_when_post_compact_under_threshold():
    """Happy path: when compaction does its job, no exception raised."""
    cm = _make_cm(model_max_tokens=100_000)
    cm.items = [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        Message(role="assistant", content="x" * 1000),
        Message(role="user", content="follow-up"),
        Message(role="assistant", content="reply"),
        Message(role="user", content="follow-up 2"),
        Message(role="assistant", content="reply 2"),
    ]
    cm.running_context_usage = 95_000

    async def fake_summarize(*args, **kwargs):
        return ("summary", 10)

    def fake_recompute(self, model_name):
        self.running_context_usage = 5_000  # well under threshold

    with (
        patch("agent.context_manager.manager.summarize_messages", side_effect=fake_summarize),
        patch.object(ContextManager, "_recompute_usage", fake_recompute),
        patch("litellm.token_counter", return_value=100),
    ):
        await cm.compact(
            model_name="anthropic/claude-opus-4-6",
            tool_specs=None,
            hf_token=None,
            session=None,
        )
    assert cm.running_context_usage == 5_000


# ── aggressive second-pass recovery ────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_recovers_via_aggressive_second_pass():
    """The summarize+truncate pass leaves us over threshold, but dropping the
    non-essential mid-conversation messages brings us under. The old
    single-pass code would have raised (killing the session); the aggressive
    second pass must recover it instead.
    """
    cm = _make_cm(model_max_tokens=100_000, untouched_messages=3)
    cm.items = [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        Message(role="assistant", content="x" * 1000),
        Message(role="user", content="follow-up 1"),
        Message(role="assistant", content="reply 1"),
        Message(role="user", content="follow-up 2"),
        Message(role="assistant", content="reply 2"),
    ]
    cm.running_context_usage = 95_000

    async def fake_summarize(*args, **kwargs):
        return ("summary", 10)

    # First recompute (after summarize) stays over threshold; the recompute
    # triggered by the aggressive reduction drops under it.
    state = {"calls": 0}

    def fake_recompute(self, model_name):
        state["calls"] += 1
        if state["calls"] == 1:
            self.running_context_usage = 95_000  # summarize pass insufficient
        else:
            self.running_context_usage = 5_000  # aggressive pass fits

    with (
        patch("agent.context_manager.manager.summarize_messages", side_effect=fake_summarize),
        patch.object(ContextManager, "_recompute_usage", fake_recompute),
        patch("litellm.token_counter", return_value=100),
    ):
        # Must NOT raise.
        await cm.compact(
            model_name="anthropic/claude-opus-4-6",
            tool_specs=None,
            hf_token=None,
            session=None,
        )

    assert cm.running_context_usage == 5_000
    # The aggressive pass dropped at least one mid-conversation message.
    assert state["calls"] == 2, "aggressive reduction must run exactly once"
    # Spine preserved: exactly one system + the first user task survive.
    assert sum(1 for m in cm.items if m.role == "system") == 1
    assert sum(
        1 for m in cm.items if m.role == "user" and (m.content or "") == "task"
    ) == 1


@pytest.mark.asyncio
async def test_compact_still_terminates_when_aggressive_pass_cannot_fit():
    """Even after dropping every non-essential message, the spine (system +
    first user + tail) is still over threshold → must raise
    CompactionFailedError. Proves the aggressive pass runs exactly once and
    does not loop.
    """
    cm = _make_cm(model_max_tokens=100_000, untouched_messages=3)
    cm.items = [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        Message(role="assistant", content="x" * 1000),
        Message(role="user", content="follow-up 1"),
        Message(role="assistant", content="reply 1"),
        Message(role="user", content="follow-up 2"),
        Message(role="assistant", content="reply 2"),
    ]
    cm.running_context_usage = 95_000

    async def fake_summarize(*args, **kwargs):
        return ("summary", 10)

    recompute_calls = {"n": 0}

    def fake_recompute(self, model_name):
        # Always over threshold — even the aggressive pass can't fit.
        recompute_calls["n"] += 1
        self.running_context_usage = 95_000

    with (
        patch("agent.context_manager.manager.summarize_messages", side_effect=fake_summarize),
        patch.object(ContextManager, "_recompute_usage", fake_recompute),
        patch("litellm.token_counter", return_value=100),
    ):
        with pytest.raises(CompactionFailedError):
            await cm.compact(
                model_name="anthropic/claude-opus-4-6",
                tool_specs=None,
                hf_token=None,
                session=None,
            )

    # recompute fires once for the summarize pass and once for the single
    # aggressive pass — never more (no infinite loop).
    assert recompute_calls["n"] == 2, (
        "aggressive reduction must run exactly once, not loop"
    )


# ── _compact_and_notify behavior on CompactionFailedError ──────────────


@pytest.mark.asyncio
async def test_compact_and_notify_terminates_session_on_failure():
    """The PR's #213's P0 bug-class: setting ``is_running = False`` is
    only effective if the agent loop checks it. This test asserts the
    flag IS set AND a ``session_terminated`` event is emitted, so a
    follow-up assertion in the agent loop test catches the loop-exit.
    """
    from agent.core.agent_loop import _compact_and_notify

    session = MagicMock()
    session.session_id = "sess-123"
    session.is_running = True
    session.config.model_name = "anthropic/claude-opus-4-6"
    session.hf_token = None
    session.tool_router.get_tool_specs_for_llm.return_value = []
    session.send_event = AsyncMock()

    cm = MagicMock()
    cm.running_context_usage = 95_000
    cm.compaction_threshold = 90_000
    cm.model_max_tokens = 100_000
    cm.items = []
    cm.needs_compaction = True
    cm.compact = AsyncMock(side_effect=CompactionFailedError("ineffective"))
    session.context_manager = cm

    await _compact_and_notify(session)

    assert session.is_running is False, (
        "_compact_and_notify must set is_running=False so the agent loop "
        "can exit. P0 caught by bot review on PR #213 was that the loop "
        "didn't actually check this flag."
    )
    assert session.send_event.await_count == 1
    event = session.send_event.await_args.args[0]
    assert event.event_type == "session_terminated"
    assert event.data["reason"] == "compaction_failed"
    assert event.data["context_usage"] == 95_000


@pytest.mark.asyncio
async def test_compact_and_notify_passes_through_on_success():
    """When compaction succeeds, no termination event, is_running stays True."""
    from agent.core.agent_loop import _compact_and_notify

    session = MagicMock()
    session.session_id = "sess-456"
    session.is_running = True
    session.config.model_name = "anthropic/claude-opus-4-6"
    session.hf_token = None
    session.tool_router.get_tool_specs_for_llm.return_value = []
    session.send_event = AsyncMock()

    cm = MagicMock()
    cm.running_context_usage = 5_000
    cm.compaction_threshold = 90_000
    cm.model_max_tokens = 100_000
    cm.items = []
    cm.needs_compaction = False
    cm.compact = AsyncMock(return_value=None)  # success
    session.context_manager = cm

    # Pretend old_usage == new_usage so the "compacted" event is also skipped
    await _compact_and_notify(session)

    assert session.is_running is True
    # No session_terminated event emitted
    for call in session.send_event.await_args_list:
        ev = call.args[0]
        assert ev.event_type != "session_terminated"


# ── _recompute_usage failure path ──────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_raises_when_token_counter_fails():
    """Codex review: if ``token_counter`` raises on the post-compact items
    (e.g. a message carrying large ``thinking_blocks`` / ``tool_calls``
    metadata that the previous char-based fallback ignored), a silent
    undercount would let ``compact()`` return without raising. The next
    LLM call would then exceed the context window and re-enter the same
    compaction path — exactly the loop this PR is supposed to break.

    Fail-fast: on ``token_counter`` failure ``_recompute_usage`` marks
    usage as ``model_max_tokens + 1`` so ``compact()`` raises
    ``CompactionFailedError`` and the caller terminates the session.
    """
    cm = _make_cm(model_max_tokens=100_000)
    cm.items = [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        Message(role="assistant", content="x" * 1000),
        Message(role="user", content="follow-up"),
        Message(role="assistant", content="reply"),
        Message(role="user", content="follow-up 2"),
        Message(role="assistant", content="reply 2"),
    ]
    cm.running_context_usage = 95_000

    async def fake_summarize(*args, **kwargs):
        return ("summary", 10)

    # token_counter raises on every call — simulates LiteLLM choking on a
    # message that carries metadata in fields it doesn't know how to size.
    with (
        patch("agent.context_manager.manager.summarize_messages", side_effect=fake_summarize),
        patch("litellm.token_counter", side_effect=RuntimeError("counter blew up")),
    ):
        with pytest.raises(CompactionFailedError):
            await cm.compact(
                model_name="anthropic/claude-opus-4-6",
                tool_specs=None,
                hf_token=None,
                session=None,
            )

    # And the running usage must be marked over-threshold, not undercounted.
    assert cm.running_context_usage > cm.model_max_tokens, (
        "On token_counter failure _recompute_usage must mark usage > model_max_tokens "
        "so the compact() threshold check raises. Found "
        f"running_context_usage={cm.running_context_usage}, model_max_tokens="
        f"{cm.model_max_tokens}."
    )
