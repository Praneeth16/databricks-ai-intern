"""Tests for Session trajectory logging and effort-heal caching.

Coverage:
  1. ``logged_events`` is a ring buffer — capped at ``_MAX_LOGGED_EVENTS``,
     oldest-first trim, with ``dropped_event_count`` surfaced in the
     trajectory so saved logs are explicit about the gap.
  2. ``reconcile_actual_cost`` failure warns once per session (drift being
     invisible was the bug), then drops back to debug.
  3. ``_heal_effort_and_rebuild_params`` on a transient (inconclusive)
     probe strips thinking for that retry only — it must NOT cache ``None``,
     or one network blip permanently disables thinking for the model.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest

from agent.core import session as session_mod
from agent.core.session import Event, Session


class _FakeContext:
    def __init__(self):
        self.items = []

    def add_message(self, message, token_count=None):
        self.items.append(message)


def _make_session() -> Session:
    return Session(asyncio.Queue(), context_manager=_FakeContext())


# ── logged_events ring buffer ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_logged_events_ring_buffer_caps_and_counts(monkeypatch):
    monkeypatch.setattr(session_mod, "_MAX_LOGGED_EVENTS", 10)
    session = _make_session()
    for i in range(25):
        await session.send_event(Event(event_type="tool_log", data={"i": i}))

    assert len(session.logged_events) == 10
    assert session.dropped_event_count == 15
    # Oldest trimmed, newest kept.
    assert session.logged_events[0]["data"] == {"i": 15}
    assert session.logged_events[-1]["data"] == {"i": 24}

    trajectory = session.get_trajectory()
    assert len(trajectory["events"]) == 10
    assert trajectory["events_dropped"] == 15


@pytest.mark.asyncio
async def test_logged_events_below_cap_drops_nothing():
    session = _make_session()
    for i in range(5):
        await session.send_event(Event(event_type="tool_log", data={"i": i}))

    assert len(session.logged_events) == 5
    assert session.dropped_event_count == 0
    assert session.get_trajectory()["events_dropped"] == 0


# ── reconcile_actual_cost warning ───────────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_failure_warns_once_then_debug(monkeypatch, caplog):
    session = _make_session()
    session.user_email = "user@example.com"

    def _boom(_config):
        raise RuntimeError("no warehouse configured")

    monkeypatch.setattr("agent.core.db_client.resolve_settings", _boom)

    with caplog.at_level(logging.DEBUG, logger="agent.core.session"):
        await session.reconcile_actual_cost()
        session._last_reconcile_ts = None  # bypass the 60s rate limit
        await session.reconcile_actual_cost()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "reconciliation unavailable" in msg
    assert "no warehouse configured" in msg
    assert session.actual_cost_usd is None


# ── effort heal: inconclusive probe must not poison the thinking cache ──


@pytest.mark.asyncio
async def test_heal_inconclusive_probe_strips_thinking_without_caching():
    from agent.core import agent_loop
    from agent.core.effort_probe import ProbeInconclusive

    session = _make_session()
    model = "anthropic/claude-sonnet-4-6"
    session.config.model_name = model
    session.config.reasoning_effort = "high"

    error = Exception("effort 'high' is invalid for this model")
    with patch(
        "agent.core.effort_probe.probe_effort",
        new=AsyncMock(side_effect=ProbeInconclusive("503 service unavailable")),
    ):
        params = await agent_loop._heal_effort_and_rebuild_params(
            session, error, {"model": model},
        )

    # Stripped for this retry only — nothing cached, so the probe re-runs
    # next time and the raw preference is still sent on the next turn.
    assert model not in session.model_effective_effort
    assert session.effective_effort_for(model) == "high"
    assert "thinking" not in params
    assert "output_config" not in params


@pytest.mark.asyncio
async def test_heal_thinking_unsupported_still_caches_strip():
    from agent.core import agent_loop

    session = _make_session()
    model = "anthropic/claude-sonnet-4-6"
    session.config.model_name = model
    session.config.reasoning_effort = "high"

    error = Exception("thinking.type.enabled is not supported for this model")
    params = await agent_loop._heal_effort_and_rebuild_params(
        session, error, {"model": model},
    )

    # Definitive verdict — cached so we stop sending thinking params.
    assert session.model_effective_effort[model] is None
    assert "thinking" not in params
    assert "output_config" not in params
