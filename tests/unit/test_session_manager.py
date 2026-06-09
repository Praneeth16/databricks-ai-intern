"""Unit tests for backend/session_manager.py.

Covers the EventBroadcaster fan-out lifecycle (bounded queues, slow-subscriber
drop, stale-subscription cull, close() sentinel delivery) and the
verify_session_access dev-mode gating. Pure asyncio — no workspace needed.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

# The backend package imports bare-module style (CWD=backend/). Add backend/
# to sys.path so `import session_manager` resolves.
_BACKEND = str(Path(__file__).resolve().parents[2] / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import session_manager as sm  # noqa: E402
from session_manager import BROADCAST_CLOSED, EventBroadcaster, SessionManager  # noqa: E402


def _event(n: int):
    return SimpleNamespace(event_type="msg", data={"n": n})


async def _run(broadcaster: EventBroadcaster) -> asyncio.Task:
    task = asyncio.create_task(broadcaster.run())
    await asyncio.sleep(0)  # let the loop start
    return task


async def _stop(task: asyncio.Task) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


# ── EventBroadcaster ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_receives_broadcast_and_unsubscribe_stops():
    src: asyncio.Queue = asyncio.Queue()
    b = EventBroadcaster(src)
    task = await _run(b)
    try:
        sub_id, q = b.subscribe()
        await src.put(_event(1))
        msg = await asyncio.wait_for(q.get(), timeout=2)
        assert msg == {"event_type": "msg", "data": {"n": 1}}

        b.unsubscribe(sub_id)
        await src.put(_event(2))
        await asyncio.sleep(0.05)
        assert q.empty()
    finally:
        await _stop(task)


@pytest.mark.asyncio
async def test_slow_subscriber_dropped_on_full_queue(monkeypatch):
    monkeypatch.setattr(sm, "_SUBSCRIBER_QUEUE_MAXSIZE", 2)
    src: asyncio.Queue = asyncio.Queue()
    b = EventBroadcaster(src)
    task = await _run(b)
    try:
        sub_id, q = b.subscribe()
        fast_id, fast_q = b.subscribe()
        for i in range(3):
            await src.put(_event(i))
            # Keep the fast subscriber drained so only the slow one fills up.
            await asyncio.wait_for(fast_q.get(), timeout=2)
        assert sub_id not in b._subscribers
        assert fast_id in b._subscribers
        assert q.qsize() == 2  # third event dropped; queue too full for sentinel
    finally:
        await _stop(task)


@pytest.mark.asyncio
async def test_stale_subscriber_culled_with_sentinel():
    src: asyncio.Queue = asyncio.Queue()
    b = EventBroadcaster(src)
    task = await _run(b)
    try:
        sub_id, q = b.subscribe()
        b._subscribed_at[sub_id] = (
            time.monotonic() - sm._SUBSCRIPTION_MAX_AGE_SECONDS - 1
        )
        await src.put(_event(1))
        msg = await asyncio.wait_for(q.get(), timeout=2)
        assert msg is BROADCAST_CLOSED
        assert sub_id not in b._subscribers
    finally:
        await _stop(task)


@pytest.mark.asyncio
async def test_close_delivers_sentinel_and_unsubscribe_stays_safe():
    b = EventBroadcaster(asyncio.Queue())
    sub_id, q = b.subscribe()
    b.close()
    assert q.get_nowait() is BROADCAST_CLOSED
    assert sub_id not in b._subscribers
    b.unsubscribe(sub_id)  # no raise on a closed broadcaster
    b.close()  # idempotent


@pytest.mark.asyncio
async def test_subscribe_after_close_gets_immediate_sentinel():
    # Models the delete-during-subscribe race: a request that grabs the
    # broadcaster just after deletion must terminate, not hang.
    b = EventBroadcaster(asyncio.Queue())
    b.close()
    sub_id, q = b.subscribe()
    assert q.get_nowait() is BROADCAST_CLOSED
    assert sub_id not in b._subscribers


@pytest.mark.asyncio
async def test_run_cancellation_closes_subscribers():
    src: asyncio.Queue = asyncio.Queue()
    b = EventBroadcaster(src)
    task = await _run(b)
    _, q = b.subscribe()
    await _stop(task)
    assert q.get_nowait() is BROADCAST_CLOSED


# ── verify_session_access ───────────────────────────────────────────


def _manager_with(owner: str) -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr.sessions = {"s1": SimpleNamespace(user_id=owner)}
    return mgr


def _local_mode(monkeypatch):
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)


def test_verify_access_owner_match(monkeypatch):
    _local_mode(monkeypatch)
    mgr = _manager_with("alice")
    assert mgr.verify_session_access("s1", "alice") is True
    assert mgr.verify_session_access("s1", "bob") is False
    assert mgr.verify_session_access("missing", "alice") is False


def test_verify_access_dev_bypass_local_only(monkeypatch):
    _local_mode(monkeypatch)
    assert _manager_with("alice").verify_session_access("s1", "dev") is True
    assert _manager_with("dev").verify_session_access("s1", "bob") is True


def test_verify_access_dev_bypass_blocked_in_apps_mode(monkeypatch):
    monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("DATABRICKS_APP_NAME", "databricks-ai-intern")
    assert _manager_with("alice").verify_session_access("s1", "dev") is False
    assert _manager_with("dev").verify_session_access("s1", "bob") is False
    assert _manager_with("dev").verify_session_access("s1", "dev") is True
