"""Integration tests for the backend /api/chat SSE route.

Drives the real FastAPI app (``backend/main.py``) through httpx's
ASGITransport with auth overridden and a fake agent session injected into
the real ``session_manager`` singleton. The real ``EventBroadcaster`` runs
in-loop, so these tests exercise the patched bounded-queue + close()
sentinel path end to end:

  1. happy path — submitted input streams events and the stream terminates
     on the terminal event (turn_complete), unsubscribing cleanly.
  2. session deleted mid-stream — delete_session() closes the broadcaster;
     the BROADCAST_CLOSED sentinel ends the stream without a terminal event.
  3. missing / inactive session — 404 before any stream is opened.

Note: httpx ASGITransport runs the app to completion before returning the
response, so "mid-stream" interactions are scheduled as asyncio tasks from
inside the stubbed submit — they run while the SSE generator awaits its
subscriber queue on the same event loop.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

# The backend package imports bare-module style (CWD=backend/). Add backend/
# to sys.path so `import main` / `import session_manager` resolve.
_BACKEND = str(Path(__file__).resolve().parents[2] / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import main as backend_main  # noqa: E402
import session_manager as sm  # noqa: E402
from dependencies import get_current_user  # noqa: E402
from session_manager import AgentSession, EventBroadcaster  # noqa: E402

from agent.core.session import Event  # noqa: E402

_USER = {
    "user_id": "tester",
    "user_name": "tester",
    "display_name": "Tester",
    "email": "tester@example.com",
    "workspace_url": "https://example.cloud.databricks.com",
    "authenticated": True,
}


def _fake_agent_session(session_id: str) -> tuple[AgentSession, asyncio.Queue]:
    """Real AgentSession wrapper + real EventBroadcaster around a fake
    core Session — no workspace, no LLM, no tool router."""
    source: asyncio.Queue = asyncio.Queue()
    broadcaster = EventBroadcaster(source)
    fake_session = SimpleNamespace(
        context_manager=SimpleNamespace(items=[]),
        pending_approval=None,
        config=SimpleNamespace(
            model_name="databricks/databricks-claude-opus-4",
            save_sessions=False,
        ),
        sandbox=None,
    )
    agent_session = AgentSession(
        session_id=session_id,
        session=fake_session,
        tool_router=None,
        submission_queue=asyncio.Queue(),
        user_id=_USER["user_id"],
        broadcaster=broadcaster,
    )
    return agent_session, source


class _Harness:
    def __init__(self, session_id, agent_session, source, broadcast_task):
        self.session_id = session_id
        self.agent_session = agent_session
        self.source = source
        self.broadcast_task = broadcast_task

    @property
    def broadcaster(self) -> EventBroadcaster:
        return self.agent_session.broadcaster


@pytest_asyncio.fixture
async def harness(monkeypatch):
    """Install a fake session into the real session_manager and run its
    broadcaster (on the test's event loop). Auth is overridden on the
    real app."""
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)

    session_id = "sse-test-session"
    agent_session, source = _fake_agent_session(session_id)
    sm.session_manager.sessions[session_id] = agent_session
    broadcast_task = asyncio.create_task(agent_session.broadcaster.run())

    backend_main.app.dependency_overrides[get_current_user] = lambda: _USER
    try:
        yield _Harness(session_id, agent_session, source, broadcast_task)
    finally:
        backend_main.app.dependency_overrides.pop(get_current_user, None)
        sm.session_manager.sessions.pop(session_id, None)
        broadcast_task.cancel()
        await asyncio.gather(broadcast_task, return_exceptions=True)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=backend_main.app),
        base_url="http://testserver",
    )


def _parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


# ── 1. happy path: events stream, terminal event ends the stream ─────────


@pytest.mark.asyncio
async def test_chat_streams_events_until_terminal(harness, monkeypatch):
    submitted: list[tuple[str, str]] = []

    async def fake_submit_user_input(session_id: str, text: str) -> bool:
        submitted.append((session_id, text))
        # Simulate the agent loop emitting a turn onto the session's event
        # queue. The broadcaster fans these out to the SSE subscriber.
        harness.source.put_nowait(Event("processing", {"message": "Processing"}))
        harness.source.put_nowait(Event("assistant_message", {"content": "hello"}))
        harness.source.put_nowait(Event("turn_complete", {"history_size": 2}))
        # Anything after the terminal event must NOT reach the client.
        harness.source.put_nowait(Event("assistant_message", {"content": "late"}))
        return True

    monkeypatch.setattr(
        sm.session_manager, "submit_user_input", fake_submit_user_input
    )

    async with _client() as client:
        resp = await asyncio.wait_for(
            client.post(f"/api/chat/{harness.session_id}", json={"text": "hi"}),
            timeout=15,
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert submitted == [(harness.session_id, "hi")]

    payloads = _parse_sse(resp.text)
    assert [p["event_type"] for p in payloads] == [
        "processing", "assistant_message", "turn_complete",
    ]
    assert payloads[1]["data"] == {"content": "hello"}
    assert payloads[-1]["event_type"] == "turn_complete"  # terminated on terminal

    # The handler's finally block unsubscribed — no leaked subscriber queues.
    assert harness.broadcaster._subscribers == {}


# ── 2. delete during stream → close() sentinel ends the stream cleanly ───


@pytest.mark.asyncio
async def test_chat_stream_ends_cleanly_on_session_delete(harness, monkeypatch):
    async def fake_submit_user_input(session_id: str, text: str) -> bool:
        # One non-terminal event, then the session is deleted while the SSE
        # generator is parked on its subscriber queue.
        harness.source.put_nowait(Event("processing", {"message": "Processing"}))

        async def _delete_later():
            await asyncio.sleep(0.05)
            assert await sm.session_manager.delete_session(session_id) is True

        asyncio.create_task(_delete_later())
        return True

    monkeypatch.setattr(
        sm.session_manager, "submit_user_input", fake_submit_user_input
    )

    async with _client() as client:
        # If the close() sentinel were missing, this request would hang on
        # the subscriber queue until the 15 s keepalive — the wait_for is
        # the no-stall assertion.
        resp = await asyncio.wait_for(
            client.post(f"/api/chat/{harness.session_id}", json={"text": "hi"}),
            timeout=10,
        )

    assert resp.status_code == 200
    payloads = _parse_sse(resp.text)
    # Got the pre-delete event, then end-of-stream — no terminal event.
    assert [p["event_type"] for p in payloads] == ["processing"]

    # delete_session removed the session and close() cleared all subscribers.
    assert harness.session_id not in sm.session_manager.sessions
    assert harness.broadcaster._closed is True
    assert harness.broadcaster._subscribers == {}

    # A late subscriber on the closed broadcaster gets the sentinel
    # immediately — the delete-during-subscribe race can't hang a request.
    _, late_q = harness.broadcaster.subscribe()
    assert late_q.get_nowait() is sm.BROADCAST_CLOSED


# ── 3. missing / inactive session → 404, no stream ───────────────────────


@pytest.mark.asyncio
async def test_chat_missing_session_returns_404(harness):
    async with _client() as client:
        resp = await client.post("/api/chat/does-not-exist", json={"text": "hi"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found"


@pytest.mark.asyncio
async def test_chat_inactive_session_returns_404(harness):
    harness.agent_session.is_active = False
    async with _client() as client:
        resp = await client.post(
            f"/api/chat/{harness.session_id}", json={"text": "hi"}
        )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found or inactive"
