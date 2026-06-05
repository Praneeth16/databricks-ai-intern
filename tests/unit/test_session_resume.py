"""Tests for ``agent.core.session_resume``.

Coverage:
  1. Listing prefers Lakebase, falls back to filesystem on empty/missing pool.
  2. ``resolve_session_arg`` handles index / id-prefix / path inputs.
  3. ``restore_session_from_entry`` continues vs forks based on user_id.
  4. Filesystem path also exercises the redact + invalid-model warnings.
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from litellm import Message

from agent.core import session_resume
from agent.core.session_resume import SessionLogEntry


# ── helpers ────────────────────────────────────────────────────────────


def _write_session_log(
    directory: Path,
    name: str,
    *,
    session_id: str,
    content: str,
    mtime: float,
    user_id: str | None = "user-a@example.com",
    model_name: str = "databricks/databricks-claude-opus-4-7",
    extra_messages: list[dict] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "session_start_time": "2026-01-01T00:00:00",
        "session_end_time": "2026-01-01T00:05:00",
        "model_name": model_name,
        "messages": [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": content},
            *(extra_messages or []),
        ],
        "events": [{"event_type": "turn_complete", "data": {}}],
    }
    path.write_text(json.dumps(payload))
    os.utime(path, (mtime, mtime))
    return path


class _FakeContext:
    def __init__(self) -> None:
        self.items = [Message(role="system", content="current system")]
        self.running_context_usage = 0
        self.recompute_calls: list[str] = []

    def _recompute_usage(self, model_name: str) -> None:
        self.recompute_calls.append(model_name)
        self.running_context_usage = 123


class _FakeSession:
    def __init__(self, *, user_email: str | None = "user-a@example.com") -> None:
        self.context_manager = _FakeContext()
        self.config = SimpleNamespace(model_name="databricks/databricks-claude-opus-4-7")
        self.session_id = "current-session"
        self.session_start_time = "2026-01-02T00:00:00"
        self.user_email = user_email
        self.logged_events: list[dict] = []
        self._local_save_path: str | None = None
        self.turn_count = 0
        self.last_auto_save_turn = 0
        self.pending_approval: dict | None = {"tool_calls": ["pending"]}

    def update_model(self, model_name: str) -> None:
        self.config.model_name = model_name


# ── listing ────────────────────────────────────────────────────────────


def test_filesystem_listing_newest_first(tmp_path):
    log_dir = tmp_path / "session_logs"
    older = _write_session_log(
        log_dir, "older.json",
        session_id="older-session", content="older prompt",
        mtime=time.time() - 10,
    )
    newer = _write_session_log(
        log_dir, "newer.json",
        session_id="newer-session", content="newer prompt",
        mtime=time.time(),
    )

    # No user_id → straight to filesystem fallback.
    entries = session_resume.list_session_logs(user_id=None, directory=log_dir)

    assert [e.path for e in entries] == [newer, older]
    assert entries[0].session_id == "newer-session"
    assert entries[0].preview == "newer prompt"
    assert entries[0].source == "filesystem"


def test_lakebase_listing_wins_when_pool_returns_rows(tmp_path):
    """When Lakebase has rows for the user, they must be returned and the
    filesystem fallback must NOT be consulted at all (it would dilute the
    cross-device source of truth).
    """
    log_dir = tmp_path / "session_logs"
    _write_session_log(
        log_dir, "fs-only.json",
        session_id="fs-only", content="filesystem-only prompt",
        mtime=time.time(),
    )

    fake_rows = [
        {
            "session_id": "lake-1",
            "last_active_at": "2026-05-08T12:00:00",
            "model_name": "databricks/databricks-claude-opus-4-7",
            "message_count": 4,
            "preview": "from lakebase",
        }
    ]
    with patch.object(
        session_resume, "_list_from_lakebase", return_value=[
            SessionLogEntry(
                session_id=row["session_id"],
                session_start_time=None,
                session_end_time=row["last_active_at"],
                model_name=row["model_name"],
                message_count=row["message_count"],
                preview=row["preview"],
                mtime=0.0,
                path=None,
                source="lakebase",
            ) for row in fake_rows
        ],
    ):
        entries = session_resume.list_session_logs(
            user_id="user-a@example.com", directory=log_dir,
        )

    assert len(entries) == 1
    assert entries[0].session_id == "lake-1"
    assert entries[0].source == "lakebase"
    # Filesystem entry must NOT leak through.
    assert all(e.session_id != "fs-only" for e in entries)


def test_lakebase_empty_falls_back_to_filesystem(tmp_path):
    log_dir = tmp_path / "session_logs"
    _write_session_log(
        log_dir, "fs.json", session_id="fs-session", content="fs prompt",
        mtime=time.time(),
    )
    with patch.object(session_resume, "_list_from_lakebase", return_value=[]):
        entries = session_resume.list_session_logs(
            user_id="user-a@example.com", directory=log_dir,
        )
    assert len(entries) == 1
    assert entries[0].source == "filesystem"


# ── resolve_session_arg ────────────────────────────────────────────────


def _entry(session_id: str, path: Path | None = None, source: str = "lakebase") -> SessionLogEntry:
    return SessionLogEntry(
        session_id=session_id, session_start_time=None, session_end_time=None,
        model_name=None, message_count=0, preview="", mtime=0.0,
        path=path, source=source,
    )


def test_resolve_by_index():
    entries = [_entry("a"), _entry("b"), _entry("c")]
    assert session_resume.resolve_session_arg("2", entries).session_id == "b"
    assert session_resume.resolve_session_arg("99", entries) is None


def test_resolve_by_id_prefix():
    entries = [_entry("abc123def"), _entry("xyz789")]
    assert session_resume.resolve_session_arg("abc", entries).session_id == "abc123def"
    # Ambiguous prefix yields None.
    assert session_resume.resolve_session_arg("", entries) is None


def test_resolve_by_filesystem_path(tmp_path):
    log = _write_session_log(
        tmp_path, "x.json", session_id="x-id", content="hi", mtime=time.time(),
    )
    entries = [_entry("x-id", path=log, source="filesystem")]
    out = session_resume.resolve_session_arg(str(log), entries)
    assert out.session_id == "x-id"


# ── restore ────────────────────────────────────────────────────────────


def test_restore_continues_when_user_id_matches(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir, "match.json",
        session_id="prior-session", content="task X",
        mtime=time.time(), user_id="user-a@example.com",
    )
    entry = _entry("prior-session", path=path, source="filesystem")
    session = _FakeSession(user_email="user-a@example.com")

    result = session_resume.restore_session_from_entry(session, entry)

    assert result["forked"] is False
    assert session.session_id == "prior-session"
    # Restored messages drop the system entry but keep user turn.
    assert any(m.role == "user" for m in session.context_manager.items[1:])
    # turn_count reflects the single user turn in the log.
    assert session.turn_count == 1
    # Pending approval cleared so prior-session UI state doesn't leak.
    assert session.pending_approval is None
    # _local_save_path forked so the next save won't overwrite source.
    assert session._local_save_path is None
    # Resume marker appended for trajectory cost-accounting.
    assert session.logged_events[0]["event_type"] == "resumed_from"
    assert session.logged_events[0]["data"]["forked"] is False


def test_restore_forks_when_user_id_differs(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir, "other.json",
        session_id="someone-elses-session", content="not mine",
        mtime=time.time(), user_id="other-user@example.com",
    )
    entry = _entry("someone-elses-session", path=path, source="filesystem")
    session = _FakeSession(user_email="user-a@example.com")
    original_id = session.session_id

    result = session_resume.restore_session_from_entry(session, entry)

    assert result["forked"] is True
    # Session id MUST stay put — we're forking, not continuing.
    assert session.session_id == original_id
    assert session.logged_events[0]["data"]["forked"] is True


def test_restore_flags_invalid_saved_model(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir, "badmodel.json",
        session_id="badmodel-session", content="task",
        mtime=time.time(), user_id="user-a@example.com",
        model_name="not-a-real/model-prefix",
    )
    entry = _entry("badmodel-session", path=path, source="filesystem")
    session = _FakeSession(user_email="user-a@example.com")
    original_model = session.config.model_name

    with patch.object(session_resume, "is_valid_model_id", return_value=False):
        result = session_resume.restore_session_from_entry(session, entry)

    assert result["invalid_saved_model"] == "not-a-real/model-prefix"
    # Current model preserved when saved id failed validation.
    assert session.config.model_name == original_model


def test_restore_detects_redacted_content(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir, "redacted.json",
        session_id="redacted-session", content="bearer [REDACTED_TOKEN]",
        mtime=time.time(), user_id="user-a@example.com",
    )
    entry = _entry("redacted-session", path=path, source="filesystem")
    session = _FakeSession(user_email="user-a@example.com")

    result = session_resume.restore_session_from_entry(session, entry)

    assert result["had_redacted_content"] is True


def test_restore_raises_on_missing_messages_array(tmp_path):
    log_dir = tmp_path / "session_logs"
    bad = log_dir / "bad.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    bad.write_text(json.dumps({"session_id": "bad", "messages": "not-a-list"}))
    entry = _entry("bad", path=bad, source="filesystem")
    session = _FakeSession()

    try:
        session_resume.restore_session_from_entry(session, entry)
    except ValueError as e:
        assert "messages array" in str(e)
    else:
        raise AssertionError("Expected ValueError")
