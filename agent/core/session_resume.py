"""Reload a previously saved session into the active CLI/backend session.

Storage layout: Lakebase row in ``databricks_ai_intern_sessions`` carries the full
trajectory JSONB (messages + events + metadata). Frontend, CLI, and
future API replay all read the same Postgres row, so a conversation that
started in the browser can be picked up from ``databricks-ai-intern
--resume <id>`` on the user's laptop, and vice versa.

Filesystem fallback: when ``backend.lakebase.get_pool()`` returns None
(no ``DATABRICKS_AI_INTERN_LAKEBASE_INSTANCE`` configured — local dev, unit tests,
offline CLI), the list/load helpers degrade to scanning ``session_logs/``
on disk, the same shape the upstream HF#233 port used. Keeps the CLI
usable when the workspace isn't reachable.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from litellm import Message

from agent.core.model_switcher import is_valid_model_id
from agent.core.session import DEFAULT_SESSION_LOG_DIR

logger = logging.getLogger(__name__)

_REDACTED_MARKER = re.compile(r"\[REDACTED_[A-Z_]+\]")


@dataclass
class SessionLogEntry:
    """Metadata for a resumable session.

    ``path`` points to the on-disk JSON when the entry came from the
    filesystem fallback. For Lakebase-backed entries ``path`` is None and
    ``session_id`` is the row identifier.
    """

    session_id: str
    session_start_time: str | None
    session_end_time: str | None
    model_name: str | None
    message_count: int
    preview: str
    mtime: float
    path: Path | None = None
    source: str = "lakebase"  # "lakebase" or "filesystem"


def _message_preview(content: Any, max_chars: int = 72) -> str:
    """Return a one-line preview for string or block content."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("text") or block.get("content")
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(block, str):
                parts.append(block)
        text = " ".join(parts)
    else:
        text = ""
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _first_user_preview(messages: list[Any]) -> str:
    for raw in messages:
        if isinstance(raw, dict) and raw.get("role") == "user":
            preview = _message_preview(raw.get("content"))
            if preview:
                return preview
    return "(no user prompt preview)"


def _list_from_lakebase(user_id: str, limit: int) -> list[SessionLogEntry]:
    """Pull recent sessions for ``user_id`` from Lakebase. Empty on miss."""
    try:
        from backend import lakebase
    except Exception:
        return []
    rows = lakebase.list_sessions(user_id, limit=limit)
    out: list[SessionLogEntry] = []
    for row in rows:
        mtime = 0.0
        last = row.get("last_active_at")
        if isinstance(last, str):
            try:
                mtime = datetime.fromisoformat(last).timestamp()
            except ValueError:
                pass
        out.append(SessionLogEntry(
            session_id=row.get("session_id", ""),
            session_start_time=None,
            session_end_time=row.get("last_active_at"),
            model_name=row.get("model_name"),
            message_count=int(row.get("message_count", 0) or 0),
            preview=row.get("preview") or "(no preview)",
            mtime=mtime,
            path=None,
            source="lakebase",
        ))
    return out


def _list_from_filesystem(
    directory: Path,
) -> list[SessionLogEntry]:
    """Fallback path: scan ``directory`` for session_*.json files."""
    if not directory.exists():
        return []
    entries: list[SessionLogEntry] = []
    for path in directory.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        messages = data.get("messages") or []
        if not isinstance(messages, list):
            continue

        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            session_id = path.stem

        stat = path.stat()
        entries.append(SessionLogEntry(
            session_id=session_id,
            session_start_time=data.get("session_start_time"),
            session_end_time=data.get("session_end_time"),
            model_name=data.get("model_name"),
            message_count=len(messages),
            preview=_first_user_preview(messages),
            mtime=stat.st_mtime,
            path=path,
            source="filesystem",
        ))
    entries.sort(key=lambda item: item.mtime, reverse=True)
    return entries


def list_session_logs(
    user_id: str | None = None,
    *,
    directory: Path = DEFAULT_SESSION_LOG_DIR,
    limit: int = 20,
) -> list[SessionLogEntry]:
    """Return resumable sessions, Lakebase first then filesystem fallback.

    ``user_id`` is required for the Lakebase query (Postgres index is on
    ``(user_id, last_active_at)``). When None or when Lakebase is
    unreachable, returns the filesystem listing instead.
    """
    if user_id:
        rows = _list_from_lakebase(user_id, limit)
        if rows:
            return rows
    return _list_from_filesystem(directory)


def format_session_log_entry(index: int, entry: SessionLogEntry) -> str:
    timestamp = entry.session_end_time or entry.session_start_time
    label = "unknown time"
    if isinstance(timestamp, str) and timestamp:
        try:
            label = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            label = timestamp[:16]
    short_id = entry.session_id[:8]
    model = entry.model_name or "unknown model"
    return (
        f"{index:>2}. {label}  {short_id}  "
        f"{entry.message_count} msgs  {model}\n"
        f"    {entry.preview}"
    )


def resolve_session_arg(
    arg: str,
    entries: list[SessionLogEntry],
) -> SessionLogEntry | None:
    """Resolve ``/resume <arg>`` as index, exact id, or id-prefix.

    Path-based resolution stays available only for filesystem entries so
    a user copy-pasting a ``session_logs/...json`` path still works in
    offline mode. Lakebase entries are id-only.
    """
    value = arg.strip()
    if not value:
        return None

    if value.isdigit():
        idx = int(value)
        if 1 <= idx <= len(entries):
            return entries[idx - 1]

    # Filesystem path lookup.
    candidate = Path(value).expanduser()
    if candidate.exists() and candidate.is_file():
        for entry in entries:
            if entry.path == candidate:
                return entry
        # Path matches a file we didn't list (e.g. user passed an arbitrary
        # path). Build a synthetic entry on the fly.
        try:
            with open(candidate) as f:
                data = json.load(f)
            messages = data.get("messages") or []
            return SessionLogEntry(
                session_id=data.get("session_id") or candidate.stem,
                session_start_time=data.get("session_start_time"),
                session_end_time=data.get("session_end_time"),
                model_name=data.get("model_name"),
                message_count=len(messages) if isinstance(messages, list) else 0,
                preview=(
                    _first_user_preview(messages)
                    if isinstance(messages, list) else "(no preview)"
                ),
                mtime=candidate.stat().st_mtime,
                path=candidate,
                source="filesystem",
            )
        except Exception as e:
            logger.debug("resolve_session_arg: cannot read %s: %s", candidate, e)

    matches = [e for e in entries if e.session_id.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    return None


def _load_trajectory(entry: SessionLogEntry) -> dict[str, Any] | None:
    """Fetch the saved trajectory for ``entry`` from its source layer."""
    if entry.source == "lakebase":
        try:
            from backend import lakebase
        except Exception:
            return None
        return lakebase.load_trajectory(entry.session_id)
    if entry.path is not None:
        try:
            with open(entry.path) as f:
                return json.load(f)
        except Exception as e:
            logger.debug("_load_trajectory: cannot read %s: %s", entry.path, e)
            return None
    return None


def _turn_count_from_messages(messages: list[Any]) -> int:
    return sum(
        1 for raw in messages if isinstance(raw, dict) and raw.get("role") == "user"
    )


def _has_redacted_content(messages: list[Any]) -> bool:
    """Whether any message body contains a ``[REDACTED_*]`` marker."""
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content")
        if isinstance(content, str) and _REDACTED_MARKER.search(content):
            return True
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                    if isinstance(text, str) and _REDACTED_MARKER.search(text):
                        return True
    return False


def restore_session_from_entry(session: Any, entry: SessionLogEntry) -> dict[str, Any]:
    """Replace the active session context with messages from ``entry``.

    Continues the saved session (reuses its id) when the log's user_id
    matches the current session, and forks otherwise: the caller's
    session id stays put and future saves go to a fresh slot rather than
    overwriting the source.

    Returns metadata for the ``resume_complete`` event.
    """
    data = _load_trajectory(entry)
    if data is None:
        raise ValueError(f"Cannot load trajectory for {entry.session_id}")

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("Selected log does not contain a messages array")

    restored_messages: list[Message] = []
    dropped_count = 0
    for raw in raw_messages:
        if not isinstance(raw, dict) or raw.get("role") == "system":
            continue
        try:
            restored_messages.append(Message.model_validate(raw))
        except Exception as e:
            dropped_count += 1
            logger.warning("Dropping malformed message during resume: %s", e)

    if not restored_messages:
        raise ValueError("Selected log has no restorable non-system messages")

    cm = session.context_manager
    system_msg = cm.items[0] if cm.items and cm.items[0].role == "system" else None
    cm.items = ([system_msg] if system_msg else []) + restored_messages

    # Validate saved model id before switching. ``update_model`` doesn't
    # check availability; an unrecognised id silently sticks and the next
    # LLM call fails with a cryptic routing error.
    saved_model = data.get("model_name")
    invalid_saved_model: str | None = None
    if isinstance(saved_model, str) and saved_model:
        if is_valid_model_id(saved_model):
            session.update_model(saved_model)
        else:
            invalid_saved_model = saved_model
            logger.warning(
                "Saved log model %r failed format validation; keeping %r",
                saved_model,
                session.config.model_name,
            )

    cm._recompute_usage(session.config.model_name)

    saved_session_id = data.get("session_id")
    saved_user_id = data.get("user_id")
    # Our identity surface is user_email today (from Apps OBO / SDK chain).
    # Match on either user_id or user_email to cover both backend-saved and
    # legacy filesystem-saved trajectories.
    current_user_id = getattr(session, "user_id", None) or getattr(session, "user_email", None)
    is_continuation = (
        saved_user_id is not None and saved_user_id == current_user_id
    )

    if is_continuation:
        if isinstance(saved_session_id, str) and saved_session_id:
            session.session_id = saved_session_id
        session.session_start_time = (
            data.get("session_start_time") or session.session_start_time
        )

    # Always fork the on-disk save path. The source log is an immutable
    # snapshot; the next save fork-writes to a fresh filename so we don't
    # destroy the original.
    session._local_save_path = None

    saved_event_count = (
        len(data.get("events", [])) if isinstance(data.get("events"), list) else 0
    )
    session.logged_events = [
        {
            "timestamp": datetime.now().isoformat(),
            "event_type": "resumed_from",
            "data": {
                "session_id": saved_session_id if isinstance(saved_session_id, str) else None,
                "source": entry.source,
                "path": str(entry.path) if entry.path else None,
                "original_event_count": saved_event_count,
                "forked": not is_continuation,
            },
        }
    ]
    session.turn_count = _turn_count_from_messages(raw_messages)
    session.last_auto_save_turn = session.turn_count
    session.pending_approval = None

    return {
        "session_id": entry.session_id,
        "source": entry.source,
        "path": str(entry.path) if entry.path else None,
        "restored_count": len(restored_messages),
        "dropped_count": dropped_count,
        "model_name": session.config.model_name,
        "invalid_saved_model": invalid_saved_model,
        "forked": not is_continuation,
        "had_redacted_content": _has_redacted_content(raw_messages),
    }
