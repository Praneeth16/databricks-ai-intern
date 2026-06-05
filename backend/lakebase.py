"""Lakebase (managed Postgres) connection pool for the backend.

Lakebase issues short-lived OAuth tokens (~1h). The pool's ``max_lifetime``
recycles connections before tokens expire so the agent never sees an
auth-related disconnect mid-turn. Conninfo is materialised lazily via
``db_client.build_lakebase_conninfo`` — calling it again refreshes the token
on the next pool-fill.

The pool is used by ``session_manager`` (P8) to persist session metadata so
backend restarts don't drop the session list, and the dashboard can join
sessions to traces in MLflow.

If ``ML_INTERN_LAKEBASE_INSTANCE`` isn't configured (local dev, unit tests),
``init`` is a no-op and ``get_pool()`` returns None — callers should check.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.config import Config
from agent.core import db_client

logger = logging.getLogger(__name__)

_POOL = None  # type: ignore[var-annotated]


def init(config: Config) -> bool:
    """Build the connection pool. Idempotent. Returns True on success."""
    global _POOL
    if _POOL is not None:
        return True
    settings = db_client.resolve_settings(config)
    if not settings.lakebase_instance:
        logger.info("Lakebase not configured — skipping pool init.")
        return False
    try:
        from psycopg_pool import ConnectionPool

        def _conninfo() -> str:
            # Re-resolved on each refill so the OAuth token can rotate.
            return db_client.build_lakebase_conninfo(settings)

        _POOL = ConnectionPool(
            conninfo=_conninfo(),
            min_size=1,
            max_size=10,
            max_lifetime=2700,  # 45 min — tokens last ~1h, leave a margin
            kwargs={"autocommit": True},
            open=True,
        )
        _ensure_schema(_POOL)
        logger.info("Lakebase pool initialised (instance=%s).", settings.lakebase_instance)
        return True
    except Exception as e:
        logger.warning("Lakebase pool init failed (%s) — proceeding without persistence.", e)
        _POOL = None
        return False


def shutdown() -> None:
    global _POOL
    if _POOL is None:
        return
    try:
        _POOL.close()
    except Exception as e:
        logger.debug("Lakebase pool close suppressed: %s", e)
    _POOL = None


def get_pool():
    """Return the active pool or None if Lakebase isn't configured."""
    return _POOL


def _ensure_schema(pool) -> None:
    """Create the tables the backend persists to. Idempotent."""
    with pool.connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_intern_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                user_email TEXT,
                model_name TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_active_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                message_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        # /resume reads message + event history out of this column so a CLI
        # user can pick up a conversation that started in the frontend (and
        # vice versa). Idempotent — first deploy adds the column, later
        # deploys are no-op.
        conn.execute("""
            ALTER TABLE ml_intern_sessions
                ADD COLUMN IF NOT EXISTS trajectory JSONB
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS ml_intern_sessions_user_idx
                ON ml_intern_sessions(user_id, last_active_at DESC)
        """)


def upsert_session(*, session_id: str, user_id: str, user_email: str | None,
                   model_name: str, message_count: int = 0,
                   is_active: bool = True) -> None:
    pool = get_pool()
    if pool is None:
        return
    try:
        with pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO ml_intern_sessions
                  (session_id, user_id, user_email, model_name, message_count, is_active)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    user_email = EXCLUDED.user_email,
                    model_name = EXCLUDED.model_name,
                    message_count = EXCLUDED.message_count,
                    is_active = EXCLUDED.is_active,
                    last_active_at = now()
                """,
                (session_id, user_id, user_email, model_name, message_count, is_active),
            )
    except Exception as e:
        logger.debug("upsert_session suppressed: %s", e)


def mark_session_inactive(session_id: str) -> None:
    pool = get_pool()
    if pool is None:
        return
    try:
        with pool.connection() as conn:
            conn.execute(
                "UPDATE ml_intern_sessions SET is_active = FALSE, last_active_at = now() "
                "WHERE session_id = %s",
                (session_id,),
            )
    except Exception as e:
        logger.debug("mark_session_inactive suppressed: %s", e)


def save_trajectory(
    *,
    session_id: str,
    user_id: str,
    user_email: str | None,
    model_name: str,
    trajectory: dict,
) -> bool:
    """Persist the full agent trajectory (messages + events + metadata) into
    the same ``ml_intern_sessions`` row that ``upsert_session`` keeps current.

    Used as the storage layer for ``/resume``. JSONB lets us pick up an
    older conversation from either the CLI or the frontend without
    needing a separate file store.

    Returns True when the row was written. False on missing pool, or
    when the write was suppressed by an exception (caller can fall back
    to filesystem-only recovery).
    """
    import json

    pool = get_pool()
    if pool is None:
        return False
    message_count = len(trajectory.get("messages") or [])
    try:
        with pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO ml_intern_sessions
                  (session_id, user_id, user_email, model_name,
                   message_count, is_active, trajectory)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (session_id) DO UPDATE SET
                    user_email = EXCLUDED.user_email,
                    model_name = EXCLUDED.model_name,
                    message_count = EXCLUDED.message_count,
                    trajectory = EXCLUDED.trajectory,
                    last_active_at = now()
                """,
                (
                    session_id, user_id, user_email, model_name,
                    message_count, True, json.dumps(trajectory),
                ),
            )
        return True
    except Exception as e:
        logger.debug("save_trajectory suppressed: %s", e)
        return False


def list_sessions(user_id: str, limit: int = 20) -> list[dict]:
    """List recent sessions for ``user_id``, newest first.

    Returns metadata only (session_id, last_active_at, model_name,
    message_count, first-user preview). Callers fetch the full trajectory
    via :func:`load_trajectory` once the user picks one — the picker
    grid stays small even when a user has hundreds of past sessions.
    """
    pool = get_pool()
    if pool is None:
        return []
    try:
        with pool.connection() as conn:
            rows = list(conn.execute(
                """
                SELECT session_id, last_active_at, model_name, message_count,
                       trajectory -> 'messages' AS messages
                  FROM ml_intern_sessions
                 WHERE user_id = %s
                   AND trajectory IS NOT NULL
                 ORDER BY last_active_at DESC
                 LIMIT %s
                """,
                (user_id, limit),
            ).fetchall())
    except Exception as e:
        logger.debug("list_sessions suppressed: %s", e)
        return []

    out: list[dict] = []
    for sid, last_active, model, msg_count, messages in rows:
        preview = _first_user_preview(messages)
        out.append({
            "session_id": sid,
            "last_active_at": last_active.isoformat() if last_active else None,
            "model_name": model,
            "message_count": msg_count or 0,
            "preview": preview,
        })
    return out


def load_trajectory(session_id: str) -> dict | None:
    """Fetch the full saved trajectory for ``session_id``, or None."""
    pool = get_pool()
    if pool is None:
        return None
    try:
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT user_id, trajectory FROM ml_intern_sessions "
                "WHERE session_id = %s",
                (session_id,),
            ).fetchone()
    except Exception as e:
        logger.debug("load_trajectory suppressed: %s", e)
        return None
    if row is None:
        return None
    user_id, trajectory = row
    if trajectory is None:
        return None
    # psycopg already decodes JSONB to dict via the default adapter; some
    # adapters return a str — coerce in that case.
    if isinstance(trajectory, str):
        import json
        try:
            trajectory = json.loads(trajectory)
        except Exception:
            return None
    if not isinstance(trajectory, dict):
        return None
    # Stamp the user_id alongside so the resume path can decide whether to
    # continue or fork without a second query.
    trajectory.setdefault("user_id", user_id)
    return trajectory


def _first_user_preview(messages, max_chars: int = 72) -> str:
    """Extract a short one-line preview of the first user message."""
    if not isinstance(messages, list):
        return "(no preview)"
    for raw in messages:
        if not isinstance(raw, dict) or raw.get("role") != "user":
            continue
        content = raw.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    v = block.get("text") or block.get("content")
                    if isinstance(v, str):
                        parts.append(v)
                elif isinstance(block, str):
                    parts.append(block)
            text = " ".join(parts)
        text = " ".join(text.split())
        if not text:
            continue
        if len(text) > max_chars:
            return text[: max_chars - 1].rstrip() + "…"
        return text
    return "(no user prompt preview)"
