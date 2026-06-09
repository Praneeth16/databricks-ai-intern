"""Shared session→workspace plumbing for agent-facing tool handlers.

Every handler does the same dance: load the session's config (or the default
config file), resolve Databricks settings, and build a workspace client —
preferring the per-user OBO token when the session carries one. The ledger
and jobs-tool factories live here too so ``sweep_tool``, ``model_serving_tool``
and friends share one implementation.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from agent.core import db_client
from agent.core.experiment_ledger import ExperimentLedger


def _load_default_config():
    from agent.config import load_config

    cfg_path = os.environ.get(
        "DATABRICKS_AI_INTERN_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json"),
    )
    return load_config(cfg_path)


def resolve_session_settings(session: Any) -> tuple[db_client.DatabricksSettings, Optional[str]]:
    """Resolve (settings, OBO token) from the session, falling back to the default config."""
    cfg = session.config if session and getattr(session, "config", None) else _load_default_config()
    settings = db_client.resolve_settings(cfg)
    token = getattr(session, "databricks_user_token", None) if session else None
    return settings, token


def get_session_workspace_client(session: Any) -> tuple[Any, db_client.DatabricksSettings, Optional[str]]:
    """Build a workspace client for the session, preferring OBO.

    Returns ``(wc, settings, token)`` where ``token`` is the OBO token used
    (None when the session has none and the App SP client was built).
    """
    settings, token = resolve_session_settings(session)
    if token and settings.host:
        wc = db_client.get_workspace_client_for_user(token, settings.host)
    else:
        wc = db_client.get_workspace_client(settings)
    return wc, settings, token


def _get_ledger(session: Any) -> ExperimentLedger:
    """Build the ledger from the session's resolved settings.

    Factored out so tests can monkeypatch ``<tool module>._get_ledger`` to a
    JSONL ledger on a tmp path (no workspace).
    """
    settings, token = resolve_session_settings(session)
    return ExperimentLedger(settings=settings, user_token=token)


async def _get_jobs_tool(session: Any):
    """Build a ``DatabricksJobsTool`` from the session, preferring OBO.

    Factored out so tests can monkeypatch ``<tool module>._get_jobs_tool`` to
    return ``None`` and skip Databricks entirely.
    """
    from agent.tools.databricks_jobs_tool import DatabricksJobsTool

    wc, settings, _ = get_session_workspace_client(session)

    user_email = getattr(session, "user_email", None) if session else None
    if not user_email:
        try:
            me = await asyncio.to_thread(wc.current_user.me)
            user_email = me.user_name or (me.emails[0].value if me.emails else None)
        except Exception:
            user_email = None

    return DatabricksJobsTool(wc=wc, settings=settings, user_email=user_email, session=session)
