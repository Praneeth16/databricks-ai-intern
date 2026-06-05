"""Unit tests for backend/lakebase.py.

Lakebase persistence is best-effort: when ML_INTERN_LAKEBASE_INSTANCE isn't
configured (unit tests, local dev) every helper must no-op silently rather
than raise.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_BACKEND = str(Path(__file__).resolve().parents[2] / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import lakebase  # noqa: E402


def setup_function():
    lakebase.shutdown()


def _config(lakebase_instance: str | None = None):
    from agent.config import Config, DatabricksConfig

    return Config(
        model_name="databricks/databricks-claude-opus-4",
        databricks=DatabricksConfig(lakebase_instance=lakebase_instance),
    )


def test_init_returns_false_without_instance():
    assert lakebase.init(_config()) is False
    assert lakebase.get_pool() is None


def test_helpers_noop_without_pool():
    # No raises even when pool isn't initialised.
    lakebase.upsert_session(
        session_id="s1", user_id="u1", user_email="u@x", model_name="m", is_active=True,
    )
    lakebase.mark_session_inactive("s1")


def test_init_swallows_pool_construction_failure():
    cfg = _config(lakebase_instance="ml-intern-state")
    with patch(
        "agent.core.db_client.build_lakebase_conninfo",
        side_effect=RuntimeError("no creds"),
    ):
        ok = lakebase.init(cfg)
    assert ok is False
    assert lakebase.get_pool() is None
