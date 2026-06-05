"""Integration test scaffolding.

Every test in this directory needs a live Databricks workspace. The
``databricks_settings`` fixture is the single gate: it auto-skips the
test if ``DATABRICKS_HOST`` isn't set so unit-only CI runs stay green.

To run integration tests::

    export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
    export DATABRICKS_TOKEN=dapi...
    export ML_INTERN_UC_CATALOG=ml_intern        # optional overrides
    export ML_INTERN_UC_SCHEMA=agent
    export ML_INTERN_UC_VOLUME=scratch
    export DATABRICKS_WAREHOUSE_ID=<id>
    uv run pytest tests/integration
"""

from __future__ import annotations

import os

import pytest


def _have_workspace_creds() -> bool:
    return bool(
        os.environ.get("DATABRICKS_HOST")
        or os.environ.get("DATABRICKS_CONFIG_PROFILE")
        or (os.environ.get("DATABRICKS_CLIENT_ID") and os.environ.get("DATABRICKS_CLIENT_SECRET"))
    )


@pytest.fixture(scope="session")
def databricks_settings():
    if not _have_workspace_creds():
        pytest.skip(
            "No workspace credentials (DATABRICKS_HOST / DATABRICKS_CONFIG_PROFILE / M2M) — "
            "skipping integration test."
        )
    from agent.config import load_config
    from agent.core import db_client

    # When only a profile is set, the SDK resolves the host from
    # ~/.databrickscfg. Backfill DATABRICKS_HOST so resolve_settings can
    # populate ``settings.host`` (used by URL builders and SQL helpers).
    if not os.environ.get("DATABRICKS_HOST"):
        try:
            from databricks.sdk.core import Config as SdkConfig

            cfg = SdkConfig()
            if cfg.host:
                os.environ["DATABRICKS_HOST"] = cfg.host
        except Exception:
            pass

    cfg_path = os.environ.get(
        "ML_INTERN_CONFIG_PATH",
        os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json",
        ),
    )
    return db_client.resolve_settings(load_config(cfg_path))


@pytest.fixture(scope="session")
def workspace_client(databricks_settings):
    from agent.core import db_client

    return db_client.get_workspace_client(databricks_settings)
