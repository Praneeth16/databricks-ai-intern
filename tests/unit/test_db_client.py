"""Unit tests for agent.core.db_client.

Coverage:
    - resolve_settings: env overrides config-file values, empty-string → None
    - get_workspace_client: cached per process, host resolution
    - get_workspace_client_for_user: fresh client with OBO token
    - volume_root / full_schema derivations
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agent.config import Config, DatabricksConfig
from agent.core import db_client


def _make_config(**overrides) -> Config:
    db = DatabricksConfig(
        host=overrides.get("host"),
        warehouse_id=overrides.get("warehouse_id"),
        experiment_path=overrides.get("experiment_path", "/Shared/ml-intern"),
        uc_catalog=overrides.get("uc_catalog", "ml_intern"),
        uc_schema=overrides.get("uc_schema", "agent"),
        uc_volume=overrides.get("uc_volume", "scratch"),
        secret_scope=overrides.get("secret_scope", "ml-intern"),
        lakebase_instance=overrides.get("lakebase_instance"),
        instance_pool_id=overrides.get("instance_pool_id"),
    )
    return Config(model_name="databricks/databricks-claude-opus-4", databricks=db)


@pytest.fixture(autouse=True)
def _reset():
    db_client.reset_clients_for_tests()
    yield
    db_client.reset_clients_for_tests()


def test_resolve_settings_defaults_from_config():
    cfg = _make_config(host="https://ws.cloud.databricks.com/", warehouse_id="abc123")
    with patch.dict(os.environ, {}, clear=True):
        s = db_client.resolve_settings(cfg)
    assert s.host == "https://ws.cloud.databricks.com"  # trailing slash stripped
    assert s.warehouse_id == "abc123"
    assert s.uc_catalog == "ml_intern"
    assert s.full_schema == "ml_intern.agent"
    assert s.volume_root == "/Volumes/ml_intern/agent/scratch"


def test_resolve_settings_env_overrides_config():
    cfg = _make_config(host="https://config-host", warehouse_id="from-config")
    env = {
        "DATABRICKS_HOST": "https://env-host",
        "DATABRICKS_WAREHOUSE_ID": "from-env",
        "ML_INTERN_INSTANCE_POOL_ID": "pool-xyz",
    }
    with patch.dict(os.environ, env, clear=True):
        s = db_client.resolve_settings(cfg)
    assert s.host == "https://env-host"
    assert s.warehouse_id == "from-env"
    assert s.instance_pool_id == "pool-xyz"


def test_empty_string_env_coerced_to_none_in_model():
    # This happens at Config.model_validate time (the field_validator).
    db = DatabricksConfig(host="", warehouse_id="   ", lakebase_instance="")
    assert db.host is None
    assert db.warehouse_id is None
    assert db.lakebase_instance is None


def test_get_workspace_client_is_cached():
    cfg = _make_config(host="https://ws")
    with patch.dict(os.environ, {"DATABRICKS_HOST": "https://ws"}, clear=True), \
         patch("agent.core.db_client.WorkspaceClient") as MockWC:
        MockWC.return_value = MagicMock()
        s = db_client.resolve_settings(cfg)
        wc1 = db_client.get_workspace_client(s)
        wc2 = db_client.get_workspace_client(s)
    assert wc1 is wc2
    assert MockWC.call_count == 1


def test_get_workspace_client_for_user_is_not_cached():
    with patch("agent.core.db_client.WorkspaceClient") as MockWC, \
         patch("agent.core.db_client.SdkConfig") as MockSdkCfg:
        MockWC.return_value = MagicMock()
        db_client.get_workspace_client_for_user("token-a", "https://ws")
        db_client.get_workspace_client_for_user("token-b", "https://ws")
    # Two distinct user tokens → two distinct clients, each with its own SdkConfig.
    assert MockWC.call_count == 2
    assert MockSdkCfg.call_count == 2


def test_resolve_settings_missing_host_does_not_raise():
    cfg = _make_config()
    with patch.dict(os.environ, {}, clear=True):
        s = db_client.resolve_settings(cfg)
    assert s.host == ""
