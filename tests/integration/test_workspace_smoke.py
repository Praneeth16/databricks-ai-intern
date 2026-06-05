"""Smoke tests against a real Databricks workspace.

Auto-skipped when ``DATABRICKS_HOST`` is unset. Every test exercises a
read-only API path so running them carries near-zero cost.
"""

from __future__ import annotations

import pytest


def test_current_user_resolves(workspace_client):
    me = workspace_client.current_user.me()
    assert me.user_name


def test_serving_endpoints_listable(workspace_client):
    eps = list(workspace_client.serving_endpoints.list())
    # Workspaces with no endpoints still return [] (not 403).
    assert isinstance(eps, list)


def test_uc_catalog_visible(workspace_client, databricks_settings):
    from databricks.sdk.errors import NotFound

    try:
        info = workspace_client.catalogs.get(name=databricks_settings.uc_catalog)
    except NotFound:
        pytest.skip(
            f"Catalog {databricks_settings.uc_catalog!r} not provisioned — "
            "run `databricks bundle deploy` first."
        )
    assert info.name == databricks_settings.uc_catalog


def test_serving_endpoint_for_default_model(workspace_client, databricks_settings):
    """Verify the agent's default model id maps to an endpoint we can query."""
    from agent.core import model_catalog

    info = model_catalog.lookup("databricks/databricks-claude-opus-4")
    if info is None:
        pytest.skip("databricks-claude-opus-4 endpoint not available in this workspace.")
    assert info.is_chat


def test_volume_listable(workspace_client, databricks_settings):
    """List the configured scratch volume root — proves Files API is reachable."""
    from databricks.sdk.errors import NotFound

    try:
        list(workspace_client.files.list_directory_contents(
            directory_path=databricks_settings.volume_root,
        ))
    except NotFound:
        pytest.skip(
            f"Volume {databricks_settings.volume_root} not yet provisioned. "
            "Run `databricks bundle deploy` first."
        )
