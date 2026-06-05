"""Unit tests for agent.core.model_catalog and the databricks/ branch in
agent.core.llm_params.

The catalog must:
  - Strip databricks/ prefixes on lookup.
  - Cache results in memory.
  - Surface fuzzy suggestions on typos.
  - Survive a SDK failure (returns empty rather than raising).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.core import llm_params, model_catalog


def setup_function():
    model_catalog.reset_cache_for_tests()


def _ep(name, ready="READY", task="llm/v1/chat", entities=("a",)):
    ep = MagicMock()
    ep.name = name
    state = MagicMock()
    state.ready = ready
    state.config_update = None
    ep.state = state
    ep.task = task
    cfg = MagicMock()
    se = []
    for e in entities:
        s = MagicMock()
        s.entity_name = e
        s.name = e
        se.append(s)
    cfg.served_entities = se
    ep.config = cfg
    ep.creator = "user"
    et = MagicMock()
    et.value = "FOUNDATION_MODEL_API"
    ep.endpoint_type = et
    ep.creation_timestamp = 1
    return ep


def test_lookup_strips_databricks_prefix_and_caches():
    wc = MagicMock()
    wc.serving_endpoints.list.return_value = iter([
        _ep("databricks-claude-opus-4"),
        _ep("databricks-meta-llama-3-3-70b-instruct"),
    ])
    with patch("agent.core.model_catalog.db_client.get_workspace_client", return_value=wc):
        info = model_catalog.lookup("databricks/databricks-claude-opus-4")
        info2 = model_catalog.lookup("databricks-claude-opus-4")  # bare name
    assert info is not None
    assert info.name == "databricks-claude-opus-4"
    assert info.is_ready
    assert info.is_chat
    assert info2.name == info.name
    # cached after first list
    assert wc.serving_endpoints.list.call_count == 1


def test_lookup_returns_none_when_endpoint_missing():
    wc = MagicMock()
    wc.serving_endpoints.list.return_value = iter([])
    with patch("agent.core.model_catalog.db_client.get_workspace_client", return_value=wc):
        assert model_catalog.lookup("databricks/no-such-endpoint") is None


def test_fuzzy_suggest():
    wc = MagicMock()
    wc.serving_endpoints.list.return_value = iter([
        _ep("databricks-claude-opus-4"),
        _ep("databricks-claude-sonnet-4"),
    ])
    with patch("agent.core.model_catalog.db_client.get_workspace_client", return_value=wc):
        suggestions = model_catalog.fuzzy_suggest("databricks/databricks-clude-opus-4")
    assert "databricks-claude-opus-4" in suggestions


def test_fetch_swallows_sdk_failure():
    with patch(
        "agent.core.model_catalog.db_client.get_workspace_client",
        side_effect=RuntimeError("auth"),
    ):
        endpoints = model_catalog.list_endpoints()
    assert endpoints == []  # no raise


# ---------------------------------------------------------------------------
# llm_params.databricks branch
# ---------------------------------------------------------------------------


def test_databricks_provider_minimal_params():
    p = llm_params._resolve_llm_params("databricks/databricks-claude-opus-4")
    assert p == {"model": "databricks/databricks-claude-opus-4"}


def test_databricks_reasoning_effort_in_extra_body():
    p = llm_params._resolve_llm_params(
        "databricks/databricks-claude-opus-4", reasoning_effort="high",
    )
    assert p["extra_body"] == {"reasoning_effort": "high"}


def test_databricks_minimal_normalized_to_low():
    p = llm_params._resolve_llm_params(
        "databricks/databricks-claude-opus-4", reasoning_effort="minimal",
    )
    assert p["extra_body"] == {"reasoning_effort": "low"}


def test_databricks_max_silently_dropped_in_non_strict():
    p = llm_params._resolve_llm_params(
        "databricks/databricks-claude-opus-4", reasoning_effort="max",
    )
    assert "extra_body" not in p


def test_databricks_max_strict_raises():
    with pytest.raises(llm_params.UnsupportedEffortError):
        llm_params._resolve_llm_params(
            "databricks/x", reasoning_effort="max", strict=True,
        )
