"""Pre-call cost estimator tests (issue #16)."""

from __future__ import annotations

from types import SimpleNamespace

from agent.core.cost_estimation import (
    DATABRICKS_NODE_PRICE_USD_PER_HOUR,
    DEFAULT_SANDBOX_RESERVATION_HOURS,
    FMAPI_PRICE_USD_PER_MTOK,
    CostEstimate,
    estimate_jobs_cost,
    estimate_llm_cost,
    estimate_sandbox_cost,
    estimate_tool_cost,
    parse_duration_hours,
)


# ── parse_duration_hours ───────────────────────────────────────────────


def test_parse_duration_int_treated_as_seconds():
    assert parse_duration_hours(3600) == 1.0
    assert parse_duration_hours(1800) == 0.5


def test_parse_duration_string_suffixes():
    assert parse_duration_hours("30m") == 0.5
    assert parse_duration_hours("2h") == 2.0
    assert parse_duration_hours("1d") == 24.0
    # Default unit when no suffix is seconds, matching int handling.
    assert parse_duration_hours("3600") == 1.0


def test_parse_duration_returns_default_for_empty():
    assert parse_duration_hours(None) == 0.5  # DEFAULT_JOB_TIMEOUT_HOURS
    assert parse_duration_hours("") == 0.5


def test_parse_duration_rejects_garbage():
    assert parse_duration_hours("not-a-duration") is None
    assert parse_duration_hours(True) is None
    assert parse_duration_hours(-100) is None


# ── estimate_llm_cost ──────────────────────────────────────────────────


def test_estimate_llm_cost_claude_opus_matches_catalog():
    rates = FMAPI_PRICE_USD_PER_MTOK["databricks-claude-opus-4-7"]
    est = estimate_llm_cost(
        "databricks/databricks-claude-opus-4-7", 1_000_000, 1_000_000,
    )
    # 1M input + 1M output = input_rate + output_rate USD.
    assert est.estimated_cost_usd == round(rates["input"] + rates["output"], 6)
    assert est.billable is True


def test_estimate_llm_cost_zero_tokens_is_free_but_billable_only_if_nonzero():
    # Zero-token call (rare but possible — error or empty completion).
    est = estimate_llm_cost(
        "databricks/databricks-claude-opus-4-7", 0, 0,
    )
    assert est.estimated_cost_usd == 0.0
    # Catalog hit but no spend → billable False is fine; we accept either
    # since the YOLO gate only blocks on > cap, not on "billable but $0".
    assert est.billable is False


def test_estimate_llm_cost_unknown_model_returns_none():
    est = estimate_llm_cost("openai/gpt-5", 1000, 500)
    assert est.estimated_cost_usd is None
    assert est.billable is True  # safe default → human approval
    assert "No FMAPI price catalog entry" in (est.block_reason or "")


# ── estimate_jobs_cost ─────────────────────────────────────────────────


def test_estimate_jobs_uses_node_rate_and_timeout():
    est = estimate_jobs_cost({
        "kind": "serverless_gpu",
        "node_type_id": "GPU_1xA10",
        "timeout_seconds": 3600,
    })
    expected = DATABRICKS_NODE_PRICE_USD_PER_HOUR["GPU_1xA10"] * 1.0
    assert est.estimated_cost_usd == round(expected, 4)
    assert est.label == "GPU_1xA10"


def test_estimate_jobs_default_node_for_serverless_kind():
    # No explicit node + kind=serverless → catalog's "serverless" entry.
    est = estimate_jobs_cost({"kind": "serverless", "timeout_seconds": 1800})
    expected = DATABRICKS_NODE_PRICE_USD_PER_HOUR["serverless"] * 0.5
    assert est.estimated_cost_usd == round(expected, 4)


def test_estimate_jobs_unknown_node_returns_block_reason():
    est = estimate_jobs_cost({"node_type_id": "fictional-monster", "timeout_seconds": 600})
    assert est.estimated_cost_usd is None
    assert est.billable is True
    assert "price catalog" in (est.block_reason or "")


def test_estimate_jobs_garbage_timeout_returns_block_reason():
    est = estimate_jobs_cost({"node_type_id": "GPU_1xA10", "timeout": "next-tuesday"})
    assert est.estimated_cost_usd is None
    assert "timeout" in (est.block_reason or "")


# ── estimate_sandbox_cost ──────────────────────────────────────────────


def test_estimate_sandbox_returns_zero_when_session_already_has_one():
    fake_session = SimpleNamespace(sandbox=object())
    est = estimate_sandbox_cost({"hardware": "GPU_1xA10"}, session=fake_session)
    assert est.estimated_cost_usd == 0.0
    assert est.billable is False
    assert est.label == "existing"


def test_estimate_sandbox_uses_reservation_hours():
    est = estimate_sandbox_cost({"hardware": "GPU_1xA10"}, session=None)
    expected = (
        DATABRICKS_NODE_PRICE_USD_PER_HOUR["GPU_1xA10"]
        * DEFAULT_SANDBOX_RESERVATION_HOURS
    )
    assert est.estimated_cost_usd == round(expected, 4)


# ── estimate_tool_cost (dispatcher) ────────────────────────────────────


def test_dispatcher_routes_to_jobs_estimator():
    est = estimate_tool_cost(
        "databricks_jobs",
        {"node_type_id": "GPU_1xA10", "timeout_seconds": 1800},
    )
    assert est.label == "GPU_1xA10"


def test_dispatcher_treats_unknown_tool_as_free():
    """Free / read-only tools (uc_inspect_dataset, docs, github_*, etc.)
    fall into the default branch and report 0 cost so the YOLO gate
    auto-approves without further checks."""
    est = estimate_tool_cost("uc_inspect_dataset", {})
    assert est.estimated_cost_usd == 0.0
    assert est.billable is False


# ── Session.add_estimated_spend integration ────────────────────────────


def test_session_accumulator_is_none_safe():
    """The accumulator must NOT crash on a ``CostEstimate(None, ...)``
    return — that's the "billable but unknown" signal, and the YOLO
    policy surfaces it to the human. The accumulator stays put.
    """
    from agent.core.session import Session

    # Use __new__ to skip the heavy __init__ (loads system prompt etc.)
    s = Session.__new__(Session)
    s.total_cost_usd = 0.0
    s.add_estimated_spend(None)
    assert s.total_cost_usd == 0.0
    s.add_estimated_spend(0.001)
    assert s.total_cost_usd == 0.001
    s.add_estimated_spend(0.002)
    assert s.total_cost_usd == round(0.003, 6)
