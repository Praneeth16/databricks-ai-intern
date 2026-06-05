"""YOLO auto-approval policy tests (issue #17)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.core.approval_policy import (
    is_scheduled_operation,
    normalize_tool_operation,
    should_auto_approve,
)
from agent.core.cost_estimation import CostEstimate
from agent.core.session import Session


# ── helpers ────────────────────────────────────────────────────────────


def _fake_session(*, enabled: bool, cap: float | None) -> Session:
    """Build a Session bypassing the heavy __init__ — we only need the
    yolo fields + the accumulator method.
    """
    s = Session.__new__(Session)
    s.auto_approval_enabled = enabled
    s.auto_approval_cost_cap_usd = cap
    s.auto_approval_estimated_spend_usd = 0.0
    s.total_cost_usd = 0.0
    s.sandbox = None
    return s


def _est(usd: float | None, *, billable: bool = True) -> CostEstimate:
    return CostEstimate(estimated_cost_usd=usd, billable=billable)


# ── normalize / scheduled predicate ────────────────────────────────────


def test_normalize_tool_operation_strips_and_lowercases():
    assert normalize_tool_operation("  Run  ") == "run"
    assert normalize_tool_operation(None) == ""
    assert normalize_tool_operation(123) == "123"


def test_is_scheduled_operation_matches_prefix():
    assert is_scheduled_operation("scheduled run") is True
    assert is_scheduled_operation("Scheduled Inspect") is True
    assert is_scheduled_operation("run") is False


# ── policy off ─────────────────────────────────────────────────────────


def test_policy_off_never_auto_approves():
    s = _fake_session(enabled=False, cap=100.0)
    approved, _, reason = should_auto_approve(s, "uc_inspect_dataset", {})
    assert approved is False
    assert "YOLO not enabled" in (reason or "")


# ── policy on, free tool ───────────────────────────────────────────────


def test_free_tool_auto_approved_without_touching_budget():
    s = _fake_session(enabled=True, cap=1.0)
    with patch(
        "agent.core.approval_policy.estimate_tool_cost",
        return_value=_est(0.0, billable=False),
    ):
        approved, est, reason = should_auto_approve(s, "uc_inspect_dataset", {})
    assert approved is True
    assert reason is None
    # Free tools don't bump the spend — that's the caller's job and only
    # for billable estimates.
    assert s.auto_approval_estimated_spend_usd == 0.0


# ── policy on, billable under cap ──────────────────────────────────────


def test_billable_under_cap_auto_approved():
    s = _fake_session(enabled=True, cap=5.0)
    s.auto_approval_estimated_spend_usd = 2.0
    with patch(
        "agent.core.approval_policy.estimate_tool_cost",
        return_value=_est(1.5),
    ):
        approved, est, reason = should_auto_approve(
            s, "databricks_jobs",
            {"kind": "script", "node_type_id": "i3.xlarge", "timeout_seconds": 60},
        )
    assert approved is True
    assert reason is None
    assert est.estimated_cost_usd == 1.5


def test_billable_over_cap_blocked_with_reason():
    s = _fake_session(enabled=True, cap=5.0)
    s.auto_approval_estimated_spend_usd = 4.5
    with patch(
        "agent.core.approval_policy.estimate_tool_cost",
        return_value=_est(2.0),
    ):
        approved, est, reason = should_auto_approve(
            s, "databricks_jobs",
            {"kind": "script", "node_type_id": "i3.xlarge", "timeout_seconds": 600},
        )
    assert approved is False
    assert "Would exceed budget" in (reason or "")


def test_billable_uncapped_always_approves():
    s = _fake_session(enabled=True, cap=None)
    with patch(
        "agent.core.approval_policy.estimate_tool_cost",
        return_value=_est(999.99),
    ):
        approved, _, reason = should_auto_approve(
            s, "databricks_jobs",
            {"kind": "serverless_gpu", "node_type_id": "GPU_1xH100"},
        )
    assert approved is True
    assert reason is None


# ── unknown price → human approval ─────────────────────────────────────


def test_unknown_price_falls_back_to_human():
    """Catalog-miss surface (``estimated_cost_usd=None``) must NOT
    auto-approve even when YOLO is on with a wide cap. The CostEstimate
    carries the block_reason that the UI can render."""
    s = _fake_session(enabled=True, cap=1000.0)
    with patch(
        "agent.core.approval_policy.estimate_tool_cost",
        return_value=CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason="No price catalog entry for node 'monster'.",
        ),
    ):
        approved, _, reason = should_auto_approve(
            s, "databricks_jobs", {"node_type_id": "monster"},
        )
    assert approved is False
    assert "price catalog" in (reason or "")


# ── NEVER_AUTO_APPROVE list ────────────────────────────────────────────


def test_never_auto_approve_blocked_even_under_cap():
    """Destructive ops never auto-approve, regardless of policy."""
    s = _fake_session(enabled=True, cap=1000.0)
    with patch(
        "agent.core.approval_policy.estimate_tool_cost",
        return_value=_est(0.0, billable=False),
    ):
        approved, _, reason = should_auto_approve(s, "uc_volume_rm", {})
    assert approved is False
    assert "NEVER_AUTO_APPROVE" in (reason or "")


def test_scheduled_op_blocked():
    s = _fake_session(enabled=True, cap=1000.0)
    with patch(
        "agent.core.approval_policy.estimate_tool_cost",
        return_value=_est(0.5),
    ):
        approved, _, reason = should_auto_approve(
            s, "sandbox_create", {"operation": "scheduled run"},
        )
    assert approved is False
    assert "Scheduled" in (reason or "")


# ── Session state methods ─────────────────────────────────────────────


def test_set_auto_approval_policy_keeps_spend_across_reconfigure():
    """A user toggling YOLO off and back on must NOT reset accumulated
    spend — that would let them game a capped session by flicking the
    toggle. The accumulator zeroes only on a new Session, never on
    reconfigure.
    """
    s = _fake_session(enabled=True, cap=5.0)
    s.auto_approval_estimated_spend_usd = 3.5

    s.set_auto_approval_policy(enabled=False, cost_cap_usd=None)
    assert s.auto_approval_estimated_spend_usd == 3.5

    s.set_auto_approval_policy(enabled=True, cost_cap_usd=10.0)
    assert s.auto_approval_estimated_spend_usd == 3.5
    assert s.auto_approval_enabled is True
    assert s.auto_approval_cost_cap_usd == 10.0


def test_auto_approval_remaining_usd_none_when_uncapped():
    s = _fake_session(enabled=True, cap=None)
    s.auto_approval_estimated_spend_usd = 1.0
    assert s.auto_approval_remaining_usd() is None


def test_auto_approval_remaining_usd_floors_at_zero():
    """Once we've blown past the cap (catalog drift, race, etc.) the
    remaining display should be 0 not negative — that confuses the UI."""
    s = _fake_session(enabled=True, cap=5.0)
    s.auto_approval_estimated_spend_usd = 7.5
    assert s.auto_approval_remaining_usd() == 0.0


def test_add_auto_approval_estimated_spend_none_safe():
    s = _fake_session(enabled=True, cap=5.0)
    s.add_auto_approval_estimated_spend(None)
    assert s.auto_approval_estimated_spend_usd == 0.0
    s.add_auto_approval_estimated_spend(0.1)
    s.add_auto_approval_estimated_spend(0.2)
    assert s.auto_approval_estimated_spend_usd == 0.3
