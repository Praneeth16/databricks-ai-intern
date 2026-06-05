"""YOLO auto-approval policy (issue #17).

Single source of truth for "should this tool call bypass the human
approval prompt?" — read by the agent loop's approval gate. Keeps the
decision logic out of the call site so the test surface is small.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.core.cost_estimation import (
    NEVER_AUTO_APPROVE,
    CostEstimate,
    estimate_tool_cost,
)

logger = logging.getLogger(__name__)


def normalize_tool_operation(operation: Any) -> str:
    """Normalise a tool's ``operation`` arg for comparison."""
    return str(operation or "").strip().lower()


def is_scheduled_operation(operation: Any) -> bool:
    """``databricks_jobs`` scheduled operations are user-visible changes
    to a recurring job and always go through human approval — same as
    upstream HF#201's predicate."""
    return normalize_tool_operation(operation).startswith("scheduled ")


def should_auto_approve(
    session: Any,
    tool_name: str,
    args: dict[str, Any],
) -> tuple[bool, CostEstimate, str | None]:
    """Decide approval for one tool call.

    Returns ``(approved, estimate, block_reason)``.

    Approves only when:
      1. ``session.auto_approval_enabled`` is True.
      2. ``tool_name`` not in :data:`NEVER_AUTO_APPROVE`.
      3. The tool's operation arg is not a scheduled-job mutation.
      4. The pre-call estimate has a concrete number (not ``None``).
      5. ``estimated_spend + estimate <= cost_cap`` (uncapped → always).

    On approve, the caller is expected to bump
    ``session.add_auto_approval_estimated_spend(estimate.estimated_cost_usd)``
    AND ``session.add_estimated_spend(estimate.estimated_cost_usd)`` so
    the next call site reads a consistent budget.
    """
    estimate = estimate_tool_cost(tool_name, args, session=session)

    if not getattr(session, "auto_approval_enabled", False):
        return False, estimate, "YOLO not enabled."

    if tool_name in NEVER_AUTO_APPROVE:
        return False, estimate, (
            f"{tool_name!r} is in NEVER_AUTO_APPROVE — always requires "
            "human approval."
        )

    if is_scheduled_operation(args.get("operation")):
        return False, estimate, (
            "Scheduled-job operations always require human approval."
        )

    if not estimate.billable:
        # Free / read-only — auto-approve without touching the budget.
        return True, estimate, None

    if estimate.estimated_cost_usd is None:
        return False, estimate, (
            estimate.block_reason
            or "Cost catalog miss — falling back to human approval."
        )

    cap = getattr(session, "auto_approval_cost_cap_usd", None)
    if cap is None:
        return True, estimate, None

    projected = (
        getattr(session, "auto_approval_estimated_spend_usd", 0.0)
        + estimate.estimated_cost_usd
    )
    if projected > cap:
        return False, estimate, (
            f"Would exceed budget: ${projected:.4f} > ${cap:.4f}. "
            "Approve manually if you still want to run it."
        )

    return True, estimate, None
