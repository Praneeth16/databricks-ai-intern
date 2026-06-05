"""Reproduction-gap gate — block premature complexity escalation.

When the agent implements a technique from a paper (which carries an expected
metric) and the run underperforms by a wide margin, its instinct is to ADD
complexity: more features, more models, more seeds, pseudo-labels. That is the
exact failure logged in ``examples/kaggle-f1-pitstops-s6e5/FINAL_RESULTS.md`` —
every "add more" move regressed the leaderboard.

A large shortfall vs an expectation is almost always a reproduction failure
(wrong data format, wrong LR schedule, a missing trick, or leakage), not a
signal to escalate. This gate compares actual vs expected and, when the gap is
large, blocks escalation and emits a corrective directive telling the agent to
reproduce first — mirroring the corrective-prompt injection in ``doom_loop``.

Pure functions, no I/O. Downstream wiring (Phase 2) calls these after
``ExperimentLedger.record_result``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # typing only — avoid a hard import cycle with the ledger
    from agent.core.experiment_ledger import ExperimentRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateDecision:
    blocked: bool          # True => escalation blocked; reproduce before adding complexity
    severity: str          # "ok" | "minor" | "major" | "unknown"
    gap: float | None      # signed shortfall vs expectation (positive = underperformed)
    directive: str | None  # corrective prompt to inject when blocked or advisory


def evaluate_gap(
    *,
    expected_metric: float | None,
    actual_metric: float | None,
    higher_is_better: bool = True,
    minor_threshold: float = 0.005,
    major_threshold: float = 0.02,
) -> GateDecision:
    """Compare an outcome against its expectation and decide whether to gate.

    ``shortfall`` is the signed amount by which the run fell short of
    expectation (positive = underperformed), oriented by ``higher_is_better``
    so the same thresholds apply to AUC (higher better) and eval_loss (lower
    better).

    - No expectation (either side None) -> "unknown", not blocked.
    - shortfall <= minor_threshold (incl. met/beat) -> "ok", not blocked.
    - minor < shortfall <= major -> "minor", NOT blocked, advisory directive.
    - shortfall > major -> "major", BLOCKED, reproduction-first directive.
    """
    if expected_metric is None or actual_metric is None:
        return GateDecision(blocked=False, severity="unknown", gap=None, directive=None)

    shortfall = (
        expected_metric - actual_metric
        if higher_is_better
        else actual_metric - expected_metric
    )

    if shortfall <= minor_threshold:
        return GateDecision(blocked=False, severity="ok", gap=shortfall, directive=None)

    if shortfall <= major_threshold:
        directive = (
            f"[SYSTEM: REPRODUCTION GAP] This run is {shortfall:.4f} short of the "
            f"expected metric — close, but not yet reproduced. Sanity-check the run "
            f"before iterating: confirm the metric is computed the same way as the "
            f"reference and that the config matches what you intended. Do not escalate "
            f"complexity to close a gap this small."
        )
        logger.info("Reproduction gap minor: shortfall=%.4f", shortfall)
        return GateDecision(
            blocked=False, severity="minor", gap=shortfall, directive=directive
        )

    directive = (
        f"[SYSTEM: REPRODUCTION GAP] This run underperformed the expected metric by "
        f"{shortfall:.4f} — a large shortfall that almost always means the technique "
        f"was not reproduced, NOT that it needs more complexity. DO NOT add features, "
        f"models, seeds, or ensembling yet. First reproduce: re-read the source paper/"
        f"section for the exact recipe, verify the dataset format and preprocessing "
        f"match the reference, and check the learning rate / schedule / epochs. Then "
        f"look for a missing trick and run a leakage / target-confusion check before "
        f"trusting this number or iterating."
    )
    logger.warning("Reproduction gap major: shortfall=%.4f (escalation blocked)", shortfall)
    return GateDecision(
        blocked=True, severity="major", gap=shortfall, directive=directive
    )


def gate_from_row(row: ExperimentRow, **kwargs) -> GateDecision:
    """Evaluate the gap from a ledger row's ``expected_metric``/``actual_metric``.

    Attributes are read duck-typed so this never imports the ledger at runtime.
    ``higher_is_better`` and the thresholds pass through to ``evaluate_gap``.
    """
    return evaluate_gap(
        expected_metric=row.expected_metric,
        actual_metric=row.actual_metric,
        **kwargs,
    )
