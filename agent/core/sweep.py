"""Parallel hypothesis sweep orchestrator for the auto-researcher loop.

The Kaggle demo (``examples/kaggle-f1-pitstops-s6e5``) showed the agent
iterating serially like a human — v1, v2, … v13 — one hypothesis at a time.
That is the slow path. The agent's real edge over a human is **breadth**: fan
out N hypotheses at once on Databricks Jobs, collect the results, rank them,
keep the best. This module orchestrates exactly that.

It is deliberately decoupled from the Databricks Jobs API. The two side
effects — submitting an experiment and scoring its result — are **injected
callables** (``submit_fn`` / ``score_fn``). The architect wires the real
Databricks Jobs implementation in later (``databricks_jobs_tool.py``); this
module never imports Databricks. That keeps it fully unit-testable and reusable
across tabular / finetune / NLP tasks.

Persistence flows through the frozen :class:`ExperimentLedger` interface
(plan.md Phase 0): ``find_similar_config`` for dedup, ``propose`` to record the
hypothesis, ``mark_running`` / ``record_result`` / ``reject`` for lifecycle.

Two-phase structure (models concurrency without threads):

1. **Submit phase** — for each hypothesis: dedup → ``propose`` → ``submit_fn``.
   Dedup hits are skipped. Submit failures are recorded ``failed`` and the row
   rejected. Surviving rows carry an opaque run handle into phase 2.
2. **Score phase** — for each submitted row: ``mark_running`` → ``score_fn`` →
   ``record_result``. Score failures are recorded ``failed``.

**Budget-gate interpretation (chosen for simplicity + correctness):** real cost
is only known *after* a result is scored, so the budget is enforced in the
score phase. All deduped/proposed rows are submitted in phase 1; in phase 2 we
accumulate cost as each result comes in, and *before scoring* the next row we
check whether accumulated cost already meets or exceeds ``budget_usd`` — if so
we stop scoring and ``reject`` every remaining unscored row with reason
``"budget exhausted"``. This means the budget caps how much we *charge against*
the ledger / how many results we accept, not how many jobs were physically
submitted (submission is assumed cheap relative to the scored compute, and the
ledger total is what downstream budget logic reads).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Injected side effects.
SubmitFn = Callable[[dict], Any]
ScoreFn = Callable[[Any], dict]

_BUDGET_REASON = "budget exhausted"


@dataclass(frozen=True)
class SweepOutcome:
    """The result of a single hypothesis in the sweep."""

    experiment_id: str | None
    method: str
    config: dict
    actual_metric: float | None
    cost_usd: float | None
    status: str  # "done" | "failed" | "skipped" | "rejected"
    error: str | None = None


@dataclass(frozen=True)
class SweepResult:
    """The full sweep: outcomes ranked best-first, plus roll-up counters."""

    outcomes: list[SweepOutcome]
    best: SweepOutcome | None
    total_cost_usd: float
    n_submitted: int
    n_skipped: int
    n_failed: int


def _rank_key(higher_is_better: bool):
    """Sort key putting the best metric first; ``None`` metrics always last."""

    def key(outcome: SweepOutcome) -> tuple[int, float]:
        if outcome.actual_metric is None:
            # Second element is irrelevant once the first sorts it to the end.
            return (1, 0.0)
        metric = outcome.actual_metric
        # Stable ascending sort: negate for higher-is-better so best leads.
        return (0, -metric if higher_is_better else metric)

    return key


def run_sweep(
    *,
    task_id: str,
    hypotheses: list[dict],
    submit_fn: SubmitFn,
    score_fn: ScoreFn,
    ledger: Any,
    metric_name: str,
    higher_is_better: bool = True,
    budget_usd: float | None = None,
    session_id: str | None = None,
) -> SweepResult:
    """Fan out ``hypotheses``, collect results to ``ledger``, rank best-first.

    Each hypothesis is a plain dict::

        {
            "hypothesis": str,
            "method": str,
            "config": dict,
            "expected_metric": float | None,
            "source_paper": str | None,
            "source_section": str | None,
        }

    ``submit_fn(config) -> handle`` submits one experiment and returns an opaque
    run handle; it may raise. ``score_fn(handle) -> dict`` returns
    ``{"actual_metric", "cost_usd", "wall_clock_s", "artifacts",
    "mlflow_run_id"}``; it may raise.

    Returns a :class:`SweepResult` whose ``outcomes`` are ranked best-first by
    ``actual_metric`` (honoring ``higher_is_better``; ``None`` metrics sort
    last). See the module docstring for the budget-gate interpretation.
    """
    # ── Phase 1: dedup + propose + submit ────────────────────────────────
    submitted: list[tuple[str, Any, dict]] = []  # (experiment_id, handle, hyp)
    outcomes: list[SweepOutcome] = []

    for hyp in hypotheses:
        method = hyp["method"]
        config = hyp["config"]

        existing = ledger.find_similar_config(task_id, method, config)
        if existing is not None:
            outcomes.append(
                SweepOutcome(
                    experiment_id=existing.experiment_id,
                    method=method,
                    config=config,
                    actual_metric=existing.actual_metric,
                    cost_usd=existing.cost_usd,
                    status="skipped",
                )
            )
            continue

        experiment_id = ledger.propose(
            task_id=task_id,
            hypothesis=hyp["hypothesis"],
            method=method,
            config=config,
            metric_name=metric_name,
            expected_metric=hyp.get("expected_metric"),
            source_paper=hyp.get("source_paper"),
            source_section=hyp.get("source_section"),
            session_id=session_id,
        )

        try:
            handle = submit_fn(config)
        except Exception as exc:  # noqa: BLE001 — injected callable, any error
            logger.warning("submit_fn failed for %s: %s", experiment_id, exc)
            ledger.reject(experiment_id, f"submit failed: {exc}")
            outcomes.append(
                SweepOutcome(
                    experiment_id=experiment_id,
                    method=method,
                    config=config,
                    actual_metric=None,
                    cost_usd=None,
                    status="failed",
                    error=str(exc),
                )
            )
            continue

        submitted.append((experiment_id, handle, hyp))

    # ── Phase 2: score + record (budget enforced here) ───────────────────
    accumulated_cost = 0.0
    budget_hit = False

    for experiment_id, handle, hyp in submitted:
        method = hyp["method"]
        config = hyp["config"]

        if budget_hit or (
            budget_usd is not None and accumulated_cost >= budget_usd
        ):
            budget_hit = True
            ledger.reject(experiment_id, _BUDGET_REASON)
            outcomes.append(
                SweepOutcome(
                    experiment_id=experiment_id,
                    method=method,
                    config=config,
                    actual_metric=None,
                    cost_usd=None,
                    status="rejected",
                    error=_BUDGET_REASON,
                )
            )
            continue

        ledger.mark_running(experiment_id)

        try:
            scored = score_fn(handle)
        except Exception as exc:  # noqa: BLE001 — injected callable, any error
            logger.warning("score_fn failed for %s: %s", experiment_id, exc)
            ledger.reject(experiment_id, f"score failed: {exc}")
            outcomes.append(
                SweepOutcome(
                    experiment_id=experiment_id,
                    method=method,
                    config=config,
                    actual_metric=None,
                    cost_usd=None,
                    status="failed",
                    error=str(exc),
                )
            )
            continue

        actual_metric = scored["actual_metric"]
        cost_usd = scored.get("cost_usd")

        # mlflow_run_id is part of mark_running's signature, but only known
        # post-score — re-stamp it now that we have it.
        mlflow_run_id = scored.get("mlflow_run_id")
        if mlflow_run_id is not None:
            ledger.mark_running(experiment_id, mlflow_run_id=mlflow_run_id)

        ledger.record_result(
            experiment_id,
            actual_metric=actual_metric,
            cost_usd=cost_usd,
            wall_clock_s=scored.get("wall_clock_s"),
            artifacts=scored.get("artifacts"),
            status="done",
        )

        if cost_usd is not None:
            accumulated_cost += cost_usd

        outcomes.append(
            SweepOutcome(
                experiment_id=experiment_id,
                method=method,
                config=config,
                actual_metric=actual_metric,
                cost_usd=cost_usd,
                status="done",
            )
        )

    # ── Rank + roll up ───────────────────────────────────────────────────
    ranked = sorted(outcomes, key=_rank_key(higher_is_better))
    done = [o for o in ranked if o.status == "done"]
    best = done[0] if done else None
    total_cost = sum(o.cost_usd for o in outcomes if o.cost_usd is not None)

    return SweepResult(
        outcomes=ranked,
        best=best,
        total_cost_usd=total_cost,
        n_submitted=len(submitted),
        n_skipped=sum(1 for o in outcomes if o.status == "skipped"),
        n_failed=sum(1 for o in outcomes if o.status == "failed"),
    )


def _rollup(outcomes: list[SweepOutcome], higher_is_better: bool, n_submitted: int) -> SweepResult:
    ranked = sorted(outcomes, key=_rank_key(higher_is_better))
    done = [o for o in ranked if o.status == "done"]
    return SweepResult(
        outcomes=ranked,
        best=done[0] if done else None,
        total_cost_usd=sum(o.cost_usd for o in outcomes if o.cost_usd is not None),
        n_submitted=n_submitted,
        n_skipped=sum(1 for o in outcomes if o.status == "skipped"),
        n_failed=sum(1 for o in outcomes if o.status == "failed"),
    )


async def run_sweep_async(
    *,
    task_id: str,
    hypotheses: list[dict],
    submit_fn: Callable[[dict], Any],
    score_fn: Callable[[Any], Any],
    ledger: Any,
    metric_name: str,
    higher_is_better: bool = True,
    budget_usd: float | None = None,
    session_id: str | None = None,
) -> SweepResult:
    """Async sweep driver — same contract as :func:`run_sweep`, but ``submit_fn``
    and ``score_fn`` are coroutines and submissions fan out concurrently.

    This is the real path: ``submit_fn`` kicks off a Databricks Job and returns
    a run handle without blocking, so phase 1 submits all surviving hypotheses
    in parallel (``asyncio.gather``). Phase 2 scores sequentially so the budget
    gate stays correct (cost is only known after a result lands). Dedup, ledger
    lifecycle, ranking, and the budget interpretation match :func:`run_sweep`.
    """
    outcomes: list[SweepOutcome] = []
    to_submit: list[tuple[str, dict]] = []  # (experiment_id, hyp)

    for hyp in hypotheses:
        method, config = hyp["method"], hyp["config"]
        existing = ledger.find_similar_config(task_id, method, config)
        if existing is not None:
            outcomes.append(
                SweepOutcome(
                    experiment_id=existing.experiment_id, method=method, config=config,
                    actual_metric=existing.actual_metric, cost_usd=existing.cost_usd,
                    status="skipped",
                )
            )
            continue
        experiment_id = ledger.propose(
            task_id=task_id, hypothesis=hyp["hypothesis"], method=method, config=config,
            metric_name=metric_name, expected_metric=hyp.get("expected_metric"),
            source_paper=hyp.get("source_paper"), source_section=hyp.get("source_section"),
            session_id=session_id,
        )
        to_submit.append((experiment_id, hyp))

    # ── Phase 1: fan out submissions concurrently ────────────────────────
    results = await asyncio.gather(
        *(submit_fn(hyp["config"]) for _, hyp in to_submit),
        return_exceptions=True,
    )
    submitted: list[tuple[str, Any, dict]] = []
    for (experiment_id, hyp), res in zip(to_submit, results):
        method, config = hyp["method"], hyp["config"]
        if isinstance(res, Exception):
            logger.warning("submit_fn failed for %s: %s", experiment_id, res)
            ledger.reject(experiment_id, f"submit failed: {res}")
            outcomes.append(
                SweepOutcome(experiment_id, method, config, None, None, "failed", str(res))
            )
        else:
            submitted.append((experiment_id, res, hyp))

    # ── Phase 2: score sequentially (budget enforced here) ───────────────
    accumulated_cost = 0.0
    budget_hit = False
    for experiment_id, handle, hyp in submitted:
        method, config = hyp["method"], hyp["config"]
        if budget_hit or (budget_usd is not None and accumulated_cost >= budget_usd):
            budget_hit = True
            ledger.reject(experiment_id, _BUDGET_REASON)
            outcomes.append(
                SweepOutcome(experiment_id, method, config, None, None, "rejected", _BUDGET_REASON)
            )
            continue

        ledger.mark_running(experiment_id)
        try:
            scored = await score_fn(handle)
        except Exception as exc:  # noqa: BLE001 — injected callable, any error
            logger.warning("score_fn failed for %s: %s", experiment_id, exc)
            ledger.reject(experiment_id, f"score failed: {exc}")
            outcomes.append(
                SweepOutcome(experiment_id, method, config, None, None, "failed", str(exc))
            )
            continue

        actual_metric = scored["actual_metric"]
        cost_usd = scored.get("cost_usd")
        mlflow_run_id = scored.get("mlflow_run_id")
        if mlflow_run_id is not None:
            ledger.mark_running(experiment_id, mlflow_run_id=mlflow_run_id)
        ledger.record_result(
            experiment_id, actual_metric=actual_metric, cost_usd=cost_usd,
            wall_clock_s=scored.get("wall_clock_s"), artifacts=scored.get("artifacts"),
            status="done",
        )
        if cost_usd is not None:
            accumulated_cost += cost_usd
        outcomes.append(
            SweepOutcome(experiment_id, method, config, actual_metric, cost_usd, "done")
        )

    return _rollup(outcomes, higher_is_better, len(submitted))
