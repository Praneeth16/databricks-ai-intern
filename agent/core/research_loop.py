"""Deterministic research-loop runner — the LLM out of the control flow.

The auto-researcher's primitives (ledger, hypothesis ranking, parallel sweep,
reproduce-gate) were previously chained by the *agent* following system-prompt
guidance. That puts the model in charge of control flow (ordering, batching,
acceptance, stopping) — non-deterministic and token-hungry. This runner chains
the same tested primitives with explicit, deterministic control flow. The model
only supplies *content* through ``hypothesis_source`` (a static pool, or an
LLM/research-backed generator); it never drives the loop.

Per round: ask the source for candidate hypotheses → drop ones already tried in
this loop → (optionally) clamp the batch to what the remaining budget affords →
``run_sweep_async`` (parallel fan-out + dedup + per-call budget) → take the best
scored outcome → reproduce-gate it (and an optional ``verify_fn``) → accept iff
it improves the running best and isn't blocked → check stop conditions.

Stop precedence (checked pre-round and post-round): budget > target > max_rounds
> patience > source exhaustion.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent.core.repro_gate import gate_from_row
from agent.core.sweep import run_sweep_async

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoundSummary:
    round_idx: int
    n_hypotheses: int          # fresh (not-yet-seen) hypotheses swept this round
    best_metric: float | None  # best scored metric this round
    best_experiment_id: str | None
    accepted: bool             # did this round improve the running best?
    gate_severity: str         # reproduce-gate on the round's best: ok|minor|major|unknown
    cost_usd: float            # incremental spend this round (done outcomes only)


@dataclass(frozen=True)
class LoopResult:
    rounds: list[RoundSummary]
    best_experiment_id: str | None
    best_metric: float | None
    total_cost_usd: float
    accepted_count: int
    stop_reason: str           # exhausted|budget|target|patience|max_rounds


@dataclass(frozen=True)
class LoopContext:
    round_idx: int
    current_best: float | None
    best_experiment_id: str | None
    history: tuple[RoundSummary, ...]
    remaining_budget_usd: float | None
    ledger: Any                # so a source can reason over full history if it wants


# Supplies one round's candidate hypotheses (sweep/ledger-shaped dicts). Sync or
# async. Return [] to signal "no more ideas".
HypothesisSource = Callable[[LoopContext], "list[dict] | Awaitable[list[dict]]"]
# Optional richer verifier on the round's best outcome; block findings veto acceptance.
VerifyFn = Callable[[Any, Any], list]


def _config_key(h: dict) -> tuple:
    return (h.get("method"), json.dumps(h.get("config") or {}, sort_keys=True, default=str))


async def run_research_loop(
    *,
    task_id: str,
    hypothesis_source: HypothesisSource,
    submit_fn,
    score_fn,
    ledger: Any,
    metric_name: str,
    higher_is_better: bool = True,
    current_best: float | None = None,
    budget_usd: float | None = None,
    max_rounds: int = 5,
    patience: int = 2,
    min_delta: float = 0.0,
    target_metric: float | None = None,
    est_cost_per_variant: float | None = None,
    verify_fn: VerifyFn | None = None,
    session_id: str | None = None,
) -> LoopResult:
    """Drive the research loop deterministically. See module docstring."""
    rounds: list[RoundSummary] = []
    best_metric = current_best
    best_experiment_id: str | None = None
    total_cost = 0.0
    accepted_count = 0
    patience_ctr = 0
    seen: set[tuple] = set()

    def _target_met(m: float | None) -> bool:
        if target_metric is None or m is None:
            return False
        return m >= target_metric if higher_is_better else m <= target_metric

    def _improved(m: float | None) -> bool:
        if m is None:
            return False
        if best_metric is None:
            return True
        return (m - best_metric > min_delta) if higher_is_better else (best_metric - m > min_delta)

    if max_rounds <= 0:
        return LoopResult([], None, best_metric, 0.0, 0, "max_rounds")

    stop_reason: str | None = None

    for round_idx in range(max_rounds):
        # ── pre-round hard stops ─────────────────────────────────────────
        if budget_usd is not None and total_cost >= budget_usd:
            stop_reason = "budget"
            break
        if _target_met(best_metric):
            stop_reason = "target"
            break

        remaining = None if budget_usd is None else max(0.0, budget_usd - total_cost)
        ctx = LoopContext(
            round_idx=round_idx,
            current_best=best_metric,
            best_experiment_id=best_experiment_id,
            history=tuple(rounds),
            remaining_budget_usd=remaining,
            ledger=ledger,
        )

        raw = hypothesis_source(ctx)
        if inspect.isawaitable(raw):
            raw = await raw
        # Drop hypotheses already tried in this loop (covers failed/rejected
        # configs that the ledger's dedup would otherwise let rerun forever).
        fresh = [h for h in (raw or []) if _config_key(h) not in seen]
        if not fresh:
            stop_reason = "exhausted"
            break

        # Pre-submission budget clamp: only meaningful when a per-variant cost
        # estimate is supplied (real Jobs cost is unknown at submit time).
        if est_cost_per_variant and remaining is not None:
            affordable = int(remaining // est_cost_per_variant)
            if affordable <= 0:
                stop_reason = "budget"
                break
            fresh = fresh[:affordable]

        for h in fresh:
            seen.add(_config_key(h))

        result = await run_sweep_async(
            task_id=task_id,
            hypotheses=fresh,
            submit_fn=submit_fn,
            score_fn=score_fn,
            ledger=ledger,
            metric_name=metric_name,
            higher_is_better=higher_is_better,
            budget_usd=remaining,
            session_id=session_id,
        )

        # Count only THIS round's real spend — skipped dedup rows copy historical
        # cost and must not be re-added to the cumulative total.
        round_cost = sum(
            o.cost_usd for o in result.outcomes
            if o.status == "done" and o.cost_usd is not None
        )
        total_cost += round_cost

        best_round = result.best
        gate_severity = "unknown"
        accepted_this = False
        if best_round is not None:
            row = ledger.get(best_round.experiment_id)
            blocked = False
            if row is not None:
                gate = gate_from_row(row, higher_is_better=higher_is_better)
                gate_severity = gate.severity
                blocked = gate.blocked
            if verify_fn is not None:
                try:
                    findings = verify_fn(best_round, ledger) or []
                    if any(getattr(f, "severity", "") == "block" for f in findings):
                        blocked = True
                except Exception:  # noqa: BLE001 — verifier must not break the loop
                    logger.exception("verify_fn failed; ignoring")
            if _improved(best_round.actual_metric) and not blocked:
                best_metric = best_round.actual_metric
                best_experiment_id = best_round.experiment_id
                accepted_count += 1
                accepted_this = True
                patience_ctr = 0
            else:
                patience_ctr += 1
        else:
            patience_ctr += 1

        rounds.append(
            RoundSummary(
                round_idx=round_idx,
                n_hypotheses=len(fresh),
                best_metric=best_round.actual_metric if best_round else None,
                best_experiment_id=best_round.experiment_id if best_round else None,
                accepted=accepted_this,
                gate_severity=gate_severity,
                cost_usd=round_cost,
            )
        )

        # ── post-round stops: budget > target > max_rounds > patience ────
        if budget_usd is not None and total_cost >= budget_usd:
            stop_reason = "budget"
            break
        if _target_met(best_metric):
            stop_reason = "target"
            break
        if round_idx + 1 >= max_rounds:
            stop_reason = "max_rounds"
            break
        if patience_ctr >= patience:
            stop_reason = "patience"
            break

    return LoopResult(
        rounds=rounds,
        best_experiment_id=best_experiment_id,
        best_metric=best_metric,
        total_cost_usd=total_cost,
        accepted_count=accepted_count,
        stop_reason=stop_reason or "max_rounds",
    )
