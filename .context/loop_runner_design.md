# Design — deterministic research-loop runner (Phase 6)

## Problem
The auto-researcher loop is guidance-driven: the system prompt orchestrates the
agent through research → hypotheses → sweep → record → critic → reproduce-gate.
That puts the LLM in the driver's seat for *control flow* (ordering, batching,
stopping, acceptance), which is non-deterministic and burns tokens. The
primitives already exist and are tested (ledger, hypothesis, sweep,
run_sweep_async, critic, repro_gate). We want a deterministic runner that chains
them with explicit stopping logic — the LLM only supplies *content* (the
hypothesis pool / findings), never the control flow.

## Existing primitives (on main, tested)
- `experiment_ledger.ExperimentLedger`: propose / mark_running / record_result /
  best_for_task / find_similar_config / get. SQL + JSONL backends.
- `hypothesis.generate_hypotheses(findings, *, task_metric, current_best, higher_is_better, ledger, task_id) -> list[Hypothesis]` and `rank_hypotheses`.
- `sweep.run_sweep_async(*, task_id, hypotheses, submit_fn, score_fn, ledger, metric_name, higher_is_better, budget_usd, session_id) -> SweepResult` (parallel fan-out, dedup, budget gate). SweepResult.best / .outcomes / .total_cost_usd.
- `sweep_jobs.build_jobs_callables(jobs_tool, *, render_script, base_args, kind) -> (submit_fn, score_fn)`.
- `repro_gate.gate_from_row(row, higher_is_better) -> GateDecision(blocked, severity, gap, directive)`.
- `critic.audit(signals) -> list[Finding]` (overfit/leakage/target/correlation).

## Proposed: agent/core/research_loop.py

```python
@dataclass(frozen=True)
class RoundSummary:
    round_idx: int
    n_hypotheses: int        # proposed this round (post-dedup is sweep's job)
    best_metric: float | None
    best_experiment_id: str | None
    accepted: bool           # did this round improve the running best?
    gate_severity: str       # repro-gate on the round's best: ok|minor|major|unknown
    cost_usd: float

@dataclass(frozen=True)
class LoopResult:
    rounds: list[RoundSummary]
    best_experiment_id: str | None
    best_metric: float | None
    total_cost_usd: float
    accepted_count: int
    stop_reason: str         # "exhausted"|"budget"|"target"|"patience"|"max_rounds"

@dataclass(frozen=True)
class LoopContext:
    round_idx: int
    current_best: float | None
    best_experiment_id: str | None
    history: tuple[RoundSummary, ...]
    remaining_budget_usd: float | None

# hypothesis_source supplies the round's candidate hypotheses (list of dicts in
# the sweep/ledger shape). Sync or async. Returns [] to signal "no more ideas".
HypothesisSource = Callable[[LoopContext], list[dict] | Awaitable[list[dict]]]

async def run_research_loop(
    *, task_id, hypothesis_source, submit_fn, score_fn, ledger,
    metric_name, higher_is_better=True, current_best=None,
    budget_usd=None, max_rounds=5, patience=2, min_delta=0.0,
    target_metric=None, session_id=None,
) -> LoopResult: ...
```

### Per-round control flow (deterministic)
1. Build `LoopContext` (round_idx, current_best, best_experiment_id, history, remaining_budget = budget_usd - total_cost so far).
2. `hyps = await maybe_await(hypothesis_source(ctx))`. If empty → stop `"exhausted"`.
3. `result = await run_sweep_async(..., budget_usd=remaining_budget)`. Sweep handles parallel fan-out + dedup + per-call budget.
4. `total_cost += result.total_cost_usd`.
5. `best_round = result.best` (highest scored "done" outcome).
6. If `best_round`:
   - `row = ledger.get(best_round.experiment_id)`; `gate = repro_gate.gate_from_row(row, higher_is_better=higher_is_better)`.
   - `improved = current_best is None or (best_round.actual_metric - current_best > min_delta) if higher_is_better else (current_best - best_round.actual_metric > min_delta)`.
   - `accepted = improved and not gate.blocked`.
   - if accepted: update current_best + best_experiment_id; reset patience.
   - else: patience += 1.
   else (no scored result): patience += 1; gate_severity = "unknown".
7. Append RoundSummary.
8. Stop checks (in order): target reached (current_best beats target_metric) → "target"; budget exhausted (budget_usd is not None and total_cost >= budget_usd) → "budget"; patience counter >= patience → "patience"; (round_idx + 1) >= max_rounds → "max_rounds". Else continue.
9. Return LoopResult.

### Key decisions / rationale
- **LLM out of control flow**: the runner decides ordering, batching, acceptance, stopping. `hypothesis_source` is the only content seam (inject a static pool, or an LLM/research-backed generator later). Fully testable with a fake source.
- **Built-in verification = reproduce-gate**, not full critic. The loop only reliably has expected vs actual (on the ledger row). Full critic.audit needs richer signals (cv-vs-lb, feature importances) the generic sweep outcome doesn't carry — so critic stays the agent's explicit pre-ship check, while the loop's automatic gate is the reproduce-gap (block escalation on major shortfall). This avoids fabricating signals.
- **Budget**: loop passes `remaining_budget` to each sweep; sweep enforces it per-call; loop stops when cumulative cost >= budget_usd. (Note: sweep can overshoot by ~one variant per its documented gate; the loop's cumulative check inherits that.)
- **Dedup**: handled inside run_sweep_async via find_similar_config — no double-run across rounds.

## Proposed: agent/tools/research_loop_tool.py
`research_loop` tool — agent-facing autonomous closure. Args: task_id, metric_name, script_template, hypotheses (the POOL), higher_is_better, current_best, budget_usd, max_rounds, batch_size, min_delta, patience, target_metric, kind, timeout, dependencies. Builds submit/score via sweep_jobs (like sweep_tool, reuse `_get_ledger`/`_get_jobs_tool`). hypothesis_source = pool-batcher: each round yields the next `batch_size` not-yet-tried (ranked) hypotheses from the pool; [] when pool exhausted. Approval-gated (runs jobs). Returns LoopResult summary (rounds table, best, stop_reason, total cost).

## Open questions for review
1. Is reproduce-gate the right automatic verifier, or should the loop accept a pluggable `verify_fn(best_outcome, ledger) -> findings` so the agent can wire critic with real signals?
2. Budget overshoot across rounds — acceptable, or hard-cap by pre-estimating next round?
3. Stopping precedence order (target > budget > patience > max_rounds) — correct?
4. Should `hypothesis_source` also receive the full ledger (to re-rank against all history), or just LoopContext?
5. Batch sizing in the tool: fixed batch_size vs adaptive (shrink as budget runs low)?
```
