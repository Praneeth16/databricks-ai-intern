"""Unit tests for the deterministic research-loop runner.

Fake async submit/score + a real JSONL ``ExperimentLedger`` (no workspace).
Covers the control-flow + stop-logic edge cases codex flagged.
"""

from __future__ import annotations

import pytest

from agent.core.experiment_ledger import ExperimentLedger
from agent.core.research_loop import LoopContext, run_research_loop


@pytest.fixture
def ledger(tmp_path):
    return ExperimentLedger(local_path=tmp_path / "exp.jsonl")


def _hyp(v, method=None, expected=None):
    return {
        "hypothesis": f"try {v}",
        "method": method or f"m{v}",
        "config": {"v": v},
        "expected_metric": expected,
    }


def _make_submit_score(metric_of):
    """Fakes: submit returns the config; score returns metric_of(config)."""

    async def submit_fn(config):
        return {"config": config}

    async def score_fn(handle):
        v = handle["config"]["v"]
        return {"actual_metric": metric_of(v), "cost_usd": 1.0, "wall_clock_s": 1.0}

    return submit_fn, score_fn


def _static_source(batches):
    """Yield one batch per round, then [] forever."""
    state = {"i": 0}

    def source(ctx: LoopContext):
        i = state["i"]
        state["i"] += 1
        return batches[i] if i < len(batches) else []

    return source


# ── happy path: improves, accepts, ranks ──────────────────────────────────


@pytest.mark.asyncio
async def test_improves_and_accepts_best(ledger):
    submit_fn, score_fn = _make_submit_score(lambda v: v)  # metric == v
    source = _static_source([[_hyp(0.90)], [_hyp(0.95)], [_hyp(0.92)]])
    res = await run_research_loop(
        task_id="t", hypothesis_source=source, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", max_rounds=5, patience=5,
    )
    assert res.best_metric == 0.95
    assert res.accepted_count == 2  # 0.90 then 0.95 accepted; 0.92 not
    assert res.stop_reason == "exhausted"


# ── stop conditions ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stops_on_exhausted_source(ledger):
    submit_fn, score_fn = _make_submit_score(lambda v: v)
    source = _static_source([[_hyp(0.9)]])  # one round then []
    res = await run_research_loop(
        task_id="t", hypothesis_source=source, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", max_rounds=10, patience=10,
    )
    assert res.stop_reason == "exhausted"


@pytest.mark.asyncio
async def test_stops_on_target(ledger):
    submit_fn, score_fn = _make_submit_score(lambda v: v)
    source = _static_source([[_hyp(0.99)], [_hyp(0.999)]])
    res = await run_research_loop(
        task_id="t", hypothesis_source=source, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", target_metric=0.95, max_rounds=10, patience=10,
    )
    assert res.stop_reason == "target"
    assert res.best_metric == 0.99  # stopped after first round met target


@pytest.mark.asyncio
async def test_stops_on_max_rounds(ledger):
    submit_fn, score_fn = _make_submit_score(lambda v: v)
    # endless distinct improving hypotheses
    src_state = {"v": 0.90}

    def source(ctx):
        src_state["v"] += 0.001
        return [_hyp(round(src_state["v"], 5))]

    res = await run_research_loop(
        task_id="t", hypothesis_source=source, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", max_rounds=3, patience=10,
    )
    assert res.stop_reason == "max_rounds"
    assert len(res.rounds) == 3


@pytest.mark.asyncio
async def test_stops_on_patience(ledger):
    submit_fn, score_fn = _make_submit_score(lambda v: v)
    # first round sets 0.95, subsequent rounds worse → no improvement
    source = _static_source([[_hyp(0.95)], [_hyp(0.80)], [_hyp(0.81)], [_hyp(0.82)]])
    res = await run_research_loop(
        task_id="t", hypothesis_source=source, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", max_rounds=10, patience=2,
    )
    assert res.stop_reason == "patience"
    assert res.best_metric == 0.95


# ── budget (pre-submission clamp via est_cost_per_variant) ────────────────


@pytest.mark.asyncio
async def test_budget_clamps_and_stops(ledger):
    submit_fn, score_fn = _make_submit_score(lambda v: v)
    # pool of 10 but budget only affords ~2 variants
    source = _static_source([[_hyp(0.90 + i * 0.001) for i in range(10)]])
    res = await run_research_loop(
        task_id="t", hypothesis_source=source, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", budget_usd=2.0, est_cost_per_variant=1.0,
        max_rounds=10, patience=10,
    )
    # round 1 ran 2 variants (cost 2.0) → budget hit
    assert res.stop_reason == "budget"
    assert res.rounds[0].n_hypotheses == 2


# ── lower-is-better, current_best=None (the precedence bug) ────────────────


@pytest.mark.asyncio
async def test_lower_is_better_first_round_no_crash(ledger):
    submit_fn, score_fn = _make_submit_score(lambda v: v)  # eval_loss == v
    source = _static_source([[_hyp(0.30)], [_hyp(0.25)]])
    res = await run_research_loop(
        task_id="t", hypothesis_source=source, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="eval_loss", higher_is_better=False,
        max_rounds=5, patience=5,
    )
    assert res.best_metric == 0.25  # lower is better → 0.25 accepted over 0.30
    assert res.accepted_count == 2


# ── seen-set prevents reburning the same configs ──────────────────────────


@pytest.mark.asyncio
async def test_repeated_configs_exhaust(ledger):
    submit_fn, score_fn = _make_submit_score(lambda v: v)
    same = _hyp(0.9)
    source = _static_source([[same], [same], [same]])  # source keeps repeating
    res = await run_research_loop(
        task_id="t", hypothesis_source=source, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", max_rounds=10, patience=10,
    )
    # round 1 runs it; round 2 sees the same config already tried → exhausted
    assert res.stop_reason == "exhausted"
    assert len(res.rounds) == 1


# ── reproduce-gate blocks acceptance on a major shortfall ──────────────────


@pytest.mark.asyncio
async def test_major_repro_gap_blocks_acceptance(ledger):
    # expected 0.95 but the job scores 0.70 → major gap → not accepted as best
    submit_fn, score_fn = _make_submit_score(lambda v: 0.70)
    source = _static_source([[_hyp(0.70, expected=0.95)]])
    res = await run_research_loop(
        task_id="t", hypothesis_source=source, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", max_rounds=3, patience=3,
    )
    assert res.accepted_count == 0
    assert res.best_experiment_id is None
    assert res.rounds[0].gate_severity == "major"


# ── pluggable verify_fn can veto acceptance ───────────────────────────────


@pytest.mark.asyncio
async def test_verify_fn_block_vetoes(ledger):
    submit_fn, score_fn = _make_submit_score(lambda v: v)
    source = _static_source([[_hyp(0.95)]])

    class _F:
        severity = "block"

    res = await run_research_loop(
        task_id="t", hypothesis_source=source, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", verify_fn=lambda o, l: [_F()],
        max_rounds=3, patience=3,
    )
    assert res.accepted_count == 0


@pytest.mark.asyncio
async def test_max_rounds_zero_is_noop(ledger):
    submit_fn, score_fn = _make_submit_score(lambda v: v)
    source = _static_source([[_hyp(0.9)]])
    res = await run_research_loop(
        task_id="t", hypothesis_source=source, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", max_rounds=0,
    )
    assert res.rounds == []
    assert res.stop_reason == "max_rounds"
