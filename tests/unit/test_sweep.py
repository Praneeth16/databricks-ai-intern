"""Tests for the parallel hypothesis sweep orchestrator.

Uses fake ``submit_fn`` / ``score_fn`` callables and a real
:class:`ExperimentLedger` over a JSONL file (no live workspace), so the full
propose → submit → score → record → rank flow is exercised end-to-end.
"""

from __future__ import annotations

import pytest

from agent.core.experiment_ledger import ExperimentLedger
from agent.core.sweep import SweepOutcome, SweepResult, run_sweep

TASK = "kaggle-f1-pitstops-s6e5"
METRIC = "roc_auc"


@pytest.fixture
def ledger(tmp_path):
    return ExperimentLedger(local_path=tmp_path / "exp.jsonl")


def _hyp(method: str, config: dict, expected: float | None = None) -> dict:
    return {
        "hypothesis": f"{method} should help",
        "method": method,
        "config": config,
        "expected_metric": expected,
        "source_paper": None,
        "source_section": None,
    }


def _make_score_fn(metric_by_method: dict, cost: float = 1.0):
    """score_fn that maps a handle (the config dict) to a fixed metric."""

    def score_fn(handle):
        method = handle["__method__"]
        return {
            "actual_metric": metric_by_method[method],
            "cost_usd": cost,
            "wall_clock_s": 10.0,
            "artifacts": {"preds": f"/Volumes/x/{method}.csv"},
            "mlflow_run_id": f"run-{method}",
        }

    return score_fn


def _submit_fn(config):
    # Echo the config back as the opaque handle.
    return config


# ── happy path ───────────────────────────────────────────────────────────


def test_happy_path_ranks_best_first(ledger):
    hyps = [
        _hyp("lgbm", {"__method__": "lgbm", "lr": 0.1}),
        _hyp("xgb", {"__method__": "xgb", "lr": 0.05}),
        _hyp("catboost", {"__method__": "catboost", "depth": 6}),
    ]
    score_fn = _make_score_fn(
        {"lgbm": 0.948, "xgb": 0.951, "catboost": 0.949}, cost=2.0
    )

    result = run_sweep(
        task_id=TASK,
        hypotheses=hyps,
        submit_fn=_submit_fn,
        score_fn=score_fn,
        ledger=ledger,
        metric_name=METRIC,
    )

    assert isinstance(result, SweepResult)
    # Best-first ranking.
    assert [o.method for o in result.outcomes] == ["xgb", "catboost", "lgbm"]
    assert result.best is not None
    assert result.best.method == "xgb"
    assert result.best.actual_metric == 0.951
    # Costs summed across all three.
    assert result.total_cost_usd == pytest.approx(6.0)
    assert result.n_submitted == 3
    assert result.n_skipped == 0
    assert result.n_failed == 0
    assert all(o.status == "done" for o in result.outcomes)


def test_happy_path_records_ledger_rows_done(ledger):
    hyps = [_hyp("lgbm", {"__method__": "lgbm"})]
    result = run_sweep(
        task_id=TASK,
        hypotheses=hyps,
        submit_fn=_submit_fn,
        score_fn=_make_score_fn({"lgbm": 0.95}),
        ledger=ledger,
        metric_name=METRIC,
    )

    rows = ledger.list_for_task(TASK)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "done"
    assert row.actual_metric == 0.95
    assert row.cost_usd == 1.0
    assert result.outcomes[0].experiment_id == row.experiment_id


# ── dedup ─────────────────────────────────────────────────────────────────


def test_dedup_skips_already_tried_config(ledger):
    hyps = [_hyp("lgbm", {"__method__": "lgbm", "lr": 0.1})]
    score_fn = _make_score_fn({"lgbm": 0.95})

    first = run_sweep(
        task_id=TASK,
        hypotheses=hyps,
        submit_fn=_submit_fn,
        score_fn=score_fn,
        ledger=ledger,
        metric_name=METRIC,
    )
    assert first.outcomes[0].status == "done"

    # Re-run the identical hypothesis: should be skipped, not resubmitted.
    submitted = []

    def tracking_submit(config):
        submitted.append(config)
        return config

    second = run_sweep(
        task_id=TASK,
        hypotheses=hyps,
        submit_fn=tracking_submit,
        score_fn=score_fn,
        ledger=ledger,
        metric_name=METRIC,
    )

    assert submitted == []  # nothing resubmitted
    assert second.n_submitted == 0
    assert second.n_skipped == 1
    assert second.outcomes[0].status == "skipped"
    # Skipped outcome carries the prior result's metric.
    assert second.outcomes[0].actual_metric == 0.95
    # Ledger still has exactly one row.
    assert len(ledger.list_for_task(TASK)) == 1


# ── submit failure ─────────────────────────────────────────────────────────


def test_submit_failure_marks_failed_others_succeed(ledger):
    hyps = [
        _hyp("lgbm", {"__method__": "lgbm"}),
        _hyp("boom", {"__method__": "boom"}),
        _hyp("xgb", {"__method__": "xgb"}),
    ]

    def flaky_submit(config):
        if config["__method__"] == "boom":
            raise RuntimeError("cluster denied")
        return config

    result = run_sweep(
        task_id=TASK,
        hypotheses=hyps,
        submit_fn=flaky_submit,
        score_fn=_make_score_fn({"lgbm": 0.94, "xgb": 0.95}),
        ledger=ledger,
        metric_name=METRIC,
    )

    by_method = {o.method: o for o in result.outcomes}
    assert by_method["boom"].status == "failed"
    assert "cluster denied" in by_method["boom"].error
    assert by_method["lgbm"].status == "done"
    assert by_method["xgb"].status == "done"
    assert result.n_failed == 1
    assert result.n_submitted == 2
    assert result.best.method == "xgb"

    # The failed row was rejected in the ledger.
    rows = {r.method: r for r in ledger.list_for_task(TASK)}
    assert rows["boom"].status == "rejected"


# ── score failure ───────────────────────────────────────────────────────────


def test_score_failure_marks_failed(ledger):
    hyps = [
        _hyp("lgbm", {"__method__": "lgbm"}),
        _hyp("xgb", {"__method__": "xgb"}),
    ]

    def flaky_score(handle):
        if handle["__method__"] == "xgb":
            raise ValueError("scorer crashed")
        return {
            "actual_metric": 0.94,
            "cost_usd": 1.0,
            "wall_clock_s": None,
            "artifacts": None,
            "mlflow_run_id": None,
        }

    result = run_sweep(
        task_id=TASK,
        hypotheses=hyps,
        submit_fn=_submit_fn,
        score_fn=flaky_score,
        ledger=ledger,
        metric_name=METRIC,
    )

    by_method = {o.method: o for o in result.outcomes}
    assert by_method["xgb"].status == "failed"
    assert "scorer crashed" in by_method["xgb"].error
    assert by_method["lgbm"].status == "done"
    assert result.n_failed == 1

    rows = {r.method: r for r in ledger.list_for_task(TASK)}
    assert rows["xgb"].status == "rejected"


# ── budget gate ──────────────────────────────────────────────────────────────


def test_budget_stops_sweep_and_rejects_remainder(ledger):
    hyps = [
        _hyp("a", {"__method__": "a"}),
        _hyp("b", {"__method__": "b"}),
        _hyp("c", {"__method__": "c"}),
    ]
    # Each scored result costs 4.0; budget of 5.0 admits one, then the
    # accumulated cost (4.0) is < 5.0 so the second is scored too (8.0), then
    # the third is rejected.
    score_fn = _make_score_fn({"a": 0.9, "b": 0.91, "c": 0.92}, cost=4.0)

    result = run_sweep(
        task_id=TASK,
        hypotheses=hyps,
        submit_fn=_submit_fn,
        score_fn=score_fn,
        ledger=ledger,
        metric_name=METRIC,
        budget_usd=5.0,
    )

    by_method = {o.method: o for o in result.outcomes}
    assert by_method["a"].status == "done"
    assert by_method["b"].status == "done"
    assert by_method["c"].status == "rejected"
    assert by_method["c"].error == "budget exhausted"
    assert result.total_cost_usd == pytest.approx(8.0)

    rows = {r.method: r for r in ledger.list_for_task(TASK)}
    assert rows["c"].status == "rejected"


def test_budget_zero_rejects_everything(ledger):
    hyps = [_hyp("a", {"__method__": "a"}), _hyp("b", {"__method__": "b"})]
    result = run_sweep(
        task_id=TASK,
        hypotheses=hyps,
        submit_fn=_submit_fn,
        score_fn=_make_score_fn({"a": 0.9, "b": 0.9}, cost=1.0),
        ledger=ledger,
        metric_name=METRIC,
        budget_usd=0.0,
    )
    assert all(o.status == "rejected" for o in result.outcomes)
    assert result.best is None
    assert result.total_cost_usd == 0.0


# ── ranking direction ───────────────────────────────────────────────────────


def test_lower_is_better_ranks_ascending(ledger):
    hyps = [
        _hyp("big_loss", {"__method__": "big_loss"}),
        _hyp("small_loss", {"__method__": "small_loss"}),
        _hyp("mid_loss", {"__method__": "mid_loss"}),
    ]
    score_fn = _make_score_fn(
        {"big_loss": 0.9, "small_loss": 0.1, "mid_loss": 0.5}
    )

    result = run_sweep(
        task_id=TASK,
        hypotheses=hyps,
        submit_fn=_submit_fn,
        score_fn=score_fn,
        ledger=ledger,
        metric_name="eval_loss",
        higher_is_better=False,
    )

    assert [o.method for o in result.outcomes] == [
        "small_loss",
        "mid_loss",
        "big_loss",
    ]
    assert result.best.method == "small_loss"


def test_none_metrics_sort_last(ledger):
    hyps = [
        _hyp("ok", {"__method__": "ok"}),
        _hyp("crash", {"__method__": "crash"}),
    ]

    def score_fn(handle):
        if handle["__method__"] == "crash":
            raise RuntimeError("boom")
        return {
            "actual_metric": 0.9,
            "cost_usd": 1.0,
            "wall_clock_s": None,
            "artifacts": None,
            "mlflow_run_id": None,
        }

    result = run_sweep(
        task_id=TASK,
        hypotheses=hyps,
        submit_fn=_submit_fn,
        score_fn=score_fn,
        ledger=ledger,
        metric_name=METRIC,
    )

    # The failed (None-metric) outcome must sort after the scored one.
    assert result.outcomes[0].method == "ok"
    assert result.outcomes[-1].method == "crash"
    assert result.outcomes[-1].actual_metric is None


def test_outcome_is_frozen():
    o = SweepOutcome(
        experiment_id="x",
        method="m",
        config={},
        actual_metric=0.5,
        cost_usd=1.0,
        status="done",
    )
    with pytest.raises(Exception):
        o.status = "failed"  # type: ignore[misc]


# ── async driver (run_sweep_async) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_sweep_async_ranks_and_records(tmp_path):
    from agent.core.experiment_ledger import ExperimentLedger
    from agent.core.sweep import run_sweep_async

    ledger = ExperimentLedger(local_path=tmp_path / "exp.jsonl")
    submitted_order = []

    async def submit_fn(config):
        submitted_order.append(config["v"])
        return {"v": config["v"]}

    async def score_fn(handle):
        return {"actual_metric": handle["v"], "cost_usd": 1.0, "wall_clock_s": 5.0}

    hyps = [
        {"hypothesis": "a", "method": "m", "config": {"v": 0.90}},
        {"hypothesis": "b", "method": "m", "config": {"v": 0.95}},
        {"hypothesis": "c", "method": "m", "config": {"v": 0.92}},
    ]
    res = await run_sweep_async(
        task_id="t", hypotheses=hyps, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc",
    )
    assert [o.actual_metric for o in res.outcomes] == [0.95, 0.92, 0.90]
    assert res.best.actual_metric == 0.95
    assert res.n_submitted == 3
    assert res.total_cost_usd == pytest.approx(3.0)
    assert all(r.status == "done" for r in ledger.list_for_task("t"))


@pytest.mark.asyncio
async def test_run_sweep_async_isolates_submit_failure(tmp_path):
    from agent.core.experiment_ledger import ExperimentLedger
    from agent.core.sweep import run_sweep_async

    ledger = ExperimentLedger(local_path=tmp_path / "exp.jsonl")

    async def submit_fn(config):
        if config["v"] == 0.90:
            raise RuntimeError("submit boom")
        return {"v": config["v"]}

    async def score_fn(handle):
        return {"actual_metric": handle["v"], "cost_usd": None, "wall_clock_s": None}

    hyps = [
        {"hypothesis": "a", "method": "m", "config": {"v": 0.90}},
        {"hypothesis": "b", "method": "m", "config": {"v": 0.95}},
    ]
    res = await run_sweep_async(
        task_id="t", hypotheses=hyps, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc",
    )
    statuses = {o.status for o in res.outcomes}
    assert statuses == {"failed", "done"}
    assert res.best.actual_metric == 0.95
    assert res.n_failed == 1


def test_parse_metric():
    from agent.tools.sweep_jobs import parse_metric

    assert parse_metric("noise\nSWEEP_METRIC=0.9482031234567\nmore") == pytest.approx(
        0.9482031234567
    )
    with pytest.raises(ValueError):
        parse_metric("no sentinel here")
