"""Eval harness tests: task-spec parse, scorers, runner, report."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from evals import scorers
from evals.report import write_report
from evals.runner import EvalResult, run_eval
from evals.task_spec import EvalTask, load_task

_REPO_ROOT = Path(__file__).resolve().parents[2]

_KAGGLE_TASK = _REPO_ROOT / "evals" / "tasks" / "kaggle-f1-pitstops-s6e5.yaml"


# ── in-memory ledger fake (Phase 0 frozen interface, just the 3 methods) ──


class FakeLedger:
    def __init__(self) -> None:
        self.proposed: list[dict] = []
        self.running: list[tuple] = []
        self.results: list[tuple] = []
        self._next_id = 0

    def propose(self, **kwargs) -> str:
        self.proposed.append(kwargs)
        self._next_id += 1
        return f"exp-{self._next_id}"

    def mark_running(self, experiment_id, mlflow_run_id=None) -> None:
        self.running.append((experiment_id, mlflow_run_id))

    def record_result(self, experiment_id, **kwargs) -> None:
        self.results.append((experiment_id, kwargs))


# ── task_spec ─────────────────────────────────────────────────────────


def test_load_kaggle_task():
    task = load_task(_KAGGLE_TASK)
    assert task.id == "kaggle-f1-pitstops-s6e5"
    assert task.kind == "tabular"
    assert task.metric == "roc_auc"
    assert task.higher_is_better is True
    assert task.baseline_score == pytest.approx(0.94820)
    assert task.human_ceiling == pytest.approx(0.94924)
    assert task.leaderboard["top_public"] == pytest.approx(0.9545)
    assert task.holdout["type"] == "temporal"
    assert task.holdout["column"] == "Year"
    assert task.budget["max_iterations"] == 30


def test_load_task_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_task(tmp_path / "nope.yaml")


def test_load_task_malformed_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("::not valid::\n: : :")
    with pytest.raises(ValueError):
        load_task(p)


def test_load_task_missing_required_field(tmp_path):
    p = tmp_path / "incomplete.yaml"
    p.write_text(textwrap.dedent("""\
        id: t1
        kind: tabular
    """))
    with pytest.raises(ValueError, match="required field"):
        load_task(p)


def test_load_task_invalid_kind(tmp_path):
    p = tmp_path / "badkind.yaml"
    p.write_text(textwrap.dedent("""\
        id: t1
        kind: quantum
        metric: roc_auc
        higher_is_better: true
    """))
    with pytest.raises(ValueError, match="kind"):
        load_task(p)


# ── scorers ───────────────────────────────────────────────────────────


def test_roc_auc_perfectly_separable():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert scorers.roc_auc(y_true, y_score) == pytest.approx(1.0)


def test_roc_auc_inverted_is_zero():
    y_true = [0, 0, 1, 1]
    y_score = [0.9, 0.8, 0.2, 0.1]
    assert scorers.roc_auc(y_true, y_score) == pytest.approx(0.0)


def test_roc_auc_rank_fallback_matches():
    # Force the pure-Python path and compare to a known value.
    y_true = [0, 1, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8, 0.7]
    expected = scorers._roc_auc_rank(y_true, y_score)
    assert scorers.roc_auc(y_true, y_score) == pytest.approx(expected)


def test_accuracy():
    assert scorers.accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == pytest.approx(0.75)


def test_rank_percentile_ordering():
    lb = [0.90, 0.92, 0.94]
    assert scorers.rank_percentile(0.95, lb) == pytest.approx(100.0)
    assert scorers.rank_percentile(0.93, lb) == pytest.approx(200.0 / 3.0)
    assert scorers.rank_percentile(0.80, lb) == pytest.approx(0.0)


def test_rank_percentile_empty_leaderboard():
    assert scorers.rank_percentile(0.5, []) == pytest.approx(100.0)


def test_eval_loss_mean():
    assert scorers.eval_loss([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert scorers.eval_loss([0.7]) == pytest.approx(0.7)


# ── runner end-to-end ─────────────────────────────────────────────────


def test_run_eval_with_predictions():
    task = load_task(_KAGGLE_TASK)
    ledger = FakeLedger()

    def stub_agent(t: EvalTask) -> dict:
        return {
            "predictions": [0.1, 0.2, 0.8, 0.9],
            "y_true": [0, 0, 1, 1],
            "lb_score": 0.90,
            "iterations": 5,
            "cost_usd": 3.5,
            "wall_clock_s": 120.0,
            "self_recovered_failures": 2,
            "mlflow_run_id": "run-abc",
        }

    result = run_eval(task, stub_agent, ledger, session_id="sess-1")

    assert isinstance(result, EvalResult)
    assert result.task_id == task.id
    assert result.score == pytest.approx(1.0)
    assert result.cv_lb_gap == pytest.approx(1.0 - 0.90)
    # Score 1.0 beats the single top_public anchor → 100th percentile.
    assert result.leaderboard_percentile == pytest.approx(100.0)
    assert result.cost_usd == pytest.approx(3.5)
    assert result.wall_clock_s == pytest.approx(120.0)
    assert result.iterations == 5
    assert result.self_recovered_failures == 2

    # Ledger lifecycle: proposed once, marked running once, recorded once.
    assert len(ledger.proposed) == 1
    assert ledger.proposed[0]["task_id"] == task.id
    assert ledger.proposed[0]["session_id"] == "sess-1"
    assert ledger.running == [(result.experiment_id, "run-abc")]
    assert len(ledger.results) == 1
    rec_id, rec_kwargs = ledger.results[0]
    assert rec_id == result.experiment_id
    assert rec_kwargs["actual_metric"] == pytest.approx(1.0)
    assert rec_kwargs["status"] == "done"


def test_run_eval_with_direct_score():
    task = load_task(_KAGGLE_TASK)
    ledger = FakeLedger()

    def stub_agent(t: EvalTask) -> dict:
        return {
            "score": 0.94924,
            "iterations": 12,
            "cost_usd": 8.0,
            "wall_clock_s": 600.0,
            "self_recovered_failures": 0,
        }

    result = run_eval(task, stub_agent, ledger)
    assert result.score == pytest.approx(0.94924)
    assert result.cv_lb_gap is None  # no lb_score supplied
    # Below top_public anchor (0.9545) → 0th percentile.
    assert result.leaderboard_percentile == pytest.approx(0.0)
    assert len(ledger.results) == 1


def test_run_eval_predictions_without_y_true_raises():
    task = load_task(_KAGGLE_TASK)
    ledger = FakeLedger()

    def stub_agent(t: EvalTask) -> dict:
        return {"predictions": [0.1, 0.9], "iterations": 1,
                "cost_usd": 0.0, "wall_clock_s": 1.0, "self_recovered_failures": 0}

    with pytest.raises(ValueError, match="y_true"):
        run_eval(task, stub_agent, ledger)


# ── report ────────────────────────────────────────────────────────────


def test_write_report_produces_both_files(tmp_path):
    task = load_task(_KAGGLE_TASK)
    result = EvalResult(
        task_id=task.id,
        metric_name="roc_auc",
        score=0.94924,
        experiment_id="exp-1",
        cv_lb_gap=0.042,
        leaderboard_percentile=0.0,
        cost_usd=8.0,
        wall_clock_s=600.0,
        iterations=12,
        self_recovered_failures=1,
    )
    md_path = write_report(result, task, tmp_path)
    json_path = md_path.with_suffix(".json")

    assert md_path.exists()
    assert json_path.exists()

    md = md_path.read_text()
    assert task.id in md
    assert "0.94924" in md
    assert "CV↔LB gap" in md

    payload = json.loads(json_path.read_text())
    assert payload["score"] == pytest.approx(0.94924)
    assert payload["cv_lb_gap"] == pytest.approx(0.042)
    assert payload["experiment_id"] == "exp-1"
