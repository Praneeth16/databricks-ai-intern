"""Tests for the hypothesis generator — findings -> ranked, testable hypotheses.

Pure transformation + ranking, plus dedup against a real
:class:`ExperimentLedger` over a JSONL file (no live workspace).
"""

from __future__ import annotations

import pytest

from agent.core.experiment_ledger import ExperimentLedger
from agent.core.hypothesis import Hypothesis, generate_hypotheses, rank_hypotheses

TASK = "kaggle-f1-pitstops-s6e5"
METRIC = "roc_auc"


@pytest.fixture
def ledger(tmp_path):
    return ExperimentLedger(local_path=tmp_path / "exp.jsonl")


def _finding(
    method: str,
    reported: float | None = None,
    *,
    metric_name: str = METRIC,
    config: dict | None = None,
    title: str | None = None,
) -> dict:
    return {
        "title": title or f"{method} result",
        "method": method,
        "reported_metric": reported,
        "metric_name": metric_name,
        "config": config,
        "source_paper": f"paper-{method}",
        "source_section": "Table 1",
    }


# ── mapping ────────────────────────────────────────────────────────────────


def test_finding_maps_to_hypothesis_fields():
    findings = [_finding("lgbm-dart", 0.951, config={"lr": 0.05})]
    hyps = generate_hypotheses(findings, task_metric=METRIC)

    assert len(hyps) == 1
    h = hyps[0]
    assert isinstance(h, Hypothesis)
    assert h.method == "lgbm-dart"
    assert h.config == {"lr": 0.05}
    assert h.metric_name == METRIC
    assert h.expected_metric == 0.951  # == reported_metric
    assert h.source_paper == "paper-lgbm-dart"
    assert h.source_section == "Table 1"
    assert METRIC in h.hypothesis  # one-line testable claim mentions the metric


def test_empty_config_carries_method():
    hyps = generate_hypotheses([_finding("xgb", 0.94)], task_metric=METRIC)
    assert hyps[0].config == {"method": "xgb"}


# ── metric filter ────────────────────────────────────────────────────────────


def test_non_matching_metric_dropped():
    findings = [
        _finding("on-task", 0.95),
        _finding("off-task", 0.30, metric_name="rmse"),
    ]
    hyps = generate_hypotheses(findings, task_metric=METRIC)
    assert [h.method for h in hyps] == ["on-task"]


def test_metric_filter_is_case_insensitive():
    findings = [_finding("m", 0.9, metric_name="ROC_AUC")]
    hyps = generate_hypotheses(findings, task_metric="roc_auc")
    assert len(hyps) == 1


# ── ranking ──────────────────────────────────────────────────────────────────


def test_ranks_by_expected_lift_higher_first():
    findings = [
        _finding("small", 0.951),
        _finding("big", 0.970),
        _finding("mid", 0.960),
    ]
    hyps = generate_hypotheses(findings, task_metric=METRIC, current_best=0.950)

    assert [h.method for h in hyps] == ["big", "mid", "small"]
    # expected_lift = reported - current_best, positive = improvement.
    assert hyps[0].expected_lift == pytest.approx(0.020)
    assert hyps[-1].expected_lift == pytest.approx(0.001)


def test_lower_is_better_lift_direction():
    # loss: current best 0.40; a *lower* reported loss is a positive lift.
    findings = [
        _finding("worse", 0.45, metric_name="eval_loss"),
        _finding("better", 0.30, metric_name="eval_loss"),
    ]
    hyps = generate_hypotheses(
        findings,
        task_metric="eval_loss",
        current_best=0.40,
        higher_is_better=False,
    )
    assert [h.method for h in hyps] == ["better", "worse"]
    assert hyps[0].expected_lift == pytest.approx(0.10)   # 0.40 - 0.30
    assert hyps[1].expected_lift == pytest.approx(-0.05)  # 0.40 - 0.45


def test_no_current_best_falls_back_to_metric():
    findings = [
        _finding("lo", 0.90),
        _finding("hi", 0.96),
    ]
    hyps = generate_hypotheses(findings, task_metric=METRIC)
    assert all(h.expected_lift is None for h in hyps)
    # Fallback ranks by reported metric, higher first.
    assert [h.method for h in hyps] == ["hi", "lo"]


def test_lift_anchored_outranks_fallback_only():
    findings = [
        _finding("fallback_only", None),   # no reported metric -> no lift, score 0.0
        _finding("anchored", 0.951),       # has reported metric + current_best -> lift
    ]
    hyps = generate_hypotheses(findings, task_metric=METRIC, current_best=0.950)
    # anchored has a real lift; fallback_high has neither lift nor metric -> 0.0.
    assert hyps[0].method == "anchored"


# ── dedup ────────────────────────────────────────────────────────────────────


def test_dedup_drops_already_tried_keeps_novel(ledger):
    tried_config = {"lr": 0.05, "leaves": 31}
    novel_config = {"lr": 0.10, "leaves": 63}

    # Seed the ledger: propose + record a matching (task, method, config).
    eid = ledger.propose(
        task_id=TASK,
        hypothesis="seeded",
        method="lgbm",
        config=tried_config,
        metric_name=METRIC,
        expected_metric=0.95,
    )
    ledger.record_result(eid, actual_metric=0.949)

    findings = [
        _finding("lgbm", 0.951, config=tried_config),  # dup -> dropped
        _finding("lgbm", 0.951, config=novel_config),  # novel -> kept
    ]
    hyps = generate_hypotheses(
        findings, task_metric=METRIC, ledger=ledger, task_id=TASK
    )

    assert len(hyps) == 1
    assert hyps[0].config == novel_config


def test_no_dedup_without_ledger_or_task():
    config = {"lr": 0.05}
    findings = [_finding("lgbm", 0.95, config=config)]
    # ledger given but no task_id -> dedup disabled.
    hyps = generate_hypotheses(findings, task_metric=METRIC, ledger=None, task_id=TASK)
    assert len(hyps) == 1


# ── degenerate ───────────────────────────────────────────────────────────────


def test_empty_findings_returns_empty():
    assert generate_hypotheses([], task_metric=METRIC) == []


def test_rank_hypotheses_stable_on_ties():
    hyps = [
        Hypothesis("a", "a", {}, METRIC, None, None, None, None, 0.0),
        Hypothesis("b", "b", {}, METRIC, None, None, None, None, 0.0),
    ]
    ranked = rank_hypotheses(hyps)
    assert [h.method for h in ranked] == ["a", "b"]


def test_hypothesis_is_frozen():
    h = Hypothesis("c", "m", {}, METRIC, 0.9, None, None, None, 1.0)
    with pytest.raises(Exception):
        h.score = 2.0  # type: ignore[misc]
