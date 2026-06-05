"""ExperimentLedger DAO tests — JSONL backend, no live workspace."""

from __future__ import annotations

import pytest

from agent.core.experiment_ledger import ExperimentLedger, ExperimentRow


@pytest.fixture
def ledger(tmp_path):
    return ExperimentLedger(local_path=tmp_path / "experiments.jsonl")


# ── backend selection ────────────────────────────────────────────────────


def test_no_warehouse_uses_jsonl(ledger):
    assert ledger._use_sql is False
    assert ledger.table is None


def test_warehouse_settings_select_sql():
    from agent.core.db_client import DatabricksSettings

    settings = DatabricksSettings(
        host="https://x.databricks.com",
        warehouse_id="wh123",
        experiment_path="/exp",
        uc_catalog="cat",
        uc_schema="sch",
        uc_volume="vol",
        secret_scope="scope",
        lakebase_instance=None,
        instance_pool_id=None,
        default_node_type_id="node",
        default_runtime_version="15.4",
        prompt_registry_name="reg",
    )
    led = ExperimentLedger(settings=settings)
    assert led._use_sql is True
    assert led.table == "`cat`.`sch`.`experiments`"


def test_unsafe_catalog_identifier_rejected():
    from agent.core.db_client import DatabricksSettings

    settings = DatabricksSettings(
        host="https://x.databricks.com",
        warehouse_id="wh123",
        experiment_path="/exp",
        uc_catalog="cat`; DROP TABLE x; --",
        uc_schema="sch",
        uc_volume="vol",
        secret_scope="scope",
        lakebase_instance=None,
        instance_pool_id=None,
        default_node_type_id="node",
        default_runtime_version="15.4",
        prompt_registry_name="reg",
    )
    with pytest.raises(ValueError):
        ExperimentLedger(settings=settings)


# ── lifecycle end-to-end ──────────────────────────────────────────────────


def test_propose_returns_id_and_stores_proposed(ledger):
    exp_id = ledger.propose(
        task_id="t1",
        hypothesis="lean beats rich",
        method="lgbm_lean",
        config={"lr": 0.05, "depth": 6},
        metric_name="roc_auc",
        expected_metric=0.95,
        source_paper="arxiv:1234",
    )
    rows = ledger.list_for_task("t1")
    assert len(rows) == 1
    row = rows[0]
    assert row.experiment_id == exp_id
    assert row.status == "proposed"
    assert row.config == {"lr": 0.05, "depth": 6}
    assert row.created_at is not None
    assert row.expected_metric == 0.95


def test_mark_running_sets_status_and_run_id(ledger):
    exp_id = ledger.propose(
        task_id="t1", hypothesis="h", method="m", config={}, metric_name="roc_auc"
    )
    ledger.mark_running(exp_id, mlflow_run_id="run-abc")
    row = ledger.list_for_task("t1")[0]
    assert row.status == "running"
    assert row.mlflow_run_id == "run-abc"


def test_record_result_computes_repro_gap(ledger):
    exp_id = ledger.propose(
        task_id="t1",
        hypothesis="h",
        method="m",
        config={"a": 1},
        metric_name="roc_auc",
        expected_metric=0.95,
    )
    ledger.record_result(
        exp_id,
        actual_metric=0.94,
        cost_usd=1.5,
        wall_clock_s=120.0,
        artifacts={"model": "/Volumes/x/model.pkl"},
    )
    row = ledger.list_for_task("t1")[0]
    assert row.actual_metric == 0.94
    assert row.repro_gap == pytest.approx(0.95 - 0.94)
    assert row.status == "done"
    assert row.cost_usd == 1.5
    assert row.wall_clock_s == 120.0
    assert row.artifacts == {"model": "/Volumes/x/model.pkl"}


def test_record_result_no_expected_metric_leaves_gap_none(ledger):
    exp_id = ledger.propose(
        task_id="t1", hypothesis="h", method="m", config={}, metric_name="roc_auc"
    )
    ledger.record_result(exp_id, actual_metric=0.9)
    row = ledger.list_for_task("t1")[0]
    assert row.repro_gap is None


def test_reject_sets_status_and_reason(ledger):
    exp_id = ledger.propose(
        task_id="t1", hypothesis="h", method="m", config={}, metric_name="roc_auc"
    )
    ledger.reject(exp_id, "leakage detected")
    row = ledger.list_for_task("t1")[0]
    assert row.status == "rejected"
    assert row.notes == "leakage detected"


# ── queries ────────────────────────────────────────────────────────────────


def test_list_for_task_filters_by_task(ledger):
    ledger.propose(task_id="t1", hypothesis="h", method="m", config={}, metric_name="roc_auc")
    ledger.propose(task_id="t2", hypothesis="h", method="m", config={}, metric_name="roc_auc")
    assert len(ledger.list_for_task("t1")) == 1
    assert len(ledger.list_for_task("t2")) == 1


def test_best_for_task_both_directions(ledger):
    low = ledger.propose(task_id="t1", hypothesis="h", method="m", config={"x": 1}, metric_name="roc_auc")
    high = ledger.propose(task_id="t1", hypothesis="h", method="m", config={"x": 2}, metric_name="roc_auc")
    ledger.record_result(low, actual_metric=0.90)
    ledger.record_result(high, actual_metric=0.95)

    best_high = ledger.best_for_task("t1", "roc_auc", higher_is_better=True)
    assert best_high.experiment_id == high
    best_low = ledger.best_for_task("t1", "roc_auc", higher_is_better=False)
    assert best_low.experiment_id == low


def test_best_for_task_ignores_unscored_and_wrong_metric(ledger):
    scored = ledger.propose(task_id="t1", hypothesis="h", method="m", config={}, metric_name="roc_auc")
    ledger.propose(task_id="t1", hypothesis="h", method="m", config={}, metric_name="roc_auc")  # unscored
    ledger.propose(task_id="t1", hypothesis="h", method="m", config={}, metric_name="accuracy")  # other metric
    ledger.record_result(scored, actual_metric=0.93)
    best = ledger.best_for_task("t1", "roc_auc")
    assert best.experiment_id == scored


def test_best_for_task_empty_returns_none(ledger):
    assert ledger.best_for_task("nope", "roc_auc") is None


# ── dedup ──────────────────────────────────────────────────────────────────


def test_find_similar_config_hit_ignores_key_order(ledger):
    ledger.propose(
        task_id="t1", hypothesis="h", method="lgbm",
        config={"lr": 0.05, "depth": 6}, metric_name="roc_auc",
    )
    hit = ledger.find_similar_config("t1", "lgbm", {"depth": 6, "lr": 0.05})
    assert hit is not None
    assert hit.method == "lgbm"


def test_find_similar_config_miss_on_method_and_config(ledger):
    ledger.propose(
        task_id="t1", hypothesis="h", method="lgbm",
        config={"lr": 0.05}, metric_name="roc_auc",
    )
    assert ledger.find_similar_config("t1", "xgb", {"lr": 0.05}) is None
    assert ledger.find_similar_config("t1", "lgbm", {"lr": 0.1}) is None
    assert ledger.find_similar_config("other", "lgbm", {"lr": 0.05}) is None


# ── robustness ───────────────────────────────────────────────────────────────


def test_missing_file_lists_empty(tmp_path):
    led = ExperimentLedger(local_path=tmp_path / "does-not-exist.jsonl")
    assert led.list_for_task("t1") == []
    assert led.best_for_task("t1", "roc_auc") is None


def test_persists_across_instances(tmp_path):
    path = tmp_path / "experiments.jsonl"
    exp_id = ExperimentLedger(local_path=path).propose(
        task_id="t1", hypothesis="h", method="m", config={"a": 1}, metric_name="roc_auc"
    )
    reopened = ExperimentLedger(local_path=path)
    rows = reopened.list_for_task("t1")
    assert len(rows) == 1
    assert rows[0].experiment_id == exp_id
    assert isinstance(rows[0], ExperimentRow)


# ── metric direction: loss metrics must orient repro_gap correctly ────────


def test_repro_gap_oriented_for_loss_metric(ledger):
    exp_id = ledger.propose(
        task_id="ft", hypothesis="lower loss", method="sft",
        config={"lr": 1e-4}, metric_name="eval_loss", expected_metric=0.30,
    )
    # Missed the target (loss higher than expected) → positive gap (underperformed).
    ledger.record_result(exp_id, actual_metric=0.45)
    assert ledger.get(exp_id).repro_gap == pytest.approx(0.15)


def test_repro_gap_negative_when_loss_beats_target(ledger):
    exp_id = ledger.propose(
        task_id="ft", hypothesis="lower loss", method="sft",
        config={"lr": 1e-4}, metric_name="eval_loss", expected_metric=0.30,
    )
    ledger.record_result(exp_id, actual_metric=0.20)  # beat the target
    assert ledger.get(exp_id).repro_gap == pytest.approx(-0.10)


def test_find_similar_ignores_rows_without_result(ledger):
    cfg = {"lr": 0.05}
    rejected = ledger.propose(
        task_id="t", hypothesis="h", method="m", config=cfg, metric_name="roc_auc",
    )
    ledger.reject(rejected, "transient submit failure")
    # A rejected (no-result) row must not block a re-run of the same config.
    assert ledger.find_similar_config("t", "m", cfg) is None
