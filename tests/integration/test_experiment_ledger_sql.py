"""Live SQL-backend test for the experiment ledger.

Unit tests cover the JSONL fallback; this exercises the path they cannot —
real UC Delta writes through databricks.sql: ``?``-parameter binding,
``CAST(? AS DOUBLE)`` (so AUC survives past 5 decimals), backtick-quoted
identifier, INSERT/UPDATE/WHERE pushdown, and JSON round-trip.

Gated by the ``databricks_settings`` fixture (auto-skips without creds). Scopes
every row to a unique task_id and deletes them in teardown, so it is safe to
run against the shared ``experiments`` table.
"""

from __future__ import annotations

import uuid

import pytest

from agent.core import db_client
from agent.core.experiment_ledger import ExperimentLedger


@pytest.fixture
def sql_ledger(databricks_settings):
    if not databricks_settings.warehouse_id:
        pytest.skip("DATABRICKS_WAREHOUSE_ID not set — SQL ledger needs a warehouse.")
    ledger = ExperimentLedger(settings=databricks_settings)
    assert ledger._use_sql is True, "expected SQL backend with a warehouse configured"
    ledger.ensure_table()
    task_id = f"itest-{uuid.uuid4().hex[:12]}"
    yield ledger, task_id
    # Teardown: drop only this test's rows from the shared table.
    conn = db_client.get_sql_connection(databricks_settings)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {ledger.table} WHERE task_id = ?", [task_id])
    finally:
        conn.close()


def test_sql_ledger_full_lifecycle_and_double_precision(sql_ledger):
    ledger, task_id = sql_ledger

    # AUC value chosen so float32 narrowing (~1e-8 error) would fail the 1e-9
    # tolerance below — this is the regression test for the DOUBLE-cast fix.
    expected_auc = 0.9492410
    actual_auc = 0.9482031234567
    auc_id = ledger.propose(
        task_id=task_id,
        hypothesis="xgb+cb blend reproduces the v5.2 ceiling",
        method="xgb_cb_blend",
        config={"lr": 0.05, "n_estimators": 800, "feats": 14},
        metric_name="roc_auc",
        expected_metric=expected_auc,
        source_paper="kaggle:s6e5",
    )
    ledger.mark_running(auc_id, mlflow_run_id="run-xyz")
    ledger.record_result(
        auc_id,
        actual_metric=actual_auc,
        cost_usd=1.2345,
        wall_clock_s=99.5,
        artifacts={"model": "/Volumes/x/m.pkl"},
    )

    row = ledger.get(auc_id)
    assert row is not None
    assert row.status == "done"
    assert row.mlflow_run_id == "run-xyz"
    # DOUBLE, not FLOAT: full precision survives the round-trip.
    assert abs(row.actual_metric - actual_auc) < 1e-9
    assert row.repro_gap == pytest.approx(expected_auc - actual_auc, abs=1e-9)
    # JSON columns round-trip back to dicts.
    assert row.config == {"lr": 0.05, "n_estimators": 800, "feats": 14}
    assert row.artifacts == {"model": "/Volumes/x/m.pkl"}

    # Loss metric: repro_gap must be oriented so positive = underperformed,
    # through the SQL path too.
    loss_id = ledger.propose(
        task_id=task_id,
        hypothesis="lower eval_loss",
        method="sft",
        config={"lr": 1e-4},
        metric_name="eval_loss",
        expected_metric=0.30,
    )
    ledger.record_result(loss_id, actual_metric=0.45)  # missed (loss too high)
    assert ledger.get(loss_id).repro_gap == pytest.approx(0.15, abs=1e-9)

    # WHERE-pushdown query returns exactly this task's rows.
    rows = ledger.list_for_task(task_id)
    assert {r.experiment_id for r in rows} == {auc_id, loss_id}

    # best_for_task picks the AUC row for roc_auc.
    best = ledger.best_for_task(task_id, "roc_auc")
    assert best is not None and best.experiment_id == auc_id

    # Dedup hits the completed config.
    hit = ledger.find_similar_config(
        task_id, "xgb_cb_blend", {"feats": 14, "lr": 0.05, "n_estimators": 800}
    )
    assert hit is not None and hit.experiment_id == auc_id
