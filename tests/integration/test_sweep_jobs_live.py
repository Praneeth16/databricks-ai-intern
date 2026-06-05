"""Live end-to-end test for the sweep -> Databricks Jobs wiring.

Runs a real 2-config sweep: each variant is a serverless job that prints its
metric via the ``SWEEP_METRIC`` stdout sentinel. Exercises the full stack the
unit tests cannot — ``run_sweep_async`` fanning out two concurrent ``runs/submit``
through ``sweep_jobs.build_jobs_callables``, terminal-state polling, output
parsing, and recording to the SQL-backed ledger. Then re-runs the same configs
to prove dedup skips them.

Gated by ``databricks_settings`` (auto-skips without creds). Scopes ledger rows
to a unique task_id and deletes them in teardown. Submits two short serverless
jobs, so it takes a couple of minutes.
"""

from __future__ import annotations

import uuid

import pytest

from agent.core import db_client
from agent.core.experiment_ledger import ExperimentLedger
from agent.core.sweep import run_sweep_async
from agent.tools.databricks_jobs_tool import DatabricksJobsTool
from agent.tools.sweep_jobs import build_jobs_callables


def _render(config: dict) -> str:
    # Trivial "training": echo the config's target metric via the sentinel.
    return f"print('SWEEP_METRIC={config['v']}')\n"


@pytest.fixture
def sql_ledger(databricks_settings):
    if not databricks_settings.warehouse_id:
        pytest.skip("DATABRICKS_WAREHOUSE_ID not set — SQL ledger needs a warehouse.")
    ledger = ExperimentLedger(settings=databricks_settings)
    ledger.ensure_table()
    task_id = f"sweep-itest-{uuid.uuid4().hex[:12]}"
    yield ledger, task_id
    conn = db_client.get_sql_connection(databricks_settings)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {ledger.table} WHERE task_id = ?", [task_id])
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_live_sweep_over_serverless_jobs(sql_ledger, databricks_settings):
    ledger, task_id = sql_ledger
    wc = db_client.get_workspace_client(databricks_settings)
    jobs_tool = DatabricksJobsTool(
        wc, databricks_settings, user_email="praneeth.paikray@databricks.com"
    )
    submit_fn, score_fn = build_jobs_callables(
        jobs_tool, render_script=_render, base_args={"timeout": "10m"}, kind="serverless",
    )

    hyps = [
        {"hypothesis": "low variant", "method": "const", "config": {"v": 0.911}},
        {"hypothesis": "high variant", "method": "const", "config": {"v": 0.953}},
    ]

    res = await run_sweep_async(
        task_id=task_id, hypotheses=hyps, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", higher_is_better=True,
    )

    assert res.n_submitted == 2
    assert res.n_failed == 0
    assert res.best is not None
    assert res.best.actual_metric == pytest.approx(0.953)
    assert [o.actual_metric for o in res.outcomes] == [0.953, 0.911]

    rows = ledger.list_for_task(task_id)
    assert len(rows) == 2
    assert all(r.status == "done" for r in rows)
    assert all(r.mlflow_run_id for r in rows)  # jobs run id stamped

    # Re-running the same configs must skip via dedup (no new jobs submitted).
    res2 = await run_sweep_async(
        task_id=task_id, hypotheses=hyps, submit_fn=submit_fn, score_fn=score_fn,
        ledger=ledger, metric_name="roc_auc", higher_is_better=True,
    )
    assert res2.n_submitted == 0
    assert res2.n_skipped == 2
