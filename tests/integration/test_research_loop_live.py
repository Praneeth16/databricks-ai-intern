"""Live end-to-end test for the deterministic research loop.

Runs a real multi-round loop: a pool of 4 variants, batch_size 2, so round 1
sweeps two serverless jobs in parallel and round 2 sweeps the other two. Proves
the runner closes the loop on real Databricks Jobs — parallel sweep per round,
ledger recording, deterministic acceptance + stop — not just in unit fakes.

Gated by ``databricks_settings`` (auto-skips without creds). Scopes ledger rows
to a unique task_id and deletes them in teardown. Submits 4 short serverless
jobs, so it takes a few minutes.
"""

from __future__ import annotations

import uuid

import pytest

from agent.core import db_client
from agent.core.experiment_ledger import ExperimentLedger
from agent.core.research_loop import run_research_loop
from agent.tools.databricks_jobs_tool import DatabricksJobsTool
from agent.tools.sweep_jobs import build_jobs_callables
from agent.tools.sweep_tool import _render_script

_TEMPLATE = "print(f'SWEEP_METRIC={CONFIG[\"v\"]}')\n"


@pytest.fixture
def sql_ledger(databricks_settings):
    if not databricks_settings.warehouse_id:
        pytest.skip("DATABRICKS_WAREHOUSE_ID not set — SQL ledger needs a warehouse.")
    ledger = ExperimentLedger(settings=databricks_settings)
    ledger.ensure_table()
    task_id = f"loop-itest-{uuid.uuid4().hex[:12]}"
    yield ledger, task_id
    conn = db_client.get_sql_connection(databricks_settings)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {ledger.table} WHERE task_id = ?", [task_id])
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_live_research_loop_over_serverless_jobs(sql_ledger, databricks_settings):
    ledger, task_id = sql_ledger
    wc = db_client.get_workspace_client(databricks_settings)
    jobs_tool = DatabricksJobsTool(
        wc, databricks_settings, user_email="praneeth.paikray@databricks.com"
    )
    submit_fn, score_fn = build_jobs_callables(
        jobs_tool,
        render_script=lambda cfg: _render_script(_TEMPLATE, cfg),
        base_args={"timeout": "10m"},
        kind="serverless",
    )

    pool = [
        {"hypothesis": "a", "method": "const", "config": {"v": 0.90}},
        {"hypothesis": "b", "method": "const", "config": {"v": 0.95}},
        {"hypothesis": "c", "method": "const", "config": {"v": 0.91}},
        {"hypothesis": "d", "method": "const", "config": {"v": 0.96}},
    ]
    cursor = {"i": 0}

    def source(ctx):
        i = cursor["i"]
        cursor["i"] = i + 2
        return pool[i:i + 2]

    result = await run_research_loop(
        task_id=task_id,
        hypothesis_source=source,
        submit_fn=submit_fn,
        score_fn=score_fn,
        ledger=ledger,
        metric_name="roc_auc",
        higher_is_better=True,
        max_rounds=2,
        patience=5,
    )

    assert result.stop_reason == "max_rounds"
    assert len(result.rounds) == 2
    assert result.rounds[0].n_hypotheses == 2  # parallel batch per round
    assert result.best_metric == pytest.approx(0.96)
    assert result.accepted_count == 2  # 0.95 in r1, 0.96 in r2
    # All four variants recorded to the ledger as done.
    done = [r for r in ledger.list_for_task(task_id) if r.status == "done"]
    assert len(done) == 4
