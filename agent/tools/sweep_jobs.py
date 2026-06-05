"""Adapter: drive ``sweep.run_sweep_async`` over real Databricks Jobs.

Reuses :class:`DatabricksJobsTool`'s async building blocks (stage script,
build submit body, wait for terminal, fetch output) so job-submission
correctness — env filtering, serverless task shape, retry-aware output picking
— is not duplicated here. Each sweep variant renders to its own script with a
unique filename, submits as a serverless run (``submit_fn`` returns the run id
without blocking so phase-1 submissions fan out), and reports its metric via a
stdout sentinel line::

    SWEEP_METRIC=<float>

Verified to survive ``runs/get-output`` on a serverless ``spark_python_task``.
``score_fn`` waits for the run, requires SUCCESS, and parses that line.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Any, Callable

from agent.tools.databricks_jobs_tool import DatabricksJobsTool, _JOBS_SUBMIT_PATH

logger = logging.getLogger(__name__)

SENTINEL = "SWEEP_METRIC"
_METRIC_RE = re.compile(rf"^{SENTINEL}=([-+0-9.eE]+)\s*$", re.MULTILINE)

# config dict -> a Python script that computes the metric and prints
# ``SWEEP_METRIC=<float>`` on its own line.
RenderFn = Callable[[dict], str]


def parse_metric(output: str) -> float:
    """Extract the metric from a job's stdout sentinel line."""
    m = _METRIC_RE.search(output or "")
    if not m:
        raise ValueError(
            f"no '{SENTINEL}=<float>' line in job output "
            f"({len(output or '')} chars captured)"
        )
    return float(m.group(1))


def build_jobs_callables(
    jobs_tool: DatabricksJobsTool,
    *,
    render_script: RenderFn,
    base_args: dict | None = None,
    kind: str = "serverless",
):
    """Build ``(submit_fn, score_fn)`` for :func:`sweep.run_sweep_async`.

    ``render_script(config)`` must return a script that prints
    ``SWEEP_METRIC=<float>``. ``base_args`` is merged into every job's args
    (e.g. ``timeout``, ``dependencies``). Each config gets a unique filename so
    staging never collides across variants sharing one session.
    """
    base = dict(base_args or {})
    as_notebook = kind == "serverless_gpu"

    async def submit_fn(config: dict) -> int:
        args = {
            **base,
            "kind": kind,
            "script": render_script(config),
            "filename": f"sweep_{uuid.uuid4().hex[:12]}.py",
        }
        ws_path = await jobs_tool._resolve_or_stage_script(args, as_notebook=as_notebook)
        body = await jobs_tool._build_submit_body(args, ws_path, kind)
        resp = await asyncio.to_thread(
            jobs_tool.wc.api_client.do, "POST", _JOBS_SUBMIT_PATH, body=body
        )
        run_id = resp.get("run_id")
        if not run_id:
            raise RuntimeError(f"runs/submit returned no run_id: {resp}")
        return run_id

    async def score_fn(run_id: int) -> dict:
        run = await jobs_tool._wait_for_run(run_id)
        state = run.get("state") or {}
        if state.get("result_state") != "SUCCESS":
            raise RuntimeError(
                f"run {run_id} ended "
                f"{state.get('result_state') or state.get('life_cycle_state')!r}: "
                f"{state.get('state_message') or '—'}"
            )
        output = await jobs_tool._fetch_run_output(run)
        metric = parse_metric(output)
        wall_clock_s = None
        start, end = run.get("start_time"), run.get("end_time")
        if start and end and end >= start:
            wall_clock_s = (end - start) / 1000.0
        return {
            "actual_metric": metric,
            "cost_usd": None,  # serverless run cost lands in system.billing.usage (lagged)
            "wall_clock_s": wall_clock_s,
            "mlflow_run_id": str(run_id),  # jobs run id, for traceability
            "artifacts": None,
        }

    return submit_fn, score_fn
