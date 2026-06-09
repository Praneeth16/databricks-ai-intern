"""Live tests for the model_serving tool against a real Databricks workspace.

Two tiers:
- ``test_serving_cheap_ops_live`` — probe / plan / list. Cheap, no GPU. Runs
  whenever ``databricks_settings`` has creds.
- ``test_serving_full_roundtrip_live`` — build_and_register (serverless-GPU job)
  → deploy (SOD endpoint) → query → benchmark → delete + UC cleanup. ~20 min and
  spins real GPU compute, so it is opt-in behind ``RUN_SERVING_BUILD=1``.

This is the path that validated the tool end to end on fe-vm-lakebase-praneeth
(Qwen2.5-0.5B → A10, ~338 tok/s peak) and surfaced the FIPS/opencv,
notebook.exit, smoke-endpoint, and readiness bugs.

The registration target defaults to the praneeth test schema; override with
``SERVING_TEST_MODEL=<cat>.<schema>.<model>``.
"""

from __future__ import annotations

import json
import os

import pytest

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json")

# Qwen2.5-0.5B-Instruct architecture (small + cheap to build).
TINY_FACTS = {
    "params_billions": 0.5, "num_layers": 24, "hidden_size": 896, "kv_dim": 128,
    "num_attention_heads": 14, "num_kv_heads": 2, "native_precision": "fp16",
}
TINY_SOURCE = "Qwen/Qwen2.5-0.5B-Instruct"


class _Session:
    def __init__(self, cfg):
        self.config = cfg
        self.session_id = "itest-serving"
        self.databricks_user_token = None
        self.is_cancelled = False
        self.user_email = None


def _session():
    from agent.config import load_config
    return _Session(load_config(CFG_PATH))


@pytest.mark.asyncio
async def test_serving_cheap_ops_live(databricks_settings):
    """probe / plan / list against the live workspace — no GPU spend."""
    from agent.tools import model_serving_tool as mst
    sess = _session()

    probe = await mst.model_serving_handler({"operation": "probe_serving"}, session=sess)
    assert probe["isError"] is False and "GPU_MEDIUM" in probe["formatted"]

    listing = await mst.model_serving_handler({"operation": "list"}, session=sess)
    assert listing["isError"] is False

    plan = await mst.model_serving_handler({
        "operation": "plan_deployment", "model_facts": TINY_FACTS, "accuracy_budget": "max",
        "available_workload_types": ["GPU_MEDIUM"], "source_model": TINY_SOURCE,
        "served_model_name": "q",
    }, session=sess)
    assert plan["isError"] is False
    plans = json.loads(plan["formatted"].splitlines()[-1])
    assert plans and plans[0]["serving_config"]["workload_type"] == "GPU_MEDIUM"
    assert plans[0]["plan_hash"]


@pytest.mark.skipif(
    not os.environ.get("RUN_SERVING_BUILD"),
    reason="set RUN_SERVING_BUILD=1 to run the ~20min serverless-GPU build + deploy round-trip",
)
@pytest.mark.asyncio
async def test_serving_full_roundtrip_live(databricks_settings):
    """build_and_register → deploy → query → benchmark → delete, with UC cleanup."""
    from agent.core import db_client
    from agent.tools import model_serving_tool as mst

    registered = os.environ.get(
        "SERVING_TEST_MODEL",
        "serverless_lakebase_praneeth_catalog.ml_intern_test.itest_qwen05b",
    )
    endpoint = "itest-qwen05b-serving"
    sess = _session()

    plan_res = await mst.model_serving_handler({
        "operation": "plan_deployment", "model_facts": TINY_FACTS, "accuracy_budget": "max",
        "available_workload_types": ["GPU_MEDIUM"], "source_model": TINY_SOURCE,
        "served_model_name": "qwen05b", "registered_model_name": registered,
    }, session=sess)
    plan = json.loads(plan_res["formatted"].splitlines()[-1])[0]

    try:
        build = await mst.model_serving_handler(
            {"operation": "build_and_register", "plan": plan, "timeout": "1h"}, session=sess)
        assert build["isError"] is False, build["formatted"]
        assert "PASSED" in build["formatted"]

        deploy = await mst.model_serving_handler({
            "operation": "deploy", "plan": plan, "endpoint_name": endpoint,
            "model_version": "1", "on_exists": "update", "wait": True,
        }, session=sess)
        assert deploy["isError"] is False and "READY" in deploy["formatted"]

        query = await mst.model_serving_handler({
            "operation": "query", "endpoint_name": endpoint,
            "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 32,
        }, session=sess)
        assert query["isError"] is False and "responded" in query["formatted"]

        bench = await mst.model_serving_handler({
            "operation": "benchmark", "endpoint_name": endpoint,
            "concurrency_levels": [1, 4], "max_tokens": 64, "plan": plan, "model_version": "1",
        }, session=sess)
        assert bench["isError"] is False and "sys_tps" in bench["formatted"]
    finally:
        await mst.model_serving_handler(
            {"operation": "delete", "endpoint_name": endpoint}, session=sess)
        wc = db_client.get_workspace_client(databricks_settings)
        try:
            for v in wc.model_versions.list(registered):
                wc.model_versions.delete(registered, v.version)
            wc.registered_models.delete(registered)
        except Exception:
            pass
