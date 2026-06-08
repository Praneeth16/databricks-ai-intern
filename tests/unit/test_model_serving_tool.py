"""Unit tests for the model_serving tool (offline surface).

Covers the deterministic boundary (plan_hash contract), the REST deploy body
(fixed concurrency, no autoscaling), query 429 backoff, registration + approval
gating, and ledger recording. The serverless-GPU build + live benchmark land in
Phase 7 build-step 4 and are integration-gated.
"""

from __future__ import annotations

import base64
import json

import pytest

from agent.core.experiment_ledger import ExperimentLedger
from agent.core.serving_strategy import ModelFacts, feasible_configs
from agent.tools import model_serving_tool as mst


# --- fixtures ----------------------------------------------------------------

class _FakeServingEndpoints:
    def __init__(self, existing=None):
        self._existing = set(existing or [])
        self.deleted: list[str] = []

    def get(self, name):
        if name not in self._existing:
            raise RuntimeError(f"{name} does not exist")
        return type("EP", (), {"name": name, "state": None, "config": None})()

    def list(self):
        return []

    def delete(self, name):
        self.deleted.append(name)


class _FakeApiClient:
    def __init__(self, responses=None, errors=None):
        self.calls: list[tuple] = []
        self._responses = list(responses or [])
        self._errors = list(errors or [])

    def do(self, method, path, body=None):
        self.calls.append((method, path, body))
        if self._errors:
            err = self._errors.pop(0)
            if err:
                raise RuntimeError(err)
        return self._responses.pop(0) if self._responses else {}


class _FakeWC:
    def __init__(self, existing=None, responses=None, errors=None):
        self.serving_endpoints = _FakeServingEndpoints(existing)
        self.api_client = _FakeApiClient(responses, errors)


def _patch_context(monkeypatch, wc):
    monkeypatch.setattr(mst, "_context", lambda session: (None, wc))


def _plan(**overrides):
    model = ModelFacts(params_billions=4.0, num_layers=36, hidden_size=2560, kv_dim=1024,
                       native_precision="fp16")
    cfg = feasible_configs(model=model, available_workload_types=["GPU_MEDIUM"],
                           accuracy_budget="max")[0]
    args = {"source_model": "org/model-4b", "served_model_name": "m",
            "registered_model_name": "cat.sch.m4b"}
    args.update(overrides)
    return mst._make_plan(cfg, args, model)


# --- registration + approval -------------------------------------------------

def test_registered_in_builtin_tools():
    from agent.core.tools import create_builtin_tools
    assert "model_serving" in {t.name for t in create_builtin_tools()}


def test_approval_gating_by_operation():
    from agent.core.agent_loop import _needs_approval
    for op in ("build_and_register", "deploy", "benchmark", "delete"):
        assert _needs_approval("model_serving", {"operation": op}, None) is True
    for op in ("plan_deployment", "query", "list", "probe_serving"):
        assert _needs_approval("model_serving", {"operation": op}, None) is False


# --- plan_deployment + plan_hash contract ------------------------------------

@pytest.mark.asyncio
async def test_plan_deployment_returns_plans_with_hash():
    res = await mst.model_serving_handler({
        "operation": "plan_deployment",
        "model_facts": {"params_billions": 4.0, "num_layers": 36, "hidden_size": 2560, "kv_dim": 1024},
        "available_workload_types": ["GPU_MEDIUM"], "accuracy_budget": "max",
        "source_model": "org/m", "served_model_name": "m",
    })
    assert res["isError"] is False
    payload = json.loads(res["formatted"].splitlines()[-1])
    assert payload and all("plan_hash" in p and "entrypoint" in p for p in payload)
    assert all(p["serving_config"]["workload_type"] == "GPU_MEDIUM" for p in payload)


@pytest.mark.asyncio
async def test_plan_deployment_missing_model_facts_errors():
    res = await mst.model_serving_handler({"operation": "plan_deployment"})
    assert res["isError"] is True


def test_plan_hash_is_stable_and_detects_tampering():
    p = _plan()
    assert mst._plan_hash(p) == p["plan_hash"]      # stable
    mst._validate_plan(p)                            # round-trips
    p["serving_config"]["precision"] = "fp8"         # tamper
    with pytest.raises(mst._ToolError):
        mst._validate_plan(p)


# --- deploy body + execution -------------------------------------------------

def test_deploy_body_is_fixed_concurrency_no_autoscale():
    body = mst._build_deploy_body(
        endpoint_name="ep", served_name="m", registered_model_name="cat.sch.m",
        model_version="3", workload_type="MULTIGPU_MEDIUM", provisioned_concurrency=6,
    )
    se = body["config"]["served_entities"][0]
    assert se["min_provisioned_concurrency"] == se["max_provisioned_concurrency"] == 8  # snapped ×4
    assert se["scale_to_zero_enabled"] is False
    assert se["entity_name"] == "cat.sch.m" and se["entity_version"] == "3"
    assert se["workload_type"] == "MULTIGPU_MEDIUM"


@pytest.mark.asyncio
async def test_deploy_creates_via_rest(monkeypatch):
    wc = _FakeWC(existing=[])
    _patch_context(monkeypatch, wc)
    res = await mst.model_serving_handler(
        {"operation": "deploy", "plan": _plan(), "endpoint_name": "ep", "model_version": "2"})
    assert res["isError"] is False and "created" in res["formatted"]
    method, path, _ = wc.api_client.calls[0]
    assert method == "POST" and path == "/api/2.0/serving-endpoints"


@pytest.mark.asyncio
async def test_deploy_fails_when_endpoint_exists_without_update(monkeypatch):
    wc = _FakeWC(existing=["ep"])
    _patch_context(monkeypatch, wc)
    res = await mst.model_serving_handler(
        {"operation": "deploy", "plan": _plan(), "endpoint_name": "ep", "model_version": "2"})
    assert res["isError"] is True and "already exists" in res["formatted"]


@pytest.mark.asyncio
async def test_deploy_updates_existing_with_on_exists(monkeypatch):
    wc = _FakeWC(existing=["ep"])
    _patch_context(monkeypatch, wc)
    res = await mst.model_serving_handler(
        {"operation": "deploy", "plan": _plan(), "endpoint_name": "ep",
         "model_version": "2", "on_exists": "update"})
    assert res["isError"] is False and "updated" in res["formatted"]
    method, path, _ = wc.api_client.calls[0]
    assert method == "PUT" and path.endswith("/config")


@pytest.mark.asyncio
async def test_deploy_rejects_tampered_plan(monkeypatch):
    _patch_context(monkeypatch, _FakeWC())
    bad = _plan()
    bad["serving_config"]["precision"] = "fp8"  # hash no longer matches
    res = await mst.model_serving_handler(
        {"operation": "deploy", "plan": bad, "endpoint_name": "ep", "model_version": "2"})
    assert res["isError"] is True and "plan_hash mismatch" in res["formatted"]


# --- query 429 backoff -------------------------------------------------------

@pytest.mark.asyncio
async def test_query_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(mst.time, "sleep", lambda *_: None)
    wc = _FakeWC(
        responses=[{"choices": [{"message": {"content": "hi there"}}]}],
        errors=["HTTP 429 Too many parallel requests", None],  # first call 429, second ok
    )
    _patch_context(monkeypatch, wc)
    res = await mst.model_serving_handler(
        {"operation": "query", "endpoint_name": "ep", "messages": [{"role": "user", "content": "hi"}]})
    assert res["isError"] is False and "hi there" in res["formatted"]
    assert len(wc.api_client.calls) == 2  # retried once


# --- list / delete / probe ---------------------------------------------------

@pytest.mark.asyncio
async def test_delete_calls_sdk(monkeypatch):
    wc = _FakeWC(existing=["ep"])
    _patch_context(monkeypatch, wc)
    res = await mst.model_serving_handler({"operation": "delete", "endpoint_name": "ep"})
    assert res["isError"] is False and wc.serving_endpoints.deleted == ["ep"]


def test_poll_ready_does_not_false_positive_on_not_ready():
    # "NOT_READY".endswith("READY") is True — the readiness check must compare the
    # trailing enum token exactly, or a still-provisioning endpoint reads as READY.
    class _State:
        def __init__(self, ready, cfg=None):
            self.ready, self.config_update = ready, cfg

    class _EP:
        def __init__(self, st):
            self.state = st

    def _wc(state):
        return type("W", (), {"serving_endpoints": type("SE", (), {
            "get": lambda self, n: _EP(state)})()})()

    assert "READY" in mst._poll_ready(_wc(_State("EndpointStateReady.READY")), "ep",
                                      timeout_s=5, interval_s=0)
    # NOT_READY + a failed config update must NOT report READY.
    msg = mst._poll_ready(_wc(_State("EndpointStateReady.NOT_READY", "EndpointStateConfigUpdate.UPDATE_FAILED")),
                          "ep", timeout_s=5, interval_s=0)
    assert "FAILED" in msg and "is READY" not in msg


@pytest.mark.asyncio
async def test_probe_serving_reports_tiers(monkeypatch):
    _patch_context(monkeypatch, _FakeWC())
    res = await mst.model_serving_handler({"operation": "probe_serving"})
    assert res["isError"] is False
    assert "GPU_XLARGE" in res["formatted"] and "GPU_MEDIUM" in res["formatted"]


# --- build_and_register ------------------------------------------------------

def _build_sentinel(plan, *, version="7", smoke=True, hash_override=None):
    result = {
        "registered_model_name": plan["registered_model_name"], "model_version": version,
        "mlflow_run_id": "run-1", "plan_hash": hash_override or plan["plan_hash"],
        "artifact_manifest_sha256": "abc", "smoke_passed": smoke,
        "vllm_version": "0.11.2", "transformers_version": "4.57.6",
    }
    b64 = base64.b64encode(json.dumps(result).encode()).decode()
    return f"...job log...\n{mst._BUILD_SENTINEL}{b64}\n...more log...\n"


def test_render_build_script_has_pins_smoke_register_sentinel():
    plan = _plan()
    script = mst._render_build_script(plan, plan["registered_model_name"])
    assert "snapshot_download" in script and plan["source_model"] in script
    assert 'env_pack="databricks_model_serving"' in script
    assert '"task": "llm/v1/chat"' in script and '"entrypoint": SERVING_ENTRYPOINT' in script
    assert '"plan_hash": PLAN_HASH' in script
    assert mst._BUILD_SENTINEL in script
    assert "Application startup complete" in script  # local smoke gate
    assert plan["plan_hash"] in script
    # Regression guards for bugs found in live validation:
    assert "/v1/chat/completions" in script   # smoke hits the stock vLLM route, NOT /invocations
    assert "dbutils.notebook.exit" in script  # results flow via notebook_output, not stdout


def test_build_accelerator_sizing():
    small = {"est_vram_per_gpu_gb": 10.0, "tensor_parallel_size": 1}
    big = {"est_vram_per_gpu_gb": 9.0, "tensor_parallel_size": 4}  # full ≈ 36 GB
    assert mst._build_accelerator_for(small) == "GPU_1xA10"
    assert mst._build_accelerator_for(big) == "GPU_1xH100"


def test_parse_and_validate_build_result():
    plan = _plan()
    out = _build_sentinel(plan)
    result = mst._parse_build_result(out)
    assert result["model_version"] == "7"
    mst._validate_build_result(plan, result)  # ok

    with pytest.raises(mst._ToolError):
        mst._parse_build_result("no sentinel here")
    with pytest.raises(mst._ToolError):  # smoke failed
        mst._validate_build_result(plan, mst._parse_build_result(_build_sentinel(plan, smoke=False)))
    with pytest.raises(mst._ToolError):  # hash mismatch
        mst._validate_build_result(plan, mst._parse_build_result(_build_sentinel(plan, hash_override="deadbeef")))


@pytest.mark.asyncio
async def test_build_and_register_happy_path(monkeypatch):
    plan = _plan()

    async def fake_jobs(session):
        return object()

    async def fake_submit(jobs_tool, **kwargs):
        assert kwargs["hardware_accelerator"] in ("GPU_1xA10", "GPU_1xH100")
        return _build_sentinel(plan, version="9")

    monkeypatch.setattr(mst, "_get_jobs_tool", fake_jobs)
    monkeypatch.setattr(mst, "_submit_gpu_build", fake_submit)
    res = await mst.model_serving_handler({"operation": "build_and_register", "plan": plan})
    assert res["isError"] is False
    assert "v9" in res["formatted"] and "PASSED" in res["formatted"]


@pytest.mark.asyncio
async def test_build_requires_source_and_registered():
    no_source = _plan(source_model=None)
    res = await mst.model_serving_handler({"operation": "build_and_register", "plan": no_source})
    assert res["isError"] is True and "source_model" in res["formatted"]


@pytest.mark.asyncio
async def test_build_quant_build_needs_calibration():
    plan = _plan()
    plan["serving_config"]["quant_source"] = "build"
    plan["plan_hash"] = mst._plan_hash(plan)  # re-hash after tampering so validation passes
    res = await mst.model_serving_handler({"operation": "build_and_register", "plan": plan})
    assert res["isError"] is True and "calibration_dataset" in res["formatted"]


# --- benchmark ---------------------------------------------------------------

def test_summarize_sweep_math():
    results = [{"ok": True, "total": 1.0, "tokens": 100},
               {"ok": True, "total": 2.0, "tokens": 100},
               {"ok": False, "total": 0.1, "code": 429}]
    row = mst._summarize_sweep(results, wall=2.0, level=3)
    assert row["C"] == 3 and row["ok"] == 2 and row["http429"] == 1
    assert row["sys_tps"] == 100.0  # 200 tokens / 2.0s wall
    assert row["mean_tok"] == 100


@pytest.mark.asyncio
async def test_benchmark_runs_sweep(monkeypatch):
    class _BenchWC:
        class _Api:
            def do(self, method, path, body=None):
                return {"choices": [{"message": {"content": "x"}}], "usage": {"completion_tokens": 50}}
        api_client = _Api()
    monkeypatch.setattr(mst, "_context", lambda s: (None, _BenchWC()))
    res = await mst.model_serving_handler(
        {"operation": "benchmark", "endpoint_name": "ep", "concurrency_levels": [1, 2]})
    assert res["isError"] is False and "sys_tps" in res["formatted"]


@pytest.mark.asyncio
async def test_unknown_operation_errors():
    res = await mst.model_serving_handler({"operation": "frobnicate"})
    assert res["isError"] is True


# --- ledger recording --------------------------------------------------------

def test_record_deployment_writes_ledger_row(tmp_path, monkeypatch):
    ledger = ExperimentLedger(local_path=tmp_path / "exp.jsonl")
    monkeypatch.setattr(mst, "_get_ledger", lambda s: ledger)
    plan = _plan()
    exp_id = mst._record_deployment(
        session=None, plan=plan, endpoint_name="ep", model_version="2",
        metric_name="tokens_per_second", metric_value=417.0,
        extra={"http429": 1},
    )
    rows = ledger.list_for_task(f"serve:{plan['registered_model_name']}")
    assert len(rows) == 1
    r = rows[0]
    assert r.experiment_id == exp_id and r.method == "custom_llm_serving"
    assert r.actual_metric == 417.0 and r.status == "done"
    assert r.config["endpoint"] == "ep" and r.artifacts["http429"] == 1
