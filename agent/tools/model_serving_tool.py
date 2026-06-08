"""Custom LLM Serving tool — autonomous model deployment on Databricks.

Wires the deterministic ``serving_strategy`` math to real Databricks Custom LLM
Serving (Serverless Optimized Deployments, vLLM-backed, OpenAI-compatible
``/invocations``). The intern figures out *how* to host a model — GPU tier,
single-GPU vs tensor-parallel, precision, fixed provisioned concurrency — then
deploys, queries, and benchmarks it.

Operations (``operation`` enum):
  read-only : plan_deployment, query, list, probe_serving
  mutating  : build_and_register, deploy, delete, benchmark   (approval-gated)

The deterministic boundary is enforced by a *plan contract*: ``plan_deployment``
returns canonical plan objects each carrying a ``plan_hash``; ``build_and_register``
and ``deploy`` require that object and recompute the hash, so the agent can't
hand-assemble workload_type/precision/entrypoint at build or deploy time.

Endpoint creation goes through ``wc.api_client.do`` (centralizes OBO/SDK auth),
NOT the SDK's ``EndpointCoreConfigInput`` — entrypoint-based endpoints reject
autoscaling, so the body uses a fixed ``min==max`` provisioned concurrency
(multiple of 4) with ``scale_to_zero_enabled=false``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Callable

from agent.core import db_client
from agent.core import serving_strategy as ss
from agent.core.serving_strategy import ModelFacts, ServingConfig, feasible_configs, render_entrypoint
from agent.tools.databricks_jobs_tool import _JOBS_SUBMIT_PATH
from agent.tools.sweep_tool import _get_jobs_tool, _get_ledger, _load_default_config

logger = logging.getLogger(__name__)

# The MLflow artifact key the weights are logged under; Serving runs vLLM with
# --model pointed at this path, so the entrypoint and log_model must agree on it.
_ARTIFACT_KEY = "model"
_LOCAL_SMOKE_PORT = 3080  # serverless-GPU notebooks allow ports 3000-3999
_BENCH_PROMPT = "Write a detailed multi-paragraph explanation of how transformers work."

_BUILD_SENTINEL = "SERVING_BUILD_RESULT_B64="
_BUILD_RESULT_RE = re.compile(rf"^{re.escape(_BUILD_SENTINEL)}([A-Za-z0-9+/=]+)\s*$", re.MULTILINE)
# vLLM/transformers pins couple to the model architecture (FE: 4B→0.11.2/4.57.6,
# Qwen3.5→0.19.1/5.5.4). These are the conservative defaults; the agent overrides
# per arch via ``extra_pip_requirements`` in the plan.
_DEFAULT_PINS = ("mlflow>=3.12.0", "vllm==0.11.2", "transformers==4.57.6", "hf_transfer==0.1.9")

MODEL_SERVING_TOOL_SPEC: dict[str, Any] = {
    "name": "model_serving",
    "description": (
        "Deploy and operate models on Databricks Custom LLM Serving (vLLM, OpenAI-compatible). "
        "Figure out the deployment strategy from model size, workspace GPUs, and the team's "
        "accuracy/latency/cost priorities — then build, deploy, query, and benchmark.\n"
        "Operations:\n"
        "- plan_deployment: given model_facts + budgets, return the ranked feasible serving "
        "configs (GPU tier, tensor-parallel, precision, provisioned concurrency) with a plan_hash. "
        "Pick one and pass it to build_and_register/deploy.\n"
        "- build_and_register: run a serverless-GPU job that downloads weights, smoke-tests vLLM, "
        "and registers a Serverless Optimized Deployment to Unity Catalog. Requires a plan object.\n"
        "- deploy: create/update the serving endpoint (fixed concurrency, no autoscaling).\n"
        "- query: send a chat request to an endpoint's /invocations.\n"
        "- benchmark: concurrency sweep (throughput, TTFT, TPOT, 429s).\n"
        "- list / delete: manage endpoints. probe_serving: report available GPU serving tiers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["plan_deployment", "build_and_register", "deploy", "query",
                         "benchmark", "list", "delete", "probe_serving"],
            },
            "model_facts": {
                "type": "object",
                "description": "(plan_deployment) params_billions, num_layers, hidden_size, and "
                               "optionally kv_dim, num_attention_heads, num_kv_heads, "
                               "native_precision, available_quant_formats.",
            },
            "accuracy_budget": {"type": "string", "enum": ["max", "balanced", "aggressive"]},
            "objective": {"type": "string",
                          "enum": ["balanced", "cost_first", "accuracy_first", "latency_first"]},
            "available_workload_types": {
                "type": "array", "items": {"type": "string"},
                "description": "(plan_deployment) restrict to these workload_types; omit to use "
                               "the default set (A10/A10×4/H100; T4 excluded unless allow_small_gpu).",
            },
            "max_model_len": {"type": "integer"},
            "max_num_seqs": {"type": "integer"},
            "provisioned_concurrency": {"type": "integer", "description": "snapped to a multiple of 4"},
            "allow_small_gpu": {"type": "boolean"},
            "allow_quantization_build": {"type": "boolean"},
            "source_model": {"type": "string",
                             "description": "HF repo id, UC model URI, or /Volumes path to the weights."},
            "served_model_name": {"type": "string"},
            "registered_model_name": {"type": "string",
                                      "description": "three-level UC name <cat>.<schema>.<model>."},
            "extra_pip_requirements": {"type": "array", "items": {"type": "string"}},
            "plan": {"type": "object", "description": "a plan object returned by plan_deployment."},
            "endpoint_name": {"type": "string"},
            "model_version": {"type": "string"},
            "on_exists": {"type": "string", "enum": ["fail", "update"]},
            "wait": {"type": "boolean"},
            "messages": {"type": "array", "items": {"type": "object"},
                         "description": "(query) OpenAI-style chat messages."},
            "max_tokens": {"type": "integer"},
            "concurrency_levels": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["operation"],
    },
}


async def model_serving_handler(
    arguments: dict[str, Any], session: Any = None, tool_call_id: str | None = None
) -> dict[str, Any]:
    op = arguments.get("operation")
    try:
        if op == "plan_deployment":
            return _ok(_plan_deployment(arguments))
        if op == "deploy":
            return _ok(_deploy(arguments, session))
        if op == "query":
            return _ok(_query(arguments, session))
        if op == "list":
            return _ok(_list(session))
        if op == "delete":
            return _ok(_delete(arguments, session))
        if op == "probe_serving":
            return _ok(_probe_serving(session))
        if op == "build_and_register":
            return _ok(await _build_and_register(arguments, session))
        if op == "benchmark":
            return _ok(_benchmark(arguments, session))
        return _err(f"unknown operation {op!r}.")
    except _ToolError as e:
        return _err(str(e))
    except Exception as e:  # noqa: BLE001 — surface any backend failure to the agent
        logger.exception("model_serving %s failed", op)
        return _err(f"model_serving {op} failed: {e}")


# --- plan_deployment ---------------------------------------------------------

def _plan_deployment(args: dict[str, Any]) -> str:
    facts = args.get("model_facts")
    if not isinstance(facts, dict):
        raise _ToolError("model_facts (object) is required for plan_deployment.")
    try:
        model = ModelFacts(
            params_billions=float(facts["params_billions"]),
            num_layers=int(facts["num_layers"]),
            hidden_size=int(facts["hidden_size"]),
            kv_dim=facts.get("kv_dim"),
            num_attention_heads=facts.get("num_attention_heads"),
            num_kv_heads=facts.get("num_kv_heads"),
            native_precision=facts.get("native_precision", "fp16"),
            available_quant_formats=tuple(facts.get("available_quant_formats", ())),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise _ToolError(f"invalid model_facts: {e}")

    cfgs = feasible_configs(
        model=model,
        available_workload_types=args.get("available_workload_types"),
        accuracy_budget=args.get("accuracy_budget", "balanced"),
        objective=args.get("objective", "balanced"),
        max_model_len=int(args.get("max_model_len", 8192)),
        max_num_seqs=int(args.get("max_num_seqs", 64)),
        provisioned_concurrency=int(args.get("provisioned_concurrency", 4)),
        allow_small_gpu=bool(args.get("allow_small_gpu", False)),
        allow_quantization_build=bool(args.get("allow_quantization_build", False)),
    )
    if not cfgs:
        return ("No feasible serving config. The model may be too large for the available GPUs, "
                "the precision may need a pre-quantized artifact (set available_quant_formats or "
                "allow_quantization_build), or available_workload_types may be empty.")

    plans = [_make_plan(cfg, args, model) for cfg in cfgs]
    return _format_plans(plans)


def _make_plan(cfg: ServingConfig, args: dict[str, Any], model: ModelFacts) -> dict[str, Any]:
    """A canonical, hashable deployment plan derived from one ServingConfig."""
    served = args.get("served_model_name") or "served_model"
    # The served entrypoint points at the logged artifact dir, NOT the HF source.
    entrypoint = render_entrypoint(cfg, artifacts_path=_ARTIFACT_KEY, served_model_name=served)
    plan = {
        "serving_config": _config_to_dict(cfg),
        "model_facts": _facts_to_dict(model),
        "source_model": args.get("source_model"),
        "served_model_name": served,
        "registered_model_name": args.get("registered_model_name"),
        "extra_pip_requirements": list(args.get("extra_pip_requirements") or _DEFAULT_PINS),
        "entrypoint": entrypoint,
    }
    plan["plan_hash"] = _plan_hash(plan)
    return plan


def _plan_hash(plan: dict[str, Any]) -> str:
    """SHA256 over the canonical plan, excluding the hash field itself."""
    payload = {k: v for k, v in plan.items() if k != "plan_hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or "plan_hash" not in plan:
        raise _ToolError("a plan object from plan_deployment is required (with its plan_hash).")
    claimed = plan.get("plan_hash")
    if _plan_hash(plan) != claimed:
        raise _ToolError("plan_hash mismatch — the plan was modified after plan_deployment. "
                         "Re-run plan_deployment and pass the returned plan verbatim.")
    return plan


# --- deploy / query / list / delete (REST via api_client / SDK) --------------

def _build_deploy_body(
    *, endpoint_name: str, served_name: str, registered_model_name: str,
    model_version: str, workload_type: str, provisioned_concurrency: int,
) -> dict[str, Any]:
    """Custom LLM Serving create/update body — fixed concurrency, NO autoscaling."""
    pc = ss._snap_to_multiple_of_four(provisioned_concurrency)
    return {
        "name": endpoint_name,
        "config": {
            "served_entities": [{
                "name": served_name,
                "entity_name": registered_model_name,
                "entity_version": str(model_version),
                "workload_type": workload_type,
                "min_provisioned_concurrency": pc,
                "max_provisioned_concurrency": pc,
                "scale_to_zero_enabled": False,
            }]
        },
    }


def _deploy(args: dict[str, Any], session: Any) -> str:
    plan = _validate_plan(args.get("plan"))
    endpoint = args.get("endpoint_name")
    version = args.get("model_version")
    registered = plan.get("registered_model_name") or args.get("registered_model_name")
    if not (endpoint and version and registered):
        raise _ToolError("deploy requires endpoint_name, model_version, and a registered_model_name "
                         "(in the plan or args).")
    cfg = plan["serving_config"]
    body = _build_deploy_body(
        endpoint_name=endpoint, served_name=plan.get("served_model_name", "served_model"),
        registered_model_name=registered, model_version=version,
        workload_type=cfg["workload_type"], provisioned_concurrency=cfg["provisioned_concurrency"],
    )
    _settings, wc = _context(session)

    exists = _endpoint_exists(wc, endpoint)
    on_exists = args.get("on_exists", "fail")
    if exists and on_exists != "update":
        raise _ToolError(f"endpoint {endpoint!r} already exists. Pass on_exists='update' to replace "
                         f"its config, or choose a new endpoint_name.")
    if exists:
        wc.api_client.do("PUT", f"/api/2.0/serving-endpoints/{endpoint}/config", body=body["config"])
        action = "updated"
    else:
        wc.api_client.do("POST", "/api/2.0/serving-endpoints", body=body)
        action = "created"

    lines = [f"Endpoint {endpoint!r} {action} ({cfg['workload_type']}, TP={cfg['tensor_parallel_size']}, "
             f"{cfg['precision']}, provisioned_concurrency={cfg['provisioned_concurrency']}, no scale-to-zero)."]
    if args.get("wait"):
        lines.append(_poll_ready(wc, endpoint))
    else:
        lines.append("Not waiting for readiness (cold start = minutes). Use list to check state, "
                     "or the Serving UI for live logs (the logs API only serves the active config).")
    return "\n".join(lines)


def _query(args: dict[str, Any], session: Any) -> str:
    endpoint = args.get("endpoint_name")
    if not endpoint:
        raise _ToolError("query requires endpoint_name.")
    messages = args.get("messages") or [{"role": "user", "content": "Hello"}]
    body = {"messages": messages, "max_tokens": int(args.get("max_tokens", 256)), "temperature": 0}
    _settings, wc = _context(session)
    path = f"/serving-endpoints/{endpoint}/invocations"

    delay = 1.0
    for attempt in range(5):
        try:
            resp = wc.api_client.do("POST", path, body=body)
            choices = (resp or {}).get("choices") or []
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            return f"{endpoint} responded:\n{content}" if content else f"{endpoint} returned: {resp}"
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) and attempt < 4:
                # provisioned_concurrency is an admission ceiling; back off and retry.
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise _ToolError(f"{endpoint} kept returning HTTP 429 (admission ceiling) after retries.")


def _list(session: Any) -> str:
    _settings, wc = _context(session)
    eps = list(wc.serving_endpoints.list())
    if not eps:
        return "No serving endpoints in this workspace."
    out = []
    for e in eps:
        state = getattr(getattr(e, "state", None), "ready", None) or getattr(e, "state", "")
        out.append(f"- {getattr(e, 'name', '?')}  [{state}]")
    return "Serving endpoints:\n" + "\n".join(out)


def _delete(args: dict[str, Any], session: Any) -> str:
    endpoint = args.get("endpoint_name")
    if not endpoint:
        raise _ToolError("delete requires endpoint_name.")
    _settings, wc = _context(session)
    wc.serving_endpoints.delete(endpoint)
    return f"Deleted serving endpoint {endpoint!r}."


def _probe_serving(session: Any) -> str:
    """Confidence-rated capability report — there's no clean enumerate-tiers API."""
    _settings, wc = _context(session)
    observed: set[str] = set()
    try:
        for e in wc.serving_endpoints.list():
            for se in (getattr(getattr(e, "config", None), "served_entities", None) or []):
                wt = getattr(se, "workload_type", None)
                if wt:
                    observed.add(str(wt))
    except Exception as e:  # noqa: BLE001
        logger.debug("probe_serving list failed: %s", e)

    lines = ["Serving capability (best-effort — no enumerate API):"]
    for name, tier in ss.GPU_TIERS.items():
        src = "observed" if name in observed else "assumed"
        lines.append(f"- {name}: {tier.gpu_count}× {tier.gpu} ({tier.vram_gb}GB) [{src}]")
    lines.append("H100 (GPU_XLARGE) serving enrollment is 'unknown' until a deploy succeeds.")
    return "\n".join(lines)


# --- ledger ------------------------------------------------------------------

def _record_deployment(
    session: Any, *, plan: dict[str, Any], endpoint_name: str, model_version: str,
    metric_name: str, metric_value: float, mlflow_run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Record a deployment trial as an ExperimentRow (config → latency/$/quality)."""
    ledger = _get_ledger(session)
    cfg = plan["serving_config"]
    task_id = f"serve:{plan.get('registered_model_name') or endpoint_name}"
    hypo = (f"Serve as {cfg['precision']} on {cfg['workload_type']} "
            f"(TP={cfg['tensor_parallel_size']}, PC={cfg['provisioned_concurrency']}).")
    exp_id = ledger.propose(
        task_id=task_id, hypothesis=hypo, method="custom_llm_serving",
        config={"endpoint": endpoint_name, "model_version": model_version, **cfg},
        metric_name=metric_name,
        session_id=getattr(session, "session_id", None) if session else None,
    )
    if mlflow_run_id:
        ledger.mark_running(exp_id, mlflow_run_id=mlflow_run_id)
    ledger.record_result(
        exp_id, actual_metric=float(metric_value),
        artifacts={"plan": plan, "endpoint": endpoint_name, **(extra or {})},
    )
    return exp_id


# --- build_and_register (serverless-GPU job) ---------------------------------

async def _build_and_register(args: dict[str, Any], session: Any) -> str:
    plan = _validate_plan(args.get("plan"))
    source = plan.get("source_model")
    registered = plan.get("registered_model_name") or args.get("registered_model_name")
    if not source or source == "<source_model>":
        raise _ToolError("plan.source_model (HF repo id, UC URI, or /Volumes path) is required.")
    if not registered:
        raise _ToolError("a three-level registered_model_name (<cat>.<schema>.<model>) is required.")
    cfg = plan["serving_config"]
    if cfg["quant_source"] == "build" and not args.get("calibration_dataset"):
        # AWQ/GPTQ/int8 quantization needs calibration data — fail closed rather
        # than silently produce a low-quality artifact.
        raise _ToolError("serving_config.quant_source='build' requires a calibration_dataset "
                         "(AWQ/GPTQ/int8 calibration). Provide one or pick a native/online precision.")

    script = _render_build_script(plan, registered)
    jobs_tool = await _get_jobs_tool(session)
    accelerator = args.get("build_accelerator") or _build_accelerator_for(cfg)
    output = await _submit_gpu_build(
        jobs_tool, script=script, hardware_accelerator=accelerator,
        dependencies=plan.get("extra_pip_requirements"), timeout=args.get("timeout", "2h"),
    )
    result = _parse_build_result(output)
    _validate_build_result(plan, result)

    topology = "exact (TP=1)" if cfg["tensor_parallel_size"] == 1 else \
        f"partial — smoked at TP=1 on the build GPU, serves at TP={cfg['tensor_parallel_size']}"
    lines = [
        f"Built + registered {result['registered_model_name']} v{result['model_version']}.",
        f"  precision={cfg['precision']}/{cfg['quant_source']}, build GPU={accelerator}, "
        f"vLLM smoke {'PASSED' if result.get('smoke_passed') else 'FAILED'} ({topology}).",
        f"  pins: {', '.join(plan.get('extra_pip_requirements', []))}",
        f"  plan_hash {plan['plan_hash'][:12]} (verified). Deploy with operation=deploy, "
        f"endpoint_name=<name>, model_version={result['model_version']}, the same plan.",
    ]
    if cfg["tensor_parallel_size"] > 1:
        lines.append("  NOTE: TP>1 serving topology was not smoke-tested on the build box — "
                     "watch the first deploy's logs and delete on crash-loop.")
    return "\n".join(lines)


def _build_accelerator_for(cfg: dict[str, Any]) -> str:
    """Pick a serverless-GPU build accelerator big enough to load the model once.

    The build only needs to load weights + run a TP=1 smoke; serving topology
    (TP) lives in the entrypoint metadata, not the build box. Approximate the
    full-model footprint from the per-GPU estimate × serving TP.
    """
    full_gb = cfg.get("est_vram_per_gpu_gb", 0) * cfg.get("tensor_parallel_size", 1)
    return "GPU_1xA10" if full_gb < 22 else "GPU_1xH100"


async def _submit_gpu_build(
    jobs_tool: Any, *, script: str, hardware_accelerator: str,
    dependencies: list[str] | None, timeout: str,
) -> str:
    """Stage + run the build notebook on serverless GPU; return its stdout.

    Reuses DatabricksJobsTool's async building blocks (same path as sweep_jobs)
    so env filtering / serverless task shape / retry-aware output picking aren't
    duplicated.
    """
    job_args: dict[str, Any] = {
        "kind": "serverless_gpu",
        "script": script,
        "filename": f"serving_build_{uuid.uuid4().hex[:12]}.py",
        "hardware_accelerator": hardware_accelerator,
        "timeout": timeout,
    }
    if dependencies:
        job_args["dependencies"] = dependencies
    ws_path = await jobs_tool._resolve_or_stage_script(job_args, as_notebook=True)
    body = await jobs_tool._build_submit_body(job_args, ws_path, "serverless_gpu")
    resp = await asyncio.to_thread(
        jobs_tool.wc.api_client.do, "POST", _JOBS_SUBMIT_PATH, body=body
    )
    run_id = resp.get("run_id")
    if not run_id:
        raise _ToolError(f"runs/submit returned no run_id: {resp}")
    run = await jobs_tool._wait_for_run(run_id)
    state = run.get("state") or {}
    if state.get("result_state") != "SUCCESS":
        raise _ToolError(
            f"build run {run_id} ended "
            f"{state.get('result_state') or state.get('life_cycle_state')!r}: "
            f"{state.get('state_message') or '—'}. Read the Serving/Jobs UI logs."
        )
    return await jobs_tool._fetch_run_output(run)


def _parse_build_result(output: str) -> dict[str, Any]:
    m = _BUILD_RESULT_RE.search(output or "")
    if not m:
        raise _ToolError(f"no '{_BUILD_SENTINEL}<b64>' sentinel in build output "
                         f"({len(output or '')} chars) — the build likely failed before registering.")
    try:
        return json.loads(base64.b64decode(m.group(1)).decode())
    except Exception as e:  # noqa: BLE001
        raise _ToolError(f"could not decode build result sentinel: {e}")


def _validate_build_result(plan: dict[str, Any], result: dict[str, Any]) -> None:
    if result.get("plan_hash") != plan["plan_hash"]:
        raise _ToolError("build result plan_hash does not match the plan — refusing to trust it.")
    if not result.get("smoke_passed"):
        raise _ToolError("the local vLLM smoke test did not pass — not registering for deploy.")
    if not result.get("registered_model_name") or not result.get("model_version"):
        raise _ToolError(f"build result missing registered_model_name/model_version: {result}")


def _render_build_script(plan: dict[str, Any], registered: str) -> str:
    """Render the serverless-GPU build notebook source.

    Downloads weights → (optional quantize) → validates the artifact → runs a
    LOCAL vLLM smoke test (TP=1) → logs a placeholder ChatModel whose metadata
    carries the real serving entrypoint → registers a Serverless Optimized
    Deployment to UC → prints a base64 result sentinel. Injected values are
    JSON-encoded so entrypoint strings (which contain quotes) embed safely.
    """
    cfg = plan["serving_config"]
    served = plan.get("served_model_name", "served_model")
    smoke_cfg = replace(_reconstruct_cfg(cfg), tensor_parallel_size=1)
    smoke_ep = render_entrypoint(smoke_cfg, artifacts_path=_ARTIFACT_KEY,
                                 served_model_name=served, serving_port=_LOCAL_SMOKE_PORT)
    pins = plan.get("extra_pip_requirements") or list(_DEFAULT_PINS)
    return f'''# Databricks notebook source
# Auto-generated by model_serving.build_and_register — Custom LLM Serving (SOD) build.
import base64, hashlib, json, os, subprocess, tempfile, time
import requests

SOURCE_MODEL = {json.dumps(plan["source_model"])}
ARTIFACTS_PATH = {json.dumps(_ARTIFACT_KEY)}
SERVED_NAME = {json.dumps(served)}
REGISTERED_NAME = {json.dumps(registered)}
SERVING_ENTRYPOINT = {json.dumps(plan["entrypoint"])}
SMOKE_ENTRYPOINT = {json.dumps(smoke_ep)}
PLAN_HASH = {json.dumps(plan["plan_hash"])}
PINS = {json.dumps(pins)}
LOCAL_PORT = {_LOCAL_SMOKE_PORT}

os.chdir(tempfile.mkdtemp())  # /Workspace can't hold large weights

# 1) Materialize weights at ARTIFACTS_PATH (HF id -> download; local/Volume -> reuse).
if os.path.isdir(SOURCE_MODEL):
    if os.path.abspath(SOURCE_MODEL) != os.path.abspath(ARTIFACTS_PATH):
        os.symlink(SOURCE_MODEL, ARTIFACTS_PATH)
else:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=SOURCE_MODEL, local_dir=ARTIFACTS_PATH)

# 2) Validate the artifact + manifest hash.
assert os.path.exists(os.path.join(ARTIFACTS_PATH, "config.json")), "no config.json in artifact"
names = sorted(os.listdir(ARTIFACTS_PATH))
manifest = hashlib.sha256(
    "".join(f"{{n}}:{{os.path.getsize(os.path.join(ARTIFACTS_PATH, n))}}" for n in names).encode()
).hexdigest()

# 3) Local vLLM smoke test (TP=1) against the rendered smoke entrypoint.
log = open("vllm.log", "w")
proc = subprocess.Popen(["bash", "-lc", SMOKE_ENTRYPOINT], stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
smoke_passed = False
try:
    deadline = time.time() + 1500
    ready = False
    while time.time() < deadline:
        if os.path.exists("vllm.log") and "Application startup complete" in open("vllm.log").read():
            ready = True
            break
        if proc.poll() is not None:
            break
        time.sleep(5)
    if ready:
        r = requests.post(f"http://localhost:{{LOCAL_PORT}}/invocations",
                          json={{"messages": [{{"role": "user", "content": "Hello"}}], "max_tokens": 16}},
                          timeout=120)
        smoke_passed = r.status_code == 200 and bool(
            (r.json().get("choices") or [{{}}])[0].get("message", {{}}).get("content")
        )
finally:
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"])
assert smoke_passed, "vLLM local smoke test failed — see vllm.log"

# 4) Log placeholder ChatModel (Serving runs the entrypoint, not predict) + register SOD.
import mlflow
from mlflow.pyfunc.model import ChatModel, ChatCompletionResponse

class LLMModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({{"choices": []}})

info = mlflow.pyfunc.log_model(
    name=SERVED_NAME, python_model=LLMModel(), artifacts={{"model_dir": ARTIFACTS_PATH}},
    metadata={{"task": "llm/v1/chat", "entrypoint": SERVING_ENTRYPOINT,
              "plan_hash": PLAN_HASH, "artifact_manifest_sha256": manifest}},
    extra_pip_requirements=PINS,
)
mlflow.set_registry_uri("databricks-uc")
mv = mlflow.register_model(info.model_uri, REGISTERED_NAME, env_pack="databricks_model_serving")

import vllm, transformers
result = {{
    "registered_model_name": REGISTERED_NAME, "model_version": str(mv.version),
    "mlflow_run_id": info.run_id, "plan_hash": PLAN_HASH,
    "artifact_manifest_sha256": manifest, "smoke_passed": bool(smoke_passed),
    "vllm_version": vllm.__version__, "transformers_version": transformers.__version__,
}}
print("{_BUILD_SENTINEL}" + base64.b64encode(json.dumps(result).encode()).decode())
'''


def _reconstruct_cfg(d: dict[str, Any]) -> ServingConfig:
    return ServingConfig(
        workload_type=d["workload_type"], tensor_parallel_size=d["tensor_parallel_size"],
        precision=d["precision"], quant_source=d["quant_source"], max_model_len=d["max_model_len"],
        gpu_memory_utilization=d["gpu_memory_utilization"], max_num_seqs=d["max_num_seqs"],
        provisioned_concurrency=d["provisioned_concurrency"], est_vram_per_gpu_gb=d["est_vram_per_gpu_gb"],
        vram_headroom_gb=d["vram_headroom_gb"], est_max_concurrent_seqs=d["est_max_concurrent_seqs"],
        quality_delta_pct=d["quality_delta_pct"], est_cost_per_hr_usd=d["est_cost_per_hr_usd"],
        fits=True, notes=tuple(d.get("notes", [])),
    )


# --- benchmark (concurrency sweep) -------------------------------------------

def _benchmark(args: dict[str, Any], session: Any) -> str:
    endpoint = args.get("endpoint_name")
    if not endpoint:
        raise _ToolError("benchmark requires endpoint_name.")
    levels = [int(c) for c in (args.get("concurrency_levels") or [1, 4, 8, 16])]
    max_tokens = int(args.get("max_tokens", 128))
    _settings, wc = _context(session)
    one = _make_request_fn(wc, endpoint, max_tokens)

    rows = []
    for c in levels:
        results, wall = _run_level(one, c)
        rows.append(_summarize_sweep(results, wall, c))

    primary = max((r["sys_tps"] or 0.0 for r in rows), default=0.0)
    plan = args.get("plan")
    note = ""
    if isinstance(plan, dict) and "serving_config" in plan:
        try:
            exp_id = _record_deployment(
                session, plan=plan, endpoint_name=endpoint,
                model_version=str(args.get("model_version", "")),
                metric_name="tokens_per_second", metric_value=primary,
                extra={"benchmark": rows},
            )
            note = f"\nRecorded to ledger as {exp_id}."
        except Exception as e:  # noqa: BLE001 — benchmarking still succeeds if ledger is down
            logger.debug("benchmark ledger record failed: %s", e)

    header = f"Benchmark of {endpoint!r} (peak {primary:.0f} tok/s):"
    table = ["  C   ok  429   sys_tps  p50_lat  p95_lat  mean_tok"]
    for r in rows:
        table.append(f"  {r['C']:<3} {r['ok']:<3} {r['http429']:<4} "
                     f"{(r['sys_tps'] or 0):>8.1f}  {_s(r['p50_lat'])}  {_s(r['p95_lat'])}  "
                     f"{r['mean_tok'] or 0:>8}")
    table.append("(non-streaming: TTFT/TPOT need a streaming client — future.)")
    return header + "\n" + "\n".join(table) + note


def _make_request_fn(wc: Any, endpoint: str, max_tokens: int) -> Callable[[], dict]:
    path = f"/serving-endpoints/{endpoint}/invocations"
    body = {"messages": [{"role": "user", "content": _BENCH_PROMPT}],
            "max_tokens": max_tokens, "temperature": 0}

    def one() -> dict:
        t0 = time.perf_counter()
        try:
            resp = wc.api_client.do("POST", path, body=dict(body))
            tokens = ((resp or {}).get("usage") or {}).get("completion_tokens", 0)
            return {"ok": True, "total": time.perf_counter() - t0, "tokens": tokens}
        except Exception as e:  # noqa: BLE001
            code = 429 if "429" in str(e) else "ERR"
            return {"ok": False, "total": time.perf_counter() - t0, "code": code}

    return one


def _run_level(request_fn: Callable[[], dict], c: int) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex:
        results = list(ex.map(lambda _: request_fn(), range(c)))
    return results, time.perf_counter() - t0


def _summarize_sweep(results: list[dict], wall: float, level: int) -> dict[str, Any]:
    oks = [r for r in results if r.get("ok")]
    lats = sorted(r["total"] for r in oks)
    tot_tokens = sum(r.get("tokens", 0) for r in oks)
    n429 = sum(1 for r in results if r.get("code") == 429)

    def pct(a: list[float], q: float):
        return round(a[min(len(a) - 1, int(q * (len(a) - 1)))], 2) if a else None

    return {
        "C": level, "ok": len(oks), "http429": n429,
        "sys_tps": round(tot_tokens / wall, 1) if wall > 0 else None,
        "p50_lat": pct(lats, 0.5), "p95_lat": pct(lats, 0.95),
        "mean_tok": round(tot_tokens / len(oks)) if oks else 0,
    }


def _s(v) -> str:
    return f"{v:>7.2f}" if v is not None else "      -"


# --- helpers -----------------------------------------------------------------

class _ToolError(Exception):
    """A user-facing validation/precondition error (returned as isError, not a crash)."""


def _context(session: Any):
    cfg = session.config if session and getattr(session, "config", None) else _load_default_config()
    settings = db_client.resolve_settings(cfg)
    token = getattr(session, "databricks_user_token", None) if session else None
    wc = (db_client.get_workspace_client_for_user(token, settings.host)
          if token else db_client.get_workspace_client(settings))
    return settings, wc


def _endpoint_exists(wc: Any, name: str) -> bool:
    try:
        wc.serving_endpoints.get(name)
        return True
    except Exception:  # noqa: BLE001 — get raises when not found
        return False


def _poll_ready(wc: Any, name: str, timeout_s: int = 1800, interval_s: int = 20) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            ep = wc.serving_endpoints.get(name)
            ready = getattr(getattr(ep, "state", None), "ready", None)
            if ready and str(ready).upper().endswith("READY"):
                return f"Endpoint {name!r} is READY."
            cfg_update = getattr(getattr(ep, "state", None), "config_update", None)
            if cfg_update and "FAILED" in str(cfg_update).upper():
                return (f"Endpoint {name!r} config update FAILED — read the Serving UI logs "
                        f"(the logs API only serves the active config, so a failed deploy loses them).")
        except Exception as e:  # noqa: BLE001
            logger.debug("poll %s: %s", name, e)
        time.sleep(interval_s)
    return f"Endpoint {name!r} not READY within {timeout_s}s — check the Serving UI."


def _config_to_dict(cfg: ServingConfig) -> dict[str, Any]:
    return {
        "workload_type": cfg.workload_type, "tensor_parallel_size": cfg.tensor_parallel_size,
        "precision": cfg.precision, "quant_source": cfg.quant_source,
        "max_model_len": cfg.max_model_len, "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "max_num_seqs": cfg.max_num_seqs, "provisioned_concurrency": cfg.provisioned_concurrency,
        "est_vram_per_gpu_gb": cfg.est_vram_per_gpu_gb, "vram_headroom_gb": cfg.vram_headroom_gb,
        "est_max_concurrent_seqs": cfg.est_max_concurrent_seqs,
        "quality_delta_pct": cfg.quality_delta_pct, "est_cost_per_hr_usd": cfg.est_cost_per_hr_usd,
        "notes": list(cfg.notes),
    }


def _facts_to_dict(m: ModelFacts) -> dict[str, Any]:
    return {
        "params_billions": m.params_billions, "num_layers": m.num_layers,
        "hidden_size": m.hidden_size, "kv_dim": m.kv_dim,
        "num_attention_heads": m.num_attention_heads, "num_kv_heads": m.num_kv_heads,
        "native_precision": m.native_precision,
        "available_quant_formats": list(m.available_quant_formats),
    }


def _format_plans(plans: list[dict[str, Any]]) -> str:
    lines = [f"{len(plans)} feasible serving config(s), best first:"]
    for i, p in enumerate(plans, 1):
        c = p["serving_config"]
        lines.append(
            f"  {i}. {c['workload_type']} TP={c['tensor_parallel_size']} {c['precision']}"
            f"/{c['quant_source']} — ${c['est_cost_per_hr_usd']}/hr, "
            f"~{c['est_max_concurrent_seqs']} concurrent seqs, Δacc≈{c['quality_delta_pct']}%, "
            f"PC={c['provisioned_concurrency']}  (plan_hash {p['plan_hash'][:12]})"
        )
        for n in c["notes"]:
            lines.append(f"       · {n}")
    lines.append("\nPick one and pass its full plan object to build_and_register (to produce the "
                 "UC model version) then deploy. Plans are JSON below.\n")
    lines.append(json.dumps(plans, separators=(",", ":")))
    return "\n".join(lines)


def _ok(formatted: str) -> dict[str, Any]:
    return {"formatted": formatted, "isError": False}


def _err(msg: str) -> dict[str, Any]:
    return {"formatted": f"Error: {msg}", "isError": True}
