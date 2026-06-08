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

import hashlib
import json
import logging
import time
from typing import Any

from agent.core import db_client
from agent.core import serving_strategy as ss
from agent.core.serving_strategy import ModelFacts, ServingConfig, feasible_configs, render_entrypoint
from agent.tools.sweep_tool import _get_ledger, _load_default_config

logger = logging.getLogger(__name__)

_BUILD_SENTINEL = "SERVING_BUILD_RESULT_B64="
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
        if op in ("build_and_register", "benchmark"):
            # Increment 2 — the serverless-GPU build script + live load test land
            # with a workspace to validate against (see plan.md Phase 7 build order).
            return _err(f"{op} is not implemented yet (Phase 7 build-step 4). "
                        f"plan_deployment + deploy + query are available now.")
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
    artifacts_path = args.get("source_model") or "<source_model>"
    entrypoint = render_entrypoint(cfg, artifacts_path=artifacts_path, served_model_name=served)
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
