"""``sweep`` builtin — fan out N training-job variants in parallel.

The auto-researcher's edge over a human is **breadth**: instead of iterating
serially (v1, v2, … vN — the slow path the Kaggle demo exhibited), this tool
launches every hypothesis as its own Databricks training job *at once*, scores
each, records the outcome to the experiment ledger, ranks them best-first, and
dedups configs already tried.

It is a thin agent-facing wrapper over the decoupled orchestrator in
``agent.core.sweep`` (``run_sweep_async``) and the Databricks Jobs adapter in
``agent.tools.sweep_jobs`` (``build_jobs_callables``). The agent supplies a
``script_template``; each variant's ``config`` is injected as a ``CONFIG`` dict
the script reads, and the script must print the metric as ``SWEEP_METRIC=<float>``
on its own stdout line. (Config is injected as a JSON literal, not via
``str.format`` — so scripts with brace literals / f-strings are safe and string
config values can't inject code.)

The ledger and jobs tool are built in module-level factories (``_get_ledger`` /
``_get_jobs_tool``) so tests can monkeypatch them onto a tmp JSONL ledger and
skip Databricks entirely — mirroring ``experiment_tool``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict

from agent.core import db_client
from agent.core.experiment_ledger import ExperimentLedger
from agent.core.sweep import SweepResult, run_sweep_async
from agent.tools.sweep_jobs import build_jobs_callables

logger = logging.getLogger(__name__)


SWEEP_TOOL_SPEC: Dict[str, Any] = {
    "name": "sweep",
    "description": (
        "Launch N training-job variants in PARALLEL on Databricks, score each, "
        "record every outcome to the experiment ledger, rank them best-first, "
        "and skip configs already tried for this task.\n\n"
        "This is the breadth play: instead of iterating one hypothesis at a "
        "time, you fan out all of them at once and keep the winner.\n\n"
        "How scoring works: each variant runs your `script_template`. That "
        "variant's `config` dict is injected as a module-level `CONFIG` "
        "variable your script reads (CONFIG['lr'], etc.) — do NOT use "
        "`{placeholder}` formatting. The script MUST print its metric as a line "
        "`SWEEP_METRIC=<float>` (on its own line) — that printed value is what "
        "the sweep ranks on. A script that never prints it is scored as failed.\n\n"
        "Example script_template:\n"
        "    import lightgbm as lgb\n"
        "    lr = CONFIG['lr']; n_est = CONFIG['n_estimators']\n"
        "    # ... train, compute auc ...\n"
        "    print(f'SWEEP_METRIC={auc}')\n\n"
        "Each hypothesis carries its own `config` dict (read via CONFIG). Set "
        "higher_is_better=false for loss-like metrics."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Eval-task or user-task id the sweep belongs to.",
            },
            "metric_name": {
                "type": "string",
                "description": "Metric name to rank on (e.g. 'roc_auc', 'eval_loss').",
            },
            "script_template": {
                "type": "string",
                "description": (
                    "Python training script. Reads the variant's config via a "
                    "module-level `CONFIG` dict (e.g. CONFIG['lr']) — NOT "
                    "`{placeholder}` formatting. MUST print `SWEEP_METRIC=<float>` "
                    "on its own line."
                ),
            },
            "hypotheses": {
                "type": "array",
                "description": (
                    "The variants to fan out. Each entry tests one config."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "hypothesis": {
                            "type": "string",
                            "description": "One-line testable claim.",
                        },
                        "method": {
                            "type": "string",
                            "description": "Technique name (e.g. 'lightgbm').",
                        },
                        "config": {
                            "type": "object",
                            "description": (
                                "Hyperparameters / feature set; fills the "
                                "script_template placeholders and is used for "
                                "dedup."
                            ),
                        },
                        "expected_metric": {
                            "type": "number",
                            "description": "Paper-reported or prior-best value.",
                        },
                        "source_paper": {
                            "type": "string",
                            "description": "arXiv id/url the hypothesis came from.",
                        },
                        "source_section": {
                            "type": "string",
                            "description": "Section of the source paper.",
                        },
                    },
                    "required": ["hypothesis", "method", "config"],
                },
            },
            "higher_is_better": {
                "type": "boolean",
                "description": (
                    "Whether a larger metric is better. Default true; set false "
                    "for loss-like metrics."
                ),
            },
            "budget_usd": {
                "type": "number",
                "description": (
                    "Optional dollar cap; stop scoring further variants once "
                    "accumulated cost meets it."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["serverless", "serverless_gpu", "script"],
                "description": "Job compute kind for each variant. Default 'serverless'.",
            },
            "timeout": {
                "type": "string",
                "description": "Per-job timeout (e.g. '30m'). Default '30m'.",
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional pip specs installed into each job.",
            },
        },
        "required": ["task_id", "metric_name", "script_template", "hypotheses"],
    },
}


def _load_default_config():
    from agent.config import load_config

    cfg_path = os.environ.get(
        "ML_INTERN_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json"),
    )
    return load_config(cfg_path)


def _get_ledger(session: Any) -> ExperimentLedger:
    """Build the ledger from the session's resolved settings.

    Factored out so tests can monkeypatch ``sweep_tool._get_ledger`` to a JSONL
    ledger on a tmp path (no workspace).
    """
    cfg = session.config if session and getattr(session, "config", None) else _load_default_config()
    settings = db_client.resolve_settings(cfg)
    token = getattr(session, "databricks_user_token", None) if session else None
    return ExperimentLedger(settings=settings, user_token=token)


async def _get_jobs_tool(session: Any):
    """Build a ``DatabricksJobsTool`` from the session, preferring OBO.

    Factored out so tests can monkeypatch ``sweep_tool._get_jobs_tool`` to
    return ``None`` and skip Databricks entirely.
    """
    from agent.tools.databricks_jobs_tool import DatabricksJobsTool

    cfg = session.config if session and getattr(session, "config", None) else _load_default_config()
    settings = db_client.resolve_settings(cfg)

    user_token = getattr(session, "databricks_user_token", None) if session else None
    user_email = getattr(session, "user_email", None) if session else None
    if user_token and settings.host:
        wc = db_client.get_workspace_client_for_user(user_token, settings.host)
    else:
        wc = db_client.get_workspace_client(settings)

    if not user_email:
        try:
            me = await asyncio.to_thread(wc.current_user.me)
            user_email = me.user_name or (me.emails[0].value if me.emails else None)
        except Exception:
            user_email = None

    return DatabricksJobsTool(wc=wc, settings=settings, user_email=user_email, session=session)


def _err(msg: str) -> Dict[str, Any]:
    return {"formatted": f"Error: {msg}", "isError": True}


def _missing(args: Dict[str, Any], required: list[str]) -> str | None:
    for field in required:
        if args.get(field) in (None, "", []):
            return field
    return None


def _render_script(script_template: str, config: dict) -> str:
    """Inject ``config`` as a module-level ``CONFIG`` dict the script reads.

    Uses a JSON literal (not ``str.format``) so scripts containing brace
    literals / f-strings don't raise KeyError and string config values can't
    inject Python code — ``repr`` of a JSON string is a safe string literal.
    """
    cfg_json = json.dumps(config or {})
    prelude = "import json as _json\nCONFIG = _json.loads(%r)\n" % cfg_json
    return prelude + script_template


def _validate_hypotheses(hypotheses: Any) -> str | None:
    if not isinstance(hypotheses, list) or not hypotheses:
        return "hypotheses must be a non-empty list."
    for i, h in enumerate(hypotheses):
        if not isinstance(h, dict):
            return f"hypotheses[{i}] must be an object."
        for f in ("hypothesis", "method", "config"):
            if h.get(f) in (None, ""):
                return f"hypotheses[{i}] is missing required field {f!r}."
        if not isinstance(h["config"], dict):
            return f"hypotheses[{i}].config must be an object."
    return None


def _summarize_config(config: dict | None, limit: int = 80) -> str:
    if not config:
        return "{}"
    parts = [f"{k}={v}" for k, v in sorted(config.items())]
    s = ", ".join(parts)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _format_result(result: SweepResult, metric_name: str) -> str:
    lines = [f"Sweep ranked {len(result.outcomes)} variant(s) best-first:"]
    for i, o in enumerate(result.outcomes, 1):
        metric = "—" if o.actual_metric is None else f"{o.actual_metric:g}"
        suffix = f"  ({o.error})" if o.error else ""
        lines.append(
            f"  {i}. {o.method}  [{_summarize_config(o.config)}]  "
            f"{metric_name}={metric}  {o.status}{suffix}"
        )

    if result.best is not None:
        b = result.best
        lines.append(
            f"\nBest: {b.method}  [{_summarize_config(b.config)}]  "
            f"{metric_name}={b.actual_metric:g}  (experiment {b.experiment_id})"
        )
    else:
        lines.append("\nBest: none — no variant scored successfully.")

    lines.append(
        f"\nTotals: submitted={result.n_submitted}, skipped={result.n_skipped}, "
        f"failed={result.n_failed}, total_cost_usd={result.total_cost_usd:g}"
    )
    return "\n".join(lines)


async def sweep_handler(arguments: Dict[str, Any], session: Any = None) -> Dict[str, Any]:
    field = _missing(arguments, ["task_id", "metric_name", "script_template", "hypotheses"])
    if field:
        return _err(f"{field} is required for sweep.")

    bad = _validate_hypotheses(arguments["hypotheses"])
    if bad:
        return _err(bad)

    task_id = arguments["task_id"]
    metric_name = arguments["metric_name"]
    script_template = arguments["script_template"]
    hypotheses = arguments["hypotheses"]
    higher_is_better = arguments.get("higher_is_better", True)
    budget_usd = arguments.get("budget_usd")
    kind = arguments.get("kind", "serverless")
    timeout = arguments.get("timeout", "30m")
    dependencies = arguments.get("dependencies")

    try:
        ledger = _get_ledger(session)
        jobs_tool = await _get_jobs_tool(session)

        def render_script(config: dict) -> str:
            return _render_script(script_template, config)

        base_args = {"timeout": timeout, "dependencies": dependencies}
        submit_fn, score_fn = build_jobs_callables(
            jobs_tool, render_script=render_script, base_args=base_args, kind=kind
        )

        result = await run_sweep_async(
            task_id=task_id,
            hypotheses=hypotheses,
            submit_fn=submit_fn,
            score_fn=score_fn,
            ledger=ledger,
            metric_name=metric_name,
            higher_is_better=higher_is_better,
            budget_usd=budget_usd,
            session_id=getattr(session, "session_id", None) if session else None,
        )
        return {"formatted": _format_result(result, metric_name), "isError": False}
    except Exception as e:
        logger.exception("sweep handler crashed (task_id=%s)", task_id)
        return _err(f"sweep failed: {e}")
