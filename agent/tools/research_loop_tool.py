"""``research_loop`` builtin — autonomous, deterministic closure.

The agent hands over a *pool* of hypotheses + a script template + stop budget,
and the runner (``agent.core.research_loop``) drives the whole loop itself:
each round it sweeps the next not-yet-tried batch in parallel, records to the
ledger, reproduce-gates the best, accepts it only if it improves, and stops on
budget / target / patience / max-rounds / pool-exhaustion. The LLM is out of the
control flow — it only supplied the pool.

Reuses ``sweep_tool``'s ledger / jobs-tool factories, config renderer, and
hypothesis validator so behavior matches the ``sweep`` tool exactly. Approval-
gated (it launches Databricks training jobs).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from agent.core.research_loop import LoopResult, run_research_loop
from agent.tools.sweep_jobs import build_jobs_callables
from agent.tools.sweep_tool import (
    _get_jobs_tool,
    _get_ledger,
    _render_script,
    _summarize_config,
    _validate_hypotheses,
)

logger = logging.getLogger(__name__)


RESEARCH_LOOP_TOOL_SPEC: Dict[str, Any] = {
    "name": "research_loop",
    "description": (
        "Run the auto-researcher loop autonomously over a POOL of hypotheses. "
        "Each round the runner sweeps the next untried batch in parallel, scores "
        "and records each, reproduce-gates the best, accepts it only if it beats "
        "the running best, and stops on its own (budget / target / patience / "
        "max_rounds / pool exhausted). You supply the candidates and the budget; "
        "the loop owns the control flow.\n\n"
        "Same scoring contract as `sweep`: each variant runs `script_template` "
        "with its `config` injected as a `CONFIG` dict (read CONFIG['lr'], NOT "
        "`{placeholders}`), and the script MUST print `SWEEP_METRIC=<float>`.\n\n"
        "Use this once you have a batch of ideas to burn down hands-off. Use "
        "`sweep` for a single fan-out, `experiment` for one-off runs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task id the loop belongs to."},
            "metric_name": {"type": "string", "description": "Metric to rank on."},
            "script_template": {
                "type": "string",
                "description": (
                    "Python script reading the variant config via CONFIG['x']; "
                    "MUST print SWEEP_METRIC=<float> on its own line."
                ),
            },
            "hypotheses": {
                "type": "array",
                "description": "The POOL of candidates to burn down across rounds.",
                "items": {
                    "type": "object",
                    "properties": {
                        "hypothesis": {"type": "string"},
                        "method": {"type": "string"},
                        "config": {"type": "object"},
                        "expected_metric": {"type": "number"},
                        "source_paper": {"type": "string"},
                        "source_section": {"type": "string"},
                    },
                    "required": ["hypothesis", "method", "config"],
                },
            },
            "higher_is_better": {"type": "boolean", "description": "Default true."},
            "current_best": {"type": "number", "description": "Known best to beat (optional)."},
            "budget_usd": {"type": "number", "description": "Cumulative dollar cap (optional)."},
            "est_cost_per_variant": {
                "type": "number",
                "description": (
                    "Estimated $/variant; lets the loop clamp each batch to the "
                    "remaining budget BEFORE launching jobs. Recommended when "
                    "budget_usd is set (real Jobs cost is unknown at submit time)."
                ),
            },
            "max_rounds": {"type": "integer", "description": "Max rounds. Default 5."},
            "batch_size": {"type": "integer", "description": "Hypotheses per round. Default 4."},
            "patience": {"type": "integer", "description": "Stop after N rounds w/o improvement. Default 2."},
            "min_delta": {"type": "number", "description": "Min metric gain to count as improvement. Default 0."},
            "target_metric": {"type": "number", "description": "Stop early once reached (optional)."},
            "kind": {
                "type": "string",
                "enum": ["serverless", "serverless_gpu", "script"],
                "description": "Job compute kind. Default 'serverless'.",
            },
            "timeout": {"type": "string", "description": "Per-job timeout. Default '30m'."},
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional pip specs per job.",
            },
        },
        "required": ["task_id", "metric_name", "script_template", "hypotheses"],
    },
}


def _missing(args: Dict[str, Any], required: list[str]) -> str | None:
    for field in required:
        if args.get(field) in (None, "", []):
            return field
    return None


def _pool_source(pool: list[dict], batch_size: int):
    """A hypothesis source that yields the next ``batch_size`` slice each round.

    The runner's own seen-set dedups across rounds, so a plain forward cursor is
    enough; an empty slice signals exhaustion.
    """
    state = {"i": 0}

    def source(ctx) -> list[dict]:
        i = state["i"]
        state["i"] = i + batch_size
        return pool[i:i + batch_size]

    return source


def _format_result(result: LoopResult, metric_name: str) -> str:
    lines = [
        f"Research loop stopped: {result.stop_reason}. "
        f"{len(result.rounds)} round(s), {result.accepted_count} accepted, "
        f"total_cost_usd={result.total_cost_usd:g}."
    ]
    for r in result.rounds:
        m = "—" if r.best_metric is None else f"{r.best_metric:g}"
        flag = "✓accepted" if r.accepted else "—"
        lines.append(
            f"  round {r.round_idx}: {r.n_hypotheses} variant(s), "
            f"{metric_name}={m}  gate={r.gate_severity}  {flag}  "
            f"cost={r.cost_usd:g}"
        )
    if result.best_metric is not None:
        lines.append(
            f"\nBest: {metric_name}={result.best_metric:g} "
            f"(experiment {result.best_experiment_id})"
        )
    else:
        lines.append("\nBest: none accepted.")
    return "\n".join(lines)


async def research_loop_handler(arguments: Dict[str, Any], session: Any = None) -> Dict[str, Any]:
    field = _missing(arguments, ["task_id", "metric_name", "script_template", "hypotheses"])
    if field:
        return {"formatted": f"Error: {field} is required for research_loop.", "isError": True}

    bad = _validate_hypotheses(arguments["hypotheses"])
    if bad:
        return {"formatted": f"Error: {bad}", "isError": True}

    script_template = arguments["script_template"]
    kind = arguments.get("kind", "serverless")
    base_args = {"timeout": arguments.get("timeout", "30m"), "dependencies": arguments.get("dependencies")}

    try:
        ledger = _get_ledger(session)
        jobs_tool = await _get_jobs_tool(session)

        def render_script(config: dict) -> str:
            return _render_script(script_template, config)

        submit_fn, score_fn = build_jobs_callables(
            jobs_tool, render_script=render_script, base_args=base_args, kind=kind
        )
        source = _pool_source(arguments["hypotheses"], int(arguments.get("batch_size", 4)))

        result = await run_research_loop(
            task_id=arguments["task_id"],
            hypothesis_source=source,
            submit_fn=submit_fn,
            score_fn=score_fn,
            ledger=ledger,
            metric_name=arguments["metric_name"],
            higher_is_better=arguments.get("higher_is_better", True),
            current_best=arguments.get("current_best"),
            budget_usd=arguments.get("budget_usd"),
            est_cost_per_variant=arguments.get("est_cost_per_variant"),
            max_rounds=int(arguments.get("max_rounds", 5)),
            patience=int(arguments.get("patience", 2)),
            min_delta=arguments.get("min_delta", 0.0),
            target_metric=arguments.get("target_metric"),
            session_id=getattr(session, "session_id", None) if session else None,
        )
        return {"formatted": _format_result(result, arguments["metric_name"]), "isError": False}
    except Exception as e:  # noqa: BLE001 — boundary
        logger.exception("research_loop handler crashed (task_id=%s)", arguments.get("task_id"))
        return {"formatted": f"Error: research_loop failed: {e}", "isError": True}
