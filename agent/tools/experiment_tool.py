"""``experiment`` builtin — the agent's read/write surface over the ledger.

The experiment ledger (``agent.core.experiment_ledger``) is the persistence
spine of the auto-researcher loop. This tool exposes pure ledger CRUD/query so
the agent reasons over its own experiment history within and across sessions:

- ``propose`` a hypothesis BEFORE submitting a training job (records the claim,
  method, config, and expected metric so the run is attributable later).
- ``record`` the outcome AFTER the job finishes (actual metric, cost, wall
  clock) — the ledger computes the reproduction gap.
- ``best`` / ``list`` to reason over prior runs for a task.
- ``find_similar`` to avoid re-running a config already tried (dedup).

There is no gate logic here — escalation/repro-gap decisions live elsewhere.

Backend selection (SQL vs JSONL) is decided by ``ExperimentLedger`` from the
resolved settings, mirroring ``uc_dataset_tools``. The ledger is built in a
module-level factory so tests can monkeypatch it onto a tmp JSONL path.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.core import db_client
from agent.core.experiment_ledger import (
    ExperimentLedger,
    ExperimentRow,
    metric_higher_is_better,
)
from agent.core.repro_gate import gate_from_row

logger = logging.getLogger(__name__)


EXPERIMENT_TOOL_SPEC: Dict[str, Any] = {
    "name": "experiment",
    "description": (
        "Read and write the experiment ledger — your durable record of every "
        "experiment proposed, run, and scored, within and across sessions. "
        "Use it to reason over your own history instead of repeating work.\n\n"
        "Workflow:\n"
        "- propose: BEFORE submitting a training job, record the hypothesis "
        "(one-line testable claim), method, config (hyperparams/feature set), "
        "metric_name, and optional expected_metric (paper-reported or "
        "prior-best). Returns an experiment_id.\n"
        "- record: AFTER the job finishes, record the outcome for that "
        "experiment_id (actual_metric, optional cost_usd / wall_clock_s / "
        "notes). The ledger computes the reproduction gap (expected − actual).\n"
        "- list: list all experiments for a task_id to review what you tried.\n"
        "- best: the best-scoring experiment for a task_id + metric_name "
        "(set higher_is_better=false for loss-like metrics).\n"
        "- find_similar: BEFORE proposing, check whether a (task_id, method, "
        "config) was already tried — skip re-running an identical config.\n\n"
        "Always propose before a run and record after, so best/list/find_similar "
        "stay accurate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["propose", "record", "list", "best", "find_similar"],
                "description": "The ledger operation to perform.",
            },
            "task_id": {
                "type": "string",
                "description": (
                    "(propose, list, best, find_similar) Eval-task or user-task "
                    "id the experiment belongs to."
                ),
            },
            "hypothesis": {
                "type": "string",
                "description": "(propose) One-line testable claim.",
            },
            "method": {
                "type": "string",
                "description": (
                    "(propose, find_similar) Technique name "
                    "(e.g. 'lightgbm', 'pseudo_label')."
                ),
            },
            "config": {
                "type": "object",
                "description": (
                    "(propose, find_similar) JSON of hyperparameters / feature "
                    "set. Used for dedup matching in find_similar."
                ),
            },
            "metric_name": {
                "type": "string",
                "description": (
                    "(propose, best) Metric name (e.g. 'roc_auc', 'accuracy', "
                    "'eval_loss')."
                ),
            },
            "expected_metric": {
                "type": "number",
                "description": (
                    "(propose) Paper-reported or prior-best value; used to "
                    "compute the reproduction gap on record."
                ),
            },
            "source_paper": {
                "type": "string",
                "description": "(propose) arXiv id/url the hypothesis came from.",
            },
            "source_section": {
                "type": "string",
                "description": "(propose) Section of the source paper.",
            },
            "parent_id": {
                "type": "string",
                "description": "(propose) experiment_id this one descends from (lineage).",
            },
            "experiment_id": {
                "type": "string",
                "description": "(record) The id returned by a prior propose.",
            },
            "actual_metric": {
                "type": "number",
                "description": "(record) Measured metric value for the run.",
            },
            "cost_usd": {
                "type": "number",
                "description": "(record) Dollar cost of the run.",
            },
            "wall_clock_s": {
                "type": "number",
                "description": "(record) Wall-clock seconds the run took.",
            },
            "status": {
                "type": "string",
                "description": "(record) Outcome status; defaults to 'done'.",
            },
            "notes": {
                "type": "string",
                "description": "(record) Free-form notes about the run.",
            },
            "higher_is_better": {
                "type": "boolean",
                "description": (
                    "(best) Whether a larger metric is better. Default true; "
                    "set false for loss-like metrics."
                ),
            },
        },
        "required": ["op"],
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

    Factored out so tests can monkeypatch ``experiment_tool._get_ledger`` to a
    JSONL ledger on a tmp path (no workspace).
    """
    cfg = session.config if session and getattr(session, "config", None) else _load_default_config()
    settings = db_client.resolve_settings(cfg)
    token = getattr(session, "databricks_user_token", None) if session else None
    return ExperimentLedger(settings=settings, user_token=token)


def _ok(formatted: str) -> Dict[str, Any]:
    return {"formatted": formatted, "isError": False}


def _err(msg: str) -> Dict[str, Any]:
    return {"formatted": f"Error: {msg}", "isError": True}


def _missing(args: Dict[str, Any], required: list[str]) -> str | None:
    for field in required:
        if args.get(field) in (None, ""):
            return field
    return None


def _summarize_config(config: dict | None, limit: int = 80) -> str:
    if not config:
        return "{}"
    parts = [f"{k}={v}" for k, v in sorted(config.items())]
    s = ", ".join(parts)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _short_id(experiment_id: str | None) -> str:
    return (experiment_id or "")[:8]


def _format_row(row: ExperimentRow) -> str:
    metric = "—" if row.actual_metric is None else f"{row.actual_metric:g}"
    gap = "—" if row.repro_gap is None else f"{row.repro_gap:+g}"
    return (
        f"{_short_id(row.experiment_id)}  {row.method}  "
        f"[{_summarize_config(row.config)}]  "
        f"{row.metric_name}={metric}  gap={gap}  {row.status}"
    )


def _format_rows(rows: list[ExperimentRow]) -> str:
    return "\n".join(_format_row(r) for r in rows)


async def experiment_handler(arguments: Dict[str, Any], session: Any = None) -> Dict[str, Any]:
    op = (arguments.get("op") or "").strip().lower()
    if not op:
        return _err("op is required (propose | record | list | best | find_similar).")

    try:
        ledger = _get_ledger(session)

        if op == "propose":
            field = _missing(arguments, ["task_id", "hypothesis", "method", "config", "metric_name"])
            if field:
                return _err(f"{field} is required for op=propose.")
            experiment_id = ledger.propose(
                task_id=arguments["task_id"],
                hypothesis=arguments["hypothesis"],
                method=arguments["method"],
                config=arguments["config"],
                metric_name=arguments["metric_name"],
                source_paper=arguments.get("source_paper"),
                source_section=arguments.get("source_section"),
                expected_metric=arguments.get("expected_metric"),
                parent_id=arguments.get("parent_id"),
                session_id=getattr(session, "session_id", None) if session else None,
            )
            return _ok(f"Proposed experiment {experiment_id} (status=proposed).")

        if op == "record":
            field = _missing(arguments, ["experiment_id", "actual_metric"])
            if field:
                return _err(f"{field} is required for op=record.")
            ledger.record_result(
                arguments["experiment_id"],
                actual_metric=arguments["actual_metric"],
                cost_usd=arguments.get("cost_usd"),
                wall_clock_s=arguments.get("wall_clock_s"),
                status=arguments.get("status", "done"),
                notes=arguments.get("notes"),
            )
            msg = f"Recorded result for experiment {arguments['experiment_id']}."
            # Reproduction-gap gate: if the run fell well short of its expected
            # metric, surface the reproduce-first directive instead of letting the
            # agent escalate complexity (the documented Kaggle failure mode).
            row = ledger.get(arguments["experiment_id"])
            if row is not None:
                # Infer metric direction from the row's metric_name so loss-like
                # metrics aren't gated backwards (the agent never has to pass it).
                decision = gate_from_row(
                    row, higher_is_better=metric_higher_is_better(row.metric_name)
                )
                if decision.directive:
                    msg = f"{msg}\n\n{decision.directive}"
            return _ok(msg)

        if op == "list":
            field = _missing(arguments, ["task_id"])
            if field:
                return _err(f"{field} is required for op=list.")
            rows = ledger.list_for_task(arguments["task_id"])
            if not rows:
                return _ok(f"No experiments for task {arguments['task_id']!r}.")
            return _ok(
                f"{len(rows)} experiment(s) for task {arguments['task_id']!r}:\n"
                f"{_format_rows(rows)}"
            )

        if op == "best":
            field = _missing(arguments, ["task_id", "metric_name"])
            if field:
                return _err(f"{field} is required for op=best.")
            row = ledger.best_for_task(
                arguments["task_id"],
                arguments["metric_name"],
                higher_is_better=arguments.get("higher_is_better", True),
            )
            if row is None:
                return _ok(f"No scored experiments for task {arguments['task_id']!r}.")
            return _ok(f"Best for task {arguments['task_id']!r}:\n{_format_row(row)}")

        if op == "find_similar":
            field = _missing(arguments, ["task_id", "method", "config"])
            if field:
                return _err(f"{field} is required for op=find_similar.")
            row = ledger.find_similar_config(
                arguments["task_id"], arguments["method"], arguments["config"]
            )
            if row is None:
                return _ok("No matching config found — this config has not been tried.")
            return _ok(f"Already tried this config:\n{_format_row(row)}")

        return _err(f"Unknown op {op!r}. Use propose | record | list | best | find_similar.")
    except Exception as e:
        logger.exception("experiment handler crashed (op=%s)", op)
        return _err(f"{op} failed: {e}")
