"""Eval runner: drive an agent against a task, score it, record it.

The runner is the auto-researcher's scorer in code form. It is
deliberately synchronous and side-effect-light: propose a ledger row,
mark it running, invoke the agent, score the output, record the result,
return an ``EvalResult``.

Ground truth comes from the task spec (``ground_truth:`` — a UC table or
a local csv/parquet/jsonl file), **never** from agent output. A
``"y_true"`` key in the agent output is ignored.

Agent callable contract
------------------------
``agent_callable(task: EvalTask) -> dict`` must return a mapping with:

    - one of:
        - ``"predictions"``: the agent's holdout predictions, which the
          runner scores with ``task.metric`` against the task's declared
          ground truth. A mapping ``{id: prediction}`` when the spec
          declares ``id_column`` (coverage is validated — missing or
          extra ids fail the run); else a positional sequence with a
          strict length check.
        - ``"score"``: a float — the agent's own self-asserted metric
          value. Recorded, but the result carries
          ``self_reported=True`` and must never be presented as
          verified.
    - ``"iterations"``: int — agent loop iterations spent.
    - ``"cost_usd"``: float — USD spent.
    - ``"wall_clock_s"``: float — wall-clock seconds.
    - ``"self_recovered_failures"``: int — failures the agent recovered
      from without human help.

Optional keys the runner uses when present:
    - ``"lb_score"``: the leaderboard / hidden-test score, used to
      compute the CV↔LB gap against the local score. Falls back to
      ``task.leaderboard["top_public"]`` only for the percentile, never
      for the gap.
    - ``"artifacts"``: dict of artifact paths to forward to the ledger.
    - ``"mlflow_run_id"``: str, forwarded to ``mark_running``.

The ``ledger`` is duck-typed (see plan.md Phase 0 frozen interface):
``propose``, ``mark_running``, ``record_result``. A scoring failure
(coverage mismatch, missing ground truth, …) is recorded to the ledger
with ``status="failed"`` and its reason before the error propagates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from evals import scorers
from evals.task_spec import EvalTask

logger = logging.getLogger(__name__)

AgentCallable = Callable[[EvalTask], dict[str, Any]]


@dataclass(frozen=True)
class EvalResult:
    """Scorecard for one eval run."""

    task_id: str
    metric_name: str
    score: float
    experiment_id: str
    cv_lb_gap: Optional[float] = None
    leaderboard_percentile: Optional[float] = None
    cost_usd: Optional[float] = None
    wall_clock_s: Optional[float] = None
    iterations: int = 0
    self_recovered_failures: int = 0
    self_reported: bool = False


def _coerce(value: Any) -> Any:
    """Numeric-normalize a label/id so csv strings join against ints."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _ground_truth_rows_from_file(task: EvalTask, gt: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(gt["path"])
    if not path.is_absolute() and task.source_path is not None:
        path = task.source_path.parent / path
    if not path.exists():
        raise ValueError(f"Task {task.id}: ground_truth file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import csv

        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".jsonl":
        import json

        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".parquet":
        import pandas as pd  # lazy — heavy import

        return pd.read_parquet(path).to_dict("records")
    raise ValueError(
        f"Task {task.id}: unsupported ground_truth file type '{suffix}' "
        "(expected csv, parquet, or jsonl)"
    )


def _ground_truth_rows_from_table(task: EvalTask, gt: dict[str, Any]) -> list[dict[str, Any]]:
    from types import SimpleNamespace

    from agent.config import DatabricksConfig
    from agent.core import db_client  # lazy — pulls the SDK

    settings = db_client.resolve_settings(SimpleNamespace(databricks=DatabricksConfig()))
    columns = [c for c in (gt.get("id_column"), gt["label_column"]) if c]
    conn = db_client.get_sql_connection(settings)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT {} FROM {}".format(
                    ", ".join(f"`{c}`" for c in columns), gt["table"]
                )
            )
            rows = cursor.fetchall()
    finally:
        conn.close()
    return [dict(zip(columns, row)) for row in rows]


def _load_ground_truth(task: EvalTask) -> tuple[Optional[list[Any]], list[Any]]:
    """Return ``(ids, labels)`` from the task's ground_truth source.

    ``ids`` is None when the spec declares no ``id_column`` (positional
    join).
    """
    gt = task.ground_truth
    rows = (
        _ground_truth_rows_from_file(task, gt)
        if gt.get("path")
        else _ground_truth_rows_from_table(task, gt)
    )
    id_column = gt.get("id_column")
    try:
        labels = [_coerce(row[gt["label_column"]]) for row in rows]
        ids = None if not id_column else [_coerce(row[id_column]) for row in rows]
    except KeyError as e:
        raise ValueError(f"Task {task.id}: ground_truth rows missing column {e}") from e
    if not labels:
        raise ValueError(f"Task {task.id}: ground_truth source is empty")
    return ids, labels


def _align_predictions(
    task: EvalTask, predictions: Any, ids: Optional[list[Any]], n_true: int
) -> list[Any]:
    """Order predictions against ground truth, validating coverage."""
    if ids is None:
        if isinstance(predictions, dict):
            raise ValueError(
                f"Task {task.id}: ground_truth declares no id_column; "
                "'predictions' must be a positional sequence"
            )
        if len(predictions) != n_true:
            raise ValueError(
                f"Task {task.id}: prediction coverage mismatch — got "
                f"{len(predictions)} predictions for {n_true} ground-truth rows"
            )
        return list(predictions)

    if not isinstance(predictions, dict):
        raise ValueError(
            f"Task {task.id}: ground_truth declares id_column "
            f"'{task.ground_truth['id_column']}'; 'predictions' must map id -> prediction"
        )
    pred_by_id = {_coerce(k): v for k, v in predictions.items()}
    missing = [i for i in ids if i not in pred_by_id]
    extra = sorted(set(pred_by_id) - set(ids), key=repr)
    if missing or extra:
        raise ValueError(
            f"Task {task.id}: prediction coverage mismatch — "
            f"{len(missing)} missing id(s) (e.g. {missing[:5]}), "
            f"{len(extra)} extra id(s) (e.g. {extra[:5]})"
        )
    return [pred_by_id[i] for i in ids]


def _score_output(task: EvalTask, output: dict[str, Any]) -> tuple[float, bool]:
    """Resolve ``(score, self_reported)`` from the agent output.

    Predictions are always scored against the task's declared ground
    truth — agent-supplied labels are never trusted. A bare ``"score"``
    with no predictions is recorded as self-reported.
    """
    predictions = output.get("predictions")
    if predictions is None:
        if output.get("score") is not None:
            return float(output["score"]), True
        raise ValueError(
            f"Task {task.id}: agent output must contain 'predictions' or 'score'"
        )

    if not task.ground_truth:
        raise ValueError(
            f"Task {task.id}: task spec declares no ground_truth source; "
            "cannot verify 'predictions'"
        )

    ids, y_true = _load_ground_truth(task)
    y_pred = _align_predictions(task, predictions, ids, len(y_true))

    metric = task.metric
    if metric == "roc_auc":
        return scorers.roc_auc(y_true, y_pred), False
    if metric == "accuracy":
        return scorers.accuracy(y_true, y_pred), False
    if metric == "eval_loss":
        return scorers.eval_loss(y_pred), False
    raise ValueError(f"Task {task.id}: cannot score metric '{metric}' from predictions")


def _leaderboard_percentile(task: EvalTask, score: float) -> Optional[float]:
    lb = task.leaderboard or {}
    competitor_scores = lb.get("scores")
    if isinstance(competitor_scores, (list, tuple)) and competitor_scores:
        return scorers.rank_percentile(score, list(competitor_scores))
    top_public = lb.get("top_public")
    if top_public is not None:
        # Single-anchor leaderboard → percentile against the top score.
        return scorers.rank_percentile(score, [float(top_public)])
    return None


def run_eval(
    task: EvalTask,
    agent_callable: AgentCallable,
    ledger: Any,
    *,
    session_id: Optional[str] = None,
) -> EvalResult:
    """Run ``task`` through ``agent_callable``, score it, record to ``ledger``."""
    experiment_id = ledger.propose(
        task_id=task.id,
        hypothesis=f"Agent run on {task.id}",
        method="agent_callable",
        config={"kind": task.kind, "metric": task.metric},
        metric_name=task.metric,
        expected_metric=task.human_ceiling,
        session_id=session_id,
    )

    output = agent_callable(task)
    ledger.mark_running(experiment_id, mlflow_run_id=output.get("mlflow_run_id"))

    try:
        score, self_reported = _score_output(task, output)
    except ValueError as e:
        ledger.record_result(
            experiment_id, actual_metric=None, status="failed", notes=str(e)
        )
        raise

    cv_lb_gap: Optional[float] = None
    lb_score = output.get("lb_score")
    if lb_score is not None:
        cv_lb_gap = float(score) - float(lb_score)

    leaderboard_percentile = _leaderboard_percentile(task, score)

    cost_usd = output.get("cost_usd")
    wall_clock_s = output.get("wall_clock_s")
    iterations = int(output.get("iterations", 0))
    self_recovered_failures = int(output.get("self_recovered_failures", 0))

    notes = []
    if self_reported:
        notes.append("self_reported=true (agent-asserted score, not verified)")
    if cv_lb_gap is not None:
        notes.append(f"cv_lb_gap={cv_lb_gap:.5f}")

    ledger.record_result(
        experiment_id,
        actual_metric=score,
        cost_usd=cost_usd,
        wall_clock_s=wall_clock_s,
        artifacts=output.get("artifacts"),
        status="done",
        notes="; ".join(notes) or None,
    )

    return EvalResult(
        task_id=task.id,
        metric_name=task.metric,
        score=float(score),
        experiment_id=experiment_id,
        cv_lb_gap=cv_lb_gap,
        leaderboard_percentile=leaderboard_percentile,
        cost_usd=None if cost_usd is None else float(cost_usd),
        wall_clock_s=None if wall_clock_s is None else float(wall_clock_s),
        iterations=iterations,
        self_recovered_failures=self_recovered_failures,
        self_reported=self_reported,
    )
