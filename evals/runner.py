"""Eval runner: drive an agent against a task, score it, record it.

The runner is the auto-researcher's scorer in code form. It is
deliberately synchronous and side-effect-light: propose a ledger row,
mark it running, invoke the agent, score the output, record the result,
return an ``EvalResult``.

Agent callable contract
------------------------
``agent_callable(task: EvalTask) -> dict`` must return a mapping with:

    - one of:
        - ``"score"``: a float — the agent's own absolute metric value
          (used directly when the agent already scored on the holdout);
        - ``"predictions"``: a sequence of scores/labels for the holdout,
          which the runner scores with ``task.metric`` against
          ``"y_true"`` (also required in that case).
    - ``"iterations"``: int — agent loop iterations spent.
    - ``"cost_usd"``: float — USD spent.
    - ``"wall_clock_s"``: float — wall-clock seconds.
    - ``"self_recovered_failures"``: int — failures the agent recovered
      from without human help.

Optional keys the runner uses when present:
    - ``"y_true"``: holdout labels (required iff ``"predictions"`` given).
    - ``"lb_score"``: the leaderboard / hidden-test score, used to
      compute the CV↔LB gap against the local ``"score"``. Falls back to
      ``task.leaderboard["top_public"]`` only for the percentile, never
      for the gap.
    - ``"artifacts"``: dict of artifact paths to forward to the ledger.
    - ``"mlflow_run_id"``: str, forwarded to ``mark_running``.

The ``ledger`` is duck-typed (see plan.md Phase 0 frozen interface):
``propose``, ``mark_running``, ``record_result``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


def _score_output(task: EvalTask, output: dict[str, Any]) -> float:
    """Resolve the absolute metric value from the agent output."""
    if "score" in output and output["score"] is not None:
        return float(output["score"])

    if "predictions" not in output:
        raise ValueError(
            f"Task {task.id}: agent output must contain 'score' or 'predictions'"
        )
    predictions = output["predictions"]
    y_true = output.get("y_true")
    if y_true is None:
        raise ValueError(
            f"Task {task.id}: agent output with 'predictions' must also supply 'y_true'"
        )

    metric = task.metric
    if metric == "roc_auc":
        return scorers.roc_auc(y_true, predictions)
    if metric == "accuracy":
        return scorers.accuracy(y_true, predictions)
    if metric == "eval_loss":
        return scorers.eval_loss(predictions)
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

    score = _score_output(task, output)

    cv_lb_gap: Optional[float] = None
    lb_score = output.get("lb_score")
    if lb_score is not None:
        cv_lb_gap = float(score) - float(lb_score)

    leaderboard_percentile = _leaderboard_percentile(task, score)

    cost_usd = output.get("cost_usd")
    wall_clock_s = output.get("wall_clock_s")
    iterations = int(output.get("iterations", 0))
    self_recovered_failures = int(output.get("self_recovered_failures", 0))

    ledger.record_result(
        experiment_id,
        actual_metric=score,
        cost_usd=cost_usd,
        wall_clock_s=wall_clock_s,
        artifacts=output.get("artifacts"),
        status="done",
        notes=None if cv_lb_gap is None else f"cv_lb_gap={cv_lb_gap:.5f}",
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
    )
