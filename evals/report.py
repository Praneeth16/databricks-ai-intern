"""Scorecard writer: markdown + json sidecar for one eval run."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from evals.runner import EvalResult
from evals.task_spec import EvalTask

logger = logging.getLogger(__name__)


def _fmt(value: object, spec: str = ".5f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def _scorecard_md(result: EvalResult, task: EvalTask) -> str:
    direction = "higher is better" if task.higher_is_better else "lower is better"
    lines = [
        f"# Eval scorecard — {task.id}",
        "",
    ]
    if result.self_reported:
        lines += [
            "> **WARNING: SELF-REPORTED SCORE.** The agent asserted this score "
            "itself; it was NOT verified against the task's ground truth.",
            "",
        ]
    score_suffix = " (self-reported, unverified)" if result.self_reported else ""
    lines += [
        f"- **Metric:** {result.metric_name} ({direction})",
        f"- **Final score:** {_fmt(result.score)}{score_suffix}",
        f"- **Baseline:** {_fmt(task.baseline_score)}",
        f"- **Human ceiling:** {_fmt(task.human_ceiling)}",
        f"- **CV↔LB gap:** {_fmt(result.cv_lb_gap)}",
        f"- **LB percentile:** {_fmt(result.leaderboard_percentile, '.2f')}",
        f"- **Cost (USD):** {_fmt(result.cost_usd, '.4f')}",
        f"- **Wall-clock (s):** {_fmt(result.wall_clock_s, '.2f')}",
        f"- **Iterations:** {result.iterations}",
        f"- **Self-recovered failures:** {result.self_recovered_failures}",
        f"- **Experiment ID:** {result.experiment_id}",
        "",
    ]
    return "\n".join(lines)


def write_report(result: EvalResult, task: EvalTask, out_dir: str | Path) -> Path:
    """Write a markdown scorecard + json sidecar under ``out_dir``.

    Returns the path to the markdown file. The json sidecar sits beside
    it with the same stem so downstream tooling can parse it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"scorecard-{task.id}"
    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"

    md_path.write_text(_scorecard_md(result, task))

    payload = {
        "task_id": result.task_id,
        "metric_name": result.metric_name,
        "score": result.score,
        "self_reported": result.self_reported,
        "baseline_score": task.baseline_score,
        "human_ceiling": task.human_ceiling,
        "cv_lb_gap": result.cv_lb_gap,
        "leaderboard_percentile": result.leaderboard_percentile,
        "cost_usd": result.cost_usd,
        "wall_clock_s": result.wall_clock_s,
        "iterations": result.iterations,
        "self_recovered_failures": result.self_recovered_failures,
        "experiment_id": result.experiment_id,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    logger.info("Wrote eval scorecard to %s", md_path)
    return md_path
