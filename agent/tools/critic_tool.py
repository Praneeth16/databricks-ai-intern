"""``critic`` builtin — audit a result for the classic ML self-deception modes.

Thin agent-facing wrapper over :mod:`agent.core.critic`. The agent calls this
BEFORE declaring a win or registering a model, passing whatever signals it has
(CV vs LB, the predicted target vs the submission columns, feature importances
or an adversarial-validation AUC, blend-member rank correlation). The critic
flags overfit, target confusion, leakage, and correlation-floor blends — the
exact failure classes the F1 Kaggle demo logged. Pure analysis, no I/O.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from agent.core import critic

logger = logging.getLogger(__name__)


CRITIC_TOOL_SPEC: Dict[str, Any] = {
    "name": "critic",
    "description": (
        "Audit an experiment result for self-deception BEFORE you trust it. "
        "Pass any signals you have; each detector runs only when its inputs are "
        "present. Catches:\n"
        "- overfit: CV much better than LB (give cv_score + lb_score + "
        "higher_is_better).\n"
        "- target_confusion: predicting the wrong column (give "
        "sample_submission_columns + predicted_target, optional id_columns).\n"
        "- leakage: one feature dominates importance, or train/test are "
        "separable (give feature_importances and/or adversarial_auc).\n"
        "- correlation_floor: blend members are near-identical so blending "
        "won't help (give spearman).\n"
        "Returns findings with severity warn|block. A 'block' means do NOT "
        "register/submit until resolved."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "cv_score": {"type": "number", "description": "Cross-val / holdout score."},
            "lb_score": {"type": "number", "description": "Leaderboard / hidden-test score."},
            "higher_is_better": {"type": "boolean", "description": "Metric direction (default true)."},
            "sample_submission_columns": {
                "type": "array", "items": {"type": "string"},
                "description": "Columns of sample_submission.csv.",
            },
            "predicted_target": {"type": "string", "description": "The column you trained to predict."},
            "id_columns": {
                "type": "array", "items": {"type": "string"},
                "description": "Id columns to exclude when inferring the target.",
            },
            "feature_importances": {
                "type": "object",
                "description": "Map of feature -> importance (any scale).",
            },
            "adversarial_auc": {
                "type": "number",
                "description": "AUC of a train-vs-test classifier; high = distribution split.",
            },
            "spearman": {
                "type": "number",
                "description": "Rank correlation between two blend members' predictions.",
            },
        },
        "required": [],
    },
}


async def critic_handler(arguments: Dict[str, Any], session: Any = None) -> Dict[str, Any]:
    try:
        findings = critic.audit(arguments or {})
    except Exception as e:  # noqa: BLE001 — defensive boundary
        logger.exception("critic audit failed")
        return {"formatted": f"Error: critic failed: {e}", "isError": True}

    if not findings:
        return {
            "formatted": "No issues detected by the critic (given the signals provided).",
            "isError": False,
        }
    blocks = [f for f in findings if f.severity == "block"]
    lines = [f"{f.severity.upper()} [{f.kind}] {f.message}" for f in findings]
    header = (
        f"{len(findings)} finding(s), {len(blocks)} blocking — "
        "do NOT register/submit until blocks are resolved:"
        if blocks
        else f"{len(findings)} advisory finding(s):"
    )
    return {"formatted": header + "\n" + "\n".join(lines), "isError": False}
