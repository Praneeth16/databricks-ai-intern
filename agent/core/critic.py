"""Critic / verifier — audit an experiment result for false-win failure modes.

An agent chasing a leaderboard will happily report a "win" that is actually
leakage, the wrong target, an overfit (CV up but LB down), or a blend of
near-duplicate models with no real diversity. Each of these is documented as a
concrete regression in ``examples/kaggle-f1-pitstops-s6e5/FINAL_RESULTS.md``:

- Overfit: Race-grouped KFold (v7) and StratGroupKFold (v13) both produced
  OOF ↑ but LB ↓ — the local score improved while the real score fell.
- Target confusion: predicting a column that is not the submission target.
- Leakage: a single dominant feature, or a train/test split that an
  adversarial classifier can separate (high adversarial AUC = distribution
  shift the model can exploit).
- Correlation floor: ``spearman > 0.998`` between two blend members means they
  are the same model — blending buys nothing (v5.4, v8, v9 all hit this wall).

These detectors are pure functions over signals the caller supplies. NO I/O,
NO LLM. Each detector returns a ``Finding`` when it fires, else ``None``;
``audit`` dispatches to whichever detectors have their inputs present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Finding:
    kind: str  # "overfit" | "target_confusion" | "leakage" | "correlation_floor"
    severity: str  # "warn" | "block"
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


def check_overfit(
    cv_score: float | None,
    lb_score: float | None,
    *,
    threshold: float = 0.01,
    higher_is_better: bool = True,
) -> Finding | None:
    """Flag a model whose local CV is better than the leaderboard by > threshold.

    When ``higher_is_better`` the gap is ``cv - lb`` (CV scored higher locally);
    otherwise it is ``lb - cv`` (CV scored a lower/"better" loss locally). A
    positive gap past ``threshold`` is the OOF↑LB↓ signature. Severity is
    "block" when the gap exceeds ``2 * threshold``, else "warn". Returns ``None``
    when either score is missing or CV did not beat LB by more than threshold.
    """
    if cv_score is None or lb_score is None:
        return None

    gap = cv_score - lb_score if higher_is_better else lb_score - cv_score
    if gap <= threshold:
        return None

    severity = "block" if gap > 2 * threshold else "warn"
    logger.info("Overfit detected: cv=%s lb=%s gap=%.4f (%s)", cv_score, lb_score, gap, severity)
    return Finding(
        kind="overfit",
        severity=severity,
        message=(
            f"CV beats LB by {gap:.4f} (> {threshold:.4f}) — local score improved "
            f"while the real score fell. Classic overfit / wrong CV geometry; trust "
            f"the leaderboard, not the local number."
        ),
        evidence={"cv": cv_score, "lb": lb_score, "gap": gap},
    )


def check_target_confusion(
    sample_submission_columns: list[str] | tuple[str, ...],
    predicted_target: str,
    *,
    id_columns: list[str] | tuple[str, ...] = (),
) -> Finding | None:
    """Flag predicting a column absent from the submission's non-id targets.

    The valid targets are the sample-submission columns minus ``id_columns``.
    If ``predicted_target`` is not among them, the agent is modelling the wrong
    thing — always a "block". Returns ``None`` when the predicted target is a
    valid candidate.
    """
    id_set = set(id_columns)
    candidates = [c for c in sample_submission_columns if c not in id_set]
    if predicted_target in candidates:
        return None

    logger.warning(
        "Target confusion: predicted=%s not in candidates=%s", predicted_target, candidates
    )
    return Finding(
        kind="target_confusion",
        severity="block",
        message=(
            f"Predicted target {predicted_target!r} is not a submission target "
            f"(expected one of {candidates}). The model is predicting the wrong column."
        ),
        evidence={"expected": candidates, "predicted": predicted_target},
    )


def check_leakage(
    feature_importances: dict[str, float] | None,
    *,
    dominance: float = 0.6,
    adversarial_auc: float | None = None,
    adv_threshold: float = 0.8,
) -> Finding | None:
    """Flag a suspected leak from a dominant feature or a separable train/test split.

    Two independent paths:

    - Dominance: if a single feature's share of total importance >= ``dominance``,
      that feature likely encodes the target (a leak) — "warn".
    - Adversarial AUC: if a classifier can tell train from test with AUC >=
      ``adv_threshold``, the split is detectably shifted — "warn", escalating to
      "block" at AUC >= 0.95.

    The dominance path takes precedence when both fire. Returns ``None`` when
    neither path triggers.
    """
    if feature_importances:
        total = sum(abs(v) for v in feature_importances.values())
        if total > 0:
            top_feat, top_val = max(
                feature_importances.items(), key=lambda kv: abs(kv[1])
            )
            share = abs(top_val) / total
            if share >= dominance:
                logger.warning("Leakage: feature %s share=%.3f", top_feat, share)
                return Finding(
                    kind="leakage",
                    severity="warn",
                    message=(
                        f"Feature {top_feat!r} holds {share:.1%} of total importance "
                        f"(>= {dominance:.0%}) — likely leaks the target."
                    ),
                    evidence={"top_feature": top_feat, "share": share},
                )

    if adversarial_auc is not None and adversarial_auc >= adv_threshold:
        severity = "block" if adversarial_auc >= 0.95 else "warn"
        logger.warning("Leakage: adversarial_auc=%.3f (%s)", adversarial_auc, severity)
        return Finding(
            kind="leakage",
            severity=severity,
            message=(
                f"Adversarial validation AUC {adversarial_auc:.3f} (>= {adv_threshold:.2f}) "
                f"— a classifier can separate train from test, so the split is shifted "
                f"and any local score may not transfer."
            ),
            evidence={"adversarial_auc": adversarial_auc},
        )

    return None


def check_correlation_floor(
    spearman: float | None,
    *,
    threshold: float = 0.998,
) -> Finding | None:
    """Flag two blend members that are effectively the same model.

    A Spearman rank correlation >= ``threshold`` means the members rank rows
    identically; blending them adds no diversity and cannot help. Returns
    ``None`` when correlation is below the floor or missing.
    """
    if spearman is None or spearman < threshold:
        return None

    logger.info("Correlation floor: spearman=%.4f >= %.4f", spearman, threshold)
    return Finding(
        kind="correlation_floor",
        severity="warn",
        message=(
            f"Blend members correlate at spearman={spearman:.4f} (>= {threshold:.3f}) "
            f"— effectively the same model. Blending them won't help; seek real diversity."
        ),
        evidence={"spearman": spearman, "threshold": threshold},
    )


def audit(signals: dict) -> list[Finding]:
    """Run every detector whose inputs are present in ``signals`` and collect findings.

    Recognized keys (a detector runs only when its required key(s) are present):

    - ``cv_score``, ``lb_score`` (both required) — overfit check.
      Optional: ``higher_is_better``, ``overfit_threshold``.
    - ``sample_submission_columns``, ``predicted_target`` (both required) —
      target-confusion check. Optional: ``id_columns``.
    - ``feature_importances`` and/or ``adversarial_auc`` (either present) —
      leakage check. Optional: ``dominance``, ``adv_threshold``.
    - ``spearman`` (present) — correlation-floor check. Optional:
      ``correlation_threshold``.

    Returns all non-None findings (possibly empty). Detectors with absent inputs
    are skipped silently.
    """
    findings: list[Finding] = []

    if "cv_score" in signals and "lb_score" in signals:
        kwargs: dict[str, Any] = {}
        if "higher_is_better" in signals:
            kwargs["higher_is_better"] = signals["higher_is_better"]
        if "overfit_threshold" in signals:
            kwargs["threshold"] = signals["overfit_threshold"]
        f = check_overfit(signals["cv_score"], signals["lb_score"], **kwargs)
        if f is not None:
            findings.append(f)

    if "sample_submission_columns" in signals and "predicted_target" in signals:
        kwargs = {}
        if "id_columns" in signals:
            kwargs["id_columns"] = signals["id_columns"]
        f = check_target_confusion(
            signals["sample_submission_columns"], signals["predicted_target"], **kwargs
        )
        if f is not None:
            findings.append(f)

    if "feature_importances" in signals or "adversarial_auc" in signals:
        kwargs = {}
        if "dominance" in signals:
            kwargs["dominance"] = signals["dominance"]
        if "adversarial_auc" in signals:
            kwargs["adversarial_auc"] = signals["adversarial_auc"]
        if "adv_threshold" in signals:
            kwargs["adv_threshold"] = signals["adv_threshold"]
        f = check_leakage(signals.get("feature_importances"), **kwargs)
        if f is not None:
            findings.append(f)

    if "spearman" in signals:
        kwargs = {}
        if "correlation_threshold" in signals:
            kwargs["threshold"] = signals["correlation_threshold"]
        f = check_correlation_floor(signals["spearman"], **kwargs)
        if f is not None:
            findings.append(f)

    return findings
