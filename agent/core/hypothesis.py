"""Hypothesis generator — turn extracted paper findings into ranked, testable ledger rows.

The auto-researcher's research sub-agent (``papers_tool`` / ``research_tool``)
reads papers and extracts structured *findings* — "method M reported metric V on
dataset D". A finding is not yet actionable: the sweep and the
:class:`~agent.core.experiment_ledger.ExperimentLedger` run *hypotheses*, each a
concrete (method, config, expected_metric) bundle with a one-line testable claim.
This module is the bridge.

It is pure transformation + ranking — **no LLM calls, no I/O**. Extraction from
prose happens upstream (the agent fills the ``Finding`` dict); the heavy ledger
side effects are injected via the frozen ledger interface for dedup only.

A ``Finding`` is a plain dict (documented on :func:`generate_hypotheses`):

    {
        "title":          str,            # paper / result title (claim source)
        "method":         str,            # technique name, e.g. "lightgbm-dart"
        "reported_metric": float | None,  # the number the paper reports
        "metric_name":    str,            # what that number measures, e.g. "roc_auc"
        "dataset":        str | None,     # optional, informational
        "source_paper":   str | None,     # citation / URL
        "source_section": str | None,     # section / table the number came from
        "config":         dict | None,    # hyperparameters to reproduce the result
        "snippet":        str | None,     # optional supporting quote
    }

Only findings whose ``metric_name`` matches the task metric (case-insensitive)
become hypotheses; the rest are dropped as off-task.

**Ranking scheme** (best-first), implemented in :func:`rank_hypotheses`:

- A hypothesis with a known ``expected_lift`` (both ``reported_metric`` and
  ``current_best`` present) ranks by that lift — larger expected improvement
  over the current best ranks higher. ``expected_lift`` is already oriented so
  positive always means "expected to beat current best", for loss metrics too.
- A hypothesis with no ``expected_lift`` (no ``current_best``, or no
  ``reported_metric``) falls back to its ``expected_metric``, oriented by
  ``higher_is_better`` — a higher reported AUC (or lower reported loss) ranks
  above a worse one. Findings with neither lift nor metric score 0.0.
- Hypotheses *with* a known lift always outrank fallback-only ones, so a
  measured improvement is preferred over an unanchored reported number.
- Ties (and equal scores) break by original input order (stable sort).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hypothesis:
    """A testable claim derived from a finding, ready for the sweep / ledger."""

    hypothesis: str           # one-line testable claim
    method: str               # technique name
    config: dict              # hyperparameters to reproduce the result
    metric_name: str          # the metric this hypothesis targets
    expected_metric: float | None   # the number the source reports (== reported_metric)
    source_paper: str | None
    source_section: str | None
    expected_lift: float | None     # signed improvement over current_best (positive = better)
    score: float              # ranking score (best-first); see module docstring



def _claim(title: str, method: str, reported_metric: float | None, metric_name: str) -> str:
    """A concise one-line testable claim derived from the finding."""
    base = (title or method or "this method").strip()
    if reported_metric is not None:
        return f"{base} should reach {metric_name} ~= {reported_metric:g}"
    return f"{base} should improve {metric_name}"


def generate_hypotheses(
    findings: list[dict],
    *,
    task_metric: str,
    current_best: float | None = None,
    higher_is_better: bool = True,
    ledger=None,
    task_id: str | None = None,
) -> list[Hypothesis]:
    """Transform structured findings into ranked, testable hypotheses.

    For each finding whose ``metric_name`` matches ``task_metric``
    (case-insensitive) a :class:`Hypothesis` is built:

    - ``hypothesis`` — a concise testable claim from title / method / metric.
    - ``config`` — ``finding["config"]`` if present, else ``{"method": method}``
      so the row is never configless.
    - ``expected_metric`` — the finding's ``reported_metric``.
    - ``expected_lift`` — ``reported_metric - current_best`` when higher is
      better (``current_best - reported_metric`` when lower is better), so a
      positive lift always means "expected to beat current best". ``None`` when
      either side is missing.

    Off-task findings (metric mismatch) are dropped.

    Dedup: when both ``ledger`` and ``task_id`` are given, a hypothesis whose
    ``(task_id, method, config)`` already has a live ledger row
    (``ledger.find_similar_config(...) is not None``) is skipped.

    Returns hypotheses sorted best-first (see module docstring), stable by input
    order on ties.
    """
    target = task_metric.strip().lower()
    hyps: list[Hypothesis] = []

    for finding in findings:
        metric_name = finding.get("metric_name")
        if not metric_name or metric_name.strip().lower() != target:
            continue

        method = finding.get("method") or "unknown"
        config = finding.get("config") or {"method": method}
        reported = finding.get("reported_metric")

        if reported is not None and current_best is not None:
            expected_lift = (
                reported - current_best
                if higher_is_better
                else current_best - reported
            )
        else:
            expected_lift = None

        if ledger is not None and task_id is not None:
            if ledger.find_similar_config(task_id, method, config) is not None:
                logger.debug(
                    "dedup: skipping already-tried %s on %s", method, task_id
                )
                continue

        hyps.append(
            Hypothesis(
                hypothesis=_claim(
                    finding.get("title"), method, reported, metric_name
                ),
                method=method,
                config=config,
                metric_name=metric_name,
                expected_metric=reported,
                source_paper=finding.get("source_paper"),
                source_section=finding.get("source_section"),
                expected_lift=expected_lift,
                score=_score(reported, expected_lift, higher_is_better),
            )
        )

    return rank_hypotheses(hyps, higher_is_better=higher_is_better)


def _score(
    expected_metric: float | None,
    expected_lift: float | None,
    higher_is_better: bool,
) -> float:
    """Within-tier ranking score (higher = better). The lift-vs-fallback tier
    is enforced separately in :func:`rank_hypotheses`, so no magnitude offset is
    needed here — this only orders hypotheses inside the same tier."""
    if expected_lift is not None:
        return expected_lift
    if expected_metric is not None:
        return expected_metric if higher_is_better else -expected_metric
    return 0.0


def rank_hypotheses(
    hyps: list[Hypothesis], higher_is_better: bool = True
) -> list[Hypothesis]:
    """Sort hypotheses best-first (stable on ties).

    Two explicit tiers: every lift-anchored hypothesis (a measured
    ``expected_lift``) outranks every fallback-only one regardless of
    magnitude; within a tier, by ``score``. ``higher_is_better`` is accepted for
    call-site symmetry — the score is already oriented at construction.
    """
    return sorted(
        hyps, key=lambda h: (h.expected_lift is not None, h.score), reverse=True
    )
