"""Pure deterministic scorers for the eval harness.

Each function takes plain Python sequences and returns a ``float``. No
hidden state, no I/O — so the runner can dispatch by metric name and
the unit tests can pin exact values.

sklearn is used for ``roc_auc``/``accuracy`` when importable (it is in
the eval extra), but it is *not* a hard dependency: ``roc_auc`` falls
back to the rank / Mann–Whitney-U formula and ``accuracy`` is trivial,
so the harness scores correctly in a bare environment too.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

Number = float | int


def roc_auc(y_true: Sequence[Number], y_score: Sequence[Number]) -> float:
    """Area under the ROC curve for binary labels.

    Prefers ``sklearn.metrics.roc_auc_score``; falls back to the
    rank-based Mann–Whitney-U estimator (tie-aware via average ranks)
    so the result is identical without sklearn installed.
    """
    if len(y_true) != len(y_score):
        raise ValueError("roc_auc: y_true and y_score length mismatch")
    if len(y_true) == 0:
        raise ValueError("roc_auc: empty input")

    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(list(y_true), list(y_score)))
    except Exception:  # noqa: BLE001 — any sklearn issue → pure fallback
        return _roc_auc_rank(y_true, y_score)


def _roc_auc_rank(y_true: Sequence[Number], y_score: Sequence[Number]) -> float:
    n_pos = sum(1 for y in y_true if y > 0)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc: only one class present in y_true")

    # Average ranks (1-indexed), tie-aware.
    order = sorted(range(len(y_score)), key=lambda i: y_score[i])
    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # mean of 1-indexed ranks in the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    sum_pos_ranks = sum(ranks[idx] for idx, y in enumerate(y_true) if y > 0)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def accuracy(y_true: Sequence[Number], y_pred: Sequence[Number]) -> float:
    """Fraction of exact label matches."""
    if len(y_true) != len(y_pred):
        raise ValueError("accuracy: y_true and y_pred length mismatch")
    if len(y_true) == 0:
        raise ValueError("accuracy: empty input")
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return float(correct / len(y_true))


def rank_percentile(score: Number, leaderboard_scores: Sequence[Number]) -> float:
    """Percentile position of ``score`` against competitor scores.

    Returns a value in [0, 100]: the fraction of leaderboard entries
    that ``score`` is greater than or equal to, ×100. Empty leaderboard
    → 100.0 (nothing to beat).
    """
    if not leaderboard_scores:
        return 100.0
    at_or_below = sum(1 for s in leaderboard_scores if score >= s)
    return float(100.0 * at_or_below / len(leaderboard_scores))


def eval_loss(losses: Sequence[Number]) -> float:
    """Mean of a sequence of losses (passthrough for a single value)."""
    if len(losses) == 0:
        raise ValueError("eval_loss: empty input")
    return float(sum(losses) / len(losses))
