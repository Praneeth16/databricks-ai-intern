"""Eval task spec: load + validate the task yaml.

Mirrors the defensive parse pattern in ``agent/skills/loader.py`` — a
malformed or schema-violating task file raises a clear ``ValueError``
naming the offending file, never a bare ``KeyError``/``TypeError``.

Schema (see plan.md Phase 1)::

    id: kaggle-f1-pitstops-s6e5
    kind: tabular            # tabular | finetune | nlp | cv
    metric: roc_auc
    higher_is_better: true
    baseline_score: 0.94820
    human_ceiling: 0.94924
    leaderboard:
      top_public: 0.9545
      proxy: rank_percentile
    data:
      train: ...
      test: ...
    ground_truth:               # holdout labels the *runner* scores against
      table: cat.schema.tbl     # UC table — exactly one of table | path
      # path: holdout.csv       # or a local csv / parquet / jsonl file
      id_column: id             # optional; omit for positional join
      label_column: target
    holdout: { type: temporal, column: Year, value: 2025 }
    budget: { max_cost_usd: 25.0, max_iterations: 30 }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

_VALID_KINDS = {"tabular", "finetune", "nlp", "cv"}
_REQUIRED_FIELDS = ("id", "kind", "metric", "higher_is_better")


@dataclass(frozen=True)
class EvalTask:
    """One eval task, parsed from yaml.

    ``leaderboard``, ``data``, ``holdout`` and ``budget`` are kept as
    plain dicts — they are heterogeneous per task kind, and the runner
    only reads the keys it needs.
    """

    id: str
    kind: str
    metric: str
    higher_is_better: bool
    baseline_score: Optional[float] = None
    human_ceiling: Optional[float] = None
    leaderboard: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    holdout: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None


def _as_float(value: Any, *, path: Path, key: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Task {path}: field '{key}' must be a number, got {value!r}") from e


def _as_dict(value: Any, *, path: Path, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Task {path}: field '{key}' must be a mapping, got {type(value).__name__}")
    return value


def _validate_ground_truth(gt: dict[str, Any], *, path: Path) -> dict[str, Any]:
    if not gt:
        return gt
    if bool(gt.get("table")) == bool(gt.get("path")):
        raise ValueError(
            f"Task {path}: 'ground_truth' must declare exactly one of 'table' or 'path'"
        )
    if not gt.get("label_column"):
        raise ValueError(f"Task {path}: 'ground_truth' requires 'label_column'")
    return gt


def load_task(path: str | Path) -> EvalTask:
    """Parse and validate a task yaml into an ``EvalTask``.

    Raises ``ValueError`` on a missing file, non-mapping yaml, missing
    required fields, or a type-invalid field — the message always names
    the file so the caller can fix it.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Task file not found: {path}")

    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Task {path} failed to parse as YAML: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Task {path} is not a YAML mapping.")

    missing = [k for k in _REQUIRED_FIELDS if data.get(k) is None]
    if missing:
        raise ValueError(f"Task {path} missing required field(s): {', '.join(missing)}")

    kind = str(data["kind"])
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"Task {path}: kind '{kind}' invalid; expected one of {sorted(_VALID_KINDS)}"
        )

    return EvalTask(
        id=str(data["id"]),
        kind=kind,
        metric=str(data["metric"]),
        higher_is_better=bool(data["higher_is_better"]),
        baseline_score=_as_float(data.get("baseline_score"), path=path, key="baseline_score"),
        human_ceiling=_as_float(data.get("human_ceiling"), path=path, key="human_ceiling"),
        leaderboard=_as_dict(data.get("leaderboard"), path=path, key="leaderboard"),
        data=_as_dict(data.get("data"), path=path, key="data"),
        ground_truth=_validate_ground_truth(
            _as_dict(data.get("ground_truth"), path=path, key="ground_truth"), path=path
        ),
        holdout=_as_dict(data.get("holdout"), path=path, key="holdout"),
        budget=_as_dict(data.get("budget"), path=path, key="budget"),
        source_path=path,
    )
