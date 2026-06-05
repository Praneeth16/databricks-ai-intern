"""Experiment Ledger — the persistence spine of the auto-researcher loop.

Findings die at session end unless they are written down. Everything
downstream (reproduction-gap gate, dedup, eval-driven iteration, cross-session
learning) reasons over a structured, durable record of every experiment the
agent proposed, ran, and scored. This module is that record.

Two backends sit behind one interface, mirroring ``telemetry.py``'s
best-effort / offline-friendly ethos:

- **SQL (UC Delta)** when ``settings.warehouse_id`` is set — writes to
  ``{catalog}.{schema}.experiments`` via ``db_client.get_sql_connection``.
- **Local JSONL** otherwise (default ``session_logs/experiments.jsonl``) — so
  offline runs and unit tests work without a workspace.

The interface is frozen (plan.md Phase 0); downstream phases code against it.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from agent.core.db_client import DatabricksSettings

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_PATH = Path("session_logs/experiments.jsonl")

# Columns that carry a JSON-serialized dict in their STRING storage column.
_JSON_COLUMNS = ("config", "artifacts")
# Float columns are DOUBLE in the table. A bare Python float binds as FLOAT on
# the databricks-sql connector, which silently narrows to ~7 sig digits — fatal
# when comparing AUC to 5 decimals (0.94924 vs 0.94920). Cast to DOUBLE.
_FLOAT_COLUMNS = (
    "expected_metric",
    "actual_metric",
    "repro_gap",
    "cost_usd",
    "wall_clock_s",
)
# created_at stores an ISO string; the column is TIMESTAMP — cast on the way in.
_TIMESTAMP_COLUMNS = ("created_at",)

# UC identifier gate (matches the regex style in agent/tools/uc_dataset_tools.py).
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_table(catalog: str, schema: str, name: str = "experiments") -> str:
    """Validate + backtick-quote a three-level UC name.

    Identifiers can't be bound as SQL params, so they're interpolated — gate
    them against an identifier regex so a stray quote/space in config can't
    break or inject the statement.
    """
    for part in (catalog, schema, name):
        if not part or not _IDENT_RE.match(part):
            raise ValueError(f"Unsafe UC identifier part: {part!r}")
    return f"`{catalog}`.`{schema}`.`{name}`"


def _placeholder(col: str) -> str:
    if col in _FLOAT_COLUMNS:
        return "CAST(? AS DOUBLE)"
    if col in _TIMESTAMP_COLUMNS:
        return "CAST(? AS TIMESTAMP)"
    return "?"


@contextlib.contextmanager
def _file_lock(path: Path):
    """Exclusive interprocess lock around the JSONL read-modify-write window.

    The fallback file is append+rewrite; without a lock a concurrent update can
    clobber an append. POSIX flock on a sidecar ``.lock`` serializes writers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


@dataclass(frozen=True)
class ExperimentRow:
    """One experiment, mirroring the ``experiments`` Delta table columns."""

    experiment_id: str
    task_id: str
    hypothesis: str
    method: str
    config: dict
    metric_name: str
    created_at: str | None = None
    session_id: str | None = None
    source_paper: str | None = None
    source_section: str | None = None
    expected_metric: float | None = None
    actual_metric: float | None = None
    repro_gap: float | None = None
    cost_usd: float | None = None
    wall_clock_s: float | None = None
    status: str = "proposed"
    parent_id: str | None = None
    mlflow_run_id: str | None = None
    artifacts: dict | None = None
    notes: str | None = None


# Column order used for SQL INSERT and for reading rows back positionally.
_COLUMNS = [f.name for f in fields(ExperimentRow)]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_config(config: dict) -> str:
    """Canonical JSON for config-equality comparisons (dedup)."""
    return json.dumps(config or {}, sort_keys=True, default=str)


# Substrings that mark a lower-is-better metric. Everything else (auc, accuracy,
# f1, ndcg, …) is higher-is-better. Used to orient repro_gap so a positive gap
# always means "underperformed expectation", for loss metrics too.
_LOWER_IS_BETTER_HINTS = (
    "loss", "error", "rmse", "mae", "mse", "perplexity", "logloss", "mape",
    "wer", "cer",
)


def metric_higher_is_better(metric_name: str | None) -> bool:
    """Best-effort metric direction inferred from its name."""
    if not metric_name:
        return True
    name = metric_name.lower()
    return not any(h in name for h in _LOWER_IS_BETTER_HINTS)


class ExperimentLedger:
    """DAO over the experiment ledger. SQL when a warehouse is configured,
    JSONL fallback otherwise."""

    def __init__(
        self,
        settings: DatabricksSettings | None = None,
        user_token: str | None = None,
        local_path: Path | None = None,
    ):
        self.settings = settings
        self.user_token = user_token
        self._use_sql = bool(settings and settings.warehouse_id)
        self.local_path = Path(local_path) if local_path else DEFAULT_LOCAL_PATH
        if self._use_sql:
            self.table = _quote_table(settings.uc_catalog, settings.uc_schema)
        else:
            self.table = None

    # ── public interface ────────────────────────────────────────────────

    def ensure_table(self) -> None:
        """Idempotent CREATE TABLE IF NOT EXISTS (SQL backend only)."""
        if not self._use_sql:
            return
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            experiment_id STRING,
            session_id STRING,
            created_at TIMESTAMP,
            task_id STRING,
            hypothesis STRING,
            source_paper STRING,
            source_section STRING,
            method STRING,
            config STRING,
            metric_name STRING,
            expected_metric DOUBLE,
            actual_metric DOUBLE,
            repro_gap DOUBLE,
            cost_usd DOUBLE,
            wall_clock_s DOUBLE,
            status STRING,
            parent_id STRING,
            mlflow_run_id STRING,
            artifacts STRING,
            notes STRING
        ) USING DELTA
        """
        self._execute(ddl)

    def propose(
        self,
        *,
        task_id: str,
        hypothesis: str,
        method: str,
        config: dict,
        metric_name: str,
        source_paper: str | None = None,
        source_section: str | None = None,
        expected_metric: float | None = None,
        parent_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Insert a new proposed experiment. Returns its experiment_id."""
        row = ExperimentRow(
            experiment_id=str(uuid.uuid4()),
            created_at=_now_iso(),
            session_id=session_id,
            task_id=task_id,
            hypothesis=hypothesis,
            source_paper=source_paper,
            source_section=source_section,
            method=method,
            config=config or {},
            metric_name=metric_name,
            expected_metric=expected_metric,
            status="proposed",
            parent_id=parent_id,
        )
        self._insert(row)
        return row.experiment_id

    def mark_running(self, experiment_id: str, mlflow_run_id: str | None = None) -> None:
        self._update(
            experiment_id,
            {"status": "running", "mlflow_run_id": mlflow_run_id},
        )

    def record_result(
        self,
        experiment_id: str,
        *,
        actual_metric: float,
        cost_usd: float | None = None,
        wall_clock_s: float | None = None,
        artifacts: dict | None = None,
        status: str = "done",
        notes: str | None = None,
    ) -> None:
        """Record an outcome and compute repro_gap = expected − actual."""
        existing = self._get(experiment_id)
        repro_gap = None
        if existing is not None and existing.expected_metric is not None:
            # Orient so positive = underperformed, regardless of metric direction.
            shortfall = existing.expected_metric - actual_metric
            repro_gap = (
                shortfall
                if metric_higher_is_better(existing.metric_name)
                else -shortfall
            )
        self._update(
            experiment_id,
            {
                "actual_metric": actual_metric,
                "repro_gap": repro_gap,
                "cost_usd": cost_usd,
                "wall_clock_s": wall_clock_s,
                "artifacts": artifacts,
                "status": status,
                "notes": notes,
            },
        )

    def reject(self, experiment_id: str, reason: str) -> None:
        self._update(experiment_id, {"status": "rejected", "notes": reason})

    def get(self, experiment_id: str) -> ExperimentRow | None:
        """Fetch a single experiment by id (None if absent)."""
        return self._get(experiment_id)

    def list_for_task(self, task_id: str) -> list[ExperimentRow]:
        if self._use_sql:
            return self._sql_query("WHERE task_id = ?", [task_id])
        return [r for r in self._jsonl_all() if r.task_id == task_id]

    def best_for_task(
        self, task_id: str, metric_name: str, higher_is_better: bool = True
    ) -> ExperimentRow | None:
        candidates = [
            r
            for r in self.list_for_task(task_id)
            if r.metric_name == metric_name and r.actual_metric is not None
        ]
        if not candidates:
            return None
        return (max if higher_is_better else min)(
            candidates, key=lambda r: r.actual_metric
        )

    def find_similar_config(
        self, task_id: str, method: str, config: dict
    ) -> ExperimentRow | None:
        """Dedup: first live row matching (task_id, method, config).

        Matches proposed / running / done rows (avoid duplicating in-flight or
        completed work) but skips failed / rejected rows so a transient failure
        doesn't block a legitimate re-run of the same config.
        """
        target = _normalize_config(config)
        for r in self.list_for_task(task_id):
            if r.status in ("failed", "rejected"):
                continue
            if r.method == method and _normalize_config(r.config) == target:
                return r
        return None

    # ── backend dispatch ────────────────────────────────────────────────

    def _insert(self, row: ExperimentRow) -> None:
        if self._use_sql:
            self._sql_insert(row)
        else:
            self._jsonl_append(row)

    def _update(self, experiment_id: str, changes: dict) -> None:
        if self._use_sql:
            self._sql_update(experiment_id, changes)
        else:
            self._jsonl_update(experiment_id, changes)

    def _get(self, experiment_id: str) -> ExperimentRow | None:
        if self._use_sql:
            rows = self._sql_query("WHERE experiment_id = ?", [experiment_id])
            return rows[0] if rows else None
        for r in self._jsonl_all():
            if r.experiment_id == experiment_id:
                return r
        return None

    # ── SQL backend ─────────────────────────────────────────────────────

    def _connect(self):
        from agent.core import db_client

        return db_client.get_sql_connection(self.settings, user_token=self.user_token)

    def _execute(self, sql: str, params: list | None = None) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                if params:
                    cur.execute(sql, params)
                else:
                    cur.execute(sql)
        finally:
            conn.close()

    def _sql_insert(self, row: ExperimentRow) -> None:
        values = _row_to_storage(row)
        placeholders = ", ".join(_placeholder(c) for c in _COLUMNS)
        sql = (
            f"INSERT INTO {self.table} ({', '.join(_COLUMNS)}) "
            f"VALUES ({placeholders})"
        )
        self._execute(sql, [values[c] for c in _COLUMNS])

    def _sql_update(self, experiment_id: str, changes: dict) -> None:
        storage = _changes_to_storage(changes)
        set_clause = ", ".join(f"{col} = {_placeholder(col)}" for col in storage)
        sql = f"UPDATE {self.table} SET {set_clause} WHERE experiment_id = ?"
        params = list(storage.values()) + [experiment_id]
        self._execute(sql, params)

    def _sql_query(self, where: str = "", params: list | None = None) -> list[ExperimentRow]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = f"SELECT {', '.join(_COLUMNS)} FROM {self.table} {where}".strip()
                if params:
                    cur.execute(sql, params)
                else:
                    cur.execute(sql)
                cols = [d[0] for d in cur.description]
                return [_row_from_storage(dict(zip(cols, r))) for r in cur.fetchall()]
        finally:
            conn.close()

    # ── JSONL backend ───────────────────────────────────────────────────

    def _jsonl_append(self, row: ExperimentRow) -> None:
        with _file_lock(self.local_path):
            with self.local_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(row), default=str) + "\n")

    def _jsonl_all(self) -> list[ExperimentRow]:
        if not self.local_path.exists():
            return []
        rows: list[ExperimentRow] = []
        with self.local_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(ExperimentRow(**json.loads(line)))
        return rows

    def _jsonl_update(self, experiment_id: str, changes: dict) -> None:
        with _file_lock(self.local_path):
            rows = self._jsonl_all()
            found = False
            for i, r in enumerate(rows):
                if r.experiment_id == experiment_id:
                    rows[i] = ExperimentRow(**{**asdict(r), **changes})
                    found = True
                    break
            if not found:
                logger.warning(
                    "update for unknown experiment_id %s ignored", experiment_id
                )
                return
            tmp = self.local_path.with_suffix(self.local_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(asdict(r), default=str) + "\n")
            tmp.replace(self.local_path)


# ── storage (de)serialization for the SQL backend ───────────────────────


def _double_str(v) -> str | None:
    """Full-precision string for a DOUBLE param.

    The databricks-sql connector binds a bare Python float as a 32-bit FLOAT,
    so ``CAST(? AS DOUBLE)`` on a float value casts an already-narrowed number
    (AUC loses precision past ~7 digits). Binding the round-trippable string
    instead lets the CAST parse full double precision. None stays NULL.
    """
    return None if v is None else repr(float(v))


def _row_to_storage(row: ExperimentRow) -> dict:
    d = asdict(row)
    for col in _JSON_COLUMNS:
        d[col] = json.dumps(d[col], default=str) if d[col] is not None else None
    for col in _FLOAT_COLUMNS:
        d[col] = _double_str(d.get(col))
    return d


def _changes_to_storage(changes: dict) -> dict:
    out = dict(changes)
    for col in _JSON_COLUMNS:
        if col in out and out[col] is not None:
            out[col] = json.dumps(out[col], default=str)
    for col in _FLOAT_COLUMNS:
        if col in out:
            out[col] = _double_str(out[col])
    return out


def _row_from_storage(d: dict) -> ExperimentRow:
    for col in _JSON_COLUMNS:
        v = d.get(col)
        if isinstance(v, str):
            d[col] = json.loads(v)
    if d.get("created_at") is not None:
        d["created_at"] = str(d["created_at"])
    return ExperimentRow(**{k: d.get(k) for k in _COLUMNS})
