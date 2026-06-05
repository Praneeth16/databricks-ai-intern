-- Experiment Ledger — UC Delta table backing agent/core/experiment_ledger.py.
--
-- The persistence spine of the auto-researcher loop: one row per experiment
-- (proposed → running → done/failed/rejected), with the reproduction gap and
-- cost/wall-clock needed by the downstream gap-gate, dedup, and eval-driven
-- iteration phases.
--
-- The DAO self-creates this table via ensure_table() (idempotent), so this
-- file is the documented source of truth — keep the two in sync.
--
-- Replace {{catalog}}.{{schema}} with the configured UC binding
-- (defaults: databricks_ai_intern.agent). The DAO resolves it as
-- {settings.uc_catalog}.{settings.uc_schema}.experiments.

CREATE TABLE IF NOT EXISTS {{catalog}}.{{schema}}.experiments (
    experiment_id    STRING,      -- uuid, PK
    session_id       STRING,      -- nullable
    created_at       TIMESTAMP,
    task_id          STRING,      -- eval task or user-task id
    hypothesis       STRING,      -- one-line testable claim
    source_paper     STRING,      -- arxiv id/url, nullable
    source_section   STRING,      -- nullable
    method           STRING,      -- technique name
    config           STRING,      -- JSON of hyperparams / feature set
    metric_name      STRING,      -- auc / accuracy / eval_loss / rank_pct
    expected_metric  DOUBLE,      -- paper-reported or prior-best, nullable
    actual_metric    DOUBLE,      -- nullable until done
    repro_gap        DOUBLE,      -- expected - actual, computed on record
    cost_usd         DOUBLE,      -- nullable
    wall_clock_s     DOUBLE,      -- nullable
    status           STRING,      -- proposed / running / done / failed / rejected
    parent_id        STRING,      -- lineage, nullable
    mlflow_run_id    STRING,      -- nullable
    artifacts        STRING,      -- JSON paths, nullable
    notes            STRING       -- nullable
) USING DELTA;
