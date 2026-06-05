# plan.md — ml-intern → Auto-Researcher

Action plan derived from two reviews:
1. **Capability review** — engineering is strong; the agent escalates complexity and can't tell when it's done; **there is no way to measure the agent itself** (eval = placeholder).
2. **Auto-researcher review** — discovery + finetune tools are real (9/10), but the research→implement→eval→iterate **loop is prose in the system prompt, not code** (3/10). Missing: persistence, reproduction-gap gate, eval-driven iteration, parallel hypothesis fan-out, novelty/dedup.

Goal: turn ml-intern from a capable one-shot ML engineer into a **measurable, self-iterating auto-researcher** that works across tabular, LLM finetune, and general ML tasks.

---

## Guiding principles

- **Can't improve what you can't measure.** The eval harness is the spine — every other phase is validated against it.
- **Evidence, not dogma.** Skills/playbooks emit testable hypotheses, not laws. The agent ablates before escalating.
- **The trace is the dataset.** Persist every experiment (config, metric, cost, repro-gap) so the agent reasons over its own history and across sessions.
- **Breadth over patience.** Parallel hypothesis fan-out on Databricks Jobs is the agent's edge over a human — use it.
- **Fail-soft, Databricks-native.** Reuse `db_client`, MLflow, UC, cost_estimation. No new infra primitives. Local fallback for offline/unit tests, mirroring `telemetry.py`.

---

## Architecture overview

```
papers_tool / research_tool  ──┐
                               ├─►  Hypothesis Generator  ──►  Experiment Ledger (UC Delta + local JSONL)
github / docs tools          ──┘            │                          ▲
                                            ▼                          │
                                  Sweep Op (parallel Jobs)  ──►  Eval Harness / Scorers
                                            │                          │
                                            ▼                          │
                                  Reproduction-Gap Gate  ──────────────┘
                                            │
                                            ▼
                                  Critic / Verifier (leakage, target, overfit)
```

---

## Phase 0 — Experiment Ledger  *(FOUNDATION — build first)*

**Why:** persistence is the missing spine. Findings currently die at session end. Everything downstream (gap-gate, dedup, eval-driven iteration, cross-session learning) needs a structured, durable record.

**Deliverables**
- `agent/core/experiment_ledger.py` — `ExperimentLedger` DAO. SQL backend over UC Delta via `db_client.get_sql_connection`; **local JSONL fallback** when no warehouse (offline + unit tests).
- `resources/experiment_ledger.sql` — `CREATE TABLE IF NOT EXISTS` DDL (documented; DAO also self-creates).
- `tests/unit/test_experiment_ledger.py` — full DAO coverage against the JSONL backend (no live workspace).

**Table:** `{catalog}.{schema}.experiments` (Delta)

| column | type | notes |
|---|---|---|
| experiment_id | STRING | uuid, PK |
| session_id | STRING | nullable |
| created_at | TIMESTAMP | |
| task_id | STRING | eval task or user-task id |
| hypothesis | STRING | one-line testable claim |
| source_paper | STRING | arxiv id/url, nullable |
| source_section | STRING | nullable |
| method | STRING | technique name |
| config | STRING | JSON of hyperparams / feature set |
| metric_name | STRING | auc / accuracy / eval_loss / rank_pct |
| expected_metric | DOUBLE | paper-reported or prior-best, nullable |
| actual_metric | DOUBLE | nullable until done |
| repro_gap | DOUBLE | expected − actual, computed on record |
| cost_usd | DOUBLE | nullable |
| wall_clock_s | DOUBLE | nullable |
| status | STRING | proposed / running / done / failed / rejected |
| parent_id | STRING | lineage, nullable |
| mlflow_run_id | STRING | nullable |
| artifacts | STRING | JSON paths, nullable |
| notes | STRING | nullable |

**DAO interface (frozen — downstream codes against this):**
```python
class ExperimentLedger:
    def __init__(self, settings: DatabricksSettings | None = None,
                 user_token: str | None = None,
                 local_path: Path | None = None): ...
    def ensure_table(self) -> None              # idempotent
    def propose(self, *, task_id, hypothesis, method, config: dict,
                metric_name, source_paper=None, source_section=None,
                expected_metric=None, parent_id=None,
                session_id=None) -> str          # returns experiment_id
    def mark_running(self, experiment_id, mlflow_run_id: str | None = None) -> None
    def record_result(self, experiment_id, *, actual_metric: float,
                       cost_usd=None, wall_clock_s=None, artifacts: dict | None = None,
                       status: str = "done", notes: str | None = None) -> None  # computes repro_gap
    def reject(self, experiment_id, reason: str) -> None
    def list_for_task(self, task_id) -> list[ExperimentRow]
    def best_for_task(self, task_id, metric_name, higher_is_better=True) -> ExperimentRow | None
    def find_similar_config(self, task_id, method, config: dict) -> ExperimentRow | None  # dedup
```
- Backend selection: SQL when `settings.warehouse_id` present, else JSONL at `local_path` (default `session_logs/experiments.jsonl`).
- Use parameterized SQL. Delta supports INSERT + UPDATE (MERGE for upsert).
- `ExperimentRow` = frozen dataclass mirroring the table.

**Verify:** `uv run pytest tests/unit/test_experiment_ledger.py`.

---

## Phase 1 — Eval Harness  *(FOUNDATION — build in parallel with Phase 0)*

**Why:** the #1 critical gap. No way to know if a prompt/model/playbook change made the agent better or worse. Also the auto-researcher's scorer to close the loop.

**Deliverables**
- `evals/__init__.py`
- `evals/task_spec.py` — load + validate task yaml (`EvalTask` dataclass).
- `evals/scorers.py` — `roc_auc`, `accuracy`, `rank_percentile` (vs leaderboard), `eval_loss`. Pure functions, deterministic.
- `evals/runner.py` — `run_eval(task, agent_callable, ledger)` → drives a task, scores the result, records to ledger, returns `EvalResult`.
- `evals/report.py` — scorecard writer (markdown + json): final score, LB percentile, # iterations, cost_usd, wall_clock_s, # self-recovered failures.
- `evals/tasks/kaggle-f1-pitstops-s6e5.yaml` — first real task, referencing `examples/kaggle-f1-pitstops-s6e5`.
- `tests/unit/test_eval_harness.py` — task-spec parse, each scorer on fixtures, runner with a stub agent_callable + JSONL ledger.

**Task yaml schema**
```yaml
id: kaggle-f1-pitstops-s6e5
kind: tabular            # tabular | finetune | nlp | cv
metric: roc_auc
higher_is_better: true
baseline_score: 0.94820  # agent v1 one-shot
human_ceiling: 0.94924   # v5.2
leaderboard:
  top_public: 0.9545
  proxy: rank_percentile
data:
  train: ...
  test: ...
holdout: { type: temporal, column: Year, value: 2025 }
budget: { max_cost_usd: 25.0, max_iterations: 30 }
```

**Scorer must report, per run:** absolute score, **CV↔LB gap**, leaderboard percentile, cost, wall-clock, iterations. (The CV↔LB gap is the distribution-shift signal the Kaggle demo proved decisive.)

**Verify:** `uv run pytest tests/unit/test_eval_harness.py`. Offline (no workspace) using the example's stored predictions as fixtures.

---

## Phase 2 — Reproduction-Gap Gate + Sweep Op  *(after 0 & 1)*

- `agent/core/repro_gate.py` — after `record_result`, compute `repro_gap`; if `gap > threshold`, **block escalation** and emit a corrective directive ("re-read paper §X; check data format / LR schedule / missing trick") instead of "add more complexity." Reuses the `doom_loop` corrective-prompt injection mechanism.
- Sweep op in `databricks_jobs_tool.py` (new `kind="sweep"` or wrapper): agent emits N configs → submit N Jobs concurrently → collect to ledger → critic ranks on holdout. Budget-capped via `cost_estimation` against the ledger running total.
- New tool: `experiment` (propose / record / query ledger) so the agent reads its own history.
- Tests: gap-gate thresholds, sweep fan-out (mocked jobs), budget cap enforcement.

## Phase 3 — Hypothesis Generator + Critic  *(after 2)*

- `agent/core/hypothesis.py` — turn `papers_tool` citation-graph snippets into **testable ledger rows** (method + expected_metric + config), ranked by expected-lift / cost. Reuse `research-companion` brainstormer/idea-critic agents as generator + critic rather than rebuild.
- Verifier/critic pass: audit each win for **leakage, target confusion, overfit (CV↑LB↓), correlation-floor blends (spearman>0.998)**. These are the exact failure classes the Kaggle demo logged.
- Dedup: `find_similar_config` blocks re-running tried configs.

## Phase 4 — Wire the loop + skills as evidence  *(after 3)*

- Implement the loop the system prompt only describes: `generate → rank → fan out top-k → eval → gap-gate → ledger → regenerate`, budget-bounded.
- Rewrite `kaggle-tabular-classification/playbook.md` phases to emit evidence ("pseudo-labeling MAY help — ablate, keep only on ≥0.001 gain"), not laws.
- Add finetune + NLP/CV playbooks on the same evidence-first skeleton.
- Self-improving skill loop: `/retro` proposes new skill entries; **harness validates the lift before keeping** (gate strictly — no memorizing noise).

## Phase 5 — Observability + reliability  *(parallelizable, ongoing)*

- Trace LLM payloads (per-turn token/cost), per-tool latency, effort-probe cascade, compaction cost. The trace becomes the eval dataset.
- Lakeview cost dashboard (system tables exist, nothing reads them).
- Reliability fixes for long unattended runs: compaction-failure fallback (don't terminate), effort-probe mid-convo drift guard, serverless-GPU `NotImplementedError`, orphaned-cluster cleanup, per-tool timeout.

---

## Build process — multi-agent

- **Architect/supervisor-in-chief (main session):** owns this plan + the frozen interfaces; integrates; runs tests; final gate.
- **Expert agents (parallel implementers):** one module each, non-overlapping files.
- **Supervisor/reviewer agents:** cross-check each module against spec + repo conventions + `REVIEW.md` severities (P0/P1/P2).
- **Codex (`/codex`):** independent cross-check on the persistent/critical surfaces (ledger schema + DAO, harness contract).

**This session builds Phase 0 + Phase 1** (the dependency root, independent of each other). Phases 2–5 follow once the foundation is green.

---

## Status

- [x] Phase 0 — Experiment Ledger (`agent/core/experiment_ledger.py`, `resources/experiment_ledger.sql`, 16 tests). Reviewed by 1 supervisor agent + codex; P1s applied (backtick-quoted+validated UC identifier, JSONL interprocess lock, `with conn.cursor()`, WHERE-pushdown instead of full scans, float→DOUBLE casts for AUC precision).
- [x] Phase 1 — Eval Harness (`evals/` task_spec/scorers/runner/report + kaggle-f1 task, 16 tests). Reviewed by 1 supervisor agent; `evals*` packaged (sys.path hack removed), yaml placeholders marked honest.
- [x] Phase 2 — Repro-gap gate + experiment tool + sweep orchestrator (`agent/core/repro_gate.py`, `agent/tools/experiment_tool.py`, `agent/core/sweep.py`; registered in `tools.py`; loop discipline + reproduce-first rule in `system_prompt_v3.yaml`). Reviewed by 1 supervisor agent; P0 fixed (loss-like metrics inverted the gap — now metric direction inferred from `metric_name`, `repro_gap` oriented so positive=underperformed) + P1 fixed (dedup skips failed/rejected rows so transient failures don't block retry). 39 Phase-2 tests + loss-orientation tests.
  - [x] **Sweep→Jobs wired + live-verified.** `sweep.run_sweep_async` (concurrent fan-out via gather) + `agent/tools/sweep_jobs.py` adapter reusing `DatabricksJobsTool` async building blocks (stage/build/wait/fetch). Metric channel = stdout sentinel `SWEEP_METRIC=<float>` (probed: survives `runs/get-output` on serverless `spark_python_task`). `tests/integration/test_sweep_jobs_live.py` ran a real 2-config serverless sweep on the lakebase workspace (100s): parallel submit, metric parse, SQL-ledger record, ranking, dedup re-run skip — all green.
- [ ] Phase 3 — Hypothesis generator + critic
- [ ] Phase 4 — Wire loop + evidence-first skills
- [x] Phase 5 — Observability + reliability (PR #21: tracing LLM/tool spans wired into the loop; compaction aggressive-retry; effort-probe no silent drift)
- [x] Phase 6 — Deterministic loop runner (`agent/core/research_loop.py` + `research_loop` tool). Chains the primitives with explicit control flow (LLM out of the driver's seat): per-round generate→dedup→budget-clamp→sweep→reproduce-gate→accept→stop. Stop precedence budget>target>max_rounds>patience>exhausted. Codex reviewed the design (2 P0 + 3 P1 folded in: round-only cost accounting, pre-submission budget clamp via est_cost_per_variant, loop-local seen-set, parenthesized improvement check, pre-round stops, pluggable verify_fn). 16 unit tests + live 2-round loop over real serverless jobs (191s).

### Build log
- Multi-agent: 2 expert implementers (parallel, non-overlapping files) → 2 supervisor reviewers → codex independent cross-check on the persistence spine → architect integrated all P1 fixes. Both reviewers returned 0 P0 / ready-to-merge; codex caught the float→FLOAT-narrowing precision bug the reviewers missed.
- Known pre-existing failures (NOT from this work): `tests/unit/test_model_catalog.py` 3 failures in `llm_params` effort handling.

---

## Parked — upstream port backlog (huggingface/ml-intern → fork)

Triaged 2026-06-06 against upstream `main` (fork base `2a2e170`). Most upstream
churn is HF-infra (HF Router/FAL/credits/quota/Hub/Spaces/Trackio/OAuth) — N/A to
this fork. Big-ticket provider-agnostic fixes already landed here (compaction
infinite-loop + oversized-truncation #213, drop LLM-msg timestamps #209, CLI help
align #248, stale tool error badges #247, `/resume` #233, session YOLO budget #201).
**Three items worth porting**, ranked:

1. **[security, S] claude-review.yml prompt-injection (#231 `fbc10a2`).** Our CI
   still `cat`s `REVIEW.md` straight into the reviewer prompt as authoritative
   instructions — a malicious PR editing `REVIEW.md` hijacks the auto-reviewer.
   Fix: move REVIEW.md *below* base instructions, label untrusted, sanitize
   backticks, `head -n 100`. ~16 lines. **Do first.**
2. **[correctness, S] session-capacity race (subset of #277 `6cf406a`).**
   `backend/session_manager.py` check-then-create race overshoots `MAX_SESSIONS`
   under concurrent creates (blocking constructors run outside the lock before the
   count is incremented). Fix: `_pending_creates` counter reserved under lock;
   capacity check uses `active + pending`. Port *only* this — rest of #277 is
   HF-sandbox/Mongo coupled.
3. **[UX, M] `/clear` + `/new` CLI commands (#256 `021580f`).** Fresh conversation,
   keep warm sandbox/model cache. One adaptation: HF dataset-upload detach →
   Lakebase/UC session persistence.
