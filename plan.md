# plan.md — databricks-ai-intern → Auto-Researcher

Action plan derived from two reviews:
1. **Capability review** — engineering is strong; the agent escalates complexity and can't tell when it's done; **there is no way to measure the agent itself** (eval = placeholder).
2. **Auto-researcher review** — discovery + finetune tools are real (9/10), but the research→implement→eval→iterate **loop is prose in the system prompt, not code** (3/10). Missing: persistence, reproduction-gap gate, eval-driven iteration, parallel hypothesis fan-out, novelty/dedup.

Goal: turn databricks-ai-intern from a capable one-shot ML engineer into a **measurable, self-iterating auto-researcher** that works across tabular, LLM finetune, and general ML tasks.

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
- [~] Phase 7 — Custom LLM Serving + deployment strategy. **Build-step 2 done:** `agent/core/serving_strategy.py` (pure size-driven VRAM/TP/precision math + `render_entrypoint` baking the opencv-FIPS/fork/`bash -lc` fixes) + **24 unit tests**. Math reproduces both FE verdicts (Qwen3-4B fits 1×A10 TP=1; Qwen3.5-27B-FP8 overflows one A10 → escalate to single H100, fallback TP=4 on A10×4). Reference example NOT vendored (knowledge in this plan; agent generates entrypoints at runtime). **Codex review (1 P0 + 3 P1) applied:** (P0) quantized precisions are no longer free runtime flags — `quant_source` ∈ native/online/artifact/build; AWQ/GPTQ/int8 require an existing artifact or `allow_quantization_build`, only fp8 is online-capable (vLLM dynamic), `--quantization` emitted online-only (native/artifact auto-detected); (P1) explicit empty capability → no configs (was "all GPUs"); (P1) `est_max_concurrent_seqs` from KV budget + `max_num_seqs` clamp; (P1) T4 default-excluded (opt-in), explicit `objective` ∈ balanced/cost_first/accuracy_first/latency_first. Also: TP head-divisibility gate (`num_kv_heads % tp`), `shlex.quote` on entrypoint paths. **Deferred refinements (low-risk):** MoE total-vs-active params, GiB-vs-GB, quantized-KV cache. **Build-step 3 done (codex-planned):** `agent/tools/model_serving_tool.py` — spine ops `plan_deployment` (→ feasible set + canonical plan objects with `plan_hash`), `deploy` (REST `wc.api_client.do`, fixed ×4 concurrency, no autoscale; create/update; readiness poll), `query` (429 backoff), `list`/`delete`/`probe_serving` (confidence-rated), `_record_deployment` (ExperimentRow). plan_hash contract enforced (`_validate_plan` recomputes; tamper → error). Registered in `tools.py`; `_needs_approval` gates build/deploy/benchmark/delete. **16 tool tests** (registration, approval, plan_hash tamper, deploy golden body, create/update, 429 retry, ledger row) — 40 serving tests total, full unit suite 437 pass (3 pre-existing model_catalog failures unrelated). **Build-step 4 done:** `build_and_register` — renders a serverless-GPU build notebook (download weights → validate artifact + manifest hash → local vLLM TP=1 smoke → log placeholder `ChatModel` with `{task, entrypoint, plan_hash, manifest}` metadata → `register_model(env_pack="databricks_model_serving")` → base64 result sentinel), submitted via the `sweep_jobs` reuse path (`_resolve_or_stage_script`/`_build_submit_body`/`_wait_for_run`/`_fetch_run_output`); recomputes plan_hash + requires smoke_passed before trusting; build accelerator auto-sized (GPU_1xA10/H100); quant_source="build" fails closed without calibration; honest TP>1 smoke caveat. `benchmark` — threaded concurrency sweep over `/invocations` (sys_tps, p50/p95 lat, http429; TTFT/TPOT noted as streaming-only future), records peak tok/s to the ledger. **+7 tests (23 tool tests total, 47 serving, full suite 444 pass).** **Next:** live-validate against `fe-vm-lakebase-praneeth` (the build/register/deploy round-trip — unit mocks can't prove the SOD env_pack or vLLM startup); then skill playbook + Lakeview panel.
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

---

## Improvement backlog — SOTA-shaped search loop  *(codex consult + web research, 2026-06-08)*

Diagnosis (codex + MLE-STAR/AIDE/Operand-Quant survey): **execution is solid, the
search policy is flat.** The loop consumes whatever `hypothesis_source` emits,
sweeps, keeps best — no tree, no ablation. MLE-STAR (64% MLE-bench-Lite medal vs
AIDE ~17%) gets its edge from *ablation-guided targeted refinement*, not breadth.
All file claims below verified against the repo.

**Tier 1 — highest leverage**

1. **[search policy, M] Ablation-guided refinement.** Add `agent/core/search_policy.py`:
   seed baseline → ablate named blocks (val-split, feature block, model family,
   hyperparams, ensemble) → pick the highest-marginal-lift block → generate children
   *only* in that block → repeat to budget/patience. Reuses ledger `parent_id`
   (already in schema, line 114/193). This is the MLE-STAR advantage without a
   fragile general planner. **Single highest-ROI unlock.**
2. **[hypothesis gen, M] Wire Phase 3 as structured extraction, not prose.**
   `agent/core/hypothesis.py` (`generate_hypotheses`) exists but is referenced by
   *nothing but itself* — stranded. Add a `hypothesis_generate` tool: structured
   `Finding[]` → `generate_hypotheses` → dedup vs ledger → sweep-ready rows. Make
   `research_tool` optionally emit JSON findings. (Second to #1 — good seeds still
   need targeted refinement.)
3. **[ledger, S] Flat rows → experiment lineage.** Schema stores `parent_id`,
   source paper/section, expected/actual metric, artifacts, MLflow run id — but the
   public API is only `list_for_task` / `best_for_task` / exact-config dedup. Add
   `children()`, `lineage()`, `best_by_block()`, `find_similar_across_tasks(dataset_fingerprint,…)`.
   The last = **cross-session learning** (today each task restarts cold).

**Tier 2 — make "numbers go up" real + the showcase land**

4. **[eval, M] Real eval + leaderboard stop.** `scripts/eval_model.py` literally
   writes `eval=placeholder`; the Kaggle task yaml has placeholder data paths. Wire
   `research_loop` to `EvalTask` (metric direction, baseline, human ceiling, LB
   percentile, max iters/cost). "Numbers go up" = *rank improved under budget*, not
   "stdout float rose."
5. **[showcase, M] Make the Databricks moat visible.** The differentiator AIDE/MLE-STAR
   *cannot* copy: governed, observable, workspace-native autonomous experimentation
   (UC Delta ledger + Jobs fanout + UC Volumes + MLflow traces + Model Registry +
   system billing). But `resources/lakeview_dashboard.yml.disabled` is OFF and Jobs
   carry **no billing tags**. Enable Lakeview, tag jobs, surface: hypothesis tree,
   metric-over-time, repro gaps, spend, accept/reject decisions, model lineage.

**Tier 3 — cheap robustness**

6. **[robustness, S] Sweep scoring durability.** Metric contract depends on Databricks
   stdout capture. Write `/Volumes/…/sweep_results/<id>.json` + `mlflow.log_metric`;
   stdout sentinel = fallback only; reject conflicting sentinels; record metric source.
7. **[robustness, S] Budget = launched compute, not scored results.** `sweep.py`
   submits all jobs then budget-gates *scoring* — money already spent. Require cost
   estimates, add `max_concurrent`, cancel unscored jobs after the cap is hit.

**Cut / defer.** Ouroboros/Darwin-Gödel self-rewriting (optimizes vibes without a
real eval harness — defer behind #4). Multi-agent pipeline gen (Jobs already give
reliable parallelism; the gap is search policy, not more agents). More prose
playbook rules — convert to executable checks or don't add.

---

## Phase 7 — Custom LLM Serving + Autonomous Deployment Strategy  *(net-new; large team unlock)*

**Why.** Databricks recently shipped **Custom LLM Serving** (vLLM-backed, OpenAI-compatible
`/invocations`) — host any HF / fine-tuned / PEFT / multimodal model not in the
Foundation Model API. Teams that fine-tune today get stuck on *hosting*: which GPU,
what precision, will it fit, what does it cost. The intern should **figure this out**
— pick deployment size + precision/quantization + scaling from workspace capability,
model facts, task, and the team's accuracy/latency/cost priorities. **Not a lookup
table — a reasoned decision over a feasible set computed by deterministic math.**
Mirrors the existing split: mechanism is code, judgment is the agent's.

> **FE reference code — GROUND TRUTH (explored 2026-06-08).** A working FE-team
> implementation (`custom_llm_serving.zip`, file `F0B7Y3RLS10`, channel `sme-ai-apj`)
> was unzipped + read: `README.md`, `REPORT.md`, `create_endpoint.py`, `benchmark.py`,
> `notebooks/serve_qwen3_4b.py`, `notebooks/serve_qwen35_27b_fp8.py`. **Action: vendor
> into `examples/custom-llm-serving/`** as the canonical reference. The mechanics below
> are corrected against it — the public docs were materially incomplete (they describe a
> `workload_size`/`scale_to_zero` autoscaling path that the **entrypoint-based** deploy
> actually rejects).

### Native surface (corrected against FE code — this is the **SOD / entrypoint** path)
- **Endpoint** = UC registered model, but the MLflow artifact is a **placeholder** `ChatModel`
  (predict returns `{}`). Serving runs the **`entrypoint` command string** from MLflow
  `metadata={"task":"llm/v1/chat","entrypoint": "<vllm launch cmd>"}` — *not* `python_model.predict`.
  Registered with **`mlflow.register_model(..., env_pack="databricks_model_serving")`** (builds
  the **Serverless Optimized Deployment**). MLflow ≥ 3.12, port **8080**.
- **The entrypoint string is the entire adaptation surface** — GPU/TP sizing, dtype, dep quirks,
  worker-process env all live there (`bash -lc '...; exec python -m vllm.entrypoints.openai.api_server …'`).
- **`workload_type`** (from FE, richer than docs): `GPU_MEDIUM` (1×A10, 24 GB) ·
  **`MULTIGPU_MEDIUM` (4×A10, 96 GB → `--tensor-parallel-size 4`)** · (docs also list `GPU_SMALL`
  1×T4 / `GPU_XLARGE` 1×H100, **available to us**; serverless-GPU **build jobs** expose
  `GPU_1xH100`). **Selection is size-driven, smallest-GPU-that-fits:** model fits A10 → stay on
  A10 (cheapest, single-GPU); too big for A10 → escalate to a single **H100 (GPU_XLARGE)**; too
  big for one H100 → fall back to **A10×4 tensor-parallel (`MULTIGPU_MEDIUM`)**. H100 is the
  escalation step for big models, **not a default**. (The FE notebooks used A10×4 only because
  H100 wasn't in their workspace — a workspace constraint, not the canonical pattern; the
  notebooks are reference for the *mechanics*, not the deploy default.)
- **NO autoscaling for entrypoint endpoints.** Config (via **REST** `/api/2.0/serving-endpoints`,
  *not* the SDK's autoscaling-default `EndpointCoreConfigInput`):
  `min_provisioned_concurrency == max_provisioned_concurrency`, **multiple of 4 (min 4)**,
  `scale_to_zero_enabled=False`. Fixed, always-on capacity → size to peak, budget continuous GPU cost.
- **`provisioned_concurrency` is an ADMISSION CEILING, not the batch size.** vLLM continuous-batches
  far beyond it (4B: batched ~32 concurrent at provisioned=4); exceeding it → **HTTP 429
  "Too many parallel requests"** → clients need retry/backoff. Scale by raising in multiples of 4.
- **vLLM entrypoint args** (seen in FE): `--dtype`, `--max-model-len`, `--gpu-memory-utilization`,
  `--tensor-parallel-size`, `--max-num-seqs` (batch cap). FP8 model uses `--dtype bfloat16` (compute
  dtype; weights FP8 on disk).
- **Observability**: live vLLM logs, Prometheus `/metrics`, logs+metrics to UC Delta. **Caveat: logs
  API serves only the ACTIVE config** — a fully-failed deploy returns "does not exist with config
  version 0"; read Serving-UI service logs or poll *during* the crash-loop.

### Build/serve split + non-obvious requirements (from FE REPORT.md — encode as intern knowledge)
- **Build/register MUST run on a GPU job** (serverless GPU / AI Runtime) — registering on plain
  (non-GPU) serverless yields a non-SOD artifact the endpoint rejects ("...only supported with
  Serverless Optimized Deployments"). Big models: build on **1×H100** (fast TP=1 load test) → **serve
  on A10×4 TP=4**.
- **Non-network-restricted workspace** required for register (serverless can't fetch UC temp storage
  creds when restricted → `PERMISSION_DENIED`).
- **Gotcha table (each cost FE real debugging):**
  1. **opencv FIPS crash** — vLLM 0.19.1 → `import gguf` (FP8 path) → opencv's bundled libcrypto fails
     FIPS self-test, aborts model server (exitCode=1). **Fix: `pip uninstall -y opencv-python-headless
     opencv-python` *inside the entrypoint*** before launching vLLM (text inference needs no cv2).
  2. **TP>1 worker crash** — `libstdc++ CXXABI not found` on EngineCore workers. **Fix:
     `VLLM_WORKER_MULTIPROC_METHOD=fork`** in the entrypoint.
  3. **Shell wrap** — entrypoint must be `bash -lc '...'` so the uninstall + env-vars run before
     `exec python` (robust whether runtime argv-execs or shell-execs).
  4. **Version coupling** — vLLM/transformers pins track model architecture (4B: vllm 0.11.2 / tf
     4.57.6; Qwen3.5-27B: vllm 0.19.1 / tf 5.5.4). New arch ⇒ bump runtime; carry pins in
     `extra_pip_requirements`.

### Design — deterministic mechanism vs agent judgment

**A. `agent/core/serving_strategy.py`** — pure, tested, no Databricks contact:
- `estimate_vram(params, dtype_bytes, max_model_len, kv_dtype, concurrency, hidden, layers, tp)` → GB
  *per GPU*. Weights = `params × bytes/param` (fp16/bf16=2, fp8/int8=1, int4=0.5), **divided by `tp`**;
  KV cache ≈ `2 × layers × max_model_len × hidden × kv_bytes × concurrency / tp`; + ~15–20 %
  activation/fragmentation overhead; respect `--gpu-memory-utilization` headroom.
- `feasible_configs(model_facts, workspace_caps, accuracy_budget, latency_budget, cost_budget)`
  → **ranked list**, each entry = `(workload_type, tensor_parallel_size, precision/quant,
  max_model_len, gpu_mem_util, max_num_seqs, provisioned_concurrency)` that *fit per-GPU after TP
  split*, annotated: fits?, per-GPU VRAM headroom, expected quality-delta band, est $/hr (always-on
  — no scale-to-zero), cold-start risk. **`provisioned_concurrency` snapped to a multiple of 4.**
  When a model doesn't fit one GPU, emit a **tensor-parallel** option (e.g. `MULTIGPU_MEDIUM` TP=4)
  rather than only escalating single-GPU size.
- `render_entrypoint(cfg, artifacts_path, served_name)` → the `bash -lc '…'` string, auto-including
  the opencv-uninstall + `VLLM_WORKER_MULTIPROC_METHOD=fork` (when TP>1) fixes. This is the exact
  contract Serving executes.
- **Precision ⨯ hardware coupling** (data table the planner reads, NOT agent prose):
  - **H100 / GPU_XLARGE (Hopper):** fp16/bf16 baseline; **fp8 (W8A8)** ~2× throughput, <1 %
    loss; hosts ~13B fp16 / ~70B fp8.
  - **A10 / GPU_MEDIUM (Ampere):** fp16 ≤ ~7B; **AWQ / GPTQ-Marlin int4** for ~13B; **no fp8**.
  - **T4 / GPU_SMALL (Turing):** small only; AWQ/GPTQ int4 ≤ ~7B; no fp8/bf16 (fp16 ok).
  - Accuracy budget → allowed precision set: tight → fp16/bf16 (≈0 loss) → fp8 (<1 %) →
    AWQ/GPTQ int4 (1–3 %, 4× memory cut) → bitsandbytes (experiment). vLLM refs:
    AWQ/GPTQ (Turing+), Marlin (fastest GPTQ/AWQ/FP8 kernel), FP8 (Ada/Hopper only).

**B. Agent judgment (the intern reasons — do NOT hard-bake):**
- Gather inputs: model size/arch (UC model metadata / HF `config.json`), team's stated
  accuracy expectation + latency/throughput target + cost ceiling, **workspace capability
  probe** (which `workload_type`s exist, region for H100, enrollment).
- Pick from the feasible set per priorities: accuracy-first → fp16 on the biggest GPU that
  fits; cost-first → smallest GPU + int4 + scale-to-zero; latency-first → enough replicas,
  higher gpu-mem-util, `--enforce-eager` off.
- Deploy → smoke-query → benchmark → record. **Deployment is a ledger experiment**
  (model version → config → latency p50/p99, tokens/s, $/1k-tok, quality-delta-vs-fp16).

### New tool — `agent/tools/model_serving_tool.py`  *(approval-gated — spins GPU $$)*
Ops:
- `probe_serving` — capability probe (which `workload_type`s incl. `MULTIGPU_*` exist, region/
  enrollment, workspace network-restricted?). Mirror `databricks_sandbox.probe_compute` cascade.
- `plan_deployment` — → `serving_strategy.feasible_configs`, returns ranked set + rationale.
- `build_and_register` — stage the **build/register notebook on a GPU job** (download weights →
  local vLLM smoke test → `mlflow.pyfunc.log_model` placeholder `ChatModel` + `entrypoint` metadata →
  `register_model(env_pack="databricks_model_serving")`). Reuse `databricks_jobs_tool` serverless-GPU
  path. **Must be a GPU job + non-restricted workspace** (see gotchas).
- `deploy` — create endpoint via **REST** `/api/2.0/serving-endpoints` with fixed
  `min==max provisioned_concurrency` (×4), `scale_to_zero_enabled=False` (the SDK default path is wrong).
- `query` — smoke `/invocations` (OpenAI-compatible chat).
- `benchmark` — **vendor `examples/custom-llm-serving/benchmark.py`**: concurrency sweep →
  sys_tps, p50/p95 lat, **TTFT** (the RAG/interactive metric — degrades under load from prefill
  queuing), TPOT, eff_decode_C, http429.
- `list` / `delete`.

Reuse `db_client`, OBO for user-scoped deploys, secrets via `{{secrets/scope/key}}`, no plaintext
creds in entrypoint.

### Tool-layer design (codex-planned, build-step 3)
- **One tool, `operation` enum** (matches `sweep_tool`/`research_loop_tool`). Handler returns the
  repo's `{"formatted", "isError"}` dict (NOT a tuple). Read-only: `plan_deployment`, `query`,
  `list`, `probe_serving` (shallow). Mutating/approval-gated: `build_and_register`, `deploy`,
  `delete`, `benchmark`, `probe_serving(deep=true)`.
- **`plan_hash` contract — enforce the deterministic boundary.** `plan_deployment` returns a
  *canonical plan object* (model_facts, serving_config, served/registered names, source_model,
  extra_pip_requirements=version pins, entrypoint, `plan_hash`=sha256 of canonical JSON).
  `build_and_register` REQUIRES that object and recomputes the hash — the agent may not
  hand-assemble `workload_type`/`precision`/`entrypoint` at build time.
- **REST via `wc.api_client.do`**, not raw urllib (centralizes OBO/SDK auth): `POST
  /api/2.0/serving-endpoints` (create), `PUT …/{name}/config` (update). Body uses `served_entities`
  + `min_provisioned_concurrency == max_provisioned_concurrency` + `scale_to_zero_enabled=false`.
  Never `EndpointCoreConfigInput` (its defaults force autoscaling, which entrypoint endpoints reject).
- **`build_and_register` (riskiest — artifact lifecycle).** Renders a build script run as a
  `databricks_jobs` `kind="serverless_gpu"` (reuse `_resolve_or_stage_script(as_notebook=True)`,
  `_build_submit_body`, `_wait_for_run`, `_fetch_run_output`; track run_id for cancel). The script:
  pin vLLM/transformers (+ autoawq/llmcompressor only for `quant_source="build"`, needs a calibration
  set → fail closed if missing) → download weights → [quantize] → validate artifact (config, tokenizer,
  precision, manifest hash) → **local vLLM smoke** (`/invocations` non-empty) → `log_model` placeholder
  `ChatModel` + metadata `{task, entrypoint, plan_hash, manifest}` → `register_model(env_pack=
  "databricks_model_serving")` → emit `SERVING_BUILD_RESULT_B64=<b64-json>` sentinel
  (registered_model_name, version, mlflow_run_id, pins, smoke_passed). **Honest TP caveat:** exact
  TP=4 smoke needs a 4-GPU build box; if built on 1×H100, mark `partial_tp_mismatch` + require
  post-deploy smoke + cleanup-on-failure — not "proof."
- **Ledger:** record each deployment as an `ExperimentRow` (method=`custom_llm_serving`, config=plan,
  metric=benchmark primary e.g. `p95_latency_ms`/`tokens_per_second`, artifacts=full metadata). New
  table only later if Lakeview needs time-series.
- **probe_serving** can't truly enumerate serving capability → confidence-rated: `serving_endpoints.list()`
  shows API visibility + already-used workload types; otherwise return `GPU_TIERS` defaults tagged
  `source="assumed"`; H100 enrollment `unknown` until observed/deployed. `deep=true` may submit a tiny
  GPU probe (mutating, gated).
- **query** retries HTTP 429 with jitter/`Retry-After`. **delete** never auto-approves.
- **Build order:** spine first (skeleton + `plan_deployment` + `deploy` REST + `query`/`list`/`delete`
  + probe + ledger + tests, all offline-testable) → then `build_and_register` + `benchmark` (the
  GPU-build piece, best landed with a workspace to live-test). Bites: MLflow ≥3.12 pin for `env_pack`,
  HF license/egress, vLLM/transformers pin drift, TP-smoke≠serving topology, OBO UC/Serving perms,
  logs vanishing after a failed config.

### Close the loop / showcase
Team fine-tunes (existing Mosaic AI path) → intern auto-deploys at the right size →
benchmarks → if accuracy within the team's budget at int4, keeps the cheaper config.
"Numbers go up" extends to serving: **$/token down, latency down, accuracy held.**
Deployment trials feed the Lakeview dashboard (ties into backlog #5).

### Skill — `skills/llm-serving-deployment/playbook.md`  *(evidence-first)*
Deploy fp16 baseline → measure quality + latency + cost → step one precision down →
measure delta → keep only if quality within the team's budget. Never trust a precision
choice without measuring (same discipline as the ablation rule).

### Build order
1. **Vendor** `examples/custom-llm-serving/` (FE reference — 6 files) as canonical ground truth.
2. `serving_strategy.py` — VRAM/TP/precision math + `render_entrypoint` (offline-testable).
   Validate `estimate_vram` against the two known FE points: Qwen3-4B fits 1×A10 (TP=1);
   Qwen3.5-27B-FP8 (~27 GB) needs TP=4 across A10×4 — the math must reproduce that verdict.
3. `model_serving_tool.py` — probe → plan → build_and_register → deploy (REST, fixed ×4) →
   query → benchmark (reuse FE `benchmark.py`) → list/delete.
4. `skills/llm-serving-deployment/playbook.md` (evidence-first + the 4 gotchas baked in).
5. Ledger deployment rows → Lakeview panel ($/tok, TTFT, accepted config).

Tests: VRAM/TP math vs the two FE model/GPU pairs, feasibility gating (precision⨯hardware,
single-GPU vs TP), entrypoint renders the opencv/fork/bash-lc fixes, provisioned_concurrency
snaps to ×4, capability-probe fallback, deploy/query mocked.
