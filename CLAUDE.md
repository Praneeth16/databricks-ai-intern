# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo

Port is staged over 10 phases (see `.plan/` if present, or ask before deep work). Phase 1 (config + auth + `db_client`) has landed — phases 2+ will swap the jobs tool, storage tools, sandbox, and prompts.

## Databricks-native component map

| Concern | Native primitive |
|---|---|
| LLM | Foundation Model API serving endpoints via LiteLLM's native `databricks/` provider (auth auto-resolved from SDK chain). Mosaic AI Gateway applies transparently when enabled on the endpoint — not explicitly routed in code. No direct Bedrock/Anthropic. |
| Job submission | Databricks Jobs API + Mosaic AI Model Training (`databricks-genai`) for fine-tune. |
| Files | UC Volumes (`/Volumes/<cat>/<schema>/<vol>/…`) + Workspace Files (`/Workspace/Users/<u>/…`). |
| Registry | UC registered models (`<cat>.<schema>.<name>`) via MLflow with `registry_uri=databricks-uc`. |
| Telemetry | MLflow Tracing only — no custom Delta KPIs, no APScheduler. Token/cost come from `system.serving.endpoint_usage`. |
| Session state | Lakebase (managed Postgres). No in-memory dicts. |
| Secrets | Databricks Secrets scopes. Jobs use `{{secrets/scope/key}}` dynamic refs — never baked into env. |
| Sandbox | Serverless GPU compute → serverless compute `commands/execute` → pool-backed cluster → on-demand. Adaptive probe at session start. |
| Deploy | Databricks Asset Bundles (`databricks.yml` + `resources/*.yml`). One `databricks bundle deploy`. |
| Auth | On Apps: `X-Forwarded-Access-Token` header (OBO). Locally: SDK unified chain (PAT → profile → M2M). |
| Dashboards | Lakeview dashboards in the bundle — no custom frontend KPI code. |
| Prompts | MLflow Prompt Registry under `databricks_ai_intern.agent.system_prompt`, yaml fallback. |

## Commands

### Install & run (CLI)
```bash
uv sync
uv tool install -e .
databricks-ai-intern                      # interactive
databricks-ai-intern "your prompt"        # headless
python -m agent.main                      # same, without install
```

Auth: set `DATABRICKS_HOST` + `DATABRICKS_TOKEN`, or run `databricks auth login`.

### Backend (FastAPI)
```bash
cd backend && bash start.sh               # uvicorn main:app :7860 (or :$DATABRICKS_APP_PORT on Apps)
uvicorn main:app --reload --port 7860     # dev
```
Backend runs with CWD=`backend/` — imports like `from routes.agent import ...` are bare. Running from repo root breaks them.

### Frontend (Vite/React/TS)
```bash
cd frontend
npm install
npm run dev       # :5173
npm run build     # tsc -b && vite build -> frontend/dist
npm run lint
```

### Tests
```bash
uv sync --extra dev
uv run pytest tests/unit
uv run pytest tests/unit/test_db_client.py::test_resolve_settings
```
Integration tests gated on `DATABRICKS_HOST` — skip cleanly otherwise.

### Bundle deploy
```bash
export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com  # bundle does NOT interpolate ${VAR} into workspace.host
databricks bundle validate
databricks bundle deploy --target dev     # or prod
databricks bundle run databricks_ai_intern           # open the App

# One-shot post-deploy bootstrap (each idempotent):
python scripts/bootstrap_pool.py --name databricks-ai-intern-warm
python scripts/register_prompt.py
databricks bundle run databricks_ai_intern_register_prompt --target dev   # or run as a Job
python scripts/wire_eval_trigger.py --job-id <eval_job_id_from_deploy>
```

### Code review graph (~10s, ~6× token reduction on blast-radius queries)
```bash
code-review-graph build                   # first time on this repo
# Already built — incremental updates auto on edits.
```

## Architecture

Three deployables, one `agent/` package:

1. **`agent/`** — pure Python agent + tools. CLI binary `databricks-ai-intern`.
2. **`backend/`** — FastAPI wrapper over WebSocket, deployed as a Databricks App.
3. **`frontend/`** — React + MUI + Zustand, served by backend.

### Agent core (`agent/core/`)
Queue-based async loop. Operations → `submission_loop()` → events. Provider-agnostic; only tool layer + config vary.

- `session.py` — `Session` owns `Config`, `ContextManager`, `ToolRouter`, queues.
- `agent_loop.py` — per-turn loop (max 300 iters): `litellm.acompletion` → parse tool_calls → approval gate → `ToolRouter.execute_tool` → repeat.
- `context_manager/manager.py` — `litellm.Message[]` history, auto-compact at ~170k tokens.
- `tools.py` — `ToolRouter` merges builtin ToolSpecs + MCP tools.
- `doom_loop.py` — detects repeated tool patterns, injects corrective prompts.
- `db_client.py` — **Databricks gateway**: `resolve_settings`, `get_workspace_client`, `get_workspace_client_for_user` (OBO), `get_sql_connection`, `get_mlflow_client`, `build_lakebase_conninfo`. Use this, never instantiate `WorkspaceClient` directly.
- `tracing.py` — MLflow Tracing wrapper: `init_tracing(experiment_path)`, `trace_span(name, attrs)`, `@traced(name)`. Fail-soft — no-op when MLflow can't reach the workspace. Replaces HF dataset upload + Delta KPIs.
- `model_catalog.py` — Databricks serving-endpoint catalog (replaces `hf_router_catalog.py`). Pulls `wc.serving_endpoints.list()` with 5-min cache. `lookup`, `fuzzy_suggest`, `prewarm`.
- `prompt_registry.py` — `load_system_prompt(name, version=...)`. MLflow Prompt Registry first, bundled YAML fallback.
- `llm_params.py`, `prompt_caching.py`, `model_switcher.py`, `effort_probe.py` — per-provider param shaping. `databricks/` branch lives here.
- `redact.py` — secret scrubbing before MLflow artifact logging.

### Tools (`agent/tools/`)
- `databricks_jobs_tool.py` — Dispatches by `kind`: `finetune` → Mosaic AI Model Training REST, `script` → Jobs `runs/submit` with `new_cluster`, `serverless` → `runs/submit` with `environment_key`. Stages inline scripts to Workspace Files. Filters DATABRICKS_*/cloud creds from agent env.
- `uc_volume_tools.py` — UC Volume read/write/ls/rm/mkdir via Files API. Path validation forces `/Volumes/` prefix.
- `uc_dataset_tools.py` — UC table inspection (`uc_inspect_dataset` op vocab: list_tables/describe/sample/query). Read-only SQL via warehouse.
- `uc_model_tools.py` — UC Registered Models list/inspect/list_versions/set_alias/delete_alias.
- `hf_to_uc_tool.py` — Hugging Face → UC ingestion. `ingest_dataset` (optional CTAS into Delta), `ingest_model`, `ingest_file`. Snapshot to local tmp, then stream to `wc.files.upload`.
- `repos_tool.py` — Databricks Git Folders: clone/list/inspect/pull/delete. Provider inferred from URL host.
- `sandbox_tool.py` + `databricks_sandbox.py` — Databricks-native sandbox. `probe_compute` cascade: serverless GPU (env-gated, not GA) → pool-backed single-node cluster → on-demand. `bash` runs subprocess via `command_execution` Python kernel. `read`/`write`/`edit` route `/Volumes/*` → `wc.files`, `/Workspace/*` → `wc.workspace`.
- `papers_tool.py`, `research_tool.py`, `docs_tools.py`, `github_*.py` — research corpus. Kept — orthogonal to infra.
- `plan_tool.py`, `local_tools.py`, `edit_utils.py`, `utilities.py` — provider-agnostic.

### Backend (`backend/`)
- `main.py` — FastAPI + CORS, routes.
- `routes/agent.py` — WebSocket agent I/O.
- `routes/auth.py` — minimal `/auth/me`, `/auth/status`. No OAuth flow (Apps proxy handles it).
- `dependencies.py` — `get_current_user` reads `X-Forwarded-*` headers in Apps mode, SDK chain locally. `extract_obo_token` for per-request OBO plumbing.
- `session_manager.py` — in-memory WS→Session map for hot path; per-create / per-cleanup writes to Lakebase via `backend/lakebase.py` (best-effort, fail-soft when not configured).
- `lakebase.py` — psycopg connection pool over Lakebase. `init(config)` builds the pool with a 45-min connection lifetime so OAuth token rotation never strands a request. Schema: `databricks_ai_intern_sessions`. Helpers: `upsert_session`, `mark_session_inactive`. `get_pool()` returns None when Lakebase isn't configured.

### Config (`agent/config.py` + `configs/main_agent_config.json`)
JSON config with `${VAR:-default}` env interpolation. `DatabricksConfig` submodel carries workspace binding (catalog, schema, volume, warehouse, lakebase, prompt registry name). Env overrides config-file values.

## Conventions

- **Never instantiate `WorkspaceClient` directly.** Go through `agent.core.db_client`. Tests bust caches via `reset_clients_for_tests()`.
- **Never accept `DATABRICKS_TOKEN` / `DATABRICKS_CLIENT_SECRET` from LLM-emitted `env` dicts.** The jobs tool filters these out before submitting. Auth is resolved server-side only.
- **User-scoped actions use OBO.** Route handlers should pass `extract_obo_token(request)` into `get_workspace_client_for_user(token, host)` for tool-triggered ops so the audit log names the user, not the App SP.
- **Jobs reference Workspace Files, not base64-wrapped inline scripts.** When the agent authors a training script, write to `/Workspace/Users/<user>/databricks-ai-intern/<session>/train.py` and reference it from the job spec.
- **Secrets in jobs use `{{secrets/scope/key}}` dynamic references** — never plaintext.
- **Model registration is `registered_model_name="databricks_ai_intern.agent.<name>"`** (three-level UC name), with `mlflow.set_registry_uri("databricks-uc")`.
- **Session telemetry = MLflow Tracing (`@mlflow.trace`).** Don't write custom Delta tables for agent activity.

## Review gate

`REVIEW.md` overrides default review behavior for this repo (P0/P1/P2 severities, P1 cap of 3, verdict line, skip list). Read it before reviewing.
