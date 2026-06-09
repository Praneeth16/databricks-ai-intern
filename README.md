# Databricks AI Intern

*An autonomous AI engineer that reads the literature, ingests Unity Catalog
datasets, runs Mosaic AI jobs, and registers trained models — natively on the
Databricks AI runtime, until the numbers go up.*

---

Databricks AI Intern is an agentic AI engineer. Give it a goal — *"fine-tune Llama on this
UC table"*, *"get me a top-decile model on this Kaggle dataset"*, *"read the
latest papers on X and try the most promising idea"* — and it runs the full loop:
research → hypothesize → train → measure → reproduce-gate → iterate, all inside
**your** Databricks workspace with full MLflow lineage.

Databricks AI Intern is Databricks-native end to end — LLM inference, job
submission, storage, model registry, serving, telemetry, and session state all run
on first-class Databricks primitives. There is no external-provider fallback: your
workspace is the only backend, so data and compute never leave it.

## Databricks-native component map

| Concern | Native primitive |
|---|---|
| **LLM inference** | Foundation Model API serving endpoints (api_base + key auto-resolved from the SDK auth chain). Mosaic AI Gateway governance + usage logging apply transparently when enabled on the endpoint. No direct Bedrock/Anthropic. |
| **Job submission** | Databricks Jobs API (`runs/submit`) + Mosaic AI Model Training (`databricks-genai`) for fine-tunes. |
| **Files** | UC Volumes (`/Volumes/<cat>/<schema>/<vol>/…`) + Workspace Files. |
| **Datasets** | Unity Catalog tables (read-only SQL via a SQL warehouse). |
| **Model registry** | UC registered models (`<cat>.<schema>.<name>`) via MLflow, `registry_uri=databricks-uc`. |
| **Model serving** | Custom LLM Serving — vLLM-backed Serverless Optimized Deployments. The agent plans GPU tier / tensor-parallel / precision, builds + registers, deploys, and benchmarks the endpoint. |
| **Telemetry** | MLflow Tracing — every turn, tool call, and LLM invocation is a span. Token/cost from `system.serving.endpoint_usage`. |
| **Session state** | Lakebase (managed Postgres). |
| **Secrets** | Databricks Secrets scopes; jobs use `{{secrets/scope/key}}` dynamic refs. |
| **Sandbox** | Serverless GPU → serverless compute → pool-backed cluster → on-demand (adaptive probe). |
| **Deploy** | Databricks Asset Bundles (`databricks.yml` + `resources/*.yml`). |
| **Prompts** | MLflow Prompt Registry (`databricks_ai_intern.agent.system_prompt`), YAML fallback. |

## What makes it an AI *researcher*, not just an AI *engineer*

Beyond one-shot training, Databricks AI Intern closes a measurable, self-iterating loop:

- **Experiment ledger** (`agent/core/experiment_ledger.py`) — every run persisted
  to a UC Delta table (JSONL fallback), with config, metric, and reproduce status.
- **Eval harness** (`evals/`) — task-spec-driven scoring (ROC-AUC, accuracy, rank
  percentile, eval-loss) so improvements are measured, not asserted.
- **Parallel sweeps** (`agent/core/sweep.py` + `sweep` tool) — fan out top-k
  hypotheses across Databricks Jobs concurrently; metric returned via a stdout
  sentinel that survives `runs/get-output`.
- **Reproduce-gate** (`agent/core/repro_gate.py`) — blocks escalation when a result
  fails to reproduce; metric direction inferred from the metric name.
- **Critic** (`agent/core/critic.py` + `critic` tool) — overfit / target-confusion
  / leakage / correlation-floor detectors.
- **Deterministic loop runner** (`agent/core/research_loop.py` + `research_loop`
  tool) — chains the primitives with explicit control flow (the LLM is out of the
  driver's seat): per round → generate → dedup → budget-clamp → sweep →
  reproduce-gate → accept → stop, with a budget > target > max-rounds > patience
  stop precedence.
- **Autonomous serving** (`agent/core/serving_strategy.py` + `agent/tools/model_serving_tool.py`
  + `model_serving` tool) — once a model is trained, the intern figures out *how to
  host it*: size-driven GPU-tier / tensor-parallel / precision / concurrency
  selection (deterministic VRAM math returns a ranked feasible set; the LLM picks by
  the team's accuracy / latency / cost priorities), then build → register → deploy
  (Custom LLM Serving) → query → benchmark, recording each deployment trial to the
  ledger. Validated end to end on a serverless-GPU workspace.

## Deployment

Every model the intern trains or fine-tunes is registered to **Unity Catalog**
(`<cat>.<schema>.<name>` via MLflow with `registry_uri=databricks-uc`) — versioned,
governed, lineage-tracked. How it's served depends on the model type; both land on
**Mosaic AI Model Serving** in your workspace (real-time REST + batch), under UC
governance and Mosaic AI Gateway.

### LLMs & fine-tuned LLMs → Custom LLM Serving (GPU)

The `model_serving` tool takes these from a registered checkpoint to a live endpoint
**autonomously** — the path validated end to end (Qwen2.5-0.5B on an A10, ~338 tok/s):

- **`plan_deployment`** — size-driven strategy: deterministic VRAM math returns the
  ranked *feasible* configs (GPU tier, tensor-parallel, precision, fixed concurrency);
  the LLM picks by the team's accuracy / latency / cost priorities.
- **`build_and_register`** — a serverless-GPU job downloads weights, smoke-tests vLLM,
  and registers a Serverless Optimized Deployment to UC.
- **`deploy`** — creates the endpoint (vLLM, OpenAI-compatible `/invocations`, fixed
  provisioned concurrency, no autoscaling).
- **`query` / `benchmark`** — smoke test + concurrency sweep (throughput, latency,
  HTTP 429s), recorded to the experiment ledger.

| | |
|---|---|
| **GPU tiers** | `GPU_MEDIUM` (1×A10, 24 GB) · `MULTIGPU_MEDIUM` (4×A10, tensor-parallel) · `GPU_XLARGE` (1×H100, 80 GB) |
| **Precision** | bf16 / fp16, online fp8, or pre-quantized AWQ / GPTQ / int8 (with the artifact) |
| **Selection** | smallest GPU that fits → escalate to H100 → tensor-parallel across A10×4 |
| **Source** | a HF repo id, a UC model URI, or weights on a `/Volumes` path |

Base/foundation LLMs that don't need fine-tuning are called directly via the
**Foundation Model API** (provisioned-throughput or pay-per-token) — no deploy step.

### Classical ML models (scikit-learn / XGBoost / LightGBM / …) → Model Serving (CPU)

Logged as MLflow models and registered to UC the same way, then served as:

- **Real-time** — a Mosaic AI Model Serving CPU endpoint (scale-to-zero capable) for
  low-latency REST scoring.
- **Batch** — `ai_query(...)` over a UC table, an MLflow `spark_udf`, or a scheduled
  Databricks Job writing scores straight to Delta.

Today the intern automates the *LLM* serving path end to end; classical-ML models are
trained + registered by the agent and promoted to a Model Serving endpoint via the
bundle / Serving UI / API.

### Where everything runs

All endpoints live in **your** workspace under Unity Catalog governance, fronted by
Mosaic AI Gateway (rate limits, usage logging → `system.serving.endpoint_usage`,
guardrails) when enabled. Nothing leaves the workspace.

## Quick Start

### Install (CLI)

```bash
git clone https://github.com/Praneeth16/databricks-ai-intern.git
cd databricks-ai-intern
uv sync
uv tool install -e .
```

Now `databricks-ai-intern` works from any directory:

```bash
databricks-ai-intern                      # interactive
databricks-ai-intern "your prompt"        # headless (auto-approve)
python -m agent.main                      # same, without install
```

### Authenticate to Databricks

Set workspace credentials via the SDK unified auth chain — environment, profile,
or M2M:

```bash
export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
export DATABRICKS_TOKEN=<your-pat>
# — or —
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

When running on Databricks Apps, auth is on-behalf-of the signed-in user via the
`X-Forwarded-Access-Token` header — no token plumbing required.

### Usage

```bash
databricks-ai-intern --model databricks/databricks-claude-sonnet-4-6 "fine-tune llama on my UC table"
databricks-ai-intern --max-iterations 100 "your prompt"
databricks-ai-intern --no-stream "your prompt"
```

### Web app (backend + frontend)

```bash
cd backend && bash start.sh        # FastAPI on :7860 (or :$DATABRICKS_APP_PORT)
cd frontend && npm install && npm run dev   # Vite/React on :5173
```

### Deploy with Databricks Asset Bundles (DAB)

The whole stack ships as one bundle (`databricks.yml` + `resources/*.yml`): the App
(backend + built frontend), the UC catalog/schema/volume, a `CU_1` Lakebase instance
for session state, the secret scope, an optional warm GPU instance pool, and the
prompt/eval jobs. One `databricks bundle deploy` provisions all of it.

```bash
# 0. Build the frontend — the App serves <repo>/static, which is gitignored.
cd frontend && npm install && npm run build && cd ..

# 1. Host comes from the env var; the bundle does NOT interpolate ${VAR} into workspace.host.
export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com

# 2. Validate + deploy. `dev` is the default target; `prod` pins a /Workspace/Shared root_path.
databricks bundle validate
databricks bundle deploy --target dev          # or: --target prod
#    override defaults with --var, e.g.:
#    databricks bundle deploy --var uc_catalog=my_cat --var warehouse_id=<id>

# 3. Open the App.
databricks bundle run databricks_ai_intern --target dev
```

The UC **catalog must already exist** — catalog creation needs the direct-deployment
engine, so the bundle only manages the schema + volume under it. Pass an existing
catalog via `--var uc_catalog=<name>` (default `databricks_ai_intern`).

#### Post-deploy bootstrap (each idempotent)

```bash
python scripts/bootstrap_pool.py --name databricks-ai-intern-warm        # warm GPU pool for the sandbox
databricks bundle run databricks_ai_intern_register_prompt --target dev  # system prompt → MLflow Prompt Registry
python scripts/wire_eval_trigger.py --job-id <eval_job_id_from_deploy>   # fire eval on each new model version
```

UC grants for the App service principal are applied out-of-band after deploy (dev
mode prefixes the App name, which mangles the SP lookup) — see the GRANT statements
in the bootstrap scripts.

## Architecture

Three deployables, one `agent/` package:

1. **`agent/`** — pure Python agent + tools; CLI binary `databricks-ai-intern`.
2. **`backend/`** — FastAPI WebSocket wrapper, deployed as a Databricks App.
3. **`frontend/`** — React + MUI + Zustand, served by the backend.

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       User / CLI / App                       │
└────────────┬─────────────────────────────────────┬──────────┘
             │ Operations                          │ Events
             ↓ (user_input, exec_approval,         ↑
      submission_queue  interrupt, compact, ...)  event_queue
             │                                          │
             ↓                                          │
┌────────────────────────────────────────────────────┐  │
│            submission_loop (agent_loop.py)         │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │  1. Receive Operation from queue             │  │  │
│  │  2. Route to handler (run_agent/compact/...) │  │  │
│  └──────────────────────────────────────────────┘  │  │
│                      ↓                             │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │         Handlers.run_agent()                 │  ├──┤
│  │                                              │  │  │
│  │  ┌────────────────────────────────────────┐  │  │  │
│  │  │  Agentic Loop (max 300 iterations)     │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Session                          │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ContextManager             │  │  │  │  │  │
│  │  │  │  │ • Message history          │  │  │  │  │  │
│  │  │  │  │   (litellm.Message[])      │  │  │  │  │  │
│  │  │  │  │ • Auto-compaction (~170k)  │  │  │  │  │  │
│  │  │  │  │ • Session state → Lakebase │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  │                                  │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ToolRouter                 │  │  │  │  │  │
│  │  │  │  │  ├─ UC datasets / volumes  │  │  │  │  │  │
│  │  │  │  │  ├─ Databricks Jobs +      │  │  │  │  │  │
│  │  │  │  │  │  Mosaic AI fine-tune    │  │  │  │  │  │
│  │  │  │  │  ├─ UC registered models   │  │  │  │  │  │
│  │  │  │  │  ├─ Custom LLM Serving     │  │  │  │  │  │
│  │  │  │  │  │  (plan/build/deploy)    │  │  │  │  │  │
│  │  │  │  │  ├─ Research loop: ledger, │  │  │  │  │  │
│  │  │  │  │  │  sweep, critic          │  │  │  │  │  │
│  │  │  │  │  ├─ Papers / docs / GitHub │  │  │  │  │  │
│  │  │  │  │  ├─ Sandbox & local tools  │  │  │  │  │  │
│  │  │  │  │  └─ MCP server tools       │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Doom Loop Detector               │  │  │  │  │
│  │  │  │ • Detects repeated tool patterns │  │  │  │  │
│  │  │  │ • Injects corrective prompts     │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  Loop:                                 │  │  │  │
│  │  │    1. LLM call (litellm.acompletion)   │  │  │  │
│  │  │       ↓  (databricks/ via AI Gateway)  │  │  │  │
│  │  │    2. Parse tool_calls[]               │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    3. Approval check                   │  │  │  │
│  │  │       (jobs, sweeps, destructive ops)  │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    4. Execute via ToolRouter           │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    5. Add results to ContextManager    │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    6. Repeat if tool_calls exist       │  │  │  │
│  │  └────────────────────────────────────────┘  │  │  │
│  └──────────────────────────────────────────────┘  │  │
└────────────────────────────────────────────────────┴──┘
```

### Agentic Loop Flow

```
User Message
     ↓
[Add to ContextManager]
     ↓
     ╔═══════════════════════════════════════════╗
     ║      Iteration Loop (max 300)             ║
     ║                                           ║
     ║  Get messages + tool specs                ║
     ║         ↓                                 ║
     ║  litellm.acompletion()  (databricks/)     ║
     ║         ↓                                 ║
     ║  Has tool_calls? ──No──> Done             ║
     ║         │                                 ║
     ║        Yes                                ║
     ║         ↓                                 ║
     ║  Add assistant msg (with tool_calls)      ║
     ║         ↓                                 ║
     ║  Doom loop check                          ║
     ║         ↓                                 ║
     ║  For each tool_call:                      ║
     ║    • Needs approval? ──Yes──> Wait for    ║
     ║    │                         user confirm ║
     ║    No                                     ║
     ║    ↓                                      ║
     ║    • ToolRouter.execute_tool()            ║
     ║    • Add result to ContextManager         ║
     ║         ↓                                 ║
     ║  Continue loop ─────────────────┐         ║
     ║         ↑                       │         ║
     ║         └───────────────────────┘         ║
     ╚═══════════════════════════════════════════╝
```

## Events

The agent emits the following events via `event_queue`:

- `processing` — starting to process user input
- `ready` — agent is ready for input
- `assistant_chunk` — streaming token chunk
- `assistant_message` — complete LLM response text
- `assistant_stream_end` — token stream finished
- `tool_call` — tool being called with arguments
- `tool_output` — tool execution result
- `tool_log` — informational tool log message
- `tool_state_change` — tool execution state transition
- `approval_required` — requesting user approval for sensitive operations
- `turn_complete` — agent finished processing
- `error` — error occurred during processing
- `interrupted` — agent was interrupted
- `compacted` — context was compacted
- `undo_complete` — undo operation completed
- `shutdown` — agent shutting down

## Development

### Adding Built-in Tools

Edit `agent/core/tools.py`:

```python
def create_builtin_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="your_tool",
            description="What your tool does",
            parameters={
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Parameter description"}
                },
                "required": ["param"]
            },
            handler=your_async_handler
        ),
        # ... existing tools
    ]
```

### Adding MCP Servers

Edit `configs/main_agent_config.json`:

```json
{
  "model_name": "databricks/databricks-claude-sonnet-4.7",
  "mcpServers": {
    "your-server-name": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${YOUR_TOKEN}"
      }
    }
  }
}
```

Environment variables like `${YOUR_TOKEN}` are auto-substituted from `.env`.

### Tests

```bash
uv sync --extra dev
uv run pytest tests/unit
```

Integration tests are gated on `DATABRICKS_HOST` and skip cleanly without it.

## Citation

If you use Databricks AI Intern in your work, please cite it:

```bibtex
@software{databricks_ai_intern,
  title  = {Databricks AI Intern: An Autonomous AI Engineer on the Databricks AI Runtime},
  author = {Paikray, Praneeth},
  year   = {2026},
  url    = {https://github.com/Praneeth16/databricks-ai-intern}
}
```
