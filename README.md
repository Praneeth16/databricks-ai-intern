# Databricks AI Intern

*An autonomous ML engineer that reads the literature, ingests Unity Catalog
datasets, runs Mosaic AI jobs, and registers trained models — natively on the
Databricks AI runtime, until the numbers go up.*

---

Databricks AI Intern is an agentic ML engineer. Give it a goal — *"fine-tune Llama on this
UC table"*, *"get me a top-decile model on this Kaggle dataset"*, *"read the
latest papers on X and try the most promising idea"* — and it runs the full loop:
research → hypothesize → train → measure → reproduce-gate → iterate, all inside
**your** Databricks workspace with full MLflow lineage.

This is a Databricks-native port of the agent: every HuggingFace primitive (HF
Router, HF Jobs, HF Hub, HF Spaces, HF OAuth) has been replaced with a Databricks
equivalent. There is no HF fallback — Databricks is the only supported backend.

## Databricks-native component map

| Concern | Native primitive |
|---|---|
| **LLM inference** | Foundation Model API + AI Gateway (LiteLLM `databricks/` prefix). No direct Bedrock/Anthropic. |
| **Job submission** | Databricks Jobs API (`runs/submit`) + Mosaic AI Model Training (`databricks-genai`) for fine-tunes. |
| **Files** | UC Volumes (`/Volumes/<cat>/<schema>/<vol>/…`) + Workspace Files. |
| **Datasets** | Unity Catalog tables (read-only SQL via a SQL warehouse). |
| **Model registry** | UC registered models (`<cat>.<schema>.<name>`) via MLflow, `registry_uri=databricks-uc`. |
| **Telemetry** | MLflow Tracing — every turn, tool call, and LLM invocation is a span. Token/cost from `system.serving.endpoint_usage`. |
| **Session state** | Lakebase (managed Postgres). |
| **Secrets** | Databricks Secrets scopes; jobs use `{{secrets/scope/key}}` dynamic refs. |
| **Sandbox** | Serverless GPU → serverless compute → pool-backed cluster → on-demand (adaptive probe). |
| **Deploy** | Databricks Asset Bundles (`databricks.yml` + `resources/*.yml`). |
| **Prompts** | MLflow Prompt Registry (`databricks_ai_intern.agent.system_prompt`), YAML fallback. |

## What makes it an ML *researcher*, not just an ML *engineer*

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
databricks-ai-intern --model databricks/databricks-claude-sonnet-4 "fine-tune llama on my UC table"
databricks-ai-intern --max-iterations 100 "your prompt"
databricks-ai-intern --no-stream "your prompt"
```

### Web app (backend + frontend)

```bash
cd backend && bash start.sh        # FastAPI on :7860 (or :$DATABRICKS_APP_PORT)
cd frontend && npm install && npm run dev   # Vite/React on :5173
```

### Deploy as a Databricks App

```bash
export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
databricks bundle validate
databricks bundle deploy --target dev      # or prod
databricks bundle run databricks_ai_intern            # open the App
```

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
  "model_name": "databricks/databricks-claude-sonnet-4",
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
  title  = {Databricks AI Intern: An Autonomous ML Engineer on the Databricks AI Runtime},
  author = {Paikray, Praneeth},
  year   = {2026},
  url    = {https://github.com/Praneeth16/databricks-ai-intern}
}
```

This project is a Databricks-native port of the original
[**ml-intern**](https://github.com/huggingface/ml-intern) by Hugging Face, whose
agent loop and tooling design it builds on:

```bibtex
@software{huggingface_ml_intern,
  title  = {ml-intern: An open-source ML engineer that reads papers, trains models, and ships ML models},
  author = {{Hugging Face}},
  url    = {https://github.com/huggingface/ml-intern}
}
```
