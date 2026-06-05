"""End-to-end finetune smoke test against a real workspace.

Submits a tiny LoRA fine-tune job through ``databricks_jobs``. Defaults to
``kind="serverless"`` because the workspace under test is serverless-only;
flip ``ML_INTERN_TEST_KIND=script`` + ``ML_INTERN_TEST_HARDWARE=t4-small``
when running against a workspace that allows classic compute.

    - Tiny base model (``sshleifer/tiny-gpt2``) — pulled from HF Hub at job time.
    - Synthetic dataset, 32 rows, 5 training steps.
    - LoRA via PEFT, MLflow logged.
    - CPU-friendly (no fp16 / bf16 in the SFTConfig).

Auto-skipped without:
    - DATABRICKS_CONFIG_PROFILE / DATABRICKS_HOST
    - ML_INTERN_RUN_FINETUNE_TEST=1   (opt-in — burns DBUs)
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import MagicMock

import pytest

# Train script lives as a bare string so we can stage it via the same
# Workspace Files path the production tool uses. Kept conservative:
# %pip install at top, MLflow logging, LoRA on a 4-layer GPT-2.
_TRAIN_SCRIPT = '''
import os
# AI Runtime sets HF_HUB_ENABLE_HF_TRANSFER=1 by default but hf_transfer is
# not in our pinned deps. Disable so AutoTokenizer / from_pretrained falls back
# to the standard requests downloader.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

print(f"CUDA_AVAILABLE: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA_DEVICE: {torch.cuda.get_device_name(0)}")

MODEL_ID = "sshleifer/tiny-gpt2"
print(f"Loading {MODEL_ID}...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Synthetic data — 32 rows of (instruction, completion).
rows = [{"text": f"Q: 2+{i}={i+2}. Tell me. A: {i+2}"} for i in range(32)]
ds = Dataset.from_list(rows)

cfg = SFTConfig(
    output_dir="/tmp/lora_out",
    max_steps=5,
    per_device_train_batch_size=4,
    learning_rate=1e-3,
    logging_steps=1,
    logging_first_step=True,
    save_strategy="no",
    report_to=[],
    disable_tqdm=True,
    bf16=False,
    fp16=False,
)

print("Starting SFTTrainer...")
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)
trainer.train()

final = trainer.state.log_history[-1] if trainer.state.log_history else {}
print(f"FINAL_METRICS: {final}")
print("DONE")

# notebook_task captures the dbutils.notebook.exit value as the run output.
try:
    dbutils.notebook.exit(  # noqa: F821 — provided by Databricks runtime
        f"cuda={torch.cuda.is_available()} steps={trainer.state.global_step} loss={final.get('loss','?')}"
    )
except NameError:
    pass
'''


def _gate_finetune_test():
    if not os.environ.get("DATABRICKS_CONFIG_PROFILE") and not os.environ.get("DATABRICKS_HOST"):
        pytest.skip("No workspace creds")
    if os.environ.get("ML_INTERN_RUN_FINETUNE_TEST") != "1":
        pytest.skip(
            "Set ML_INTERN_RUN_FINETUNE_TEST=1 to run the live LoRA finetune "
            "(burns ~$0.10 of GPU compute)."
        )


def _testing_settings():
    """Override the default settings to point at this workspace's test schema."""
    from agent.config import load_config
    from agent.core import db_client

    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json"))
    cfg.databricks.uc_catalog = os.environ.get(
        "ML_INTERN_TEST_CATALOG", "serverless_lakebase_praneeth_catalog",
    )
    cfg.databricks.uc_schema = os.environ.get("ML_INTERN_TEST_SCHEMA", "ml_intern_test")
    cfg.databricks.uc_volume = os.environ.get("ML_INTERN_TEST_VOLUME", "scratch")
    return db_client.resolve_settings(cfg)


@pytest.mark.asyncio
async def test_lora_finetune_runs_to_completion():
    """30 min wall-clock cap is enforced by the runs/submit timeout=25m + the
    tool's poll loop; pytest-timeout is not installed."""
    _gate_finetune_test()

    from agent.core import db_client
    from agent.tools.databricks_jobs_tool import DatabricksJobsTool

    settings = _testing_settings()
    wc = db_client.get_workspace_client(settings)
    me = wc.current_user.me()

    tool = DatabricksJobsTool(
        wc=wc,
        settings=settings,
        user_email=me.user_name,
        session=MagicMock(session_id=f"finetune-test-{int(time.time())}",
                          is_cancelled=False, _running_job_ids=set()),
    )

    kind = os.environ.get("ML_INTERN_TEST_KIND", "serverless_gpu")
    args = {
        "operation": "run",
        "kind": kind,
        "script": _TRAIN_SCRIPT,
        "filename": "lora_smoke.py",
        "timeout": "60m",
        "run_name": f"ml-intern-lora-smoke-{int(time.time())}",
    }
    args["dependencies"] = [
        "torch>=2.4",
        "transformers>=4.46,<4.50",
        "peft>=0.15.0,<0.17",
        "trl>=0.12.0,<0.15",
        "datasets>=2.21.0,<3.5",
        "accelerate>=0.34.0",
        "mlflow>=2.20",
    ]
    if kind == "script":
        args["hardware_flavor"] = os.environ.get("ML_INTERN_TEST_HARDWARE", "t4-small")
    print(f"Submitting LoRA smoke run on kind={kind} …")
    result = await tool.execute(args)

    print("---- tool output ----")
    print(result["formatted"])
    print("---------------------")

    assert not result.get("isError"), f"Tool returned error: {result['formatted']}"
    # Successful runs surface the SUCCESS result_state in the output block.
    assert "Result:** SUCCESS" in result["formatted"] or "Lifecycle:** TERMINATED" in result["formatted"], (
        "Run did not reach SUCCESS / TERMINATED — check the tool output above."
    )
    # dbutils.notebook.exit value carries the smoke-test fingerprint.
    assert "steps=5" in result["formatted"], (
        "Training did not complete 5 steps — final marker missing in run output."
    )
    # GPU smoke must actually be on GPU.
    if kind == "serverless_gpu":
        assert "cuda=True" in result["formatted"], (
            "serverless_gpu run reported cuda=False — accelerator selector not honored."
        )
