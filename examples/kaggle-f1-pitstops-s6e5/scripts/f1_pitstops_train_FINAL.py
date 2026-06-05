# Databricks notebook source
import sys as _ml_sys, io as _ml_io
_ML_INTERN_BUF = _ml_io.StringIO()
class _ML_INTERN_TEE:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, b):
        for s in self._streams:
            try: s.write(b)
            except Exception: pass
        return len(b) if isinstance(b, str) else 0
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass
    def isatty(self):
        return False
_ml_sys.stdout = _ML_INTERN_TEE(_ml_sys.__stdout__, _ML_INTERN_BUF)
_ml_sys.stderr = _ML_INTERN_TEE(_ml_sys.__stderr__, _ML_INTERN_BUF)
try:

    """
    PTB-smoke: Fine-tune Qwen3-1.7B-Base on GSM8K (200 samples) with LoRA + SFTTrainer.
    Evaluate on 8 test examples. Save adapter to UC Volume.
    """
    import subprocess
    import sys

    # Install required packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "trl", "peft", "accelerate"])

    import os
    import re
    import json
    import mlflow
    import torch
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    # ─── MLflow setup ───
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    # The experiment /Shared/ml-intern directory already exists - use it
    # The error "For input string: None" seems to be a workspace-specific issue
    # Try with the experiment path differently
    import mlflow.tracking
    client = mlflow.tracking.MlflowClient()
    try:
        exp = client.get_experiment_by_name("/Shared/ml-intern")
        if exp:
            print(f"Found experiment: {exp.experiment_id}")
            os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Shared/ml-intern"
        else:
            exp_id = client.create_experiment("/Shared/ml-intern")
            print(f"Created experiment: {exp_id}")
            os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Shared/ml-intern"
    except Exception as e:
        print(f"Experiment lookup: {e}, will use default")

    # ─── Config ───
    MODEL_ID = "Qwen/Qwen3-1.7B-Base"
    N_TRAIN = 200
    N_EVAL = 8
    MAX_SEQ_LEN = 512
    OUTPUT_BASE = "/Volumes/serverless_lakebase_praneeth_catalog/ml_intern_test/scratch/ptb_smoke"

    # ─── Load dataset via direct parquet download ───
    print("Loading GSM8K dataset...")
    train_file = hf_hub_download(repo_id="openai/gsm8k", filename="main/train-00000-of-00001.parquet", repo_type="dataset")
    test_file = hf_hub_download(repo_id="openai/gsm8k", filename="main/test-00000-of-00001.parquet", repo_type="dataset")

    train_table = pq.read_table(train_file)
    test_table = pq.read_table(test_file)
    train_data = train_table.to_pydict()
    test_data = test_table.to_pydict()
    print(f"Train total: {len(train_data['question'])}, Test total: {len(test_data['question'])}")

    # Create formatted training texts
    train_texts = []
    for i in range(min(N_TRAIN, len(train_data['question']))):
        text = f"Question: {train_data['question'][i]}\nAnswer: {train_data['answer'][i]}"
        train_texts.append(text)

    train_ds = Dataset.from_dict({"text": train_texts})
    print(f"Train samples: {len(train_ds)}")
    print(f"Sample: {train_ds[0]['text'][:200]}")

    # ─── Load model + tokenizer ───
    print(f"Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    print("Model loaded.")

    # ─── LoRA config ───
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ─── SFT Training ───
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")
    
        output_dir = f"{OUTPUT_BASE}/{run_id}"
        final_model_dir = f"{output_dir}/final_model"
    
        training_args = SFTConfig(
            output_dir="/tmp/sft_output",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_length=MAX_SEQ_LEN,
            logging_steps=10,
            logging_first_step=True,
            logging_strategy="steps",
            save_strategy="no",
            bf16=True,
            disable_tqdm=True,
            report_to="none",
            dataset_text_field="text",
            gradient_checkpointing=True,
        )
    
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            peft_config=lora_config,
            processing_class=tokenizer,
        )
    
        print("Starting training...")
        train_result = trainer.train()
        train_loss = train_result.training_loss
        print(f"Training complete. Loss: {train_loss:.4f}")
    
        # ─── Save adapter to UC Volume ───
        print(f"Saving adapter to {final_model_dir}...")
        os.makedirs(final_model_dir, exist_ok=True)
        trainer.model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        print("Adapter saved.")
    
        # ─── Evaluation ───
        print("\n=== EVALUATION ===")
        model_for_eval = trainer.model
        model_for_eval.eval()
    
        def extract_answer(text):
            """Extract the final numeric answer after ####"""
            match = re.search(r'####\s*([\-\d,]+)', text)
            if match:
                return match.group(1).replace(",", "").strip()
            return None
    
        correct = 0
        results = []
    
        for i in range(N_EVAL):
            question = test_data['question'][i]
            answer = test_data['answer'][i]
        
            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model_for_eval.device)
        
            with torch.no_grad():
                outputs = model_for_eval.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
        
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
            # Ground truth
            gt_answer = extract_answer(answer)
            pred_answer = extract_answer(generated)
        
            is_correct = (pred_answer == gt_answer) if (pred_answer and gt_answer) else False
            if is_correct:
                correct += 1
        
            results.append({
                "idx": i,
                "question": question[:100],
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "correct": is_correct,
                "generated": generated[:300],
            })
        
            print(f"  [{i}] GT={gt_answer} | Pred={pred_answer} | {'CORRECT' if is_correct else 'WRONG'}")
    
        accuracy = correct / N_EVAL
        print(f"\nAccuracy: {correct}/{N_EVAL} = {accuracy:.2%}")
    
        # ─── Log metrics ───
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("eval_accuracy_at_8", accuracy)
        mlflow.log_param("n_train_samples", str(N_TRAIN))
        mlflow.log_param("n_eval_samples", str(N_EVAL))
        mlflow.log_param("model_id", MODEL_ID)
        mlflow.log_param("lora_r", str(16))
        mlflow.log_param("lora_alpha", str(32))
        mlflow.log_param("epochs", str(3))
        mlflow.log_param("learning_rate", str(2e-4))
    
        # ─── Print summary ───
        print("\n" + "="*60)
        print("FINAL REPORT")
        print("="*60)
        print(f"run_id: {run_id}")
        print(f"UC Volume path: {final_model_dir}")
        print(f"eval_accuracy_at_8: {accuracy}")
        print(f"train_loss: {train_loss:.4f}")
        print(f"\nSample outputs:")
        for r in results[:2]:
            print(f"  Q: {r['question']}")
            print(f"  GT: {r['gt_answer']}")
            print(f"  Pred: {r['pred_answer']}")
            print(f"  Generated: {r['generated'][:200]}")
            print()
        print("="*60)

except BaseException as _ml_intern_err:
    _ml_tail = _ML_INTERN_BUF.getvalue()[-4000:]
    try:
        dbutils.notebook.exit(_ml_tail + '\n[ml-intern] error: ' + repr(_ml_intern_err))
    except Exception:
        pass
    raise
else:
    try:
        dbutils.notebook.exit(_ML_INTERN_BUF.getvalue()[-4000:])
    except Exception:
        pass
