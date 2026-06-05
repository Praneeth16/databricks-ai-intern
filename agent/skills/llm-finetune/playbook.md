# Skill: llm-finetune

Evidence-first recipe for fine-tuning LLMs on **Mosaic AI Model Training** via
`databricks_jobs` with `kind=finetune`. Every "do X" below is a hypothesis with
a decision threshold, not a law. The default posture: confirm the setup, set a
target, reproduce a known number, then ablate one knob at a time. The metric
here (`eval_loss` / `perplexity`) is **LOWER is better** — keep that direction
straight everywhere, especially in the reproduce-gate.

## How to use the loop tools (read first)

- **`experiment`** — `find_similar` before every run (skip a base×data×config
  already tried), `propose` with `expected_metric` (your target eval_loss) before
  launching, `record` after (including failures), `best`/`list` to reason over
  history. Log every run.
- **`sweep`** — fan out LR / epochs / context_length / LoRA-rank variants in
  parallel instead of launching jobs one at a time.
- **Reproduce-first gate** — a run whose `eval_loss` is far WORSE (higher) than
  `expected_metric` was NOT reproduced. Do NOT add LoRA rank, epochs, or data.
  Fix the data format / template / metric wiring, reproduce the expected number,
  THEN tune.

## Phase 0 — Confirm the setup (MANDATORY, no ablation)

The single biggest source of wasted GPU-hours is a wrong-shaped dataset or the
wrong `task_type`. Confirm all of these before launching anything:

1. **Base model.** Exact registered name / HF id. Confirm it's a supported
   Mosaic AI base model and that you have access. Note its context length and
   tokenizer.
2. **Dataset format + `task_type`.** Pick ONE and confirm the rows match it:
   - `INSTRUCTION_FINETUNE` — rows have a `prompt`/`response` (or
     instruction/output) pair.
   - `CHAT_COMPLETION` — rows are OpenAI-style `messages` lists with roles.
   - `CONTINUED_PRETRAIN` — rows are raw `text` (no labels). Use only when the
     goal is domain adaptation, not task behavior.
   Print 3 sample rows and confirm the schema literally matches the chosen
   `task_type`. A CHAT dataset fed as INSTRUCTION (or vice versa) trains garbage.
3. **Train/eval split.** A held-out eval set is mandatory — see Phase 1.
4. **Metric + direction.** `eval_loss` and `perplexity` are LOWER = better.
   This orients `experiment propose` and the reproduce-gate. Don't invert it.

## Phase 1 — Held-out eval set + leakage check

**Rule (not ablatable):** evaluate on data the model never trained on, or every
later number is a lie.

- Hold out a real eval split (or pass `eval_data_path` to Mosaic AI). Never let
  eval examples appear in train.
- **Leakage checks before trusting any eval number:**
  - De-dup train vs eval on a content hash. Near-duplicates count as leakage.
  - For instruction data, check the *response* isn't trivially reconstructable
    from a train row (templated/synthetic data often repeats answers verbatim).
  - If `eval_loss` is suspiciously low on epoch 1, suspect leakage before
    celebrating.
- Record the eval set's size and source via `experiment` so later runs compare
  against the same yardstick.

## Phase 2 — Reproduce a known number BEFORE trying to beat it

**Rule:** your first real run targets a KNOWN baseline, not a new SOTA.

- Set `expected_metric` from a credible source: the base model's reported
  perplexity on a standard set, a model-card fine-tune number, or a paper's
  reported `eval_loss` for the same recipe. `experiment propose` with it.
- Launch the smallest faithful reproduction (default LoRA, modest epochs).
- If the run lands far worse than `expected_metric`, the reproduce-gate fires:
  do NOT escalate. Re-check data format, chat template, tokenizer, label masking,
  and the metric wiring. Reproduce first.
- Only once you've matched (or beaten) the expected number do you start ablating.

## Phase 3 — LoRA vs full fine-tune (ablation, not a default)

**Hypothesis:** full fine-tune *may* beat LoRA — but at much higher cost, and
often the gap is small. LoRA/QLoRA is the cheaper starting point, not an
automatic winner.

- Start with LoRA (or QLoRA if memory-bound). Record its eval_loss as the anchor.
- Run a full fine-tune as ONE ablation variant.
- **Decision rule:** adopt full fine-tune only if it improves eval_loss by a
  margin that justifies the cost (e.g. **≥ ~2% relative perplexity** AND the
  budget allows). Otherwise keep LoRA — it's cheaper to train, store, and serve.
- For LoRA, treat `lora_rank` / `lora_alpha` as sweepable knobs (Phase 4), not
  fixed magic numbers.

## Phase 4 — Sweep LR / epochs / context_length via `sweep`

**Hypothesis:** the defaults are rarely optimal, but more is not automatically
better (over-training raises eval_loss; too-long context wastes compute).

- Fan out a `sweep` over the few knobs that matter most, one at a time or in a
  small grid:
  - `learning_rate` — usually the highest-leverage knob. Sweep ~3 values across
    an order of magnitude around the recipe default.
  - `epochs` / training duration — watch for the eval_loss U-turn (it bottoms
    out then rises = over-fitting; stop at the minimum).
  - `context_length` — only raise it if your data actually needs it; longer
    context costs compute and memory for no gain on short examples.
- **Decision rule:** keep a knob change only if eval_loss improves beyond run-to-
  run noise (re-run the anchor twice to estimate noise; require the gain to
  exceed it). `experiment record` every trial — including the ones that
  regressed, so you never re-sweep the same space.
- Respect the per-job compute budget; prefer several small sweep children over
  one giant job.

## Phase 5 — Register + qualitative check

- Register the winning adapter/model to UC:
  `registered_model_name="databricks_ai_intern.agent.<name>"` with
  `mlflow.set_registry_uri("databricks-uc")`.
- eval_loss is necessary but not sufficient — spot-check generations on held-out
  prompts for format compliance, refusals, and obvious regressions vs the base
  model. A lower eval_loss with degraded behavior means the eval set doesn't
  capture what you care about; fix the eval, not the model.

## When to stop

If `experiment best` shows no sweep variant beating the reproduced baseline
beyond noise, stop. More epochs/rank/data past this point typically over-fits
(eval_loss rises). Report the best run, its eval_loss vs the target, and the
marginal-gain-vs-cost call.
