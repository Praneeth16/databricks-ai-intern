# Skill: nlp-text-classification

Evidence-first recipe for text classification (sentiment, topic, intent, spam).
Every move is a hypothesis with a decision threshold, not a law. Default
posture: a strong cheap baseline first, reproduce its number, then ablate up to
a transformer only if it earns its keep. Higher is better for accuracy/F1 —
keep the direction straight in the reproduce-gate.

## How to use the loop tools (read first)

- **`experiment`** — `find_similar` before each run, `propose` with
  `expected_metric` (target macro-F1/accuracy) before training, `record` after
  (including regressions), `best`/`list` over history. Log every variant.
- **`sweep`** — fan out LR / epochs / max_length / model-size variants in
  parallel rather than launching jobs serially.
- **Reproduce-first gate** — a run far BELOW its `expected_metric` was NOT
  reproduced. Do NOT jump to a bigger model. Fix the split, tokenization, or
  label mapping and reproduce the baseline first.

## Phase 0 — Confirm labels, split, and metric (MANDATORY)

1. **Label schema.** Print the class set, the per-class counts, and the dtype.
   Confirm the label column is what the task actually asks for (binary vs
   multiclass vs multilabel are different problems).
2. **Class balance.** Compute the imbalance ratio now — it decides the metric.
3. **Metric + direction.** Both higher = better:
   - Imbalanced → **macro-F1** (weights each class equally; accuracy lies when
     one class dominates).
   - Balanced → accuracy is fine; report macro-F1 alongside anyway.
   Lock the metric before training; `experiment propose` uses it.
4. **Stratified split.** Use a stratified train/val/test split so every class is
   represented in val. Never let the same document (or near-duplicate) sit in
   both train and val — de-dup on a content hash first. Leakage here inflates
   every later number.

## Phase 1 — Strong cheap baseline FIRST (reproduce before scaling)

**Rule (strong prior):** a TF-IDF + linear model, or a small pretrained encoder,
is often within a point or two of a large transformer on standard text
classification — at a fraction of the cost. Establish it before reaching for
anything bigger.

- **Baseline A — TF-IDF + linear** (logistic regression or linear SVM). Minutes
  on CPU, no GPU. Word + char n-grams, sublinear TF. This is your anchor.
- **Baseline B (optional) — small pretrained encoder** (e.g. a DistilBERT-class
  model) head-only or lightly fine-tuned, if the task clearly needs semantics
  the bag-of-words can't capture.
- `experiment propose` with `expected_metric` from a published number for this
  dataset (or Baseline A's own result) and `record` the outcome.
- **Reproduce-gate:** if the baseline lands far below a known published number,
  fix the pipeline (tokenization, label mapping, split) before escalating. A
  bigger model will NOT fix a broken split.

## Phase 2 — Class-imbalance handling (ablation, not a default)

**Hypothesis:** on imbalanced data, imbalance handling *may* lift macro-F1 — but
it can also just trade recall for precision with no net gain.

- Candidates: class weights (`class_weight="balanced"` / weighted loss),
  threshold tuning on val, focal loss (transformers), or resampling.
- **Decision rule:** ablate each against the Phase-1 anchor on val macro-F1.
  Keep ONLY if macro-F1 improves beyond run-to-run noise (re-run the anchor
  twice to estimate noise). Class weighting usually helps macro-F1 on skew;
  oversampling often overfits the minority class — verify, don't assume.
- Prefer threshold tuning + class weights (cheap, no data duplication) before
  resampling.

## Phase 3 — Transformer fine-tune via `databricks_jobs` (only if it earns it)

**Hypothesis:** a fine-tuned transformer beats the baseline — sometimes. Test
it; don't assume it.

- Fine-tune a sequence-classification encoder via `databricks_jobs`
  (`kind=script` on a GPU cluster, or `serverless`). Stage the script to
  Workspace Files; don't inline-base64 it.
- Start small (DistilBERT/base-size) before large. `max_length` should match the
  real token-length distribution — don't pad to 512 if the 95th percentile is 80
  (it wastes compute for no gain).
- **Decision rule:** adopt the transformer over the baseline only if val
  macro-F1 (or accuracy) improves by a margin that justifies the GPU cost
  (e.g. **≥ ~0.5–1 macro-F1 point** AND the budget allows). A 0.1-point win is
  not worth a GPU pipeline — ship the TF-IDF model.

## Phase 4 — Sweep LR / epochs / max_length via `sweep`

**Hypothesis:** transformer defaults are rarely optimal; over-training hurts.

- Fan out a `sweep` over the high-leverage knobs:
  - `learning_rate` — top knob; sweep ~3 values (e.g. 1e-5, 2e-5, 5e-5).
  - `epochs` — watch the val-metric U-turn; stop at the peak (over-fitting drops
    it again). 2–4 epochs is the usual sweet spot for fine-tuning.
  - `max_length` — only as long as the data needs.
- **Decision rule:** keep a knob change only if val macro-F1 beats noise.
  `experiment record` every trial, including regressions, so the same space
  isn't re-swept.

## Phase 5 — Confirm on the held-out test split + register

- Evaluate the single best config ONCE on the held-out test split (not val) for
  the reported number. Tuning against test = leakage.
- Print the per-class F1, not just the macro average — a high macro-F1 hiding a
  near-zero minority-class F1 is a different story than the headline suggests.
- Register the chosen model to UC:
  `registered_model_name="ml_intern.agent.<name>"` with
  `mlflow.set_registry_uri("databricks-uc")`.

## When to stop

If `experiment best` shows no variant beating the baseline beyond noise, ship
the baseline — a cheap reproducible model that ties an expensive one is the
better engineering outcome. Report the best run, its metric vs target, and the
gain-vs-cost call.
