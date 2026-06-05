# Skill: kaggle-tabular-classification

Evidence-first playbook for Kaggle tabular AUC/log-loss/accuracy competitions.
The structure is fixed (7 phases). The *moves* inside each phase are
**hypotheses, not laws** — every one cites prior evidence (which way it cut and
by how much) and a decision threshold. Run the ablation on THIS data; keep the
move only if it clears the threshold. The lessons below come from one comp
(Playground S6E5, F1 Pit Stops); they are strong priors, not guarantees.

## How to use the loop tools (read first)

You have three tools that turn this from a guess-and-check grind into a search:

- **`experiment`** — `find_similar` BEFORE every run (skip configs already tried),
  `propose` (record the hypothesis + `expected_metric` before submitting),
  `record` (log the result after), `best`/`list` (reason over history). Log
  EVERY variant. Never re-run a config the ledger already has.
- **`sweep`** — fan out N variants in parallel (hyperparams, feature sets, seeds).
  Use it whenever you'd otherwise run the same script 3+ times by hand.
- **Reproduce-first gate** — when a run lands far BELOW its `expected_metric`,
  that means it was NOT reproduced. Do NOT add features/models/seeds. Re-read
  the source, fix the bug, reproduce the number, THEN escalate complexity.

## Phase 0 — Confirm the target + metric (MANDATORY, no ablation)

This is the one phase with no hypothesis — it is always done, always first.

> **Anti-pattern: Wrong target column.** The most expensive mistake on this
> family of tasks. On S6E5, 4 iterations were burned training on `PitStop`
> instead of `PitNextLap` — CV looked great, LB ~0.46. A high CV on the wrong
> target is worse than useless: it hides the bug. Confirm the target before
> writing a single line of model code.

1. **Read `sample_submission.csv` header.** The non-id column IS the target.
2. **Train-test column diff:** target must be in train, absent from test.
3. **Print target distribution + dtype** before training. If the task says
   "binary classification" and your target has 17 unique values, you're on the
   wrong column.
4. **Confirm the metric and its direction.** AUC/accuracy: higher is better.
   Log-loss: lower is better. The reproduce-gate orients on this — make sure
   `expected_metric` and the comparison direction match.

## Phase 1 — Strong baseline + the validation hypothesis

Two jobs of one purpose: establish an LB anchor AND test which CV scheme tracks LB.

**Baseline (1 job, ~5 min):** XGBoost, sensible defaults, submit once. This is
your LB anchor. Common starting params (tune later, don't tune now):
```python
XGBClassifier(
    max_depth=10, learning_rate=0.02, n_estimators=1500,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=8, gamma=0.5,
    reg_alpha=0.001, reg_lambda=5.0,
    early_stopping_rounds=50, eval_metric="auc", tree_method="hist",
)
```

**Validation is a HYPOTHESIS — verify it, don't assume it.** The single most
important number in the whole comp is the **CV↔LB gap**: does your val move the
same direction as LB?

- If a temporal column exists (Year, Date, Season, Order) AND test rows are the
  most-recent value(s), a **time-holdout** is a candidate. Evidence (S6E5):
  Year=2025 holdout had a stable +0.042 val→LB gap across every version — it
  was the ONLY scheme that tracked LB.
- Grouped/StratifiedGroupKFold is the other candidate. Evidence (S6E5): Race-
  grouped KFold (v7) gave OOF 0.929 but LB regressed by 0.0026; StratGroupKFold
  by RaceYear (v13) gave OOF 0.925 and LB regressed 0.003. On *this* shift-heavy
  data, grouped KFold's OOF lied.
- **Decision rule:** submit one model under each candidate scheme; keep the
  scheme whose OOF/val ranks the two submissions in the same order as LB. Do
  NOT inherit "time-holdout always wins" — on IID data grouped KFold is usually
  the better, lower-variance estimator. Test it here.
- **Stop rule:** on 2 consecutive LB regressions while val rises, your CV is
  broken. STOP tuning. Switch the validation scheme before anything else.

Record the chosen scheme and its gap via `experiment record` — every later
phase is judged against it.

## Phase 2 — Cross-entity / Race-context features (often highest lift)

**Hypothesis:** when multiple entities compete in the same event (drivers per
race, players per game, sellers per category), single-entity features leave the
biggest signal on the table. Cross-entity features grouped by (event_id, time)
MAY be the largest single win.

Concrete for S6E5 — group by (Race, Year, LapNumber), per row compute:
`pits_this_lap_in_race`, `driver_ahead_pitted_last_lap`,
`driver_behind_pitted_last_lap`, `tyrelife_rank_in_lap`, `lap_time_rank_in_lap`,
`pit_pressure_3lap`. For other tasks: within-group ranks, leads, lags,
cumulative counts on whichever (entity, event) split defines the domain.

Implementation: sort once by (event_id, time), chain pandas groupby
shift/cumsum/rank. ~60 LOC, <30s on 500k rows.

- **Evidence both ways:** the [F1 pit-stop paper, Frontiers 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12626961/)
  reports +0.02 AUC from race-context features on *real* FastF1 data. But on the
  *synthetic* S6E5 data, richer feature sets consistently regressed (v4.1's 26
  features: LB -0.00411; v13's 35 features: LB -0.00290). Real data → big lift;
  synthetic Playground data → frequently noise.
- **Decision rule:** add the feature block as ONE variant, ablate against the
  Phase-1 baseline on your verified val. Keep ONLY if val improves **≥ 0.002**
  AND the CV↔LB gap doesn't widen. Use `sweep` to test feature-block subsets in
  parallel rather than one giant block. `experiment find_similar` first.
- **Leakage check:** any feature that peeks at the current or future row's
  target leaks. On S6E5, lag of LapTime/Position leaked at the stint boundary,
  inflating CV to 0.948 while LB fell. All features must be causal (past only).

## Phase 3 — Within-entity sequence features

**Hypothesis:** trend/acceleration features (not just levels) within each
(entity, sub-event) sequence MAY add signal on top of Phase 2.

Concrete for S6E5 — group by (Race, Year, Driver, Stint), order by LapNumber:
`lap_in_stint` (cumcount), `laptime_slope_last3` (rolling OLS slope),
`laptime_delta_vs_stint_mean`, `degradation_accel` (diff), `tyrelife_X_stintprogress`.
All causal → no leakage by construction.

- **Evidence:** plausible +0.002–0.005 on real data; on S6E5 synthetic data the
  net of all richer features was negative (see Phase 2 evidence).
- **Decision rule:** ablate as one variant; keep only on val **≥ 0.002** with a
  non-widening CV↔LB gap. Log via `experiment`.

## Phase 4 — Hyperparameter search (Optuna) via `sweep`

**Hypothesis:** tuned params beat defaults. On S6E5 this was the single biggest
single-model win (20-trial Optuna on XGB + CB).

- Run Optuna (or a coarse grid) as a `sweep` so trials fan out in parallel
  instead of serializing into one 86-min job that times out (S6E5 anti-pattern:
  3-model × 5-fold × Optuna in one job → serverless-CPU timeout).
- One model family per sweep child. Respect the ~12-min per-job budget.
- **Decision rule:** keep tuned params only on val **≥ 0.001** over defaults.
  `experiment record` every trial's outcome so you never re-search the same space.
- GPU note: `tree_method="hist", device="cuda"` (XGB) / `task_type="GPU"` (CB)
  gave 5–10× speedup on S6E5 — use it when GPU compute is available.

## Phase 5 — Many-seed retrain on 100% data (insurance)

**Hypothesis:** averaging the winning model over 5–10 seeds on full train shaves
variance. NVIDIA Grandmasters Playbook §7 calls this reliable +0.001–0.002.

- **Evidence both ways:** generally cheap insurance; but on S6E5 a 5-seed
  ensemble (v8) actually *regressed* LB -0.00057 — same-family seeds were too
  correlated to decorrelate error.
- **Decision rule:** run as a `sweep` over seeds, average test probs, keep only
  if val **≥ 0.001**. Otherwise ship the single best model.

Reference: [NVIDIA Grandmasters Playbook §7](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/).

## Phase 6 — Hill-climb blend over diverse OOF preds

**Hypothesis:** blending genuinely diverse, comparably-strong models beats the
best single model.

1. Train OOF for 3–4 diverse models: XGBoost, LightGBM, CatBoost (native cat
   handling = real diversity), optionally one MLP. Split across jobs / use `sweep`.
2. Hill-climb weights on OOF:
   ```python
   from scipy.optimize import minimize
   from sklearn.metrics import roc_auc_score
   def neg_auc(w, oofs, y):
       w = np.abs(w); w /= w.sum()
       return -roc_auc_score(y, (w[:, None] * oofs).sum(0))
   best = minimize(neg_auc, np.ones(len(oofs))/len(oofs),
                   args=(np.array(oofs), y), method="Nelder-Mead")
   ```
3. Apply the same weights to test preds.

- **Diversity diagnostic (Spearman of OOF):** > 0.998 = same model, no value to
  extract; < 0.995 = real diversity. Evidence (S6E5): v5.2-vs-v5.4 = 0.999 (no
  help), lgb-vs-cb = 0.780 (real diversity, but base AUCs too low to help).
- **Hard rule from S6E5:** diversity ONLY helps when both bases are comparably
  strong. A diverse-but-weak model gets weight ≈ 0 (v10 hazard model: LB -0.0007;
  MLP in two blends: weight ≈ 0). Don't add a weak model for "diversity."
- **Decision rule:** keep the blend only on val **≥ 0.001** over the best single.
  If your CV is overfitting, blending averages the same overfit — fix CV first.
- If one model's OOF is Δ > 0.005 below best, the optimizer zeros it; don't waste
  compute retraining it.

Reference: [Matt-OP hillclimbers](https://github.com/Matt-OP/hillclimbers),
[S5E12 1st place](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl).

## Phase 7 — Pseudo-labeling (one round, high suspicion)

**Hypothesis:** adding confident test rows as pseudo-labels to train MAY help
when train/test distributions differ.

1. Predict test. Keep only `pred > 0.97 or pred < 0.03` (~10–20% of test).
2. Concat those rows + pseudo-labels onto train, refit, predict full test.

- **Strong prior AGAINST on high-AUC models:** pseudo-labeling on a model that's
  already strong is structurally circular — the model learns to predict its own
  predictions, val rises, LB falls. Evidence (S6E5): v12 (HI=0.97) LB -0.00027;
  v13.2 (HI=0.92 + rebuild) LB -0.01070. Both regressed.
- **Decision rule:** ALWAYS validate the pseudo model on a held-out fold before
  submitting. Keep only on held-out val **≥ 0.001** AND a non-regressing CV↔LB
  gap. On synthetic data where test ≈ train, expect this to do nothing — skip it
  unless you have a distribution-shift reason to believe otherwise.

References: [Deotte pseudo-labeling QDA 0.969](https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969),
[Regularized pseudo-labeling arXiv 2302.14013](https://arxiv.org/pdf/2302.14013).

## Usually-skip list (still ablatable if you have a reason)

- **SMOTE / class-weight tweaks** — AUC is rank-only; doesn't move it. (Does
  move log-loss/accuracy — ablate there.)
- **Isotonic / Platt calibration** — rank-preserving; no AUC gain. (Helps
  log-loss — ablate there.)
- **NN-only solutions** — GBDTs dominate categorical-heavy tabular. On S6E5 the
  MLP got weight ≈ 0 in every blend. Use NN only as a diversity ingredient, and
  only if it's comparably strong.

## When you've hit the wall

If `experiment best` shows no variant beating the anchor by ≥ 0.001 over several
honest attempts, you may be at the honest ceiling for the model family (S6E5
capped at 0.94924 after 9 failed attempts to beat it). Breaking it needs true
model-class diversity (TabPFN/transformer) — not more features, seeds, or
GBDT blends. Stop when marginal gain < compute cost and say so.

## Submission discipline (also in core system prompt)

- Most Playground comps = 5 submissions / 24h.
- Submit ONLY when val improves by **≥ 0.001** over the current best LB-mapped val.
- "Would-submit" line before each: `(prior_val, new_val, delta, reason)`.
- Never two consecutive submissions differing only in hyperparameters.
- Last line of every job: `READY FOR SUBMIT: <path> | expected LB ~Y based on val Z`.
  User submits manually.

## Per-iteration job template

1. `experiment find_similar` → skip if config already tried.
2. `experiment propose` with `expected_metric` (your LB estimate) before submit.
3. `mlflow.set_experiment("/Shared/databricks-ai-intern/<comp_slug>")` with workspace-dir
   collision fallback.
4. Wrap the script with the stdout-tee prelude so `runs/get-output` carries the tail.
5. Save submission to `/Volumes/<cat>/<schema>/<vol>/<comp_slug>/submission_iter<N>_<method>.csv`
   and OOF preds alongside.
6. End with `READY FOR SUBMIT: ...`.
7. `experiment record` the result — including regressions; the ledger's negative
   results are what keep you from repeating S6E5's 9 dead ends.

## CV↔LB gap calculator

- Anchor: best LB `LB_best`, its val `val_best`, gap `gap = LB_best - val_best`.
- New iter val `val_new` → estimated LB = `val_new + gap`.
- Submit if `val_new + gap > LB_best + 0.001`; hold otherwise.
- If actual LB diverges from estimate by more than the historical gap variance,
  the validation hypothesis (Phase 1) is breaking — re-verify it before tuning more.
