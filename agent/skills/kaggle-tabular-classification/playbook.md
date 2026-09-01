# Skill: kaggle-tabular-classification

Evidence-first playbook for Kaggle tabular AUC/log-loss/accuracy competitions.
The structure is fixed (7 phases). The *moves* inside each phase are
**hypotheses, not laws** — every one cites prior evidence (which way it cut and
by how much) and a decision threshold. Run the ablation on THIS data; keep the
move only if it clears the threshold. The lessons below come from two comps —
Playground **S6E5** (F1 Pit Stops, temporal) and Playground **S6E8** (Smartphone
Addiction, IID with heavy missingness) — and they are strong priors, not guarantees.

**The two comps disagree, which is the point.** S6E5's single hardest-won lesson was
"a temporal holdout is the only valid proxy". S6E8 has no temporal column at all, and
there StratifiedKFold is correct. Where the two disagree, the disagreement itself is
the lesson: verify the geometry on *this* data rather than inheriting a rule.

> **Where the S6E8 campaign actually finished, and why this playbook was revised.**
> 458 / 3,532 — private 0.97080, **0.00023 short of top-50**. Not a compute deficit. Three
> process failures, each now a section here:
> 1. It falsified missingness-as-*pattern* (−0.00001, correct) and retired the whole
>    category. The 1st-place winner's best single model is `realmlp_repair20` — missing-value
>    reconstruction, iteration 20 — worth ~4th place alone. → **Phase 1b-missing**.
> 2. It generated feature ideas from priors, so three misses read as "features exhausted".
>    The winning agents generated them from the model's own errors, gaining +0.0013 on a
>    single model in six days. → **Phase 1c**.
> 3. It declared an honest ceiling from the public pool's ceiling. One RealMLP at CV 0.97070
>    beat that entire pool stacked. → **Know when the leaderboard stops being evidence**, and
>    the wall checklist.
>
> And the finding that dwarfs all three: the winner's **Phase One was a single autonomous
> agent, unattended for four days with a self-submit loop, and it reached 0.97127 public
> before a second agent existed** — past what a 45-minute run reached with other people's
> predictions blended in. Wall-clock autonomy beat both feature cleverness and worker count.
> Full comparison: `examples/kaggle-smartphone-addiction-s6e8/FINAL_RESULTS.md` § Posthoc.

## Phase -1 — Pick CPU or GPU before submitting anything (cheap, always do it)

Call `compute_advisor.recommend_compute(task_shape=..., model_family=..., n_rows=...,
n_features=...)` and state the answer. For tabular gradient boosting at Playground
scale the answer is **CPU**, and reaching for GPU is an active mistake:

- Below roughly 10M rows, multicore CPU `hist` beats GPU — kernel launch and
  host-to-device transfer cost more than the histogram work saved. S6E8 is 691k x 12.
- **LightGBM cannot use GPU on the Databricks serverless GPU image at all**: its GPU
  build needs OpenCL, which that image does not ship. S6E5 lost a whole job to this.
  XGBoost is fine there — it uses CUDA directly via `device="cuda"`.
- CPU is also far cheaper per DBU, and on S6E8 a 5-fold XGBoost over 691k rows took
  ~10 minutes on plain CPU.

Reach for GPU when the model does gradient descent on dense tensors (transformer,
CNN, MLP/TabM, any fine-tune) — not because the row count is "big".

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
- **If NO temporal or group column exists, use `StratifiedKFold(5, shuffle=True,
  random_state=42)`** and stop looking for a cleverer split. Evidence (S6E8): 12
  independent features, no time column, and an id carrying no signal (measured
  corr(id, label) = 0.0011, flat across deciles). That split gave a stable, *positive*
  CV→LB offset of +0.00109 to +0.00150 across submissions — my own v4 measured
  OOF 0.968434 → LB 0.96982, an offset of +0.00139. Note the sign: unlike S6E5's
  val→LB gap, here LB reads slightly *higher* than OOF, because test predictions
  average five fold-models while OOF rows come from one.
- **Use the same seed and split as the community if you plan to blend.** OOF arrays are
  only stackable if they share fold assignments on original row order.
- **Single-holdout noise floor.** On S6E8 a single 20% holdout had a ~±0.0003 noise
  floor — enough to make three separate feature ideas look like small wins when they
  were nothing. Anything below ~0.0005 needs full k-fold OOF to resolve.
- **Decision rule:** submit one model under each candidate scheme; keep the
  scheme whose OOF/val ranks the two submissions in the same order as LB. Do
  NOT inherit "time-holdout always wins" — on IID data grouped KFold is usually
  the better, lower-variance estimator. Test it here.
- **Stop rule:** on 2 consecutive LB regressions while val rises, your CV is
  broken. STOP tuning. Switch the validation scheme before anything else.

Record the chosen scheme and its gap via `experiment record` — every later
phase is judged against it.

## Phase 1b — Target + frequency encoding (highest-lift feature move on IID comps)

**Hypothesis:** on a synthetic Playground table where the generator memorised
value→label, encoding each column's *levels* is worth more than any derived ratio.

Evidence (S6E8): target + frequency encoding was **+0.00324 OOF measured here**
(0.964070 raw -> 0.967309), against +0.0023 reported by published work — the single
biggest feature win on that competition, and ~3.5x the entire composition/lattice block
that followed it (+0.00093).

**It also dwarfed hyperparameters, which is the opposite of what I first concluded.**
Tuning the raw-feature model was worth roughly +0.0008; features were worth ~+0.0042. I
had decided features were exhausted after three of my *own* feature ideas measured as
noise. Three failed ideas is evidence about those three ideas — read what the field
already found before writing off a whole category.

- Cast **all** columns to string levels, **numerics included**. Bounded low-cardinality
  numerics (age had 18 distinct values, notifications 231) leave ~500 rows per level,
  which is plenty for a well-estimated smoothed mean.
- Target-encode with smoothing ≈ 10: `(sum_y + prior*10) / (count + 10)`. Smoothing 50
  and 200 both measured worse (−0.00006 / −0.00030).
- **Nest the encoding inside the folds.** Each outer fold's training rows must be encoded
  out-of-fold via an inner split, or the model learns a feature that partly *is* the
  label and OOF flatters badly.
- Frequency-encode transductively (counts over train+test). It uses no labels, so it is
  not leakage.
- `pandas >= 3.0` trap: `.astype(str)` preserves NA instead of creating a `"nan"` level,
  which silently drops those rows from every subsequent group-by. Use
  `.astype(object).where(notna(), "__missing__").astype(str)`.

**Missing values: augment, never replace.** GBMs learn native NaN split directions;
imputing drags rows toward the mean and loses information. Keep imputed columns
*alongside* the originals (+0.0012 on S6E8).

### Phase 1b-missing — missingness has THREE independent sub-families. Test all three.

**This is where the S6E8 campaign lost its top-50 finish, so read the whole subsection.**
We measured NaN-indicator features at −0.00001, concluded "missingness is MCAR, dead", and
wrote that into this skill and the agent prompt. The 1st-place winner's answer to what broke
his plateau: *"One new large source of signal was all the missing values."* His winning
single model's filename is `realmlp_repair20` — missing-value **repair**, iteration 20 —
and it scored ~4th place on its own. We falsified one sub-family and retired the category.

The three sub-families are unrelated mechanisms. A null result on one says nothing about
the others:

1. **Pattern-as-predictor** — indicators, row-level NaN counts, missingness combinations.
   *Genuinely dead on S6E8:* −0.00001 here; count-of-missing scores AUC 0.5017 standalone,
   and the largest gap any single column opens between its missing and present rows is
   0.0042 against a 0.709 base rate (Georgy Mamarin, measured independently). An adversarial
   train/test probe at 0.566 sitting entirely in the missingness is safe to ignore.
2. **Value recovery — the live one, and the one we skipped.** When the generator enforces an
   arithmetic identity, a missing driver is partly *recoverable* from the columns that
   remain. On S6E8 `daily ≥ social + gaming + work` holds for 100% of rows; we verified that
   and then only used the residual as a feature on rows where everything was observed. Run it
   backwards and each missing driver gets an exact bound from the others. Size of the effect:
   **46% of what `gaming_hours` and `work_study_hours` add on top of the three screen columns
   disappears once you keep only rows where both drivers are observed** — nearly half of that
   apparent signal was recovery. Build: identity-derived bounds per missing pattern,
   per-column "null-secondary" imputation models (25th place ran these), reconstruction
   residuals.
3. **Missingness-shaped augmentation on NN members** — mask inputs during training in the
   *same shape* as real missingness. Measured **+0.0014 on a solo NN** (Talha Tursun),
   shrinking to ~0 inside a large blend. So it is a single-model lever that pool-blending
   campaigns systematically discard. Our one test (masking aug 0.25 on a lookup transformer,
   −0.00014) was the wrong architecture and an arbitrary rate — it does not close this.

**Rule.** Before recording any missingness verdict, name which of the three you tested.
"Missingness doesn't help here" is not a finding a single ablation can support.

## Phase 1c — Error forensics: derive the next feature from the model's mistakes

**Do this before every feature block after Phase 1b.** It is the process difference between
the S6E8 campaign that finished 458th and the one that won.

Our loop was: pick an idea family → measure OOF → accept or reject. A rejection taught
nothing about what to try next, so after three misses we concluded a category was exhausted.
The 1st-place agents ran the other loop, in their operator's words: *"they continually
studied the single model and analyzed where it made mistakes and what types of patterns
different features or model architectures could replicate and constantly added just what was
needed to address every shortcoming."* Their best single model gained **+0.0013 in six days
in ~8-10 discrete steps** (each step one discovery landing), and their *starting* point was
above our best single model's final score.

Procedure, on OOF predictions of the current best single model:

1. **Rank rows by loss.** For AUC, work with the misranked pairs: sample high-scoring
   negatives and low-scoring positives.
2. **Slice the worst decile** by each raw column's value bucket, by missing-pattern, by
   every categorical level. Compare slice error rate against the global rate.
3. **Name the top 3 slices and ask what representation would separate them.** A slice the
   model cannot fix with more trees is telling you about a *representation* gap, not a
   capacity gap — e.g. on S6E8 `notifications_per_day` has univariate AUC 0.492 (no monotone
   signal at all) but its per-value residuals correlate 0.72 across independent halves, which
   is what makes an exact-value embedding table the right answer and axis-aligned splits the
   wrong one.
4. **Build the one feature or member that addresses the top slice**, ablate it, and re-run
   this phase. Not a batch of ten guesses.
5. `experiment record` the slice diagnosis alongside the metric. The diagnosis is the part
   that compounds; the metric alone does not.

**Stop rule inversion.** "N feature ideas in a row measured as noise" is NOT a signal to stop
engineering — it is a signal that you are generating ideas from priors instead of from this
model's errors. Switch to this phase before concluding a category is dead.

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

## Already-falsified hypotheses — do not spend compute re-deriving these

Both comps are synthetic Playground tables. These were each measured, and each lost.
If you think one applies anyway, say why *this* data differs before running it.

**Read the Scope column before generalising.** Every row is a verdict on exactly what its
scope says and nothing wider. This column exists because the S6E8 campaign recorded
"NaN-indicator features: −0.00001, missingness is MCAR", read it later as *missingness is
dead*, and lost a top-50 finish to a feature family the winner called his largest new source
of signal. A narrow measurement written down without its scope becomes a false general law
the next reader cannot audit.

| Idea | Measured result | Scope — what this does NOT falsify | Why it failed |
|---|---|---|---|
| NaN-indicator features, row-level NaN counts | S6E8 −0.00001 / +0.00002 | **Only missingness-as-pattern.** Says nothing about value *recovery* through a generator identity, per-column imputation models, or missingness-shaped NN augmentation (+0.0014 solo NN). See Phase 1b-missing | The pattern of which cells are absent carries no signal on the label (MCAR); count-of-missing scores AUC 0.5017 standalone |
| Concatenating the original source dataset | S6E8 −0.0001 | Only *this* generator's identity mismatch; check the identity before assuming it transfers | Generator manufactured structure the original lacks: `daily ≥ social+gaming+work` holds for **all** 421k synthetic rows, violated by 60.7% of original rows |
| Pseudo-labelling confident test rows | S6E8 −0.0034 (worst single move); S6E5 −0.00027 | Only high-AUC models with test ≈ train; a genuine distribution shift is a different case | Circular — it learns to predict its own predictions, so val rises while LB falls |
| Monte-Carlo marginalising over missing features | S6E8 0.95240 vs 0.96328 | Only *marginal* sampling. Joint/conditional reconstruction is untested and is the live sub-family | Sampling from marginals discards feature correlations and injects noise |
| Constrained mutual imputation, `daily`↔`weekend` | S6E8 0.96309 vs 0.96324 | **Only that column pair.** The `daily ≥ social+gaming+work` identity holds on 100% of rows and was never run as recovery — do that one | The U(0.5,3) window only survives in ~54% of comparable synthetic rows |
| Neural members (TabM / ResNet / MLP-PLR) as *diversity* | ≤ +0.00002 to a GBM stack | Only neural members *below* the strength cliff. A neural member AT GBDT strength is the highest-value member there is — S6E8's winning single model was a RealMLP at 0.97070, above the entire public pool stacked | Below a ~0.966 solo-OOF cliff, contribution tracks solo strength, not decorrelation |
| Rank-averaging a saturated ensemble | S6E8 −0.0027 vs logit stack | Only plain rank-averaging. Rank-normalised inputs into a *fitted* LogReg C-bag reached 0.970879 for 7th place | Ranks lose resolution when members correlate above 0.99 |
| Richer feature sets on shift-heavy synthetic data | S6E5 −0.0041 (26 feats), −0.0029 (35 feats) | Only S6E5's shift-heavy geometry — S6E8 rewarded 600+ engineered features (25th place) | Over-engineering adds noise when the generator's signal is already exposed |
| 30-trial Optuna from published params | S6E8 0.966850 vs incumbent 0.966853 | Only when the incumbent was already a *published* config for this comp. From your own guesses, tuning was S6E5's biggest single win | Hyperparameters were already at the family's ceiling |

**Blending strategy that did work (S6E8):** logit-space stack
(`logit(p)` clipped to ±30) with a LogisticRegression meta-learner, fitted honestly so
no row is scored by a combiner that saw its own OOF value — **+0.00187** over the best
single model, and +0.00047 over raw-probability averaging. Keep predictions in float64;
float32 rounding reordered 28% of test ranks at blend level.

**Know when the leaderboard stops being evidence.** On S6E8, the top of the public LB
(0.97142) sits above the best score reachable from published work (0.97117) — the gap is
people blending each other's submissions. A widely-upvoted analysis showed that above
~0.97110 the OOF signal and the public LB actively disagree, and that in 3 of 7 completed
Season-6 episodes *zero* public top-10 teams survived into the private top-10. Optimising
against a 59k-row public split is multiple-testing, not modelling.

> **But do not turn that into a ceiling claim.** The S6E8 campaign went one step further and
> argued the 0.9711 band was unreachable by honest modelling, reasoning from "the best score
> reachable from published work is 0.97117". That was wrong, and it capped the campaign at
> 458th. The 1st-place **single** model measured CV 0.97070 — roughly +0.0006 above what the
> *entire* public pool of ~155 OOFs reaches when stacked (~0.9701, measured independently).
> **The ceiling of what the field has shared is not the ceiling of the problem.** When a
> leaderboard compresses into a 0.0002 band, that is evidence everyone stopped in the same
> place, not evidence the place is optimal. Distinguish two claims and only ever make the
> first: "the public *pool* has converged" (measurable, usually true) versus "the *problem*
> has converged" (almost never demonstrable).

## When you've hit the wall

If `experiment best` shows no variant beating the anchor by ≥ 0.001 over several
honest attempts, you may be at the honest ceiling for the model family (S6E5
capped at 0.94924 after 9 failed attempts to beat it). Breaking it needs true
model-class diversity (TabPFN/transformer) — not more features, seeds, or
GBDT blends. Stop when marginal gain < compute cost and say so.

**Before you declare a wall, run this checklist.** S6E8 declared one at 0.9705 and the
winner was 0.0010 further up the same road:

1. **Phase 1c error forensics on the best single model** — have you looked at *where* it is
   wrong, or only at whether ideas cleared a threshold? If the latter, you have not hit a
   wall, you have run out of priors.
2. **All three missingness sub-families tested?** (Phase 1b-missing.) Recovery and NN
   augmentation are the two usually skipped.
3. **Is your best single model at the strength of the best *published* single model?** If it
   is below, the wall is your model, not the problem. On S6E8 the winner's single model beat
   the entire public pool stacked, so a strong single model was the highest-leverage object
   in the competition and every "the pool has converged" argument was beside the point.
4. **Ensemble breadth vs single-model strength — pick deliberately, and know the trade.** 7th
   place stacked 556 verified streams to OOF 0.970879 (50 individually *weak* members were
   worth +0.000060 against +0.000013 for a random control — "care about different errors more
   than solo scores"). 1st place got 0.97070 from one model, and his 456-member ensemble
   added only +0.00028 on top of it. Both routes reached the top; running a half-sized version
   of each reaches neither.
5. **How long has the loop actually run?** S6E8's 1st-place **Phase One was a single
   autonomous agent, unattended for four days, submitting to Kaggle itself — it reached
   0.97127 public before any second agent existed**, above what our 45-minute run reached
   even with other people's predictions blended in. His 12 extra workers added +0.0008; the
   lone agent had already added +0.0008. If your campaign is measured in minutes, wall-clock
   is your binding constraint and nothing in this playbook will substitute for it.
6. **Is your ledger single-writer?** His coordination primitive was one shared discovery store
   that every agent wrote to and re-read. A ledger only one process ever writes is a ledger
   whose wrong entries are never challenged — see the missingness entry that cost S6E8.

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
