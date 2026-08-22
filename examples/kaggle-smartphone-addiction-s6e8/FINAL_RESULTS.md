# Smartphone Addiction S6E8 — Final Results

**Competition:** [Kaggle Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8) — Predicting Smartphone Addiction
**Metric:** ROC-AUC on a hidden test set (296,302 rows)
**CV protocol:** `StratifiedKFold(5, shuffle=True, random_state=42)` on original row order, predictions in float64
**Status:** competition closes 2026-08-31; results below are real submissions, not estimates

> Numbers marked **LB** are measured public-leaderboard scores. Numbers marked **OOF**
> are 5-fold out-of-fold on train. Nothing here is extrapolated.

## Leaderboard — every attempt ranked

Ranked by OOF. Only two were submitted — the rest were dominated and spending a
leaderboard slot on them would have bought nothing.

Feature counts are what the model actually sees: static features plus the 12
target-encoded columns that are rebuilt inside each fold.

| Version | OOF AUC | LB AUC | Features | Approach |
|---|---|---|---|---|
| v6_logit_stack | **0.968503** | 0.96979 | — | Logit-space stack of the four below, LogisticRegression meta-learner |
| **v4_xgb_full_fe** | 0.968434 | **0.96982** | 42 + 12 TE | XGBoost: raw + nested target encoding + frequency + composition + decimal lattice |
| v3_lgbm_full_fe | 0.968243 | — | 42 + 12 TE | Same feature set, LightGBM |
| v2_lgbm_te_freq | 0.967309 | — | 24 + 12 TE | LightGBM, target + frequency encoding only (no composition/lattice) |
| v5_cat_native | 0.961154 | — | 30 | CatBoost native categoricals + ordered target statistics, no hand-rolled encoding |
| v1_lgbm_raw | 0.964070 | — | 12 | LightGBM on the 12 raw columns — the params-vs-features ablation |
| (naive baseline) | 0.96324 | — | 12 | Hand-picked params, raw columns, single holdout — the starting point |

**Best leaderboard score is the single XGBoost model, not the stack.**

## The stack did essentially nothing, and that is the interesting part

Published work on this competition reports **+0.00187** from a 74-member logit stack. Mine
gained **+0.000069** OOF and came out **0.00003 worse on the leaderboard**. That is not a
bug — it is what the theory predicts once you look at the members:

```
Spearman correlation between members
  v2 vs v3   0.9967      v2 vs v5   0.9797
  v2 vs v4   0.9961      v3 vs v5   0.9822
  v3 vs v4   ~0.996      v4 vs v5   0.9811

Meta-learner weights
  v4_xgb 0.6289 | v3_lgbm 0.2333 | v2_lgbm 0.0845 | v5_cat 0.0483
```

Three of the four members are near-identical GBMs (correlating above 0.996 — there is no
disagreement to exploit), and the only structurally different member, CatBoost at 0.9612,
sits **below the ~0.966 solo-OOF cliff** where contribution stops tracking diversity and
starts tracking strength. It got 4.8% weight. The published 74-member stack works because
it spans 19 model families including a Lookup-Transformer whose max correlation to the
pack is 0.9869 — a genuinely different mechanism, still individually strong.

**Ensembling is not a free +0.002.** It pays for disagreement between comparably strong
models, and four variations on gradient boosting are one model wearing four hats.

## The measured CV→LB relationship

Two calibration points, both submitted:

```
v4  OOF 0.968434  ->  LB 0.96982     offset = +0.001386
v6  OOF 0.968503  ->  LB 0.96979     offset = +0.001287
```

The offset is **positive**, the opposite sign to the s6e5 example's val→LB gap, and not a
quirk: test predictions average five fold-models while each OOF row comes from a single
fold model, so the test side carries less variance. Published submissions report +0.00109
to +0.00150, so both land mid-range. The offset also shrank on the stronger model, which
is the reported behaviour — a larger ensemble has already averaged away variance that the
test-side averaging would otherwise remove.

Practical consequence: an OOF gain below roughly 0.0005 will not survive to the
leaderboard. The single-20%-holdout noise floor I measured is about ±0.0003, and the stack
is the worked example of a sub-noise "improvement" going the other way in reality.

## Where the ~0.005 actually was — I got this wrong first time

Mid-session I concluded the gap was "almost entirely hyperparameters, not features",
because every feature idea I *invented* measured as noise. The `v1_lgbm_raw` ablation
says the opposite, and it is the cleaner experiment:

| Step | OOF | Delta |
|---|---|---|
| Naive first attempt (my params, 12 raw cols, single holdout) | 0.963240 | — |
| **v1** — 12 raw cols, published-style params, 5-fold | 0.964070 | +0.00083 |
| **v2** — + nested target encoding + frequency encoding | 0.967309 | **+0.00324** |
| **v3** — + composition/ratio + decimal lattice | 0.968243 | +0.00093 |
| **v4** — same features, XGBoost instead of LightGBM | 0.968434 | +0.00019 |

**Features were worth ~+0.00417; hyperparameters ~+0.00083** — roughly five to one the
other way from what I claimed. (The parameter step compares a single holdout against
5-fold OOF, so treat it as approximate. The feature steps all share one estimator and one
split, so those are clean.)

Why I got it backwards: the features I reached for first — NaN indicators, ratio
features, marginalisation over missing values — were the wrong ones and each measured as
nothing, so I generalised from three failures to "features don't matter here". The
feature that *did* matter was target encoding, which I had not tried and which published
work on this competition had already identified.

The real lesson is narrower and more useful than "tune before you engineer":

> Three failed feature ideas is not evidence that features are exhausted. It is evidence
> about those three ideas. Read what the field already found before concluding a whole
> category is dead — and before inventing a fourth idea of your own.

## What did not work (each measured, not assumed)

| Idea | Result | Why |
|---|---|---|
| NaN-indicator features | 0.96295 vs 0.96301 | Missingness is MCAR — the pattern carries no signal. Independently confirmed at −0.00001 by published work. |
| Monte-Carlo marginalising over missing features | 0.95240 vs 0.96328 | Sampling missing values from their *marginals* discards feature correlations and injects noise. |
| Constrained mutual imputation of `daily` ↔ `weekend` | 0.96309 vs 0.96324 | The relation is exact in the original data but the generator only preserved it for ~54% of comparable synthetic rows. |
| Concatenating the original 7,500-row source dataset | −0.0001 (published) | The two distributions violate the generator's accounting identity differently — see below. |
| Pseudo-labelling confident test rows | −0.0034 (published) | Circular on a high-AUC model: it learns to predict its own predictions, so val rises and LB falls. Matches the s6e5 finding. |
| CatBoost's native target statistics *instead of* explicit encoding | 0.961154 vs 0.967309 | Reported to beat hand-rolled encoding by +0.0004 when features are otherwise equal; here withholding the explicit encoding cost far more than the native handling returned. My design error, not a CatBoost limitation — it should have received the same encoded features. |
| Logit-space stacking of four GBM variants | +0.000069 OOF, −0.00003 LB | Members correlated above 0.996 with one weak outlier. Stacking pays for disagreement between comparably strong models, and four variations on gradient boosting are one model wearing four hats. |

## Two structural facts about the generated data

Both verified directly, not taken on trust:

1. **`daily_screen_time_hours ≥ social_media + gaming + work_study` holds for every
   single row** — all 421,427 train and 182,287 test rows where those four are present,
   with a minimum residual of exactly 0.0000. The original source dataset violates it in
   **60.7%** of rows. The generator manufactured the constraint, which is why mixing in
   the original data hurts and why the residual is a legitimate feature.
2. **`weekend_screen_time = daily_screen_time_hours + Uniform(0.5, 3.0)` exactly** in the
   original data: residual mean 1.7439, std 0.7200, bounds [0.500, 3.000] against
   theoretical 1.75 and 0.7217. The synthetic data keeps the correlation (r = 0.96) but
   only ~54% of comparable rows fall inside the window, so it is a feature, not an
   imputation rule.

Neither leak exists: `corr(id, label) = 0.0011` with a flat mean across id deciles, and
all 691,369 train feature-tuples are unique with only 2 of 296,302 test rows matching one.

## Honest ceiling, and where the leaderboard stops being evidence

![Leaderboard position](artifacts/leaderboard-position.png)

The public LB tops out at **0.97142** across 1,700 teams. The best score reachable from
*published* work is **0.97117** — the visible spike just below the top of that histogram
is teams blending each other's public submissions.

A widely-upvoted analysis of this competition showed that above roughly **0.97110** the
OOF signal and the public LB actively disagree: repeated pseudo-public splits gained
+0.00008 on the exposed rows and lost −0.0001 on untouched ones, and in 3 of 7 completed
Season-6 episodes *zero* public top-10 teams survived into the private top-10 (in S6E7
the public winner finished 440th). Optimising against a 59k-row public split is
multiple-testing, not modelling.

So this example stops at honest modelling and says where that leaves it, rather than
borrowing submissions to buy a rank. That is the same call the s6e5 example made.

## Compute

Everything ran on **CPU**, which is the right answer here and is now enforced by the
`compute_advice` tool:

- 691,369 × 12 tabular gradient boosting does not saturate a GPU — below roughly 10M rows
  the kernel-launch and host-to-device transfer cost exceeds the histogram work saved.
- **LightGBM cannot use GPU on the Databricks serverless GPU image at all**: its GPU build
  needs OpenCL, which that image does not ship. The s6e5 run lost a job to this.
- A full 5-fold XGBoost over 691k rows took **~10 minutes** on local CPU.

A deliberate tradeoff worth recording: the published reference LightGBM config
(`lr=0.01`, `max_bin=1023`, 10,000 rounds) measured **~15× slower per fold** than the
XGBoost member, which put the full ladder out of reach in one session. Dropping to
`lr=0.02` / `max_bin=255` costs on the order of 0.0002–0.0005 OOF and returns most of the
wall clock. The LightGBM rows above are therefore slightly understated relative to the
XGBoost row — that is a budget decision, not a modelling finding.

## Reproducing

```bash
# 1. Data (bearer token from Kaggle -> Settings -> API)
mkdir -p /tmp/s6e8 && cd /tmp/s6e8
TK=$(tr -d '\n' < ~/.kaggle/access_token)
for f in train.csv test.csv sample_submission.csv; do
  curl -sL -H "Authorization: Bearer $TK" -o "$f" \
    "https://www.kaggle.com/api/v1/competitions/data/download/playground-series-s6e8/$f"
done

# 2. Run any version from versions.yaml
cd examples/kaggle-smartphone-addiction-s6e8/scripts
python train.py --version v4_xgb_full_fe --data-dir /tmp/s6e8 --out-dir /tmp/s6e8work
python train.py --version all           --data-dir /tmp/s6e8 --out-dir /tmp/s6e8work

# 3. Submit (validates row count and id order first)
python kaggle_submit.py --file /tmp/s6e8work/submission_v4_xgb_full_fe.csv \
  --message "v4 xgb full-FE, OOF 0.968434"
```

Deps: `pandas numpy scikit-learn lightgbm xgboost catboost pyyaml kaggle`.
