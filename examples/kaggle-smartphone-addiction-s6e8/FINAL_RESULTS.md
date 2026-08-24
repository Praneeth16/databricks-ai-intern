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

## Second pass — 0.96982 to 0.97052 on our own models, 0.97104 with borrowed ones

The first pass stopped at a single XGBoost, 0.96982, and concluded the four-GBM stack was
worth nothing. Both of those held up. What moved the score was finding out *why* the stack
was worth nothing, and the answer was not the meta-learner.

### Every measurement, ranked

| version | OOF AUC | LB | what it is |
|---|---|---|---|
| **v19_lookup_bag_10f** | **0.968908** | — | lookup transformer, 3 seeds, 10 folds — best single model here |
| v20_xgb_ident_bag_10f | 0.968778 | — | v7 at 10 folds |
| v14_lookup_bag | 0.968727 | — | lookup transformer, 3 seeds, 5 folds |
| v7_xgb_ident_bag | 0.968655 | 0.96992 | v4 + identity block + 3 seeds, lr 0.015 |
| v8_lgbm_strong | 0.968548 | — | lr 0.01, max_bin 511, 2 seeds — the config the first pass could not afford |
| v13_lookup | 0.968293 | — | lookup transformer, 1 seed |
| v17_realmlp | 0.968282 | — | RealMLP-TD |
| v9_cat_encoded | 0.968241 | — | CatBoost given the same features as everyone else |
| v16_lookup_aug | 0.968154 | — | lookup, aug 0.25 |
| v15_lookup_wide | 0.968147 | — | lookup, d=192, 6 layers |
| v10_tabm | 0.968006 | — | TabM on the GBM feature set |
| **stack, own models only** | **0.969450** | **0.97052** | hill-climb: v19 0.50, v20 0.31, realmlp 0.12, tabm 0.05, cat 0.02 |
| stack + public OOF library | 0.969680 | 0.97084 | 83 members, 74 of them other people's |
| rank blend with najiama 19_blend | — | 0.97104 | mostly other people's |

For scale: the strongest *own-models* pipeline published on this competition scored 0.97041
(lookup + CatBoost + LightGBM at 11 folds). 0.97052 is slightly past that.

### The finding that mattered: diversity is a property of the input, not the architecture

The first pass blamed its dead stack on member correlation, which was right, and inferred it
needed a different *architecture*, which was wrong. Both were tested here on identical folds:

| member | mechanism | features it saw | Spearman vs trees | hill-climb weight |
|---|---|---|---|---|
| v10_tabm | batch-ensembled MLP | the GBM feature set | **0.9945** | 0.05 |
| v19_lookup | transformer over exact values | raw columns only | **0.9635** | **0.50** |

**Correction, added after a controlled follow-up.** The paragraph below draws the wrong
conclusion from these two rows, and `notebooks/lookup-transformer.ipynb` is the experiment
that found the error. Holding the trainer fixed and varying only the representation, a
strength-matched MLP on the encoded features measures **0.9665** against the trees, not
0.9945, which is statistically level with the lookup's 0.9658; the two networks correlate
0.9782 with each other. So the decorrelation boundary is tree-versus-neural, and the
lookup's real contribution is that it is the only non-tree member reaching GBDT strength
(0.968305 against 0.965032 for the same trainer on hand-built target encoding). Why TabM
specifically measured 0.9945 is still unexplained. The per-value evidence in the next
paragraph stands; only the causal claim about decorrelation does not.

TabM is not a decision tree and does not split on thresholds, and it still agreed with the
GBMs as closely as they agree with each other — because it was fed the same target-encoded
columns, so it learned the same function through different machinery. The lookup transformer
disagrees because it sees a different representation: an embedding table over each column's
exact quantised value.

That premise is measurable and was measured by the notebook this architecture came from:
`notifications_per_day` has univariate AUC **0.492** — no monotone signal at all — yet its
per-value residuals correlate **0.72** across two independent halves of the data. The
generator memorised value-to-label associations from its small source dataset, so the exact
value is a lookup key. Neither that nor the 4-term budget identity is representable by
axis-aligned splits at any data volume.

Independent confirmation, found after the fact: the author of the public 74-model OOF library
writes that adding this same architecture was "worth +0.000109 to my blend, which is more
than any other single change I have made in this competition," at max correlation 0.9869
against 73 other models.

### What else paid, and what did not

| move | delta | verdict |
|---|---|---|
| CatBoost given the shared feature set (v5 -> v9) | **+0.0071** | the first pass's worst design error |
| lookup transformer as a member | **+0.00054** on the stack | the only real decorrelation |
| 3 seeds averaged inside each fold | +0.00043 on the lookup member | variance, not bias |
| 10 folds instead of 5 | +0.00018 (lookup), +0.00012 (xgb) | real but small |
| full-strength LightGBM (lr 0.01, max_bin 511) | +0.0003 over v3 | worth the wall clock on CPU |
| weekend-window identity block | +0.00019 | the only feature idea that paid |
| RealMLP-TD as a member | 0.12 stack weight | strength, not diversity |
| pairwise TE, all 66 pairs | **-0.00024** | rejected |
| pairwise frequency, all 66 pairs | -0.00021 | rejected |
| coarse-bin TE alongside exact-level TE | -0.00006 | rejected |
| TE smoothing 50 / 3 (vs 10) | -0.00013 / -0.00007 | 10 was already right |
| identity *bounds* features (7 cols) | +0.00010 | noise for 7 columns; skipped |
| row-level NaN counts | +0.00002 | rejected, same verdict as NaN indicators |
| wider/deeper lookup (d=192, 6 layers) | -0.00015 | capacity is not the constraint |
| heavier masking augmentation (aug 0.25) | -0.00014 | — |
| 30-trial Optuna on XGBoost | **0.966850 vs incumbent 0.966853** | hyperparameters are exhausted |

That last row is worth its own sentence. On the s6e5 example an Optuna pass was the single
biggest single-model win. Here 30 TPE trials could not match hand-copied published params, on
an identical subsample and fold. The difference is where each started: s6e5 began from my own
guesses, this began from params published for this competition. A sweep is only evidence if
the incumbent was measured under the sweep's exact conditions — which is why the incumbent
arm was run at all.

### Where top-10 actually lives

Top-10 on the public leaderboard is 0.97130, and 203 teams sit inside 0.9711-0.9713 while the
gap from 1st to 10th is 0.00012 — smaller than the noise floor of a 59k-row public split.
Pulling the notebooks in that band shows what they are: the 0.97113 and 0.97117 entries are
`pd.read_csv` over najiama's and Szymon Kłapiński's shared OOF libraries, and the "#1 Public
LB 0.97068 | Honest 55-Model Stack" is honest in its *validation* while its 55 members are
other people's predictions.

Two things we measured ourselves support treating that band as unreachable by modelling:

1. Our honest nested meta over 83 members scored **0.969904**, *below* najiama's 19_blend
   solo OOF of 0.970099. His blend weights were fitted on all OOF rows, so his OOF is
   in-sample for those weights; an honest meta correctly refuses to reproduce it.
2. A 50/50 rank blend of our variant-B stack with his submission scored 0.97104 — *below* his
   0.97113 alone. At that correlation (Spearman 0.9962) our models dilute rather than add.

So the borrowed track converges on whatever the best borrowed artifact scores, and beating it
means blending several authors and tuning weights against the public leaderboard. One public
notebook demonstrates precisely what that costs: submitted at student weights -0.12, -0.08,
0.00 and +0.12, its OOF said one direction and the leaderboard rewarded the opposite.

Both variants were submitted and are recorded above, separately and labelled. Variant A is
the one this example stands behind.

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
