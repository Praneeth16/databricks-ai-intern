# F1 Pit Stops S6E5 — Final Results

**Competition:** Kaggle Playground Series S6E5 — Predicting F1 Pit Stops
**Metric:** ROC-AUC on hidden test set
**Final submission:** `v5.2` — Optuna-tuned XGB + CatBoost blend
**Final LB score:** **0.94924**
**Status:** Locked. 9 attempts to beat v5.2 all failed.

## Leaderboard — all attempts ranked

| Version | LB AUC | Δ vs v5.2 | Approach |
|---------|--------|-----------|----------|
| **v5.2** | **0.94924** | **0.00000** | **Optuna 20-trial XGB + CB, hill-climb blend, 14 feats, Year=2025 val (FINAL)** |
| v11a    | 0.94921 | -0.00003 | Logit-rank blend (v5.2 + v5.4 + v8 + v10), v5.2 anchor 0.55 |
| v9      | 0.94915 | -0.00009 | Logit-rank blend (v5.2 + v5.4 + v8) |
| v12     | 0.94897 | -0.00027 | Pseudo-label v5.2 confident test rows, re-Optuna |
| v5.4    | 0.94896 | -0.00028 | v5.3 + 20-trial CB |
| v8      | 0.94867 | -0.00057 | 5-seed XGB+LGBM+CB ensemble, Optuna params |
| v10     | 0.94852 | -0.00072 | Hazard-feature model (TyreLifeBin, target enc) |
| v1      | 0.94820 | -0.00104 | Baseline XGB |
| v7      | 0.94668 | -0.00256 | StratifiedGroupKFold(5) by Race, 3 seeds |
| v13     | 0.94634 | -0.00290 | GM recipe v1: ext data + 35 feats + 4-model + StratGroupKFold by RaceYear |
| v4.1    | 0.94513 | -0.00411 | 26 hazard features, single XGB |
| v13.2   | 0.93854 | -0.01070 | v13 → drop ext+TE-in-CV, keep feats + pseudo + 4-model, Year=2025 val |

## Final model (v5.2)

```python
# 14 features only
FEATS = ["Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint",
         "TyreLife", "Position", "LapTime (s)", "LapTime_Delta",
         "Cumulative_Degradation", "RaceProgress", "Position_Change"]
CAT_COLS = ["Driver", "Compound", "Race"]

# Year=2025 holdout val
val_mask = train["Year"] == 2025

# Optuna 20 trials each, hill-climb 2-model blend
# XGB: tree_method="hist", device="cuda", scale_pos_weight, early_stopping_rounds=50
# CB: task_type="GPU", native cat_features=CAT_COLS, early_stopping_rounds=80
# Blend: w ∈ [0,1] @ 0.01 grid on val AUC
```

## Validation Strategy

- **Year=2025 holdout** = LB proxy (val→LB gap +0.042, stable across versions)
- **NOT** Race-grouped KFold (v7: OOF 0.929 → LB 0.947 regressed by 0.0026)
- **NOT** StratGroupKFold by RaceYear (v13: OOF 0.925 → LB 0.946 regressed by 0.003)
- AUC is prevalence-invariant, so val→LB gap is pure distribution shift, NOT class-balance

## What Worked

- **Optuna 20 trials each on XGB + CB** — single biggest single-model win
- **Year=2025 holdout val** — only correct LB proxy
- **Hill-climb 2-model blend** — +0.001 over best single
- **GPU acceleration** (`tree_method=hist device=cuda` / `task_type=GPU`) — 5-10× speedup
- **Lean 14-feature set** — every richer feature set regressed

## What Did Not Work (Each Tried Honestly)

| Move | Result | Lesson |
|------|--------|--------|
| 26 hazard features (v4.1) | LB -0.00411 | More features ≠ better when distribution shifts |
| Race-grouped KFold (v7) | LB -0.00256 | Wrong CV geometry beats no CV |
| 5-seed ensemble (v8) | LB -0.00057 | Same-class seeds don't decorrelate enough |
| Hazard model for diversity (v10) | LB -0.00072 | Diversity from weakness ≠ useful diversity |
| 3-sub logit-rank blend (v9) | LB -0.00009 | Spearman > 0.998 → no rank diversity to extract |
| 4-sub logit-rank w/ hazard (v11a) | LB -0.00003 | 15% diverse-weak weight too small to help |
| Pseudo-label HI=0.97/LO=0.03 (v12) | LB -0.00027 | Pseudo = circular learning; val ↑ LB ↓ |
| GM recipe full (v13) | LB -0.00290 | StratGroupKFold + ext data + 4-model all wrong on this comp |
| GM recipe trimmed (v13.2) | LB -0.01070 | Even with Year=val, features + pseudo + MLP all hurt |

## Five Locked Lessons

1. **The honest ceiling for a single GBM family on this problem is 0.94924.**
   9 attempts to beat it, all failed. Tried: more features, more models, more seeds, more data,
   pseudo-labels, neural net diversity, KFold CV, logit-rank blending. None lifted LB.

2. **Lean beats rich on this comp.**
   v5.2's 14 features + 2 models > 35 features / 4 models / 50 features / external data.
   The data already has the signal; over-engineering adds noise.

3. **Pseudo-labeling is structurally circular on a high-AUC model.**
   Model learns to predict its own predictions → val rises, LB falls.
   Both v12 (HI=0.97) and v13.2 (HI=0.92 with model rebuild) regressed.

4. **MLP / NN add no value on this categorical-heavy problem.**
   In 2 separate 4-model blends (v13, v13.2), Optuna assigned MLP weight ≈ 0.
   Compound/Driver/Race interactions are GBM's strength, not NN's.

5. **The only valid val is Year=2025 holdout.**
   Race-shift CV (v7), Race×Year shift CV (v13) both produced OOF↑LB↓.
   Public LB top scores (~0.9545) come from blending other Kagglers' submissions, not pure modeling.

## Diversity diagnostic (spearman as guide)

```
v5.2 vs v5.4   = 0.999   (same family)
v5.2 vs v8     = 0.998   (same family)
v5.2 vs v10    = 0.994   (real diversity, but v10 weaker → didn't help)
v13 lgb vs xgb = 0.990   (same family)
v13 lgb vs cb  = 0.780   (real diversity, but base AUCs too low)
v13 lgb vs mlp = 0.898   (some diversity, but MLP useless)
```

Rule: spearman > 0.998 = same model. < 0.995 = real diversity. But diversity ONLY helps when both base models are comparably strong.

## Compute Spend

- Platform: Databricks serverless GPU (GPU_1xA10), AI Runtime, environment_version 4
- API: `/api/2.2/jobs/runs/submit` (notebook_task only)
- Total job-minutes: ~7 hours across 11 versions (v1, v4.1, v4.2, v5.2, v5.3, v5.4, v6, v7, v8, v9, v10, v11a, v12, v13, v13.2)
- Per-job: ~10-30min (Optuna runs longer than non-Optuna)
- Total Kaggle subs used: 11

## Artifacts

- Submissions: `/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops/submission_v*.csv`
- OOF files: same directory, `oof_v*.csv`
- MLflow experiment: `/Shared/databricks-ai-intern/f1_pitstops`
- Scripts: `scripts/f1_pitstops_*_v*.py`
- **Final script: `scripts/f1_pitstops_train_v5_2_optuna.py`**
- **Final sub CSV: `/Volumes/.../submission_v5_2_optuna.csv`**

## Final note

The public LB top scores of ~0.9545 are reached by blending OTHER KAGGLERS' submissions
(confirmed via /codex analysis of public notebooks). Pure honest modeling on this problem
caps at ~0.949 for the GBM family. Breaking that wall requires either:
(a) TabPFN or transformer architecture (true model-class diversity)
(b) Multi-author blend exploitation

Neither was pursued. v5.2 stands as the honest ceiling.
