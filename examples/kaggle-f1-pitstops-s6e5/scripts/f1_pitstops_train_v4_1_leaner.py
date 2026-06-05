# Databricks notebook source
"""
v4.1 — Leaner Phase 2.

Drop from v4:
  - race_pit_progress_mean (broken alignment bug)
  - All raw rl_mean/std/min/max (redundant, distribution-shift in val)
  - Rolling-3 std (noisy)

Keep:
  - *__vs_field deltas (driver minus race-lap mean) — the actual signal
  - laptime_gap_to_leader, pos_gap_to_leader
  - progress_x_tyrelife, tyrelife_x_stint (interactions)
  - tyrelife_rolling3_mean, laptime_rolling3_mean, laptime_diff_prev, stint_lap_idx (within-driver sequence)

Regularize harder: min_child_weight=20, reg_lambda=5, max_depth=7.
"""
import os, sys, time, subprocess, warnings, json
warnings.filterwarnings("ignore")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "xgboost", "scikit-learn", "pandas", "numpy", "mlflow"])

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import mlflow

DATA_DIR = "/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{DATA_DIR}/submission_v4_1_leaner.csv"
OOF_PATH = f"{DATA_DIR}/oof_v4_1_leaner.npy"

TARGET = "PitNextLap"
ID_COL = "id"

print("Loading...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print(f"train={train.shape}  test={test.shape}")
assert TARGET in train.columns and TARGET not in test.columns
print(f"target pos rate: {train[TARGET].mean():.4f}")

# ─── Feature engineering ──────────────────────────────────────────────────────
combined = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)

race_lap_key = ["Year", "Race", "LapNumber"]
driver_race_key = ["Year", "Race", "Driver"]

# Cross-driver deltas only (no raw mean/std/min/max kept as features)
vs_field_cols = ["LapTime (s)", "TyreLife", "Position", "Cumulative_Degradation"]
print("Building vs_field deltas...")
for col in vs_field_cols:
    m = combined.groupby(race_lap_key)[col].transform("mean")
    combined[f"{col}__vs_field"] = combined[col] - m

# Gap to leader (laptime + position)
lt_min = combined.groupby(race_lap_key)["LapTime (s)"].transform("min")
combined["laptime_gap_to_leader"] = combined["LapTime (s)"] - lt_min
combined["pos_gap_to_leader"] = combined["Position"] - 1

# Interactions
combined["progress_x_tyrelife"] = combined["RaceProgress"] * combined["TyreLife"]
combined["tyrelife_x_stint"] = combined["TyreLife"] * combined["Stint"]

# Within-driver-race sequence — sort then group
combined = combined.sort_values(driver_race_key + ["LapNumber"]).reset_index(drop=True)
gdr = combined.groupby(driver_race_key)
combined["tyrelife_rolling3_mean"] = gdr["TyreLife"].transform(
    lambda s: s.rolling(3, min_periods=1).mean())
combined["laptime_rolling3_mean"] = gdr["LapTime (s)"].transform(
    lambda s: s.rolling(3, min_periods=1).mean())
combined["laptime_diff_prev"] = gdr["LapTime (s)"].diff()
combined["stint_lap_idx"] = gdr.cumcount()

# Label-encode cats
for c in ["Driver", "Compound", "Race"]:
    combined[c] = LabelEncoder().fit_transform(combined[c].astype(str))

# Realign by id
train_ids = set(train[ID_COL].tolist())
train_enc = combined[combined[ID_COL].isin(train_ids)].copy()
test_enc = combined[~combined[ID_COL].isin(train_ids)].copy()
target_map = dict(zip(train[ID_COL], train[TARGET]))
train_enc[TARGET] = train_enc[ID_COL].map(target_map)
assert train_enc[TARGET].notna().all()

FEATURE_COLS = [c for c in train_enc.columns if c not in (TARGET, ID_COL)]
print(f"n_features={len(FEATURE_COLS)}")
print("features:", FEATURE_COLS)

X_all = train_enc[FEATURE_COLS].values
y_all = train_enc[TARGET].values.astype(int)
X_test = test_enc[FEATURE_COLS].values
test_ids = test_enc[ID_COL].values

year_idx = FEATURE_COLS.index("Year")
val_mask = X_all[:, year_idx] == 2025
X_tr, X_va = X_all[~val_mask], X_all[val_mask]
y_tr, y_va = y_all[~val_mask], y_all[val_mask]
print(f"tr={X_tr.shape}  va={X_va.shape}  pos_rate tr={y_tr.mean():.4f}  va={y_va.mean():.4f}")

scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
print(f"scale_pos_weight={scale_pos:.2f}")

# ─── XGB with tighter regularization ──────────────────────────────────────────
print("\nTraining XGB v4.1...")
t0 = time.time()
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=20,        # was 4
    reg_lambda=5.0,             # was 1
    reg_alpha=0.5,
    gamma=0.5,
    scale_pos_weight=scale_pos,
    eval_metric="auc",
    early_stopping_rounds=80,
    random_state=42,
    tree_method="hist",
    device="cuda",
    verbosity=0,
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
val_pred = model.predict_proba(X_va)[:, 1]
val_auc = roc_auc_score(y_va, val_pred)
print(f"v4.1 val AUC = {val_auc:.6f}  (took {time.time()-t0:.1f}s, best_iter={model.best_iteration})")

V1 = 0.9485
delta = val_auc - V1
print(f"Δ vs v1 ({V1}): {delta:+.4f}")

fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_}).sort_values(
    "importance", ascending=False)
print("\nTop-20 features:")
print(fi.head(20).to_string(index=False))

# Retrain on full data
n_best = model.best_iteration + 1
print(f"\nRetraining on full data, n_est={n_best}...")
final = xgb.XGBClassifier(
    n_estimators=n_best,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=20,
    reg_lambda=5.0,
    reg_alpha=0.5,
    gamma=0.5,
    scale_pos_weight=scale_pos,
    random_state=42,
    tree_method="hist",
    device="cuda",
    verbosity=0,
)
final.fit(X_all, y_all, verbose=False)
test_preds = final.predict_proba(X_test)[:, 1]

sub = pd.DataFrame({"id": test_ids, "PitNextLap": test_preds})
sub.to_csv(SUB_PATH, index=False)
np.save(OOF_PATH, val_pred)
print(f"submission -> {SUB_PATH}  shape={sub.shape}")

mlflow.set_tracking_uri("databricks")
try:
    mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
except Exception:
    mlflow.set_experiment("/Shared/databricks-ai-intern")

with mlflow.start_run(run_name="f1_pitstops_v4_1_leaner") as run:
    mlflow.log_param("phase", "2_leaner")
    mlflow.log_param("model", "xgboost")
    mlflow.log_param("val_strategy", "year_2025_holdout")
    mlflow.log_param("n_features", len(FEATURE_COLS))
    mlflow.log_param("best_iter", int(n_best))
    mlflow.log_param("min_child_weight", 20)
    mlflow.log_param("reg_lambda", 5.0)
    mlflow.log_metric("val_auc", float(val_auc))
    mlflow.log_metric("delta_vs_v1", float(delta))
    fi.to_csv("/tmp/fi_v4_1.csv", index=False)
    mlflow.log_artifact("/tmp/fi_v4_1.csv")
    mlflow.log_artifact(SUB_PATH)
    run_id = run.info.run_id

print("\n" + "=" * 60)
print("V4.1 REPORT")
print("=" * 60)
print(json.dumps({
    "version": "v4.1_leaner",
    "val_auc": float(val_auc),
    "v1_baseline": V1,
    "delta_vs_v1": float(delta),
    "n_features": len(FEATURE_COLS),
    "best_iter": int(n_best),
    "top5_features": fi["feature"].head(5).tolist(),
    "submission_path": SUB_PATH,
    "mlflow_run": run_id,
    "submit_decision": "SUBMIT" if delta > 0.003 else "HOLD_AND_ITERATE",
}, indent=2))
