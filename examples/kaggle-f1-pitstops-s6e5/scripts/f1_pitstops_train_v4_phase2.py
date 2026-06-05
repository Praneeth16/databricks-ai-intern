# Databricks notebook source
"""
F1 Pit Stops — Phase 2 race-context / cross-entity features.

Encodes the kaggle-tabular-classification skill playbook Phase 2:
  - Per-race-lap aggregates (mean/std/min/max across drivers in same race-lap)
  - Driver-vs-field deltas (own value minus race-lap mean)
  - Position-relative features (gap to leader by laptime + position)
  - Pit-window indicators (RaceProgress x TyreLife interaction)

Locked guardrails from skill:
  - Target = PitNextLap (confirmed against sample_submission)
  - Validation = time-split Year=2025 holdout (NOT 5-fold GroupKF — leaks year)
  - No lag of target (leakage)
  - Single XGB model (Phase 1 baseline + Phase 2 features). Blend comes later.
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

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUBMISSION_PATH = f"{DATA_DIR}/submission_v4_phase2.csv"
OOF_PATH = f"{DATA_DIR}/oof_v4_phase2.npy"

TARGET = "PitNextLap"
ID_COL = "id"

# ─── Load ─────────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print(f"train={train.shape} test={test.shape}")

# Confirm target column (skill playbook Phase 0)
assert TARGET in train.columns, f"target {TARGET!r} not in train"
assert TARGET not in test.columns, f"target leaked into test"
print(f"target positive rate: {train[TARGET].mean():.4f}")

# ─── Phase 2 feature engineering ──────────────────────────────────────────────
combined = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
combined["__is_train"] = np.r_[np.ones(len(train), dtype=bool),
                               np.zeros(len(test), dtype=bool)]

race_lap_key = ["Year", "Race", "LapNumber"]
driver_race_key = ["Year", "Race", "Driver"]
race_key = ["Year", "Race"]

# 2a) Per-race-lap aggregates across drivers
agg_cols = ["LapTime (s)", "TyreLife", "Position",
            "Cumulative_Degradation", "LapTime_Delta"]
print("Building race-lap aggregates...")
for col in agg_cols:
    g = combined.groupby(race_lap_key)[col]
    combined[f"{col}__rl_mean"] = g.transform("mean")
    combined[f"{col}__rl_std"] = g.transform("std")
    combined[f"{col}__rl_min"] = g.transform("min")
    combined[f"{col}__rl_max"] = g.transform("max")
    combined[f"{col}__vs_field"] = combined[col] - combined[f"{col}__rl_mean"]

# 2b) Position-relative
combined["pos_gap_to_leader"] = combined["Position"] - 1
combined["laptime_gap_to_leader"] = combined["LapTime (s)"] - combined["LapTime (s)__rl_min"]
combined["laptime_pct_of_leader"] = combined["LapTime (s)"] / (combined["LapTime (s)__rl_min"] + 1e-6)

# Position change vs field
combined["poschg_vs_field"] = (combined["Position_Change"]
                                - combined.groupby(race_lap_key)["Position_Change"].transform("mean"))

# 2c) Within-driver-race sequence (Phase 3 cheap wins, no target lag)
print("Building within-driver-race rollups...")
gdr = combined.sort_values(driver_race_key + ["LapNumber"]).groupby(driver_race_key)
combined = combined.sort_values(driver_race_key + ["LapNumber"]).reset_index(drop=True)
gdr = combined.groupby(driver_race_key)

combined["tyrelife_rolling3_mean"] = gdr["TyreLife"].transform(
    lambda s: s.rolling(3, min_periods=1).mean())
combined["laptime_rolling3_mean"] = gdr["LapTime (s)"].transform(
    lambda s: s.rolling(3, min_periods=1).mean())
combined["laptime_rolling3_std"] = gdr["LapTime (s)"].transform(
    lambda s: s.rolling(3, min_periods=1).std())
combined["laptime_diff_prev"] = gdr["LapTime (s)"].diff()
combined["position_diff_prev"] = gdr["Position"].diff()
combined["stint_lap_idx"] = gdr.cumcount()  # laps into this race for the driver

# 2d) Pit-window indicators
# Typical pit timing in this race (training-derived; test rows fill by race-mean)
combined["progress_x_tyrelife"] = combined["RaceProgress"] * combined["TyreLife"]
combined["progress_x_pitstop"] = combined["RaceProgress"] * combined["PitStop"]
combined["tyrelife_x_stint"] = combined["TyreLife"] * combined["Stint"]

# Race-level pit timing stats from train only — broadcast to all rows
train_only = combined[combined["__is_train"]].copy()
train_only["__pit_event"] = train.set_index(train.index)[TARGET].values  # aligned
# Mean pit-event RaceProgress per race (only where pit event happens)
pit_progress = (train_only[train_only["__pit_event"] == 1]
                .groupby(race_key)["RaceProgress"].mean()
                .rename("race_pit_progress_mean"))
combined = combined.merge(pit_progress, left_on=race_key, right_index=True, how="left")
combined["race_pit_progress_mean"] = combined["race_pit_progress_mean"].fillna(
    combined["race_pit_progress_mean"].mean())
combined["progress_to_pit_window"] = combined["RaceProgress"] - combined["race_pit_progress_mean"]

# Drop the helper col
combined = combined.drop(columns=["__is_train"])

# ─── Label encode cats ────────────────────────────────────────────────────────
CAT_COLS = ["Driver", "Compound", "Race"]
for c in CAT_COLS:
    le = LabelEncoder()
    combined[c] = le.fit_transform(combined[c].astype(str))

# ─── Split back ───────────────────────────────────────────────────────────────
# preserve original train/test split via id column
train_ids = set(train[ID_COL].tolist())
train_enc = combined[combined[ID_COL].isin(train_ids)].copy()
test_enc = combined[~combined[ID_COL].isin(train_ids)].copy()

# realign train target by id (we sorted earlier)
target_map = dict(zip(train[ID_COL], train[TARGET]))
train_enc[TARGET] = train_enc[ID_COL].map(target_map)
assert train_enc[TARGET].notna().all(), "target alignment failed"

FEATURE_COLS = [c for c in train_enc.columns if c not in (TARGET, ID_COL)]
print(f"feature count: {len(FEATURE_COLS)}")

X_all = train_enc[FEATURE_COLS].values
y_all = train_enc[TARGET].values.astype(int)
X_test = test_enc[FEATURE_COLS].values
test_ids = test_enc[ID_COL].values

# Year is needed for split — find index
year_idx = FEATURE_COLS.index("Year")
val_mask = X_all[:, year_idx] == 2025
train_mask = ~val_mask

X_tr, X_va = X_all[train_mask], X_all[val_mask]
y_tr, y_va = y_all[train_mask], y_all[val_mask]
print(f"tr={X_tr.shape} va={X_va.shape}  pos_rate tr={y_tr.mean():.4f} va={y_va.mean():.4f}")

scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
print(f"scale_pos_weight={scale_pos:.2f}")

# ─── XGB w/ early stopping ────────────────────────────────────────────────────
print("\nTraining XGB Phase 2...")
t0 = time.time()
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=4,
    reg_lambda=1.0,
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
print(f"Phase 2 val AUC = {val_auc:.6f}  (took {time.time()-t0:.1f}s, "
      f"best_iter={model.best_iteration})")

# ─── Compare against locked v1 baseline ───────────────────────────────────────
V1_VAL_AUC = 0.9485  # approx from v1 run
delta = val_auc - V1_VAL_AUC
print(f"Δ vs v1 baseline ({V1_VAL_AUC}): {delta:+.4f}")

# ─── Feature importances ──────────────────────────────────────────────────────
fi = pd.DataFrame({"feature": FEATURE_COLS,
                   "importance": model.feature_importances_}).sort_values(
    "importance", ascending=False)
print("\nTop-15 features:")
print(fi.head(15).to_string(index=False))

# ─── Retrain on all data using best_iteration ─────────────────────────────────
print("\nRetraining on full data for submission...")
n_best = model.best_iteration + 1
final = xgb.XGBClassifier(
    n_estimators=n_best,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=4,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos,
    random_state=42,
    tree_method="hist",
    device="cuda",
    verbosity=0,
)
final.fit(X_all, y_all, verbose=False)
test_preds = final.predict_proba(X_test)[:, 1]

# ─── Save submission + OOF + feature importance ───────────────────────────────
sub = pd.DataFrame({"id": test_ids, "PitNextLap": test_preds})
sub.to_csv(SUBMISSION_PATH, index=False)
np.save(OOF_PATH, val_pred)
print(f"submission -> {SUBMISSION_PATH}  shape={sub.shape}")

# ─── MLflow ───────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("databricks")
try:
    mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
except Exception:
    mlflow.set_experiment("/Shared/databricks-ai-intern")

with mlflow.start_run(run_name="f1_pitstops_v4_phase2") as run:
    mlflow.log_param("phase", "2_race_context")
    mlflow.log_param("model", "xgboost")
    mlflow.log_param("val_strategy", "year_2025_holdout")
    mlflow.log_param("n_features", len(FEATURE_COLS))
    mlflow.log_param("best_iter", int(n_best))
    mlflow.log_metric("val_auc", float(val_auc))
    mlflow.log_metric("delta_vs_v1", float(delta))
    fi.to_csv("/tmp/fi_v4.csv", index=False)
    mlflow.log_artifact("/tmp/fi_v4.csv")
    mlflow.log_artifact(SUBMISSION_PATH)
    run_id = run.info.run_id

# ─── Final report ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 REPORT")
print("=" * 60)
print(json.dumps({
    "phase": 2,
    "val_auc": float(val_auc),
    "v1_baseline": V1_VAL_AUC,
    "delta_vs_v1": float(delta),
    "n_features": len(FEATURE_COLS),
    "best_iter": int(n_best),
    "top5_features": fi["feature"].head(5).tolist(),
    "submission": SUBMISSION_PATH,
    "mlflow_run": run_id,
    "submit_decision": "SUBMIT" if delta > 0.003 else "HOLD_AND_ITERATE",
}, indent=2))
