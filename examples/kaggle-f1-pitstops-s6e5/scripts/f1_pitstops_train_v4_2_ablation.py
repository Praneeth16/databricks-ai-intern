# Databricks notebook source
"""
v4.2 ABLATION — Three configs in one job to find what broke v4/v4.1.

A: v1 baseline features + v1 hparams (expect ~0.948)
B: v1 features + ONLY laptime_gap_to_leader + LapTime__vs_field  (minimal Phase 2)
C: v1 features + all vs_field deltas only (no rolling, no interactions)
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
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
TARGET = "PitNextLap"
ID_COL = "id"

print(f"train={train.shape}  test={test.shape}")

# Same FE pipeline as v4.1 — but we'll cherry-pick features per config
combined = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
race_lap_key = ["Year", "Race", "LapNumber"]

for col in ["LapTime (s)", "TyreLife", "Position", "Cumulative_Degradation"]:
    m = combined.groupby(race_lap_key)[col].transform("mean")
    combined[f"{col}__vs_field"] = combined[col] - m

lt_min = combined.groupby(race_lap_key)["LapTime (s)"].transform("min")
combined["laptime_gap_to_leader"] = combined["LapTime (s)"] - lt_min

for c in ["Driver", "Compound", "Race"]:
    combined[c] = LabelEncoder().fit_transform(combined[c].astype(str))

# Realign
train_ids = set(train[ID_COL].tolist())
train_enc = combined[combined[ID_COL].isin(train_ids)].copy()
test_enc = combined[~combined[ID_COL].isin(train_ids)].copy()
target_map = dict(zip(train[ID_COL], train[TARGET]))
train_enc[TARGET] = train_enc[ID_COL].map(target_map)
assert train_enc[TARGET].notna().all()

V1_FEATS = ["Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint",
            "TyreLife", "Position", "LapTime (s)", "LapTime_Delta",
            "Cumulative_Degradation", "RaceProgress", "Position_Change"]

CONFIGS = {
    "A_v1_only": V1_FEATS,
    "B_v1_plus_min": V1_FEATS + ["laptime_gap_to_leader", "LapTime (s)__vs_field"],
    "C_v1_plus_vs_field": V1_FEATS + [c for c in train_enc.columns if c.endswith("__vs_field")],
}

V1_HPARAMS = dict(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="auc", early_stopping_rounds=50,
    random_state=42, tree_method="hist", device="cuda", verbosity=0,
)

y_all = train_enc[TARGET].values.astype(int)
year_col = train_enc["Year"].values
val_mask = year_col == 2025

results = {}
for name, feats in CONFIGS.items():
    print(f"\n=== {name}  ({len(feats)} feats) ===")
    feats = [f for f in feats if f in train_enc.columns]
    X_all = train_enc[feats].values
    X_tr, X_va = X_all[~val_mask], X_all[val_mask]
    y_tr, y_va = y_all[~val_mask], y_all[val_mask]
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = dict(V1_HPARAMS, scale_pos_weight=scale_pos)
    t0 = time.time()
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred = m.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred)
    results[name] = {"val_auc": float(auc), "n_feats": len(feats),
                     "best_iter": int(m.best_iteration), "time_s": round(time.time()-t0, 1)}
    print(f"  val AUC = {auc:.6f}  best_iter={m.best_iteration}  time={time.time()-t0:.1f}s")

print("\n=== ABLATION SUMMARY ===")
print(json.dumps(results, indent=2))

mlflow.set_tracking_uri("databricks")
try:
    mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
except Exception:
    mlflow.set_experiment("/Shared/databricks-ai-intern")

with mlflow.start_run(run_name="f1_pitstops_v4_2_ablation") as run:
    for name, r in results.items():
        mlflow.log_metric(f"{name}_val_auc", r["val_auc"])
        mlflow.log_metric(f"{name}_best_iter", r["best_iter"])
    mlflow.log_param("v1_baseline_lb", 0.94820)
    mlflow.log_dict(results, "results.json")
