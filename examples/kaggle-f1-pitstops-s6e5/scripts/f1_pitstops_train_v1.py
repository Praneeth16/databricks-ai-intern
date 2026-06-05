"""
F1 Pit Stops Prediction — Kaggle Playground S6E5
XGBoost + LightGBM + Optuna tuning, MLflow logging, UC model registration.
"""
import os
import time
import warnings
warnings.filterwarnings("ignore")

# Install deps
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "xgboost", "lightgbm", "optuna", "scikit-learn", "pandas", "mlflow"])

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import optuna
import mlflow
import mlflow.sklearn
import pickle

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = "/Volumes/serverless_lakebase_praneeth_catalog/ml_intern_test/scratch/f1_pitstops"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUBMISSION_PATH = f"{DATA_DIR}/submission.csv"
MODEL_NAME = "serverless_lakebase_praneeth_catalog.ml_intern_test.f1_pitstops_clf"

# ─── MLflow setup ────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/ml-intern")

# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Class balance
print(f"\nTarget distribution:\n{train['PitNextLap'].value_counts(normalize=True)}")
print(f"Positive rate: {train['PitNextLap'].mean():.4f}")

# ─── Feature Engineering ─────────────────────────────────────────────────────
TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Compound", "Race"]
NUM_COLS = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
            "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
            "RaceProgress", "Position_Change"]

# Combine for consistent encoding
combined = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)

# Label encode categoricals (works well with tree models)
label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
    label_encoders[col] = le

# Split back
train_enc = combined.iloc[:len(train)].copy()
test_enc = combined.iloc[len(train):].copy()

# Feature columns (drop id)
FEATURE_COLS = CAT_COLS + NUM_COLS
X_all = train_enc[FEATURE_COLS].values
y_all = train[TARGET].values
X_test = test_enc[FEATURE_COLS].values
test_ids = test[ID_COL].values

# ─── Train/Val Split: Year-based (2025 as val) ──────────────────────────────
year_col_idx = FEATURE_COLS.index("Year")
years_encoded = train_enc["Year"].values

# Get the encoded value for 2025
year_2025_enc = label_encoders["Year"].transform(["2025"])[0] if "Year" in label_encoders else None

# Actually Year is numeric, not label-encoded. Let me use original year values
train_years = train["Year"].values

val_mask = train_years == 2025
train_mask = ~val_mask

X_train, X_val = X_all[train_mask], X_all[val_mask]
y_train, y_val = y_all[train_mask], y_all[val_mask]

print(f"\nTrain set: {X_train.shape[0]} rows (Years < 2025)")
print(f"Val set:   {X_val.shape[0]} rows (Year = 2025)")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Val positive rate:   {y_val.mean():.4f}")

# ─── Baseline Models ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING BASELINE MODELS")
print("="*60)

# Scale pos weight for imbalanced classes
scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"scale_pos_weight: {scale_pos:.2f}")

# XGBoost baseline
print("\n--- XGBoost Baseline ---")
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=42,
    tree_method="hist",
    device="cuda",
    verbosity=0,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_val_pred)
print(f"XGBoost val AUC: {xgb_auc:.6f} (took {time.time()-t0:.1f}s)")

# LightGBM baseline
print("\n--- LightGBM Baseline ---")
t0 = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    metric="auc",
    random_state=42,
    device="gpu",
    verbose=-1,
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)
lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
lgb_auc = roc_auc_score(y_val, lgb_val_pred)
print(f"LightGBM val AUC: {lgb_auc:.6f} (took {time.time()-t0:.1f}s)")

# Pick winner
winner = "xgboost" if xgb_auc >= lgb_auc else "lightgbm"
winner_auc = max(xgb_auc, lgb_auc)
print(f"\n>>> Winner: {winner} (AUC={winner_auc:.6f})")

# ─── Optuna Hyperparameter Tuning on Winner ──────────────────────────────────
print("\n" + "="*60)
print(f"OPTUNA TUNING ({winner.upper()}, 20 trials)")
print("="*60)

best_auc = winner_auc
best_model = xgb_model if winner == "xgboost" else lgb_model
best_params = {}

def objective(trial):
    global best_auc, best_model, best_params
    
    if winner == "xgboost":
        params = {
            "n_estimators": 1000,
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos,
            "eval_metric": "auc",
            "early_stopping_rounds": 50,
            "random_state": 42,
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict_proba(X_val)[:, 1]
    else:
        params = {
            "n_estimators": 1000,
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 31, 256),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos,
            "metric": "auc",
            "random_state": 42,
            "device": "gpu",
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, pred)
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_params = params.copy()
        print(f"  New best AUC: {auc:.6f}")
    return auc

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, show_progress_bar=False)

print(f"\nBest Optuna AUC: {study.best_value:.6f}")
print(f"Best params: {study.best_params}")

# ─── Retrain best model on ALL training data (optional boost) ────────────────
# Actually, let's keep the model trained on train split (not val) for proper eval reporting
# But we'll also train a final model on ALL data for submission
print("\n" + "="*60)
print("RETRAINING BEST MODEL ON ALL DATA FOR SUBMISSION")
print("="*60)

if winner == "xgboost":
    final_params = {k: v for k, v in best_params.items() 
                    if k not in ["eval_metric", "early_stopping_rounds"]}
    # Use best n_estimators from early stopping
    if hasattr(best_model, 'best_iteration'):
        final_params["n_estimators"] = best_model.best_iteration + 1
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_all, y_all, verbose=False)
else:
    final_params = {k: v for k, v in best_params.items() if k != "metric"}
    if hasattr(best_model, 'best_iteration_') and best_model.best_iteration_ > 0:
        final_params["n_estimators"] = best_model.best_iteration_ + 1
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_all, y_all)

print(f"Final model trained with n_estimators={final_params.get('n_estimators', 'default')}")

# ─── Feature Importance ──────────────────────────────────────────────────────
if winner == "xgboost":
    importances = best_model.feature_importances_
else:
    importances = best_model.feature_importances_

feat_imp = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop-10 Feature Importances:")
print(feat_imp.head(10).to_string(index=False))

# ─── Generate Submission ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("GENERATING SUBMISSION")
print("="*60)

test_preds = final_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({"id": test_ids, "PitNextLap": test_preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(f"Submission shape: {submission.shape}")
print(f"\nFirst 10 rows:")
print(submission.head(10).to_string(index=False))
print(f"\nPrediction stats: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

# ─── MLflow Logging ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOGGING TO MLFLOW")
print("="*60)

with mlflow.start_run(run_name="f1_pitstops_best") as run:
    # Log params
    mlflow.log_param("model_type", winner)
    mlflow.log_param("n_trials_optuna", 20)
    mlflow.log_param("val_strategy", "year_2025_holdout")
    mlflow.log_param("n_train", X_train.shape[0])
    mlflow.log_param("n_val", X_val.shape[0])
    for k, v in study.best_params.items():
        mlflow.log_param(f"best_{k}", v)
    
    # Log metrics
    mlflow.log_metric("xgb_baseline_auc", xgb_auc)
    mlflow.log_metric("lgb_baseline_auc", lgb_auc)
    mlflow.log_metric("tuned_val_auc", best_auc)
    mlflow.log_metric("best_optuna_auc", study.best_value)
    
    # Log feature importances as artifact
    feat_imp.to_csv("/tmp/feature_importances.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importances.csv")
    
    # Log submission as artifact
    mlflow.log_artifact(SUBMISSION_PATH)
    
    # Log model
    if winner == "xgboost":
        mlflow.xgboost.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
    else:
        mlflow.lightgbm.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
    
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    print(f"Model registered to: {MODEL_NAME}")

# ─── Final Summary ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"XGBoost baseline AUC:  {xgb_auc:.6f}")
print(f"LightGBM baseline AUC: {lgb_auc:.6f}")
print(f"Winner: {winner}")
print(f"Tuned winner AUC:      {best_auc:.6f}")
print(f"Top-5 features: {feat_imp['feature'].head(5).tolist()}")
print(f"UC Model: {MODEL_NAME}")
print(f"MLflow Run ID: {run_id}")
print(f"Submission: {SUBMISSION_PATH}")
print(f"\nFirst 5 rows of submission:")
print(submission.head(5).to_string(index=False))
print("\nDONE!")
