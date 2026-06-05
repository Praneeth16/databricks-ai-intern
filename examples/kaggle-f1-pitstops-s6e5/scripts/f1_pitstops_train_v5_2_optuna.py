# Databricks notebook source
"""
v5.2 — Optuna-tuned XGB + Optuna-tuned CatBoost blend.

Tune both on time-split val (Year=2025). 20 trials each.
XGB: encoded cats. CB: native cats.
Hill-climb blend of 2 OOF columns.
"""
import sys as _ml_sys, io as _ml_io
_BUF = _ml_io.StringIO()
class _T:
    def __init__(self, *s): self._s = s
    def write(self, b):
        for x in self._s:
            try: x.write(b)
            except: pass
        return len(b) if isinstance(b, str) else 0
    def flush(self):
        for x in self._s:
            try: x.flush()
            except: pass
_ml_sys.stdout = _T(_ml_sys.__stdout__, _BUF)
_ml_sys.stderr = _T(_ml_sys.__stderr__, _BUF)

try:
    import os, sys, time, subprocess, warnings, json
    warnings.filterwarnings("ignore")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "xgboost", "catboost", "optuna", "scikit-learn",
                           "pandas", "numpy", "mlflow"])

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    from catboost import CatBoostClassifier, Pool
    import optuna
    import mlflow
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    DATA_DIR = "/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops"
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    TARGET = "PitNextLap"
    ID_COL = "id"
    print(f"train={train.shape}  test={test.shape}  pos_rate={train[TARGET].mean():.4f}")

    FEATS = ["Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint",
             "TyreLife", "Position", "LapTime (s)", "LapTime_Delta",
             "Cumulative_Degradation", "RaceProgress", "Position_Change"]
    CAT_COLS = ["Driver", "Compound", "Race"]

    combined = pd.concat([train[FEATS], test[FEATS]], axis=0, ignore_index=True)
    n_train = len(train)
    years = train["Year"].values
    val_mask = years == 2025
    y_all = train[TARGET].values.astype(int)
    y_tr = y_all[~val_mask]; y_va = y_all[val_mask]
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"tr_size={(~val_mask).sum()}  va_size={val_mask.sum()}  scale_pos={scale_pos:.2f}")

    # XGB pipeline: encoded
    combined_xgb = combined.copy()
    for c in CAT_COLS:
        combined_xgb[c] = LabelEncoder().fit_transform(combined_xgb[c].astype(str))
    X_xgb_all = combined_xgb.iloc[:n_train].values
    X_xgb_test = combined_xgb.iloc[n_train:].values
    X_xgb_tr, X_xgb_va = X_xgb_all[~val_mask], X_xgb_all[val_mask]

    # CB pipeline: string cats
    combined_cb = combined.copy()
    for c in CAT_COLS:
        combined_cb[c] = combined_cb[c].astype(str)
    X_cb_all = combined_cb.iloc[:n_train].reset_index(drop=True)
    X_cb_test = combined_cb.iloc[n_train:].reset_index(drop=True)
    X_cb_tr = X_cb_all[~val_mask].reset_index(drop=True)
    X_cb_va = X_cb_all[val_mask].reset_index(drop=True)

    # ─── Optuna XGB ───
    def xgb_objective(trial):
        p = dict(
            n_estimators=1500,
            max_depth=trial.suggest_int("max_depth", 5, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 3.0),
            scale_pos_weight=scale_pos,
            eval_metric="auc",
            early_stopping_rounds=50,
            tree_method="hist",
            device="cuda",
            random_state=42,
            verbosity=0,
        )
        m = xgb.XGBClassifier(**p)
        m.fit(X_xgb_tr, y_tr, eval_set=[(X_xgb_va, y_va)], verbose=False)
        return roc_auc_score(y_va, m.predict_proba(X_xgb_va)[:, 1])

    print("\n--- Optuna XGB 20 trials ---")
    t0 = time.time()
    study_x = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=42))
    study_x.optimize(xgb_objective, n_trials=20, show_progress_bar=False)
    print(f"XGB best val AUC: {study_x.best_value:.6f}  ({time.time()-t0:.1f}s)")
    print(f"XGB best params: {study_x.best_params}")

    # Refit best XGB to get OOF + best_iter
    xp_best = dict(study_x.best_params, n_estimators=1500,
                   scale_pos_weight=scale_pos, eval_metric="auc",
                   early_stopping_rounds=50, tree_method="hist",
                   device="cuda", random_state=42, verbosity=0)
    m_xgb = xgb.XGBClassifier(**xp_best)
    m_xgb.fit(X_xgb_tr, y_tr, eval_set=[(X_xgb_va, y_va)], verbose=False)
    oof_xgb = m_xgb.predict_proba(X_xgb_va)[:, 1]
    auc_xgb = roc_auc_score(y_va, oof_xgb)
    print(f"XGB refit val AUC: {auc_xgb:.6f}  best_iter={m_xgb.best_iteration}")

    # Retrain XGB on full data
    xf = dict(xp_best, n_estimators=m_xgb.best_iteration + 1)
    xf.pop("early_stopping_rounds", None); xf.pop("eval_metric", None)
    m_xgb_full = xgb.XGBClassifier(**xf)
    m_xgb_full.fit(X_xgb_all, y_all, verbose=False)
    test_xgb = m_xgb_full.predict_proba(X_xgb_test)[:, 1]

    # ─── Optuna CatBoost ───
    def cb_objective(trial):
        p = dict(
            iterations=2000,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            depth=trial.suggest_int("depth", 5, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            border_count=trial.suggest_categorical("border_count", [64, 128, 254]),
            eval_metric="AUC",
            random_seed=42,
            verbose=0,
            early_stopping_rounds=80,
            scale_pos_weight=scale_pos,
            task_type="GPU",
        )
        m = CatBoostClassifier(**p)
        m.fit(Pool(X_cb_tr, y_tr, cat_features=CAT_COLS),
              eval_set=Pool(X_cb_va, y_va, cat_features=CAT_COLS),
              verbose=0)
        return roc_auc_score(y_va, m.predict_proba(X_cb_va)[:, 1])

    print("\n--- Optuna CatBoost 20 trials ---")
    t0 = time.time()
    study_c = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=42))
    study_c.optimize(cb_objective, n_trials=20, show_progress_bar=False)
    print(f"CB best val AUC: {study_c.best_value:.6f}  ({time.time()-t0:.1f}s)")
    print(f"CB best params: {study_c.best_params}")

    cp_best = dict(study_c.best_params, iterations=2000,
                   eval_metric="AUC", random_seed=42, verbose=0,
                   early_stopping_rounds=80, scale_pos_weight=scale_pos,
                   task_type="GPU")
    m_cb = CatBoostClassifier(**cp_best)
    m_cb.fit(Pool(X_cb_tr, y_tr, cat_features=CAT_COLS),
             eval_set=Pool(X_cb_va, y_va, cat_features=CAT_COLS), verbose=0)
    oof_cb = m_cb.predict_proba(X_cb_va)[:, 1]
    auc_cb = roc_auc_score(y_va, oof_cb)
    print(f"CB refit val AUC: {auc_cb:.6f}  best_iter={m_cb.get_best_iteration()}")

    cf = dict(cp_best, iterations=m_cb.get_best_iteration() + 1)
    cf.pop("early_stopping_rounds", None); cf.pop("eval_metric", None)
    m_cb_full = CatBoostClassifier(**cf)
    m_cb_full.fit(Pool(X_cb_all, y_all, cat_features=CAT_COLS), verbose=0)
    test_cb = m_cb_full.predict_proba(X_cb_test)[:, 1]

    # ─── Hill-climb blend ───
    OOF = np.column_stack([oof_xgb, oof_cb])
    TEST = np.column_stack([test_xgb, test_cb])
    names = ["xgb_opt", "cb_opt"]
    per_auc = [roc_auc_score(y_va, OOF[:, i]) for i in range(2)]
    print(f"\nPer-model val AUC: xgb={per_auc[0]:.6f}  cb={per_auc[1]:.6f}")

    eq = OOF.mean(axis=1)
    eq_auc = roc_auc_score(y_va, eq)
    print(f"Equal-weight blend: {eq_auc:.6f}")

    # Fine grid for 2-model blend
    best_w, best_a = 0.5, 0.0
    for w in np.linspace(0.0, 1.0, 101):
        blend = w * OOF[:, 0] + (1 - w) * OOF[:, 1]
        a = roc_auc_score(y_va, blend)
        if a > best_a:
            best_a = a; best_w = w
    print(f"Best XGB weight = {best_w:.2f}  blend AUC = {best_a:.6f}")

    weights = np.array([best_w, 1 - best_w])
    test_blend = TEST @ weights

    best_single = max(per_auc)
    blend_lift = best_a - best_single
    v1_lb_anchor = 0.94820
    pred_lb = best_a + 0.046  # observed val→LB gap
    print(f"Blend lift over best single: {blend_lift:+.6f}")
    print(f"Predicted LB (val+0.046 gap): {pred_lb:.4f}  vs v1 LB {v1_lb_anchor}")

    SUB_PATH = f"{DATA_DIR}/submission_v5_2_optuna.csv"
    OOF_PATH = f"{DATA_DIR}/oof_v5_2_optuna.csv"
    test_ids = test[ID_COL].values
    pd.DataFrame({"id": test_ids, "PitNextLap": test_blend}).to_csv(SUB_PATH, index=False)
    pd.DataFrame(OOF, columns=names).to_csv(OOF_PATH, index=False)
    print(f"submission -> {SUB_PATH}")

    mlflow.set_tracking_uri("databricks")
    try:
        mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
    except Exception:
        mlflow.set_experiment("/Shared/databricks-ai-intern")

    with mlflow.start_run(run_name="f1_pitstops_v5_2_optuna") as run:
        mlflow.log_param("version", "v5_2_optuna")
        mlflow.log_param("n_trials_xgb", 20)
        mlflow.log_param("n_trials_cb", 20)
        mlflow.log_metric("xgb_val_auc", float(auc_xgb))
        mlflow.log_metric("cb_val_auc", float(auc_cb))
        mlflow.log_metric("blend_val_auc", float(best_a))
        mlflow.log_metric("blend_lift", float(blend_lift))
        mlflow.log_metric("xgb_weight", float(best_w))
        mlflow.log_metric("predicted_lb", float(pred_lb))
        for k, v in study_x.best_params.items():
            mlflow.log_param(f"xgb_{k}", v)
        for k, v in study_c.best_params.items():
            mlflow.log_param(f"cb_{k}", v)
        mlflow.log_artifact(SUB_PATH)
        run_id = run.info.run_id

    print("\n" + "=" * 60)
    print("V5.2 REPORT")
    print("=" * 60)
    print(json.dumps({
        "version": "v5_2_optuna",
        "xgb_val_auc": float(auc_xgb),
        "cb_val_auc": float(auc_cb),
        "blend_val_auc": float(best_a),
        "xgb_weight": float(best_w),
        "blend_lift_vs_best_single": float(blend_lift),
        "predicted_lb": float(pred_lb),
        "v1_lb": v1_lb_anchor,
        "submission": SUB_PATH,
        "mlflow_run": run_id,
        "submit_decision": "SUBMIT" if pred_lb > v1_lb_anchor + 0.001 else "HOLD",
    }, indent=2))
except BaseException as _err:
    print(f"FATAL: {_err!r}")
    raise
finally:
    try:
        dbutils.notebook.exit(_BUF.getvalue()[-6000:])
    except Exception:
        pass
