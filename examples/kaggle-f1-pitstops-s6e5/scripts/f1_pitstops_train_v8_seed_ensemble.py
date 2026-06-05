# Databricks notebook source
"""
v8 — v5.4 Optuna params + 5-seed ensemble per model. Year=2025 holdout.

Why:
  v7 (race-grouped CV) regressed to LB 0.94668 — race-shift CV picked
  wrong weights for year-shift test set. The noisy Year=2025 holdout
  was the right LB proxy. Keep it. Add seed averaging on top.

Pipeline:
  1. Same data prep as v5.4 (14 bare feats, label-enc for XGB/LGBM, native CB).
  2. Optuna 20 trials per model on Year=2025 val (reproduces v5.4 since same
     TPE seed=42 + same trial count).
  3. EACH refit (val + full-train) becomes 5-seed ensemble.
     val_oof = mean(5 seeds × predict(val))
     test    = mean(5 seeds × full-train refit × predict(test))
  4. Optuna 200-trial weight search on seed-avg OOF + 41-grid sanity.

Comparison anchors (Year=2025 val_auc → LB):
  v5.2  0.9049 → 0.94924
  v5.4  0.9054 → 0.94896
  v7    OOF 0.9296 → 0.94668  (CV mismatch, ignore as LB predictor)
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
                           "xgboost", "lightgbm", "catboost", "optuna",
                           "scikit-learn", "pandas", "numpy", "mlflow"])

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier, Pool
    import optuna
    import mlflow
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    DATA_DIR = "/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops"
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    TARGET, ID_COL = "PitNextLap", "id"
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
    y_tr, y_va = y_all[~val_mask], y_all[val_mask]
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"tr={(~val_mask).sum()}  va={val_mask.sum()}  scale_pos={scale_pos:.2f}")

    combined_enc = combined.copy()
    for c in CAT_COLS:
        combined_enc[c] = LabelEncoder().fit_transform(combined_enc[c].astype(str))
    X_enc_all = combined_enc.iloc[:n_train].values
    X_enc_test = combined_enc.iloc[n_train:].values
    X_enc_tr, X_enc_va = X_enc_all[~val_mask], X_enc_all[val_mask]

    combined_cb = combined.copy()
    for c in CAT_COLS:
        combined_cb[c] = combined_cb[c].astype(str)
    X_cb_all_df = combined_cb.iloc[:n_train].reset_index(drop=True)
    X_cb_test_df = combined_cb.iloc[n_train:].reset_index(drop=True)
    X_cb_tr = X_cb_all_df[~val_mask].reset_index(drop=True)
    X_cb_va = X_cb_all_df[val_mask].reset_index(drop=True)

    SEEDS = [42, 7, 555, 2024, 13]
    N_SEEDS = len(SEEDS)
    print(f"seeds = {SEEDS}")

    # ─── Optuna XGB ───
    def xgb_obj(trial):
        p = dict(
            n_estimators=1500,
            max_depth=trial.suggest_int("max_depth", 5, 11),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 3.0),
            scale_pos_weight=scale_pos, eval_metric="auc",
            early_stopping_rounds=50, tree_method="hist",
            device="cuda", random_state=42, verbosity=0,
        )
        m = xgb.XGBClassifier(**p)
        m.fit(X_enc_tr, y_tr, eval_set=[(X_enc_va, y_va)], verbose=False)
        return roc_auc_score(y_va, m.predict_proba(X_enc_va)[:, 1])

    print("\n--- Optuna XGB 20 trials ---")
    t0 = time.time()
    sx = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
    sx.optimize(xgb_obj, n_trials=20, show_progress_bar=False)
    print(f"XGB best val_auc={sx.best_value:.6f}  ({time.time()-t0:.1f}s)")
    print(f"XGB best_params={sx.best_params}")

    # ─── XGB 5-seed ensemble: val + test ───
    print("\n--- XGB 5-seed ensemble ---")
    xp_base = dict(sx.best_params, n_estimators=1500, scale_pos_weight=scale_pos,
                   eval_metric="auc", early_stopping_rounds=50, tree_method="hist",
                   device="cuda", verbosity=0)
    oof_x_seeds = np.zeros(val_mask.sum())
    test_x_seeds = np.zeros(len(X_enc_test))
    for sd in SEEDS:
        t0 = time.time()
        m = xgb.XGBClassifier(**xp_base, random_state=sd)
        m.fit(X_enc_tr, y_tr, eval_set=[(X_enc_va, y_va)], verbose=False)
        oof_x_seeds += m.predict_proba(X_enc_va)[:, 1] / N_SEEDS
        bi = m.best_iteration + 1
        # full-train refit, fixed n_estimators
        xp_full = dict(xp_base, n_estimators=bi, random_state=sd)
        xp_full.pop("early_stopping_rounds", None); xp_full.pop("eval_metric", None)
        mf = xgb.XGBClassifier(**xp_full)
        mf.fit(X_enc_all, y_all, verbose=False)
        test_x_seeds += mf.predict_proba(X_enc_test)[:, 1] / N_SEEDS
        print(f"  seed{sd}: best_iter={bi-1}  ({time.time()-t0:.1f}s)")
    oof_x = oof_x_seeds
    test_x = test_x_seeds
    auc_x = roc_auc_score(y_va, oof_x)
    print(f"XGB 5-seed avg val_auc={auc_x:.6f}")

    # ─── Optuna LGBM ───
    def lgb_obj(trial):
        p = dict(
            n_estimators=1500,
            max_depth=trial.suggest_int("max_depth", 5, 11),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 0.1, 10.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            scale_pos_weight=scale_pos, metric="auc",
            random_state=42, verbose=-1, n_jobs=-1,
        )
        m = lgb.LGBMClassifier(**p)
        m.fit(X_enc_tr, y_tr, eval_set=[(X_enc_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(0)])
        return roc_auc_score(y_va, m.predict_proba(X_enc_va)[:, 1])

    print("\n--- Optuna LGBM 20 trials ---")
    t0 = time.time()
    sl = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
    sl.optimize(lgb_obj, n_trials=20, show_progress_bar=False)
    print(f"LGBM best val_auc={sl.best_value:.6f}  ({time.time()-t0:.1f}s)")
    print(f"LGBM best_params={sl.best_params}")

    print("\n--- LGBM 5-seed ensemble ---")
    lp_base = dict(sl.best_params, n_estimators=1500, scale_pos_weight=scale_pos,
                   metric="auc", verbose=-1, n_jobs=-1)
    oof_l_seeds = np.zeros(val_mask.sum())
    test_l_seeds = np.zeros(len(X_enc_test))
    for sd in SEEDS:
        t0 = time.time()
        m = lgb.LGBMClassifier(**lp_base, random_state=sd)
        m.fit(X_enc_tr, y_tr, eval_set=[(X_enc_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(0)])
        oof_l_seeds += m.predict_proba(X_enc_va)[:, 1] / N_SEEDS
        bi = m.best_iteration_ + 1
        lp_full = dict(lp_base, n_estimators=bi, random_state=sd)
        lp_full.pop("metric", None)
        mf = lgb.LGBMClassifier(**lp_full)
        mf.fit(X_enc_all, y_all)
        test_l_seeds += mf.predict_proba(X_enc_test)[:, 1] / N_SEEDS
        print(f"  seed{sd}: best_iter={bi-1}  ({time.time()-t0:.1f}s)")
    oof_l = oof_l_seeds
    test_l = test_l_seeds
    auc_l = roc_auc_score(y_va, oof_l)
    print(f"LGBM 5-seed avg val_auc={auc_l:.6f}")

    # ─── Optuna CatBoost ───
    def cb_obj(trial):
        p = dict(
            iterations=2000,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            depth=trial.suggest_int("depth", 5, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            border_count=trial.suggest_categorical("border_count", [64, 128, 254]),
            eval_metric="AUC", random_seed=42, verbose=0,
            early_stopping_rounds=80,
            scale_pos_weight=scale_pos, task_type="GPU",
        )
        m = CatBoostClassifier(**p)
        m.fit(Pool(X_cb_tr, y_tr, cat_features=CAT_COLS),
              eval_set=Pool(X_cb_va, y_va, cat_features=CAT_COLS), verbose=0)
        return roc_auc_score(y_va, m.predict_proba(X_cb_va)[:, 1])

    print("\n--- Optuna CatBoost 20 trials ---")
    t0 = time.time()
    sc = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
    sc.optimize(cb_obj, n_trials=20, show_progress_bar=False)
    print(f"CB best val_auc={sc.best_value:.6f}  ({time.time()-t0:.1f}s)")
    print(f"CB best_params={sc.best_params}")

    print("\n--- CB 5-seed ensemble ---")
    cp_base = dict(sc.best_params, iterations=2000, eval_metric="AUC",
                   verbose=0, early_stopping_rounds=80,
                   scale_pos_weight=scale_pos, task_type="GPU")
    oof_c_seeds = np.zeros(val_mask.sum())
    test_c_seeds = np.zeros(len(X_cb_test_df))
    for sd in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**cp_base, random_seed=sd)
        m.fit(Pool(X_cb_tr, y_tr, cat_features=CAT_COLS),
              eval_set=Pool(X_cb_va, y_va, cat_features=CAT_COLS), verbose=0)
        oof_c_seeds += m.predict_proba(X_cb_va)[:, 1] / N_SEEDS
        bi = m.get_best_iteration() + 1
        cp_full = dict(cp_base, iterations=bi, random_seed=sd)
        cp_full.pop("early_stopping_rounds", None); cp_full.pop("eval_metric", None)
        mf = CatBoostClassifier(**cp_full)
        mf.fit(Pool(X_cb_all_df, y_all, cat_features=CAT_COLS), verbose=0)
        test_c_seeds += mf.predict_proba(X_cb_test_df)[:, 1] / N_SEEDS
        print(f"  seed{sd}: best_iter={bi-1}  ({time.time()-t0:.1f}s)")
    oof_c = oof_c_seeds
    test_c = test_c_seeds
    auc_c = roc_auc_score(y_va, oof_c)
    print(f"CB 5-seed avg val_auc={auc_c:.6f}")

    # ─── Blend ───
    OOF = np.column_stack([oof_x, oof_l, oof_c])
    TEST = np.column_stack([test_x, test_l, test_c])
    names = ["xgb", "lgbm", "cb"]
    per = [auc_x, auc_l, auc_c]
    print("\nPer-model 5-seed val_auc:")
    for n, a in zip(names, per): print(f"  {n:5s} {a:.6f}")
    eq = OOF.mean(axis=1); eq_auc = roc_auc_score(y_va, eq)
    print(f"Equal-weight blend: {eq_auc:.6f}")

    def blend_obj(trial):
        a = trial.suggest_float("w_xgb_raw", 0.0, 1.0)
        b = trial.suggest_float("w_lgbm_raw", 0.0, 1.0)
        c = trial.suggest_float("w_cb_raw", 0.0, 1.0)
        s = a + b + c
        if s < 1e-9: return 0.0
        w = np.array([a, b, c]) / s
        return roc_auc_score(y_va, OOF @ w)

    print("\n--- Optuna blend weights 300 trials ---")
    t0 = time.time()
    sw = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
    sw.optimize(blend_obj, n_trials=300, show_progress_bar=False)
    bp = sw.best_params
    s = bp["w_xgb_raw"] + bp["w_lgbm_raw"] + bp["w_cb_raw"]
    weights = np.array([bp["w_xgb_raw"], bp["w_lgbm_raw"], bp["w_cb_raw"]]) / s
    blend_auc = sw.best_value
    print(f"Optuna blend val_auc={blend_auc:.6f}  ({time.time()-t0:.1f}s)")
    print(f"Weights: xgb={weights[0]:.4f}  lgbm={weights[1]:.4f}  cb={weights[2]:.4f}")

    print("\n--- 41-grid sanity-check ---")
    best_g = 0.0; best_wg = None
    grid = np.linspace(0, 1, 41)
    for wx in grid:
        for wl in grid:
            if wx + wl > 1: continue
            wc = 1 - wx - wl
            blend = wx * OOF[:, 0] + wl * OOF[:, 1] + wc * OOF[:, 2]
            a = roc_auc_score(y_va, blend)
            if a > best_g:
                best_g = a; best_wg = (wx, wl, wc)
    print(f"Grid blend val_auc={best_g:.6f}  weights={best_wg}")
    if best_g > blend_auc:
        weights = np.array(best_wg); blend_auc = best_g
        print("Using grid weights (better)")
    else:
        print("Using Optuna weights")

    test_blend = TEST @ weights
    best_single = max(per)
    blend_lift = blend_auc - best_single
    pred_lb = blend_auc + 0.044
    print(f"\nBlend lift vs best single: {blend_lift:+.6f}")
    print(f"Predicted LB (val+0.044): {pred_lb:.4f}  vs v5.2 LB 0.94924, v5.4 LB 0.94896")

    SUB_PATH = f"{DATA_DIR}/submission_v8_seed_ensemble.csv"
    OOF_PATH = f"{DATA_DIR}/oof_v8_seed_ensemble.csv"
    test_ids = test[ID_COL].values
    pd.DataFrame({"id": test_ids, "PitNextLap": test_blend}).to_csv(SUB_PATH, index=False)
    pd.DataFrame(OOF, columns=names).to_csv(OOF_PATH, index=False)
    print(f"submission -> {SUB_PATH}")

    mlflow.set_tracking_uri("databricks")
    try:
        mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
    except Exception:
        mlflow.set_experiment("/Shared/databricks-ai-intern")

    with mlflow.start_run(run_name="f1_pitstops_v8_seed_ensemble") as run:
        mlflow.log_param("version", "v8_seed_ensemble")
        mlflow.log_param("seeds", str(SEEDS))
        mlflow.log_param("n_seeds", N_SEEDS)
        mlflow.log_param("models", ",".join(names))
        for k, v in sx.best_params.items(): mlflow.log_param(f"xgb_{k}", v)
        for k, v in sl.best_params.items(): mlflow.log_param(f"lgb_{k}", v)
        for k, v in sc.best_params.items(): mlflow.log_param(f"cb_{k}", v)
        for n, a, w in zip(names, per, weights):
            mlflow.log_metric(f"{n}_val_auc_5seed", float(a))
            mlflow.log_metric(f"{n}_weight", float(w))
        mlflow.log_metric("equal_blend_auc", float(eq_auc))
        mlflow.log_metric("final_blend_auc", float(blend_auc))
        mlflow.log_metric("blend_lift", float(blend_lift))
        mlflow.log_metric("predicted_lb", float(pred_lb))
        mlflow.log_artifact(SUB_PATH)
        mlflow.log_artifact(OOF_PATH)
        run_id = run.info.run_id

    print("\n" + "=" * 60)
    print("V8 REPORT")
    print("=" * 60)
    report = {
        "version": "v8_seed_ensemble",
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "per_model_val_auc": dict(zip(names, [float(a) for a in per])),
        "equal_blend_auc": float(eq_auc),
        "final_blend_auc": float(blend_auc),
        "weights": dict(zip(names, [float(w) for w in weights])),
        "blend_lift_vs_best_single": float(blend_lift),
        "predicted_lb": float(pred_lb),
        "v5_2_lb": 0.94924, "v5_4_lb": 0.94896, "v7_lb": 0.94668,
        "submission": SUB_PATH, "mlflow_run": run_id,
        "submit_decision": ("SUBMIT" if pred_lb > 0.94924 + 0.0005 else "HOLD"),
    }
    print(json.dumps(report, indent=2))
except BaseException as _err:
    print(f"FATAL: {_err!r}")
    raise
finally:
    try:
        dbutils.notebook.exit(_BUF.getvalue()[-7000:])
    except Exception:
        pass
