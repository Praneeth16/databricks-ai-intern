# Databricks notebook source
"""
v7 — StratifiedGroupKFold(by Race) + 3-seed ensembling + Optuna blend.

Why:
  Val differences <0.002 on Year=2025 split don't map to LB ordering
  (see v5.2 val 0.9049 LB 0.94924  vs  v5.4 val 0.9054 LB 0.94896).
  Single-year holdout is at noise floor. Replace with:
    1. 5-fold StratifiedGroupKFold by Race → robust full-train OOF
    2. 3 seeds per model → averages away seed noise (+0.001 typical)
  Tuning kept FIXED at near-v5.4 reasonable defaults — re-tuning would
  add noise without lift. The OOF + seed averaging is the real ask.

Features: v5.4 bare 14 (no FE — v6 showed FE hurt distribution-aligned val).
Cats: LabelEncoded for XGB/LGBM, native strings for CatBoost.

Fits: 5 folds × 3 seeds × 3 models = 45 total. Budget ~50min serverless A10.
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
    from sklearn.model_selection import StratifiedGroupKFold
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
    y_all = train[TARGET].values.astype(int)
    groups_race_str = train["Race"].astype(str).values  # groupkfold key

    # Encoded for XGB/LGBM (fit on train+test for stable encoding)
    combined_enc = combined.copy()
    for c in CAT_COLS:
        combined_enc[c] = LabelEncoder().fit_transform(combined_enc[c].astype(str))
    X_enc_all = combined_enc.iloc[:n_train].values
    X_enc_test = combined_enc.iloc[n_train:].values

    # String for CatBoost
    combined_cb = combined.copy()
    for c in CAT_COLS:
        combined_cb[c] = combined_cb[c].astype(str)
    X_cb_all = combined_cb.iloc[:n_train].reset_index(drop=True)
    X_cb_test = combined_cb.iloc[n_train:].reset_index(drop=True)

    # scale_pos uses full train (rough estimate — per-fold computed below)
    scale_pos_global = (y_all == 0).sum() / max((y_all == 1).sum(), 1)
    print(f"scale_pos_global={scale_pos_global:.2f}")

    # 5-fold StratifiedGroupKFold by Race
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(sgkf.split(X_enc_all, y_all, groups=groups_race_str))
    print("Fold sizes:")
    for i, (tr, va) in enumerate(fold_splits):
        u_tr = len(np.unique(groups_race_str[tr]))
        u_va = len(np.unique(groups_race_str[va]))
        print(f"  fold{i}: tr={len(tr)} ({u_tr} races)  va={len(va)} ({u_va} races)  "
              f"pos_va={y_all[va].mean():.4f}")

    SEEDS = [42, 7, 555]
    N_SEEDS = len(SEEDS)
    N_FOLDS = 5

    # ─── Frozen hyperparameters (sane mid-range from v5.4 Optuna spaces) ──
    XGB_PARAMS = dict(
        n_estimators=2000, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_lambda=2.0, reg_alpha=0.01, gamma=0.5,
        eval_metric="auc", early_stopping_rounds=80,
        tree_method="hist", device="cuda", verbosity=0,
    )
    LGB_PARAMS = dict(
        n_estimators=2000, max_depth=8, num_leaves=127, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=2.0,
        reg_lambda=2.0, reg_alpha=0.01,
        metric="auc", verbose=-1, n_jobs=-1,
    )
    CB_PARAMS = dict(
        iterations=2500, learning_rate=0.05, depth=8,
        l2_leaf_reg=5.0, random_strength=2.0, bagging_temperature=0.5,
        border_count=128, eval_metric="AUC", verbose=0,
        early_stopping_rounds=100, task_type="GPU",
    )

    # OOF + test accumulators
    oof_x = np.zeros(n_train); oof_l = np.zeros(n_train); oof_c = np.zeros(n_train)
    test_x = np.zeros(len(X_enc_test))
    test_l = np.zeros(len(X_enc_test))
    test_c = np.zeros(len(X_cb_test))

    t_start = time.time()
    for fi, (tr_idx, va_idx) in enumerate(fold_splits):
        X_tr_enc, X_va_enc = X_enc_all[tr_idx], X_enc_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]
        X_tr_cb = X_cb_all.iloc[tr_idx].reset_index(drop=True)
        X_va_cb = X_cb_all.iloc[va_idx].reset_index(drop=True)
        scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        fold_oof_x = np.zeros(len(va_idx))
        fold_oof_l = np.zeros(len(va_idx))
        fold_oof_c = np.zeros(len(va_idx))
        fold_te_x = np.zeros(len(X_enc_test))
        fold_te_l = np.zeros(len(X_enc_test))
        fold_te_c = np.zeros(len(X_cb_test))

        for seed in SEEDS:
            # XGB
            t0 = time.time()
            mx = xgb.XGBClassifier(**XGB_PARAMS,
                                   random_state=seed,
                                   scale_pos_weight=scale_pos)
            mx.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)], verbose=False)
            p_va = mx.predict_proba(X_va_enc)[:, 1]
            p_te = mx.predict_proba(X_enc_test)[:, 1]
            fold_oof_x += p_va / N_SEEDS
            fold_te_x += p_te / N_SEEDS
            print(f"  fold{fi} seed{seed} XGB  auc={roc_auc_score(y_va, p_va):.5f}  "
                  f"best_iter={mx.best_iteration}  ({time.time()-t0:.1f}s)")

            # LGBM
            t0 = time.time()
            ml = lgb.LGBMClassifier(**LGB_PARAMS,
                                    random_state=seed,
                                    scale_pos_weight=scale_pos)
            ml.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)],
                   callbacks=[lgb.early_stopping(80, verbose=False),
                              lgb.log_evaluation(0)])
            p_va = ml.predict_proba(X_va_enc)[:, 1]
            p_te = ml.predict_proba(X_enc_test)[:, 1]
            fold_oof_l += p_va / N_SEEDS
            fold_te_l += p_te / N_SEEDS
            print(f"  fold{fi} seed{seed} LGBM auc={roc_auc_score(y_va, p_va):.5f}  "
                  f"best_iter={ml.best_iteration_}  ({time.time()-t0:.1f}s)")

            # CatBoost
            t0 = time.time()
            mc = CatBoostClassifier(**CB_PARAMS,
                                    random_seed=seed,
                                    scale_pos_weight=scale_pos)
            mc.fit(Pool(X_tr_cb, y_tr, cat_features=CAT_COLS),
                   eval_set=Pool(X_va_cb, y_va, cat_features=CAT_COLS), verbose=0)
            p_va = mc.predict_proba(X_va_cb)[:, 1]
            p_te = mc.predict_proba(X_cb_test)[:, 1]
            fold_oof_c += p_va / N_SEEDS
            fold_te_c += p_te / N_SEEDS
            print(f"  fold{fi} seed{seed} CB   auc={roc_auc_score(y_va, p_va):.5f}  "
                  f"best_iter={mc.get_best_iteration()}  ({time.time()-t0:.1f}s)")

        # Per-fold seed-averaged metrics
        f_auc_x = roc_auc_score(y_va, fold_oof_x)
        f_auc_l = roc_auc_score(y_va, fold_oof_l)
        f_auc_c = roc_auc_score(y_va, fold_oof_c)
        print(f"  fold{fi} seed-avg: xgb={f_auc_x:.5f} lgbm={f_auc_l:.5f} cb={f_auc_c:.5f}  "
              f"elapsed={time.time()-t_start:.0f}s")

        oof_x[va_idx] = fold_oof_x
        oof_l[va_idx] = fold_oof_l
        oof_c[va_idx] = fold_oof_c
        test_x += fold_te_x / N_FOLDS
        test_l += fold_te_l / N_FOLDS
        test_c += fold_te_c / N_FOLDS

    # ─── Per-model OOF + blend ───
    auc_x = roc_auc_score(y_all, oof_x)
    auc_l = roc_auc_score(y_all, oof_l)
    auc_c = roc_auc_score(y_all, oof_c)
    print(f"\nFull-OOF per-model AUC (seed+fold averaged):")
    print(f"  xgb  {auc_x:.6f}")
    print(f"  lgbm {auc_l:.6f}")
    print(f"  cb   {auc_c:.6f}")

    OOF = np.column_stack([oof_x, oof_l, oof_c])
    TEST = np.column_stack([test_x, test_l, test_c])
    names = ["xgb", "lgbm", "cb"]
    per = [auc_x, auc_l, auc_c]

    eq = OOF.mean(axis=1); eq_auc = roc_auc_score(y_all, eq)
    print(f"Equal-weight blend OOF: {eq_auc:.6f}")

    # Optuna weights on FULL OOF (not single-fold val)
    def blend_obj(trial):
        a = trial.suggest_float("w_xgb_raw", 0.0, 1.0)
        b = trial.suggest_float("w_lgbm_raw", 0.0, 1.0)
        c = trial.suggest_float("w_cb_raw", 0.0, 1.0)
        s = a + b + c
        if s < 1e-9: return 0.0
        w = np.array([a, b, c]) / s
        return roc_auc_score(y_all, OOF @ w)

    print("\n--- Optuna OOF blend weights 300 trials ---")
    t0 = time.time()
    sw = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
    sw.optimize(blend_obj, n_trials=300, show_progress_bar=False)
    bp = sw.best_params
    s = bp["w_xgb_raw"] + bp["w_lgbm_raw"] + bp["w_cb_raw"]
    weights = np.array([bp["w_xgb_raw"], bp["w_lgbm_raw"], bp["w_cb_raw"]]) / s
    blend_auc = sw.best_value
    print(f"Optuna OOF blend AUC={blend_auc:.6f}  ({time.time()-t0:.1f}s)")
    print(f"Weights: xgb={weights[0]:.4f}  lgbm={weights[1]:.4f}  cb={weights[2]:.4f}")

    print("\n--- 41-grid sanity-check ---")
    best_g = 0.0; best_wg = None
    grid = np.linspace(0, 1, 41)
    for wx in grid:
        for wl in grid:
            if wx + wl > 1: continue
            wc = 1 - wx - wl
            blend = wx * OOF[:, 0] + wl * OOF[:, 1] + wc * OOF[:, 2]
            a = roc_auc_score(y_all, blend)
            if a > best_g:
                best_g = a; best_wg = (wx, wl, wc)
    print(f"Grid OOF blend AUC={best_g:.6f}  weights={best_wg}")
    if best_g > blend_auc:
        weights = np.array(best_wg); blend_auc = best_g
        print("Using grid weights (better)")
    else:
        print("Using Optuna weights")

    test_blend = TEST @ weights
    best_single = max(per)
    blend_lift = blend_auc - best_single

    # Predicted LB: OOF AUC has different scale than Year=2025 holdout AUC,
    # but apply rough offset from v5.4 (val 0.9054 → LB 0.94896).
    # Note: OOF AUC tends to be HIGHER than single-year-val AUC because
    # train data leaks year structure into folds. Don't over-interpret.
    pred_lb_naive = blend_auc + (0.94896 - 0.9054)  # apply v5.4 gap
    print(f"\nBlend lift vs best single: {blend_lift:+.6f}")
    print(f"Predicted LB (naive v5.4 gap): {pred_lb_naive:.4f}  vs v5.2 LB 0.94924")
    print("NOTE: OOF AUC is not directly comparable to prior val AUC numbers.")

    SUB_PATH = f"{DATA_DIR}/submission_v7_kfold_seeds.csv"
    OOF_PATH = f"{DATA_DIR}/oof_v7_kfold_seeds.csv"
    test_ids = test[ID_COL].values
    pd.DataFrame({"id": test_ids, "PitNextLap": test_blend}).to_csv(SUB_PATH, index=False)
    pd.DataFrame(OOF, columns=names).to_csv(OOF_PATH, index=False)
    print(f"submission -> {SUB_PATH}")

    mlflow.set_tracking_uri("databricks")
    try:
        mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
    except Exception:
        mlflow.set_experiment("/Shared/databricks-ai-intern")

    with mlflow.start_run(run_name="f1_pitstops_v7_kfold_seeds") as run:
        mlflow.log_param("version", "v7_kfold_seeds")
        mlflow.log_param("models", ",".join(names))
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("n_seeds", N_SEEDS)
        mlflow.log_param("seeds", str(SEEDS))
        mlflow.log_param("cv", "StratifiedGroupKFold(by Race)")
        for k, v in XGB_PARAMS.items(): mlflow.log_param(f"xgb_{k}", str(v))
        for k, v in LGB_PARAMS.items(): mlflow.log_param(f"lgb_{k}", str(v))
        for k, v in CB_PARAMS.items(): mlflow.log_param(f"cb_{k}", str(v))
        for n, a, w in zip(names, per, weights):
            mlflow.log_metric(f"{n}_oof_auc", float(a))
            mlflow.log_metric(f"{n}_weight", float(w))
        mlflow.log_metric("equal_blend_oof", float(eq_auc))
        mlflow.log_metric("optuna_blend_oof", float(sw.best_value))
        mlflow.log_metric("grid_blend_oof", float(best_g))
        mlflow.log_metric("final_blend_oof", float(blend_auc))
        mlflow.log_metric("blend_lift", float(blend_lift))
        mlflow.log_metric("pred_lb_naive", float(pred_lb_naive))
        mlflow.log_artifact(SUB_PATH)
        mlflow.log_artifact(OOF_PATH)
        run_id = run.info.run_id

    print("\n" + "=" * 60)
    print("V7 REPORT")
    print("=" * 60)
    report = {
        "version": "v7_kfold_seeds",
        "n_folds": N_FOLDS, "n_seeds": N_SEEDS, "seeds": SEEDS,
        "cv": "StratifiedGroupKFold(by Race)",
        "per_model_oof_auc": dict(zip(names, [float(a) for a in per])),
        "equal_blend_oof": float(eq_auc),
        "final_blend_oof": float(blend_auc),
        "weights": dict(zip(names, [float(w) for w in weights])),
        "blend_lift_vs_best_single": float(blend_lift),
        "pred_lb_naive": float(pred_lb_naive),
        "v5_2_lb": 0.94924, "v5_4_lb": 0.94896,
        "submission": SUB_PATH, "mlflow_run": run_id,
        "submit_decision": "SUBMIT_IF_OOF_HIGHER_THAN_V5_4_OOF",
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
