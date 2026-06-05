# Databricks notebook source
"""
v6 — v4.1's leaner 26-feature engineering + v5.3's 3-model Optuna-tuned blend.

  Features (26 total):
    Base 14: Driver, Compound, Race, Year, PitStop, LapNumber, Stint, TyreLife,
             Position, LapTime (s), LapTime_Delta, Cumulative_Degradation,
             RaceProgress, Position_Change
    vs_field deltas (4): LapTime__vs_field, TyreLife__vs_field,
                         Position__vs_field, Cumulative_Degradation__vs_field
    Leader gaps (2):     laptime_gap_to_leader, pos_gap_to_leader
    Interactions (2):    progress_x_tyrelife, tyrelife_x_stint
    Within-driver (4):   tyrelife_rolling3_mean, laptime_rolling3_mean,
                         laptime_diff_prev, stint_lap_idx

  Models:
    XGB  Optuna 20 trials   (cats label-encoded, GPU)
    LGBM Optuna 20 trials   (cats label-encoded, CPU n_jobs=-1)
    CB   Optuna 20 trials   (cats native strings, GPU)

  Weight search: Optuna 200 trials over 3-simplex + 41-grid sanity check.

  Validation: time-split, Year=2025 holdout.
  Hypothesis: feature engineering, not tuning depth, is the dominant lever.
              v4.1 (XGB only + these feats) hit LB 0.94513 with no blend;
              v5.4 (no FE + 3-model blend) hit LB 0.94896.
              Combining both targets >0.95.
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
    TRAIN_PATH = f"{DATA_DIR}/train.csv"
    TEST_PATH = f"{DATA_DIR}/test.csv"
    TARGET, ID_COL = "PitNextLap", "id"

    print("Loading...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"train={train.shape}  test={test.shape}  pos_rate={train[TARGET].mean():.4f}")

    # ─── Feature engineering (combined train+test, cats still strings) ─────
    combined = pd.concat([train.drop(columns=[TARGET]), test],
                         axis=0, ignore_index=True)

    race_lap_key = ["Year", "Race", "LapNumber"]
    driver_race_key = ["Year", "Race", "Driver"]
    vs_field_cols = ["LapTime (s)", "TyreLife", "Position", "Cumulative_Degradation"]

    print("Building vs_field deltas...")
    for col in vs_field_cols:
        m = combined.groupby(race_lap_key)[col].transform("mean")
        combined[f"{col}__vs_field"] = combined[col] - m

    print("Building leader gaps...")
    lt_min = combined.groupby(race_lap_key)["LapTime (s)"].transform("min")
    combined["laptime_gap_to_leader"] = combined["LapTime (s)"] - lt_min
    combined["pos_gap_to_leader"] = combined["Position"] - 1

    print("Building interactions...")
    combined["progress_x_tyrelife"] = combined["RaceProgress"] * combined["TyreLife"]
    combined["tyrelife_x_stint"] = combined["TyreLife"] * combined["Stint"]

    print("Building within-driver rollups...")
    combined = combined.sort_values(driver_race_key + ["LapNumber"]).reset_index(drop=True)
    gdr = combined.groupby(driver_race_key)
    combined["tyrelife_rolling3_mean"] = gdr["TyreLife"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    combined["laptime_rolling3_mean"] = gdr["LapTime (s)"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    combined["laptime_diff_prev"] = gdr["LapTime (s)"].diff()
    combined["stint_lap_idx"] = gdr.cumcount()

    # ─── Split rows back into train/test by id ─────────────────────────────
    train_ids = set(train[ID_COL].tolist())
    train_enc_full = combined[combined[ID_COL].isin(train_ids)].copy().reset_index(drop=True)
    test_enc_full = combined[~combined[ID_COL].isin(train_ids)].copy().reset_index(drop=True)
    target_map = dict(zip(train[ID_COL], train[TARGET]))
    train_enc_full[TARGET] = train_enc_full[ID_COL].map(target_map)
    assert train_enc_full[TARGET].notna().all(), "Target alignment failed"

    CAT_COLS = ["Driver", "Compound", "Race"]
    FEATURE_COLS = [c for c in train_enc_full.columns if c not in (TARGET, ID_COL)]
    print(f"n_features={len(FEATURE_COLS)}")
    print("features:", FEATURE_COLS)

    # ─── XGB/LGBM matrix: label-encode the cats ────────────────────────────
    train_enc = train_enc_full.copy()
    test_enc = test_enc_full.copy()
    for c in CAT_COLS:
        le = LabelEncoder().fit(pd.concat([train_enc[c], test_enc[c]]).astype(str))
        train_enc[c] = le.transform(train_enc[c].astype(str))
        test_enc[c] = le.transform(test_enc[c].astype(str))

    X_enc_all = train_enc[FEATURE_COLS].values
    X_enc_test = test_enc[FEATURE_COLS].values
    y_all = train_enc[TARGET].values.astype(int)
    test_ids = test_enc_full[ID_COL].values

    year_idx = FEATURE_COLS.index("Year")
    val_mask = X_enc_all[:, year_idx] == 2025
    X_enc_tr, X_enc_va = X_enc_all[~val_mask], X_enc_all[val_mask]
    y_tr, y_va = y_all[~val_mask], y_all[val_mask]
    print(f"tr={X_enc_tr.shape}  va={X_enc_va.shape}  pos_rate tr={y_tr.mean():.4f}  va={y_va.mean():.4f}")

    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"scale_pos_weight={scale_pos:.2f}")

    # ─── CatBoost matrix: same FE, cats kept as strings ────────────────────
    train_cb = train_enc_full[FEATURE_COLS].copy().reset_index(drop=True)
    test_cb = test_enc_full[FEATURE_COLS].copy().reset_index(drop=True)
    for c in CAT_COLS:
        train_cb[c] = train_cb[c].astype(str)
        test_cb[c] = test_cb[c].astype(str)
    # CatBoost can't handle NaN in numeric features by default — fill
    num_cols_cb = [c for c in FEATURE_COLS if c not in CAT_COLS]
    train_cb[num_cols_cb] = train_cb[num_cols_cb].fillna(0.0)
    test_cb[num_cols_cb] = test_cb[num_cols_cb].fillna(0.0)
    X_cb_tr = train_cb.loc[~val_mask].reset_index(drop=True)
    X_cb_va = train_cb.loc[val_mask].reset_index(drop=True)
    X_cb_all = train_cb
    X_cb_test = test_cb

    # ─── Optuna XGB ───
    def xgb_obj(trial):
        p = dict(
            n_estimators=2000,
            max_depth=trial.suggest_int("max_depth", 5, 11),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 3.0),
            scale_pos_weight=scale_pos, eval_metric="auc",
            early_stopping_rounds=60, tree_method="hist",
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

    xp = dict(sx.best_params, n_estimators=2000, scale_pos_weight=scale_pos,
              eval_metric="auc", early_stopping_rounds=60, tree_method="hist",
              device="cuda", random_state=42, verbosity=0)
    m_x = xgb.XGBClassifier(**xp)
    m_x.fit(X_enc_tr, y_tr, eval_set=[(X_enc_va, y_va)], verbose=False)
    oof_x = m_x.predict_proba(X_enc_va)[:, 1]
    auc_x = roc_auc_score(y_va, oof_x)
    print(f"XGB refit val_auc={auc_x:.6f}  best_iter={m_x.best_iteration}")
    xf = dict(xp, n_estimators=m_x.best_iteration + 1)
    xf.pop("early_stopping_rounds", None); xf.pop("eval_metric", None)
    mxf = xgb.XGBClassifier(**xf)
    mxf.fit(X_enc_all, y_all, verbose=False)
    test_x = mxf.predict_proba(X_enc_test)[:, 1]

    # ─── Optuna LGBM ───
    def lgb_obj(trial):
        p = dict(
            n_estimators=2000,
            max_depth=trial.suggest_int("max_depth", 5, 11),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 0.1, 20.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            scale_pos_weight=scale_pos, metric="auc",
            random_state=42, verbose=-1, n_jobs=-1,
        )
        m = lgb.LGBMClassifier(**p)
        m.fit(X_enc_tr, y_tr, eval_set=[(X_enc_va, y_va)],
              callbacks=[lgb.early_stopping(60, verbose=False),
                         lgb.log_evaluation(0)])
        return roc_auc_score(y_va, m.predict_proba(X_enc_va)[:, 1])

    print("\n--- Optuna LGBM 20 trials ---")
    t0 = time.time()
    sl = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
    sl.optimize(lgb_obj, n_trials=20, show_progress_bar=False)
    print(f"LGBM best val_auc={sl.best_value:.6f}  ({time.time()-t0:.1f}s)")
    print(f"LGBM best_params={sl.best_params}")

    lp = dict(sl.best_params, n_estimators=2000, scale_pos_weight=scale_pos,
              metric="auc", random_state=42, verbose=-1, n_jobs=-1)
    m_l = lgb.LGBMClassifier(**lp)
    m_l.fit(X_enc_tr, y_tr, eval_set=[(X_enc_va, y_va)],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    oof_l = m_l.predict_proba(X_enc_va)[:, 1]
    auc_l = roc_auc_score(y_va, oof_l)
    print(f"LGBM refit val_auc={auc_l:.6f}  best_iter={m_l.best_iteration_}")
    lf = dict(lp, n_estimators=m_l.best_iteration_ + 1)
    lf.pop("metric", None)
    mlf = lgb.LGBMClassifier(**lf)
    mlf.fit(X_enc_all, y_all)
    test_l = mlf.predict_proba(X_enc_test)[:, 1]

    # ─── Optuna CatBoost (GPU, native cats) ───
    def cb_obj(trial):
        p = dict(
            iterations=2500,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            depth=trial.suggest_int("depth", 5, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
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

    cp = dict(sc.best_params, iterations=2500, eval_metric="AUC", random_seed=42,
              verbose=0, early_stopping_rounds=80, scale_pos_weight=scale_pos,
              task_type="GPU")
    m_c = CatBoostClassifier(**cp)
    m_c.fit(Pool(X_cb_tr, y_tr, cat_features=CAT_COLS),
            eval_set=Pool(X_cb_va, y_va, cat_features=CAT_COLS), verbose=0)
    oof_c = m_c.predict_proba(X_cb_va)[:, 1]
    auc_c = roc_auc_score(y_va, oof_c)
    print(f"CB refit val_auc={auc_c:.6f}  best_iter={m_c.get_best_iteration()}")
    cf = dict(cp, iterations=m_c.get_best_iteration() + 1)
    cf.pop("early_stopping_rounds", None); cf.pop("eval_metric", None)
    mcf = CatBoostClassifier(**cf)
    mcf.fit(Pool(X_cb_all, y_all, cat_features=CAT_COLS), verbose=0)
    test_c = mcf.predict_proba(X_cb_test)[:, 1]

    # ─── OOF matrix + Optuna ensemble weight tuning ───
    OOF = np.column_stack([oof_x, oof_l, oof_c])
    TEST = np.column_stack([test_x, test_l, test_c])
    names = ["xgb", "lgbm", "cb"]
    per = [roc_auc_score(y_va, OOF[:, i]) for i in range(3)]
    print("\nPer-model val AUC:")
    for n, a in zip(names, per):
        print(f"  {n:5s} {a:.6f}")

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

    print("\n--- Optuna ensemble weights 200 trials ---")
    t0 = time.time()
    sw = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
    sw.optimize(blend_obj, n_trials=200, show_progress_bar=False)
    bp = sw.best_params
    s = bp["w_xgb_raw"] + bp["w_lgbm_raw"] + bp["w_cb_raw"]
    weights = np.array([bp["w_xgb_raw"], bp["w_lgbm_raw"], bp["w_cb_raw"]]) / s
    blend_auc = sw.best_value
    print(f"Optuna blend val_auc={blend_auc:.6f}  ({time.time()-t0:.1f}s)")
    print(f"Weights: xgb={weights[0]:.4f}  lgbm={weights[1]:.4f}  cb={weights[2]:.4f}")

    print("\n--- Fine grid sanity-check ---")
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
        weights = np.array(best_wg)
        blend_auc = best_g
        print("Using grid weights (better)")
    else:
        print("Using Optuna weights")

    test_blend = TEST @ weights
    best_single = max(per)
    blend_lift = blend_auc - best_single
    pred_lb_v4_1_gap = blend_auc + (0.94513 - 0.8988)
    pred_lb_v5_4_gap = blend_auc + (0.94896 - 0.9054)
    pred_lb = (pred_lb_v4_1_gap + pred_lb_v5_4_gap) / 2
    print(f"\nBlend lift vs best single: {blend_lift:+.6f}")
    print(f"Predicted LB (v4.1 gap): {pred_lb_v4_1_gap:.4f}")
    print(f"Predicted LB (v5.4 gap): {pred_lb_v5_4_gap:.4f}")
    print(f"Predicted LB (avg gap):  {pred_lb:.4f}  vs v5.2 LB 0.94924")

    SUB_PATH = f"{DATA_DIR}/submission_v6_feats_blend.csv"
    OOF_PATH = f"{DATA_DIR}/oof_v6_feats_blend.csv"
    pd.DataFrame({"id": test_ids, "PitNextLap": test_blend}).to_csv(SUB_PATH, index=False)
    pd.DataFrame(OOF, columns=names).to_csv(OOF_PATH, index=False)
    print(f"submission -> {SUB_PATH}")

    mlflow.set_tracking_uri("databricks")
    try:
        mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
    except Exception:
        mlflow.set_experiment("/Shared/databricks-ai-intern")

    with mlflow.start_run(run_name="f1_pitstops_v6_feats_blend") as run:
        mlflow.log_param("version", "v6_feats_blend")
        mlflow.log_param("models", ",".join(names))
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("features", json.dumps(FEATURE_COLS))
        for k, v in sx.best_params.items(): mlflow.log_param(f"xgb_{k}", v)
        for k, v in sl.best_params.items(): mlflow.log_param(f"lgb_{k}", v)
        for k, v in sc.best_params.items(): mlflow.log_param(f"cb_{k}", v)
        for n, a, w in zip(names, per, weights):
            mlflow.log_metric(f"{n}_val_auc", float(a))
            mlflow.log_metric(f"{n}_weight", float(w))
        mlflow.log_metric("equal_blend_auc", float(eq_auc))
        mlflow.log_metric("optuna_blend_auc", float(sw.best_value))
        mlflow.log_metric("grid_blend_auc", float(best_g))
        mlflow.log_metric("final_blend_auc", float(blend_auc))
        mlflow.log_metric("blend_lift", float(blend_lift))
        mlflow.log_metric("predicted_lb_v41_gap", float(pred_lb_v4_1_gap))
        mlflow.log_metric("predicted_lb_v54_gap", float(pred_lb_v5_4_gap))
        mlflow.log_metric("predicted_lb_avg", float(pred_lb))
        mlflow.log_artifact(SUB_PATH)
        run_id = run.info.run_id

    print("\n" + "=" * 60)
    print("V6 REPORT")
    print("=" * 60)
    report = {
        "version": "v6_feats_blend",
        "n_features": len(FEATURE_COLS),
        "per_model_val_auc": dict(zip(names, [float(a) for a in per])),
        "equal_blend_auc": float(eq_auc),
        "final_blend_auc": float(blend_auc),
        "weights": dict(zip(names, [float(w) for w in weights])),
        "blend_lift_vs_best_single": float(blend_lift),
        "predicted_lb_v41_gap": float(pred_lb_v4_1_gap),
        "predicted_lb_v54_gap": float(pred_lb_v5_4_gap),
        "predicted_lb_avg": float(pred_lb),
        "v5_2_lb": 0.94924,
        "v5_4_lb": 0.94896,
        "v4_1_lb": 0.94513,
        "submission": SUB_PATH,
        "mlflow_run": run_id,
        "submit_decision": ("SUBMIT" if pred_lb > 0.94924 + 0.001 else "HOLD"),
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
