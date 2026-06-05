# Databricks notebook source
"""
v10 — Hazard-feature model (Codex #2). Single XGB + CatBoost, equal-weight blend.

Goal: produce a *decorrelated* prediction vector vs v5.2/v5.4/v8 so v11 can
blend with real diversity. Per Codex, blending only lifts AUC if base preds
are not >0.99 spearman-correlated.

Hazard-style features (binary: will driver pit NEXT lap = next-event hazard):
  Bins (quantile, fit on train year<2025):
    TyreLifeBin       (10 bins)
    RaceProgressBin   (10 bins)
  Categorical interactions (strings; CB native cats):
    Compound x Stint
    Compound x TyreLifeBin
    Stint x TyreLifeBin
  Numeric, train-only stats (no leakage):
    pit_rate_cs_tlb      = smoothed mean(PitNextLap) per (Compound, Stint, TyreLifeBin)
    dist_to_median_pit   = TyreLife - median TyreLife on rows where PitNextLap=1 per (Compound, Stint)
    pit_rate_lap_pct     = smoothed mean(PitNextLap) per (Race, LapNumber percentile bin)

All target-encoded stats computed on TRAIN ONLY (year<2025), then applied to
val (year=2025) and test. Marginal self-leak in train (own-row contribution
to its own mean) is acceptable for 350k rows.

Models: XGB (Optuna 10 trials) + CB (Optuna 8 trials). Equal-weight blend.
Single submission v10. Print rank-correlation vs v5.2/v5.4/v8 to gauge
diversity before any v11 blend step.
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
                           "pandas", "numpy", "mlflow", "kaggle"])

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
    TARGET, ID_COL = "PitNextLap", "id"
    print(f"train={train.shape}  test={test.shape}  pos_rate={train[TARGET].mean():.4f}")

    # ─── Build hazard features on combined (FE only; target stats train-only) ───
    BASE = ["Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint",
            "TyreLife", "Position", "LapTime (s)", "LapTime_Delta",
            "Cumulative_Degradation", "RaceProgress", "Position_Change"]
    combined = pd.concat([train[BASE + [ID_COL]], test[BASE + [ID_COL]]],
                         axis=0, ignore_index=True)

    # Quantile bins — fit on train year<2025 only to avoid future leak
    train_obs = train[train["Year"] < 2025]
    print(f"train_obs (year<2025): {len(train_obs)}  pos_rate={train_obs[TARGET].mean():.4f}")

    tl_q = np.quantile(train_obs["TyreLife"].values, np.linspace(0, 1, 11))
    rp_q = np.quantile(train_obs["RaceProgress"].values, np.linspace(0, 1, 11))
    # Make edges strictly increasing
    tl_q = np.unique(tl_q); rp_q = np.unique(rp_q)
    combined["TyreLifeBin"] = pd.cut(combined["TyreLife"], bins=tl_q,
                                     labels=False, include_lowest=True).fillna(0).astype(int)
    combined["RaceProgressBin"] = pd.cut(combined["RaceProgress"], bins=rp_q,
                                         labels=False, include_lowest=True).fillna(0).astype(int)
    # Cat string interactions
    combined["compound_stint"] = combined["Compound"].astype(str) + "_" + combined["Stint"].astype(str)
    combined["compound_tlb"]   = combined["Compound"].astype(str) + "_" + combined["TyreLifeBin"].astype(str)
    combined["stint_tlb"]      = combined["Stint"].astype(str) + "_" + combined["TyreLifeBin"].astype(str)

    # ─── Target-encoded stats from train_obs only ───
    ALPHA = 20.0  # beta-prior smoothing strength
    global_rate = train_obs[TARGET].mean()

    def smoothed_mean(df, keys):
        g = df.groupby(keys)[TARGET].agg(["sum", "count"])
        smoothed = (g["sum"] + ALPHA * global_rate) / (g["count"] + ALPHA)
        return smoothed.to_dict()

    tr_for_enc = train_obs.copy()
    tr_for_enc["TyreLifeBin"] = pd.cut(tr_for_enc["TyreLife"], bins=tl_q,
                                       labels=False, include_lowest=True).fillna(0).astype(int)
    tr_for_enc["RaceProgressBin"] = pd.cut(tr_for_enc["RaceProgress"], bins=rp_q,
                                           labels=False, include_lowest=True).fillna(0).astype(int)
    tr_for_enc["compound_stint"] = tr_for_enc["Compound"].astype(str) + "_" + tr_for_enc["Stint"].astype(str)
    tr_for_enc["compound_tlb"]   = tr_for_enc["Compound"].astype(str) + "_" + tr_for_enc["TyreLifeBin"].astype(str)
    tr_for_enc["stint_tlb"]      = tr_for_enc["Stint"].astype(str) + "_" + tr_for_enc["TyreLifeBin"].astype(str)

    pr_cs_tlb = smoothed_mean(tr_for_enc, ["Compound", "Stint", "TyreLifeBin"])
    pr_cs     = smoothed_mean(tr_for_enc, ["Compound", "Stint"])
    pr_lap    = smoothed_mean(tr_for_enc, ["Race", "RaceProgressBin"])

    def lookup(d, keys_arr, default=global_rate):
        return np.array([d.get(tuple(r), default) for r in keys_arr])

    combined["pit_rate_cs_tlb"] = lookup(pr_cs_tlb,
        combined[["Compound", "Stint", "TyreLifeBin"]].astype({"Stint": int, "TyreLifeBin": int}).values)
    combined["pit_rate_cs"]     = lookup(pr_cs,
        combined[["Compound", "Stint"]].astype({"Stint": int}).values)
    combined["pit_rate_lap"]    = lookup(pr_lap,
        combined[["Race", "RaceProgressBin"]].astype({"RaceProgressBin": int}).values)

    # Median pit TyreLife per (Compound, Stint) — only rows where pit_next_lap=1
    pit_rows = tr_for_enc[tr_for_enc[TARGET] == 1]
    median_tl = pit_rows.groupby(["Compound", "Stint"])["TyreLife"].median().to_dict()
    global_median_tl = pit_rows["TyreLife"].median()
    cs_keys = combined[["Compound", "Stint"]].astype({"Stint": int}).values
    median_lookup = np.array([median_tl.get(tuple(r), global_median_tl) for r in cs_keys])
    combined["dist_to_median_pit_tl"] = combined["TyreLife"].values - median_lookup

    # ─── Final feature lists ───
    NEW_CAT = ["TyreLifeBin", "RaceProgressBin", "compound_stint", "compound_tlb", "stint_tlb"]
    NEW_NUM = ["pit_rate_cs_tlb", "pit_rate_cs", "pit_rate_lap", "dist_to_median_pit_tl"]
    BASE_CAT = ["Driver", "Compound", "Race"]
    CAT_COLS = BASE_CAT + NEW_CAT
    FEATURE_COLS = BASE + NEW_CAT + NEW_NUM
    print(f"n_features={len(FEATURE_COLS)}  cat={len(CAT_COLS)}")
    print("features:", FEATURE_COLS)

    # XGB/LGBM-style: label encode all categoricals
    combined_enc = combined.copy()
    for c in CAT_COLS:
        combined_enc[c] = LabelEncoder().fit_transform(combined_enc[c].astype(str))

    n_train = len(train)
    X_enc_all = combined_enc.iloc[:n_train][FEATURE_COLS].values
    X_enc_test = combined_enc.iloc[n_train:][FEATURE_COLS].values
    y_all = train[TARGET].values.astype(int)
    years = train["Year"].values
    val_mask = years == 2025
    X_enc_tr, X_enc_va = X_enc_all[~val_mask], X_enc_all[val_mask]
    y_tr, y_va = y_all[~val_mask], y_all[val_mask]
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"tr={(~val_mask).sum()}  va={val_mask.sum()}  scale_pos={scale_pos:.2f}")

    # CatBoost-style: keep cats as strings
    combined_cb = combined.copy()
    for c in CAT_COLS:
        combined_cb[c] = combined_cb[c].astype(str)
    # Fill NaN in numerics
    num_cols_cb = [c for c in FEATURE_COLS if c not in CAT_COLS]
    combined_cb[num_cols_cb] = combined_cb[num_cols_cb].fillna(0.0)
    X_cb_all = combined_cb.iloc[:n_train][FEATURE_COLS].reset_index(drop=True)
    X_cb_test = combined_cb.iloc[n_train:][FEATURE_COLS].reset_index(drop=True)
    X_cb_tr = X_cb_all[~val_mask].reset_index(drop=True)
    X_cb_va = X_cb_all[val_mask].reset_index(drop=True)

    # ─── Optuna XGB 10 trials (smaller — diversity matters more than tune) ───
    def xgb_obj(trial):
        p = dict(
            n_estimators=2000,
            max_depth=trial.suggest_int("max_depth", 5, 11),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 3.0),
            scale_pos_weight=scale_pos, eval_metric="auc",
            early_stopping_rounds=60, tree_method="hist",
            device="cuda", random_state=42, verbosity=0,
        )
        m = xgb.XGBClassifier(**p)
        m.fit(X_enc_tr, y_tr, eval_set=[(X_enc_va, y_va)], verbose=False)
        return roc_auc_score(y_va, m.predict_proba(X_enc_va)[:, 1])

    print("\n--- Optuna XGB 10 trials ---")
    t0 = time.time()
    sx = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
    sx.optimize(xgb_obj, n_trials=10, show_progress_bar=False)
    print(f"XGB best val_auc={sx.best_value:.6f}  ({time.time()-t0:.1f}s)")
    print(f"XGB best_params={sx.best_params}")
    xp = dict(sx.best_params, n_estimators=2000, scale_pos_weight=scale_pos,
              eval_metric="auc", early_stopping_rounds=60, tree_method="hist",
              device="cuda", random_state=42, verbosity=0)
    m_x = xgb.XGBClassifier(**xp)
    m_x.fit(X_enc_tr, y_tr, eval_set=[(X_enc_va, y_va)], verbose=False)
    oof_x = m_x.predict_proba(X_enc_va)[:, 1]
    print(f"XGB refit val_auc={roc_auc_score(y_va, oof_x):.6f}  best_iter={m_x.best_iteration}")
    xf = dict(xp, n_estimators=m_x.best_iteration + 1)
    xf.pop("early_stopping_rounds", None); xf.pop("eval_metric", None)
    mxf = xgb.XGBClassifier(**xf); mxf.fit(X_enc_all, y_all, verbose=False)
    test_x = mxf.predict_proba(X_enc_test)[:, 1]

    # ─── Optuna CB 8 trials ───
    def cb_obj(trial):
        p = dict(
            iterations=2000,
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
            depth=trial.suggest_int("depth", 5, 9),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            eval_metric="AUC", random_seed=42, verbose=0,
            early_stopping_rounds=80, scale_pos_weight=scale_pos, task_type="GPU",
        )
        m = CatBoostClassifier(**p)
        m.fit(Pool(X_cb_tr, y_tr, cat_features=CAT_COLS),
              eval_set=Pool(X_cb_va, y_va, cat_features=CAT_COLS), verbose=0)
        return roc_auc_score(y_va, m.predict_proba(X_cb_va)[:, 1])

    print("\n--- Optuna CB 8 trials ---")
    t0 = time.time()
    sc = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
    sc.optimize(cb_obj, n_trials=8, show_progress_bar=False)
    print(f"CB best val_auc={sc.best_value:.6f}  ({time.time()-t0:.1f}s)")
    print(f"CB best_params={sc.best_params}")
    cp = dict(sc.best_params, iterations=2000, eval_metric="AUC",
              random_seed=42, verbose=0, early_stopping_rounds=80,
              scale_pos_weight=scale_pos, task_type="GPU")
    m_c = CatBoostClassifier(**cp)
    m_c.fit(Pool(X_cb_tr, y_tr, cat_features=CAT_COLS),
            eval_set=Pool(X_cb_va, y_va, cat_features=CAT_COLS), verbose=0)
    oof_c = m_c.predict_proba(X_cb_va)[:, 1]
    print(f"CB refit val_auc={roc_auc_score(y_va, oof_c):.6f}  best_iter={m_c.get_best_iteration()}")
    cf = dict(cp, iterations=m_c.get_best_iteration() + 1)
    cf.pop("early_stopping_rounds", None); cf.pop("eval_metric", None)
    mcf = CatBoostClassifier(**cf)
    mcf.fit(Pool(X_cb_all, y_all, cat_features=CAT_COLS), verbose=0)
    test_c = mcf.predict_proba(X_cb_test)[:, 1]

    # ─── Equal-weight blend of XGB + CB ───
    auc_x = roc_auc_score(y_va, oof_x)
    auc_c = roc_auc_score(y_va, oof_c)
    eq = 0.5 * (oof_x + oof_c)
    eq_auc = roc_auc_score(y_va, eq)
    print(f"\nXGB val_auc={auc_x:.6f}  CB val_auc={auc_c:.6f}  EqBlend={eq_auc:.6f}")

    test_v10 = 0.5 * (test_x + test_c)

    # ─── Diversity vs prior subs (rank correlation) ───
    print("\nLoading prior subs for correlation check...")
    PRIOR_SUBS = {
        "v5.2": f"{DATA_DIR}/submission_v5_2_optuna.csv",
        "v5.4": f"{DATA_DIR}/submission_v5_4_blend3_full.csv",
        "v8":   f"{DATA_DIR}/submission_v8_seed_ensemble.csv",
    }
    test_ids_out = test[ID_COL].values
    v10_df = pd.DataFrame({"id": test_ids_out, "PitNextLap": test_v10})
    v10_sorted = v10_df.sort_values("id").reset_index(drop=True)
    v10_ranks = pd.Series(v10_sorted["PitNextLap"].values).rank().values
    print("\nRank-correlation (spearman) of v10 test preds vs prior subs:")
    for name, path in PRIOR_SUBS.items():
        df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
        assert np.array_equal(df["id"].values, v10_sorted["id"].values)
        sp = float(np.corrcoef(v10_ranks, pd.Series(df["PitNextLap"].values).rank().values)[0, 1])
        print(f"  v10 vs {name}: spearman={sp:.5f}  (lower = more diversity)")

    SUB_PATH = f"{DATA_DIR}/submission_v10_hazard.csv"
    pd.DataFrame({"id": test_ids_out, "PitNextLap": test_v10}).to_csv(SUB_PATH, index=False)
    print(f"\nsubmission -> {SUB_PATH}")

    # MLflow
    mlflow.set_tracking_uri("databricks")
    try:
        mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
    except Exception:
        mlflow.set_experiment("/Shared/databricks-ai-intern")
    with mlflow.start_run(run_name="f1_pitstops_v10_hazard") as run:
        mlflow.log_param("version", "v10_hazard")
        mlflow.log_param("n_features", len(FEATURE_COLS))
        for k, v in sx.best_params.items(): mlflow.log_param(f"xgb_{k}", v)
        for k, v in sc.best_params.items(): mlflow.log_param(f"cb_{k}", v)
        mlflow.log_metric("xgb_val_auc", float(auc_x))
        mlflow.log_metric("cb_val_auc", float(auc_c))
        mlflow.log_metric("eq_blend_val_auc", float(eq_auc))
        mlflow.log_artifact(SUB_PATH)
        run_id = run.info.run_id

    # ─── Kaggle submit ───
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    TOK = "KGAT_d7a0018cbb8131f82d3dd175d2e4cdb6"
    TP = os.path.expanduser("~/.kaggle/access_token")
    with open(TP, "w") as f: f.write(TOK)
    os.chmod(TP, 0o600)
    os.environ["KAGGLE_API_TOKEN"] = TOK
    COMP = "playground-series-s6e5"
    MSG = (f"v10 hazard: TyreLifeBin/RaceProgressBin/CompoundXStint feats + "
           f"smoothed pit-rate target encodings (train year<2025 only) + "
           f"dist-to-median-pit-tyrelife. XGB+CB eq-blend. val={eq_auc:.4f}")

    print(f"\nSubmitting v10...")
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", COMP, "-f", SUB_PATH, "-m", MSG],
                       capture_output=True, text=True)
    print("STDOUT:", r.stdout); print("rc:", r.returncode)
    print("\nPolling for v10 score...")
    final_v10 = ""
    for i in range(30):
        time.sleep(15)
        rs = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP],
                            capture_output=True, text=True)
        for line in rs.stdout.split("\n"):
            if "submission_v10_hazard.csv" in line:
                print(f"t={(i+1)*15}s: {line}")
                final_v10 = line
                if line.split() and line.split()[-1].startswith("0."): break
        if final_v10 and final_v10.split()[-1].startswith("0."): break

    print("\nDONE")
    print(json.dumps({
        "version": "v10_hazard",
        "xgb_val_auc": float(auc_x), "cb_val_auc": float(auc_c),
        "eq_blend_val_auc": float(eq_auc),
        "n_features": len(FEATURE_COLS),
        "v5_2_lb": 0.94924,
        "submission": SUB_PATH, "mlflow_run": run_id,
        "final_lb_line": final_v10[-200:] if final_v10 else "",
    }, indent=2))
except BaseException as _err:
    print(f"FATAL: {_err!r}")
    raise
finally:
    try:
        dbutils.notebook.exit(_BUF.getvalue()[-7000:])
    except Exception:
        pass
