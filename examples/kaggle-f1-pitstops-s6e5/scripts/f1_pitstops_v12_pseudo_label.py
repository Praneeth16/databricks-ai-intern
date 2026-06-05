# Databricks notebook source
"""
v12 — Pseudo-label on v5.2 confident test predictions.

Recipe:
  1. Load v5.2 submission (anchor LB 0.94924, predictions on test).
  2. Pseudo-label test rows with pred >= HI or <= LO as label 1/0.
  3. Append pseudo-rows to train; mark Year=2025 val UNCHANGED (no contamination).
  4. Re-Optuna XGB+CB (20 trials each) on augmented train.
  5. Blend (hill-climb on val) → submit as v12.

Anchor v5.2 = 0.94924. Goal: +0.001 to +0.003 from distribution-shift correction.
Standard pseudo-labeling trick used on shift-heavy Kaggle Playgrounds.
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
    v52_sub = pd.read_csv(f"{DATA_DIR}/submission_v5_2_optuna.csv").sort_values("id").reset_index(drop=True)
    TARGET = "PitNextLap"
    ID_COL = "id"
    print(f"train={train.shape}  test={test.shape}  pos_rate={train[TARGET].mean():.4f}")
    print(f"v5.2 sub loaded: {v52_sub.shape}  mean={v52_sub['PitNextLap'].mean():.4f}")

    # ─── Pseudo-label test rows ───
    HI = 0.97
    LO = 0.03
    test_aligned = test.sort_values(ID_COL).reset_index(drop=True)
    assert np.array_equal(test_aligned[ID_COL].values, v52_sub[ID_COL].values)
    p = v52_sub["PitNextLap"].values
    is_pos = p >= HI
    is_neg = p <= LO
    n_pos = int(is_pos.sum()); n_neg = int(is_neg.sum())
    print(f"\nThresholds HI={HI} LO={LO}")
    print(f"pseudo POS rows: {n_pos} ({n_pos/len(p)*100:.2f}%)")
    print(f"pseudo NEG rows: {n_neg} ({n_neg/len(p)*100:.2f}%)")
    print(f"pseudo total:    {n_pos+n_neg} ({(n_pos+n_neg)/len(p)*100:.2f}%)")

    if n_pos < 200 or n_neg < 2000:
        # Fall back to looser thresholds
        HI, LO = 0.93, 0.05
        is_pos = p >= HI; is_neg = p <= LO
        n_pos = int(is_pos.sum()); n_neg = int(is_neg.sum())
        print(f"Falling back: HI={HI} LO={LO} -> POS={n_pos} NEG={n_neg}")

    pseudo_mask = is_pos | is_neg
    pseudo = test_aligned[pseudo_mask].copy().reset_index(drop=True)
    pseudo[TARGET] = np.where(p[pseudo_mask] >= HI, 1, 0).astype(int)
    print(f"Pseudo rows added: {len(pseudo)}, pseudo pos_rate={pseudo[TARGET].mean():.4f}")

    # Build augmented train
    aug = pd.concat([train, pseudo], axis=0, ignore_index=True)
    print(f"Augmented train shape: {aug.shape}  (orig {train.shape[0]} + pseudo {len(pseudo)})")

    FEATS = ["Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint",
             "TyreLife", "Position", "LapTime (s)", "LapTime_Delta",
             "Cumulative_Degradation", "RaceProgress", "Position_Change"]
    CAT_COLS = ["Driver", "Compound", "Race"]

    # Combine: encode using train+test+pseudo as one universe so encoders see all values
    combined = pd.concat([aug[FEATS], test_aligned[FEATS]], axis=0, ignore_index=True)
    n_aug = len(aug)

    # Val mask: only ORIGINAL train Year=2025 rows (no pseudo in val)
    is_orig = np.zeros(n_aug, dtype=bool)
    is_orig[: len(train)] = True
    years = aug["Year"].values
    val_mask = (years == 2025) & is_orig          # untouched val: original 2025 train rows
    train_mask = ~val_mask                         # everything else (incl pseudo) goes to train
    y_all = aug[TARGET].values.astype(int)
    y_tr = y_all[train_mask]; y_va = y_all[val_mask]
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"tr_size={train_mask.sum()}  va_size={val_mask.sum()}  scale_pos={scale_pos:.2f}")
    print(f"val pos_rate={y_va.mean():.4f}  tr pos_rate={y_tr.mean():.4f}")

    # XGB pipeline: encoded
    combined_xgb = combined.copy()
    for c in CAT_COLS:
        combined_xgb[c] = LabelEncoder().fit_transform(combined_xgb[c].astype(str))
    X_xgb_aug = combined_xgb.iloc[:n_aug].values
    X_xgb_test = combined_xgb.iloc[n_aug:].values
    X_xgb_tr, X_xgb_va = X_xgb_aug[train_mask], X_xgb_aug[val_mask]

    # CB pipeline: string cats
    combined_cb = combined.copy()
    for c in CAT_COLS:
        combined_cb[c] = combined_cb[c].astype(str)
    X_cb_aug = combined_cb.iloc[:n_aug].reset_index(drop=True)
    X_cb_test = combined_cb.iloc[n_aug:].reset_index(drop=True)
    X_cb_tr = X_cb_aug[train_mask].reset_index(drop=True)
    X_cb_va = X_cb_aug[val_mask].reset_index(drop=True)

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

    print("\n--- Optuna XGB 20 trials (pseudo-augmented) ---")
    t0 = time.time()
    study_x = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=42))
    study_x.optimize(xgb_objective, n_trials=20, show_progress_bar=False)
    print(f"XGB best val AUC: {study_x.best_value:.6f}  ({time.time()-t0:.1f}s)")
    print(f"XGB best params: {study_x.best_params}")

    xp_best = dict(study_x.best_params, n_estimators=1500,
                   scale_pos_weight=scale_pos, eval_metric="auc",
                   early_stopping_rounds=50, tree_method="hist",
                   device="cuda", random_state=42, verbosity=0)
    m_xgb = xgb.XGBClassifier(**xp_best)
    m_xgb.fit(X_xgb_tr, y_tr, eval_set=[(X_xgb_va, y_va)], verbose=False)
    oof_xgb = m_xgb.predict_proba(X_xgb_va)[:, 1]
    auc_xgb = roc_auc_score(y_va, oof_xgb)
    print(f"XGB refit val AUC: {auc_xgb:.6f}  best_iter={m_xgb.best_iteration}")

    # Retrain XGB on full augmented data
    xf = dict(xp_best, n_estimators=m_xgb.best_iteration + 1)
    xf.pop("early_stopping_rounds", None); xf.pop("eval_metric", None)
    m_xgb_full = xgb.XGBClassifier(**xf)
    m_xgb_full.fit(X_xgb_aug, y_all, verbose=False)
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

    print("\n--- Optuna CatBoost 20 trials (pseudo-augmented) ---")
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
    m_cb_full.fit(Pool(X_cb_aug, y_all, cat_features=CAT_COLS), verbose=0)
    test_cb = m_cb_full.predict_proba(X_cb_test)[:, 1]

    # ─── Blend ───
    OOF = np.column_stack([oof_xgb, oof_cb])
    TEST = np.column_stack([test_xgb, test_cb])
    per_auc = [roc_auc_score(y_va, OOF[:, i]) for i in range(2)]
    print(f"\nPer-model val AUC: xgb={per_auc[0]:.6f}  cb={per_auc[1]:.6f}")

    eq = OOF.mean(axis=1)
    eq_auc = roc_auc_score(y_va, eq)
    print(f"Equal-weight blend: {eq_auc:.6f}")

    best_w, best_a = 0.5, 0.0
    for w in np.linspace(0.0, 1.0, 101):
        blend = w * OOF[:, 0] + (1 - w) * OOF[:, 1]
        a = roc_auc_score(y_va, blend)
        if a > best_a:
            best_a = a; best_w = w
    print(f"Best XGB weight = {best_w:.2f}  blend val AUC = {best_a:.6f}")

    weights = np.array([best_w, 1 - best_w])
    test_blend = TEST @ weights
    v52_lb = 0.94924
    pred_lb = best_a + 0.046
    print(f"Predicted LB (val+0.046): {pred_lb:.4f}  vs v5.2 LB {v52_lb}")

    SUB_PATH = f"{DATA_DIR}/submission_v12_pseudo_label.csv"
    OOF_PATH = f"{DATA_DIR}/oof_v12_pseudo_label.csv"
    test_ids = test_aligned[ID_COL].values
    pd.DataFrame({"id": test_ids, "PitNextLap": test_blend}).to_csv(SUB_PATH, index=False)
    pd.DataFrame(OOF, columns=["xgb_pl", "cb_pl"]).to_csv(OOF_PATH, index=False)
    print(f"submission -> {SUB_PATH}")

    # ─── MLflow ───
    mlflow.set_tracking_uri("databricks")
    try:
        mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
    except Exception:
        mlflow.set_experiment("/Shared/databricks-ai-intern")
    with mlflow.start_run(run_name="f1_pitstops_v12_pseudo_label") as run:
        mlflow.log_param("version", "v12_pseudo_label")
        mlflow.log_param("pseudo_HI", HI)
        mlflow.log_param("pseudo_LO", LO)
        mlflow.log_param("pseudo_n_pos", n_pos)
        mlflow.log_param("pseudo_n_neg", n_neg)
        mlflow.log_metric("xgb_val_auc", float(auc_xgb))
        mlflow.log_metric("cb_val_auc", float(auc_cb))
        mlflow.log_metric("blend_val_auc", float(best_a))
        mlflow.log_metric("predicted_lb", float(pred_lb))
        mlflow.log_metric("xgb_weight", float(best_w))
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
    MSG = (f"v12 pseudo-label: v5.2 confident test rows (HI={HI}, LO={LO}) "
           f"appended as train. n_pos={n_pos} n_neg={n_neg}. "
           f"Re-Optuna XGB+CB, val={best_a:.5f}, anchor v5.2 LB 0.94924.")

    print(f"\nSubmitting v12...")
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", COMP,
                        "-f", SUB_PATH, "-m", MSG], capture_output=True, text=True)
    print("STDOUT:", r.stdout); print("rc:", r.returncode)

    print("\nPolling for v12 score...")
    final = ""
    for i in range(30):
        time.sleep(15)
        rs = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP],
                            capture_output=True, text=True)
        for line in rs.stdout.split("\n"):
            if "submission_v12_pseudo_label.csv" in line:
                print(f"t={(i+1)*15}s: {line}")
                final = line
                if line.split() and line.split()[-1].startswith("0."): break
        if final and final.split()[-1].startswith("0."): break

    print("\n" + "=" * 60)
    print("V12 REPORT")
    print("=" * 60)
    print(json.dumps({
        "version": "v12_pseudo_label",
        "pseudo_HI": HI, "pseudo_LO": LO,
        "pseudo_n_pos": n_pos, "pseudo_n_neg": n_neg,
        "xgb_val_auc": float(auc_xgb),
        "cb_val_auc": float(auc_cb),
        "blend_val_auc": float(best_a),
        "predicted_lb": float(pred_lb),
        "v52_lb": v52_lb,
        "kaggle_final_line": final,
        "submission": SUB_PATH,
        "mlflow_run": run_id,
    }, indent=2))
    print("\nDONE")
except BaseException as _err:
    print(f"FATAL: {_err!r}")
    raise
finally:
    try:
        dbutils.notebook.exit(_BUF.getvalue()[-7000:])
    except Exception:
        pass
