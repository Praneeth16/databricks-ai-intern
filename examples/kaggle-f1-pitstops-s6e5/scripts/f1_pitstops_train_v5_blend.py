# Databricks notebook source
"""
v5 — Model diversity blend.

Stack:
  - CatBoost: native cat_features (Driver, Compound, Race), no LabelEncoder
  - XGB x5 seeds (42, 1337, 2024, 7, 13) at v1 hparams
  - Hill-climb weight search over 6 OOF columns
  - Final = weighted blend on test preds

Locked: v1's 14 features only. NO new FE (Phase 2 confirmed harmful).
Time-split val (Year=2025) for OOF.
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
                           "xgboost", "catboost", "scikit-learn", "pandas", "numpy", "mlflow"])

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    from catboost import CatBoostClassifier, Pool
    import mlflow

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

    # ─── XGB pipeline: LabelEncode cats ───
    combined = pd.concat([train[FEATS], test[FEATS]], axis=0, ignore_index=True)
    combined_xgb = combined.copy()
    for c in CAT_COLS:
        combined_xgb[c] = LabelEncoder().fit_transform(combined_xgb[c].astype(str))

    n_train = len(train)
    X_xgb_all = combined_xgb.iloc[:n_train].values
    X_xgb_test = combined_xgb.iloc[n_train:].values
    y_all = train[TARGET].values.astype(int)
    years = train["Year"].values

    val_mask = years == 2025
    X_tr, X_va = X_xgb_all[~val_mask], X_xgb_all[val_mask]
    y_tr, y_va = y_all[~val_mask], y_all[val_mask]
    print(f"tr={X_tr.shape}  va={X_va.shape}")
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"scale_pos_weight={scale_pos:.2f}")

    # ─── CatBoost pipeline: keep strings, mark cat_features ───
    cat_idx_xgb = [FEATS.index(c) for c in CAT_COLS]
    combined_cb = combined.copy()
    for c in CAT_COLS:
        combined_cb[c] = combined_cb[c].astype(str)
    X_cb_all = combined_cb.iloc[:n_train]
    X_cb_test = combined_cb.iloc[n_train:]
    X_cb_tr, X_cb_va = X_cb_all[~val_mask].reset_index(drop=True), X_cb_all[val_mask].reset_index(drop=True)

    # ─── XGB 5 seeds ───
    XGB_HP = dict(n_estimators=500, max_depth=8, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  scale_pos_weight=scale_pos, eval_metric="auc",
                  early_stopping_rounds=50, tree_method="hist",
                  device="cuda", verbosity=0)
    SEEDS = [42, 1337, 2024, 7, 13]
    oof_xgb = []   # list of (n_va,) arrays
    test_xgb = []  # list of (n_test,) arrays
    for s in SEEDS:
        t0 = time.time()
        m = xgb.XGBClassifier(**XGB_HP, random_state=s)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        v = m.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, v)
        print(f"XGB seed={s}  val_auc={auc:.6f}  best_iter={m.best_iteration}  t={time.time()-t0:.1f}s")
        oof_xgb.append(v)
        # Retrain on full data with best_iter
        n_best = m.best_iteration + 1
        fp = dict(XGB_HP, n_estimators=n_best, random_state=s)
        fp.pop("early_stopping_rounds", None)
        fp.pop("eval_metric", None)
        mf = xgb.XGBClassifier(**fp)
        mf.fit(X_xgb_all, y_all, verbose=False)
        test_xgb.append(mf.predict_proba(X_xgb_test)[:, 1])

    # ─── CatBoost 1 model (native cats) ───
    t0 = time.time()
    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=8,
        l2_leaf_reg=5, eval_metric="AUC",
        random_seed=42, verbose=0,
        early_stopping_rounds=80,
        scale_pos_weight=scale_pos,
        task_type="GPU",
    )
    cb_pool_tr = Pool(X_cb_tr, y_tr, cat_features=CAT_COLS)
    cb_pool_va = Pool(X_cb_va, y_va, cat_features=CAT_COLS)
    cb.fit(cb_pool_tr, eval_set=cb_pool_va, verbose=0)
    oof_cb = cb.predict_proba(X_cb_va)[:, 1]
    auc_cb = roc_auc_score(y_va, oof_cb)
    print(f"CatBoost val_auc={auc_cb:.6f}  best_iter={cb.get_best_iteration()}  t={time.time()-t0:.1f}s")
    # Retrain CB on full data
    cb_full = CatBoostClassifier(
        iterations=cb.get_best_iteration() + 1, learning_rate=0.05, depth=8,
        l2_leaf_reg=5, random_seed=42, verbose=0,
        scale_pos_weight=scale_pos, task_type="GPU",
    )
    cb_full.fit(Pool(X_cb_all, y_all, cat_features=CAT_COLS))
    test_cb = cb_full.predict_proba(X_cb_test)[:, 1]

    # ─── OOF matrix ───
    OOF = np.column_stack(oof_xgb + [oof_cb])     # (n_va, 6)
    TEST = np.column_stack(test_xgb + [test_cb])  # (n_test, 6)
    model_names = [f"xgb_s{s}" for s in SEEDS] + ["catboost"]
    per_model_auc = [roc_auc_score(y_va, OOF[:, i]) for i in range(OOF.shape[1])]
    print("\nPer-model val AUC:")
    for nm, a in zip(model_names, per_model_auc):
        print(f"  {nm:12s}  {a:.6f}")

    # Equal-weight baseline blend
    eq_blend = OOF.mean(axis=1)
    print(f"Equal-weight blend AUC: {roc_auc_score(y_va, eq_blend):.6f}")

    # ─── Hill-climb weight search ───
    def hill_climb(OOF, y, steps=100, lr=0.05):
        n_models = OOF.shape[1]
        w = np.ones(n_models) / n_models
        best_auc = roc_auc_score(y, OOF @ w)
        for it in range(steps):
            improved = False
            for i in range(n_models):
                for delta in (-lr, lr):
                    wn = w.copy()
                    wn[i] = max(0.0, min(1.0, wn[i] + delta))
                    s = wn.sum()
                    if s < 1e-9: continue
                    wn = wn / s
                    a = roc_auc_score(y, OOF @ wn)
                    if a > best_auc:
                        best_auc = a
                        w = wn
                        improved = True
            if not improved:
                break
        return w, best_auc

    weights, blend_auc = hill_climb(OOF, y_va, steps=200, lr=0.05)
    print(f"\nHill-climb blend AUC: {blend_auc:.6f}")
    print("Weights:")
    for nm, wv in zip(model_names, weights):
        print(f"  {nm:12s}  {wv:.4f}")

    # Best single
    best_single_idx = int(np.argmax(per_model_auc))
    print(f"Best single: {model_names[best_single_idx]}  ({per_model_auc[best_single_idx]:.6f})")
    blend_lift = blend_auc - per_model_auc[best_single_idx]
    print(f"Blend lift over best single: {blend_lift:+.6f}")

    # ─── Generate submission ───
    test_blend = TEST @ weights
    test_ids = test[ID_COL].values
    sub = pd.DataFrame({"id": test_ids, "PitNextLap": test_blend})
    SUB_PATH = f"{DATA_DIR}/submission_v5_blend.csv"
    OOF_PATH = f"{DATA_DIR}/oof_v5_blend.csv"
    sub.to_csv(SUB_PATH, index=False)
    pd.DataFrame(OOF, columns=model_names).to_csv(OOF_PATH, index=False)
    print(f"\nsubmission -> {SUB_PATH}  shape={sub.shape}")
    print(f"pred stats: mean={test_blend.mean():.4f}  std={test_blend.std():.4f}")

    # ─── MLflow ───
    mlflow.set_tracking_uri("databricks")
    try:
        mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
    except Exception:
        mlflow.set_experiment("/Shared/databricks-ai-intern")

    with mlflow.start_run(run_name="f1_pitstops_v5_blend") as run:
        mlflow.log_param("version", "v5_blend")
        mlflow.log_param("models", ",".join(model_names))
        mlflow.log_metric("blend_val_auc", float(blend_auc))
        mlflow.log_metric("eq_blend_val_auc", float(roc_auc_score(y_va, eq_blend)))
        mlflow.log_metric("best_single_val_auc", float(per_model_auc[best_single_idx]))
        mlflow.log_metric("blend_lift", float(blend_lift))
        for nm, a, w in zip(model_names, per_model_auc, weights):
            mlflow.log_metric(f"{nm}_val_auc", float(a))
            mlflow.log_metric(f"{nm}_weight", float(w))
        mlflow.log_artifact(SUB_PATH)
        mlflow.log_artifact(OOF_PATH)
        run_id = run.info.run_id

    print("\n" + "=" * 60)
    print("V5 BLEND REPORT")
    print("=" * 60)
    print(json.dumps({
        "version": "v5_blend",
        "blend_val_auc": float(blend_auc),
        "best_single_val_auc": float(per_model_auc[best_single_idx]),
        "blend_lift": float(blend_lift),
        "weights": dict(zip(model_names, [float(w) for w in weights])),
        "per_model_val_auc": dict(zip(model_names, [float(a) for a in per_model_auc])),
        "submission": SUB_PATH,
        "mlflow_run": run_id,
        "submit_decision": "SUBMIT" if blend_lift > 0.001 else "HOLD",
    }, indent=2))
except BaseException as _err:
    print(f"FATAL: {_err!r}")
    raise
finally:
    try:
        dbutils.notebook.exit(_BUF.getvalue()[-6000:])
    except Exception:
        pass
