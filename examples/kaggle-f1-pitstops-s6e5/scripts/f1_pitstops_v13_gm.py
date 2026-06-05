# Databricks notebook source
"""
v13 — GM-level recipe. Full kitchen sink informed by:
  - Our v5.2 (LB 0.94924, our best)
  - shamanthakreddymallu notebook (predict-f1-pit-stop)

Adoptions:
  1. External dataset (aadigupta1601 f1_strategy_dataset_v4) merged with sample_weight=0.6
  2. ~35 engineered features (rolling, diff, TyrePct_Used, InPitWindow, TE-in-CV)
  3. 5-fold StratifiedGroupKFold by RaceYear (Race+Year combined)
  4. Models: LGB, XGB, CB, MLP (4 model classes — TRUE diversity)
  5. Rank-norm OOFs + Optuna 100-trial weight search
  6. Adversarial validation: train_vs_test classifier → row test-likeness as weight multiplier
  7. Recency weight: 2025=1.3, 2024=1.1, 2023=1.0, ≤2022=0.7
  8. Smoothed target encoding (α=20) inside CV loop (no leakage)

Goal: break 0.949 ceiling. Anchor v5.2 LB 0.94924.
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
    import os, sys, time, subprocess, warnings, json, gc
    warnings.filterwarnings("ignore")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "xgboost", "lightgbm", "catboost", "optuna", "scikit-learn",
                           "pandas", "numpy", "mlflow", "kaggle", "torch"])

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier, Pool
    import optuna
    import mlflow
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from scipy.stats import rankdata
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    DATA_DIR = "/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops"
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    TARGET = "PitNextLap"
    ID_COL = "id"
    SEED = 42
    N_FOLDS = 5
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"comp_train={train.shape}  test={test.shape}  pos_rate={train[TARGET].mean():.4f}")

    # ─── 1. External dataset (best-effort) ───
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    TOK = "KGAT_d7a0018cbb8131f82d3dd175d2e4cdb6"
    TP = os.path.expanduser("~/.kaggle/access_token")
    with open(TP, "w") as f: f.write(TOK)
    os.chmod(TP, 0o600)
    os.environ["KAGGLE_API_TOKEN"] = TOK

    orig_loaded = False
    orig = None
    try:
        EXT_DIR = "/tmp/ext_data"
        os.makedirs(EXT_DIR, exist_ok=True)
        r = subprocess.run(["kaggle", "datasets", "download", "-d",
                            "aadigupta1601/f1-strategy-dataset-pit-stop-prediction",
                            "-p", EXT_DIR, "--unzip"],
                           capture_output=True, text=True, timeout=180)
        print(f"kaggle dataset download rc={r.returncode}")
        print(r.stdout[-500:]); print(r.stderr[-500:])
        for f in os.listdir(EXT_DIR):
            print(f"  found: {f}")
            if "v4" in f and f.endswith(".csv"):
                orig = pd.read_csv(os.path.join(EXT_DIR, f))
                print(f"orig raw shape: {orig.shape}, cols: {list(orig.columns)[:20]}")
                orig_loaded = True
                break
    except Exception as e:
        print(f"External data load FAILED: {e!r}. Continuing without it.")

    # ─── 2. Schema-align external + append ───
    train["is_orig"] = 0
    train["sample_weight"] = 1.0
    if orig_loaded:
        # Keep only columns shared with train
        shared = [c for c in train.columns if c in orig.columns and c != "sample_weight" and c != "is_orig"]
        print(f"shared cols: {shared}")
        if TARGET in shared and len(shared) >= 10:
            orig_aligned = orig[shared].copy()
            orig_aligned["is_orig"] = 1
            orig_aligned["sample_weight"] = 0.6
            # Drop NaN targets in orig
            orig_aligned = orig_aligned.dropna(subset=[TARGET]).reset_index(drop=True)
            orig_aligned[TARGET] = orig_aligned[TARGET].astype(int)
            print(f"orig aligned: {orig_aligned.shape}  pos_rate={orig_aligned[TARGET].mean():.4f}")
            all_train = pd.concat([train, orig_aligned], axis=0, ignore_index=True)
        else:
            print("orig schema mismatch — skipping merge")
            all_train = train.copy()
    else:
        all_train = train.copy()
    print(f"all_train shape after orig merge: {all_train.shape}")

    # ─── 3. Recency weighting ───
    def _recency_w(y):
        if y >= 2025: return 1.3
        if y == 2024: return 1.15
        if y == 2023: return 1.0
        if y == 2022: return 0.85
        return 0.7
    all_train["recency_w"] = all_train["Year"].apply(_recency_w)
    all_train["sample_weight"] = all_train["sample_weight"] * all_train["recency_w"]
    print(f"sample_weight stats: min={all_train['sample_weight'].min():.3f} "
          f"max={all_train['sample_weight'].max():.3f} mean={all_train['sample_weight'].mean():.3f}")

    # ─── 4. Combined frame (train + test) for feat eng ───
    test["is_orig"] = 0
    test["sample_weight"] = 1.0
    test["recency_w"] = 1.0
    test[TARGET] = np.nan
    combined = pd.concat([all_train, test], axis=0, ignore_index=True)
    n_all_train = len(all_train)
    print(f"combined shape: {combined.shape}")

    # ─── 5. Feature engineering ───
    COMPOUND_ORDER = {"WET": 0, "INTERMEDIATE": 1, "SOFT": 2, "MEDIUM": 3, "HARD": 4}
    COMPOUND_MAX_STINT = {"SOFT": 25, "MEDIUM": 35, "HARD": 50, "INTERMEDIATE": 30, "WET": 40}

    combined["Compound_ord"] = combined["Compound"].map(COMPOUND_ORDER).fillna(2)
    combined["MaxStint"] = combined["Compound"].map(COMPOUND_MAX_STINT).fillna(35)
    combined["RaceYear"] = combined["Race"].astype(str) + "_" + combined["Year"].astype(str)
    combined["TotalLaps_est"] = (combined["LapNumber"] / combined["RaceProgress"].replace(0, np.nan)).round()
    combined["LapsRemaining"] = combined["TotalLaps_est"] - combined["LapNumber"]
    combined["TyreLife_x_Compound"] = combined["TyreLife"] * combined["Compound_ord"]
    combined["TyrePct_Used"] = (combined["TyreLife"] / combined["MaxStint"]).clip(0, 2)
    combined["IsEarlyRace"] = (combined["RaceProgress"] < 0.25).astype(int)
    combined["IsMidRace"] = ((combined["RaceProgress"] >= 0.25) & (combined["RaceProgress"] < 0.70)).astype(int)
    combined["IsLateRace"] = (combined["RaceProgress"] >= 0.70).astype(int)
    combined["InPitWindow"] = (
        ((combined["RaceProgress"] >= 0.20) & (combined["RaceProgress"] <= 0.40)) |
        ((combined["RaceProgress"] >= 0.55) & (combined["RaceProgress"] <= 0.75))
    ).astype(int)
    combined["StopsAlready"] = combined["Stint"] - 1
    combined["Is1Stopper"] = (combined["Stint"] == 1).astype(int)
    combined["TyreLife_x_RaceProgress"] = combined["TyreLife"] * combined["RaceProgress"]
    combined["Degradation_per_Lap"] = (combined["Cumulative_Degradation"] / combined["TyreLife"].replace(0, 1)).clip(0, 10)
    combined["TyreLife_vs_LapsRemaining"] = combined["TyreLife"] / combined["LapsRemaining"].replace(0, 1)
    combined["CanMakeItToEnd"] = (combined["LapsRemaining"] <= (combined["MaxStint"] - combined["TyreLife"])).astype(int)
    combined["Stint_x_Compound"] = combined["Stint"] * combined["Compound_ord"]
    combined["LapsOverMaxStint"] = (combined["TyreLife"] - combined["MaxStint"]).clip(lower=0)
    combined["InPoints"] = (combined["Position"] <= 10).astype(int)
    combined["TopFive"] = (combined["Position"] <= 5).astype(int)
    combined["Position_x_RaceProgress"] = combined["Position"] * combined["RaceProgress"]

    # Rolling/diff features (need temporal sort within Driver×RaceYear)
    combined = combined.sort_values(["Driver", "RaceYear", "LapNumber"]).reset_index(drop=True)
    grp = combined.groupby(["Driver", "RaceYear"])
    combined["LapTime_roll3"] = grp["LapTime (s)"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    combined["LapTime_roll5"] = grp["LapTime (s)"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    combined["LapTime_roll10"] = grp["LapTime (s)"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    combined["Delta_roll3"] = grp["LapTime_Delta"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    combined["Delta_roll5"] = grp["LapTime_Delta"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    combined["Delta_roll10"] = grp["LapTime_Delta"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    combined["CumDeg_roll3"] = grp["Cumulative_Degradation"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    combined["LapTime_diff1"] = grp["LapTime (s)"].transform(lambda x: x.diff(1))
    combined["PaceDropoff_3v10"] = combined["LapTime_roll3"] - combined["LapTime_roll10"]
    combined["DegDropoff_3v10"] = combined["CumDeg_roll3"] - combined["Delta_roll10"]
    roll_cols = ["LapTime_roll3", "LapTime_roll5", "LapTime_roll10",
                 "Delta_roll3", "Delta_roll5", "Delta_roll10", "CumDeg_roll3",
                 "LapTime_diff1", "PaceDropoff_3v10", "DegDropoff_3v10"]
    combined[roll_cols] = combined[roll_cols].fillna(0)

    # Label encode for CB cat_features
    LE_COLS = ["Driver", "Race", "Compound", "RaceYear"]
    for c in LE_COLS:
        combined[f"LE_{c}"] = LabelEncoder().fit_transform(combined[c].astype(str))

    # Split back
    is_train_mask = combined[TARGET].notna()
    train_fe = combined[is_train_mask].copy().reset_index(drop=True)
    test_fe = combined[~is_train_mask].copy().reset_index(drop=True)
    print(f"train_fe={train_fe.shape}  test_fe={test_fe.shape}")

    # ─── 6. Smoothed target encoding (CV-internal) ───
    TE_COLS = ["Driver", "Race", "RaceYear", "Compound"]
    SMOOTH = 20
    global_mean = train_fe[TARGET].mean()
    for c in TE_COLS:
        train_fe[f"TE_{c}"] = global_mean
        test_fe[f"TE_{c}"] = global_mean

    temp_sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(
        temp_sgkf.split(train_fe, train_fe[TARGET].astype(int), train_fe["RaceYear"])
    ):
        tr_df = train_fe.iloc[tr_idx]
        for c in TE_COLS:
            agg = tr_df.groupby(c)[TARGET].agg(["mean", "count"])
            agg["te_smooth"] = (agg["mean"] * agg["count"] + global_mean * SMOOTH) / (agg["count"] + SMOOTH)
            train_fe.loc[train_fe.index[val_idx], f"TE_{c}"] = (
                train_fe.iloc[val_idx][c].map(agg["te_smooth"]).fillna(global_mean).values
            )
    # Refit TE on full train_fe for test
    for c in TE_COLS:
        agg = train_fe.groupby(c)[TARGET].agg(["mean", "count"])
        agg["te_smooth"] = (agg["mean"] * agg["count"] + global_mean * SMOOTH) / (agg["count"] + SMOOTH)
        test_fe[f"TE_{c}"] = test_fe[c].map(agg["te_smooth"]).fillna(global_mean).values
    print("Target encoding done.")

    # ─── 7. Adversarial validation reweight ───
    print("\n--- Adversarial validation ---")
    NUM_FEATS_BASE = [
        "Year", "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change", "PitStop",
        "Compound_ord", "TyrePct_Used", "TotalLaps_est", "LapsRemaining",
        "TyreLife_x_RaceProgress", "Degradation_per_Lap",
        "TyreLife_vs_LapsRemaining", "CanMakeItToEnd",
        "Stint_x_Compound", "LapsOverMaxStint",
        "IsEarlyRace", "IsMidRace", "IsLateRace", "InPitWindow",
        "StopsAlready", "Is1Stopper",
        "InPoints", "TopFive", "Position_x_RaceProgress",
        "LapTime_roll3", "LapTime_roll5", "LapTime_roll10",
        "Delta_roll3", "Delta_roll5", "Delta_roll10",
        "CumDeg_roll3", "LapTime_diff1", "PaceDropoff_3v10", "DegDropoff_3v10",
        "TyreLife_x_Compound", "MaxStint",
        "LE_Driver", "LE_Race", "LE_Compound", "LE_RaceYear",
        "TE_Driver", "TE_Race", "TE_RaceYear", "TE_Compound",
    ]
    X_av = pd.concat([train_fe[NUM_FEATS_BASE], test_fe[NUM_FEATS_BASE]], axis=0).values.astype(np.float32)
    y_av = np.concatenate([np.zeros(len(train_fe)), np.ones(len(test_fe))])
    av_clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               tree_method="hist", device="cuda", eval_metric="auc",
                               random_state=SEED, verbosity=0)
    av_clf.fit(X_av, y_av)
    av_auc = roc_auc_score(y_av, av_clf.predict_proba(X_av)[:, 1])
    print(f"adversarial val AUC: {av_auc:.4f} (1.0=perfect shift, 0.5=no shift)")
    if av_auc > 0.95:
        # AV memorized — multiplier would be uninformative (~0 for all train rows). Skip.
        print(f"AV AUC > 0.95 — multiplier is overfit, skipping reweight.")
        train_fe["adv_w"] = 1.0
    else:
        p_test = av_clf.predict_proba(train_fe[NUM_FEATS_BASE].values.astype(np.float32))[:, 1]
        train_fe["adv_w"] = 1.0 + 0.4 * p_test
        train_fe["sample_weight"] = train_fe["sample_weight"] * train_fe["adv_w"]
    print(f"final sample_weight: min={train_fe['sample_weight'].min():.3f} "
          f"max={train_fe['sample_weight'].max():.3f} mean={train_fe['sample_weight'].mean():.3f}")

    # ─── 8. Train arrays ───
    FEATURES = NUM_FEATS_BASE
    X = train_fe[FEATURES].values.astype(np.float32)
    y = train_fe[TARGET].values.astype(int)
    w = train_fe["sample_weight"].values.astype(np.float32)
    groups = train_fe["RaceYear"].values
    X_test = test_fe[FEATURES].values.astype(np.float32)
    test_ids = test_fe[ID_COL].values.astype(int)
    print(f"X={X.shape}  y={y.shape}  X_test={X_test.shape}")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(X, y, groups))

    # ─── 9. Model: LightGBM ───
    print("\n=== LightGBM ===")
    LGB_PARAMS = dict(
        objective="binary", metric="auc", learning_rate=0.02,
        num_leaves=127, max_depth=-1, min_child_samples=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.5, random_state=SEED, verbosity=-1,
    )
    oof_lgb = np.zeros(len(X))
    test_lgb = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        t0 = time.time()
        dtrain = lgb.Dataset(X[tr_idx], y[tr_idx], weight=w[tr_idx])
        dvalid = lgb.Dataset(X[val_idx], y[val_idx])
        m = lgb.train(LGB_PARAMS, dtrain, num_boost_round=5000,
                      valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(150, verbose=False),
                                 lgb.log_evaluation(0)])
        oof_lgb[val_idx] = m.predict(X[val_idx])
        test_lgb += m.predict(X_test) / N_FOLDS
        fa = roc_auc_score(y[val_idx], oof_lgb[val_idx])
        print(f"  fold{fold} AUC={fa:.5f} iter={m.best_iteration} ({time.time()-t0:.0f}s)")
    auc_lgb = roc_auc_score(y, oof_lgb)
    print(f"LGB OOF AUC: {auc_lgb:.5f}")

    # ─── 10. Model: XGBoost ───
    print("\n=== XGBoost ===")
    XGB_PARAMS = dict(
        max_depth=8, learning_rate=0.03, n_estimators=3000,
        subsample=0.85, colsample_bytree=0.7, min_child_weight=5,
        reg_lambda=2.0, reg_alpha=0.1, gamma=0.1,
        tree_method="hist", device="cuda", eval_metric="auc",
        early_stopping_rounds=100, random_state=SEED, verbosity=0,
    )
    oof_xgb = np.zeros(len(X))
    test_xgb = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        t0 = time.time()
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx],
              eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        oof_xgb[val_idx] = m.predict_proba(X[val_idx])[:, 1]
        test_xgb += m.predict_proba(X_test)[:, 1] / N_FOLDS
        fa = roc_auc_score(y[val_idx], oof_xgb[val_idx])
        print(f"  fold{fold} AUC={fa:.5f} iter={m.best_iteration} ({time.time()-t0:.0f}s)")
    auc_xgb = roc_auc_score(y, oof_xgb)
    print(f"XGB OOF AUC: {auc_xgb:.5f}")

    # ─── 11. Model: CatBoost ───
    print("\n=== CatBoost ===")
    CB_PARAMS = dict(
        iterations=3000, learning_rate=0.03, depth=8,
        l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=0.5,
        border_count=128, eval_metric="AUC",
        random_seed=SEED, verbose=0, early_stopping_rounds=100,
        task_type="GPU",
    )
    CB_CATS = ["LE_Driver", "LE_Race", "LE_Compound", "LE_RaceYear"]
    # CB needs DataFrame with int dtype on cat cols
    Xc_df = train_fe[FEATURES].copy()
    Xc_test_df = test_fe[FEATURES].copy()
    for c in CB_CATS:
        Xc_df[c] = Xc_df[c].astype(int)
        Xc_test_df[c] = Xc_test_df[c].astype(int)
    oof_cb = np.zeros(len(X))
    test_cb = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        t0 = time.time()
        tp = Pool(Xc_df.iloc[tr_idx], y[tr_idx], weight=w[tr_idx], cat_features=CB_CATS)
        vp = Pool(Xc_df.iloc[val_idx], y[val_idx], cat_features=CB_CATS)
        m = CatBoostClassifier(**CB_PARAMS)
        m.fit(tp, eval_set=vp, verbose=0)
        oof_cb[val_idx] = m.predict_proba(Xc_df.iloc[val_idx])[:, 1]
        test_cb += m.predict_proba(Xc_test_df)[:, 1] / N_FOLDS
        fa = roc_auc_score(y[val_idx], oof_cb[val_idx])
        print(f"  fold{fold} AUC={fa:.5f} iter={m.get_best_iteration()} ({time.time()-t0:.0f}s)")
    auc_cb = roc_auc_score(y, oof_cb)
    print(f"CB OOF AUC: {auc_cb:.5f}")

    # ─── 12. Model: MLP ───
    print("\n=== MLP ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MLP device: {device}")

    class MLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.Mish(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.Mish(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.LayerNorm(64), nn.Mish(), nn.Dropout(0.1),
                nn.Linear(64, 1), nn.Sigmoid(),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    oof_mlp = np.zeros(len(X))
    test_mlp = np.zeros(len(X_test))
    EPOCHS = 15
    BS = 4096

    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        t0 = time.time()
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_idx])
        Xva = scaler.transform(X[val_idx])
        Xte = scaler.transform(X_test)
        ytr = y[tr_idx].astype(np.float32)
        yva = y[val_idx].astype(np.float32)
        wtr = w[tr_idx].astype(np.float32)

        ds = TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.float32),
            torch.tensor(wtr, dtype=torch.float32),
        )
        dl = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=False)

        model = MLP(Xtr.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.BCELoss(reduction="none")

        Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
        best_auc = 0; best_oof = None; best_test = None
        for ep in range(EPOCHS):
            model.train()
            for xb, yb, wb in dl:
                xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
                opt.zero_grad()
                p = model(xb)
                loss = (loss_fn(p, yb) * wb).mean()
                loss.backward()
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pv = model(Xva_t).cpu().numpy()
            va = roc_auc_score(yva, pv)
            if va > best_auc:
                best_auc = va
                best_oof = pv
                with torch.no_grad():
                    best_test = model(Xte_t).cpu().numpy()
        oof_mlp[val_idx] = best_oof
        test_mlp += best_test / N_FOLDS
        print(f"  fold{fold} AUC={best_auc:.5f} ({time.time()-t0:.0f}s)")
        del model, opt, sched, Xva_t, Xte_t, ds, dl
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    auc_mlp = roc_auc_score(y, oof_mlp)
    print(f"MLP OOF AUC: {auc_mlp:.5f}")

    # ─── 13. Rank-norm + Optuna blend ───
    print("\n=== Blend ===")
    def rank_norm(x):
        return (rankdata(x) - 1) / (len(x) - 1)
    r_lgb = rank_norm(oof_lgb); r_xgb = rank_norm(oof_xgb)
    r_cb = rank_norm(oof_cb); r_mlp = rank_norm(oof_mlp)
    rt_lgb = rank_norm(test_lgb); rt_xgb = rank_norm(test_xgb)
    rt_cb = rank_norm(test_cb); rt_mlp = rank_norm(test_mlp)

    print(f"per-model OOF AUC: lgb={auc_lgb:.5f} xgb={auc_xgb:.5f} cb={auc_cb:.5f} mlp={auc_mlp:.5f}")
    # Spearman diag
    import numpy as _np
    rk = {"lgb": r_lgb, "xgb": r_xgb, "cb": r_cb, "mlp": r_mlp}
    print("Spearman OOF:")
    names = list(rk.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            sp = float(_np.corrcoef(rk[names[i]], rk[names[j]])[0, 1])
            print(f"  {names[i]} vs {names[j]}: {sp:.5f}")

    def blend_oof(ws):
        w_lgb, w_xgb, w_cb, w_mlp = ws
        s = w_lgb + w_xgb + w_cb + w_mlp
        if s == 0: return 0.5
        return roc_auc_score(y, (w_lgb*r_lgb + w_xgb*r_xgb + w_cb*r_cb + w_mlp*r_mlp) / s)

    def objective(trial):
        w_lgb = trial.suggest_float("w_lgb", 0.0, 1.0)
        w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0)
        w_cb = trial.suggest_float("w_cb", 0.0, 1.0)
        w_mlp = trial.suggest_float("w_mlp", 0.0, 1.0)
        return blend_oof((w_lgb, w_xgb, w_cb, w_mlp))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=200, show_progress_bar=False)
    print(f"best blend OOF AUC: {study.best_value:.5f}")
    bp = study.best_params
    s = bp["w_lgb"] + bp["w_xgb"] + bp["w_cb"] + bp["w_mlp"]
    W = {k: v/s for k, v in bp.items()}
    print(f"weights (normed): {W}")

    final_oof = (W["w_lgb"]*r_lgb + W["w_xgb"]*r_xgb + W["w_cb"]*r_cb + W["w_mlp"]*r_mlp)
    final_test = (W["w_lgb"]*rt_lgb + W["w_xgb"]*rt_xgb + W["w_cb"]*rt_cb + W["w_mlp"]*rt_mlp)
    blend_auc = roc_auc_score(y, final_oof)
    print(f"FINAL blend OOF AUC: {blend_auc:.5f}")

    # ─── 14. Save + submit ───
    SUB_PATH = f"{DATA_DIR}/submission_v13_gm.csv"
    OOF_PATH = f"{DATA_DIR}/oof_v13_gm.csv"
    pd.DataFrame({"id": test_ids, "PitNextLap": final_test}).to_csv(SUB_PATH, index=False)
    pd.DataFrame({"lgb": oof_lgb, "xgb": oof_xgb, "cb": oof_cb, "mlp": oof_mlp}).to_csv(OOF_PATH, index=False)
    print(f"submission -> {SUB_PATH}")

    mlflow.set_tracking_uri("databricks")
    try: mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
    except Exception: mlflow.set_experiment("/Shared/databricks-ai-intern")
    with mlflow.start_run(run_name="f1_pitstops_v13_gm") as run:
        mlflow.log_param("version", "v13_gm")
        mlflow.log_param("orig_loaded", orig_loaded)
        mlflow.log_param("av_auc", float(av_auc))
        mlflow.log_metric("lgb_oof", float(auc_lgb))
        mlflow.log_metric("xgb_oof", float(auc_xgb))
        mlflow.log_metric("cb_oof", float(auc_cb))
        mlflow.log_metric("mlp_oof", float(auc_mlp))
        mlflow.log_metric("blend_oof", float(blend_auc))
        for k, v in W.items(): mlflow.log_param(k, float(v))
        mlflow.log_artifact(SUB_PATH)
        run_id = run.info.run_id

    MSG = (f"v13 GM: orig_data={orig_loaded}, LGB+XGB+CB+MLP rank-blend, "
           f"adv-val reweight, recency, 5-fold StratGroupKFold RaceYear. "
           f"OOFs lgb={auc_lgb:.4f} xgb={auc_xgb:.4f} cb={auc_cb:.4f} mlp={auc_mlp:.4f} "
           f"blend={blend_auc:.4f}.")
    print(f"\nSubmitting v13...")
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", "playground-series-s6e5",
                        "-f", SUB_PATH, "-m", MSG], capture_output=True, text=True)
    print("STDOUT:", r.stdout); print("rc:", r.returncode)

    print("\nPolling for v13 score...")
    final = ""
    for i in range(40):
        time.sleep(15)
        rs = subprocess.run(["kaggle", "competitions", "submissions", "-c", "playground-series-s6e5"],
                            capture_output=True, text=True)
        for line in rs.stdout.split("\n"):
            if "submission_v13_gm.csv" in line:
                print(f"t={(i+1)*15}s: {line}")
                final = line
                if line.split() and line.split()[-1].startswith("0."): break
        if final and final.split()[-1].startswith("0."): break

    print("\n" + "=" * 60)
    print("V13 REPORT")
    print("=" * 60)
    print(json.dumps({
        "version": "v13_gm",
        "orig_loaded": orig_loaded,
        "av_auc": float(av_auc),
        "lgb_oof": float(auc_lgb),
        "xgb_oof": float(auc_xgb),
        "cb_oof": float(auc_cb),
        "mlp_oof": float(auc_mlp),
        "blend_oof": float(blend_auc),
        "weights": W,
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
