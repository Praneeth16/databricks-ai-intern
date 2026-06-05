# Databricks notebook source
"""
v13.2 — Year=2025 val + rich features + 4-model rank blend + pseudo from v5.2.

Lessons applied:
  - DROP external dataset (AV=1.0, distribution shift too severe)
  - DROP StratGroupKFold (wrong geometry; OOF↑LB↓ proved it)
  - DROP TE-in-CV (likely leaking)
  - USE Year=2025 holdout val (proven LB proxy in v5.2)
  - KEEP 35 engineered features (rolling, diff, TyrePct, InPitWindow)
  - KEEP LGB+XGB+CB+MLP 4-model rank blend
  - ADD pseudo-labels from v5.2 (HI=0.92, LO=0.04, weight=0.5)

Anchor: v5.2 LB 0.94924. Goal: +0.001-0.003 via features + pseudo + class diversity.
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
                           "xgboost", "lightgbm", "catboost", "optuna",
                           "scikit-learn", "pandas", "numpy", "mlflow",
                           "kaggle", "torch"])

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    v52_sub = pd.read_csv(f"{DATA_DIR}/submission_v5_2_optuna.csv").sort_values("id").reset_index(drop=True)
    TARGET = "PitNextLap"; ID_COL = "id"; SEED = 42
    np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"train={train.shape}  test={test.shape}  v5.2={v52_sub.shape}")

    # ─── 1. Pseudo-label from v5.2 ───
    test_sorted = test.sort_values(ID_COL).reset_index(drop=True)
    assert np.array_equal(test_sorted[ID_COL].values, v52_sub[ID_COL].values)
    HI = 0.92; LO = 0.04; PSEUDO_W = 0.5
    p = v52_sub["PitNextLap"].values
    mask_pos = p >= HI; mask_neg = p <= LO
    n_pos = int(mask_pos.sum()); n_neg = int(mask_neg.sum())
    print(f"pseudo HI={HI} LO={LO}: POS={n_pos} NEG={n_neg}")
    pseudo = test_sorted[mask_pos | mask_neg].copy().reset_index(drop=True)
    pseudo[TARGET] = np.where(p[mask_pos | mask_neg] >= HI, 1, 0).astype(int)
    pseudo["sample_weight"] = PSEUDO_W
    pseudo["is_pseudo"] = 1

    train["sample_weight"] = 1.0; train["is_pseudo"] = 0
    all_train = pd.concat([train, pseudo], axis=0, ignore_index=True)
    print(f"all_train (orig+pseudo)={all_train.shape}")

    # ─── 2. Combined for feature eng ───
    test["sample_weight"] = 1.0; test["is_pseudo"] = 0; test[TARGET] = np.nan
    combined = pd.concat([all_train, test], axis=0, ignore_index=True)
    n_all_train = len(all_train)

    # ─── 3. Feature engineering ───
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

    LE_COLS = ["Driver", "Race", "Compound", "RaceYear"]
    for c in LE_COLS:
        combined[f"LE_{c}"] = LabelEncoder().fit_transform(combined[c].astype(str))

    # Split: val=Year=2025 ORIGINAL train only. tr = all else (orig pre-2025 + pseudo).
    is_train_mask = combined[TARGET].notna()
    is_pseudo_mask = combined["is_pseudo"] == 1
    is_year2025_orig = (combined["Year"] == 2025) & is_train_mask & (~is_pseudo_mask)

    train_fe_full = combined[is_train_mask].copy().reset_index(drop=True)
    test_fe = combined[~is_train_mask].copy().reset_index(drop=True)

    val_mask = (train_fe_full["Year"] == 2025) & (train_fe_full["is_pseudo"] == 0)
    train_mask = ~val_mask  # incl pseudo
    print(f"train_fe={train_fe_full.shape}  val_size={val_mask.sum()}  tr_size={train_mask.sum()}")
    print(f"tr pos_rate={train_fe_full.loc[train_mask, TARGET].mean():.4f}  "
          f"val pos_rate={train_fe_full.loc[val_mask, TARGET].mean():.4f}")

    FEATURES = [
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
    ]
    X_full = train_fe_full[FEATURES].values.astype(np.float32)
    y_full = train_fe_full[TARGET].values.astype(int)
    w_full = train_fe_full["sample_weight"].values.astype(np.float32)
    X_test = test_fe[FEATURES].values.astype(np.float32)
    test_ids = test_fe[ID_COL].values.astype(int)
    tr_idx = np.where(train_mask)[0]
    va_idx = np.where(val_mask)[0]
    X_tr, X_va = X_full[tr_idx], X_full[va_idx]
    y_tr, y_va = y_full[tr_idx], y_full[va_idx]
    w_tr = w_full[tr_idx]
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"X_tr={X_tr.shape}  X_va={X_va.shape}  scale_pos={scale_pos:.2f}")

    # ─── 4. LightGBM ───
    print("\n=== LightGBM ===")
    t0 = time.time()
    dtrain = lgb.Dataset(X_tr, y_tr, weight=w_tr)
    dvalid = lgb.Dataset(X_va, y_va)
    LGB_PARAMS = dict(
        objective="binary", metric="auc", learning_rate=0.02,
        num_leaves=127, max_depth=-1, min_child_samples=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.5, scale_pos_weight=scale_pos,
        random_state=SEED, verbosity=-1,
    )
    m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=5000,
                      valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(150, verbose=False),
                                 lgb.log_evaluation(0)])
    val_lgb = m_lgb.predict(X_va)
    test_lgb = m_lgb.predict(X_test)
    auc_lgb = roc_auc_score(y_va, val_lgb)
    print(f"LGB val AUC: {auc_lgb:.5f}  iter={m_lgb.best_iteration}  ({time.time()-t0:.0f}s)")

    # ─── 5. XGBoost ───
    print("\n=== XGBoost ===")
    t0 = time.time()
    XGB_PARAMS = dict(
        max_depth=8, learning_rate=0.03, n_estimators=3000,
        subsample=0.85, colsample_bytree=0.7, min_child_weight=5,
        reg_lambda=2.0, reg_alpha=0.1, gamma=0.1,
        scale_pos_weight=scale_pos,
        tree_method="hist", device="cuda", eval_metric="auc",
        early_stopping_rounds=100, random_state=SEED, verbosity=0,
    )
    m_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    m_xgb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_xgb = m_xgb.predict_proba(X_va)[:, 1]
    test_xgb = m_xgb.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_va, val_xgb)
    print(f"XGB val AUC: {auc_xgb:.5f}  iter={m_xgb.best_iteration}  ({time.time()-t0:.0f}s)")

    # ─── 6. CatBoost ───
    print("\n=== CatBoost ===")
    t0 = time.time()
    CB_CATS = ["LE_Driver", "LE_Race", "LE_Compound", "LE_RaceYear"]
    Xc_df = train_fe_full[FEATURES].copy()
    Xc_test_df = test_fe[FEATURES].copy()
    for c in CB_CATS:
        Xc_df[c] = Xc_df[c].astype(int)
        Xc_test_df[c] = Xc_test_df[c].astype(int)
    CB_PARAMS = dict(
        iterations=3000, learning_rate=0.03, depth=8,
        l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=0.5,
        border_count=128, eval_metric="AUC", scale_pos_weight=scale_pos,
        random_seed=SEED, verbose=0, early_stopping_rounds=100, task_type="GPU",
    )
    tp = Pool(Xc_df.iloc[tr_idx], y_tr, weight=w_tr, cat_features=CB_CATS)
    vp = Pool(Xc_df.iloc[va_idx], y_va, cat_features=CB_CATS)
    m_cb = CatBoostClassifier(**CB_PARAMS)
    m_cb.fit(tp, eval_set=vp, verbose=0)
    val_cb = m_cb.predict_proba(Xc_df.iloc[va_idx])[:, 1]
    test_cb = m_cb.predict_proba(Xc_test_df)[:, 1]
    auc_cb = roc_auc_score(y_va, val_cb)
    print(f"CB val AUC: {auc_cb:.5f}  iter={m_cb.get_best_iteration()}  ({time.time()-t0:.0f}s)")

    # ─── 7. MLP ───
    print("\n=== MLP ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 256), nn.LayerNorm(256), nn.Mish(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.Mish(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.LayerNorm(64), nn.Mish(), nn.Dropout(0.1),
                nn.Linear(64, 1), nn.Sigmoid(),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    EPOCHS = 20; BS = 4096
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_test)

    ds = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr.astype(np.float32), dtype=torch.float32),
        torch.tensor(w_tr, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=False)
    Xva_t = torch.tensor(X_va_s, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(X_te_s, dtype=torch.float32, device=device)

    # 3-seed averaging for MLP
    val_mlp_seeds = []; test_mlp_seeds = []
    for s_seed in [42, 7, 555]:
        torch.manual_seed(s_seed)
        model = MLP(X_tr.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.BCELoss(reduction="none")
        best_a = 0; best_v = None; best_t = None
        for ep in range(EPOCHS):
            model.train()
            for xb, yb, wb in dl:
                xb=xb.to(device); yb=yb.to(device); wb=wb.to(device)
                opt.zero_grad()
                pp = model(xb)
                ll = (loss_fn(pp, yb) * wb).mean()
                ll.backward(); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pv = model(Xva_t).cpu().numpy()
            va = roc_auc_score(y_va, pv)
            if va > best_a:
                best_a = va; best_v = pv
                with torch.no_grad(): best_t = model(Xte_t).cpu().numpy()
        print(f"  seed={s_seed} best val AUC: {best_a:.5f}")
        val_mlp_seeds.append(best_v); test_mlp_seeds.append(best_t)
        del model, opt, sched
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    val_mlp = np.mean(val_mlp_seeds, axis=0)
    test_mlp = np.mean(test_mlp_seeds, axis=0)
    auc_mlp = roc_auc_score(y_va, val_mlp)
    print(f"MLP 3-seed avg val AUC: {auc_mlp:.5f}")

    # ─── 8. Rank-norm + Optuna blend ───
    print("\n=== Blend ===")
    def rank_norm(x):
        return (rankdata(x) - 1) / (len(x) - 1)
    r_lgb = rank_norm(val_lgb); r_xgb = rank_norm(val_xgb)
    r_cb = rank_norm(val_cb); r_mlp = rank_norm(val_mlp)
    rt_lgb = rank_norm(test_lgb); rt_xgb = rank_norm(test_xgb)
    rt_cb = rank_norm(test_cb); rt_mlp = rank_norm(test_mlp)
    print(f"per-model val AUC: lgb={auc_lgb:.5f} xgb={auc_xgb:.5f} cb={auc_cb:.5f} mlp={auc_mlp:.5f}")
    print("Spearman val:")
    rk = {"lgb": r_lgb, "xgb": r_xgb, "cb": r_cb, "mlp": r_mlp}
    names = list(rk.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            sp = float(np.corrcoef(rk[names[i]], rk[names[j]])[0, 1])
            print(f"  {names[i]} vs {names[j]}: {sp:.5f}")

    def blend_val(ws):
        s = sum(ws)
        if s == 0: return 0.5
        return roc_auc_score(y_va, (ws[0]*r_lgb + ws[1]*r_xgb + ws[2]*r_cb + ws[3]*r_mlp) / s)

    def objective(trial):
        wl = trial.suggest_float("w_lgb", 0.0, 1.0)
        wx = trial.suggest_float("w_xgb", 0.0, 1.0)
        wc = trial.suggest_float("w_cb", 0.0, 1.0)
        wm = trial.suggest_float("w_mlp", 0.0, 1.0)
        return blend_val((wl, wx, wc, wm))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=200, show_progress_bar=False)
    print(f"best blend val AUC: {study.best_value:.5f}")
    bp = study.best_params
    s = bp["w_lgb"] + bp["w_xgb"] + bp["w_cb"] + bp["w_mlp"]
    W = {k: v/s for k, v in bp.items()}
    print(f"weights (normed): {W}")

    final_val = W["w_lgb"]*r_lgb + W["w_xgb"]*r_xgb + W["w_cb"]*r_cb + W["w_mlp"]*r_mlp
    final_test = W["w_lgb"]*rt_lgb + W["w_xgb"]*rt_xgb + W["w_cb"]*rt_cb + W["w_mlp"]*rt_mlp
    blend_auc = roc_auc_score(y_va, final_val)
    v52_lb = 0.94924
    pred_lb = blend_auc + 0.042  # v5.2 had val=0.9075 → LB=0.94924 → gap 0.0417
    print(f"FINAL blend val AUC: {blend_auc:.5f}  pred LB={pred_lb:.4f}  vs v5.2 LB {v52_lb}")

    SUB_PATH = f"{DATA_DIR}/submission_v13_2_year_pseudo.csv"
    OOF_PATH = f"{DATA_DIR}/oof_v13_2_year_pseudo.csv"
    pd.DataFrame({"id": test_ids, "PitNextLap": final_test}).to_csv(SUB_PATH, index=False)
    pd.DataFrame({"lgb": val_lgb, "xgb": val_xgb, "cb": val_cb, "mlp": val_mlp}).to_csv(OOF_PATH, index=False)
    print(f"submission -> {SUB_PATH}")

    mlflow.set_tracking_uri("databricks")
    try: mlflow.set_experiment("/Shared/databricks-ai-intern/f1_pitstops")
    except Exception: mlflow.set_experiment("/Shared/databricks-ai-intern")
    with mlflow.start_run(run_name="f1_pitstops_v13_2_year_pseudo") as run:
        mlflow.log_param("version", "v13_2_year_pseudo")
        mlflow.log_param("pseudo_HI", HI); mlflow.log_param("pseudo_LO", LO)
        mlflow.log_param("pseudo_w", PSEUDO_W)
        mlflow.log_param("pseudo_n_pos", n_pos); mlflow.log_param("pseudo_n_neg", n_neg)
        mlflow.log_metric("lgb_val", float(auc_lgb))
        mlflow.log_metric("xgb_val", float(auc_xgb))
        mlflow.log_metric("cb_val", float(auc_cb))
        mlflow.log_metric("mlp_val", float(auc_mlp))
        mlflow.log_metric("blend_val", float(blend_auc))
        mlflow.log_metric("pred_lb", float(pred_lb))
        for k, v in W.items(): mlflow.log_param(k, float(v))
        mlflow.log_artifact(SUB_PATH)
        run_id = run.info.run_id

    # ─── 9. Submit ───
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    TOK = "KGAT_d7a0018cbb8131f82d3dd175d2e4cdb6"
    TP = os.path.expanduser("~/.kaggle/access_token")
    with open(TP, "w") as f: f.write(TOK)
    os.chmod(TP, 0o600)
    os.environ["KAGGLE_API_TOKEN"] = TOK
    COMP = "playground-series-s6e5"
    MSG = (f"v13.2 year-val+pseudo: LGB+XGB+CB+MLP rank blend on Year=2025 val. "
           f"Pseudo HI={HI}/LO={LO} weight={PSEUDO_W}. "
           f"vals lgb={auc_lgb:.4f} xgb={auc_xgb:.4f} cb={auc_cb:.4f} mlp={auc_mlp:.4f} blend={blend_auc:.4f}.")
    print(f"\nSubmitting v13.2...")
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", COMP,
                        "-f", SUB_PATH, "-m", MSG], capture_output=True, text=True)
    print("STDOUT:", r.stdout); print("rc:", r.returncode)

    print("\nPolling...")
    final = ""
    for i in range(40):
        time.sleep(15)
        rs = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP],
                            capture_output=True, text=True)
        for line in rs.stdout.split("\n"):
            if "submission_v13_2_year_pseudo.csv" in line:
                print(f"t={(i+1)*15}s: {line}")
                final = line
                if line.split() and line.split()[-1].startswith("0."): break
        if final and final.split()[-1].startswith("0."): break

    print("\n" + "=" * 60)
    print("V13.2 REPORT")
    print("=" * 60)
    print(json.dumps({
        "version": "v13_2_year_pseudo",
        "pseudo_HI": HI, "pseudo_LO": LO, "pseudo_w": PSEUDO_W,
        "pseudo_n_pos": n_pos, "pseudo_n_neg": n_neg,
        "lgb_val": float(auc_lgb),
        "xgb_val": float(auc_xgb),
        "cb_val": float(auc_cb),
        "mlp_val": float(auc_mlp),
        "blend_val": float(blend_auc),
        "weights": W,
        "pred_lb": float(pred_lb),
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
