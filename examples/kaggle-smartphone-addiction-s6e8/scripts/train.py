"""Kaggle Playground S6E8 (Predicting Smartphone Addiction) — one trainer, many versions.

Every attempt is a row in ../versions.yaml; nothing about a version lives in this file.
Runs unchanged locally (CPU) and as a Databricks job against a UC Volume.

    python train.py --version v2_lgbm_te_freq --data-dir /tmp/s6e8 --out-dir /tmp/s6e8work

CV is StratifiedKFold(5, shuffle=True, random_state=42) on original row order for every
version, so OOF arrays stay row-aligned and can be stacked. Predictions are kept in
float64: at this level of saturation float32 rounding reorders a meaningful fraction of
test ranks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "addicted_label"
ID = "id"
NUM = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
CAT = ["gender", "stress_level", "academic_work_impact"]
FEATURE_COLS = NUM + CAT

TE_SMOOTHING = 10.0
MISSING_LEVEL = "__missing__"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- feature blocks


def _levels(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cast columns to string levels, with NaN as its own explicit level.

    `.astype(str)` alone is not safe here: on pandas >= 3.0 it preserves NA rather
    than producing a "nan" level, which silently drops those rows out of every
    group-by that follows.
    """
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = df[c].astype(object).where(df[c].notna(), MISSING_LEVEL).astype(str)
    return out


def add_frequency(train_lv: pd.DataFrame, test_lv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transductive level counts over train+test. Uses no labels, so this is not leakage."""
    tr_out, te_out = pd.DataFrame(index=train_lv.index), pd.DataFrame(index=test_lv.index)
    for c in train_lv.columns:
        counts = pd.concat([train_lv[c], test_lv[c]]).value_counts()
        tr_out[f"freq_{c}"] = train_lv[c].map(counts).astype(np.float64)
        te_out[f"freq_{c}"] = test_lv[c].map(counts).astype(np.float64)
    return tr_out, te_out


def _te_map(levels: pd.Series, y: np.ndarray, prior: float) -> pd.Series:
    stats = pd.DataFrame({"lv": levels.values, "y": y}).groupby("lv")["y"].agg(["sum", "count"])
    return (stats["sum"] + prior * TE_SMOOTHING) / (stats["count"] + TE_SMOOTHING)


def target_encode(
    train_lv: pd.DataFrame,
    y: np.ndarray,
    test_lv: pd.DataFrame,
    fold_tr: np.ndarray,
    fold_va: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nested target encoding for one outer fold.

    The fold's own training rows are encoded out-of-fold via an inner 5-way split, so
    no row's encoding is built from its own label. Validation and test rows are encoded
    from the full outer-training portion. Encoding the training rows in-fold instead is
    the classic mistake: the model learns to trust a feature that partly *is* the label,
    and OOF then flatters badly.
    """
    prior = float(y[fold_tr].mean())
    tr_enc = pd.DataFrame(index=train_lv.index[fold_tr], dtype=np.float64)
    va_enc = pd.DataFrame(index=train_lv.index[fold_va], dtype=np.float64)
    te_enc = pd.DataFrame(index=test_lv.index, dtype=np.float64)

    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    inner_splits = list(inner.split(fold_tr, y[fold_tr]))

    for c in train_lv.columns:
        lv_tr = train_lv[c].iloc[fold_tr]
        col = np.empty(len(fold_tr), dtype=np.float64)
        for i_tr, i_va in inner_splits:
            m = _te_map(lv_tr.iloc[i_tr], y[fold_tr][i_tr], prior)
            col[i_va] = lv_tr.iloc[i_va].map(m).fillna(prior).to_numpy(dtype=np.float64)
        tr_enc[f"te_{c}"] = col

        full = _te_map(lv_tr, y[fold_tr], prior)
        va_enc[f"te_{c}"] = train_lv[c].iloc[fold_va].map(full).fillna(prior).to_numpy(dtype=np.float64)
        te_enc[f"te_{c}"] = test_lv[c].map(full).fillna(prior).to_numpy(dtype=np.float64)

    return tr_enc, va_enc, te_enc


def add_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio / accounting features on the screen-time block.

    `residual` is the slack in daily - (social + gaming + work). On this synthetic data
    that slack is >= 0 for every one of the 421,427 train and 182,287 test rows where all
    four are present; the original source dataset violates it 60.7% of the time. The
    generator built the constraint in, so the slack is a real quantity, not noise.
    """
    out = pd.DataFrame(index=df.index, dtype=np.float64)
    daily, weekend = df["daily_screen_time_hours"], df["weekend_screen_time"]
    sm, gm, ws = df["social_media_hours"], df["gaming_hours"], df["work_study_hours"]

    out["residual"] = daily - (sm + gm + ws)
    out["leisure"] = sm + gm
    out["leisure_frac"] = (sm + gm) / daily
    out["sm_frac"] = sm / daily
    out["gm_frac"] = gm / daily
    out["ws_frac"] = ws / daily
    out["weekend_gap"] = weekend - daily
    out["weekend_ratio"] = weekend / daily
    out["screen_sleep"] = daily / df["sleep_hours"]
    out["notif_per_open"] = df["notifications_per_day"] / df["app_opens_per_day"]
    out["notif_per_hour"] = df["notifications_per_day"] / daily
    out["sleep_plus_screen"] = df["sleep_hours"] + daily
    return out.replace([np.inf, -np.inf], np.nan)


def add_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Generator-identity probes beyond the single accounting residual.

    `weekend = daily + Uniform(0.5, 3.0)` holds exactly in the original source data; the
    synthetic generator kept the correlation but only ~54% of comparable rows land inside
    that window, so whether a row falls inside it is informative rather than a rule to
    impute with. Row-level NaN counts were in this block and measured +0.00002 on a paired
    fold-0 comparison — dropped, same verdict as the per-column NaN indicators.
    """
    out = pd.DataFrame(index=df.index, dtype=np.float64)
    daily, weekend = df["daily_screen_time_hours"], df["weekend_screen_time"]
    sm, gm, ws = df["social_media_hours"], df["gaming_hours"], df["work_study_hours"]
    gap = weekend - daily
    out["wk_in_window"] = ((gap >= 0.5) & (gap <= 3.0)).astype(np.float64).where(gap.notna())
    out["wk_window_pos"] = ((gap - 0.5) / 2.5).where(gap.notna())
    res = daily - (sm + gm + ws)
    out["residual_frac"] = res / daily
    out["residual_zero"] = (res.abs() < 1e-9).astype(np.float64).where(res.notna())
    return out.replace([np.inf, -np.inf], np.nan)


def add_lattice(df: pd.DataFrame) -> pd.DataFrame:
    """First fractional digit of each continuous column — exposes generator rounding."""
    out = pd.DataFrame(index=df.index, dtype=np.float64)
    for c in ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
              "work_study_hours", "sleep_hours", "weekend_screen_time"]:
        out[f"dec_{c}"] = np.round((df[c] * 10) % 10).where(df[c].notna())
    return out


# ---------------------------------------------------------------- models


def fit_predict(model: str, Xtr, ytr, Xva, yva, Xte, cat_cols: list[str], seed: int,
                overrides: dict | None = None):
    """Train one fold. Returns (val_pred, test_pred, best_iteration).

    `overrides` is merged over the learner's defaults, so a version that only wants
    different hyperparameters is a row in versions.yaml rather than a branch in here.
    """
    overrides = dict(overrides or {})
    if model == "lgbm":
        import lightgbm as lgb

        # Deliberately faster than the published reference config for this competition
        # (lr=0.01, max_bin=1023, 10k rounds). That config measured ~15x slower per fold
        # than the XGBoost member here — roughly 8x of it from max_bin 1023 and the extra
        # trees a 0.01 rate needs — which put a full 5-model ladder out of reach in this
        # session. lr=0.02 with max_bin=255 costs on the order of 0.0002-0.0005 OOF AUC
        # and returns most of the wall clock. Raise both back up if you have the hours.
        params = dict(
            objective="binary", metric="auc", boosting_type="gbdt",
            learning_rate=0.02, num_leaves=127, max_depth=-1,
            min_child_samples=200, feature_fraction=0.5,
            bagging_fraction=0.75, bagging_freq=5,
            reg_alpha=0.1, reg_lambda=1.0, max_bin=255,
            verbose=-1, n_jobs=-1, seed=seed, force_col_wise=True,
        )
        rounds = int(overrides.pop("num_boost_round", 6000))
        stopping = int(overrides.pop("early_stopping_rounds", 200))
        params.update(overrides)
        m = lgb.train(
            params, lgb.Dataset(Xtr, ytr), num_boost_round=rounds,
            valid_sets=[lgb.Dataset(Xva, yva)],
            callbacks=[lgb.early_stopping(stopping, verbose=False)],
        )
        it = m.best_iteration
        return (m.predict(Xva, num_iteration=it).astype(np.float64),
                m.predict(Xte, num_iteration=it).astype(np.float64), it)

    if model == "xgb":
        import xgboost as xgb

        xgb_params = dict(
            n_estimators=6000, learning_rate=0.02, max_depth=7,
            min_child_weight=15, subsample=0.85, colsample_bytree=0.5,
            reg_alpha=0.05, reg_lambda=3.0,
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            early_stopping_rounds=200, random_state=seed, n_jobs=-1,
            enable_categorical=True, max_cat_to_onehot=1,
        )
        xgb_params.update(overrides)
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        it = m.best_iteration
        return (m.predict_proba(Xva)[:, 1].astype(np.float64),
                m.predict_proba(Xte)[:, 1].astype(np.float64), it)

    if model == "cat":
        from catboost import CatBoostClassifier, Pool

        # CatBoost wants categoricals as strings with no NaN.
        def prep(X):
            X = X.copy()
            for c in cat_cols:
                X[c] = X[c].astype(object).where(X[c].notna(), MISSING_LEVEL).astype(str)
            return X

        Xtr, Xva, Xte = prep(Xtr), prep(Xva), prep(Xte)
        cat_params = dict(
            iterations=4000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            eval_metric="AUC", early_stopping_rounds=150,
            random_seed=seed, verbose=0, thread_count=-1,
            border_count=128,  # CatBoost's max_bin; 254 default is slower for little gain here
            # Without this CatBoost writes catboost_info/ into the current directory,
            # which means training litter lands in the repo when run from scripts/.
            train_dir=str(Path(tempfile.gettempdir()) / "catboost_info"),
            allow_writing_files=False,
        )
        cat_params.update(overrides)
        m = CatBoostClassifier(**cat_params)
        m.fit(Pool(Xtr, ytr, cat_features=cat_cols),
              eval_set=Pool(Xva, yva, cat_features=cat_cols))
        it = m.get_best_iteration()
        return (m.predict_proba(Xva)[:, 1].astype(np.float64),
                m.predict_proba(Xte)[:, 1].astype(np.float64), it)

    if model == "tabm":
        return _fit_tabm(Xtr, ytr, Xva, yva, Xte, cat_cols, seed, overrides)

    raise ValueError(f"unknown model: {model}")


def _fit_tabm(Xtr, ytr, Xva, yva, Xte, cat_cols: list[str], seed: int, overrides: dict):
    """TabM: a k-member batch ensemble of MLPs with piecewise-linear numeric embeddings.

    The one member here whose mechanism is not a decision tree. Two things it needs that
    the GBMs do not: numeric columns quantile-transformed to a normal marginal (an MLP
    cannot ignore scale the way a split can), and NaNs filled, since there is no
    missing-value branch to send them down. Fill is the training fold's median, fitted on
    the training rows only.
    """
    import torch
    import torch.nn as nn
    import rtdl_num_embeddings
    from sklearn.preprocessing import QuantileTransformer
    from tabm import TabM

    p = dict(k=32, epochs=40, batch_size=4096, lr=3e-3, weight_decay=3e-4,
             d_embedding=16, n_bins=48, patience=4)
    p.update(overrides)

    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_cols = [c for c in Xtr.columns if c not in cat_cols]

    def cat_mat(X):
        out = np.empty((len(X), len(cat_cols)), dtype=np.int64)
        for j, c in enumerate(cat_cols):
            codes = X[c].cat.codes.to_numpy()
            out[:, j] = np.where(codes < 0, X[c].cat.categories.size, codes)
        return out

    cards = [Xtr[c].cat.categories.size + 1 for c in cat_cols]  # +1 for the NaN code

    med = Xtr[num_cols].median()
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=200_000, random_state=seed)
    Ntr = qt.fit_transform(Xtr[num_cols].fillna(med)).astype(np.float32)
    Nva = qt.transform(Xva[num_cols].fillna(med)).astype(np.float32)
    Nte = qt.transform(Xte[num_cols].fillna(med)).astype(np.float32)
    Ctr, Cva, Cte = cat_mat(Xtr), cat_mat(Xva), cat_mat(Xte)

    tNtr, tytr = torch.tensor(Ntr), torch.tensor(ytr.astype(np.int64))
    tCtr = torch.tensor(Ctr)
    bins = rtdl_num_embeddings.compute_bins(tNtr[:100_000], n_bins=int(p["n_bins"]))
    emb = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        bins, d_embedding=int(p["d_embedding"]), activation=False, version="B")
    m = TabM.make(n_num_features=Ntr.shape[1], cat_cardinalities=cards, d_out=2,
                  k=int(p["k"]), num_embeddings=emb).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=float(p["lr"]),
                            weight_decay=float(p["weight_decay"]))

    def predict(N, C):
        m.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(N), 16384):
                o = m(torch.tensor(N[b:b + 16384]).to(dev),
                      torch.tensor(C[b:b + 16384]).to(dev))
                out.append(torch.softmax(o, -1).mean(1)[:, 1].cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    bs, k = int(p["batch_size"]), int(p["k"])
    n_batches = int(np.ceil(len(tNtr) / bs))
    best_auc, best_state, waited = -1.0, None, 0
    for ep in range(int(p["epochs"])):
        m.train()
        perm = torch.randperm(len(tNtr))
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs]
            xb, cb, yb = tNtr[idx].to(dev), tCtr[idx].to(dev), tytr[idx].to(dev)
            out = m(xb, cb)  # (batch, k, 2)
            loss = nn.functional.cross_entropy(out.reshape(-1, 2), yb.repeat_interleave(k))
            opt.zero_grad()
            loss.backward()
            opt.step()
        auc = roc_auc_score(yva, predict(Nva, Cva))
        if auc > best_auc + 1e-6:
            best_auc, waited = auc, 0
            best_state = {kk: v.detach().clone() for kk, v in m.state_dict().items()}
        else:
            waited += 1
            if waited >= int(p["patience"]):
                break
        log(f"    tabm ep{ep} val_auc={auc:.6f} best={best_auc:.6f}")

    m.load_state_dict(best_state)
    return predict(Nva, Cva), predict(Nte, Cte), ep + 1


# ---------------------------------------------------------------- run one version


def run_lookup_version(spec: dict, cv: dict, train: pd.DataFrame, y: np.ndarray,
                       test: pd.DataFrame, test_ids: np.ndarray, out_dir: Path,
                       frac: float) -> dict:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import QuantileTransformer

    p = dict(
        d=128, k=24, layers=4, heads=8, drop=0.1,
        epochs=32, batch_size=2048, lr=2e-3, aug=0.10,
        weight_decay=1e-5, embedding_weight_decay=3e-4,
        pct_start=0.15, grad_clip=1.0,
        ema_decay=0.999, patience=5, predict_batch_size=16384,
    )
    p.update(spec.get("params") or {})
    model_seeds = [int(x) for x in (spec.get("seeds") or [int(cv["seed"])])]
    folds = int(spec.get("folds") or cv["folds"])

    both = pd.concat([train[FEATURE_COLS], test[FEATURE_COLS]], ignore_index=True)

    # The synthetic generator reused exact quantised values. A shared table with disjoint
    # per-column ranges preserves that identity without letting equal strings in unrelated
    # columns share a learned vector.
    ids, vocab = [], []
    for c in FEATURE_COLS:
        s = both[c].astype(object).where(both[c].notna(), MISSING_LEVEL).astype(str)
        cats = sorted(v for v in s.unique() if v != MISSING_LEVEL)
        mapping = {v: i + 1 for i, v in enumerate(cats)}
        mapping[MISSING_LEVEL] = 0
        ids.append(s.map(mapping).to_numpy(dtype=np.int64))
        vocab.append(len(cats) + 1)
    value_ids = np.stack(ids, axis=1)
    offsets = np.concatenate([[0], np.cumsum(vocab)[:-1]]).astype(np.int64)
    value_ids += offsets[None, :]
    total_vocab = int(sum(vocab))

    comp = ["social_media_hours", "gaming_hours", "work_study_hours"]
    daily, weekend = both["daily_screen_time_hours"], both["weekend_screen_time"]
    sgw = both[comp].sum(axis=1)
    residual = daily - sgw
    derived = pd.DataFrame({
        "other_screen": residual,
        "sgw": sgw,
        "other_frac": residual / daily.clip(lower=0.1),
        "wk_minus_sgw": weekend - sgw,
        "wk_other": weekend - residual,
        "sgw_frac": sgw / daily.clip(lower=0.1),
        "weekend_gap": weekend - daily,
    })

    def rank_gauss(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros((len(df), df.shape[1]), dtype=np.float32)
        missing = np.zeros_like(values)
        for j, c in enumerate(df.columns):
            v = df[c].to_numpy(dtype=np.float64)
            observed = ~np.isnan(v)
            if observed.sum() > 10:
                qt = QuantileTransformer(
                    n_quantiles=1000, output_distribution="normal",
                    subsample=400_000, random_state=0,
                )
                values[observed, j] = qt.fit_transform(
                    v[observed].reshape(-1, 1)
                ).ravel().astype(np.float32)
            missing[~observed, j] = 1.0
        return values, missing

    # Categorical columns deliberately have no smooth branch. Their exact-value embedding
    # remains active, including a distinct learned missing vector at local index zero.
    raw_numeric = pd.DataFrame({
        c: both[c] if c in NUM else np.nan for c in FEATURE_COLS
    })
    column_numeric, column_missing = rank_gauss(raw_numeric)
    derived_numeric, derived_missing = rank_gauss(derived)
    n_columns, n_derived = len(FEATURE_COLS), derived.shape[1]
    n_tokens = 1 + n_columns + n_derived

    t_ids = torch.from_numpy(value_ids)
    t_cn = torch.from_numpy(column_numeric)
    t_cm = torch.from_numpy(column_missing)
    t_dn = torch.from_numpy(derived_numeric)
    t_dm = torch.from_numpy(derived_missing)
    t_y = torch.from_numpy(y.astype(np.float32))
    t_offsets = torch.from_numpy(offsets)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = None
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log(f"  lookup device={device.type} vocab={total_vocab} tokens={n_tokens} "
        f"({n_columns} raw + {n_derived} derived + CLS)")

    class PLR(nn.Module):
        def __init__(self, n_features: int, k: int, d: int, sigma: float = 0.5):
            super().__init__()
            self.f = nn.Parameter(torch.randn(n_features, k) * sigma)
            self.w = nn.Parameter(torch.randn(n_features, 2 * k, d) / math.sqrt(2 * k))
            self.b = nn.Parameter(torch.zeros(n_features, d))

        def forward(self, x):
            z = 2 * math.pi * x.unsqueeze(-1) * self.f.unsqueeze(0)
            z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
            return torch.einsum("bfk,fkd->bfd", z, self.w) + self.b

    class LookupTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            d = int(p["d"])
            self.emb = nn.Embedding(total_vocab, d)
            nn.init.normal_(self.emb.weight, std=0.02)
            self.plr_c = PLR(n_columns, int(p["k"]), d)
            self.plr_d = PLR(n_derived, int(p["k"]), d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.pos = nn.Parameter(torch.randn(1, n_tokens, d) * 0.02)
            self.edrop = nn.Dropout(float(p["drop"]))
            layer = nn.TransformerEncoderLayer(
                d, int(p["heads"]), d * 2, float(p["drop"]), activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, int(p["layers"]))
            self.head = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                nn.Dropout(float(p["drop"])), nn.Linear(d, 1),
            )

        def forward(self, idx, cn, cm, dn, dm):
            batch = idx.shape[0]
            tok_c = self.emb(idx) + self.plr_c(cn) * (1 - cm).unsqueeze(-1)
            tok_d = self.plr_d(dn) * (1 - dm).unsqueeze(-1)
            tokens = torch.cat([self.cls.expand(batch, -1, -1), tok_c, tok_d], dim=1)
            encoded = self.transformer(self.edrop(tokens + self.pos))
            return self.head(encoded[:, 0]).squeeze(-1)

    def fit_fold(i_tr: np.ndarray, i_va: np.ndarray, fold: int,
                 model_seed: int) -> tuple[np.ndarray, np.ndarray, int]:
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)
        generator = torch.Generator().manual_seed(model_seed)
        model = LookupTransformer().to(device)
        embedding_params = [v for name, v in model.named_parameters() if name.startswith("emb.")]
        other_params = [v for name, v in model.named_parameters() if not name.startswith("emb.")]
        optimizer = torch.optim.AdamW([
            {"params": other_params, "weight_decay": float(p["weight_decay"])},
            {"params": embedding_params,
             "weight_decay": float(p["embedding_weight_decay"])},
        ], lr=float(p["lr"]))
        batch_size = int(p["batch_size"])
        epochs = int(p["epochs"])
        total_steps = math.ceil(len(i_tr) / batch_size) * epochs + 10
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, float(p["lr"]), total_steps=total_steps,
            pct_start=float(p["pct_start"]),
        )
        scaler = None
        if amp_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda")
        loss_fn = nn.BCEWithLogitsLoss()
        parameters = list(model.parameters())
        ema = [v.detach().clone() for v in parameters]
        ema_step = 0

        I, CN, CM = t_ids[i_tr].to(device), t_cn[i_tr].to(device), t_cm[i_tr].to(device)
        DN, DM, Y = t_dn[i_tr].to(device), t_dm[i_tr].to(device), t_y[i_tr].to(device)
        Iv, CNv, CMv = t_ids[i_va].to(device), t_cn[i_va].to(device), t_cm[i_va].to(device)
        DNv, DMv = t_dn[i_va].to(device), t_dm[i_va].to(device)
        It, CNt, CMt = t_ids[len(train):].to(device), t_cn[len(train):].to(device), t_cm[len(train):].to(device)
        DNt, DMt = t_dn[len(train):].to(device), t_dm[len(train):].to(device)
        missing_ids = t_offsets.to(device)

        def forward(*args):
            if amp_dtype is None:
                return model(*args)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                return model(*args)

        def predict(idx, cn, cm, dn, dm) -> np.ndarray:
            model.eval()
            out = []
            chunk = int(p["predict_batch_size"])
            with torch.no_grad():
                for start in range(0, len(idx), chunk):
                    logits = forward(
                        idx[start:start + chunk], cn[start:start + chunk],
                        cm[start:start + chunk], dn[start:start + chunk],
                        dm[start:start + chunk],
                    )
                    out.append(torch.sigmoid(logits).float().cpu().numpy())
            return np.concatenate(out).astype(np.float64)

        best_auc, best_weights, best_epoch, bad = -1.0, None, -1, 0
        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(len(i_tr), generator=generator).to(device)
            for start in range(0, len(i_tr), batch_size):
                sl = permutation[start:start + batch_size]
                idx, cm = I[sl].clone(), CM[sl].clone()
                if float(p["aug"]) > 0:
                    drop = torch.rand(idx.shape, device=device) < float(p["aug"])
                    idx = torch.where(drop, missing_ids.expand_as(idx), idx)
                    cm = torch.maximum(cm, drop.float())
                loss = loss_fn(forward(idx, CN[sl], cm, DN[sl], DM[sl]), Y[sl])
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, float(p["grad_clip"]))
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, float(p["grad_clip"]))
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                # Warm up the EMA horizon instead of using a fixed decay from step 1.
                # At decay=0.999 a 500-step run still carries 0.999**500 = 61% of the
                # random initialisation, which made a short smoke run score AUC 0.45 and
                # look like a label bug rather than an under-trained average.
                ema_step += 1
                decay = min(float(p["ema_decay"]), (1.0 + ema_step) / (10.0 + ema_step))
                with torch.no_grad():
                    for avg, current in zip(ema, parameters):
                        avg.mul_(decay).add_(current.detach(), alpha=1 - decay)

            should_evaluate = epoch == epochs - 1 or (epoch >= 5 and epoch % 2 == 1)
            if should_evaluate:
                current_weights = [v.detach().clone() for v in parameters]
                with torch.no_grad():
                    for current, avg in zip(parameters, ema):
                        current.copy_(avg)
                auc = float(roc_auc_score(y[i_va], predict(Iv, CNv, CMv, DNv, DMv)))
                if auc > best_auc:
                    best_auc, best_epoch, bad = auc, epoch, 0
                    best_weights = [v.detach().clone() for v in ema]
                else:
                    bad += 1
                log(f"    lookup fold{fold} seed={model_seed} ep{epoch} "
                    f"val_auc={auc:.6f} best={best_auc:.6f}")
                with torch.no_grad():
                    for current, saved in zip(parameters, current_weights):
                        current.copy_(saved)
                if bad >= int(p["patience"]):
                    break

        with torch.no_grad():
            for current, best in zip(parameters, best_weights):
                current.copy_(best)
        val_pred = predict(Iv, CNv, CMv, DNv, DMv)
        test_pred = predict(It, CNt, CMt, DNt, DMt)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        return val_pred, test_pred, best_epoch + 1

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    skf = StratifiedKFold(
        folds, shuffle=bool(cv["shuffle"]), random_state=int(cv["seed"]),
    )
    fold_aucs, t0 = [], time.time()
    for fold, (i_tr, i_va) in enumerate(skf.split(np.zeros(len(train)), y)):
        p_va = np.zeros(len(i_va), dtype=np.float64)
        p_te = np.zeros(len(test), dtype=np.float64)
        best_epochs = []
        for model_seed in model_seeds:
            v, t, best_epoch = fit_fold(i_tr, i_va, fold, model_seed)
            p_va += v / len(model_seeds)
            p_te += t / len(model_seeds)
            best_epochs.append(best_epoch)
        oof[i_va] = p_va
        test_pred += p_te / folds
        auc = float(roc_auc_score(y[i_va], p_va))
        fold_aucs.append(auc)
        log(f"  fold {fold}: auc={auc:.6f} best_it={best_epochs} n_feat={n_tokens} "
            f"seeds={len(model_seeds)} [{time.time() - t0:.0f}s]")

    oof_auc = float(roc_auc_score(y, oof))
    log(f"  {spec['id']} OOF AUC = {oof_auc:.6f} "
        f"(folds: {[round(a, 6) for a in fold_aucs]})")
    if frac < 1.0:
        elapsed = time.time() - t0
        log(f"  SMOKE {spec['id']}: {elapsed:.0f}s at frac={frac}; no artifacts written")
        return {"id": spec["id"], "oof_auc": oof_auc, "fold_aucs": fold_aucs,
                "n_features": n_tokens, "smoke_frac": frac,
                "elapsed_s": round(elapsed, 1)}

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{spec['id']}.npy", oof)
    np.save(out_dir / f"testpred_{spec['id']}.npy", test_pred)
    write_submission(test_ids, test_pred, out_dir / f"submission_{spec['id']}.csv")
    return {"id": spec["id"], "oof_auc": oof_auc, "fold_aucs": fold_aucs,
            "n_features": n_tokens, "seeds": model_seeds}


def run_realmlp_version(spec: dict, cv: dict, train: pd.DataFrame, y: np.ndarray,
                        test: pd.DataFrame, test_ids: np.ndarray, out_dir: Path,
                        frac: float) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    p = dict(
        n_ens=10, embed_dim=8, onehot_thresh=8,
        hidden_dims=[768, 512, 512], dropout=0.07,
        p_drop_sched="expm4t", add_front_scale=True,
        pbld_hidden_dim=32, pbld_out_dim=6, pbld_freq_scale=10.0,
        pbld_lr_factor=0.093,
        lr=0.008, mom=0.9, sq_mom=0.98,
        lr_sched="flat_cos", flat_ratio=0.3,
        first_layer_lr_factor=1.0, first_layer_wd_factor=0.1,
        lr_scale_mult=10.0, lr_bias_mult=0.1,
        weight_decay=0.015, wd_scale_mult=0.1, wd_bias_mult=0.5,
        ema_decay=0.997875, grad_clip=1.2,
        ls_eps=0.04, ls_eps_sched="cos",
        tfms=["median_center", "robust_scale", "smooth_clip"],
        epochs=10, train_bs=256, eval_bs=10240,
        use_early_stopping=False,
        early_stopping_additive_patience=10,
        early_stopping_multiplicative_patience=1,
    )
    p.update(spec.get("params") or {})
    feats = set(spec.get("features") or [])
    model_seeds = [int(x) for x in (spec.get("seeds") or [int(cv["seed"])])]

    blocks_tr, blocks_te = [train[NUM].astype(np.float64)], [test[NUM].astype(np.float64)]
    cat_cols: list[str] = []
    if "raw" in feats:
        for c in CAT:
            blocks_tr.append(train[[c]])
            blocks_te.append(test[[c]])
        cat_cols = list(CAT)
    if "comp" in feats:
        blocks_tr.append(add_composition(train))
        blocks_te.append(add_composition(test))
    if "lattice" in feats:
        blocks_tr.append(add_lattice(train))
        blocks_te.append(add_lattice(test))
    if "identity" in feats:
        blocks_tr.append(add_identity(train))
        blocks_te.append(add_identity(test))

    train_lv, test_lv = _levels(train, FEATURE_COLS), _levels(test, FEATURE_COLS)
    if "freq" in feats:
        f_tr, f_te = add_frequency(train_lv, test_lv)
        blocks_tr.append(f_tr)
        blocks_te.append(f_te)

    X = pd.concat(blocks_tr, axis=1)
    Xtest = pd.concat(blocks_te, axis=1)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = None
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log(f"  realmlp device={device.type} n_ens={int(p['n_ens'])} "
        f"hidden={list(p['hidden_dims'])}")

    class CategoricalFeatureLayer(nn.Module):
        def __init__(self, n_ens: int, cat_dims: list[int]):
            super().__init__()
            self.n_ens = n_ens
            self.cat_dims = cat_dims
            threshold = int(p["onehot_thresh"])
            self.onehot_features = [i for i, dim in enumerate(cat_dims) if dim <= threshold]
            self.embed_feature_indices = [i for i, dim in enumerate(cat_dims) if dim > threshold]
            self.embed_layers = nn.ModuleList([
                nn.ModuleList([
                    nn.Embedding(cat_dims[i], int(p["embed_dim"])) for _ in range(n_ens)
                ])
                for i in self.embed_feature_indices
            ])

        def forward(self, x):
            batch_size = x.shape[0]
            features = []
            if self.onehot_features:
                dims = [self.cat_dims[i] for i in self.onehot_features]
                encoded = torch.zeros(
                    batch_size, self.n_ens, sum(dims), device=x.device,
                )
                start = 0
                for j, dim in enumerate(dims):
                    pos = x[:, :, self.onehot_features[j]:self.onehot_features[j] + 1].long()
                    encoded.scatter_(2, pos + start, 1.0)
                    start += dim
                features.append(encoded)
            for emb_list, feat_idx in zip(self.embed_layers, self.embed_feature_indices):
                per_head = [
                    emb_list[k](x[:, k, feat_idx:feat_idx + 1].long())
                    for k in range(self.n_ens)
                ]
                features.append(torch.cat(per_head, dim=1))
            if not features:
                return x.new_zeros((batch_size, self.n_ens, 0), dtype=torch.float32)
            return torch.cat(features, dim=2)

    class ScalingLayer(nn.Module):
        def __init__(self, n_ens: int, n_features: int):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(n_ens, n_features))

        def forward(self, x):
            return x * self.scale.unsqueeze(0)

    class NTPLinear(nn.Module):
        def __init__(self, n_ens: int, in_features: int, out_features: int):
            super().__init__()
            self.in_features = in_features
            self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
            self.bias = nn.Parameter(torch.randn(n_ens, out_features))

        def forward(self, x):
            return (torch.einsum("bki,kio->bko", x, self.weight)
                    / math.sqrt(self.in_features) + self.bias)

    class PBLDEmbedding(nn.Module):
        def __init__(self, n_ens: int, n_features: int):
            super().__init__()
            hidden = int(p["pbld_hidden_dim"])
            out_dim = int(p["pbld_out_dim"])
            self.w1 = nn.Parameter(
                torch.randn(n_ens, n_features, hidden) * float(p["pbld_freq_scale"])
            )
            self.b1 = nn.Parameter(torch.empty(n_ens, n_features, hidden))
            self.w2 = nn.Parameter(
                torch.randn(n_ens, n_features, hidden, out_dim - 1) / math.sqrt(hidden)
            )
            self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
            self.act = nn.PReLU()
            nn.init.uniform_(self.b1, -math.pi, math.pi)

        def forward(self, x):
            periodic = torch.cos(
                2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0)
                               + self.b1.unsqueeze(0))
            )
            transformed = self.act(
                torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2)
                + self.b2.unsqueeze(0)
            )
            return torch.cat([x.unsqueeze(-1), transformed], dim=-1).flatten(start_dim=2)

    class RealMLP(nn.Module):
        def __init__(self, n_numerical: int, cat_dims: list[int]):
            super().__init__()
            n_ens = int(p["n_ens"])
            self.n_ens = n_ens
            self.num_embed = PBLDEmbedding(n_ens, n_numerical)
            self.cate = CategoricalFeatureLayer(n_ens, cat_dims)
            cat_dim = sum(
                dim if dim <= int(p["onehot_thresh"]) else int(p["embed_dim"])
                for dim in cat_dims
            )
            in_dim = n_numerical * int(p["pbld_out_dim"]) + cat_dim
            layers = []
            if bool(p["add_front_scale"]):
                layers.append(ScalingLayer(n_ens, in_dim))
            self.dropout_modules = []
            for out_dim in [int(x) for x in p["hidden_dims"]]:
                linear = NTPLinear(n_ens, in_dim, out_dim)
                drop = nn.Dropout(float(p["dropout"]))
                self.dropout_modules.append(drop)
                layers.extend([linear, nn.SiLU(), drop])
                in_dim = out_dim
            self.hidden = nn.Sequential(*layers)
            self.output_layer = NTPLinear(n_ens, in_dim, 2)

        def forward(self, x_num, x_cat):
            x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
            x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
            x = torch.cat([self.num_embed(x_num), self.cate(x_cat)], dim=2)
            return self.output_layer(self.hidden(x))

    def apply_schedule(value: float, progress: float, schedule: str) -> float:
        if schedule == "constant":
            return value
        if schedule == "cos":
            return value * (math.cos(math.pi * progress) + 1) / 2
        if schedule == "flat_cos":
            if progress < float(p["flat_ratio"]):
                return value
            tail = (progress - float(p["flat_ratio"])) / (1 - float(p["flat_ratio"]))
            return value * (math.cos(math.pi * tail) + 1) / 2
        if schedule == "expm4t":
            return value * math.exp(-4 * progress)
        raise ValueError(f"unknown schedule: {schedule}")

    def fit_fold(Xtr: pd.DataFrame, Xva: pd.DataFrame, Xte: pd.DataFrame,
                 ytr: np.ndarray, yva: np.ndarray, fold: int,
                 model_seed: int) -> tuple[np.ndarray, np.ndarray, int]:
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(model_seed)

        num_cols = [c for c in Xtr.columns if c not in cat_cols]
        clean_tr = Xtr[num_cols].replace([np.inf, -np.inf], np.nan)
        clean_va = Xva[num_cols].replace([np.inf, -np.inf], np.nan)
        clean_te = Xte[num_cols].replace([np.inf, -np.inf], np.nan)
        # Both imputation and scaling are fitted inside the outer fold. A global median
        # would let validation-row distribution shifts influence the training inputs.
        fill = clean_tr.median().fillna(0.0)
        Ntr = clean_tr.fillna(fill).to_numpy(dtype=np.float32)
        Nva = clean_va.fillna(fill).to_numpy(dtype=np.float32)
        Nte = clean_te.fillna(fill).to_numpy(dtype=np.float32)
        tfms = set(p["tfms"])
        center = np.median(Ntr, axis=0)
        qdiff = np.quantile(Ntr, 0.75, axis=0) - np.quantile(Ntr, 0.25, axis=0)
        ranges = Ntr.max(axis=0) - Ntr.min(axis=0)
        qdiff = np.where(qdiff == 0, 0.5 * ranges, qdiff)
        scale = np.where(qdiff == 0, 0.0, 1.0 / (qdiff + 1e-30)).astype(np.float32)

        def transform(values: np.ndarray) -> np.ndarray:
            out = values.copy()
            if "median_center" in tfms:
                out -= center[None, :]
            if "robust_scale" in tfms:
                out *= scale[None, :]
            if "smooth_clip" in tfms:
                out = out / np.sqrt(1 + (out / 3) ** 2)
            return out.astype(np.float32)

        Ntr, Nva, Nte = transform(Ntr), transform(Nva), transform(Nte)

        def categorical_values(df: pd.DataFrame, c: str) -> pd.Series:
            return df[c].astype(object).where(df[c].notna(), MISSING_LEVEL).astype(str)

        Ctr = np.empty((len(Xtr), len(cat_cols)), dtype=np.int64)
        Cva = np.empty((len(Xva), len(cat_cols)), dtype=np.int64)
        Cte = np.empty((len(Xte), len(cat_cols)), dtype=np.int64)
        cat_dims = []
        for j, c in enumerate(cat_cols):
            levels = sorted(categorical_values(Xtr, c).unique())
            mapping = {level: i + 1 for i, level in enumerate(levels)}
            # Index zero is fold-local unknown. Validation/test categories do not get to
            # enlarge a training representation merely because they occur downstream.
            Ctr[:, j] = categorical_values(Xtr, c).map(mapping).fillna(0).to_numpy(np.int64)
            Cva[:, j] = categorical_values(Xva, c).map(mapping).fillna(0).to_numpy(np.int64)
            Cte[:, j] = categorical_values(Xte, c).map(mapping).fillna(0).to_numpy(np.int64)
            cat_dims.append(len(levels) + 1)

        model = RealMLP(Ntr.shape[1], cat_dims).to(device)
        first_weight = next(
            module.weight for module in model.hidden if isinstance(module, NTPLinear)
        )
        grouped = {"scale": [], "pbld": [], "first": [], "other": [], "bias": []}
        for name, param in model.named_parameters():
            if "num_embed" in name:
                grouped["pbld"].append(param)
            elif "scale" in name:
                grouped["scale"].append(param)
            elif param is first_weight:
                grouped["first"].append(param)
            elif "bias" in name:
                grouped["bias"].append(param)
            else:
                grouped["other"].append(param)
        lr, wd = float(p["lr"]), float(p["weight_decay"])
        group_specs = [
            ("scale", lr * float(p["lr_scale_mult"]), wd * float(p["wd_scale_mult"])),
            ("pbld", lr * float(p["pbld_lr_factor"]), wd),
            ("first", lr * float(p["first_layer_lr_factor"]),
             wd * float(p["first_layer_wd_factor"])),
            ("other", lr, wd),
            ("bias", lr * float(p["lr_bias_mult"]), wd * float(p["wd_bias_mult"])),
        ]
        param_groups = [
            {"params": grouped[name], "lr": group_lr, "lr_base": group_lr,
             "weight_decay": group_wd}
            for name, group_lr, group_wd in group_specs if grouped[name]
        ]
        optimizer = torch.optim.AdamW(
            param_groups, betas=(float(p["mom"]), float(p["sq_mom"])),
        )

        tNtr = torch.from_numpy(Ntr).to(device)
        tNva = torch.from_numpy(Nva).to(device)
        tNte = torch.from_numpy(Nte).to(device)
        tCtr = torch.from_numpy(Ctr).to(device)
        tCva = torch.from_numpy(Cva).to(device)
        tCte = torch.from_numpy(Cte).to(device)
        tytr = torch.from_numpy(ytr.astype(np.int64)).to(device)
        counts = np.bincount(ytr, minlength=2).astype(np.float32)
        class_weights = torch.from_numpy(len(ytr) / (2 * counts)).to(device)
        scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
        generator = torch.Generator().manual_seed(model_seed)
        parameters = list(model.parameters())
        ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
        ema_step = 0

        def forward(num, cat):
            if amp_dtype is None:
                return model(num, cat)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                return model(num, cat)

        def smooth_loss(logits, labels, smoothing: float):
            logits = logits.reshape(-1, 2).float()
            labels = labels.repeat_interleave(int(p["n_ens"]))
            targets = torch.full_like(logits, smoothing / 2)
            targets.scatter_(1, labels.unsqueeze(1), 1 - smoothing + smoothing / 2)
            per_row = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
            weights = class_weights[labels]
            return (per_row * weights).sum() / weights.sum()

        def predict(num, cat) -> np.ndarray:
            model.eval()
            preds = []
            with torch.no_grad():
                for start in range(0, len(num), int(p["eval_bs"])):
                    probs = torch.softmax(
                        forward(num[start:start + int(p["eval_bs"])],
                                cat[start:start + int(p["eval_bs"])]).float(), dim=2,
                    ).mean(dim=1)[:, 1]
                    preds.append(probs.cpu().numpy())
            return np.concatenate(preds).astype(np.float64)

        epochs, batch_size = int(p["epochs"]), int(p["train_bs"])
        steps_per_epoch = math.ceil(len(ytr) / batch_size)
        total_steps = max(1, epochs * steps_per_epoch)
        best_auc, best_epoch, best_state = -1.0, -1, None
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(ytr), generator=generator).to(device)
            for batch, start in enumerate(range(0, len(ytr), batch_size)):
                step = epoch * steps_per_epoch + batch
                progress = step / total_steps
                for group in optimizer.param_groups:
                    group["lr"] = apply_schedule(
                        float(group["lr_base"]), progress, str(p["lr_sched"]),
                    )
                drop = apply_schedule(
                    float(p["dropout"]), progress, str(p["p_drop_sched"]),
                )
                for module in model.dropout_modules:
                    module.p = drop
                idx = order[start:start + batch_size]
                smoothing = apply_schedule(
                    float(p["ls_eps"]), progress, str(p["ls_eps_sched"]),
                )
                loss = smooth_loss(forward(tNtr[idx], tCtr[idx]), tytr[idx], smoothing)
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, float(p["grad_clip"]))
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, float(p["grad_clip"]))
                    scaler.step(optimizer)
                    scaler.update()

                ema_step += 1
                decay = min(
                    float(p["ema_decay"]), (1.0 + ema_step) / (10.0 + ema_step),
                )
                with torch.no_grad():
                    for name, value in model.state_dict().items():
                        if torch.is_floating_point(value):
                            ema[name].mul_(decay).add_(value.detach(), alpha=1 - decay)
                        else:
                            ema[name].copy_(value)

            live_state = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}
            model.load_state_dict(ema, strict=True)
            val_pred = predict(tNva, tCva)
            auc = float(roc_auc_score(yva, val_pred))
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                best_state = {name: value.detach().clone() for name, value in ema.items()}
            log(f"    realmlp fold{fold} seed={model_seed} ep{epoch} "
                f"val_auc={auc:.6f} best={best_auc:.6f} "
                f"ls={smoothing:.4f} drop={drop:.4f}")
            model.load_state_dict(live_state, strict=True)
            if bool(p["use_early_stopping"]):
                patience = (best_epoch + 1) * float(p["early_stopping_multiplicative_patience"])
                patience += int(p["early_stopping_additive_patience"])
                if epoch + 1 > patience:
                    break

        model.load_state_dict(best_state, strict=True)
        val_pred = predict(tNva, tCva)
        test_pred = predict(tNte, tCte)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        return val_pred, test_pred, best_epoch + 1

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    folds = int(spec.get("folds") or cv["folds"])
    skf = StratifiedKFold(
        folds, shuffle=bool(cv["shuffle"]), random_state=int(cv["seed"]),
    )
    fold_aucs, t0, n_features = [], time.time(), int(X.shape[1])
    for fold, (i_tr, i_va) in enumerate(skf.split(X, y)):
        Xtr, Xva, Xte_f = X.iloc[i_tr], X.iloc[i_va], Xtest
        if "te" in feats:
            tr_enc, va_enc, te_enc = target_encode(
                train_lv, y, test_lv, i_tr, i_va, int(cv["seed"]),
            )
            Xtr = pd.concat([Xtr, tr_enc], axis=1)
            Xva = pd.concat([Xva, va_enc], axis=1)
            Xte_f = pd.concat([Xtest, te_enc], axis=1)
        n_features = int(Xtr.shape[1])
        p_va = np.zeros(len(i_va), dtype=np.float64)
        p_te = np.zeros(len(test), dtype=np.float64)
        best_epochs = []
        for model_seed in model_seeds:
            val, tst, best_epoch = fit_fold(
                Xtr, Xva, Xte_f, y[i_tr], y[i_va], fold, model_seed,
            )
            p_va += val / len(model_seeds)
            p_te += tst / len(model_seeds)
            best_epochs.append(best_epoch)
        oof[i_va] = p_va
        test_pred += p_te / folds
        auc = float(roc_auc_score(y[i_va], p_va))
        fold_aucs.append(auc)
        log(f"  fold {fold}: auc={auc:.6f} best_it={best_epochs} n_feat={n_features} "
            f"seeds={len(model_seeds)} [{time.time() - t0:.0f}s]")

    oof_auc = float(roc_auc_score(y, oof))
    log(f"  {spec['id']} OOF AUC = {oof_auc:.6f} "
        f"(folds: {[round(a, 6) for a in fold_aucs]})")
    if frac < 1.0:
        elapsed = time.time() - t0
        log(f"  SMOKE {spec['id']}: {elapsed:.0f}s at frac={frac}; no artifacts written")
        return {"id": spec["id"], "oof_auc": oof_auc, "fold_aucs": fold_aucs,
                "n_features": n_features, "smoke_frac": frac,
                "elapsed_s": round(elapsed, 1)}

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{spec['id']}.npy", oof)
    np.save(out_dir / f"testpred_{spec['id']}.npy", test_pred)
    write_submission(test_ids, test_pred, out_dir / f"submission_{spec['id']}.csv")
    return {"id": spec["id"], "oof_auc": oof_auc, "fold_aucs": fold_aucs,
            "n_features": n_features, "seeds": model_seeds}


def run_version(spec: dict, cv: dict, data_dir: Path, out_dir: Path, frac: float = 1.0) -> dict:
    vid, model = spec["id"], spec["model"]
    feats = set(spec.get("features") or [])
    seed = int(cv["seed"])
    # Model seeds are separate from the CV seed: the split must stay identical across
    # every version so OOF arrays remain row-aligned and stackable, while a version may
    # average several model seeds inside each fold to cut variance.
    model_seeds = [int(x) for x in (spec.get("seeds") or [seed])]
    overrides = dict(spec.get("params") or {})
    folds = int(spec.get("folds") or cv["folds"])

    log(f"=== {vid} (model={model}, features={sorted(feats)}) ===")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    if frac < 1.0:
        # Smoke-test mode. A run at frac<1 writes no artifacts and its OOF is not
        # comparable to a full run — it exists to answer "does this finish, and how long
        # would the full one take", which is the arithmetic the agent's timed-out job
        # skipped.
        train = train.sample(frac=frac, random_state=0).reset_index(drop=True)
        test = test.sample(frac=frac, random_state=0).reset_index(drop=True)
        log(f"SMOKE frac={frac}: train={train.shape} test={test.shape}")
    y = train[TARGET].to_numpy(dtype=np.int8)
    test_ids = test[ID].to_numpy()
    log(f"train={train.shape} test={test.shape} positive_rate={y.mean():.5f}")

    if model == "lookup":
        return run_lookup_version(spec, cv, train, y, test, test_ids, out_dir, frac)
    if model == "realmlp":
        return run_realmlp_version(spec, cv, train, y, test, test_ids, out_dir, frac)
    if model == "stack":
        return run_stack(spec, cv, train, y, test_ids, out_dir)

    # Static feature blocks (identical for every fold).
    blocks_tr, blocks_te = [train[NUM].astype(np.float64)], [test[NUM].astype(np.float64)]
    cat_cols: list[str] = []
    if "raw" in feats:
        for c in CAT:
            blocks_tr.append(train[[c]].astype("category"))
            blocks_te.append(test[[c]].astype("category"))
        cat_cols = list(CAT)
    if "comp" in feats:
        blocks_tr.append(add_composition(train))
        blocks_te.append(add_composition(test))
    if "lattice" in feats:
        blocks_tr.append(add_lattice(train))
        blocks_te.append(add_lattice(test))
    if "identity" in feats:
        blocks_tr.append(add_identity(train))
        blocks_te.append(add_identity(test))

    train_lv, test_lv = _levels(train, FEATURE_COLS), _levels(test, FEATURE_COLS)
    if "freq" in feats:
        f_tr, f_te = add_frequency(train_lv, test_lv)
        blocks_tr.append(f_tr)
        blocks_te.append(f_te)

    X = pd.concat(blocks_tr, axis=1)
    Xtest = pd.concat(blocks_te, axis=1)
    # Categoricals must share one category set or the learners see different codings.
    for c in cat_cols:
        cats = pd.api.types.union_categoricals([X[c], Xtest[c]]).categories
        X[c] = X[c].cat.set_categories(cats)
        Xtest[c] = Xtest[c].cat.set_categories(cats)

    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(Xtest), dtype=np.float64)
    skf = StratifiedKFold(folds, shuffle=bool(cv["shuffle"]), random_state=seed)
    fold_aucs, t0 = [], time.time()

    for k, (i_tr, i_va) in enumerate(skf.split(X, y)):
        Xtr, Xva, Xte_f = X.iloc[i_tr], X.iloc[i_va], Xtest
        if "te" in feats:
            tr_enc, va_enc, te_enc = target_encode(train_lv, y, test_lv, i_tr, i_va, seed)
            Xtr = pd.concat([Xtr, tr_enc], axis=1)
            Xva = pd.concat([Xva, va_enc], axis=1)
            Xte_f = pd.concat([Xtest, te_enc], axis=1)

        p_va = np.zeros(len(Xva), dtype=np.float64)
        p_te = np.zeros(len(Xte_f), dtype=np.float64)
        iters = []
        for ms in model_seeds:
            v, t, it = fit_predict(model, Xtr, y[i_tr], Xva, y[i_va], Xte_f, cat_cols, ms, overrides)
            p_va += v / len(model_seeds)
            p_te += t / len(model_seeds)
            iters.append(it)
        oof[i_va] = p_va
        test_pred += p_te / folds
        auc = roc_auc_score(y[i_va], p_va)
        fold_aucs.append(auc)
        log(f"  fold {k}: auc={auc:.6f} best_it={iters} n_feat={Xtr.shape[1]} "
            f"seeds={len(model_seeds)} [{time.time() - t0:.0f}s]")

    oof_auc = float(roc_auc_score(y, oof))
    log(f"  {vid} OOF AUC = {oof_auc:.6f}  (folds: {[round(a, 6) for a in fold_aucs]})")

    if frac < 1.0:
        log(f"  SMOKE {vid}: {time.time() - t0:.0f}s at frac={frac}; no artifacts written")
        return {"id": vid, "oof_auc": oof_auc, "fold_aucs": fold_aucs,
                "n_features": int(X.shape[1]), "smoke_frac": frac,
                "elapsed_s": round(time.time() - t0, 1)}

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{vid}.npy", oof)
    np.save(out_dir / f"testpred_{vid}.npy", test_pred)
    write_submission(test_ids, test_pred, out_dir / f"submission_{vid}.csv")
    return {"id": vid, "oof_auc": oof_auc, "fold_aucs": fold_aucs, "n_features": int(X.shape[1]),
            "seeds": model_seeds}


def hillclimb_weights(Z: np.ndarray, y: np.ndarray, step: float = 0.02,
                      rounds: int = 300) -> np.ndarray:
    """Greedy forward weight search on AUC in logit space.

    Starts from the best single member and repeatedly adds `step` of whichever member
    improves AUC most, stopping when none does. Unlike a logistic meta-learner this
    optimises the competition metric directly and cannot assign a negative weight, which
    is the failure mode when members correlate above 0.99 and the meta-learner starts
    differencing two near-identical columns.
    """
    n = Z.shape[1]
    aucs = [roc_auc_score(y, Z[:, j]) for j in range(n)]
    w = np.zeros(n, dtype=np.float64)
    w[int(np.argmax(aucs))] = 1.0
    blend = Z @ w
    best = roc_auc_score(y, blend)
    for _ in range(rounds):
        gains = []
        for j in range(n):
            cand = (blend * w.sum() + step * Z[:, j]) / (w.sum() + step)
            gains.append(roc_auc_score(y, cand))
        j = int(np.argmax(gains))
        if gains[j] <= best + 1e-9:
            break
        w[j] += step
        blend = Z @ w / w.sum()
        best = gains[j]
    return w / w.sum()


def run_stack(spec: dict, cv: dict, train: pd.DataFrame, y: np.ndarray,
              test_ids: np.ndarray, out_dir: Path) -> dict:
    """Logit-space stack with a LogisticRegression meta-learner.

    Fitted honestly: the meta-model that scores a row is trained on the other four
    folds, so no row is scored by a combiner that saw its own OOF value. Fitting one
    meta-model on all OOF rows and reporting its in-sample AUC is the usual way stacks
    look better than they are.
    """
    members = spec["members"]
    log(f"stacking {len(members)} members: {members}")

    def logit(p):
        p = np.clip(p.astype(np.float64), 1e-15, 1 - 1e-15)
        return np.clip(np.log(p / (1 - p)), -30, 30)

    Z = np.column_stack([logit(np.load(out_dir / f"oof_{m}.npy")) for m in members])
    Zt = np.column_stack([logit(np.load(out_dir / f"testpred_{m}.npy")) for m in members])

    for m, col in zip(members, Z.T):
        log(f"  member {m}: solo OOF AUC={roc_auc_score(y, col):.6f}")
    if len(members) > 1:
        log(f"  member correlation matrix (Spearman):\n{pd.DataFrame(Z, columns=members).corr(method='spearman').round(4)}")

    meta = spec.get("meta", "logreg")
    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(int(cv["folds"]), shuffle=bool(cv["shuffle"]), random_state=int(cv["seed"]))
    for i_tr, i_va in skf.split(Z, y):
        if meta == "hillclimb":
            w = hillclimb_weights(Z[i_tr], y[i_tr])
            oof[i_va] = Z[i_va] @ w
        else:
            lr = LogisticRegression(C=1.0, max_iter=5000)
            lr.fit(Z[i_tr], y[i_tr])
            if lr.n_iter_[0] >= 5000:
                raise RuntimeError("meta-learner did not converge; a non-converged stack reads higher than truth")
            oof[i_va] = lr.predict_proba(Z[i_va])[:, 1]

    oof_auc = float(roc_auc_score(y, oof))
    if meta == "hillclimb":
        w = hillclimb_weights(Z, y)
        test_pred = (Zt @ w).astype(np.float64)
        weights = dict(zip(members, w.round(6).tolist()))
    else:
        final = LogisticRegression(C=1.0, max_iter=5000).fit(Z, y)
        test_pred = final.predict_proba(Zt)[:, 1].astype(np.float64)
        weights = dict(zip(members, final.coef_[0].round(6).tolist()))
    log(f"  stack({meta}) OOF AUC = {oof_auc:.6f}  weights={weights}")

    np.save(out_dir / f"oof_{spec['id']}.npy", oof)
    np.save(out_dir / f"testpred_{spec['id']}.npy", test_pred)
    write_submission(test_ids, test_pred, out_dir / f"submission_{spec['id']}.csv")
    return {"id": spec["id"], "oof_auc": oof_auc, "members": members,
            "meta": meta, "weights": weights}


def write_submission(ids: np.ndarray, pred: np.ndarray, path: Path) -> None:
    if len(ids) != len(pred):
        raise ValueError(f"id/pred length mismatch: {len(ids)} vs {len(pred)}")
    pd.DataFrame({ID: ids, TARGET: pred}).to_csv(path, index=False)
    log(f"  wrote {path} ({len(pred)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="version id from versions.yaml, or 'all'")
    ap.add_argument("--data-dir", default=os.environ.get("S6E8_DATA_DIR", "/tmp/s6e8"))
    ap.add_argument("--out-dir", default=os.environ.get("S6E8_OUT_DIR", "/tmp/s6e8work"))
    ap.add_argument("--versions-file", default=str(Path(__file__).resolve().parent.parent / "versions.yaml"))
    ap.add_argument("--frac", type=float, default=1.0,
                    help="smoke-test on a row fraction; writes no artifacts")
    ap.add_argument("--members", default=None,
                    help="comma-separated member ids, overriding a stack version's list. "
                         "Lets a blend be re-scored over whatever members exist yet.")
    ap.add_argument("--meta", default=None, choices=["logreg", "hillclimb"],
                    help="override a stack version's meta-learner")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.versions_file).read_text())
    by_id = {v["id"]: v for v in cfg["versions"]}
    todo = list(cfg["versions"]) if args.version == "all" else [by_id[args.version]]

    if args.members or args.meta:
        # An overridden run is not the version the ledger declares, so it gets its own id.
        # Writing it under the declared id would file a 4-member result against an 8-member
        # row, which is exactly the drift versions.yaml exists to prevent.
        for spec in todo:
            if spec.get("model") != "stack":
                continue
            if args.members:
                spec["members"] = [m.strip() for m in args.members.split(",") if m.strip()]
            if args.meta:
                spec["meta"] = args.meta
            spec["id"] = f"{spec['id']}__adhoc"

    results = []
    for spec in todo:
        results.append(run_version(spec, cfg["cv"], Path(args.data_dir), Path(args.out_dir), args.frac))

    if args.frac < 1.0:
        for r in results:
            log(f"  SMOKE {r['id']}: OOF {r['oof_auc']:.6f} in {r['elapsed_s']}s at frac={args.frac} "
                f"-> a full run is ~{r['elapsed_s'] / args.frac / 60:.0f} min if cost were linear "
                f"(it is superlinear in rows for tree builds, so treat that as a floor)")
        return 0

    out = Path(args.out_dir) / "results.json"
    prior = json.loads(out.read_text()) if out.exists() else {}
    prior.update({r["id"]: r for r in results})
    out.write_text(json.dumps(prior, indent=2))
    log(f"results -> {out}")
    for r in results:
        log(f"  {r['id']}: OOF {r['oof_auc']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
