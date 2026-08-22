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


def add_lattice(df: pd.DataFrame) -> pd.DataFrame:
    """First fractional digit of each continuous column — exposes generator rounding."""
    out = pd.DataFrame(index=df.index, dtype=np.float64)
    for c in ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
              "work_study_hours", "sleep_hours", "weekend_screen_time"]:
        out[f"dec_{c}"] = np.round((df[c] * 10) % 10).where(df[c].notna())
    return out


# ---------------------------------------------------------------- models


def fit_predict(model: str, Xtr, ytr, Xva, yva, Xte, cat_cols: list[str], seed: int):
    """Train one fold. Returns (val_pred, test_pred, best_iteration)."""
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
        m = lgb.train(
            params, lgb.Dataset(Xtr, ytr), num_boost_round=6000,
            valid_sets=[lgb.Dataset(Xva, yva)],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        it = m.best_iteration
        return (m.predict(Xva, num_iteration=it).astype(np.float64),
                m.predict(Xte, num_iteration=it).astype(np.float64), it)

    if model == "xgb":
        import xgboost as xgb

        m = xgb.XGBClassifier(
            n_estimators=6000, learning_rate=0.02, max_depth=7,
            min_child_weight=15, subsample=0.85, colsample_bytree=0.5,
            reg_alpha=0.05, reg_lambda=3.0,
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            early_stopping_rounds=200, random_state=seed, n_jobs=-1,
            enable_categorical=True, max_cat_to_onehot=1,
        )
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
        m = CatBoostClassifier(
            iterations=4000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            eval_metric="AUC", early_stopping_rounds=150,
            random_seed=seed, verbose=0, thread_count=-1,
            border_count=128,  # CatBoost's max_bin; 254 default is slower for little gain here
            # Without this CatBoost writes catboost_info/ into the current directory,
            # which means training litter lands in the repo when run from scripts/.
            train_dir=str(Path(tempfile.gettempdir()) / "catboost_info"),
            allow_writing_files=False,
        )
        m.fit(Pool(Xtr, ytr, cat_features=cat_cols),
              eval_set=Pool(Xva, yva, cat_features=cat_cols))
        it = m.get_best_iteration()
        return (m.predict_proba(Xva)[:, 1].astype(np.float64),
                m.predict_proba(Xte)[:, 1].astype(np.float64), it)

    raise ValueError(f"unknown model: {model}")


# ---------------------------------------------------------------- run one version


def run_version(spec: dict, cv: dict, data_dir: Path, out_dir: Path) -> dict:
    vid, model = spec["id"], spec["model"]
    feats = set(spec.get("features") or [])
    seed = int(cv["seed"])

    log(f"=== {vid} (model={model}, features={sorted(feats)}) ===")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train[TARGET].to_numpy(dtype=np.int8)
    test_ids = test[ID].to_numpy()
    log(f"train={train.shape} test={test.shape} positive_rate={y.mean():.5f}")

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
    skf = StratifiedKFold(int(cv["folds"]), shuffle=bool(cv["shuffle"]), random_state=seed)
    fold_aucs, t0 = [], time.time()

    for k, (i_tr, i_va) in enumerate(skf.split(X, y)):
        Xtr, Xva, Xte_f = X.iloc[i_tr], X.iloc[i_va], Xtest
        if "te" in feats:
            tr_enc, va_enc, te_enc = target_encode(train_lv, y, test_lv, i_tr, i_va, seed)
            Xtr = pd.concat([Xtr, tr_enc], axis=1)
            Xva = pd.concat([Xva, va_enc], axis=1)
            Xte_f = pd.concat([Xtest, te_enc], axis=1)

        p_va, p_te, it = fit_predict(model, Xtr, y[i_tr], Xva, y[i_va], Xte_f, cat_cols, seed)
        oof[i_va] = p_va
        test_pred += p_te / int(cv["folds"])
        auc = roc_auc_score(y[i_va], p_va)
        fold_aucs.append(auc)
        log(f"  fold {k}: auc={auc:.6f} best_it={it} n_feat={Xtr.shape[1]} [{time.time() - t0:.0f}s]")

    oof_auc = float(roc_auc_score(y, oof))
    log(f"  {vid} OOF AUC = {oof_auc:.6f}  (folds: {[round(a, 6) for a in fold_aucs]})")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{vid}.npy", oof)
    np.save(out_dir / f"testpred_{vid}.npy", test_pred)
    write_submission(test_ids, test_pred, out_dir / f"submission_{vid}.csv")
    return {"id": vid, "oof_auc": oof_auc, "fold_aucs": fold_aucs, "n_features": int(X.shape[1])}


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

    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(int(cv["folds"]), shuffle=bool(cv["shuffle"]), random_state=int(cv["seed"]))
    for i_tr, i_va in skf.split(Z, y):
        lr = LogisticRegression(C=1.0, max_iter=5000)
        lr.fit(Z[i_tr], y[i_tr])
        if lr.n_iter_[0] >= 5000:
            raise RuntimeError("meta-learner did not converge; a non-converged stack reads higher than truth")
        oof[i_va] = lr.predict_proba(Z[i_va])[:, 1]

    oof_auc = float(roc_auc_score(y, oof))
    final = LogisticRegression(C=1.0, max_iter=5000).fit(Z, y)
    test_pred = final.predict_proba(Zt)[:, 1].astype(np.float64)
    log(f"  stack OOF AUC = {oof_auc:.6f}  weights={dict(zip(members, final.coef_[0].round(4)))}")

    np.save(out_dir / f"oof_{spec['id']}.npy", oof)
    np.save(out_dir / f"testpred_{spec['id']}.npy", test_pred)
    write_submission(test_ids, test_pred, out_dir / f"submission_{spec['id']}.csv")
    return {"id": spec["id"], "oof_auc": oof_auc, "members": members,
            "weights": dict(zip(members, final.coef_[0].round(6).tolist()))}


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
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.versions_file).read_text())
    by_id = {v["id"]: v for v in cfg["versions"]}
    todo = list(cfg["versions"]) if args.version == "all" else [by_id[args.version]]

    results = []
    for spec in todo:
        results.append(run_version(spec, cfg["cv"], Path(args.data_dir), Path(args.out_dir)))

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
