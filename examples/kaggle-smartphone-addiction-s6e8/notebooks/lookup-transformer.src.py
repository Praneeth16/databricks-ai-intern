# %% [markdown]
# # Lookup-Transformer and the control that refuted me
#
# Playground S6E8 is saturated. 2,692 teams, and 203 of them sit inside a 0.0002 AUC band. The
# gap from 1st to 10th is 0.00012, smaller than the noise of the public split. In that regime
# tuning a gradient-boosted tree harder is not a strategy, because every GBDT on the leaderboard
# is making almost the same mistakes on almost the same rows. What moves the number is a model
# that is **wrong differently**.
#
# I had a theory about where "differently" comes from. I thought it came from input
# representation: that a model reading each feature value as a token identity would decorrelate
# from models reading it as a magnitude, and that this mattered more than which architecture you
# picked. I built a controlled experiment to demonstrate it.
#
# The experiment refuted the theory, and then handed me a better one. Both parts are below.
#
# ### The setup
#
# Five models, one frozen fold definition. Two trees, then three models sharing a single training
# function with the same optimizer, schedule, augmentation rate and EMA, differing only in what
# they are allowed to read.
#
# | member | reads | solo OOF AUC | Spearman vs trees | blend weight | added on top of the two trees |
# |---|---|---|---|---|---|
# | `xgboost` | target + frequency encoding, raw values, budget features | 0.968454 | 0.9990 with LightGBM | 0.3251 | |
# | `lightgbm` | same | 0.968331 | 0.9990 with XGBoost | 0.1627 | |
# | `mlp_magnitudes` | rank-gauss magnitudes and missing masks only | 0.941125 | 0.9416 | 0.0000 | +0.000008 |
# | `mlp_encoded` | the trees' exact feature matrix, rank-gauss scaled | 0.965032 | 0.9665 | 0.0159 | +0.000013 |
# | `lookup_transformer` | a learned vector per **exact value**, plus gated magnitudes | 0.968305 | 0.9658 | **0.4963** | **+0.000577** |
#
# ### Where the theory died
#
# My theory predicted `mlp_encoded` would sit near 0.99 against the trees. It is a neural network
# fed the trees' own features, so on my theory it should behave like the trees. It measured
# **0.9665**, indistinguishable from the Lookup-Transformer's **0.9658**. The two neural networks
# correlate **0.9782 with each other**, higher than either correlates with either tree.
#
# The decorrelation boundary in this pool runs between trees and neural networks. It does not run
# between magnitudes and identities. Changing the representation inside the neural family bought
# no measurable diversity at all.
#
# ### Where it gets interesting
#
# Look at the last two columns. `mlp_encoded` and `lookup_transformer` have effectively the same
# correlation with the trees, 0.9665 against 0.9658, and their contributions to the blend differ
# by more than **40-fold**. One is worth +0.000013 and the other +0.000577. Removing the
# Lookup-Transformer costs 0.000563 of blend AUC; removing either MLP costs nothing measurable,
# and removing `mlp_magnitudes` costs literally 0.000000.
#
# So correlation, on its own, predicts nothing. Rank these three by how decorrelated they are
# from the trees and you get `mlp_magnitudes`, `lookup_transformer`, `mlp_encoded`. Rank them by
# what they actually contribute and you get `lookup_transformer`, `mlp_encoded`,
# `mlp_magnitudes`. The orderings are scrambled.
#
# What separates them is **strength at equal diversity**. The whole neural family was already
# decorrelated from the trees. What it lacked was a member strong enough for that decorrelation
# to be worth anything, and `mlp_encoded` at 0.965032 is not that member, because it is 0.0033
# behind the trees it is supposed to be complementing.
#
# The Lookup-Transformer is that member, at 0.968305, level with a tuned GBDT. And the reason it
# gets there is the representation: same trainer, same schedule, same everything, learned
# per-value embeddings instead of hand-built target encoding, and **+0.0033 AUC** for it.
#
# So the corrected claim, which the data does support:
#
# > Per-value embeddings did not buy diversity. They bought **strength inside a family that was
# > already diverse**, and that is what made the diversity cash out.
#
# ### The trap, stated plainly
#
# My first version of this notebook had only `mlp_magnitudes` as the control. It correlates
# 0.9416 with the trees, which is *lower* than the Lookup-Transformer's 0.9658. Read carelessly
# that looks like a spectacular confirmation: the magnitude model is far from the trees, the
# identity model is far from the trees, so representation must be what matters.
#
# It is nothing of the kind. `mlp_magnitudes` is 0.027 AUC weaker than everything else in the
# pool, and **any sufficiently weak model decorrelates**, because it is largely producing noise.
# Correlation is only interpretable between models of comparable strength. Adding the
# strength-matched control is the single step that turned a confirmation into a refutation, and it
# cost one extra cell.
#
# ### A note on the last decimal
#
# The figures in this summary are from the run that produced this notebook's outputs. Rerunning
# moves them in the sixth decimal place, because LightGBM's threaded histogram construction and
# the GPU kernels underneath the two networks are not bit-deterministic. The fold definition,
# the feature construction and the seeds are all fixed, so what drifts is arithmetic order, not
# methodology. The hill-climb weights move a little more than the AUCs do, which is what you
# should expect from a greedy search over highly correlated members: they trade weight almost
# arbitrarily among themselves. That is exactly why section 10 measures contribution by ablation
# instead of reading it off the weights, and the ablation numbers reproduced to the sixth decimal
# across two full runs.
#
# ### What you get here
#
# 1. The evidence that this dataset's floats are secretly categorical, measured before modelling.
# 2. A Lookup-Transformer built from scratch, one component at a time, with the reason for each.
#    It scores level with a tuned GBDT, so it is a working model and not only a lesson.
# 3. Two controls sharing one trainer, and the arithmetic that killed my hypothesis.
# 4. Leave-one-out and incremental blend analysis, which is what exposed that correlation and
#    contribution disagree.
# 5. The EMA bug that made my first run score 0.45 AUC and look like a flipped label.
# 6. Negative results, including a 40-trial Optuna sweep that bought 0.000003.
# 7. Every member's out-of-fold and test predictions saved on the frozen folds, so you can stack
#    them into your own blend without refitting anything.
#
# ### Credit where it is due
#
# The Lookup-Transformer idea is not mine. I ported it from
# [tamerlanomralinov's notebook](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041)
# (LB 0.97041) and retrained it from scratch on my own folds. What I add is the controlled
# comparison, the EMA fix, and the blend analysis. If this notebook is useful, upvote that one
# too.

# %% [markdown]
# ## 1. Setup
#
# **Accelerator:** set this to **GPU T4 x2**. Kaggle's other GPU option is a P100, whose
# compute capability (sm_60) is below what the preinstalled PyTorch build ships kernels for
# (sm_70 and up). XGBoost is unaffected because it bundles its own CUDA build, so on a P100 the
# tree sections pass and the neural sections fail 25 minutes later. The device probe below
# catches that at setup instead.
#
# Everything downstream depends on one frozen CV definition. `StratifiedKFold(5, shuffle=True,
# random_state=42)` over the original row order, used by every model without exception. That is
# what makes the out-of-fold arrays row-aligned, which is what makes the correlation matrix and
# the blend legitimate. If you fork this and change the split for one model, the blend section
# stops meaning anything.

# %%
import glob
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")

# Resolve the competition mount rather than hardcoding it, so this notebook survives
# being forked with the data attached under a different name.
def find_data_dir():
    candidates = (sorted(glob.glob("/kaggle/input/*"))
                  + sorted(glob.glob("/kaggle/input/*/*")) + ["."])
    for d in candidates:
        if os.path.exists(os.path.join(d, "train.csv")):
            return d
    raise FileNotFoundError(f"no train.csv under any of: {candidates}")


DATA = find_data_dir()
print("using:", DATA)

TARGET = "addicted_label"
ID = "id"

NUM = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
CAT = ["gender", "stress_level", "academic_work_impact"]
FEATURES = NUM + CAT

N_FOLDS = 5
SEED = 42
MISSING = "__missing__"
TE_SMOOTHING = 10.0

def pick_device():
    """Probe CUDA with a real operation instead of trusting `is_available()`.

    Kaggle hands out either a T4 (sm_75) or a P100 (sm_60), and the preinstalled
    torch build ships kernels for sm_70 and up. On a P100 `torch.cuda.is_available()`
    still returns True and then every kernel launch raises, so the failure lands deep
    inside training rather than at setup. Probing here turns 25 wasted minutes into
    one clear line. If you fork this and get a P100, switch the accelerator to
    "GPU T4 x2" in the notebook settings.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu"), "no CUDA device"
    try:
        probe = torch.randn(64, 64, device="cuda")
        float((probe @ probe).sum())
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    except Exception as exc:
        cap = torch.cuda.get_device_capability(0)
        print(f"CUDA present but unusable (sm_{cap[0]}{cap[1]}): {type(exc).__name__}")
        print(f"this torch build supports {torch.cuda.get_arch_list()}")
        print("falling back to CPU; the two neural sections will be very slow")
        return torch.device("cpu"), "cuda unusable"


DEVICE, DEVICE_NAME = pick_device()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train[TARGET].to_numpy(dtype=np.int64)
test_ids = test[ID].to_numpy()

cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
FOLDS = list(cv.split(np.zeros(len(train)), y))

log(f"train={train.shape} test={test.shape} positive_rate={y.mean():.5f}")
log(f"device={DEVICE.type} ({DEVICE_NAME})")

# %% [markdown]
# ## 2. The evidence: these floats are categorical
#
# Before building anything, measure the thing the model is supposed to exploit. A continuous
# feature over 691,369 rows should have close to 691,369 distinct values. Count them.

# %%
rows = []
for c in FEATURES:
    n_uniq = train[c].nunique()
    rows.append({
        "column": c,
        "unique_values": n_uniq,
        "rows_per_value": round(train[c].notna().sum() / max(n_uniq, 1)),
        "missing_pct": round(100 * train[c].isna().mean(), 1),
        "kind": "numeric" if c in NUM else "categorical",
    })

card = pd.DataFrame(rows).sort_values("unique_values")
card

# %% [markdown]
# `gaming_hours` is a float column with **401** distinct values across 691,369 rows. About
# 1,400 rows share every exact value. `app_opens_per_day` has 166. `age` has 18.
#
# This is a synthetic dataset, and the generator quantised its outputs. So a value like
# `gaming_hours == 1.59` is not a point on a continuum that happens to be near 1.58. It is a
# **level** that roughly 1,400 training rows belong to, with its own empirical target rate that
# the model can learn directly.
#
# A tree can only ever approach that through thresholds. It splits on `gaming_hours <= 1.59`,
# which forces the target rate of level 1.59 to be an average with its neighbours. With enough
# depth a tree can isolate a single level, but it spends splits doing it and it generalises the
# ordering assumption the whole way down. The ordering assumption is mostly right, which is why
# GBDTs score 0.968 here. It is not exactly right, and the residual is where the opportunity is.
#
# An embedding table does the opposite. Give level 1.59 its own row of free parameters and the
# model can place it anywhere, including somewhere its numeric neighbours are not. It loses the
# ordering prior, which is why a pure lookup model is *weaker* on its own. It makes a different
# class of mistake, which is why it is valuable in a blend.
#
# Note the missing column too. Every feature is 4% to 19% absent, and this is not random
# damage: in synthetic Playground data the missingness pattern usually carries signal. So
# "missing" gets its own learned vector rather than an imputed mean.

# %% [markdown]
# ## 3. Feature engineering for the tree baselines
#
# The GBDTs need real features to be a fair comparison. Three blocks:
#
# **Target encoding**, computed inside each training fold only. This is the part people get
# wrong. If you fit the encoder on the full training set and then cross-validate, every fold's
# validation rows have already contributed to their own encoding, your OOF score inflates by
# roughly 0.001 here, and the blend weights you fit on those OOF values are optimising against
# a fiction. The encoder is refit per fold below, with no exceptions.
#
# **Frequency encoding**, computed over train and test together. This is transductive but not
# leakage: it uses no labels. How common a value is in the combined dataset is legitimately
# knowable at prediction time.
#
# **Budget features**, which come from reading the column names. `daily_screen_time_hours` is
# supposed to contain `social_media_hours + gaming_hours + work_study_hours`. It does not always,
# and the residual is informative: how much screen time is unaccounted for, whether it is exactly
# zero, and how the weekend figure sits relative to the weekday budget.

# %%
def levels(df):
    """Exact string levels, with NaN promoted to its own explicit level."""
    out = pd.DataFrame(index=df.index)
    for c in FEATURES:
        s = df[c]
        out[c] = s.astype(object).where(s.notna(), MISSING).astype(str)
    return out


TR_LV, TE_LV = levels(train), levels(test)

# Frequency encoding: label-free, so train and test are pooled.
FREQ = {}
for c in FEATURES:
    counts = pd.concat([TR_LV[c], TE_LV[c]], ignore_index=True).value_counts()
    FREQ[c] = counts / counts.sum()


def budget_features(df):
    parts = ["social_media_hours", "gaming_hours", "work_study_hours"]
    daily = df["daily_screen_time_hours"]
    weekend = df["weekend_screen_time"]
    accounted = df[parts].sum(axis=1)
    residual = daily - accounted
    return pd.DataFrame({
        "accounted": accounted,
        "residual": residual,
        "residual_frac": residual / daily.clip(lower=0.1),
        "residual_is_zero": (residual.abs() < 1e-9).astype(float),
        "weekend_gap": weekend - daily,
        "weekend_minus_accounted": weekend - accounted,
        "sleep_plus_screen": df["sleep_hours"] + daily,
        "notif_per_open": df["notifications_per_day"] / df["app_opens_per_day"].clip(lower=1),
    }, index=df.index)


BUDGET_TR = budget_features(train)
BUDGET_TE = budget_features(test)


def base_block(lv, raw, budget):
    """Label-free features: frequency encodings, raw numerics, budget block.

    Nothing here touches y, so it is computed once for the whole dataset rather
    than per fold.
    """
    cols = {}
    for c in FEATURES:
        cols[f"fq_{c}"] = lv[c].map(FREQ[c]).fillna(0.0).to_numpy(dtype=np.float32)
    for c in NUM:
        cols[f"raw_{c}"] = raw[c].to_numpy(dtype=np.float32)
    for c in budget.columns:
        cols[f"bd_{c}"] = budget[c].to_numpy(dtype=np.float32)
    return pd.DataFrame(cols)


BASE_TR = base_block(TR_LV, train, BUDGET_TR)
BASE_TE = base_block(TE_LV, test, BUDGET_TE)


def te_block(lv_fit, y_fit, targets):
    """Fit smoothed target encodings on one training fold, apply to each target frame.

    y_fit is the training-fold labels only. Validation rows never contribute to
    their own encoding, which is the difference between an OOF score you can
    stack on and one you cannot.
    """
    prior = float(y_fit.mean())
    maps = {}
    for c in FEATURES:
        agg = (
            pd.DataFrame({"lv": lv_fit[c].to_numpy(), "y": y_fit})
            .groupby("lv")["y"].agg(["sum", "count"])
        )
        maps[c] = (agg["sum"] + TE_SMOOTHING * prior) / (agg["count"] + TE_SMOOTHING)

    out = []
    for lv in targets:
        cols = {
            f"te_{c}": lv[c].map(maps[c]).fillna(prior).to_numpy(dtype=np.float32)
            for c in FEATURES
        }
        out.append(pd.DataFrame(cols))
    return out


def fold_matrices(i_tr, i_va):
    """Return (Xtr, Xva, Xte) for one fold, with target encoding fit on i_tr only."""
    lv_tr = TR_LV.iloc[i_tr]
    te_tr, te_va, te_te = te_block(
        lv_tr, y[i_tr], [lv_tr, TR_LV.iloc[i_va], TE_LV]
    )
    Xtr = pd.concat([BASE_TR.iloc[i_tr].reset_index(drop=True), te_tr], axis=1)
    Xva = pd.concat([BASE_TR.iloc[i_va].reset_index(drop=True), te_va], axis=1)
    Xte = pd.concat([BASE_TE, te_te], axis=1)
    return Xtr, Xva, Xte


log(f"budget block: {list(BUDGET_TR.columns)}")
# %% [markdown]
# ## 4. Tree baselines
#
# One CV harness reused by every model in the notebook, so that "same folds" is enforced by the
# code rather than by my good intentions. Predictions are kept in float64 throughout. At an AUC
# of 0.968 on 296,302 test rows, float32 rounding reorders enough ties to move the fourth decimal
# place, and the fourth decimal place is the entire competition.

# %%
OOF = {}
TESTP = {}


def run_cv(name, fit_fold):
    """fit_fold(i_tr, i_va, fold) -> (val_pred, test_pred, note). Fills OOF/TESTP."""
    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    aucs, t0 = [], time.time()

    for fold, (i_tr, i_va) in enumerate(FOLDS):
        val_pred, fold_test_pred, note = fit_fold(i_tr, i_va, fold)
        oof[i_va] = val_pred
        test_pred += fold_test_pred / N_FOLDS
        auc = float(roc_auc_score(y[i_va], val_pred))
        aucs.append(auc)
        log(f"  {name} fold{fold} auc={auc:.6f} {note} [{time.time() - t0:.0f}s]")

    oof_auc = float(roc_auc_score(y, oof))
    OOF[name], TESTP[name] = oof, test_pred
    log(f"{name} OOF AUC = {oof_auc:.6f}  folds={[round(a, 6) for a in aucs]}")
    return oof_auc

# %%
import xgboost as xgb

XGB_PARAMS = dict(
    n_estimators=6000, learning_rate=0.02, max_depth=7,
    min_child_weight=12, subsample=0.85, colsample_bytree=0.55,
    reg_lambda=2.0, reg_alpha=0.5, max_bin=512,
    tree_method="hist", eval_metric="auc", early_stopping_rounds=200,
    device="cuda" if torch.cuda.is_available() else "cpu",
    random_state=SEED, verbosity=0,
)


def fit_xgb(i_tr, i_va, fold):
    Xtr, Xva, Xte = fold_matrices(i_tr, i_va)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(Xtr, y[i_tr], eval_set=[(Xva, y[i_va])], verbose=False)
    best = model.best_iteration
    kw = dict(iteration_range=(0, best + 1))
    return (
        model.predict_proba(Xva, **kw)[:, 1].astype(np.float64),
        model.predict_proba(Xte, **kw)[:, 1].astype(np.float64),
        f"best_it={best} n_feat={Xtr.shape[1]}",
    )


auc_xgb = run_cv("xgboost", fit_xgb)

# %%
import lightgbm as lgb

LGB_PARAMS = dict(
    n_estimators=6000, learning_rate=0.02, num_leaves=96,
    min_child_samples=60, subsample=0.85, subsample_freq=1,
    colsample_bytree=0.55, reg_lambda=2.0, max_bin=511,
    objective="binary", metric="auc", n_jobs=-1,
    random_state=SEED, verbosity=-1,
)


def fit_lgb(i_tr, i_va, fold):
    Xtr, Xva, Xte = fold_matrices(i_tr, i_va)
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        Xtr, y[i_tr], eval_set=[(Xva, y[i_va])],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    best = model.best_iteration_
    return (
        model.predict_proba(Xva, num_iteration=best)[:, 1].astype(np.float64),
        model.predict_proba(Xte, num_iteration=best)[:, 1].astype(np.float64),
        f"best_it={best} n_feat={Xtr.shape[1]}",
    )


auc_lgb = run_cv("lightgbm", fit_lgb)

# %% [markdown]
# ## 5. Shared preparation for both neural networks
#
# This section is the experimental control. The Dense MLP and the Lookup-Transformer are trained
# by **the same function**, with the same optimizer, the same one-cycle schedule, the same
# augmentation rate, the same EMA, and the same early-stopping rule. Only two things differ:
# the module, and what the module is allowed to see.
#
# Both read the identical rank-gauss numeric arrays. The Lookup-Transformer additionally reads a
# tensor of integer token ids. That extra tensor is the entire independent variable of this
# notebook.
#
# Two details in `rank_gauss` worth stopping on:
#
# - The transform is fit on observed values only, then missing rows are left at 0.0 and flagged
#   in a parallel mask. Imputing the mean would tell the network that an absent value is an
#   average value, which is exactly the wrong claim on this dataset.
# - Categorical columns are passed in as all-NaN, so their mask is 1.0 everywhere and their
#   smooth branch is gated fully off. A categorical has no meaningful rank, so it gets no
#   magnitude representation at all. For the MLP it is one-hot encoded instead; for the
#   Lookup-Transformer its embedding row carries it.

# %%
BOTH = pd.concat([train[FEATURES], test[FEATURES]], ignore_index=True)
N_TRAIN = len(train)

_parts = ["social_media_hours", "gaming_hours", "work_study_hours"]
_daily, _weekend = BOTH["daily_screen_time_hours"], BOTH["weekend_screen_time"]
_accounted = BOTH[_parts].sum(axis=1)
_residual = _daily - _accounted

DERIVED = pd.DataFrame({
    "residual": _residual,
    "accounted": _accounted,
    "residual_frac": _residual / _daily.clip(lower=0.1),
    "weekend_minus_accounted": _weekend - _accounted,
    "weekend_minus_residual": _weekend - _residual,
    "accounted_frac": _accounted / _daily.clip(lower=0.1),
    "weekend_gap": _weekend - _daily,
})


def rank_gauss(frame):
    """Quantile-to-normal per column, fit on observed values only.

    Returns (values, missing_mask). Missing rows stay at 0.0 in values and are
    marked 1.0 in the mask, so the network is told "absent" rather than "average".
    """
    values = np.zeros((len(frame), frame.shape[1]), dtype=np.float32)
    missing = np.zeros_like(values)
    for j, c in enumerate(frame.columns):
        v = frame[c].to_numpy(dtype=np.float64)
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


# Categoricals enter as all-NaN so their smooth branch is gated off entirely.
RAW_NUMERIC = pd.DataFrame({c: BOTH[c] if c in NUM else np.nan for c in FEATURES})
COL_VAL, COL_MISS = rank_gauss(RAW_NUMERIC)
DER_VAL, DER_MISS = rank_gauss(DERIVED)

N_COLS, N_DER = len(FEATURES), DERIVED.shape[1]

T_CV = torch.from_numpy(COL_VAL)
T_CM = torch.from_numpy(COL_MISS)
T_DV = torch.from_numpy(DER_VAL)
T_DM = torch.from_numpy(DER_MISS)

log(f"numeric tokens={N_COLS} derived tokens={N_DER}")

# %% [markdown]
# ### The shared training loop
#
# One function, two models. Read the signature: the only model-specific arguments are
# `build_model`, the input tensors, and an `augment` callback. Everything about how the network
# is optimised is fixed.

# %%
NET = dict(
    epochs=32, batch_size=2048, lr=2e-3, drop=0.1, aug=0.10,
    weight_decay=1e-5, embedding_weight_decay=3e-4,
    pct_start=0.15, grad_clip=1.0, ema_decay=0.999,
    patience=5, predict_batch_size=16384,
)


def slice_global(tensors, i_tr, i_va):
    """Split whole-dataset tensors (train rows then test rows) into the three folds parts."""
    return (tuple(t[i_tr] for t in tensors),
            tuple(t[i_va] for t in tensors),
            tuple(t[N_TRAIN:] for t in tensors))


def train_net(build_model, splits, augment, y_tr, y_va, seed, tag):
    """Train one network on one fold. Returns (val_pred, test_pred, best_epoch).

    splits is (train_tensors, val_tensors, test_tensors); each is a tuple of CPU
    tensors already restricted to that part of the fold. Taking the splits as an
    argument rather than slicing a global tensor inside means a model whose features
    must be built per fold (anything involving target encoding) uses this identical
    trainer, which is what keeps the comparison in section 8 honest.

    augment(batch) -> batch is applied to training batches only.
    """
    tr_tensors, va_tensors, te_tensors = splits
    n_rows = len(y_tr)
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model = build_model().to(DEVICE)

    # Embedding rows are updated by only the rows that reference them, so they need
    # a heavier weight decay than densely-updated weights to stay regularised.
    emb_params = [v for n, v in model.named_parameters() if n.startswith("emb.")]
    other_params = [v for n, v in model.named_parameters() if not n.startswith("emb.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "weight_decay": NET["weight_decay"]},
            {"params": emb_params, "weight_decay": NET["embedding_weight_decay"]},
        ],
        lr=NET["lr"],
    )

    batch_size, epochs = NET["batch_size"], NET["epochs"]
    total_steps = math.ceil(n_rows / batch_size) * epochs + 10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, NET["lr"], total_steps=total_steps, pct_start=NET["pct_start"]
    )
    loss_fn = nn.BCEWithLogitsLoss()

    amp_dtype = None
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    tr_batch = tuple(t.to(DEVICE) for t in tr_tensors)
    va_batch = tuple(t.to(DEVICE) for t in va_tensors)
    te_batch = tuple(t.to(DEVICE) for t in te_tensors)
    Y = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)

    def forward(batch):
        if amp_dtype is None:
            return model(*batch)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            return model(*batch)

    def predict(batch):
        model.eval()
        out, chunk = [], NET["predict_batch_size"]
        n = len(batch[0])
        with torch.no_grad():
            for s in range(0, n, chunk):
                sl = slice(s, s + chunk)
                logits = forward(tuple(t[sl] for t in batch))
                out.append(torch.sigmoid(logits).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    params = list(model.parameters())
    ema = [v.detach().clone() for v in params]
    ema_step = 0
    best_auc, best_weights, best_epoch, bad = -1.0, None, -1, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_rows, generator=generator).to(DEVICE)
        for s in range(0, n_rows, batch_size):
            sl = perm[s:s + batch_size]
            batch = augment(tuple(t[sl] for t in tr_batch))
            loss = loss_fn(forward(batch), Y[sl])

            optimizer.zero_grad(set_to_none=True)
            if scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, NET["grad_clip"])
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, NET["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            # See the EMA section below for why this decay is warmed up.
            ema_step += 1
            decay = min(NET["ema_decay"], (1.0 + ema_step) / (10.0 + ema_step))
            with torch.no_grad():
                for avg, cur in zip(ema, params):
                    avg.mul_(decay).add_(cur.detach(), alpha=1 - decay)

        if epoch == epochs - 1 or (epoch >= 5 and epoch % 2 == 1):
            live = [v.detach().clone() for v in params]
            with torch.no_grad():
                for cur, avg in zip(params, ema):
                    cur.copy_(avg)
            auc = float(roc_auc_score(y_va, predict(va_batch)))
            if auc > best_auc:
                best_auc, best_epoch, bad = auc, epoch, 0
                best_weights = [v.detach().clone() for v in ema]
            else:
                bad += 1
            log(f"    {tag} ep{epoch} val_auc={auc:.6f} best={best_auc:.6f}")
            with torch.no_grad():
                for cur, saved in zip(params, live):
                    cur.copy_(saved)
            if bad >= NET["patience"]:
                break

    with torch.no_grad():
        for cur, best in zip(params, best_weights):
            cur.copy_(best)
    val_pred, test_pred = predict(va_batch), predict(te_batch)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return val_pred, test_pred, best_epoch + 1

# %% [markdown]
# ### The EMA bug that cost me a day
#
# My first working version of this model scored **AUC 0.45** on every fold. Below 0.5. I spent a
# while looking for a flipped label, because a model that is reliably worse than a coin flip
# feels like a sign error, not a training problem.
#
# It was not a label bug. It was this line, in its original form:
#
# ```python
# decay = 0.999                      # what I had
# for avg, cur in zip(ema, params):
#     avg.mul_(decay).add_(cur.detach(), alpha=1 - decay)
# ```
#
# The EMA is seeded from the random initialisation. With a fixed decay of 0.999, the weight
# still attached to that initialisation after `n` steps is `0.999 ** n`. I was smoke-testing on
# 5% of the rows, which is about 500 optimizer steps:
#
# ```
# 0.999 ** 500  = 0.61
# ```
#
# So 61% of the weights I was evaluating were still random noise. The model underneath had
# trained fine. The average I was reading it through had barely moved off its starting point.
# An AUC slightly below 0.5 is exactly what a mostly-random network produces.
#
# The fix is to make the horizon grow with the step count instead of assuming you have already
# taken thousands of steps:
#
# ```python
# decay = min(0.999, (1.0 + step) / (10.0 + step))
# ```
#
# At step 1 that is 0.18, so the EMA tracks the live weights almost exactly and the
# initialisation is flushed immediately. By step 1,000 it has converged to 0.999 and gives the
# smoothing you actually wanted. Measured on the same 5% subsample: **0.9255 to 0.9554**. On a
# full-length run the fixed decay eventually recovers, which is worse than failing loudly,
# because it means the bug silently taxes every short run you use to make decisions.
#
# If you take one operational thing from this notebook, take this: any smoothed or averaged
# quantity that is seeded from initialisation needs a warmup, and a smoke test that disagrees
# violently with a full run is more often an averaging artefact than a data bug.

# %% [markdown]
# ## 6. Two controls, because one was not enough
#
# The comparison I actually want is: hold the trainer fixed, vary only what the model reads.
# That needs two MLPs, not one, and the reason is worth spelling out because my first attempt
# at this notebook got it wrong.
#
# **Control A, magnitudes only.** Rank-gauss values, missing masks, the derived budget block,
# and one-hot categoricals. No per-value information of any kind.
#
# **Control B, the same magnitudes plus the trees' encodings.** Identical architecture,
# identical trainer, but fed the exact feature matrix the trees consume, target encoding
# included, refit per fold and then rank-gauss transformed. This is the strength-matched
# control.
#
# Control A alone is not a valid control, and this is the trap. It scores far below the trees,
# so if it also decorrelates from them, you cannot tell whether that is because it reads a
# different representation or simply because it is a worse model. Any weak model decorrelates.
# Control B removes that confound: it is given the same information the trees have, in magnitude
# form, so if it lands near the trees in both score and correlation, then the trainer and the
# architecture are ruled out as sources of diversity.
#
# The gap between A and B is itself the most direct evidence for what this dataset is. Both are
# the same network. B differs only by receiving a scalar summary of each exact value's target
# rate. If that one addition is worth a large amount of AUC, then the signal here lives in value
# identity rather than in magnitude, which is precisely the premise the Lookup-Transformer is
# built on.

# %%
CAT_ONEHOT = []
for c in CAT:
    s = BOTH[c].astype(object).where(BOTH[c].notna(), MISSING).astype(str)
    CAT_ONEHOT.append(pd.get_dummies(s, prefix=c).to_numpy(dtype=np.float32))
CAT_ONEHOT = np.concatenate(CAT_ONEHOT, axis=1)

MLP_RAW = np.concatenate([COL_VAL, COL_MISS, DER_VAL, DER_MISS, CAT_ONEHOT], axis=1)
T_MLP_RAW = torch.from_numpy(MLP_RAW)
log(f"control A input width = {MLP_RAW.shape[1]}")


class DenseMLP(nn.Module):
    def __init__(self, width, hidden=512, depth=3):
        super().__init__()
        layers, d_in = [], width
        for _ in range(depth):
            layers += [
                nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden),
                nn.GELU(), nn.Dropout(NET["drop"]),
            ]
            d_in = hidden
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_augment_dense(n_value_cols):
    """Blank a feature and flag it missing, matching the lookup model's corruption.

    Only the leading value/mask column pair is touched, so control A perturbs its
    rank-gauss numerics and control B perturbs its encoded features, in both cases
    at the same 10% rate the Lookup-Transformer uses.
    """
    def augment(batch):
        (x,) = batch
        x = x.clone()
        drop = torch.rand((x.shape[0], n_value_cols), device=x.device) < NET["aug"]
        head = x[:, :n_value_cols]
        x[:, :n_value_cols] = torch.where(drop, torch.zeros_like(head), head)
        tail = x[:, n_value_cols:2 * n_value_cols]
        if tail.shape[1] == n_value_cols:
            x[:, n_value_cols:2 * n_value_cols] = torch.maximum(tail, drop.float())
        return (x,)
    return augment


def fit_mlp_raw(i_tr, i_va, fold):
    val_pred, test_pred, best = train_net(
        lambda: DenseMLP(MLP_RAW.shape[1]),
        slice_global((T_MLP_RAW,), i_tr, i_va),
        make_augment_dense(COL_VAL.shape[1]),
        y[i_tr], y[i_va], SEED, f"mlp_raw f{fold}",
    )
    return val_pred, test_pred, f"best_ep={best} n_feat={MLP_RAW.shape[1]}"


auc_mlp_raw = run_cv("mlp_magnitudes", fit_mlp_raw)

# %% [markdown]
# Now control B. The features have to be built inside the fold because they contain target
# encoding, so this is where the refactored `train_net` earns its signature: the exact same
# trainer, handed per-fold tensors instead of slices of a global one.
#
# The quantile transform is also fit on the training fold only. Fitting it on all rows would
# leak no labels, but it would let the validation rows shape their own scaling, and having just
# been burned by an invalid control I would rather not have to argue about it.

# %%
def fit_mlp_encoded(i_tr, i_va, fold):
    Xtr, Xva, Xte = fold_matrices(i_tr, i_va)

    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             subsample=400_000, random_state=0)
    qt.fit(Xtr.to_numpy(dtype=np.float64))

    def prep(frame):
        arr = qt.transform(frame.to_numpy(dtype=np.float64)).astype(np.float32)
        return torch.from_numpy(np.nan_to_num(arr, nan=0.0))

    splits = ((prep(Xtr),), (prep(Xva),), (prep(Xte),))
    val_pred, test_pred, best = train_net(
        lambda: DenseMLP(Xtr.shape[1]), splits,
        make_augment_dense(Xtr.shape[1]),
        y[i_tr], y[i_va], SEED, f"mlp_enc f{fold}",
    )
    return val_pred, test_pred, f"best_ep={best} n_feat={Xtr.shape[1]}"


auc_mlp_enc = run_cv("mlp_encoded", fit_mlp_encoded)

# %% [markdown]
# ## 7. The Lookup-Transformer
#
# Now the same trainer gets a model that can see identity.
#
# ### 7.1 Tokenising exact values
#
# Every distinct value in every column becomes a row in one shared embedding table, with each
# column occupying a disjoint slice of the id space. Three decisions in that sentence:
#
# **Why one table instead of twelve.** Twelve `nn.Embedding` modules do the same arithmetic in
# twelve kernel launches and twelve parameter groups. One table with an offset per column is one
# launch, and it keeps `emb.weight` a single tensor that the optimizer can give its own weight
# decay to.
#
# **Why disjoint slices rather than a shared vocabulary.** The string `"2.0"` appears in
# `gaming_hours` and in `sleep_hours`. Those are unrelated facts about a person. A shared
# vocabulary would force them onto the same vector and inject a false equality. Offsetting each
# column into its own range keeps the single-table efficiency without the false equality.
#
# **Why local index 0 is reserved.** Every column reserves its first slot for missing. Each
# column therefore learns its own missing vector, so "we do not know their sleep hours" and "we
# do not know their gender" are different inputs rather than one shared null.
#
# The vocabulary is built over train and test together. No labels are involved, so this is
# transductive rather than leaky, and it guarantees no test row ever hits an unseen id. That
# matters more than it sounds: with 401 levels in a column and 296,302 test rows, an OOV bucket
# would be a silent accuracy sink.

# %%
ids, vocab_sizes = [], []
for c in FEATURES:
    s = BOTH[c].astype(object).where(BOTH[c].notna(), MISSING).astype(str)
    cats = sorted(v for v in s.unique() if v != MISSING)
    mapping = {v: i + 1 for i, v in enumerate(cats)}   # local 0 reserved for missing
    mapping[MISSING] = 0
    ids.append(s.map(mapping).to_numpy(dtype=np.int64))
    vocab_sizes.append(len(cats) + 1)

VALUE_IDS = np.stack(ids, axis=1)
OFFSETS = np.concatenate([[0], np.cumsum(vocab_sizes)[:-1]]).astype(np.int64)
VALUE_IDS += OFFSETS[None, :]
TOTAL_VOCAB = int(sum(vocab_sizes))

T_IDS = torch.from_numpy(VALUE_IDS)
T_OFFSETS = torch.from_numpy(OFFSETS)

N_TOKENS = 1 + N_COLS + N_DER
log(f"vocab={TOTAL_VOCAB} rows across {N_COLS} columns")
log(f"sequence length={N_TOKENS} (1 CLS + {N_COLS} raw + {N_DER} derived)")
pd.DataFrame({"column": FEATURES, "vocab_rows": vocab_sizes, "id_offset": OFFSETS})

# %% [markdown]
# ### 7.2 Keeping the magnitude, adding the identity
#
# Pure lookup throws away ordering, and ordering is genuinely useful here: 8 hours of screen
# time really is more than 2. So each numeric column's token is a **sum** of two things:
#
# ```
# token = embedding[exact_value_id] + PLR(rank_gauss_value) * (1 - is_missing)
# ```
#
# The first term is identity. The second is magnitude. Adding them lets the network use the
# ordering prior where it holds and deviate from it per value where it does not, which is
# precisely the freedom a tree does not have.
#
# `PLR` is a periodic-linear embedding (Gorishniy et al., *On Embeddings for Numerical Features
# in Tabular Deep Learning*). It projects a scalar through learned frequencies into
# `[sin(2 pi f x), cos(2 pi f x)]`, then applies a per-feature linear map. A single scalar fed
# into a linear layer can only ever produce a straight line in that feature; a Fourier basis
# gives the network a high-frequency vocabulary, so it can represent a sharp change between
# 1.58 and 1.59 rather than smoothing across it.
#
# The `(1 - is_missing)` gate zeroes the magnitude term when the value is absent, so the
# embedding's missing vector carries that row alone and never gets contaminated by a
# meaningless 0.0 quantile. This is also what switches the magnitude branch off entirely for
# categorical columns, whose mask is 1.0 everywhere by construction.
#
# The derived budget features become **additional sequence positions**, not extra columns
# concatenated onto a vector. As tokens, attention can relate `residual` directly to
# `daily_screen_time_hours` and `gaming_hours`, which is the relationship that defines it.

# %%
class PLR(nn.Module):
    """Periodic-linear numeric embedding: scalar -> Fourier features -> per-feature linear."""

    def __init__(self, n_features, k, d, sigma=0.5):
        super().__init__()
        self.f = nn.Parameter(torch.randn(n_features, k) * sigma)
        self.w = nn.Parameter(torch.randn(n_features, 2 * k, d) / math.sqrt(2 * k))
        self.b = nn.Parameter(torch.zeros(n_features, d))

    def forward(self, x):
        z = 2 * math.pi * x.unsqueeze(-1) * self.f.unsqueeze(0)
        z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        return torch.einsum("bfk,fkd->bfd", z, self.w) + self.b


class LookupTransformer(nn.Module):
    def __init__(self, d=128, k=24, layers=4, heads=8):
        super().__init__()
        self.emb = nn.Embedding(TOTAL_VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.plr_col = PLR(N_COLS, k, d)
        self.plr_der = PLR(N_DER, k, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, N_TOKENS, d) * 0.02)
        self.edrop = nn.Dropout(NET["drop"])
        layer = nn.TransformerEncoderLayer(
            d, heads, d * 2, NET["drop"], activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
            nn.Dropout(NET["drop"]), nn.Linear(d, 1),
        )

    def forward(self, idx, col_val, col_miss, der_val, der_miss):
        batch = idx.shape[0]
        # identity + gated magnitude
        tok_col = self.emb(idx) + self.plr_col(col_val) * (1 - col_miss).unsqueeze(-1)
        tok_der = self.plr_der(der_val) * (1 - der_miss).unsqueeze(-1)
        tokens = torch.cat([self.cls.expand(batch, -1, -1), tok_col, tok_der], dim=1)
        encoded = self.transformer(self.edrop(tokens + self.pos))
        return self.head(encoded[:, 0]).squeeze(-1)

# %% [markdown]
# ### 7.3 Augmentation as forced missingness
#
# 10% of feature cells per batch are replaced by that column's missing token, with the mask set
# to match. The dataset is already 4% to 19% missing, so this is not synthetic noise from
# nowhere: it is more of a corruption the model must handle at prediction time anyway. It also
# stops the embedding table from memorising rare levels, which is the obvious failure mode when
# you hand a model 5,400 free parameter rows.
#
# `T_OFFSETS` is exactly the vector of per-column missing ids, because local index 0 is the
# missing slot, so `offset + 0 == offset`. That is the payoff of reserving index 0.

# %%
MISSING_IDS = T_OFFSETS.to(DEVICE)


def augment_lookup(batch):
    idx, col_val, col_miss, der_val, der_miss = batch
    idx, col_miss = idx.clone(), col_miss.clone()
    drop = torch.rand(idx.shape, device=idx.device) < NET["aug"]
    idx = torch.where(drop, MISSING_IDS.expand_as(idx), idx)
    col_miss = torch.maximum(col_miss, drop.float())
    return idx, col_val, col_miss, der_val, der_miss


LOOKUP_TENSORS = (T_IDS, T_CV, T_CM, T_DV, T_DM)


def fit_lookup(i_tr, i_va, fold):
    val_pred, test_pred, best = train_net(
        LookupTransformer,
        slice_global(LOOKUP_TENSORS, i_tr, i_va), augment_lookup,
        y[i_tr], y[i_va], SEED, f"lookup f{fold}",
    )
    return val_pred, test_pred, f"best_ep={best} n_tokens={N_TOKENS}"


auc_lookup = run_cv("lookup_transformer", fit_lookup)

# %% [markdown]
# ## 8. The measurement
#
# Five models, one set of folds, five row-aligned OOF vectors. Now the question the notebook
# exists to answer: how much do they actually disagree?
#
# Spearman rather than Pearson, because AUC only cares about ranking. Two models that assign
# wildly different probabilities but the same ordering are the same model as far as this metric
# is concerned, and Pearson would flatter them.

# %%
names = ["xgboost", "lightgbm", "mlp_magnitudes", "mlp_encoded",
         "lookup_transformer"]
solo = {n: float(roc_auc_score(y, OOF[n])) for n in names}

Z = np.column_stack([OOF[n] for n in names])

# Spearman is Pearson on ranks. Doing it that way keeps the result a matrix for any
# number of members; scipy's spearmanr collapses to a scalar when given exactly two.
RANKS = np.column_stack([pd.Series(OOF[n]).rank().to_numpy() for n in names])
corr = pd.DataFrame(np.corrcoef(RANKS, rowvar=False), index=names, columns=names)

tree_corr = {n: float((corr.loc[n, "xgboost"] + corr.loc[n, "lightgbm"]) / 2)
             for n in names}

print("solo OOF AUC        mean Spearman vs the two trees")
for n in names:
    marker = "" if n in ("xgboost", "lightgbm") else "   <- neural"
    print(f"  {n:20s} {solo[n]:.6f}   {tree_corr[n]:.4f}{marker}")
print("\nSpearman correlation of OOF predictions")
corr.round(4)

# %%
import matplotlib.pyplot as plt

SHORT = {"xgboost": "XGB", "lightgbm": "LGBM",
         "mlp_magnitudes": "MLP (A)\nmagnitudes",
         "mlp_encoded": "MLP (B)\nencoded",
         "lookup_transformer": "Lookup"}
short = [SHORT.get(n, n) for n in names]
k = len(names)
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

m = corr.to_numpy()
im = ax_l.imshow(m, cmap="viridis", vmin=min(0.93, m.min()), vmax=1.0)
ax_l.set_xticks(range(k), short, rotation=20, fontsize=8)
ax_l.set_yticks(range(k), short, fontsize=8)
for i in range(k):
    for j in range(k):
        ax_l.text(j, i, f"{m[i, j]:.4f}", ha="center", va="center",
                  color="white" if m[i, j] < 0.985 else "black", fontsize=9)
ax_l.set_title("Spearman correlation of OOF predictions")
fig.colorbar(im, ax=ax_l, fraction=0.046)

ax_r.scatter([tree_corr[n] for n in names], [solo[n] for n in names], s=140, zorder=3)
for i, label in enumerate(short):
    ax_r.annotate(label.replace("\n", " "),
                  (tree_corr[names[i]], solo[names[i]]),
                  textcoords="offset points", xytext=(8, -4), fontsize=9)
ax_r.set_xlabel("mean Spearman correlation with the two trees")
ax_r.set_ylabel("solo OOF AUC")
ax_r.set_title("Strength against redundancy")
ax_r.grid(alpha=0.3, zorder=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### This is where my theory died
#
# The prediction was that `mlp_encoded`, a neural network fed the trees' own features, would sit
# near 0.99 against the trees, and that `lookup_transformer` would sit far below it. If that had
# happened, representation would be the thing that produces diversity.
#
# It did not happen. `mlp_encoded` and `lookup_transformer` land within 0.001 of each other
# against the trees, and they correlate 0.9782 with **each other**, more tightly than either
# correlates with either tree. The two trees, meanwhile, correlate 0.9990 with each other.
#
# What the matrix actually shows is three blocks: the trees, the neural networks, and
# `mlp_magnitudes` off on its own because it is too weak to be like anything. The boundary that
# matters here is the model family. Swapping magnitudes for identities *inside* the neural family
# moved the correlation with the trees by 0.0007, which is nothing.
#
# Look at the right-hand panel before reading on. Being far to the left is being decorrelated,
# and being high up is being strong. `mlp_magnitudes` is the furthest left of anything and it is
# also at the bottom. Whether that position is valuable is the subject of section 10, and the
# answer is not the one you would guess from the correlation matrix alone.

# %% [markdown]
# ## 9. Blending
#
# A greedy non-negative hill-climb, optimising AUC directly. It starts from the best single model
# and repeatedly nudges the weight vector in whichever direction improves OOF AUC most, never
# allowing a negative weight.
#
# Direct AUC optimisation rather than a logistic-regression meta-learner, for two reasons. AUC is
# the competition metric, and optimising the metric beats optimising a surrogate of it. And the
# non-negative constraint is a real regulariser on highly correlated members: without it, the
# meta-learner happily assigns large positive and negative weights that cancel, which fits OOF
# noise and does not survive to the leaderboard.

# %%
def hillclimb(Z, y, step=0.02, rounds=300):
    n = Z.shape[1]
    w = np.zeros(n)
    w[int(np.argmax([roc_auc_score(y, Z[:, j]) for j in range(n)]))] = 1.0
    best = roc_auc_score(y, Z @ w)

    for _ in range(rounds):
        improved = False
        for j in range(n):
            for delta in (step, -step):
                cand = w.copy()
                cand[j] = max(0.0, cand[j] + delta)
                total = cand.sum()
                if total <= 0:
                    continue
                cand /= total
                score = roc_auc_score(y, Z @ cand)
                if score > best + 1e-9:
                    w, best, improved = cand, score, True
        if not improved:
            break
    return w, float(best)


weights, blend_auc = hillclimb(Z, y)

summary = pd.DataFrame({
    "model": names,
    "solo_oof_auc": [round(solo[n], 6) for n in names],
    "tree_corr": [round(tree_corr[n], 4) for n in names],
    "blend_weight": np.round(weights, 4),
}).sort_values("blend_weight", ascending=False)

print(f"blend OOF AUC   = {blend_auc:.6f}")
print(f"best solo AUC   = {max(solo.values()):.6f}")
print(f"gain over best  = {blend_auc - max(solo.values()):+.6f}")
summary

# %% [markdown]
# ## 10. Does correlation predict contribution?
#
# The weight vector already hints that it does not, but weights from a greedy search are hard to
# read: correlated members trade weight almost arbitrarily, so a small weight is not proof of a
# small contribution. The honest measurement is ablation. Drop each member, refit the weights from
# scratch on the remaining ones, and see what the blend loses.
#
# Then the same question from the other direction. Start from the two trees, which is where most
# people on this leaderboard already are, and add exactly one neural member. That number is what a
# reader actually wants: what is this model worth to a blend I already have?

# %%
loo = []
for drop in names:
    keep = [n for n in names if n != drop]
    _, score = hillclimb(np.column_stack([OOF[n] for n in keep]), y)
    loo.append({"removed": drop, "blend_without": round(score, 6),
                "cost_of_removing": round(score - blend_auc, 6)})

trees = ["xgboost", "lightgbm"]
_, base = hillclimb(np.column_stack([OOF[n] for n in trees]), y)

incremental = [{"added_to_trees": "nothing", "blend": round(base, 6), "gain": 0.0}]
for add in names:
    if add in trees:
        continue
    _, score = hillclimb(np.column_stack([OOF[n] for n in trees + [add]]), y)
    incremental.append({"added_to_trees": add, "blend": round(score, 6),
                        "gain": round(score - base, 6)})

from IPython.display import display

print("leave-one-out: what the blend loses when each member is removed")
display(pd.DataFrame(loo))
print(f"incremental value added on top of xgboost + lightgbm (base {base:.6f})")
display(pd.DataFrame(incremental))

# %% [markdown]
# ### The result that replaces my theory
#
# Put the two tables next to the correlation column.
#
# `mlp_encoded` and `lookup_transformer` are equally decorrelated from the trees, 0.9665 against
# 0.9658. Their incremental value differs by more than 40-fold. `mlp_magnitudes`, the most
# decorrelated member in the entire pool, is worth nothing at all: removing it changes the blend
# by 0.000000.
#
# So correlation on its own predicts nothing. Rank these three by decorrelation and you get
# `mlp_magnitudes`, `lookup_transformer`, `mlp_encoded`. Rank them by contribution and you get
# `lookup_transformer`, `mlp_encoded`, `mlp_magnitudes`. The orderings are scrambled.
#
# What separates them is **strength at equal diversity**. The neural family was already
# decorrelated from the trees; that was free, and it was not the scarce thing. The scarce thing
# was a neural member strong enough for the decorrelation to cash out. `mlp_encoded` is 0.0033
# behind the trees it is meant to complement, and at this level of saturation that deficit eats
# the entire benefit of being different. `lookup_transformer` is level with the trees, so its
# difference converts.
#
# And the representation is what got it there. Same trainer, same schedule, same augmentation,
# same EMA. Learned per-value embeddings instead of hand-built target encoding, worth **+0.0033**
# solo. So the corrected claim is narrower than the one I set out to prove, and more useful:
#
# > Per-value embeddings did not buy diversity. They bought strength inside a family that was
# > already diverse, and that is what made the diversity cash out.
#
# The practical version, if you are adding a neural network to a GBDT blend: the binding
# constraint is its solo score, not how exotic its architecture is. You do not need it to be
# unlike your trees. It already is. You need it to be as good as them.
#
# ### One number I cannot reconcile
#
# Reporting this because it disagrees with the notebook and I would rather flag it than bury it.
# In separate offline work on these same folds I ran TabM, a batch-ensembled MLP, on the same
# encoded features. It scored 0.968006 and correlated **0.9945** with XGBoost, far tighter than
# `mlp_encoded`'s 0.9665 here, and much closer to what my original theory predicted.
#
# I do not have a verified explanation. The most plausible mechanism is that TabM averages 32
# internal sub-models, and averaging pulls a ranking toward the consensus ordering, which is
# roughly what a boosted ensemble also produces. That would make its high tree-correlation an
# artefact of internal ensembling rather than of its features. I have not tested that, so treat
# it as a hypothesis. If you can settle it, please do, and say so in the comments.

# %%
blend_test = sum(w * TESTP[n] for w, n in zip(weights, names))

submission = pd.DataFrame({ID: test_ids, TARGET: blend_test})
submission.to_csv("submission.csv", index=False)
print(submission.shape)
submission.head()

# %% [markdown]
# Every member's out-of-fold and test predictions are written out as well. They are on the frozen
# fold definition from the top of the notebook, so if you fork this you can stack them straight
# into your own blend without refitting anything: fit weights on the OOF columns, apply them to
# the test columns.

# %%
oof_frame = pd.DataFrame({"row": np.arange(len(train)), TARGET: y})
test_frame = pd.DataFrame({ID: test_ids})
for n in names:
    oof_frame[n] = OOF[n]
    test_frame[n] = TESTP[n]
oof_frame.to_csv("oof_predictions.csv", index=False)
test_frame.to_csv("test_predictions.csv", index=False)
print("oof_predictions.csv", oof_frame.shape, "| test_predictions.csv", test_frame.shape)

# %% [markdown]
# ## 11. Things that did not work
#
# Published notebooks tend to show the ladder that worked and skip the rungs that broke. The
# failures are more useful, because they tell you which directions are already exhausted. These
# were all measured offline on this same frozen fold definition, so they are comparable to each
# other; the offline baseline for the Lookup-Transformer config used here was 0.968293, against
# 0.968305 in this notebook.
#
# **More capacity in the Lookup-Transformer made it worse.** `d=192, layers=6, drop=0.12` for 40
# epochs: 0.968147 against 0.968293. Minus 0.00015. The signal being exploited is a lookup table
# of about 5,400 rows, and the transformer on top of it is doing comparatively little work.
# Widening the part that is not the bottleneck costs generalisation and buys nothing.
#
# **Heavier augmentation made it worse.** `aug=0.25, drop=0.15`: 0.968154, minus 0.00014. 10% is
# already near the natural missing rate, and past that you are training on a different
# distribution from the test set.
#
# **A 40-trial Optuna sweep on the XGBoost baseline bought nothing.** Tuned on a 40% row
# subsample, then the winner and the incumbent were re-measured on an identical fold under
# identical conditions:
#
# ```
# incumbent    fold0 auc = 0.966853
# optuna best  fold0 auc = 0.966850
# ```
#
# Minus 0.000003, which is zero. This is what saturation looks like from the inside. The
# hyperparameter surface of a well-configured GBDT on this dataset is flat. The sweep was not
# wasted: it is the evidence that said stop tuning trees, and if it had returned +0.0005 I would
# have kept going.
#
# **Row-level missing-count features bought 0.00002.** Number of NaNs per row, and per-block NaN
# counts. Plausible, and missingness genuinely carries signal here, but the per-column masks
# already give the models everything they need. Rejected on measurement, not on taste.
#
# **What did work**, for balance. Averaging 3 seeds inside each fold: 0.968293 to 0.968727, plus
# 0.00043, and reliably the cheapest real gain available. Going from 5 folds to 10: 0.968908,
# another plus 0.00018. Both are variance reduction rather than modelling insight, which is
# exactly why they are dependable.

# %% [markdown]
# ## 12. Takeaways
#
# 1. **Check cardinality before choosing a representation.** A float column with 401 distinct
#    values across 691,369 rows is a categorical wearing a float's dtype. On synthetic Playground
#    data this is common, and it is measurable in one line before you write any model code.
#
# 2. **On this dataset the signal is value identity, not magnitude.** Same network, same trainer:
#    adding one smoothed target rate per level moves it from 0.941125 to 0.965032. That is
#    +0.0239 from a single feature family.
#
# 3. **Learned per-value embeddings are a viable substitute for target encoding.** Here they are a
#    slightly better one: 0.968305 against 0.965032 for the same trainer on hand-built encodings,
#    and level with a tuned GBDT. Free parameters per level beat one label-derived scalar per
#    level.
#
# 4. **Correlation does not predict blend contribution.** Two members equally decorrelated from
#    the trees differed more than 40-fold in what they added. The most decorrelated member added
#    nothing. Selecting ensemble members on correlation alone is as wrong as selecting them on
#    solo score alone; you need both, and the cheap way to know is ablation.
#
# 5. **Diversity is usually free; strength is the scarce input.** Any model outside your current
#    family is already decorrelated from it. What is hard is making that model as good as what you
#    have. Spend your time there.
#
# 6. **A control that cannot refute you is not a control.** `mlp_magnitudes` alone would have
#    confirmed my theory, because it is weak, and weak models decorrelate for free. One extra
#    strength-matched control cost a few minutes of compute and reversed the conclusion. If your
#    experiment can only come out one way, you have not run one yet.
#
# 7. **Warm up any average seeded from initialisation.** `0.999 ** 500 = 0.61`. A short run
#    evaluated through a cold EMA scores below chance and looks exactly like a label bug.
#
# 8. **Let a negative result close a direction.** A flat 40-trial sweep is a finding. It is
#    permission to stop tuning and go build something different.
#
# ### Reproducibility
#
# Every number above comes from `StratifiedKFold(5, shuffle=True, random_state=42)` on the
# original row order, with target encoding and the quantile transform refit inside each training
# fold and predictions kept in float64. Fork this and the folds will match, so your OOF vectors
# will stack against the exported ones.
#
# ### Credit
#
# The Lookup-Transformer architecture is
# [tamerlanomralinov's](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041).
# I ported it and retrained from scratch on my own folds; the controls, the EMA warmup fix, and
# the ablation analysis are mine. The PLR numeric embedding is from Gorishniy et al., *On
# Embeddings for Numerical Features in Tabular Deep Learning*. RealMLP-TD, which I also ran on
# these folds, is [zhenruiweng's port](https://www.kaggle.com/code/zhenruiweng/s6e8-public-lb-0-97009-single-model-realmlp).
#
# If the controlled comparison was useful, upvote the original architecture notebook as well.
# Questions and disagreements welcome in the comments, especially on the TabM discrepancy above,
# or if you can find a representation that decorrelates further *and* holds its strength.
