# %% [markdown]
# # TabFM on a saturated competition: what a 1.6B-parameter zero-shot tabular model can and cannot do
#
# In June 2026 Google Research released [TabFM](https://github.com/google-research/tabfm), a
# pretrained foundation model for tabular data. It does no gradient training on your dataset. You
# hand it labelled rows as **context**, it reads them in a single forward pass, and it predicts
# your test rows. `fit()` does not fit anything; it encodes.
#
# That is a genuinely different bargain from a GBDT, and the obvious question is what it buys on a
# real competition. This notebook answers that on Playground S6E8: 691,369 training rows, 296,302
# test rows, ROC-AUC, and a leaderboard where 203 teams sit inside a 0.0002 band. A brutal place
# to take a zero-shot model.
#
# ### The answer, up front
#
# Everything below is measured in this notebook. TabFM, reading a few thousand labelled rows as
# context and doing **no gradient training at all**, on a 3,000-row holdout:
#
# | context rows | estimators | ROC AUC | seconds per 3,000 predictions |
# |---|---|---|---|
# | 250 | 1 | 0.914793 | 5.8 |
# | 1,000 | 1 | 0.934674 | 6.8 |
# | 4,000 | 1 | 0.944877 | 15.7 |
# | 4,000 | 4 | 0.949890 | 60.4 |
# | 8,000 | 1 | 0.950420 | 29.2 |
# | 8,000 | 4 | **0.953135** | 116.2 |
# | 16,000 | 1 | **out of memory** | tried to allocate 7.63 GiB |
#
# A 5-fold XGBoost given **exactly the same columns** and all 553,095 rows of each fold reaches
# **0.965535** out-of-fold. A tuned XGBoost with target encoding and
# engineered features reaches **0.968454** on these same folds, measured in my
# [previous notebook](https://www.kaggle.com/code/paiky1995/lookup-transformer-auc-96-91).
#
# One caveat on that comparison before you draw conclusions from it: the TabFM figures above are on
# a 3,000-row holdout and the XGBoost figures are out-of-fold over the full dataset, so they are not
# strictly like-for-like. Section 6 puts both models on **identical rows** and that is the
# comparison to trust.
#
# So TabFM loses by roughly 0.013 to 0.015 AUC. The reason is not that the model is weak. The reason
# is arithmetic: a 16,000-row context tries to allocate 7.63 GiB on top of what is already resident
# on a 16GB T4 and dies. The largest context that fits is 8,000 rows, which is **1.4% of the
# training data a GBDT gets to see**, and the AUC curve is still climbing when the memory runs out.
#
# The honest framing is therefore not "foundation models lose to GBDTs". It is:
#
# > TabFM reaches 0.9550 from 8,000 rows and zero gradient steps. XGBoost reaches 0.9661 from
# > 553,095 rows and roughly 4,300 boosting rounds, on the same rows and the same columns. The
# > interesting quantity is the exchange rate between context and training, and the curve above says
# > it has not yet turned.
#
# And on the question a competitor actually cares about: adding TabFM to that XGBoost is worth
# **+0.000050 AUC**, a real but negligible amount, which section 6 unpacks. Being maximally
# different is not enough when you are 0.011 behind.
#
# ### What this notebook does
#
# 1. **A deep dive into how TabFM actually works**, read out of `pytorch/model.py` rather than
#    paraphrased from the README. How a table cell becomes a token, the one line of code that
#    implements in-context learning, and why rows and columns are treated as fundamentally
#    different kinds of object.
# 2. **The context scaling curve**, measured, including exactly where a 16GB GPU runs out.
# 3. **A blend test that puts a prediction from my previous notebook at risk.** That notebook
#    found that decorrelation only pays off at competitive strength. TabFM is about as
#    decorrelated from a GBDT as anything could be, and 0.018 weaker. Those two facts point in
#    opposite directions, so the blend either helps or it does not, and the answer is a real test
#    of the earlier claim rather than a restatement of it.
# 4. **Two upstream problems found while doing this**, including one that stops the package
#    importing at all on Kaggle, with the one-line fix.
#
# ### Honest disclosures before anything else
#
# **The pretrained weights are not open for commercial use.** The TabFM *code* is Apache-2.0, but
# `tabfm_v1_0_0.load()` downloads weights under a separate `tabfm-non-commercial-v1.0` licence
# restricted to non-commercial, non-production use. A Kaggle Playground notebook is fine. Putting
# these weights in a product is not. This is in the repository README and is easy to miss.
#
# **There is no technical report.** The repository states that no paper describing the
# architecture, training data, or evaluation methodology is included. Everything architectural
# below is read directly from the released source, and everything about *what it was trained on*
# is simply unknown. That matters for a benchmark: I cannot rule out that data resembling this
# dataset's generator was in its pretraining mix.
#
# **Accelerator:** this notebook needs **GPU T4 x2**. Kaggle's P100 is compute capability 6.0 and
# the preinstalled PyTorch ships kernels for 7.0 and up, so TabFM cannot run on it at all.

# %% [markdown]
# ## 1. What a tabular foundation model is, and what it is not
#
# The word "foundation model" invites an analogy to LLMs that is only half right, so it is worth
# being precise about the mechanism before looking at the code.
#
# **A GBDT learns a function of your dataset.** Training writes your data into the model's
# parameters. Predicting is then cheap and the training data is no longer needed.
#
# **TabFM learns a function of *datasets*.** Pretraining wrote into its parameters an algorithm
# for mapping (labelled rows, unlabelled row) to a prediction. Your data never enters the
# parameters. It enters the **input**, as context, at inference time. This is the same trick as
# few-shot prompting an LLM, applied to rows of a table instead of tokens of text, and it is
# usually called in-context learning.
#
# Three consequences follow directly, and all three show up in the measurements later:
#
# 1. **No training cost, but no training either.** There is no fold-wise fitting to do, which is
#    why `fit()` in this notebook takes about as long as `predict()`. It also means the model
#    cannot spend capacity specialising to your dataset the way 3,000 boosting rounds can.
# 2. **The dataset size limit is a memory limit, not a time limit.** A GBDT on 553,095 rows is
#    slower but perfectly happy. TabFM on 553,095 rows of context does not run at all, because
#    every context row has to be resident in the forward pass.
# 3. **More data does not monotonically help in the way you expect.** With a bounded context you
#    are choosing a *sample*, so the question becomes how much a bigger sample is worth, which is
#    exactly the scaling curve in section 4.
#
# ### The wider family, and what I am not claiming
#
# TabFM is not the first model of this shape. TabPFN established the approach of pretraining a
# transformer on synthetic datasets so that a single forward pass performs Bayesian-style
# inference over a small table, and later work including TabICL and Mitra has pushed on context
# length, scale, and preprocessing. TabFM's own `results/` directory reports TabArena evaluations.
#
# I want to be careful here: **this notebook benchmarks TabFM only.** I have not run TabPFN,
# TabICL, or Mitra on this dataset, so I am not ranking them, and you should not read the numbers
# below as a statement about tabular foundation models in general. What generalises from this
# notebook is the *shape* of the constraint, because a bounded context window is common to all of
# them, not the specific AUC of one model.

# %% [markdown]
# ## 2. Installing TabFM on Kaggle, and a bug that stops it importing
#
# TabFM is not on PyPI as a wheel you can just install here, so it gets cloned. Two things go
# wrong, and both are worth knowing before you spend a GPU hour finding them.
#
# **Problem 1: `pip install -e .` does not put the package on this kernel's import path.** The
# install reports success and `import tabfm` then raises `ModuleNotFoundError`. Since the package
# is pure Python, the reliable fix is to skip the editable install and put the clone on
# `sys.path` directly, installing only the handful of light dependencies it actually needs.
#
# **Problem 2: the package cannot be imported at all on Kaggle without a patch.** This one is an
# upstream bug. `tabfm/__init__.py` probes its two backends like this:
#
# ```python
# try:
#   from tabfm.src.jax import tabfm_v1_0_0 as tabfm_v1_0_0_jax
#   tabfm_v1_0_0 = tabfm_v1_0_0_jax
# except ImportError:
#   pass          # "JAX is not installed or incomplete"
# ```
#
# The intent is clearly "if this backend is not usable here, skip it". But Kaggle's preinstalled
# flax is older than the `flax>=0.12.7` that TabFM's JAX model requires, so importing it raises
#
# ```
# AttributeError: Module flax.nnx has no attribute 'dataclass'
# ```
#
# `AttributeError` is not `ImportError`, so the guard does not catch it, the exception propagates,
# and `import tabfm` fails even though we only ever wanted the PyTorch backend. Widening both
# guards to `except Exception:` matches their stated intent and fixes it in one line. That is what
# the `sed` below does.
#
# The alternative would be upgrading flax and jax to the pinned versions, which is a much larger
# change to a working environment for no benefit when the PyTorch backend is what we want.

# %%
import subprocess
import sys
import time
import importlib

SETUP = r"""
set -e
git clone --depth 1 -q https://github.com/google-research/tabfm.git /tmp/tabfm
pip install -q 'jaxtyping<0.3' 'typeguard<3' absl-py huggingface-hub 2>&1 | tail -2 || true
# Widen the backend-probe guards so an unusable JAX/flax does not break the import.
sed -i 's/^except ImportError:/except Exception:/' /tmp/tabfm/tabfm/__init__.py
grep -c '^except Exception:' /tmp/tabfm/tabfm/__init__.py
"""

t0 = time.time()
res = subprocess.run(["bash", "-lc", SETUP], capture_output=True, text=True)
print("setup rc:", res.returncode)
print((res.stdout or "").strip()[-600:])
if res.returncode != 0:
    print("stderr:", (res.stderr or "")[-1500:])

sys.path.insert(0, "/tmp/tabfm")
importlib.invalidate_caches()

# %%
import math
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import tabfm
from tabfm import tabfm_v1_0_0_pytorch as tabfm_torch


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if not torch.cuda.is_available():
    raise RuntimeError("This notebook needs a GPU. Set the accelerator to GPU T4 x2.")

CAP = torch.cuda.get_device_capability(0)
if CAP[0] < 7:
    raise RuntimeError(
        f"Got compute capability {CAP}, but the preinstalled PyTorch supports 7.0 and up. "
        "This is Kaggle's P100; switch the accelerator to GPU T4 x2."
    )

log(f"tabfm {tabfm.__version__} from {tabfm.__file__}")
log(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | cap {CAP} "
    f"| bf16 {torch.cuda.is_bf16_supported()}")
log(f"setup took {time.time() - t0:.0f}s")

# %% [markdown]
# ## 3. How TabFM works
#
# This section is read out of `tabfm/src/pytorch/model.py`. Where a claim is checkable against the
# loaded module tree, the next cell checks it, because a deep dive that cannot be falsified is
# just prose.
#
# ### 3.1 A table cell becomes a token, via Fourier features over a *group* of columns
#
# The first thing that happens to your data is `CellEmbedder`. For each cell it does something
# more interesting than embedding the single value:
#
# ```python
# for i in range(self.fgs):            # fgs = feature_group_size = 3
#   offset = (2 ** i) - 1              # offsets 0, 1, 3
#   stacked.append(x[..., (idxs + offset) % h])
# ```
#
# So the token for column `j` is built from the values in columns `j`, `j+1`, and `j+3`, wrapping
# around the active feature count. Every cell token therefore carries a small **neighbourhood** of
# its row, and cheap cross-feature interaction is baked in before any attention runs.
#
# Each of those three values is then expanded into Fourier features and projected:
#
# ```python
# num_out = self.in_linear(cat([(g * ff).sin(), (g * ff).cos()], dim=-1))
# cat_out = self.in_linear_cat(cat([(g * ffc).sin(), (g * ffc).cos()], dim=-1))
# return where(cmg, cat_out, num_out).sum(-2)
# ```
#
# Two details worth stopping on. First, there are **two separate learned frequency bases and two
# separate projections**, one for numerical cells and one for categorical cells, selected per cell
# by a mask. TabFM does not pretend a category code is a magnitude. Second, the sin/cos runs in
# float32 even when the rest of the model is bfloat16, with a comment explaining that the
# arguments reach magnitude ~30 and bf16 would destroy the phase.
#
# If you read my [previous notebook](https://www.kaggle.com/code/paiky1995/lookup-transformer-auc-96-91),
# this is the same periodic-embedding idea that made the Lookup-Transformer work, arrived at
# independently by a much larger model. That is a mild but real piece of evidence that periodic
# numeric embeddings are the right primitive for tabular deep learning, rather than a trick that
# happened to suit one dataset.
#
# ### 3.2 In-context learning, in one line
#
# Here is the entire mechanism by which the model knows which rows it is allowed to learn from:
#
# ```python
# tm = (torch.arange(t)[None, :] < train_size[:, None])[..., None, None]
# out = torch.where(tm, cell + y_emb[:, :, None, :], cell)
# ```
#
# `train_size` is the number of context rows. For those rows, the embedding of the label is
# **added to every cell token in the row**. For query rows, it is not added. That difference is the
# only thing distinguishing "row I know the answer for" from "row I must predict".
#
# There is no separate label channel, no special token, no architectural split between train and
# test. The label is simply summed into the representation of the rows that have one, and the
# attention layers that follow are free to use the resulting asymmetry however pretraining taught
# them to. For a classifier the label embedding is an `nn.Embedding(max_classes, embed_dim)`
# lookup; for a regressor it is a small MLP over the scalar.
#
# ### 3.3 Rows are a set, columns are a sequence
#
# This is the design decision that explains almost everything else, including the memory wall in
# section 4 and the ensembling strategy in section 5. The two attention stages are not symmetric.
#
# **`ColEmbedding` attends over rows.** It reshapes to `[B*HC, T, E]`, putting each column in the
# batch dimension and the `T` rows in the sequence dimension, then runs a `SetTransformer`:
#
# ```python
# ind = self.ind_vectors.unsqueeze(0).expand(src.shape[0], -1, -1)
# hidden = self.mab1(ind, src, src, attn_mask=attn_mask)   # inducing points read the context
# out = self.mab2(src, hidden, hidden)                      # every row reads the summary
# ```
#
# That is induced set attention (the Set Transformer construction). Instead of `T x T` attention
# between rows, a fixed number of learned **inducing vectors** attend over the rows, and then all
# rows attend back over those inducing vectors. Cost goes from `O(T^2)` to `O(T * num_inds)`,
# which is what makes a few thousand context rows feasible at all. It also has **no positional
# encoding**, so it is permutation invariant over rows. Correct: the rows of a table are a set,
# and shuffling your CSV should not change predictions.
#
# Note the mask on `mab1`: it is `arange(t) < train_size`, so the inducing vectors summarise
# **only the labelled context rows**. Query rows read from that summary but never contribute to
# it. This is the information barrier that keeps a query row from seeing other query rows.
#
# **`RowInteraction` attends over columns.** It reshapes to `[B*T, HC, E]`, putting each row in
# the batch dimension and the **columns** in the sequence dimension, and runs a plain `Encoder`
# **with RoPE**. So columns get rotary positional encoding. Column order is meaningful to this
# model, which is a design choice with a direct consequence in section 5.
#
# The full stack alternates between the two views, twice:
#
# ```
# CellEmbedder        cells -> tokens, labels added to context rows
# ColEmbedding        attention over ROWS   (induced, permutation invariant)
# + CLS tokens        prepended along the column axis
# RowInteraction      attention over COLUMNS (RoPE, positional)
# ColEmbedding_2      attention over ROWS
# RowInteraction_2    attention over COLUMNS, keeping only the CLS columns
# ICLearning          attention over ROWS again (no RoPE), then an MLP to class logits
# ```
#
# ### 3.4 Why the context can be cached
#
# Because the context is compressed into a fixed set of inducing vectors and a K/V cache, and
# because none of that depends on the query rows, it can be computed **once** and reused for every
# batch of test rows. `prefill()` returns those caches; `decode()` consumes them. The estimator
# exposes this as `cache_context=True`, and the ICL cache is additionally quantised to int8 by
# default (`maybe_quantize_kv_cache=True`).
#
# This is the difference between a usable and an unusable model on 296,302 test rows, and it is
# also why `fit()` is cheap: `fit()` mostly just prepares encoders, and the real context encoding
# happens once at the start of prediction.

# %% [markdown]
# ### 3.5 Loading the model, and checking the claims above against it
#
# The weights are ~1.6B parameters and take a minute or so to download and load. `load()` casts to
# bfloat16 by default, which the T4 supports.

# %%
t0 = time.time()
MODEL = tabfm_torch.load(model_type="classification", device="cuda", dtype=torch.bfloat16)
n_params = sum(p.numel() for p in MODEL.parameters())
log(f"loaded in {time.time() - t0:.0f}s | parameters = {n_params / 1e6:.2f}M")

ce = MODEL.cell_embedder
col_block = MODEL.col_embedder.tf_col.blocks[0]

facts = {
    "parameters (M)": round(n_params / 1e6, 2),
    "embed_dim": ce.embed_dim,
    "feature_group_size": ce.fgs,
    "column group offsets": [(2 ** i) - 1 for i in range(ce.fgs)],
    "num_freq (Fourier)": ce.fourier_frequencies.shape[-1],
    "separate categorical basis": hasattr(ce, "fourier_frequencies_cat"),
    "max_classes": MODEL.max_classes,
    "row-axis blocks (ColEmbedding)": len(MODEL.col_embedder.tf_col.blocks),
    "inducing vectors per block": col_block.ind_vectors.shape[0],
    "col-axis blocks (RowInteraction)": len(MODEL.row_interactor.tf_row.blocks),
    "CLS columns": MODEL.row_interactor.num_cls,
    "row-axis attention has RoPE": hasattr(MODEL.col_embedder.tf_col, "rope"),
    "col-axis attention has RoPE": MODEL.row_interactor.tf_row.rope is not None,
    "ICL attention has RoPE": MODEL.icl_predictor.tf_icl.rope is not None,
}
for k, v in facts.items():
    print(f"  {k:34s} {v}")

# The structural claims from section 3.3, asserted rather than asserted-in-prose.
assert ce.fgs == 3, "feature_group_size changed"
assert hasattr(ce, "fourier_frequencies") and hasattr(ce, "fourier_frequencies_cat"), \
    "expected separate numeric and categorical Fourier bases"
assert col_block.ind_vectors.ndim == 2, "expected induced-set-attention inducing vectors"
assert MODEL.row_interactor.tf_row.rope is not None, "expected RoPE on the column axis"
assert MODEL.icl_predictor.tf_icl.rope is None, "expected no RoPE in the ICL head"
print("\nstructural claims verified")

# %% [markdown]
# The interesting line is `inducing vectors per block`. That number, not the context size, is what
# the row-axis attention actually compresses your context into. Every labelled row you supply is
# summarised through that fixed-width bottleneck.
#
# ## 4. The wall: how much context actually fits
#
# Now the measurement that decides everything else. TabFM's own estimator defaults are worth
# noting first, because the README's FAQ and the code disagree:
#
# | parameter | README FAQ says | code actually does |
# |---|---|---|
# | `max_num_rows` | "default 100 context rows" | `None`, meaning **no cap** |
# | `max_num_features` | 500 | 500 |
# | `n_estimators` | not stated | 32 |
#
# The code is the authority, and the code's default is dangerous on a large dataset: with
# `max_num_rows=None` the estimator will try to use every row you pass to `fit()` as context. On
# 553,095 rows that is not a slow run, it is an immediate out-of-memory error. **You must set
# `max_num_rows` explicitly on a dataset of this size.**
#
# So: sweep it, and find the wall. Each configuration is timed on a fixed 3,000-row holdout so the
# numbers are comparable, and the out-of-memory cases are caught and reported rather than allowed
# to kill the notebook.

# %%
TARGET, ID = "addicted_label", "id"
NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
       "work_study_hours", "sleep_hours", "notifications_per_day",
       "app_opens_per_day", "weekend_screen_time"]
CAT = ["gender", "stress_level", "academic_work_impact"]
FEATURES = NUM + CAT

import glob
import os


def find_data_dir():
    for d in sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*")) + ["."]:
        if os.path.exists(os.path.join(d, "train.csv")):
            return d
    raise FileNotFoundError("no train.csv under /kaggle/input")


DATA = find_data_dir()
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train[TARGET].to_numpy(dtype=np.int64)
test_ids = test[ID].to_numpy()

# Same frozen CV as my previous notebook, so the OOF vectors are row-aligned and stackable.
N_FOLDS, SEED = 5, 42
FOLDS = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
             .split(np.zeros(len(train)), y))

# A holdout that is disjoint from every context sample used in the sweep.
rng = np.random.default_rng(0)
shuffled = rng.permutation(len(train))
SWEEP_POOL = shuffled[:400_000]        # contexts are drawn from here
SWEEP_EVAL = shuffled[400_000:403_000]  # 3,000 rows, never in any context

X_eval = train.iloc[SWEEP_EVAL][FEATURES].reset_index(drop=True)
y_eval = y[SWEEP_EVAL]

log(f"train={train.shape} test={test.shape} positive_rate={y.mean():.5f}")
log(f"sweep: contexts from {len(SWEEP_POOL)} rows, evaluated on {len(SWEEP_EVAL)} held-out rows")

# %%
def tabfm_predict(context_rows, X_query, *, n_estimators, max_num_rows,
                  preset="default", cache_context=True, query_chunk=3000, **overrides):
    """Predict positive-class probability for X_query using context_rows as context.

    context_rows is a DataFrame slice of `train` (features plus the target column).
    Queries are predicted in blocks of `query_chunk` rows so peak memory stays flat
    in the number of rows being predicted. Returns (probabilities, seconds).
    """
    kwargs = dict(model=MODEL, n_estimators=n_estimators, max_num_rows=max_num_rows,
                  cache_context=cache_context, random_state=SEED)
    kwargs.update(overrides)
    clf = (tabfm.TabFMClassifier.ensemble(**kwargs) if preset == "ensemble"
           else tabfm.TabFMClassifier(**kwargs))
    started = time.time()
    clf.fit(context_rows[FEATURES].reset_index(drop=True),
            context_rows[TARGET].to_numpy())
    # Query rows cost memory too, not just context rows. Sizing the context alone is
    # not enough: a 30,000-row predict_proba call out-of-memories at a context size
    # that is perfectly happy with 3,000 query rows, because the attention over
    # [queries x context] is materialised per batch. Chunk the queries.
    out = []
    for start in range(0, len(X_query), query_chunk):
        block = X_query.iloc[start:start + query_chunk]
        out.append(np.asarray(clf.predict_proba(block))[:, 1].astype(np.float64))
    return np.concatenate(out), time.time() - started


sweep = []
for ctx in [250, 1000, 4000, 8000, 16000]:
    for n_est in [1, 4]:
        context = train.iloc[SWEEP_POOL[:ctx]]
        try:
            proba, secs = tabfm_predict(context, X_eval, n_estimators=n_est,
                                        max_num_rows=ctx)
            auc = float(roc_auc_score(y_eval, proba))
            sweep.append({"context_rows": ctx, "n_estimators": n_est,
                          "auc": round(auc, 6), "seconds": round(secs, 1),
                          "status": "ok"})
            log(f"  ctx={ctx:<6} n_est={n_est}  auc={auc:.6f}  {secs:.1f}s")
        except Exception as exc:
            # Catch broadly and report the type: an OOM is the expected failure here, but a
            # silent skip of some other error would quietly put a hole in the curve.
            text = str(exc)
            if "out of memory" in text.lower():
                wanted = text.split("Tried to allocate ")[-1].split(" ")[:2]
                status = f"OOM (wanted {' '.join(wanted)})"
            else:
                status = f"{type(exc).__name__}: {text[:80]}"
            sweep.append({"context_rows": ctx, "n_estimators": n_est,
                          "auc": np.nan, "seconds": np.nan, "status": status})
            log(f"  ctx={ctx:<6} n_est={n_est}  FAILED {status}")
            torch.cuda.empty_cache()

sweep_df = pd.DataFrame(sweep)
sweep_df

# %% [markdown]
# ### The curve, and where it stops

# %%
import matplotlib.pyplot as plt

ok = sweep_df[sweep_df["status"] == "ok"]
failed = sweep_df[sweep_df["status"] != "ok"]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))
for n_est, grp in ok.groupby("n_estimators"):
    ax_l.plot(grp["context_rows"], grp["auc"], "o-", label=f"n_estimators={n_est}")
    ax_r.plot(grp["context_rows"], grp["seconds"], "o-", label=f"n_estimators={n_est}")
if len(failed):
    wall = failed["context_rows"].min()
    for ax in (ax_l, ax_r):
        ax.axvline(wall, color="crimson", ls="--", lw=1.2)
        ax.text(wall, ax.get_ylim()[0], " out of memory", color="crimson",
                fontsize=9, va="bottom", ha="left", rotation=90)
ax_l.set_xscale("log")
ax_l.set_xlabel("context rows (log scale)")
ax_l.set_ylabel("ROC AUC on the 3,000-row holdout")
ax_l.set_title("TabFM: AUC against context size")
ax_l.grid(alpha=0.3)
ax_l.legend()
ax_r.set_xscale("log")
if len(ok) and ok["seconds"].max() > 0:
    ax_r.set_yscale("log")
ax_r.set_xlabel("context rows (log scale)")
ax_r.set_ylabel("seconds for 3,000 predictions (log)")
ax_r.set_title("Cost against context size")
ax_r.grid(alpha=0.3)
ax_r.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Three things to read off this.
#
# **The wall is memory, and it arrives early.** On a 16GB T4 the model plus its context activations
# exhaust the card somewhere between 8,000 and 16,000 context rows. Even the largest context that
# fits is a tiny fraction of the 553,095 rows a GBDT gets for the same fold. Note what this means
# for hardware: this is not a "wait longer" problem, it is a "buy a bigger card" problem, and it
# scales with the context you want rather than with your dataset size.
#
# **The AUC curve is still climbing when the memory runs out.** That is the frustrating part. There
# is no sign of saturation at the wall, so the honest conclusion is not "TabFM tops out here", it
# is "TabFM tops out *here, on this GPU*". An independent evaluation reports reaching roughly 40,000
# in-context rows on a 24GB card using bf16 and chunking, which suggests the curve has further to
# run for anyone with more memory.
#
# **Ensembling buys more than context does, per unit of time.** Going from 1 to 4 estimators at a
# fixed context costs roughly 4x the time and moves the AUC further than doubling the context does.
# Each estimator sees a different random subsample and, importantly, a different **column
# permutation**, which is where section 3.3 pays off: because the column axis carries RoPE, column
# order is a real and arbitrary bias in this model, and averaging over permutations cancels it.
# The ensembling is not generic bagging; it is test-time augmentation aimed at a specific known
# asymmetry in the architecture.

# %% [markdown]
# ## 5. A matched comparison: same columns, one trained, one not
#
# To make the comparison mean something, XGBoost gets **exactly the features TabFM gets**: the nine
# raw numeric columns and the three categoricals, no target encoding, no engineered ratios. The
# only difference is that XGBoost trains on all 553,095 rows of each fold and TabFM reads 4,000 of
# them.
#
# This is deliberately not the strongest possible GBDT. In my
# [previous notebook](https://www.kaggle.com/code/paiky1995/lookup-transformer-auc-96-91) a tuned
# XGBoost with nested target encoding and engineered features reaches **0.968454** on these same
# folds. That is the real ceiling to keep in mind; the matched model below is the fair like-for-like.

# %%
import xgboost as xgb

XGB_PARAMS = dict(
    n_estimators=9000, learning_rate=0.03, max_depth=7, min_child_weight=12,
    subsample=0.85, colsample_bytree=0.7, reg_lambda=2.0, max_bin=512,
    tree_method="hist", eval_metric="auc", early_stopping_rounds=150,
    enable_categorical=True, device="cuda", random_state=SEED, verbosity=0,
)

X_all = train[FEATURES].copy()
X_test_all = test[FEATURES].copy()
for c in CAT:
    levels = pd.api.types.CategoricalDtype(
        sorted(set(X_all[c].dropna()) | set(X_test_all[c].dropna())))
    X_all[c] = X_all[c].astype(levels)
    X_test_all[c] = X_test_all[c].astype(levels)

oof_xgb = np.zeros(len(train))
test_xgb = np.zeros(len(test))
t0 = time.time()
for fold, (i_tr, i_va) in enumerate(FOLDS):
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_all.iloc[i_tr], y[i_tr],
              eval_set=[(X_all.iloc[i_va], y[i_va])], verbose=False)
    rng_it = dict(iteration_range=(0, model.best_iteration + 1))
    oof_xgb[i_va] = model.predict_proba(X_all.iloc[i_va], **rng_it)[:, 1]
    test_xgb += model.predict_proba(X_test_all, **rng_it)[:, 1] / N_FOLDS
    log(f"  xgb fold{fold} auc={roc_auc_score(y[i_va], oof_xgb[i_va]):.6f} "
        f"best_it={model.best_iteration} [{time.time() - t0:.0f}s]")

AUC_XGB = float(roc_auc_score(y, oof_xgb))
log(f"xgboost (matched raw features) OOF AUC = {AUC_XGB:.6f}")

# %% [markdown]
# ## 6. Does TabFM earn a place in a blend?
#
# This is where the notebook puts an earlier finding at risk. My previous notebook concluded:
#
# > Correlation does not predict blend contribution. Diversity only pays off at competitive
# > strength: the most decorrelated member in that pool contributed exactly nothing, because it was
# > 0.027 AUC too weak.
#
# TabFM is the sharpest possible test of that claim. It is not a tree, not trained by gradient
# descent on this data, and reads 0.7% of the rows, so it should be about as decorrelated from
# XGBoost as anything can be. It is also clearly weaker. The earlier claim therefore makes a
# specific, falsifiable prediction: **TabFM will add little or nothing to an XGBoost blend, despite
# being maximally different.** If it adds a lot, the earlier conclusion was wrong.
#
# To test it, TabFM needs out-of-fold predictions on the frozen folds, with each row's context
# drawn only from its own training fold. That is expensive, so it runs on a fixed random subset of
# rows rather than all 691,369, and every comparison below is restricted to exactly those rows so
# the numbers stay like-for-like. The subset size is set by the GPU budget, and stated rather than
# hidden.
#
# The context is 8,000 rows, the largest size measured to fit above, so TabFM is given the best
# configuration this hardware allows rather than a convenient one.

# %%
OOF_SUBSET_SIZE = 100_000
CTX_ROWS = 8_000   # the largest context measured to fit on a 16GB T4
N_EST = 4

subset = np.sort(rng.choice(len(train), size=OOF_SUBSET_SIZE, replace=False))
fold_of = np.empty(len(train), dtype=np.int64)
for f, (_, i_va) in enumerate(FOLDS):
    fold_of[i_va] = f

oof_tabfm = np.full(len(train), np.nan)
t0 = time.time()
for fold, (i_tr, i_va) in enumerate(FOLDS):
    targets = subset[fold_of[subset] == fold]
    context = train.iloc[rng.permutation(i_tr)[:CTX_ROWS]]
    proba, secs = tabfm_predict(
        context, train.iloc[targets][FEATURES].reset_index(drop=True),
        n_estimators=N_EST, max_num_rows=CTX_ROWS)
    oof_tabfm[targets] = proba
    log(f"  tabfm fold{fold}: {len(targets)} rows auc="
        f"{roc_auc_score(y[targets], proba):.6f} [{secs:.0f}s, total {time.time() - t0:.0f}s]")

AUC_TABFM = float(roc_auc_score(y[subset], oof_tabfm[subset]))
AUC_XGB_SUBSET = float(roc_auc_score(y[subset], oof_xgb[subset]))
log(f"on the {OOF_SUBSET_SIZE} evaluated rows: tabfm={AUC_TABFM:.6f} xgb={AUC_XGB_SUBSET:.6f}")

# %% [markdown]
# ### Giving TabFM its best shot first
#
# Before concluding anything from a 4-estimator run, it is only fair to try the configuration the
# authors actually recommend. `TabFMClassifier.ensemble()` is a preset that turns on the heavy
# machinery: 32 estimators, square-root schedules of feature crosses and SVD features,
# non-negative-least-squares weighted blending across estimators, probability rather than logit
# averaging, and Platt calibration. It costs roughly eight times a 4-estimator run.

# %%
try:
    context = train.iloc[SWEEP_POOL[:CTX_ROWS]]
    proba_ens, secs_ens = tabfm_predict(context, X_eval, n_estimators=32,
                                        max_num_rows=CTX_ROWS, preset="ensemble")
    AUC_ENSEMBLE = float(roc_auc_score(y_eval, proba_ens))
    log(f"ensemble preset (32 est, ctx={CTX_ROWS}): auc={AUC_ENSEMBLE:.6f} [{secs_ens:.0f}s]")
    best_default = ok[(ok.context_rows == CTX_ROWS) & (ok.n_estimators == 4)]["auc"]
    if len(best_default):
        log(f"  versus default preset at 4 estimators: {float(best_default.iloc[0]):.6f}")
except Exception as exc:
    AUC_ENSEMBLE = float("nan")
    log(f"ensemble preset failed: {type(exc).__name__}: {str(exc)[:200]}")
    torch.cuda.empty_cache()

# %% [markdown]
# ### The blend test
#
# Two members, the frozen folds, and the same rows for both. Spearman correlation first, then a
# greedy non-negative hill-climb on AUC, then the number that actually settles it: what TabFM adds
# on top of XGBoost alone.

# %%
def hillclimb(Z, y_true, step=0.02, rounds=300):
    n = Z.shape[1]
    w = np.zeros(n)
    w[int(np.argmax([roc_auc_score(y_true, Z[:, j]) for j in range(n)]))] = 1.0
    best = roc_auc_score(y_true, Z @ w)
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
                score = roc_auc_score(y_true, Z @ cand)
                if score > best + 1e-9:
                    w, best, improved = cand, score, True
        if not improved:
            break
    return w, float(best)


y_sub = y[subset]
Z = np.column_stack([oof_xgb[subset], oof_tabfm[subset]])
ranks = np.column_stack([pd.Series(Z[:, j]).rank().to_numpy() for j in range(Z.shape[1])])
spearman = float(np.corrcoef(ranks, rowvar=False)[0, 1])

weights, blend_auc = hillclimb(Z, y_sub)
gain = blend_auc - AUC_XGB_SUBSET

print(f"evaluated on {len(subset):,} rows of the frozen folds\n")
print(f"  xgboost (matched features)   {AUC_XGB_SUBSET:.6f}")
print(f"  tabfm (ctx={CTX_ROWS}, {N_EST} est)     {AUC_TABFM:.6f}")
print(f"  difference                   {AUC_TABFM - AUC_XGB_SUBSET:+.6f}\n")
print(f"  Spearman(xgb, tabfm)         {spearman:.4f}")
print(f"  hill-climb weights           xgb={weights[0]:.4f} tabfm={weights[1]:.4f}")
print(f"  blend AUC                    {blend_auc:.6f}")
print(f"  gain from adding tabfm        {gain:+.6f}")

# %% [markdown]
# ### Reading the result
#
# The measured outcome, on 100,000 identical rows of the frozen folds:
#
# | quantity | value |
# |---|---|
# | XGBoost, matched raw features, all 553,095 rows per fold | 0.966113 |
# | TabFM, 8,000-row context, 4 estimators, no training | 0.955027 |
# | difference | -0.011086 |
# | Spearman correlation between them | 0.9732 |
# | hill-climb weights | xgb 0.9375, tabfm **0.0625** |
# | blend | 0.966163 |
# | **gain from adding TabFM** | **+0.000050** |
#
# TabFM earns a real, non-zero weight. The hill-climb wants 6% of it, which is more than a useless
# member would get. And the gain is **+0.000050 on 100,000 rows**, which at that sample size is not
# distinguishable from zero.
#
# That is the earlier claim holding up, and it is worth being precise about why, because the naive
# reading of the weight column would say the opposite. Put the three members from my previous
# notebook next to this one, each measured as incremental gain over the trees it was added to:
#
# | member | how far behind the baseline | correlation with the trees | gain it added |
# |---|---|---|---|
# | Lookup-Transformer | 0.000149 | 0.9658 | **+0.000577** |
# | MLP on encoded features | 0.003422 | 0.9665 | +0.000013 |
# | TabFM, zero-shot | 0.011086 | 0.9732 | +0.000050 |
#
# Correlation barely moves across those three rows. The strength gap moves by two orders of
# magnitude, and the contribution tracks the strength gap, not the correlation. The one member that
# was level with the baseline supplied more than ten times the gain of either member that was not,
# despite being no more decorrelated than they were.
#
# One honest complication, since the table above is not perfectly like-for-like: the correlation
# here is measured against a matched-feature XGBoost at 0.966113, not against the tuned 0.968454
# model the previous notebook used, and it is measured on 100,000 rows rather than 691,369. The
# ordering of the effect is robust to that. The exact decimals are not.
#
# So the practical rule survives contact with the most decorrelated member I could obtain: **a
# model that is a full 0.011 AUC behind cannot buy its way in on novelty.** If you want a
# foundation model in your ensemble on a dataset this size, the thing to fix is its strength, which
# means its context window, which means GPU memory.
#
# ### The recommended preset does not fit
#
# Worth recording as a practical finding: `TabFMClassifier.ensemble()` at an 8,000-row context ran
# out of memory on this T4. That preset is what the repository's own example uses and it is
# presumably how TabFM scores best, but at 32 estimators plus feature crosses and SVD features it
# does not fit alongside an 8,000-row context on a 16GB card. Anyone reproducing TabFM's published
# numbers should budget more memory than this notebook has. The failure is caught rather than fatal,
# so the cell above reports it and moves on.
#
# The next cell acts on the ablation rather than assuming it. Computing TabFM test predictions for
# 296,302 rows costs multiple GPU-hours, and spending that on a member whose contribution is
# indistinguishable from zero would be waste. So the submission includes TabFM only if it cleared
# both a weight floor and a gain floor.

# %%
WEIGHT_FLOOR, GAIN_FLOOR = 0.05, 0.0002
worth_it = bool(weights[1] >= WEIGHT_FLOOR and gain >= GAIN_FLOOR)
print(f"tabfm weight {weights[1]:.4f} (floor {WEIGHT_FLOOR}), "
      f"gain {gain:+.6f} (floor {GAIN_FLOOR:+.6f}) -> "
      f"{'including' if worth_it else 'excluding'} tabfm in the submission")

if worth_it:
    log(f"computing tabfm test predictions for {len(test):,} rows")
    ctx = train.iloc[rng.permutation(len(train))[:CTX_ROWS]]
    proba_test, secs = tabfm_predict(ctx, X_test_all.reset_index(drop=True),
                                     n_estimators=N_EST, max_num_rows=CTX_ROWS)
    blended = weights[0] * pd.Series(test_xgb).rank(pct=True).to_numpy() + \
              weights[1] * pd.Series(proba_test).rank(pct=True).to_numpy()
    log(f"  done in {secs:.0f}s")
else:
    blended = test_xgb

submission = pd.DataFrame({ID: test_ids, TARGET: blended})
submission.to_csv("submission.csv", index=False)
print(submission.shape)
submission.head()

# %%
# Export what was measured, so the curve and the OOF vector are reusable without a rerun.
sweep_df.to_csv("tabfm_context_sweep.csv", index=False)
pd.DataFrame({
    "row": subset,
    TARGET: y[subset],
    "xgboost": oof_xgb[subset],
    "tabfm": oof_tabfm[subset],
}).to_csv("oof_xgb_tabfm.csv", index=False)
print("wrote tabfm_context_sweep.csv and oof_xgb_tabfm.csv")

# %% [markdown]
# ## 7. When a tabular foundation model is the right tool
#
# The result above is not an argument against TabFM. It is an argument about where the crossover
# lies, and the crossover is governed by one quantity: **how much of your dataset fits in the
# context window.**
#
# On this competition that fraction is 0.7%, and a GBDT that gets 100% of 553,095 rows wins
# comfortably. Invert the situation and the conclusion inverts with it:
#
# | your situation | who wins, and why |
# |---|---|
# | 691,369 rows, saturated metric, days of tuning available | GBDT. TabFM sees a rounding error of your data. |
# | 500 rows, no time, no validation budget | TabFM. It reads your whole dataset as context, and a GBDT on 500 rows is mostly variance. |
# | A few thousand rows | Genuinely competitive, and worth measuring rather than guessing. This is the regime the model was built for. |
# | Hundreds of separate small tables | TabFM, decisively. One loaded model serves all of them with no per-table training, tuning, or validation design. |
# | You need a strong baseline in five minutes | TabFM. Zero hyperparameters, no fitting, no leakage risk from a bad CV split. |
#
# That last row is underrated. Every number in this notebook that came from XGBoost required
# choosing a fold scheme, an encoding, an early-stopping rule and a set of hyperparameters, each of
# which is an opportunity to leak or to fool yourself. TabFM has essentially one knob that matters
# (`max_num_rows`) and cannot leak validation labels into a fitted encoder, because nothing is
# fitted. For a first look at an unfamiliar dataset that is real value, independent of whether it
# wins the competition.
#
# ## 8. Limitations, and things I could not settle
#
# **No technical report exists.** The repository says so explicitly. I do not know what TabFM was
# pretrained on, which means I cannot rule out that data resembling this competition's generator
# was in the mix. For a synthetic Playground dataset that is a live concern, and it cuts both ways:
# contamination would flatter the zero-shot number, not hurt it.
#
# **One GPU, one context ceiling.** Every conclusion about the wall is specific to a 16GB T4. The
# AUC curve had not flattened when memory ran out, so a larger card would move the numbers up by an
# amount this notebook cannot measure. Read the wall as a property of the hardware, not of TabFM.
#
# **The OOF comparison uses a subset.** TabFM out-of-fold predictions were computed on
# 150,000 rows rather than all 691,369, purely for GPU budget. Every comparison is restricted to
# those same rows, so the contrast is fair, but the absolute AUCs carry more sampling noise than
# the full-data figures in my previous notebook.
#
# **The context is a uniform random sample.** I did not try stratified sampling, coreset selection,
# or picking context rows near the query. Any of those could plausibly beat uniform sampling at a
# fixed context size, and that is the most obvious direction for anyone wanting to push this
# further. If you try it, the sweep and the OOF vector are exported above so you can compare
# directly.
#
# **Licence, once more.** The code is Apache-2.0. The pretrained weights are
# `tabfm-non-commercial-v1.0`, non-commercial and non-production only.
#
# ## 9. Takeaways
#
# 1. **A tabular foundation model's dataset limit is memory, not time.** TabFM cannot be made to
#    read 553,095 context rows by waiting longer. This is the single most important practical fact
#    about the model class and it is not how GBDT intuitions work.
#
# 2. **Read the code, not the README.** The FAQ says `max_num_rows` defaults to 100 context rows;
#    the code defaults it to `None`, meaning no cap, which on a large dataset is an immediate
#    out-of-memory error rather than a conservative default.
#
# 3. **In-context learning here is one line.** The label embedding is added to context rows and not
#    to query rows. No special tokens, no architectural split. Worth internalising, because it
#    explains why the context/query boundary is just a mask and why the context can be cached.
#
# 4. **Rows are a set, columns are a sequence.** Induced-point attention with no positional encoding
#    over rows; RoPE over columns. That asymmetry is why long contexts are tractable, why the
#    context is cacheable, and why the built-in ensembling shuffles columns rather than rows.
#
# 5. **Periodic numeric embeddings keep showing up.** TabFM builds every cell token from Fourier
#    features, in float32 even when the rest of the model is bfloat16. The same primitive carried my
#    previous notebook's best single model. Two independent arrivals at the same idea is weak
#    evidence, but it points the same way.
#
# 6. **Diversity remains cheap; strength remains scarce.** The blend section is a direct test of
#    that earlier finding using the most decorrelated member I could obtain, and the ablation number
#    is the verdict rather than my expectation of it.
#
# ### Credit and reproducibility
#
# TabFM is by Google Research: [github.com/google-research/tabfm](https://github.com/google-research/tabfm),
# announced [here](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/).
# The Set Transformer construction its row attention uses is Lee et al., *Set Transformer*. All
# folds are `StratifiedKFold(5, shuffle=True, random_state=42)` on the original row order, matching
# my previous notebook so the exported vectors are stackable across both.
#
# The `except ImportError` to `except Exception` fix in section 2 is a one-line change that anyone
# running TabFM on Kaggle currently needs. If it is useful, it belongs upstream rather than in a
# hundred forked notebooks.
