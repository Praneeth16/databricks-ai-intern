# Databricks notebook source
"""
v11 — Logit-rank blend with v10 hazard model providing diversity.

Inputs: v5.2 (anchor, LB 0.94924), v5.4 (LB 0.94896), v8 (LB 0.94867),
        v10 (LB 0.94852 standalone, BUT spearman ~0.994 vs others — diverse).

Weighting logic:
  - v5.2 anchor heavy weight (best LB, highest trust).
  - v5.4, v8 medium weights (similar geometry to v5.2 but slight variation).
  - v10 LOW weight, diverse — small dose to perturb tail ordering.

Two candidate weight schemes:
  v11a: 0.55*v5.2 + 0.15*v5.4 + 0.15*v8 + 0.15*v10  (anchor-heavy)
  v11b: 0.45*v5.2 + 0.20*v5.4 + 0.15*v8 + 0.20*v10  (more diversity injection)

Submit v11a only (1 daily quota). v11b saved locally for later if v11a lifts.

Logit-rank recipe (Codex): rank -> logit(clip 1e-5) -> weighted sum
-> sigmoid -> remap to v5.2 value distribution.
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
    import os, sys, subprocess, time, json, warnings
    warnings.filterwarnings("ignore")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "numpy", "pandas", "kaggle"])

    import numpy as np
    import pandas as pd

    DATA_DIR = "/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops"
    SUBS = {
        "v5.2": f"{DATA_DIR}/submission_v5_2_optuna.csv",
        "v5.4": f"{DATA_DIR}/submission_v5_4_blend3_full.csv",
        "v8":   f"{DATA_DIR}/submission_v8_seed_ensemble.csv",
        "v10":  f"{DATA_DIR}/submission_v10_hazard.csv",
    }
    ANCHOR = "v5.2"

    WEIGHTS_A = {"v5.2": 0.55, "v5.4": 0.15, "v8": 0.15, "v10": 0.15}
    WEIGHTS_B = {"v5.2": 0.45, "v5.4": 0.20, "v8": 0.15, "v10": 0.20}

    # ─── Load + align ───
    dfs = {}
    for name, path in SUBS.items():
        df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
        assert {"id", "PitNextLap"}.issubset(df.columns)
        dfs[name] = df
    ids = dfs[ANCHOR]["id"].values
    for name, df in dfs.items():
        assert np.array_equal(df["id"].values, ids), f"{name} id mismatch"
    preds = {name: np.clip(df["PitNextLap"].values.astype(float), 1e-7, 1 - 1e-7)
             for name, df in dfs.items()}
    print(f"Loaded {len(dfs)} subs, aligned on {len(ids)} ids.")

    print("\nPairwise spearman correlations:")
    names = list(preds.keys())
    rks = {n: pd.Series(preds[n]).rank().values for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sp = float(np.corrcoef(rks[names[i]], rks[names[j]])[0, 1])
            print(f"  {names[i]} vs {names[j]}: spearman={sp:.5f}")

    def percentile_rank(x):
        n = len(x); order = np.argsort(x, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1) / (n + 1)
        return ranks

    def logit(p, eps=1e-5):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    rank_pcts = {n: percentile_rank(preds[n]) for n in names}
    logits = {n: logit(rank_pcts[n]) for n in names}
    sorted_anchor = np.sort(preds[ANCHOR])

    def logit_rank_blend(weights):
        z = np.zeros_like(logits[ANCHOR])
        for n, w in weights.items():
            z += w * logits[n]
        rb = sigmoid(z)
        order = np.argsort(rb, kind="mergesort")
        out = np.empty_like(preds[ANCHOR])
        out[order] = sorted_anchor
        return np.clip(out, 1e-7, 1 - 1e-7)

    v11a = logit_rank_blend(WEIGHTS_A)
    v11b = logit_rank_blend(WEIGHTS_B)

    # Rank agreement of each variant vs inputs
    for label, vec in [("v11a", v11a), ("v11b", v11b)]:
        print(f"\n{label} weights={WEIGHTS_A if label == 'v11a' else WEIGHTS_B}")
        rb_rank = pd.Series(vec).rank().values
        for n in names:
            sp = float(np.corrcoef(rb_rank, rks[n])[0, 1])
            print(f"  {label} vs {n}: spearman={sp:.5f}")

    # Save both
    SUB_A = f"{DATA_DIR}/submission_v11a_logit_rank_hazard.csv"
    SUB_B = f"{DATA_DIR}/submission_v11b_logit_rank_hazard.csv"
    pd.DataFrame({"id": ids, "PitNextLap": v11a}).to_csv(SUB_A, index=False)
    pd.DataFrame({"id": ids, "PitNextLap": v11b}).to_csv(SUB_B, index=False)
    print(f"\nsubmission -> {SUB_A}")
    print(f"alt sub    -> {SUB_B}")

    # ─── Kaggle submit v11a ───
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    TOK = "KGAT_d7a0018cbb8131f82d3dd175d2e4cdb6"
    TP = os.path.expanduser("~/.kaggle/access_token")
    with open(TP, "w") as f: f.write(TOK)
    os.chmod(TP, 0o600)
    os.environ["KAGGLE_API_TOKEN"] = TOK
    COMP = "playground-series-s6e5"
    MSG = (f"v11a logit-rank: 0.55*v5.2 + 0.15*v5.4 + 0.15*v8 + 0.15*v10. "
           f"v10 = hazard-feature XGB+CB (spearman ~0.994 vs others — diversity). "
           f"Anchor v5.2 LB 0.94924.")

    print(f"\nSubmitting v11a...")
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", COMP,
                        "-f", SUB_A, "-m", MSG], capture_output=True, text=True)
    print("STDOUT:", r.stdout); print("rc:", r.returncode)

    print("\nPolling for v11a score...")
    final = ""
    for i in range(30):
        time.sleep(15)
        rs = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP],
                            capture_output=True, text=True)
        for line in rs.stdout.split("\n"):
            if "submission_v11a_logit_rank_hazard.csv" in line:
                print(f"t={(i+1)*15}s: {line}")
                final = line
                if line.split() and line.split()[-1].startswith("0."): break
        if final and final.split()[-1].startswith("0."): break

    print("\nDONE")
    print(f"v11a final: {final}")
except BaseException as _err:
    print(f"FATAL: {_err!r}")
    raise
finally:
    try:
        dbutils.notebook.exit(_BUF.getvalue()[-7000:])
    except Exception:
        pass
