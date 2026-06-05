# Databricks notebook source
"""
v9 — Codex #1: logit-rank blend of own submissions v5.2 + v5.4 + v8.

Recipe (Codex's prescription):
  1. Read v5.2, v5.4, v8 submission CSVs (probability vectors, id-aligned).
  2. Convert each to percentile rank.
  3. Logit transform with clip 1e-5.
  4. Linear combo:  0.55*z_v5.2 + 0.25*z_v5.4 + 0.20*z_v8
  5. Sigmoid back to rank space.
  6. Remap blended ranks to v5.2's value distribution
     (out[argsort(blended_rank)] = sort(v5.2)).
  7. Submit. Auto-poll Kaggle for score.

Why this works:
  - AUC is rank-only, so working in rank-space preserves AUC sensitivity
    where it matters (the tails).
  - Logit stretches percentiles 0.99 vs 0.999 vastly more than 0.50 vs 0.51,
    weighting tail ordering correctly.
  - Remapping to v5.2 distribution preserves calibration of best single sub.
  - Three highly-correlated probability vectors with slightly different
    decision surfaces → independent error patterns avg out.

Expected lift per Codex: +0.0003 to +0.0012. v5.2 LB = 0.94924.
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
    import os, sys, subprocess, json, time, warnings
    warnings.filterwarnings("ignore")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "numpy", "pandas", "scikit-learn", "kaggle"])

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    DATA_DIR = "/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops"
    SUBS = {
        "v5.2": f"{DATA_DIR}/submission_v5_2_optuna.csv",
        "v5.4": f"{DATA_DIR}/submission_v5_4_blend3_full.csv",
        "v8":   f"{DATA_DIR}/submission_v8_seed_ensemble.csv",
    }
    LB_SCORES = {"v5.2": 0.94924, "v5.4": 0.94896, "v8": 0.94867}
    WEIGHTS = {"v5.2": 0.55, "v5.4": 0.25, "v8": 0.20}
    ANCHOR = "v5.2"

    # ─── Load + align by id ───
    dfs = {}
    for name, path in SUBS.items():
        df = pd.read_csv(path)
        assert {"id", "PitNextLap"}.issubset(df.columns), f"{name} cols={df.columns.tolist()}"
        dfs[name] = df.sort_values("id").reset_index(drop=True)
    ids = dfs[ANCHOR]["id"].values
    for name, df in dfs.items():
        assert np.array_equal(df["id"].values, ids), f"{name} id mismatch"
    print(f"Loaded {len(dfs)} subs, aligned on {len(ids)} ids.")

    preds = {name: np.clip(df["PitNextLap"].values.astype(float), 1e-7, 1 - 1e-7)
             for name, df in dfs.items()}

    # ─── Pairwise correlations (Pearson on probs + Spearman via ranks) ───
    print("\nPairwise correlations:")
    names = list(preds.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = preds[names[i]], preds[names[j]]
            pe = float(np.corrcoef(a, b)[0, 1])
            ra = pd.Series(a).rank().values
            rb = pd.Series(b).rank().values
            sp = float(np.corrcoef(ra, rb)[0, 1])
            print(f"  {names[i]} vs {names[j]}: pearson={pe:.5f}  spearman={sp:.5f}")

    # ─── Logit-rank blend ───
    def percentile_rank(x):
        n = len(x)
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        # uniform [1/(n+1), n/(n+1)] to avoid 0/1 hitting clip
        ranks[order] = np.arange(1, n + 1) / (n + 1)
        return ranks

    def logit(p, eps=1e-5):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    rank_pcts = {name: percentile_rank(preds[name]) for name in names}
    logits = {name: logit(rank_pcts[name]) for name in names}

    z_blend = np.zeros_like(logits[ANCHOR])
    for name, w in WEIGHTS.items():
        z_blend += w * logits[name]
    rank_blend = sigmoid(z_blend)  # in (0,1), monotonic in blended logit

    # Remap to anchor (v5.2) value distribution
    order = np.argsort(rank_blend, kind="mergesort")
    sorted_anchor = np.sort(preds[ANCHOR])
    out = np.empty_like(preds[ANCHOR])
    out[order] = sorted_anchor
    out = np.clip(out, 1e-7, 1 - 1e-7)

    # ─── Sanity: AUC vs each input (using each input as pseudo-label is meaningless;
    #     just print rank-agreement of the blend vs each input)
    print("\nBlend rank-agreement (spearman) vs inputs:")
    rb_rank = pd.Series(out).rank().values
    for name in names:
        sp = float(np.corrcoef(rb_rank, pd.Series(preds[name]).rank().values)[0, 1])
        print(f"  blend vs {name}: spearman={sp:.5f}")

    # Distribution sanity
    print("\nDistribution check:")
    print(f"  anchor mean={preds[ANCHOR].mean():.6f}  std={preds[ANCHOR].std():.6f}  "
          f"min={preds[ANCHOR].min():.6f}  max={preds[ANCHOR].max():.6f}")
    print(f"  blend  mean={out.mean():.6f}  std={out.std():.6f}  "
          f"min={out.min():.6f}  max={out.max():.6f}")
    # By construction blend has the same value multiset as anchor — only ordering changes
    assert np.allclose(np.sort(out), np.sort(preds[ANCHOR])), \
        "Remap broken — blend values should be a permutation of anchor"

    SUB_PATH = f"{DATA_DIR}/submission_v9_logit_rank.csv"
    pd.DataFrame({"id": ids, "PitNextLap": out}).to_csv(SUB_PATH, index=False)
    print(f"\nsubmission -> {SUB_PATH}")

    # Also compute simple rank-average as a comparison candidate (Codex suggested testing both)
    rank_avg_z = sum(WEIGHTS[n] * rank_pcts[n] for n in names)  # weighted avg of percentile ranks
    ra_order = np.argsort(rank_avg_z, kind="mergesort")
    ra_out = np.empty_like(preds[ANCHOR])
    ra_out[ra_order] = sorted_anchor
    SUB_PATH_RA = f"{DATA_DIR}/submission_v9b_rank_avg.csv"
    pd.DataFrame({"id": ids, "PitNextLap": np.clip(ra_out, 1e-7, 1 - 1e-7)}
                 ).to_csv(SUB_PATH_RA, index=False)
    print(f"alt sub  -> {SUB_PATH_RA}")

    # Rank-disagreement between logit-rank and rank-avg variants
    sp_methods = float(np.corrcoef(pd.Series(out).rank().values,
                                   pd.Series(ra_out).rank().values)[0, 1])
    print(f"\nlogit-rank vs simple-rank-avg spearman: {sp_methods:.6f}")

    # ─── Kaggle submit ───
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    TOK = "KGAT_d7a0018cbb8131f82d3dd175d2e4cdb6"
    TP = os.path.expanduser("~/.kaggle/access_token")
    with open(TP, "w") as f: f.write(TOK)
    os.chmod(TP, 0o600)
    os.environ["KAGGLE_API_TOKEN"] = TOK
    COMP = "playground-series-s6e5"
    MSG = ("v9 logit-rank blend: percentile rank -> logit (clip 1e-5) -> "
           "weighted avg [0.55*v5.2 + 0.25*v5.4 + 0.20*v8] -> sigmoid -> "
           "remap to v5.2 distribution. Anchor=v5.2 LB 0.94924.")

    print(f"\nSubmitting v9 logit-rank to Kaggle...")
    r = subprocess.run(
        ["kaggle", "competitions", "submit", "-c", COMP, "-f", SUB_PATH, "-m", MSG],
        capture_output=True, text=True,
    )
    print("STDOUT:", r.stdout); print("STDERR:", r.stderr); print("rc:", r.returncode)

    print("\nPolling for v9 score...")
    final_v9 = ""
    for i in range(30):
        time.sleep(15)
        rs = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP],
                            capture_output=True, text=True)
        for line in rs.stdout.split("\n"):
            if "submission_v9_logit_rank.csv" in line:
                print(f"t={(i+1)*15}s: {line}")
                final_v9 = line
                if line.split() and line.split()[-1].startswith("0."):
                    break
        if final_v9 and final_v9.split()[-1].startswith("0."):
            break

    print("\nDONE")
    print(f"v9 final: {final_v9}")
    print("\nNOTE: v9b (simple rank-avg) NOT submitted — saved locally for optional follow-up sub.")
except BaseException as _err:
    print(f"FATAL: {_err!r}")
    raise
finally:
    try:
        dbutils.notebook.exit(_BUF.getvalue()[-7000:])
    except Exception:
        pass
