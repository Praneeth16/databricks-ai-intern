#!/usr/bin/env python3
"""Audit the public weak-label supply chain for RSNA Knee Abnormality Detection.

Only 58 of the 4,407 training studies carry labels. Everyone else derives labels
from the free-text radiology report, and a dozen independent label sets are now
published as Kaggle Datasets and merged into each other. This measures them.

Every number the notebook prints comes from here. `--assert` turns the findings
into pass/fail checks so a change in the upstream datasets is noticed rather than
silently reported as fact.

Runs on the CSVs alone. No pixels, no GPU, no network.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

# Every published label set we could find, keyed by the Kaggle dataset that ships
# it. `regex` marks a rule-based labeler; the rest are read by a language model.
LABEL_SETS: dict[str, tuple[str, str]] = {
    "pilkwang_v1_regex": ("pilkwang/rsna-knee-report-labels", "report_labels_v1.csv"),
    "pilkwang_v2_llm": ("pilkwang/rsna-knee-llm-labels", "report_labels_v2.csv"),
    "stevenleehans_full": ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_full.csv"),
    "stevenleehans_v4": ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv"),
    "lixin73_gpt56sol": ("lixin73/rsna-knee-llm-report-labels-sol56", "labels_llm_gpt56sol.csv"),
    "flight_hybrid": ("flight0234/rsna-knee-hybrid-report-labels", "report_labels_v4hybrid.csv"),
    "yunus_3src": ("yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels", "report_labels_v3.csv"),
    "yunus_4src": ("yunusgmsoy/rsna-knee-llm-labels-4-source-merged", "report_labels_v5.csv"),
}

# Mention lexicon over the Latin-script languages in the corpus. Deliberately not
# extended to Greek or Bulgarian: the gap is the point, and Finding 2 measures it.
MENTION = {
    "ACL": [r"\bacl\b", r"anterior cruciate", r"cruzado anterior", r"\blca\b",
            r"voorste kruisband", r"\bvkb\b", r"vorderes kreuzband",
            r"croise anterieur", r"crociato anteriore", r"on capraz"],
    "MCL": [r"\bmcl\b", r"medial collateral", r"colateral (medial|interno)",
            r"mediale collaterale", r"innenband", r"collateral medial",
            r"collateral interne", r"collaterale mediale", r"\blcm\b"],
    "Medial Meniscus": [r"medial meniscus", r"menisc\w* (interno|medial)",
                        r"mediale meniscus", r"innenmeniskus", r"menisque (interne|medial)",
                        r"binnenmeniscus", r"medial menisk", r"ic medial"],
    "Lateral Meniscus": [r"lateral meniscus", r"menisc\w* (externo|lateral)",
                         r"laterale meniscus", r"aussenmeniskus", r"menisque (externe|lateral)",
                         r"buitenmeniscus", r"lateral menisk", r"dis medial"],
    "Medial OA": [r"medial (compartment )?(osteoarthritis|oa|degenerat)",
                  r"artrosis (femorotibial )?(medial|interna)", r"gonartros\w* medial",
                  r"mediale gonarthrose", r"arthrose femoro-?tibiale interne",
                  r"medial joint space narrowing", r"artrose mediaal"],
    "Lateral OA": [r"lateral (compartment )?(osteoarthritis|oa|degenerat)",
                   r"artrosis (femorotibial )?(lateral|externa)", r"laterale gonarthrose",
                   r"arthrose femoro-?tibiale externe", r"lateral joint space narrowing",
                   r"artrose lateraal"],
    "PF OA": [r"patellofemoral", r"femoropatelar", r"retropatell", r"patello-?femoral",
              r"femoro-?patellaire", r"chondropath\w* patell"],
    "Effusion": [r"effusion", r"derrame", r"\bhydrops\b", r"gelenkerguss", r"\berguss\b",
                 r"epanchement", r"vocht in het gewricht", r"versamento", r"joint fluid",
                 r"efuzyon", r"eklem sivisi"],
    "Synovitis": [r"synovit", r"sinovit", r"synovial (thickening|proliferation)"],
    "Baker's": [r"\bbaker", r"popliteal cyst", r"poplitea\w* cyst", r"quiste de baker",
                r"bakerzyste", r"kyste de baker", r"cisti di baker"],
    "Contusion": [r"contusion", r"contusio", r"bone (bruise|contusion)",
                  r"edema (oseo|osseo|medular)", r"bone marrow edema",
                  r"beenmerg\w*oedeem", r"knochenmarkodem", r"kneuzing", r"oedeme osseux"],
    "Fracture": [r"fractur", r"fractuur", r"fraktur", r"\bfissur", r"\bfx\b",
                 r"frattura", r"breuk", r"kirik"],
}

NEGATION = re.compile(
    r"\bno\b|\bnot\b|\bwithout\b|\bnegative\b|\bintact\b|\bnormal\b|\bsin\b|\bgeen\b"
    r"|\bkein|\bpas de\b|\bnessun|\bunremarkable\b|\bpreserved\b|\bno evidence\b"
)


# --------------------------------------------------------------------------- io

def find_root() -> pathlib.Path:
    """Competition data mounts under /kaggle/input/competitions/<slug> on Kaggle."""
    for c in (
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/tmp/rsnaknee",
    ):
        if (pathlib.Path(c) / "train.csv").exists():
            return pathlib.Path(c)
    raise FileNotFoundError("train.csv not found in any known location")


def _bounded_find(root: pathlib.Path, name: str) -> pathlib.Path | None:
    """Bounded glob. An rglob under /kaggle/input walks the competition mount and its
    ~700,000 DICOM files, which costs minutes per lookup."""
    for pat in (name, f"*/{name}", f"*/*/{name}"):
        for hit in root.glob(pat):
            if "competitions" not in hit.parts:
                return hit
    return None


def load_label_sets(roots: list[pathlib.Path]) -> dict[str, pd.DataFrame]:
    """Find each published label set under any of `roots`, keyed by our short name."""
    out: dict[str, pd.DataFrame] = {}
    for name, (_slug, fname) in LABEL_SETS.items():
        for root in roots:
            hit = _bounded_find(root, fname)
            if hit is None:
                continue
            d = pd.read_csv(hit)
            if not all(c in d.columns for c in LABELS):
                continue
            out[name] = (
                d.drop_duplicates("StudyInstanceUID")
                .set_index("StudyInstanceUID")[LABELS]
                .astype(float)
            )
            break
    return out


# --------------------------------------------------------------- language (F2)

def _strip(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def script_evidence(text: str) -> str | None:
    """Independent of any statistical detector: what alphabet is this written in?

    Greek and Bulgarian are decidable from the Unicode block alone, which is far
    stronger evidence than a trained detector's guess. Turkish and Croatian get a
    diacritic test, which is suggestive rather than decisive.
    """
    greek = sum("Ͱ" <= c <= "Ͽ" for c in text)
    cyril = sum("Ѐ" <= c <= "ӿ" for c in text)
    letters = sum(c.isalpha() for c in text) or 1
    if greek / letters > 0.3:
        return "el"
    if cyril / letters > 0.3:
        return "bg"
    if any(c in text for c in "ığşİĞŞ"):
        return "tr?"
    if any(c in text for c in "čćžšđČĆŽŠĐ"):
        return "hr?"
    return None


def detect_languages(reports: pd.Series) -> pd.DataFrame:
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0

    def guess(t: str) -> str:
        try:
            return detect_langs(t)[0].lang
        except Exception:
            return "err"

    return pd.DataFrame(
        {"lang": reports.map(guess), "script": reports.map(script_evidence)},
        index=reports.index,
    )


# ------------------------------------------------------------------- leak (F4)

def leak_test(gold: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Do the annotated labels appear verbatim in a published weak-label set?

    Signature: on the annotated rows every value is exactly 0.0 or 1.0 and equals
    the annotation, while on the other rows effectively none are. Substituting real
    labels where real labels exist is a reasonable training choice -- it only breaks
    the moment those same studies are used to measure anything.
    """
    on = labels.reindex(gold.index).dropna(how="all")
    if on.empty:
        return {"testable": False}
    off = labels.loc[~labels.index.isin(gold.index)]
    binary_on = float(np.isin(on.values, [0.0, 1.0]).mean())
    binary_off = float(np.isin(off.values, [0.0, 1.0]).mean())
    match = float((on.values == gold.reindex(on.index).values).mean())
    return {
        "testable": True,
        "frac_exactly_binary_on_annotated": binary_on,
        "frac_exactly_binary_elsewhere": binary_off,
        "exact_match_to_annotations": match,
        "distinct_values_elsewhere": int(len(np.unique(off.values))),
        "leaked": bool(binary_on > 0.999 and match > 0.999 and binary_off < 0.01),
    }


# -------------------------------------------------------- auc + bootstrap (F3/F9)

def auc_with_ci(y: np.ndarray, p: np.ndarray, n_boot: int, rng: np.random.Generator
                ) -> tuple[float, float, float]:
    point = roc_auc_score(y, p)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    if not boots:
        return point, np.nan, np.nan
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def score_label_sets(gold: pd.DataFrame, sets: dict[str, pd.DataFrame], n_boot: int
                     ) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for name, d in sets.items():
        common = gold.index.intersection(d.index)
        per, lo_s, hi_s = {}, [], []
        for c in LABELS:
            y = gold.loc[common, c].values.astype(int)
            p = d.loc[common, c].values
            if len(np.unique(y)) < 2:
                continue
            p = np.nan_to_num(p, nan=float(np.nanmean(p)))
            a, lo, hi = auc_with_ci(y, p, n_boot, rng)
            per[c] = a
            lo_s.append(lo)
            hi_s.append(hi)
        rows.append(
            {"label_set": name, "n": len(common), "mean_auc": float(np.mean(list(per.values()))),
             "ci_lo": float(np.mean(lo_s)), "ci_hi": float(np.mean(hi_s)), **per}
        )
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)


# ------------------------------------------------------ cross-labeler agreement (F5)

def agreement_by_language(sets: dict[str, pd.DataFrame], lang: pd.Series,
                          min_n: int = 40) -> pd.DataFrame:
    """Mean pairwise rank correlation between independent labelers, per language.

    Needs no annotations, so unlike the 58-study score this covers all 4,407
    studies. Where independent labelers disagree, at least one is wrong; where they
    agree while all reading the same hard language, they may be wrong together.
    """
    idx = None
    for d in sets.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    ranked = {k: d.loc[idx].rank(pct=True) for k, d in sets.items()}
    names = list(ranked)
    langs = lang.reindex(idx)

    rows = []
    for lg, grp in langs.groupby(langs):
        if len(grp) < min_n:
            continue
        per = {}
        for c in LABELS:
            cors = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a = ranked[names[i]].loc[grp.index, c].values
                    b = ranked[names[j]].loc[grp.index, c].values
                    if a.std() > 0 and b.std() > 0:
                        cors.append(np.corrcoef(a, b)[0, 1])
            per[c] = float(np.mean(cors)) if cors else np.nan
        worst = min(per, key=lambda k: per[k] if per[k] == per[k] else 9)
        rows.append({"lang": lg, "n": len(grp), "agreement": float(np.nanmean(list(per.values()))),
                     "worst_finding": worst, "worst_agreement": per[worst], **per})
    return pd.DataFrame(rows).sort_values("agreement").reset_index(drop=True)


# --------------------------------------------------- lexicon coverage + silence (F2/F6)

def mention_matrix(reports: pd.Series) -> pd.DataFrame:
    flat = reports.map(_strip)
    return pd.DataFrame(
        {k: flat.str.contains("|".join(v), regex=True) for k, v in MENTION.items()},
        index=reports.index,
    )


def asserted_positive(reports: pd.Series) -> pd.DataFrame:
    """A mention in a sentence carrying no negation cue."""
    compiled = {k: re.compile("|".join(v)) for k, v in MENTION.items()}
    out = {}
    for k, pat in compiled.items():
        hits = []
        for t in reports.map(_strip):
            found = False
            for sent in re.split(r"[.;\n]", t):
                if pat.search(sent) and not NEGATION.search(sent):
                    found = True
                    break
            hits.append(found)
        out[k] = hits
    return pd.DataFrame(out, index=reports.index)


def lexicon_coverage(mentions: pd.DataFrame, lang: pd.Series) -> pd.DataFrame:
    """How often the Latin-script lexicon fires at all, split by language."""
    any_hit = mentions.any(axis=1)
    g = pd.DataFrame({"lang": lang.reindex(mentions.index), "hit": any_hit})
    return (
        g.groupby("lang")
        .agg(n=("hit", "size"), any_mention_rate=("hit", "mean"))
        .sort_values("any_mention_rate")
        .reset_index()
    )


def load_verdicts(label_roots: list[pathlib.Path]) -> pd.DataFrame | None:
    """The three-way YES / NO / UNK verdicts shipped by pilkwang/rsna-knee-llm-labels.

    This is the only published set that distinguishes "the report says absent" from
    "the report does not say", which is exactly the distinction the analysis needs.
    Every other set collapses UNK into a low probability, which is what makes the
    silent entries look like negatives.
    """
    for root in label_roots:
        hit = _bounded_find(root, "report_labels_v2.csv")
        if hit is None:
            continue
        d = pd.read_csv(hit).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        cols = {c: c.replace("__verdict", "") for c in d.columns if c.endswith("__verdict")}
        if len(cols) == len(LABELS):
            return d[list(cols)].rename(columns=cols)[LABELS]
    return None


def silence_analysis(gold: pd.DataFrame, mentions: pd.DataFrame,
                     verdicts: pd.DataFrame | None,
                     negatives: pd.DataFrame | None) -> pd.DataFrame:
    """Three different questions that all look like "silence" if you squint.

    1. `no_mention_regex` -- the Latin-script lexicon never fires. Confounded: a
       lexicon gap and a radiologist's omission leave the same trace, and Finding 2b
       shows the lexicon is blind to Bulgarian outright.
    2. `unk` -- an LLM read the report and reported it could not tell. This is the
       honest measure of what the report does not say.
    3. `neg` -- the best-scoring published set concluded the finding is absent. This
       is that set's false-negative rate, not silence at all.

    Reporting one of these as if it were the others is how the size of this problem
    gets misstated, in either direction.
    """
    rows = []
    for c in LABELS:
        y = gold[c].values.astype(int)
        rec = {"finding": c, "prevalence": float(y.mean())}
        masks = {"no_mention_regex": ~mentions.reindex(gold.index)[c].fillna(False).astype(bool).values}
        if verdicts is not None:
            masks["unk"] = (verdicts.reindex(gold.index)[c] == "UNK").fillna(False).values
        if negatives is not None:
            masks["neg"] = ~negatives.reindex(gold.index)[c].fillna(False).astype(bool).values
        for tag, m in masks.items():
            rec[f"rate_{tag}"] = float(m.mean())
            rec[f"p_pos_given_{tag}"] = float(y[m].mean()) if m.any() else np.nan
            rec[f"n_{tag}"] = int(m.sum())
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------- who the annotated studies are (F7)

def annotated_representativeness(train: pd.DataFrame, series: pd.DataFrame,
                                 gold_ids: pd.Index) -> dict:
    from scipy import stats

    is_gold = train.StudyInstanceUID.isin(gold_ids)
    g, w = train[is_gold], train[~is_gold]
    gl, wl = g.Report.str.len(), w.Report.str.len()
    counts = series.groupby("StudyInstanceUID").size()
    gc = counts.reindex(g.StudyInstanceUID).dropna()
    wc = counts.reindex(w.StudyInstanceUID).dropna()
    return {
        "n_annotated": int(is_gold.sum()),
        "n_report_only": int((~is_gold).sum()),
        "report_len_annotated_mean": float(gl.mean()),
        "report_len_other_mean": float(wl.mean()),
        "report_len_p": float(stats.mannwhitneyu(gl, wl).pvalue),
        "series_per_study_annotated": float(gc.mean()),
        "series_per_study_other": float(wc.mean()),
        "series_per_study_p": float(stats.mannwhitneyu(gc, wc).pvalue),
    }


# ------------------------------------------------------------------------ main

def run(root: pathlib.Path, label_roots: list[pathlib.Path], n_boot: int) -> dict:
    train = pd.read_csv(root / "train.csv")
    series = pd.read_csv(root / "train_series.csv")
    gold = (
        train[train[LABELS].notna().any(axis=1)]
        .set_index("StudyInstanceUID")[LABELS]
        .astype(int)
    )
    reports = train.set_index("StudyInstanceUID").Report

    print(f"studies={len(train)}  series={len(series)}  annotated={len(gold)}")

    langs = detect_languages(reports)
    lang = langs.lang
    print("\n--- Finding 2: report languages ---")
    print(lang.value_counts().to_string())
    conf = langs.dropna(subset=["script"])
    agree = (
        conf.script.str.rstrip("?") == conf.lang
    ).mean() if len(conf) else float("nan")
    print(f"script-based cross-check available for {len(conf)} reports; "
          f"agrees with langdetect on {agree:.1%}")

    sets = load_label_sets(label_roots)
    print(f"\nlabel sets found: {len(sets)} -> {list(sets)}")

    print("\n--- Finding 4: does any published set contain the annotations? ---")
    leaks = {}
    for name, d in sets.items():
        leaks[name] = leak_test(gold, d)
        if leaks[name].get("leaked"):
            print(f"  LEAKED  {name:22s} {LABEL_SETS[name][0]}/{LABEL_SETS[name][1]}")
    for name, r in leaks.items():
        if r.get("testable") and not r.get("leaked"):
            print(f"  clean   {name:22s} binary-on-annotated={r['frac_exactly_binary_on_annotated']:.3f}")

    clean = {k: v for k, v in sets.items() if not leaks.get(k, {}).get("leaked")}

    print("\n--- Findings 3 and 9: score against the 58, with bootstrap CIs ---")
    scored = score_label_sets(gold, sets, n_boot)
    print(scored[["label_set", "n", "mean_auc", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    widest = float((scored.ci_hi - scored.ci_lo).max())
    print(f"widest 95% CI across label sets: {widest:.4f}")

    print("\n--- Finding 5: cross-labeler agreement by language ---")
    agr = agreement_by_language(clean, lang)
    print(agr[["lang", "n", "agreement", "worst_finding", "worst_agreement"]].round(3).to_string(index=False))

    mentions = mention_matrix(reports)
    print("\n--- Finding 2b: Latin-script lexicon coverage by language ---")
    cov = lexicon_coverage(mentions, lang)
    print(cov.round(3).to_string(index=False))

    best = scored[~scored.label_set.isin([k for k, v in leaks.items() if v.get("leaked")])].iloc[0]
    verdicts = load_verdicts(label_roots)
    print(f"\n--- Finding 6: three things that look like silence "
          f"(negatives from {best.label_set}) ---")
    sil = silence_analysis(gold, mentions, verdicts, sets[best.label_set] > 0.5)
    show = ["finding", "prevalence", "rate_no_mention_regex", "p_pos_given_no_mention_regex",
            "rate_unk", "p_pos_given_unk", "n_unk", "rate_neg", "p_pos_given_neg"]
    print(sil[[c for c in show if c in sil.columns]].round(3).to_string(index=False))
    macro = {t: float(sil[f"p_pos_given_{t}"].mean())
             for t in ("no_mention_regex", "unk", "neg") if f"p_pos_given_{t}" in sil.columns}
    print("macro P(positive | ...): " + "  ".join(f"{k}={v:.3f}" for k, v in macro.items()))
    if "p_pos_given_unk" in sil.columns:
        worst = sil.loc[sil.p_pos_given_unk.idxmax()]
        print(f"worst finding for true silence: {worst.finding} at "
              f"P(positive | UNK)={worst.p_pos_given_unk:.3f} over n={int(worst.n_unk)}")
    print(f"reports where the lexicon fires on nothing: "
          f"{int((~mentions.any(axis=1)).sum())} of {len(mentions)}")

    print("\n--- Finding 7: are the 58 a fair sample? ---")
    rep = annotated_representativeness(train, series, gold.index)
    for k, v in rep.items():
        print(f"  {k:34s} {v}")
    print("\nannotated prevalence (%):")
    print((gold.mean() * 100).sort_values(ascending=False).round(1).to_string())

    return {
        "n_studies": len(train), "n_annotated": len(gold),
        "languages": lang.value_counts().to_dict(),
        "leaks": leaks,
        "scores": scored.to_dict(orient="records"),
        "widest_ci": widest,
        "agreement": agr.to_dict(orient="records"),
        "lexicon_coverage": cov.to_dict(orient="records"),
        "silence": sil.to_dict(orient="records"),
        "representativeness": rep,
        "annotated_prevalence": (gold.mean()).to_dict(),
    }


def check(res: dict) -> int:
    """Turn the findings into assertions so upstream drift is noticed."""
    fails = []

    def want(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    want(res["n_annotated"] == 58, f"expected 58 annotated studies, got {res['n_annotated']}")
    want(len(res["languages"]) >= 8, f"expected >=8 languages, got {len(res['languages'])}")

    leaked = [k for k, v in res["leaks"].items() if v.get("leaked")]
    want("yunus_3src" in leaked, "yunus_3src no longer shows the leak signature")
    for name in leaked:
        r = res["leaks"][name]
        want(r["frac_exactly_binary_on_annotated"] > 0.999,
             f"{name}: annotated rows not all exactly 0/1")
        want(r["frac_exactly_binary_elsewhere"] < 0.01,
             f"{name}: other rows are also binary, so this is not a leak signature")

    agr = pd.DataFrame(res["agreement"]).set_index("lang").agreement
    if "en" in agr.index:
        want(agr.idxmax() == "en", f"English is no longer the best-agreed language ({agr.idxmax()})")
        want(float(agr.max() - agr.min()) > 0.05,
             "agreement spread across languages collapsed below 0.05")

    sil = pd.DataFrame(res["silence"]).set_index("finding")
    if "p_pos_given_unk" in sil.columns:
        want(sil.p_pos_given_unk.mean() > 0.05,
             "P(positive | UNK) fell below 0.05, so unreadable entries now behave like negatives")
        want(sil.loc["Synovitis", "p_pos_given_unk"] > 0.20,
             "Synovitis is no longer the finding radiologists leave unstated")
        # The whole point of measuring it twice: a lexicon gap and a radiologist's
        # omission leave the same trace, and only the pair separates them.
        want(sil.p_pos_given_no_mention_regex.mean() > 1.5 * sil.p_pos_given_unk.mean(),
             "the regex detector no longer overstates silence relative to the LLM verdicts")

    cov = pd.DataFrame(res["lexicon_coverage"]).set_index("lang")
    if "bg" in cov.index:
        want(float(cov.loc["bg", "any_mention_rate"]) < 0.10,
             "the Latin-script lexicon now fires on Bulgarian, so Finding 2b is stale")

    want(res["widest_ci"] > 0.05,
         "58-row bootstrap CIs are narrower than 0.05, so the ranking may now be meaningful")

    if fails:
        print("\nFAILED CHECKS:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nall checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="competition data dir")
    ap.add_argument("--label-root", action="append", default=[],
                    help="dir to search for published label sets (repeatable)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root) if args.root else find_root()
    roots = [pathlib.Path(p) for p in args.label_root] or [
        pathlib.Path("/kaggle/input"), pathlib.Path("/tmp/rsnallm")
    ]
    roots = [p for p in roots if p.exists()]

    res = run(root, roots, args.boot)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(res, indent=2, default=float))
        print(f"\nwrote {args.json_out}")
    return check(res) if args.do_assert else 0


if __name__ == "__main__":
    sys.exit(main())
