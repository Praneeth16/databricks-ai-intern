# %% [markdown]
# # RSNA Knee: 58 labels, 9 languages, and a leak
#
# This competition gives you 4,407 training studies and labels for 58 of them. Everyone else
# builds labels by reading the radiology report, and about a dozen of those label sets are now
# published as Kaggle Datasets and merged into each other. Nobody has measured them. I did, and
# two of the published sets turn out to contain the 58 answers.
#
# | # | What I measured |
# |---|---|
# | 1 | Two published label sets score AUC **1.000 on all twelve findings** against the 58 annotated studies. On those 58 rows every value is exactly 0.0 or 1.0. On the other 4,349 rows not one value is. The answers were written in. |
# | 2 | A 58 row AUC has a 95% confidence interval about **0.23 wide**. The gap between the best and worst honest label set is 0.087. So the 58 studies cannot tell you which label set is better. |
# | 3 | Agreement between independent labelers needs no answers at all, so it covers all 4,407 studies. It runs from **0.819 in English down to 0.685 in Bulgarian**. The labelers get worse together, so merging three sources does not rescue the hard languages. |
# | 4 | The reports are in **nine languages**, and two of them are not written in the Latin alphabet. A keyword list covering the six Latin alphabet languages fires on 99.8% of English reports and on **0.9% of Bulgarian ones**, so a keyword labeler writes off about 500 studies as healthy. |
# | 5 | Silence is smaller than a keyword list suggests and worse where it happens. A keyword list implies 27.7% of unmentioned findings are actually present. An LLM that is allowed to answer "cannot tell" puts it at 8.2%. For Synovitis it is **34.1%**, and Synovitis is also the finding every label set scores worst on. |
# | 6 | The 58 annotated studies are not a fair sample. Their reports are longer, at 1,305 characters against 1,095 (p = 0.029), and 60.3% of them have an effusion. |
#
# The leak is not cheating. Putting real labels where real labels exist is a sensible thing to do
# when you train, and this competition invites you to build labels from the reports. It only
# becomes a problem when you then measure a labeler or a model on those same 58 studies, which is
# the one thing they are otherwise for. If you have merged either set, your validation number is
# not telling you what you think it is.
#
# ### What you get if you fork this
#
# 1. A two line test you can run on every label set you have attached, in section 2.
# 2. A per language reliability table for all 4,407 studies that needs no answers, in section 4.
# 3. A language detector with no dependencies that agrees with `langdetect` on 99.3% of the
#    corpus, so it runs with the internet switched off, in section 5.
# 4. The measurement that says which findings you can learn from text and which you cannot, in
#    section 6.
# 5. A confidence table written out at the end, with one row per study and finding, that you can
#    use directly as sample weights. It is also published as a CC0 dataset at
#    [rsna-knee-label-confidence](https://www.kaggle.com/datasets/paiky1995/rsna-knee-label-confidence)
#    so you can attach it without rerunning anything.
#
# ### Contents
#
# 1. [What the label supply chain actually looks like](#s1)
# 2. [Two published label sets contain the answers](#s2)
# 3. [Why 58 studies cannot rank anything](#s3)
# 4. [Measuring label quality without any labels](#s4)
# 5. [Nine languages, and a keyword list that reaches six](#s5)
# 6. [Three different things that all look like silence](#s6)
# 7. [Who the 58 studies are](#s7)
# 8. [What to do about it](#s8)
#
# ---

# %%
import ast, itertools, json, pathlib, re, unicodedata, warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Competition data mounts under /kaggle/input/competitions/<slug>, not /kaggle/input/<slug>.
CANDIDATES = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection", "/tmp/rsnaknee"]
ROOT = next(p for p in map(pathlib.Path, CANDIDATES) if (p / "train.csv").exists())
SEARCH = [p for p in map(pathlib.Path, ["/kaggle/input", "/tmp/rsnallm"]) if p.exists()]

train = pd.read_csv(ROOT / "train.csv")
series = pd.read_csv(ROOT / "train_series.csv")
reports = train.set_index("StudyInstanceUID").Report
gold = (train[train[LABELS].notna().any(axis=1)]
        .set_index("StudyInstanceUID")[LABELS].astype(int))

print(f"data root       {ROOT}")
print(f"studies         {len(train):,}")
print(f"series          {len(series):,}")
print(f"annotated       {len(gold)}  ({len(gold) / len(train):.2%})")
print(f"findings        {len(LABELS)}")

# %% [markdown]
# <a id="s1"></a>
# ## 1. What the label supply chain actually looks like
#
# The score is the mean of twelve ROC AUC values, one per finding. The test set has no `Report`
# column, so text cannot help you at prediction time. Text can only help you build training
# labels, and that is what everyone uses it for.
#
# Below is every published label set I could find. I load each one, keep only the sets that carry
# all twelve findings, and record how many studies each one covers.

# %%
LABEL_SETS = {
    "pilkwang_v1_regex":  ("pilkwang/rsna-knee-report-labels", "report_labels_v1.csv"),
    "pilkwang_v2_llm":    ("pilkwang/rsna-knee-llm-labels", "report_labels_v2.csv"),
    "stevenleehans_full": ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_full.csv"),
    "stevenleehans_v4":   ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv"),
    "lixin73_gpt56sol":   ("lixin73/rsna-knee-llm-report-labels-sol56", "labels_llm_gpt56sol.csv"),
    "flight_hybrid":      ("flight0234/rsna-knee-hybrid-report-labels", "report_labels_v4hybrid.csv"),
    "yunus_3src":         ("yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels", "report_labels_v3.csv"),
    "yunus_4src":         ("yunusgmsoy/rsna-knee-llm-labels-4-source-merged", "report_labels_v5.csv"),
}

sets, inventory = {}, []
for name, (slug, fname) in LABEL_SETS.items():
    hit = next((h for root in SEARCH for h in root.rglob(fname)), None)
    if hit is None:
        inventory.append({"label_set": name, "dataset": slug, "found": False, "studies": 0})
        continue
    d = pd.read_csv(hit)
    if not all(c in d.columns for c in LABELS):
        inventory.append({"label_set": name, "dataset": slug, "found": True, "studies": 0})
        continue
    sets[name] = (d.drop_duplicates("StudyInstanceUID")
                  .set_index("StudyInstanceUID")[LABELS].astype(float))
    inventory.append({"label_set": name, "dataset": slug, "found": True,
                      "studies": int(sets[name].index.nunique()),
                      "distinct_values": int(len(np.unique(sets[name].values)))})

print(pd.DataFrame(inventory).to_string(index=False))
print(f"\nusable label sets: {len(sets)}")

# %% [markdown]
# <a id="s2"></a>
# ## 2. Two published label sets contain the answers
#
# A label set that was built only from reports has no way to reproduce the 58 annotations
# exactly. If it does reproduce them exactly, the annotations were copied in.
#
# The test has two parts, and both are needed. First, on the 58 annotated rows, is every value
# exactly 0.0 or 1.0 and equal to the annotation? Second, on the other 4,349 rows, is that not
# true? The second part is what separates a copied answer key from a label set that simply
# rounds all of its outputs to 0 and 1 everywhere.

# %%
def leak_test(labels: pd.DataFrame) -> dict:
    on = labels.reindex(gold.index).dropna(how="all")
    off = labels.loc[~labels.index.isin(gold.index)]
    binary_on = float(np.isin(on.values, [0.0, 1.0]).mean())
    binary_off = float(np.isin(off.values, [0.0, 1.0]).mean())
    match = float((on.values == gold.reindex(on.index).values).mean())
    return {"binary_on_annotated": binary_on, "binary_elsewhere": binary_off,
            "matches_annotations": match, "distinct_values_elsewhere": int(len(np.unique(off.values))),
            "leaked": bool(binary_on > 0.999 and match > 0.999 and binary_off < 0.01)}

leaks = pd.DataFrame([{"label_set": k, **leak_test(v)} for k, v in sets.items()])
print(leaks.round(3).to_string(index=False))

leaked = sorted(leaks.loc[leaks.leaked, "label_set"])
clean = {k: v for k, v in sets.items() if k not in leaked}
print(f"\ncontain the answers: {leaked}")
for k in leaked:
    print(f"  {k}  ->  {LABEL_SETS[k][0]} / {LABEL_SETS[k][1]}")

# %% [markdown]
# Read the `lixin73_gpt56sol` row before you trust the test. It is 95.4% exactly 0 or 1 on the
# annotated rows, which on its own looks alarming. It is also 96.2% exactly 0 or 1 everywhere
# else, and it reproduces only 79.9% of the annotations. It is a label set that gives hard 0 and
# 1 answers, and the test correctly clears it.
#
# The two sets that fail are different in kind. They are 100% exactly 0 or 1 on the annotated
# rows and 0% exactly 0 or 1 on the other 4,349, where they take 344 distinct values. The
# annotated rows were overwritten.
#
# Here is the whole check, short enough to paste into your own notebook.

# %%
def contains_the_answers(labels: pd.DataFrame) -> bool:
    on, off = labels.reindex(gold.index), labels.loc[~labels.index.isin(gold.index)]
    return bool(np.isin(on.values, [0.0, 1.0]).all()
                and (on.values == gold.values).all()
                and not np.isin(off.values, [0.0, 1.0]).any())

for name, d in sets.items():
    print(f"{name:20s} {contains_the_answers(d)}")

# %% [markdown]
# <a id="s3"></a>
# ## 3. Why 58 studies cannot rank anything
#
# Now score each label set the way the competition scores you, as the mean of twelve per finding
# AUC values, against the 58 annotations. Then bootstrap it, so the number comes with an
# interval instead of arriving alone.

# %%
def auc_with_interval(y, p, n_boot, rng):
    point = roc_auc_score(y, p)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            boots.append(roc_auc_score(y[i], p[i]))
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

rng = np.random.default_rng(0)
rows, per_label = [], {}
for name, d in sets.items():
    common = gold.index.intersection(d.index)
    aucs, los, his = {}, [], []
    for c in LABELS:
        y = gold.loc[common, c].values.astype(int)
        p = np.nan_to_num(d.loc[common, c].values, nan=float(np.nanmean(d[c])))
        a, lo, hi = auc_with_interval(y, p, 2000, rng)
        aucs[c], _ = a, los.append(lo)
        his.append(hi)
    per_label[name] = aucs
    rows.append({"label_set": name, "n": len(common), "mean_auc": np.mean(list(aucs.values())),
                 "ci_lo": np.mean(los), "ci_hi": np.mean(his), "leaked": name in leaked})

scored = pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)
scored["ci_width"] = scored.ci_hi - scored.ci_lo
print(scored.round(4).to_string(index=False))

honest = scored[~scored.leaked]
print(f"\nbest honest set   {honest.iloc[0].label_set} at {honest.iloc[0].mean_auc:.4f}")
print(f"worst honest set  {honest.iloc[-1].label_set} at {honest.iloc[-1].mean_auc:.4f}")
print(f"spread            {honest.iloc[0].mean_auc - honest.iloc[-1].mean_auc:.4f}")
print(f"widest interval   {scored.ci_width.max():.4f}")

# %%
fig, ax = plt.subplots(figsize=(8, 3.6))
h = scored[~scored.leaked].sort_values("mean_auc")
y = np.arange(len(h))
ax.barh(y, h.mean_auc, color="#4a7fb5", height=0.6)
ax.errorbar(h.mean_auc, y, xerr=[h.mean_auc - h.ci_lo, h.ci_hi - h.mean_auc],
            fmt="none", ecolor="#22303c", capsize=4, lw=1.4)
for k in scored[scored.leaked].label_set:
    ax.axvline(1.0, color="#c0392b", lw=1.6, ls="--")
ax.text(0.999, len(h) - 0.4, f"  {len(leaked)} sets sit here, at 1.000",
        color="#c0392b", ha="right", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(h.label_set)
ax.set_xlim(0.5, 1.02); ax.set_xlabel("mean AUC over 12 findings, against 58 studies")
ax.set_title("Every honest label set has an interval wide enough to cover the others")
plt.tight_layout(); plt.show()

# %% [markdown]
# The intervals overlap almost completely. The best honest set reads 0.899 and the worst reads
# 0.813, and the widest interval is 0.230 across. On 58 studies you cannot say that any of these
# label sets is better than another, and picking one because it scored highest here is picking
# noise. The two dashed sets at 1.000 are the ones that contain the answers.
#
# This is the reason the rest of the notebook stops using the 58 studies as the main instrument.
#
# <a id="s4"></a>
# ## 4. Measuring label quality without any labels
#
# Independent labelers read the same 4,407 reports. Where they disagree, at least one of them is
# wrong. That comparison needs no annotations, so it covers every study rather than 58 of them.
#
# I rank each labeler's output per finding, then take the mean correlation over every pair of
# labelers, then split by language. I use only the honest sets, because the two leaked sets agree
# with the answers by construction on the rows that would matter most.

# %%
HEADERS = {
    "en": ["findings", "impression", "technique", "clinical", "conclusion:", "comparison", "history"],
    "es": ["tecnica", "resultados", "impresion", "hallazgos", "antecedentes", "estudio"],
    "nl": ["bevindingen", "klinische", "inlichtingen", "conclusie", "verslag", "vraagstelling"],
    "de": ["befund", "beurteilung", "fragestellung", "anamnese", "technik"],
    "fr": ["indication", "resultats", "technique:", "conclusion :", "protocole"],
    "tr": ["bulgular", "sonuc", "inceleme", "tetkik", "klinik bilgi", "oyku"],
    "hr": ["nalaz", "zaključak", "misljenje", "mr nalaz", "opis"],
}
WORDS = {
    "en": ["the", "and", "with", "there", "is", "of", "no ", "are", "was"],
    "es": ["de la", "del", "con", "sin", "en el", "se ", "los", "las", "una"],
    "nl": ["van", "met", "het", "een", "geen", "is er", "bij", "naar"],
    "de": ["und", "mit", "des", "der", "kein", "eine", "im ", "zeigt"],
    "fr": ["avec", "sans", "du ", "des ", "le ", "la ", "une", "est"],
    "tr": ["ve ", "ile", "sag", "sol", "var", "izlen", "mm ", "olan"],
    "hr": ["je ", "sa ", "uz ", "nema", "vidljiv", "desno", "lijevo", "prikaz"],
}
DIACRITICS = {"tr": "ığşİĞŞ", "hr": "čćžšđČĆŽŠĐ", "es": "ñáíóúÁÍÓÚ",
              "de": "äöüßÄÖÜ", "fr": "éèêçàùÉÈÇ", "nl": "ijëï"}

def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()

def detect_language(text: str) -> str:
    """Greek and Bulgarian are decided by alphabet, which is stronger than any guess.
    The Latin languages are scored on report section headers, then function words."""
    raw = str(text)
    letters = sum(c.isalpha() for c in raw) or 1
    if sum("Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿" for c in raw) / letters > 0.30:
        return "el"
    if sum("Ѐ" <= c <= "ӿ" for c in raw) / letters > 0.30:
        return "bg"
    f = fold(raw)
    score = {}
    for lg in WORDS:
        s = 6.0 * sum(h in f for h in HEADERS.get(lg, ()))
        s += sum(f.count(w) for w in WORDS[lg])
        s += 2.0 * sum(ch in raw for ch in DIACRITICS.get(lg, ""))
        score[lg] = s
    best = max(score, key=score.get)
    return best if score[best] > 0 else "unk"

lang = reports.map(detect_language)
print(lang.value_counts().to_string())
print(f"\nnot English: {(lang != 'en').mean():.1%} of the corpus")

# %%
common = None
for d in clean.values():
    common = d.index if common is None else common.intersection(d.index)
ranked = {k: d.loc[common].rank(pct=True) for k, d in clean.items()}
pairs = list(itertools.combinations(ranked, 2))

rows = []
for lg, grp in lang.reindex(common).groupby(lang.reindex(common)):
    if len(grp) < 40:
        continue
    per = {}
    for c in LABELS:
        cors = [np.corrcoef(ranked[a].loc[grp.index, c], ranked[b].loc[grp.index, c])[0, 1]
                for a, b in pairs
                if ranked[a].loc[grp.index, c].std() > 0 and ranked[b].loc[grp.index, c].std() > 0]
        per[c] = float(np.mean(cors)) if cors else np.nan
    worst = min(per, key=lambda k: per[k])
    rows.append({"lang": lg, "n": len(grp), "agreement": np.nanmean(list(per.values())),
                 "worst_finding": worst, "worst": per[worst], **per})

agree = pd.DataFrame(rows).sort_values("agreement").reset_index(drop=True)
print(f"{len(pairs)} labeler pairs over {len(common):,} studies\n")
print(agree[["lang", "n", "agreement", "worst_finding", "worst"]].round(3).to_string(index=False))

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
o = agree.sort_values("agreement")
c = ["#c0392b" if v < 0.72 else "#4a7fb5" for v in o.agreement]
a1.bar(o.lang, o.agreement, color=c)
a1.set_ylim(0.6, 0.86); a1.set_ylabel("mean agreement between labelers")
a1.set_title("Labelers agree least on the languages\nthey were not written for")
for i, (l, v, n) in enumerate(zip(o.lang, o.agreement, o.n)):
    a1.text(i, v + 0.004, f"{v:.3f}\nn={n}", ha="center", fontsize=7)

m = agree.set_index("lang")[LABELS].loc[o.lang]
im = a2.imshow(m.values, aspect="auto", cmap="RdYlBu", vmin=0.3, vmax=0.95)
a2.set_xticks(range(len(LABELS))); a2.set_xticklabels(LABELS, rotation=90, fontsize=7)
a2.set_yticks(range(len(m))); a2.set_yticklabels(m.index)
a2.set_title("Agreement per finding and language"); a2.grid(False)
plt.colorbar(im, ax=a2, shrink=0.85)
plt.tight_layout(); plt.show()

# %% [markdown]
# Agreement falls steadily from 0.819 in English to 0.685 in Bulgarian. English is the only
# language the labelers handle well, and it is 40% of the corpus.
#
# This is the part that should change what you do. Voting across sources removes noise when the
# sources are wrong independently. Here they are wrong together, because they are all reading a
# language none of them was built for. Merging three sources on a Bulgarian report gives you
# three copies of a similar mistake and a confident average. That is worse than a low weight,
# because it looks reliable.
#
# `Fracture` is the worst agreed finding in six of the nine languages. `Synovitis` is worst in
# Bulgarian and Dutch, at 0.336 and 0.471.
#
# <a id="s5"></a>
# ## 5. Nine languages, and a keyword list that reaches six
#
# My detector has no dependencies, so it runs with the internet switched off. Two checks say it
# is sound. Greek and Bulgarian are decided by the alphabet, which cannot be wrong for the 541
# reports written in them. For the rest, I compared it against `langdetect` offline and the two
# agree on 99.3% of the 4,407 reports. The only real disagreement is 26 Croatian reports that my
# detector reads as English.
#
# Now the point of measuring the languages. Published keyword labelers use a Latin alphabet word
# list. Below I run one over the corpus and count how often it fires at all, by language.

# %%
MENTION = {
    "ACL": [r"\bacl\b", r"anterior cruciate", r"cruzado anterior", r"\blca\b", r"voorste kruisband",
            r"vorderes kreuzband", r"croise anterieur", r"on capraz"],
    "MCL": [r"\bmcl\b", r"medial collateral", r"colateral (?:medial|interno)", r"innenband",
            r"mediale collaterale", r"\blcm\b"],
    "Medial Meniscus": [r"medial meniscus", r"menisc\w* (?:interno|medial)", r"mediale meniscus",
                        r"innenmeniskus", r"menisque (?:interne|medial)", r"medial menisk"],
    "Lateral Meniscus": [r"lateral meniscus", r"menisc\w* (?:externo|lateral)", r"laterale meniscus",
                         r"aussenmeniskus", r"menisque (?:externe|lateral)", r"lateral menisk"],
    "Medial OA": [r"medial (?:compartment )?(?:osteoarthritis|oa|degenerat)",
                  r"artrosis (?:femorotibial )?(?:medial|interna)", r"mediale gonarthrose",
                  r"gonartros\w* medial", r"artrose mediaal"],
    "Lateral OA": [r"lateral (?:compartment )?(?:osteoarthritis|oa|degenerat)",
                   r"artrosis (?:femorotibial )?(?:lateral|externa)", r"laterale gonarthrose",
                   r"artrose lateraal"],
    "PF OA": [r"patellofemoral", r"femoropatelar", r"retropatell", r"femoro-?patellaire"],
    "Effusion": [r"effusion", r"derrame", r"\bhydrops\b", r"gelenkerguss", r"\berguss\b",
                 r"epanchement", r"versamento", r"joint fluid", r"efuzyon"],
    "Synovitis": [r"synovit", r"sinovit", r"synovial (?:thickening|proliferation)"],
    "Baker's": [r"\bbaker", r"popliteal cyst", r"poplitea\w* cyst", r"bakerzyste"],
    "Contusion": [r"contusion", r"contusio", r"bone (?:bruise|contusion)",
                  r"edema (?:oseo|osseo|medular)", r"bone marrow edema", r"knochenmarkodem"],
    "Fracture": [r"fractur", r"fractuur", r"fraktur", r"\bfissur", r"frattura", r"kirik"],
}

flat = reports.map(fold)
mentions = pd.DataFrame({k: flat.str.contains("|".join(v), regex=True) for k, v in MENTION.items()},
                        index=reports.index)
coverage = (pd.DataFrame({"lang": lang, "fires": mentions.any(axis=1)})
            .groupby("lang").agg(n=("fires", "size"), fires_on=("fires", "mean"))
            .sort_values("fires_on").reset_index())
print(coverage.round(3).to_string(index=False))
print(f"\nreports where nothing fires: {int((~mentions.any(axis=1)).sum()):,} of {len(mentions):,}")

# %%
fig, ax = plt.subplots(figsize=(7.2, 3.4))
c = ["#c0392b" if v < 0.75 else "#4a7fb5" for v in coverage.fires_on]
ax.bar(coverage.lang, coverage.fires_on, color=c)
for i, v in enumerate(coverage.fires_on):
    ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=8)
ax.set_ylim(0, 1.12); ax.set_ylabel("reports where the word list fires at all")
ax.set_title("A Latin alphabet word list reaches six of the nine languages")
plt.tight_layout(); plt.show()

# %% [markdown]
# The word list fires on 99.8% of English reports and on 0.9% of Bulgarian ones. Greek reaches
# 41.1%, and that comes from the Latin abbreviations and measurements that survive inside a Greek
# report rather than from any word the list knows.
#
# A study where the word list never fires becomes a row of twelve zeros. Bulgarian is 220 studies
# and Greek is 321, so a keyword labeler quietly writes off about 500 studies as having nothing
# wrong with them. This is a large part of why the keyword set scores 0.813 while the language
# model sets reach 0.899.
#
# <a id="s6"></a>
# ## 6. Three different things that all look like silence
#
# A report says what the radiologist chose to say. If a finding is not mentioned, that is not the
# same as the radiologist ruling it out, yet every published set stores it as a low number. It is
# tempting to measure this with a word list and stop there. That number is wrong, and measuring
# it three ways shows why.
#
# 1. The word list never fires. This mixes together a radiologist's silence and a gap in the word
#    list, and section 5 shows the gaps are large.
# 2. A language model read the report and answered "cannot tell". One published set,
#    `pilkwang/rsna-knee-llm-labels`, ships three way verdicts of YES, NO and UNK, and it is the
#    only set that does. This is the honest measure of what the report does not say.
# 3. The best scoring set concluded the finding is absent. That is its error rate on negatives,
#    which is a different question again.

# %%
verdicts = None
for root in SEARCH:
    for hit in root.rglob("report_labels_v2.csv"):
        d = pd.read_csv(hit).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        cols = {c: c.replace("__verdict", "") for c in d.columns if c.endswith("__verdict")}
        if len(cols) == len(LABELS):
            verdicts = d[list(cols)].rename(columns=cols)[LABELS]
            break

best = honest.iloc[0].label_set
negatives = sets[best] > 0.5
print(f"verdict mix over all study and finding pairs (pilkwang_v2_llm):")
print(pd.concat([verdicts[c] for c in LABELS]).value_counts(normalize=True).round(3).to_string())

rows = []
for c in LABELS:
    y = gold[c].values.astype(int)
    masks = {"no_mention": ~mentions.reindex(gold.index)[c].values,
             "cannot_tell": (verdicts.reindex(gold.index)[c] == "UNK").fillna(False).values,
             "said_absent": ~negatives.reindex(gold.index)[c].fillna(False).values}
    r = {"finding": c, "prevalence": y.mean()}
    for tag, m in masks.items():
        r[f"rate_{tag}"] = m.mean()
        r[f"p_pos_{tag}"] = y[m].mean() if m.any() else np.nan
        r[f"n_{tag}"] = int(m.sum())
    rows.append(r)

silence = pd.DataFrame(rows)
print("\n" + silence[["finding", "prevalence", "rate_no_mention", "p_pos_no_mention",
                      "rate_cannot_tell", "p_pos_cannot_tell", "n_cannot_tell",
                      "p_pos_said_absent"]].round(3).to_string(index=False))
print(f"\nmean P(present | word list never fired)  {silence.p_pos_no_mention.mean():.3f}")
print(f"mean P(present | model cannot tell)     {silence.p_pos_cannot_tell.mean():.3f}")
print(f"mean P(present | model said absent)     {silence.p_pos_said_absent.mean():.3f}")

# %%
fig, ax = plt.subplots(figsize=(8.4, 3.6))
o = silence.sort_values("p_pos_cannot_tell", ascending=False)
x = np.arange(len(o))
ax.bar(x - 0.2, o.p_pos_no_mention, 0.4, label="word list never fired", color="#e2a03f")
ax.bar(x + 0.2, o.p_pos_cannot_tell, 0.4, label="model answered cannot tell", color="#4a7fb5")
ax.axhline(silence.p_pos_cannot_tell.mean(), color="#22303c", ls=":", lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(o.finding, rotation=45, ha="right")
ax.set_ylabel("share that are actually present")
ax.set_title("A word list overstates silence about three times over, "
             "and Synovitis is the finding that is genuinely unstated")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %% [markdown]
# The word list puts silence at 27.7%. The model that is allowed to say it cannot tell puts it at
# 8.2%. Most of what looks like a radiologist's silence is the word list failing to read the
# report, so a keyword labeler both misses findings and then overstates how often reports are
# silent.
#
# The 8.2% that remains is not spread evenly. `Synovitis` is the outlier. The model cannot tell
# for 70.7% of the annotated studies, and 34.1% of those studies do have synovitis, over 41
# studies. Compare that to `Effusion`, where the model cannot tell for only 3.4% of studies.
#
# Put that next to section 3, where `Synovitis` scores between 0.628 and 0.790 across every label
# set and is the worst finding for all of them. Those are the same fact. Synovitis scores badly
# from text because radiologists in this corpus mostly do not write it down. No better prompt
# fixes that, because the information is not in the report. Your image model has to carry
# Synovitis on its own.
#
# <a id="s7"></a>
# ## 7. Who the 58 studies are
#
# One more reason to be careful with the 58, separate from the leak.

# %%
is_gold = train.StudyInstanceUID.isin(gold.index)
gl, wl = train.loc[is_gold, "Report"].str.len(), train.loc[~is_gold, "Report"].str.len()
counts = series.groupby("StudyInstanceUID").size()
gc = counts.reindex(train.loc[is_gold, "StudyInstanceUID"]).dropna()
wc = counts.reindex(train.loc[~is_gold, "StudyInstanceUID"]).dropna()

print(f"report length   annotated {gl.mean():7.1f}   others {wl.mean():7.1f}   "
      f"p = {stats.mannwhitneyu(gl, wl).pvalue:.4f}")
print(f"series / study  annotated {gc.mean():7.2f}   others {wc.mean():7.2f}   "
      f"p = {stats.mannwhitneyu(gc, wc).pvalue:.4f}")
print("\nshare of the 58 that have each finding (%):")
print((gold.mean() * 100).sort_values(ascending=False).round(1).to_string())
print(f"\nmean findings per annotated study: {gold.sum(axis=1).mean():.2f} of 12")

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
a1.hist(wl, bins=60, density=True, color="#b9c6d2", label=f"other 4,349")
a1.hist(gl, bins=20, density=True, histtype="step", lw=2, color="#c0392b", label="the 58")
a1.set_xlim(0, 4000); a1.set_xlabel("report length in characters"); a1.set_ylabel("density")
a1.set_title("The annotated reports are longer"); a1.legend(fontsize=8)

p = (gold.mean() * 100).sort_values()
a2.barh(p.index, p.values, color="#4a7fb5")
for i, v in enumerate(p.values):
    a2.text(v + 0.8, i, f"{v:.1f}", va="center", fontsize=7.5)
a2.set_xlabel("% of the 58 studies"); a2.set_title("and they carry a lot of findings")
plt.tight_layout(); plt.show()

# %% [markdown]
# The annotated reports run 1,305 characters against 1,095, and the difference is unlikely to be
# chance at p = 0.029. A longer report usually means more findings were described. The
# prevalences agree with that reading. 60.3% of the 58 have an effusion and 46.6% have
# synovitis, and an average annotated study carries 4.1 of the 12 findings.
#
# I cannot tell you whether the hidden test set was drawn the same way as these 58 or the same
# way as the other 4,349, and that choice changes what your validation number is worth. What I
# can say is that the 58 are not a sample of the 4,407, so tuning a threshold on them will not
# transfer cleanly.
#
# The number of series per study is about the same, at 5.79 against 5.53 with p = 0.091, so the
# imaging protocol is not what separates them.
#
# <a id="s8"></a>
# ## 8. What to do about it
#
# Four changes follow from the measurements above.
#
# 1. Test every label set you have attached with the check in section 2. If a set contains the
#    answers, keep using it for training and stop using those 58 studies to measure anything.
# 2. Stop treating a merge of several sources as more reliable than one source. Agreement falls
#    with language, and the sources fail together, so weight each study by the agreement for its
#    language instead of counting votes.
# 3. Do not put a zero where a report is silent. Leave the entry out of the loss. For Synovitis
#    that is most of the corpus, and a zero there is wrong about a third of the time.
# 4. Expect your image model to carry Synovitis, Lateral OA and Fracture, because text
#    supervision is weakest exactly there.
#
# The table below applies the second and third points. It has one row per study and finding for
# all 4,407 studies, with the agreement for that study's language and finding, a weight you can
# multiply into a loss, and a flag for the entries you should mask instead of setting to zero. I
# publish it as a separate CC0 dataset so you can attach it without rerunning this notebook.
#
# The two corrections are in separate columns because they turn out to be independent. How well
# the labelers agree depends on the language. Whether the report says anything at all does not
# follow the same order. Spanish is the clearest case. The labelers handle Spanish second best of
# the nine languages, at 0.767 agreement, and yet Spanish reports leave the most unsaid, with 44%
# of entries unstated against 14.5% for French. Spanish reports in this corpus are short and list
# only what was found, so 40.6% of them never mention the ACL, against 1.2% of English reports.
# A single combined number would average those two effects and lose both.

# %%
long = agree.set_index("lang")[LABELS].stack().rename("agreement").reset_index()
long.columns = ["lang", "finding", "agreement"]

conf = pd.DataFrame({"StudyInstanceUID": lang.index, "lang": lang.values})
conf = conf.merge(long, on="lang", how="left")

unk = (verdicts == "UNK").stack().rename("unstated").reset_index()
unk.columns = ["StudyInstanceUID", "finding", "unstated"]
conf = conf.merge(unk, on=["StudyInstanceUID", "finding"], how="left")
conf["unstated"] = conf.unstated.fillna(False).astype(bool)

# The two corrections are kept in separate columns on purpose. `weight` covers how
# much the labelers agree for this language and finding. `unstated` covers whether
# the report says anything at all. They are different problems, and multiplying them
# together in this file would hide that, so you apply the mask yourself.
lo, hi = conf.agreement.min(), conf.agreement.max()
conf["weight"] = ((conf.agreement - lo) / (hi - lo) * 0.7 + 0.3).round(4)
conf["agreement"] = conf.agreement.round(4)
conf = conf[["StudyInstanceUID", "lang", "finding", "agreement", "unstated", "weight"]]

conf.to_csv("label_confidence.csv", index=False)
print(conf.head(8).to_string(index=False))
print(f"\nrows {len(conf):,}  studies {conf.StudyInstanceUID.nunique():,}  "
      f"findings {conf.finding.nunique()}")
print(f"entries to mask rather than zero: {conf.unstated.sum():,} ({conf.unstated.mean():.1%})\n")

axes = conf.groupby("lang").agg(n=("weight", "size"), weight=("weight", "mean"),
                                unstated=("unstated", "mean")).sort_values("weight")
print(axes.round(3).to_string())

# How usable is this? A loss that applies both corrections keeps this much of each
# language, relative to counting every entry equally.
kept = conf.assign(k=conf.weight * ~conf.unstated).groupby("lang").k.mean()
print(f"\neffective share of each language kept by a weighted, masked loss:")
print(kept.round(3).sort_values().to_string())

# %% [markdown]
# ## What I could not settle
#
# Three things are open, and I would rather say so than round them off.
#
# The 8.2% silence figure rests on one label set's UNK verdicts, so it inherits that set's
# judgement about when a report is unclear. A second set shipping three way verdicts would let
# someone check it, and no other published set does.
#
# I do not know how the 58 studies were chosen, so I can describe how they differ from the rest
# but not why, and I cannot say which population the hidden test set matches.
#
# My language detector reads 26 Croatian reports as English. That is 0.6% of the corpus and it
# will pull the Croatian agreement figure slightly toward the English one, so 0.707 for Croatian
# is a little optimistic.
#
# ## Credit
#
# This notebook measures other people's published work, so the people who did that work should be
# named. Every label set in section 1 came from a public Kaggle Dataset, and the audit is only
# possible because those authors published rather than kept them private.
#
# `pilkwang/rsna-knee-llm-labels` deserves a specific mention. It is the only set that reports
# YES, NO and UNK instead of collapsing uncertainty into a number, and section 6 exists only
# because of that choice. If you upvote one thing from this notebook's references, upvote that
# one.
#
# The two sets that contain the answers are `yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels`
# and `yunusgmsoy/rsna-knee-llm-labels-4-source-merged`. Writing real labels into a training set
# is a reasonable thing to do, and naming the files here is so that anyone who merged them can
# check their own validation, not a criticism of the author.
