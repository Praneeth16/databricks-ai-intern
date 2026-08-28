# %% [markdown]
# # RSNA Knee: the data, then a simple baseline
#
# This notebook reads the competition data, shows what is in it, trains one small image model,
# and writes a submission. Everything runs in this notebook on one T4 GPU with the internet
# switched off.
#
# The short version of what the data forces on you:
#
# 1. There are 4,407 training studies and only 58 of them carry the twelve labels. You cannot
#    train an image model on 58 studies, so the labels have to come from somewhere else.
# 2. Every training study carries the radiology report. `test.csv` has no `Report` column. Text is
#    available while you fit and gone when you predict, so a report can only ever be a training
#    label, never an input.
# 3. The score is the average of twelve ROC AUC values. Only the order of your predictions inside
#    each column is read, so calibration and thresholds are worth nothing.
#
# The model here is a ResNet18 over twelve sagittal slices per study. What it reaches:
#
# | Measurement | Average AUC |
# |---|---|
# | Model against the report labels, on a held out fifth of the studies | about 0.78 |
# | Model against a radiologist, on the 58 labelled studies, none of which it trained on | 0.76 to 0.79 |
# | Public leaderboard score of this notebook | 0.798 |
# | The report labels themselves against the radiologist, on the same 58 | 0.90 |
#
# The second row is a range because that is what 58 studies buy. Rerunning the notebook with one
# study moved between the two sides changed it by 0.03, which is the size section 3 predicts from
# the sample alone. The holdout number in the first row moved by 0.005 over the same change.
#
# The third row is the reason the second row is still worth printing. The 58 studies estimated
# 0.778 and the leaderboard came back 0.798. They cannot rank two models that are 0.02 apart, but
# they did say this model works, and the estimate landed close.
#
# The last row is the ceiling on the second. The model is learning from labels that are 0.90
# against a radiologist, so better labels move it further than a bigger network does. Section 12
# says what to change first.
#
# The top of the leaderboard is around 0.95, so this is a starting point rather than a contender.
# The whole notebook, reading all 4,407 studies and training from scratch, takes about 15 minutes
# on one T4.

# %%
import gc, hashlib, pathlib, re, time, unicodedata, warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
T0 = time.time()

def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

# One session on this competition gets a fixed amount of wall clock. Every stage below checks
# these numbers, so a slow read shrinks the training set instead of killing the run.
HARD_DEADLINE = T0 + 8.0 * 3600
PREP_BUDGET_S = 2.5 * 3600

N_SLICES = 12          # slices kept per study
SIZE = 224             # pixels per side after the resize
EPOCHS = 8
BATCH = 32
HOLDOUT = 0.20
SEED = 0
READ_THREADS = 12

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# The competition mounts under /kaggle/input/competitions/<slug> on this image, and under
# /kaggle/input/<slug> on older ones.
COMP = next(p for p in map(pathlib.Path, [
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    "/kaggle/input/rsna-knee-abnormality-detection",
    "/tmp/rsnaknee"]) if (p / "train.csv").exists())

train = pd.read_csv(COMP / "train.csv")
train_series = pd.read_csv(COMP / "train_series.csv")
test = pd.read_csv(COMP / "test.csv")
test_series = pd.read_csv(COMP / "test_series.csv")

reports = train.set_index("StudyInstanceUID").Report.fillna("")
gold = train[train[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS].astype(int)

log(f"data root        {COMP}")
log(f"train studies    {len(train):,}    train series {len(train_series):,}")
log(f"test studies     {len(test):,}     test series  {len(test_series):,}")
log(f"labelled studies {len(gold)}  ({len(gold) / len(train):.2%} of training)")

# %% [markdown]
# ## Write a valid submission before doing anything else
#
# This is a code competition. When you submit, Kaggle runs this notebook again against a test set
# you have never seen, and a run that fails part way through scores nothing. So the first thing
# the notebook does is write a submission of all 0.5 values. Every later stage overwrites it. If
# the model stage fails, you still get a scored run and you can read the log to see why.

# %%
def write_submission(pred: pd.DataFrame, path="submission.csv") -> pd.DataFrame:
    """Rank inside each column, then write one row per test study.

    The score reads only the order of the values in a column, so replacing the numbers by their
    ranks throws nothing away, and it removes any question about how the probabilities are
    spread out.
    """
    sub = pred.rank(pct=True) if len(pred) > 1 else pred.copy()
    sub.index.name = "StudyInstanceUID"
    sub = sub.reindex(test.StudyInstanceUID).fillna(0.5).reset_index()
    sub = sub[["StudyInstanceUID"] + LABELS]
    assert len(sub) == len(test), f"{len(sub)} rows against {len(test)} test studies"
    assert sub[LABELS].notna().all().all(), "a prediction is missing"
    sub.to_csv(path, index=False)
    return sub

write_submission(pd.DataFrame(0.5, index=test.StudyInstanceUID, columns=LABELS))
log("wrote a placeholder submission.csv")

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# One fixed set of colours, used in the same order everywhere below.
BLUE, ORANGE, AQUA, YELLOW, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("seq", ["#e8f1fd", "#9ec5f4", "#2a78d6", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#d03b3b"])

plt.rcParams.update({
    "figure.dpi": 120, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 8.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 9.5, "legend.frameon": False,
})

# A printed DataFrame wraps at the cell width and becomes unreadable, so every table below
# goes through display() as HTML instead.
from html import escape as _esc
from IPython.display import display, HTML, Markdown

TABLE_CSS = [
    {"selector": "caption", "props": [("caption-side", "top"), ("text-align", "left"),
                                      ("font-weight", "600"), ("font-size", "0.95rem"),
                                      ("padding", "0 0 0.45rem 0"), ("color", INK)]},
    {"selector": "th", "props": [("background-color", "#f2f1ec"), ("color", INK),
                                 ("font-weight", "600"), ("text-align", "right"),
                                 ("padding", "5px 11px"), ("border-bottom", f"1px solid #c3c2b7")]},
    {"selector": "th.row_heading", "props": [("text-align", "left"),
                                             ("background-color", "#fcfcfb")]},
    {"selector": "td", "props": [("padding", "5px 11px"), ("text-align", "right"),
                                 ("border-bottom", f"1px solid {GRID}")]},
    {"selector": "", "props": [("border-collapse", "collapse"), ("font-size", "0.86rem"),
                               ("font-variant-numeric", "tabular-nums"), ("margin", "0.3rem 0")]},
]

def _cell(v):
    """One value as text. Integers get thousands separators, floats three decimals."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{v:,.3f}" if abs(v) < 1e4 else f"{v:,.0f}"
    return str(v)

def show(df, caption="", grad=None, bars=None, vmin=None, vmax=None, hide_index=False):
    """Display a table as HTML, optionally shaded or with in cell bars."""
    st = (pd.DataFrame(df).style
          .set_caption(caption).set_table_styles(TABLE_CSS)
          .format(_cell, na_rep=""))
    if grad is not None:
        st = st.background_gradient(cmap=SEQ, subset=grad, vmin=vmin, vmax=vmax)
    if bars is not None:
        st = st.bar(subset=bars, color="#cde2fb", vmin=vmin, vmax=vmax)
    if hide_index:
        st = st.hide(axis="index")
    display(st)

def facts(pairs, caption=""):
    """A two column table for a handful of single numbers."""
    show(pd.DataFrame({"value": [v for _, v in pairs]}, index=[k for k, _ in pairs]), caption)

def note(md):
    display(Markdown(md))

def quote(text, caption=""):
    """Show a report with its line breaks intact.

    A markdown blockquote only quotes up to the first newline, and these reports are several
    lines long, so the rest would render as ordinary text.
    """
    display(HTML(
        f"<div style='margin:0.4rem 0'>"
        f"<div style='font-weight:600;font-size:0.95rem;padding-bottom:0.3rem'>{_esc(caption)}</div>"
        f"<pre style='white-space:pre-wrap;word-break:break-word;font-size:0.82rem;"
        f"line-height:1.45;background:#f7f6f2;border-left:3px solid #2a78d6;"
        f"padding:0.6rem 0.8rem;margin:0'>{_esc(str(text))}</pre></div>"))

def short_uid(u, n=10):
    """The last n characters of a study identifier, which is enough to tell rows apart."""
    return ".." + str(u)[-n:]

# %% [markdown]
# ## 1. The task and how it is scored
#
# Each study is one knee MRI session. A session holds several series, and a series is a stack of
# images taken with one set of scanner settings. You have to give each study twelve numbers, one
# for each finding:
#
# | Column | What it means |
# |---|---|
# | ACL, MCL | Tear of the anterior cruciate or medial collateral ligament |
# | Medial Meniscus, Lateral Meniscus | Tear of the meniscus on the inner or outer side |
# | Medial OA, Lateral OA, PF OA | Osteoarthritis in the inner, outer, or kneecap compartment |
# | Effusion | Extra fluid in the joint |
# | Synovitis | Inflamed joint lining |
# | Baker's | A fluid filled cyst behind the knee |
# | Contusion | A bone bruise |
# | Fracture | A break in the bone |
#
# The score is the average of twelve ROC AUC values, one per column. ROC AUC is the chance that a
# study which has the finding gets a higher number than a study which does not. A score of 0.5 is
# the same as guessing and 1.0 is perfect.
#
# Two things follow from that, and both remove work rather than adding it.
#
# The first is that only the order of your numbers inside a column is read. If you sort a column
# and replace each value by its position, the score does not change. So you do not need to
# calibrate anything, you do not need to pick a threshold, and when you combine two models you
# should average their ranks rather than their probabilities. Averaging probabilities lets
# whichever model happens to be more confident decide the answer.
#
# The second is that all twelve findings cost the same. Write M for the average AUC a good model
# reaches. A column left at 0.5 gives up (M minus 0.5) divided by 12, no matter how well the
# other eleven do. At M of 0.85 that is 0.029 of your final score, which is much larger than the
# gap between neighbouring places near the top of the board. So a rare finding deserves more
# attention than a common one, because a rare finding is where a model most easily ends up
# guessing.

# %% [markdown]
# ## 2. What is in the files
#
# There are five CSV files and two folders of images. `train.csv` holds the report and the twelve
# label columns. `train_series.csv` describes every series. The images sit at
# `train_series/<study>/<series>/<file>.dcm`, one DICOM file per slice.

# %%
show(pd.DataFrame([
    {"file": f, "rows": len(d), "columns": len(d.columns),
     "carries the report": "Report" in d.columns,
     "carries the twelve labels": all(c in d.columns for c in LABELS)}
    for f, d in [("train.csv", train), ("train_series.csv", train_series),
                 ("test.csv", test), ("test_series.csv", test_series)]]),
     "What each file holds", hide_index=True)

show(pd.DataFrame({"column in train.csv": list(train.columns),
                   "also in test.csv": [c in test.columns for c in train.columns],
                   "filled in": [f"{train[c].notna().mean():.1%}" for c in train.columns]}),
     "Every column of train.csv, and whether it survives into test.csv", hide_index=True)

quote(reports.iloc[0][:500], "One report, the first 500 characters")

sample = pd.read_csv(COMP / "sample_submission.csv").head(3)
sample.insert(0, "study", sample.StudyInstanceUID.map(short_uid))
show(sample.drop(columns="StudyInstanceUID"),
     "sample_submission.csv, with the study identifier cut to its last ten characters",
     hide_index=True)

# %% [markdown]
# `test.csv` has one column and it is the study identifier. That single fact decides the shape of
# every solution to this competition. The report is available while you fit and absent when you
# predict, so you cannot build a model that reads text and images together. The report can only
# be used to make training labels.

# %%
per_study = train_series.groupby("StudyInstanceUID").size()
planes = train_series.Anatomical_Plane.value_counts()
plane_per_study = train_series.pivot_table(index="StudyInstanceUID", columns="Anatomical_Plane",
                                           values="SeriesInstanceUID", aggfunc="count").fillna(0)

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 2.9))

a1.hist(per_study, bins=range(2, 16), color=BLUE, rwidth=0.8)
a1.set_xlabel("series in one study"); a1.set_ylabel("studies")
a1.set_title(f"A study holds {per_study.median():.0f} series in the middle case")

a2.bar(planes.index, planes.values, color=[BLUE, ORANGE, AQUA])
for i, v in enumerate(planes.values):
    a2.text(i, v + 200, f"{v:,}", ha="center", fontsize=8)
a2.set_ylim(0, planes.max() * 1.18); a2.set_ylabel("series")
a2.set_title("Sagittal is the most common plane")

share = (plane_per_study > 0).mean().sort_values(ascending=False)
a3.barh(share.index, share.values, color=BLUE)
for i, v in enumerate(share.values):
    a3.text(v - 0.06, i, f"{v:.1%}", va="center", ha="right", color="white", fontsize=8)
a3.set_xlim(0, 1.05); a3.set_xlabel("share of studies with at least one")
a3.set_title("Every study has a sagittal series")
a3.grid(axis="y", visible=False)

plt.tight_layout(); plt.show()

facts([("fewest series in a study", int(per_study.min())),
       ("middle", int(per_study.median())),
       ("most", int(per_study.max())),
       ("studies with no sagittal series", int((plane_per_study.get("Sagittal", 0) == 0).sum()))],
      "Series per study")

# %% [markdown]
# ### The two scanner flags carry one fact, not two
#
# `train_series.csv` gives each series two flags. `Fluid_Sensitive` says fluid appears bright in
# the image, which is what makes an effusion easy to see. `Fat_Suppression` says the scanner was
# set up to darken fat, which is what makes swelling in bone marrow stand out. These are two
# different scanner settings and in general they vary on their own.
#
# In this file they do not. Check whether the two columns ever disagree.

# %%
agree = (train_series.Fluid_Sensitive == train_series.Fat_Suppression).mean()
note(f"The two flags hold the same value on **{agree:.2%}** of the {len(train_series):,} "
     f"series rows.")
show(pd.crosstab(train_series.Fluid_Sensitive, train_series.Fat_Suppression),
     "Fluid_Sensitive against Fat_Suppression, series counts")
show(pd.crosstab(train_series.Anatomical_Plane, train_series.Fluid_Sensitive),
     "Anatomical_Plane against Fluid_Sensitive, series counts")

# %% [markdown]
# The two columns are equal on every one of the 24,371 rows. So the file gives you one piece of
# information about the scanner settings, under two names. If you want the two properties apart
# you have to read them out of the DICOM headers, which section 6 looks at.
#
# For a first model this is enough. It says which series show fluid brightly, and that is the
# series to feed the model, because five of the twelve findings are about fluid or swelling.

# %% [markdown]
# ## 3. The 58 studies that carry labels
#
# 58 studies have all twelve labels filled in by a radiologist looking at the images. The other
# 4,349 have every label blank. Look at what those 58 contain.

# %%
prev = gold.mean().sort_values(ascending=False)
co = pd.DataFrame(index=LABELS, columns=LABELS, dtype=float)
for a in LABELS:
    for b in LABELS:
        co.loc[a, b] = gold.loc[gold[a] == 1, b].mean() if gold[a].sum() else np.nan

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.15]})

a1.barh(prev.index[::-1], prev.values[::-1], color=BLUE)
for i, (c, v) in enumerate(zip(prev.index[::-1], prev.values[::-1])):
    a1.text(v + 0.012, i, f"{v:.0%}  ({int(gold[c].sum())})", va="center", fontsize=7.5)
a1.set_xlim(0, 0.78); a1.set_xlabel("share of the 58 studies with the finding")
a1.set_title("Every finding is common in the 58,\nand the rarest has 9 positives")
a1.grid(axis="y", visible=False)

order = prev.index.tolist()
im = a2.imshow(co.loc[order, order].values.astype(float), cmap=SEQ, vmin=0, vmax=1)
a2.set_xticks(range(12)); a2.set_xticklabels(order, rotation=90, fontsize=7)
a2.set_yticks(range(12)); a2.set_yticklabels(order, fontsize=7)
a2.set_title("Given the finding in the row,\nhow often the column is also present")
a2.grid(False)
plt.colorbar(im, ax=a2, shrink=0.85)
plt.tight_layout(); plt.show()

facts([("labelled studies", len(gold)),
       ("studies with no finding at all", int((gold.sum(axis=1) == 0).sum())),
       ("findings per study, average", f"{gold.sum(axis=1).mean():.1f}"),
       ("findings per study, most", int(gold.sum(axis=1).max())),
       ("rarest finding", f"{gold.sum().idxmin()}, {int(gold.sum().min())} positives")],
      "The 58 labelled studies")

# %% [markdown]
# These 58 studies are sick knees. Every finding is present in at least 15% of them, and the
# average study has four of the twelve findings. That is not what a random hospital week looks
# like, so treat these prevalences as a property of this subset and not of the test set.
#
# ### What 58 studies can and cannot measure
#
# It is tempting to use these 58 studies to choose between two ideas. Measure how much a number
# from 58 studies can move on its own before you do. The check below resamples the 58 studies
# with replacement 2,000 times and reads the AUC again each time.

# %%
def bootstrap_auc(y: pd.DataFrame, p: pd.DataFrame, n_boot=2000, seed=0):
    """95% interval for each finding's AUC and for the average, from resampling the rows."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    per, mean = {c: [] for c in LABELS}, []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals = []
        for c in LABELS:
            yc = y[c].values[i]
            if yc.min() == yc.max():
                continue
            a = roc_auc_score(yc, p[c].values[i])
            per[c].append(a); vals.append(a)
        if len(vals) == len(LABELS):
            mean.append(np.mean(vals))
    width = {c: float(np.diff(np.percentile(v, [2.5, 97.5]))[0]) for c, v in per.items()}
    return width, float(np.diff(np.percentile(mean, [2.5, 97.5]))[0])

# A stand in for a real model: the annotation itself with noise added, so the check measures the
# sample size rather than any particular prediction.
rng = np.random.default_rng(SEED)
noisy = gold + rng.normal(0, 0.9, gold.shape)
w_per, w_mean = bootstrap_auc(gold, pd.DataFrame(noisy, index=gold.index, columns=LABELS))

show(pd.DataFrame({"95% interval width": pd.Series(w_per)})
     .sort_values("95% interval width", ascending=False),
     "How far one finding's AUC moves on 58 studies, from resampling the rows alone",
     bars=["95% interval width"], vmin=0, vmax=0.4)
facts([("widest single finding", round(max(w_per.values()), 3)),
       ("the average over twelve findings", round(w_mean, 3))],
      "The same measurement, summarised")

# %% [markdown]
# One finding's AUC on 58 studies moves by 0.2 to 0.3 from resampling alone. The average over
# twelve findings is steadier, at about 0.07, because averaging cancels some of the noise.
#
# The rule that follows is worth keeping. Use the 58 studies to check that a model is not broken,
# because a mean AUC near 0.5 there means something is wrong. Do not use them to choose between
# two models that are 0.02 apart, because 58 studies cannot see a gap that small. For choosing,
# you need a target that covers all 4,407 studies, and section 5 builds one.

# %% [markdown]
# ## 4. The reports
#
# Every training study carries the report the radiologist wrote. This is the only description of
# 4,349 studies that you have, so it is worth knowing what it looks like.

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
    """Lower case, strip accents, and map the Turkish dotless i onto a plain i.

    The dotless i has no accent to strip, so without this line every Turkish word containing it
    fails to match a pattern written with a plain i.
    """
    s = str(s).replace("ı", "i").replace("İ", "i").replace("đ", "d").replace("Đ", "d")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()

def detect_language(text: str) -> str:
    """Greek and Bulgarian are decided by the alphabet. The rest are scored on section
    headers first, then on common words, then on which accented letters appear."""
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

NAMES = {"en": "English", "es": "Spanish", "tr": "Turkish", "hr": "Croatian", "el": "Greek",
         "de": "German", "bg": "Bulgarian", "nl": "Dutch", "fr": "French", "unk": "unknown"}
lang = reports.map(detect_language)
counts = lang.value_counts()

facts([("reports", len(reports)),
       ("distinct texts", int(reports.nunique())),
       ("studies sharing a report with another", int(len(reports) - reports.nunique())),
       ("largest group sharing one report", int(reports.value_counts().iloc[0])),
       ("shortest report, characters", int(reports.str.len().min())),
       ("middle", int(reports.str.len().median())),
       ("longest", int(reports.str.len().max())),
       ("not written in English", f"{(lang != 'en').mean():.1%}")],
      "The reports")

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})

names = [NAMES[k] for k in counts.index]
a1.bar(names, counts.values, color=[BLUE if k == "en" else ORANGE for k in counts.index])
for i, v in enumerate(counts.values):
    a1.text(i, v + 25, f"{v:,}", ha="center", fontsize=7.5)
a1.set_ylim(0, counts.max() * 1.15); a1.set_ylabel("studies")
a1.set_title("The reports are written in nine languages")
a1.tick_params(axis="x", rotation=45)
for t in a1.get_xticklabels():
    t.set_ha("right")

a2.hist(reports.str.len(), bins=50, color=BLUE)
a2.axvline(reports.str.len().median(), color=INK, ls=":", lw=1.2)
a2.text(reports.str.len().median() + 90, a2.get_ylim()[1] * 0.9,
        f"median {int(reports.str.len().median())}", fontsize=8)
a2.set_xlabel("characters in the report"); a2.set_ylabel("studies")
a2.set_title("Most reports are a few short paragraphs")
plt.tight_layout(); plt.show()

# %% [markdown]
# Two facts here change what you do next.
#
# 60% of the reports are not in English, and two of the nine languages are not written in the
# Latin alphabet. Any word list you write for reading these reports has to cover all nine or it
# will quietly return "nothing found" for whole languages. Section 5 measures that.
#
# 131 studies share a report with another study. The largest group is 37 studies with one
# identical text, which is a template a radiologist reuses for a knee with nothing wrong. If the
# training labels come from the report, then every study in such a group gets the same label
# vector. Splitting that group across a training and validation divide would let the model see
# the answer for a validation study during training. So the split below is made on a hash of the
# report text, which keeps each group whole.

# %% [markdown]
# ## 5. Where the training labels come from
#
# 58 labelled studies are not enough to train an image model, and the reports describe all 4,407.
# So the reports have to be turned into labels. There are two ways to do it and this section
# measures both, because the quality of these labels is the ceiling on everything after.
#
# ### A word list, written here
#
# The first way is to look for words. For each finding, list the words a radiologist uses for the
# body part and the words used for the problem, in all nine languages. Split the report into
# clauses, and for each clause ask whether a body part word and a problem word appear close
# together. If they do, the finding is present. If the clause also contains a word of denial such
# as "no" or "sin" or "geen", the finding is absent.
#
# Five findings need no body part word, because the word for the problem names the location
# already. There is only one place a Baker's cyst can be.

# %%
def rx(*pats): return re.compile("|".join(pats))

A_MED_MEN = rx(r"menisc\w* (?:interno|medial|mediale)", r"medial menisc\w*", r"mediale menisc\w*",
               r"innenmeniskus", r"menisque (?:interne|medial)", r"medial menisk\w*", r"ic menisk",
               r"unutrasnj\w* menisk", r"esw menisk", r"εσω μηνισκ", r"μηνισκ\w* εσω",
               r"вътрешния менис", r"медиалния менис")
A_LAT_MEN = rx(r"menisc\w* (?:externo|lateral|laterale)", r"lateral menisc\w*", r"laterale menisc\w*",
               r"aussenmeniskus", r"menisque (?:externe|lateral)", r"lateral menisk\w*", r"dis menisk",
               r"vanjsk\w* menisk", r"εξω μηνισκ", r"μηνισκ\w* εξω",
               r"външния менис", r"латералния менис")
A_ACL = rx(r"\bacl\b", r"anterior cruciate", r"cruzado anterior", r"\blca\b", r"voorste kruisband",
           r"\bvkb\b", r"vorder\w* kreuzband", r"croise anterieur", r"on capraz", r"\bocb\b",
           r"prednj\w* ukrizen", r"προσθι\w* χιαστ", r"предна кръстн")
A_MCL = rx(r"\bmcl\b", r"medial collateral", r"colateral (?:medial|interno)", r"innenband",
           r"mediale collaterale", r"\blcm\b", r"ic yan bag", r"medial yan bag",
           r"ligament collateral (?:medial|interne)", r"medijaln\w* kolateral",
           r"εσω πλαγι", r"вътрешния колатерал", r"медиалния колатерал")
A_MEDC = rx(r"\bmedial\b", r"\bmediale\b", r"\binterno\b", r"\binterna\b", r"\binnen", r"\bic\b",
            r"medijaln", r"unutrasnj", r"εσω", r"медиал", r"вътрешн")
A_LATC = rx(r"\blateral\b", r"\blaterale\b", r"\bexterno\b", r"\bexterna\b", r"\baussen",
            r"\bdis\b", r"lateraln", r"vanjsk", r"εξω", r"латерал", r"външн")
A_PF = rx(r"patell?o-?femoral", r"femoro-?patell", r"femoropatelar", r"retropatell",
          r"femoro-?patellaire", r"patellar cartilag", r"cartilag\w* (?:of the )?patell",
          r"trochlea", r"troklea", r"επιγονατιδομηριαι", r"пателофеморал")

P_TEAR = rx(r"\btear", r"\btorn\b", r"ruptur", r"rotura", r"\bdesgarr", r"disrupt", r"scheur",
            r"\briss", r"yirtik", r"yirtil", r"macerat", r"puknu", r"lezij\w* menisk",
            r"ρηξη", r"ρηγμα", r"разкъс", r"руптур")
P_OA = rx(r"osteoarthrit", r"arthros", r"artros", r"gonarthros", r"gonartros", r"degenerat",
          r"dejenerat", r"joint space narrowing", r"pincement", r"osteofit", r"osteophyt",
          r"chondral thinning", r"cartilage (?:loss|thinning)", r"kraakbeen", r"knorpel",
          r"cartilag", r"kikirdak", r"hrskavic", r"οστεοαρθρ", r"αρθριτιδ", r"χονδροπαθ",
          r"артроз", r"остеоартр", r"хондропат")
P_EFF = rx(r"effusion", r"derrame", r"\bhydrops\b", r"gelenkerguss", r"\berguss\b", r"epanchement",
           r"versamento", r"joint fluid", r"efuzyon", r"eklem (?:ici )?sivi", r"artikuler sivi",
           r"vocht in het gewricht", r"intra-?articular fluid", r"izliv",
           r"συλλογη (?:υγρου|αρθρικου)", r"αρθρικη συλλογη", r"изли[вя]", r"хидропс")
P_SYN = rx(r"synovit", r"sinovit", r"synovial (?:thickening|proliferation|hypertroph)", r"sinovyal",
           r"sinovijalni", r"υμενιτιδ", r"αρθρικου υμενα", r"синовит")
P_BAK = rx(r"\bbaker", r"popliteal cyst", r"poplitea\w* cyst", r"bakerzyste", r"quiste de baker",
           r"kyste de baker", r"cisti di baker", r"poplitealna cist", r"popliteal kist",
           r"κυστη baker", r"киста на бейкър", r"поплитеална кист")
P_CON = rx(r"contusion", r"contusio", r"bone (?:bruise|contusion)", r"edema (?:oseo|osseo|medular)",
           r"bone marrow edema", r"beenmerg\w*oedeem", r"knochenmarkodem", r"kneuzing",
           r"oedeme osseux", r"kemik kontuzyon", r"kontuzyon", r"medullar\w* odem",
           r"οιδημα (?:μυελου|οστου)", r"костномозъчен оток", r"едем на костния")
P_FRA = rx(r"fractur", r"fractuur", r"fraktur", r"\bfissur", r"frattura", r"kirik", r"prijelom",
           r"breuk", r"καταγμα", r"фрактур", r"счупван")

NEG = rx(r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative\b", r"\bintact\b", r"\bnormal\b",
         r"\bunremarkable\b", r"\bpreserved\b", r"\bsin\b", r"\bausencia\b", r"\bausente",
         r"\bgeen\b", r"\bkein", r"\bpas de\b", r"\bsans\b", r"\bnessun", r"\byok\b", r"\bnema\b",
         r"\bbez\b", r"δεν ", r"ουδεν", r"χωρις", r"без ", r"липсва")
SEV_LOW = rx(r"\btrace\b", r"\bsmall\b", r"\bminimal\b", r"\bmild\b", r"\bslight\b", r"\bscant\b",
             r"\bleve\b", r"\bminim", r"\bpeque", r"\bhafif\b", r"\bgering", r"\bdiscret",
             r"\bblag", r"\bmicro")
SEV_HIGH = rx(r"\bmoderate\b", r"\blarge\b", r"\bmassive\b", r"\bmarked\b", r"\bgross\b",
              r"\bsignificant\b", r"\bextensive\b", r"\bmoderado\b", r"\babundante\b",
              r"\bimportante\b", r"\bbelirgin\b", r"\bausgepragt", r"\bveel\b", r"\bizrazit")

CLAUSE = re.compile(r"[.;:\n]+")
PAIRED = {"ACL": (A_ACL, P_TEAR), "MCL": (A_MCL, P_TEAR),
          "Medial Meniscus": (A_MED_MEN, P_TEAR), "Lateral Meniscus": (A_LAT_MEN, P_TEAR),
          "Medial OA": (A_MEDC, P_OA), "Lateral OA": (A_LATC, P_OA), "PF OA": (A_PF, P_OA)}
SOLO = {"Effusion": P_EFF, "Synovitis": P_SYN, "Baker's": P_BAK,
        "Contusion": P_CON, "Fracture": P_FRA}

def read_report(text: str, window: int = 70):
    """Return one score per finding, and whether the report mentioned it at all.

    A score of 0.9 means a clause asserts the finding, 0.65 that it calls it small, 0.95 that it
    calls it large, and 0.1 that a clause denies it. A finding no clause touched stays at 0.0,
    which is the weakest part of this method and section 9 comes back to it.
    """
    f = fold(text)
    score = {k: 0.0 for k in LABELS}
    seen = {k: False for k in LABELS}
    for clause in CLAUSE.split(f):
        if not clause.strip():
            continue
        if NEG.search(clause):
            value = 0.10
        elif SEV_HIGH.search(clause):
            value = 0.95
        elif SEV_LOW.search(clause):
            value = 0.65
        else:
            value = 0.90
        for k, pat in SOLO.items():
            if pat.search(clause):
                seen[k] = True
                score[k] = max(score[k], value)
        for k, (anat, path) in PAIRED.items():
            for m in anat.finditer(clause):
                lo, hi = max(0, m.start() - window), min(len(clause), m.end() + window)
                if path.search(clause[lo:hi]):
                    seen[k] = True
                    score[k] = max(score[k], value)
                    break
    return score, seen

parsed = [read_report(t) for t in reports]
wordlist = pd.DataFrame([s for s, _ in parsed], index=reports.index)[LABELS]
mentioned = pd.DataFrame([m for _, m in parsed], index=reports.index)[LABELS]
log(f"read {len(reports):,} reports with the word list")

# %%
from sklearn.metrics import roc_auc_score

def score_against_gold(pred: pd.DataFrame) -> pd.Series:
    p = pred.reindex(gold.index)
    return pd.Series({c: roc_auc_score(gold[c], p[c].fillna(0.0)) for c in LABELS})

wl_auc = score_against_gold(wordlist)
show(pd.DataFrame({"AUC on the 58": wl_auc, "share of reports that mentioned it":
                   mentioned.mean()}).sort_values("AUC on the 58", ascending=False),
     f"The word list against the radiologist, average {wl_auc.mean():.3f}",
     bars=["AUC on the 58"], vmin=0.4, vmax=1.0)
note(f"The word list found nothing at all in **{int((~mentioned.any(axis=1)).sum()):,}** of "
     f"{len(mentioned):,} studies, which then look like healthy knees.")

# %%
cov = (pd.DataFrame({"lang": lang, "fires": mentioned.any(axis=1)})
       .groupby("lang").agg(n=("fires", "size"), fires=("fires", "mean"))
       .sort_values("fires"))
by_lang = mentioned.groupby(lang).mean().loc[cov.index]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1, 1.35]})

a1.bar([NAMES[k] for k in cov.index], cov.fires.values,
       color=[ORANGE if v < 0.8 else BLUE for v in cov.fires])
for i, (v, n) in enumerate(zip(cov.fires, cov.n)):
    a1.text(i, v + 0.03, f"{v:.0%}\nn={n}", ha="center", fontsize=7)
a1.set_ylim(0, 1.2); a1.set_ylabel("reports where anything at all fired")
a1.set_title("Asking only whether something fired")
a1.tick_params(axis="x", rotation=45)
for t in a1.get_xticklabels():
    t.set_ha("right")

im = a2.imshow(by_lang.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
a2.set_xticks(range(12)); a2.set_xticklabels(LABELS, rotation=90, fontsize=7)
a2.set_yticks(range(len(by_lang))); a2.set_yticklabels([NAMES[k] for k in by_lang.index], fontsize=8)
a2.set_title("Asking per finding, which is the useful question")
a2.grid(False)
plt.colorbar(im, ax=a2, shrink=0.9, label="share of reports where the finding fired")
plt.tight_layout(); plt.show()

cov_named = cov.rename(index=NAMES).rename(
    columns={"n": "reports", "fires": "share where anything fired"})
show(cov_named, "Coverage as one number per language",
     bars=["share where anything fired"], vmin=0, vmax=1)
show(by_lang.rename(index=NAMES),
     "Coverage per finding, which is where the gaps actually are",
     grad=list(by_lang.columns), vmin=0, vmax=1)

# %% [markdown]
# The left chart is the measure most people reach for, and it is the wrong one. It says Bulgarian
# is fine at 97%. The right chart shows what happens in Bulgarian. One term fires, the one for an
# effusion, and the other eleven findings are near zero. So a Bulgarian report comes out as a knee
# with fluid in it and nothing else, every time.
#
# Turkish is the weakest language by the overall measure, at 60%, and 217 Turkish studies get
# nothing at all. Turkish builds words by adding endings, so the word for a torn meniscus is one
# long word rather than two, and a fixed list of patterns misses most of the forms it takes.
#
# The same reading applies elsewhere in that chart. Croatian never fires on either meniscus and
# German never fires on the lateral one. Neither is a fact about those patients. Both are terms
# missing from the list.
#
# A study where nothing fires gets twelve zeros, and the model then learns it as a healthy knee.
# This is the part that makes a word list expensive. It does not fail loudly. It reports that a
# whole language of patients has nothing wrong with them.
#
# Notice also that PF OA came out below 0.5, which is worse than guessing. Osteoarthritis behind
# the kneecap is graded from how thin the cartilage looks, and a word list cannot grade anything.
# It matches the word "patellofemoral" whether the sentence describes damage or only names the
# images that were taken.
#
# ### A language model reading the same reports
#
# The second way is to have a language model read each report and answer the twelve questions.
# Several people have done this and published the result as a Kaggle dataset. This notebook
# attaches one of them, from `flight0234/rsna-knee-hybrid-report-labels`, and measures it the
# same way.

# %%
SEARCH = [p for p in map(pathlib.Path, ["/kaggle/input/datasets", "/kaggle/input", "/tmp/rsnallm"])
          if p.exists()]

def find_file(name: str):
    """Find an attached dataset file without walking the competition folder.

    rglob under /kaggle/input would descend into the competition mount and its ~700,000 DICOM
    files, which costs minutes for every lookup.
    """
    for root in SEARCH:
        for pat in (name, f"*/{name}", f"*/*/{name}", f"*/*/*/{name}"):
            for hit in root.glob(pat):
                if "competitions" not in hit.parts:
                    return hit
    return None

PUBLISHED = {
    "language model": "report_labels_v4hybrid.csv",   # flight0234/rsna-knee-hybrid-report-labels
    "four sets merged": "report_labels_v5.csv",       # yunusgmsoy/rsna-knee-llm-labels-4-source-merged
}
published, loaded = {}, []
for name, fname in PUBLISHED.items():
    hit = find_file(fname)
    if hit is None:
        loaded.append({"labels": name, "file": fname, "found": False, "studies": 0})
        continue
    d = pd.read_csv(hit).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    ok = all(c in d.columns for c in LABELS)
    if ok:
        published[name] = d[LABELS].astype(float)
    loaded.append({"labels": name, "file": hit.name, "found": True,
                   "studies": int(len(d)) if ok else 0})
show(pd.DataFrame(loaded), "What is attached", hide_index=True)

rows = [{"labels": "word list, written here", "mean AUC on the 58": wl_auc.mean(),
         "studies covered": len(wordlist), "distinct values": int(wordlist.stack().nunique())}]
for name, d in published.items():
    rows.append({"labels": name, "mean AUC on the 58": score_against_gold(d).mean(),
                 "studies covered": len(d), "distinct values": int(d.stack().nunique())})
show(pd.DataFrame(rows).sort_values("mean AUC on the 58", ascending=False),
     "Three ways to label 4,407 studies from 4,407 reports",
     bars=["mean AUC on the 58"], vmin=0.5, vmax=1.0, hide_index=True)

# %% [markdown]
# ### Check that a published label set does not contain the answers
#
# One of those two sets scores exactly 1.000. A label set built only from report text has no way
# to reproduce a radiologist's reading of the images perfectly, so a perfect score means the 58
# answers were copied into the file. That is a reasonable thing to do when you train, because
# real labels are better than derived ones. It stops being reasonable the moment you use those
# same 58 studies to measure anything, which is the only thing they are otherwise for.
#
# The check needs two parts. On the 58 rows, is every value exactly 0.0 or 1.0 and equal to the
# annotation? On the other 4,349 rows, is that not true? The second part is what separates a
# copied answer key from a label set that rounds all of its output to 0 and 1 everywhere.

# %%
def contains_the_answers(labels: pd.DataFrame) -> dict:
    on = labels.reindex(gold.index)
    off = labels.loc[~labels.index.isin(gold.index)]
    exact_on = float(np.isin(on.values, [0.0, 1.0]).mean())
    exact_off = float(np.isin(off.values, [0.0, 1.0]).mean())
    match = float((on.values == gold.values).mean())
    return {"exactly 0 or 1 on the 58": exact_on,
            "exactly 0 or 1 elsewhere": exact_off,
            "equals the annotation": match,
            "distinct values elsewhere": int(len(np.unique(off.values))),
            "contains the answers": bool(exact_on > 0.999 and match > 0.999 and exact_off < 0.01)}

check = pd.DataFrame({name: contains_the_answers(d) for name, d in published.items()}).T
show(check, "The two part check, run on every attached label set")

TARGET_NAME = None
for name, d in published.items():
    if not contains_the_answers(d)["contains the answers"]:
        TARGET_NAME = name
        break

if TARGET_NAME is None:
    target = wordlist.reindex(train.StudyInstanceUID)
    TARGET_NAME = "word list, written here"
    note("No clean published set is attached, so the word list supplies the targets instead.")
else:
    target = published[TARGET_NAME].reindex(train.StudyInstanceUID)
    note(f"Training target: **{TARGET_NAME}**, average AUC on the 58 "
         f"**{score_against_gold(published[TARGET_NAME]).mean():.3f}**.")

target = target.fillna(0.0)

# %% [markdown]
# The merged set is exactly 0.0 or 1.0 on all 58 rows, matches every annotation, and is exactly
# 0.0 or 1.0 on none of the other 4,349 rows, where it takes hundreds of distinct values. The
# answers were written in. The other set is not like that, so this notebook trains on it.
#
# The gap between the two honest options is large. A word list reaches about 0.73 on the 58 and a
# language model reading the same text reaches about 0.90. That difference is the single biggest
# lever in this competition, and it is bigger than anything an image model of this size will add
# on top.

# %% [markdown]
# ## 6. What is inside the DICOM files
#
# Now the images. A DICOM file holds one slice of pixels plus a header describing how it was
# taken. Read the headers of one series per study for a sample of studies and look at what varies.

# %%
import pydicom
from pydicom.errors import InvalidDicomError

HDR = ["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber", "Rows", "Columns",
       "PixelSpacing", "SliceThickness", "Laterality", "Manufacturer", "SeriesDescription",
       "RepetitionTime", "EchoTime", "MagneticFieldStrength"]

def read_header(path: pathlib.Path):
    try:
        d = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=HDR)
    except (InvalidDicomError, OSError, AttributeError):
        return None
    return d

def pick_series(series_df: pd.DataFrame, folder: pathlib.Path) -> pd.Series:
    """One series per study: sagittal first, fluid sensitive next, then the thickest stack.

    Sagittal because both cruciate ligaments and both menisci are seen end on in that plane,
    and fluid sensitive because five of the twelve findings are about fluid or swelling. Where a
    study has several equally good candidates, the one with the most slices wins, since a thicker
    stack samples the joint more finely.
    """
    d = series_df.copy()
    d["rank"] = (d.Anatomical_Plane == "Sagittal") * 2 + d.Fluid_Sensitive.fillna(0)
    best = d[d["rank"] == d.groupby("StudyInstanceUID")["rank"].transform("max")]

    def n_files(row):
        p = folder / row.StudyInstanceUID / row.SeriesInstanceUID
        try:
            return sum(1 for _ in p.iterdir())
        except OSError:
            return 0

    with ThreadPoolExecutor(READ_THREADS) as ex:
        best = best.assign(n=list(ex.map(n_files, [r for _, r in best.iterrows()])))
    return (best.sort_values("n", ascending=False)
            .groupby("StudyInstanceUID").SeriesInstanceUID.first())

chosen_train = pick_series(train_series, COMP / "train_series")
log(f"chose one series for {len(chosen_train):,} of {len(train):,} training studies")

sample_ids = list(np.random.default_rng(SEED).permutation(chosen_train.index)[:250])
rows = []
for sid in sample_ids:
    folder = COMP / "train_series" / sid / chosen_train[sid]
    files = sorted(folder.glob("*.dcm"))
    if not files:
        continue
    d = read_header(files[len(files) // 2])
    if d is None:
        continue
    ps = getattr(d, "PixelSpacing", None)
    lat = getattr(d, "Laterality", None)
    ipp = getattr(d, "ImagePositionPatient", None)
    rows.append({
        "study": sid, "slices": len(files),
        "rows_px": int(getattr(d, "Rows", 0)), "cols_px": int(getattr(d, "Columns", 0)),
        "mm_per_pixel": float(ps[0]) if ps else np.nan,
        "slice_mm": float(getattr(d, "SliceThickness", np.nan) or np.nan),
        "laterality": str(lat) if lat else "",
        "manufacturer": str(getattr(d, "Manufacturer", "") or "").split()[0][:12].upper(),
        "tesla": float(getattr(d, "MagneticFieldStrength", np.nan) or np.nan),
        "x_mm": float(ipp[0]) if ipp else np.nan,
    })
hdr = pd.DataFrame(rows)
log(f"read headers for {len(hdr)} studies")

show(hdr[["slices", "rows_px", "cols_px", "mm_per_pixel", "slice_mm", "tesla"]]
     .describe().loc[["min", "25%", "50%", "75%", "max"]]
     .rename(index={"25%": "lower quarter", "50%": "middle", "75%": "upper quarter"}),
     f"What differs between studies, over a random {len(hdr)}")
show(hdr.manufacturer.value_counts().to_frame("studies"), "Who made the scanner")
show(hdr.laterality.replace("", "not recorded").value_counts().to_frame("studies"),
     f"The Laterality tag, recorded on {(hdr.laterality != '').mean():.0%} of studies")

# %%
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 2.9))

a1.hist(hdr.slices, bins=30, color=BLUE)
a1.set_xlabel("slices in the chosen series"); a1.set_ylabel("studies")
a1.set_title(f"Median {hdr.slices.median():.0f} slices per series")

a2.hist(hdr.mm_per_pixel.dropna(), bins=30, color=BLUE)
a2.set_xlabel("millimetres per pixel"); a2.set_ylabel("studies")
a2.set_title("Physical scale differs between studies")

lat_by_maker = (hdr.assign(has=hdr.laterality != "")
                .groupby("manufacturer").has.agg(["mean", "size"])
                .sort_values("mean", ascending=False))
a3.barh(lat_by_maker.index, lat_by_maker["mean"], color=BLUE)
for i, (v, n) in enumerate(zip(lat_by_maker["mean"], lat_by_maker["size"])):
    a3.text(min(v + 0.03, 0.72), i, f"{v:.0%}  n={n}", va="center", fontsize=7)
a3.set_xlim(0, 1.05); a3.set_xlabel("share with a Laterality tag")
a3.set_title("Whether the side is recorded\ndepends on the scanner maker")
a3.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()

# %% [markdown]
# Three things in that output change how the images have to be prepared.
#
# The number of slices per series is not fixed, so a model needs a rule for picking a fixed
# number of them. The millimetres per pixel is not fixed either, so resizing every image to the
# same number of pixels hands the model knees at different physical scales. A stronger model
# would crop a fixed number of millimetres instead. This one resizes, and that is one of the
# reasons it stops where it does.
#
# The `Laterality` tag says whether the left or the right knee was scanned. It is missing for a
# large share of studies, and whether it is missing depends on which company made the scanner.
# That matters more than it sounds, and the next two cells explain why.

# %% [markdown]
# ### The file names are not in slice order
#
# A series is a folder of files. The obvious way to read it is to sort the file names. That is
# wrong here, and it fails without any error. The file name is a unique identifier assigned by
# the scanner, not a position, so sorting by it gives a stack in random anatomical order.
#
# The true order is in the header. `ImagePositionPatient` gives the position of each slice in
# millimetres, in a coordinate system fixed to the patient, where x runs from the patient's right
# to the patient's left. `ImageOrientationPatient` gives the two directions of the image plane,
# and the cross product of those two is the direction the stack travels. Projecting the position
# onto that direction gives one number per slice that increases along the stack.
#
# Measure how far off the file name order is.

# %%
def slice_geometry(path: pathlib.Path):
    """Position along the stack, the stack direction, and the patient x coordinate."""
    d = read_header(path)
    if d is None:
        return None
    ipp = getattr(d, "ImagePositionPatient", None)
    iop = getattr(d, "ImageOrientationPatient", None)
    inst = getattr(d, "InstanceNumber", None)
    if ipp is None or iop is None or len(iop) != 6:
        return {"k": float(inst) if inst is not None else np.nan, "nx": np.nan,
                "x": np.nan, "from_geometry": False}
    p = np.asarray(ipp, dtype=float)
    r = np.asarray(iop, dtype=float)
    n = np.cross(r[:3], r[3:])
    return {"k": float(p @ n), "nx": float(n[0]), "x": float(p[0]), "from_geometry": True}

from scipy.stats import spearmanr

probe = []
for sid in sample_ids[:40]:
    files = sorted((COMP / "train_series" / sid / chosen_train[sid]).glob("*.dcm"))
    if len(files) < 5:
        continue
    with ThreadPoolExecutor(READ_THREADS) as ex:
        geo = list(ex.map(slice_geometry, files))
    g = [x for x in geo if x is not None and np.isfinite(x["k"])]
    if len(g) < 5:
        continue
    ks = [x["k"] for x in g]
    rho = float(spearmanr(np.arange(len(ks)), ks)[0])
    probe.append({"study": sid, "n": len(ks), "rho_filename_vs_position": rho,
                  "geometry_present": all(x["from_geometry"] for x in g)})
probe = pd.DataFrame(probe)
assert len(probe), "no series could be read, so the order check cannot run"

show(probe.rho_filename_vs_position.describe()
     .to_frame("rank correlation")
     .rename(index={"25%": "lower quarter", "50%": "middle", "75%": "upper quarter"}),
     f"File name order against position along the stack, over {len(probe)} studies")
note(f"The header carries both position and orientation on "
     f"**{probe.geometry_present.mean():.0%}** of those studies, so the true order is always "
     f"recoverable here.")

# %%
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.hist(probe.rho_filename_vs_position, bins=np.linspace(-1, 1, 41), color=ORANGE)
ax.axvline(0, color=INK, ls=":", lw=1.2)
ax.set_xlabel("rank correlation, file name order against position")
ax.set_ylabel("studies")
ax.set_title("Sorting a series by file name gives an order unrelated to the anatomy")
plt.tight_layout(); plt.show()

# %% [markdown]
# The correlation sits around zero. So sorting by file name and then taking the middle slices
# gives a random handful of cross sections rather than the middle of the joint.
#
# ### Putting every knee the same way round
#
# Five of the twelve findings name a side of the knee. Medial means the side facing the other
# leg and lateral means the side facing away. Which side of the picture that falls on depends on
# whether the left or the right knee was scanned. If you do not correct for it, those five labels
# are being learned from a direction the model cannot see, and a horizontal flip as data
# augmentation would turn a medial tear into a lateral one.
#
# The correction can be read from the geometry, which every slice carries, rather than from the
# `Laterality` tag, which many studies do not. The knee's x position tells you which leg it is,
# because the left leg sits on the positive x side of the body's midline. The x part of the stack
# direction tells you which way the stack travels. Put together, they say whether the slices run
# from medial to lateral or the other way, and the ones that run the wrong way get reversed.
#
# Check that rule against the `Laterality` tag on the studies that have one.

# %%
def series_order(files: list, threads: int = 0):
    """Sort a series into medial to lateral order and say which knee it is.

    Returns the files in order, plus 'L' or 'R' or '' when the geometry is missing. `threads`
    is 0 when this runs inside a worker thread already, because nesting one pool inside
    another multiplies the thread count.
    """
    if threads:
        with ThreadPoolExecutor(threads) as ex:
            geo = list(ex.map(slice_geometry, files))
    else:
        geo = [slice_geometry(f) for f in files]
    pairs = [(g, f) for g, f in zip(geo, files) if g is not None and np.isfinite(g["k"])]
    if not pairs:
        return files, ""
    pairs.sort(key=lambda t: t[0]["k"])
    xs = [g["x"] for g, _ in pairs if np.isfinite(g["x"])]
    nxs = [g["nx"] for g, _ in pairs if np.isfinite(g["nx"])]
    ordered = [f for _, f in pairs]
    if not xs or not nxs:
        return ordered, ""
    side = 1.0 if np.mean(xs) > 0 else -1.0        # positive x is the patient's left
    step = 1.0 if np.mean(nxs) > 0 else -1.0       # which way x moves along the stack
    if side * step < 0:
        ordered = ordered[::-1]
    return ordered, ("L" if side > 0 else "R")

tag_by_study = hdr.set_index("study").laterality
rows = []
for sid in sample_ids[:120]:
    files = sorted((COMP / "train_series" / sid / chosen_train[sid]).glob("*.dcm"))
    if len(files) < 5:
        continue
    _, side = series_order(files, READ_THREADS)
    tag = tag_by_study.get(sid, "")
    rows.append({"study": sid, "from_geometry": side, "from_tag": tag})
sides = pd.DataFrame(rows)
both = sides[(sides.from_geometry != "") & (sides.from_tag != "")]

facts([("studies where the geometry gives a side", f"{(sides.from_geometry != '').mean():.0%}"),
       ("studies where the tag gives a side", f"{(sides.from_tag != '').mean():.0%}"),
       ("the two agree", f"{(both.from_geometry == both.from_tag).mean():.1%} of {len(both)}"
                          if len(both) else "no study carries both")],
      f"Two ways to find out which knee was scanned, over {len(sides)} studies")
show(pd.crosstab(sides.from_geometry.replace("", "not found"),
                 sides.from_tag.replace("", "not recorded")),
     "The side from the geometry, in rows, against the side from the tag, in columns")

# %% [markdown]
# ### One study, seen as a stack
#
# Before building the whole training set, look at one study in the order the code above produces.

# %%
import cv2

def read_pixels(path, size=SIZE):
    """One slice, clipped at the 0.5 and 99.5 percentiles and resized."""
    try:
        d = pydicom.dcmread(str(path))
        a = d.pixel_array.astype(np.float32)
    except Exception:
        return np.zeros((size, size), np.uint8)
    slope = float(getattr(d, "RescaleSlope", 1) or 1)
    inter = float(getattr(d, "RescaleIntercept", 0) or 0)
    a = a * slope + inter
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return (cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)

demo, demo_files = None, []
for sid in sample_ids:
    files = sorted((COMP / "train_series" / sid / chosen_train[sid]).glob("*.dcm"))
    if len(files) >= N_SLICES:
        demo, demo_files = sid, files
        break
assert demo is not None, "no study had enough slices to show"
demo_ordered, demo_side = series_order(demo_files, READ_THREADS)
pick = np.linspace(len(demo_ordered) * 0.1, len(demo_ordered) * 0.9 - 1,
                   N_SLICES).round().astype(int)
demo_stack = np.stack([read_pixels(demo_ordered[i]) for i in pick])

fig, axes = plt.subplots(2, 6, figsize=(11, 4))
for j, ax in enumerate(axes.ravel()):
    ax.imshow(demo_stack[j], cmap="gray")
    ax.set_title(f"slice {pick[j]}", fontsize=7)
    ax.axis("off")
fig.suptitle(f"One sagittal series in medial to lateral order, {demo_side} knee, "
             f"{len(demo_files)} slices reduced to {N_SLICES}", y=1.0, fontsize=9.5)
plt.tight_layout(); plt.show()

quote(reports[demo][:600], f"The report for that study, {NAMES[detect_language(reports[demo])]}, first 600 characters")

# %% [markdown]
# ## 7. Building one input per study
#
# The model gets one array per study, of twelve slices at 224 by 224. Twelve slices are taken at
# even spacing from the middle 80% of the stack, because the first and last few slices of a knee
# series are usually outside the joint.
#
# The whole set is read once and held in memory as 8 bit integers, because reading it again every
# epoch would cost far more than the training itself. Twelve slices at 224 by 224 for 4,407
# studies is about 2.7 GB, which fits.
#
# Every stage checks the clock. If reading is slower than expected, the training set gets smaller
# rather than the run failing.

# %%
def build_study(args):
    """One study, as twelve slices in medial to lateral order."""
    folder, sid, series = args
    out = np.zeros((N_SLICES, SIZE, SIZE), np.uint8)
    files = sorted((folder / sid / series).glob("*.dcm"))
    if not files:
        return sid, out, False
    ordered, _ = series_order(files)
    lo, hi = len(ordered) * 0.1, len(ordered) * 0.9 - 1
    idx = np.linspace(max(lo, 0), max(hi, 0), N_SLICES).round().astype(int)
    idx = np.clip(idx, 0, len(ordered) - 1)
    for j, i in enumerate(idx):
        out[j] = read_pixels(ordered[i])
    return sid, out, True

def build_set(ids, chosen, folder, budget_s, tag):
    X = np.zeros((len(ids), N_SLICES, SIZE, SIZE), np.uint8)
    kept, failed = [], 0
    start = time.time()
    jobs = [(folder, sid, chosen[sid]) for sid in ids]
    with ThreadPoolExecutor(READ_THREADS) as ex:
        for n, (sid, vol, ok) in enumerate(ex.map(build_study, jobs)):
            if ok:
                X[len(kept)] = vol
                kept.append(sid)
            else:
                failed += 1
            if (n + 1) % 250 == 0:
                rate = (n + 1) / (time.time() - start)
                log(f"{tag}: {n + 1:,}/{len(ids):,} studies, {rate:.1f}/s, "
                    f"{failed} unreadable")
            if time.time() - start > budget_s or time.time() > HARD_DEADLINE - 1800:
                log(f"{tag}: stopping at {n + 1:,} studies to stay inside the time budget")
                break
    log(f"{tag}: kept {len(kept):,} studies, {failed} unreadable, "
        f"{time.time() - start:.0f}s")
    return X[:len(kept)], kept

rng = np.random.default_rng(SEED)
train_ids = list(chosen_train.index)
rng.shuffle(train_ids)
# Keep every labelled study, so the check in section 9 is always possible.
train_ids = [s for s in gold.index if s in chosen_train.index] + \
            [s for s in train_ids if s not in set(gold.index)]

Xtr, kept_ids = build_set(train_ids, chosen_train, COMP / "train_series", PREP_BUDGET_S, "train")
gc.collect()
facts([("studies read", len(kept_ids)),
       ("array shape", " by ".join(str(d) for d in Xtr.shape)),
       ("memory held", f"{Xtr.nbytes / 1e9:.2f} GB"),
       ("labelled studies inside it", f"{len(set(kept_ids) & set(gold.index))} of {len(gold)}")],
      "The training array")

# %% [markdown]
# ### Splitting on the report text
#
# One fifth of the studies are held out. The split is decided by a hash of the report text so
# that studies sharing a report stay on the same side, for the reason section 4 gives.
#
# The 58 labelled studies are held out as well, whichever bucket their hash falls in, and so is
# any study that shares a report with one of them. Without that, the model would be trained on
# the images of the studies used to check it against the radiologist, and that number would say
# how well the model remembers rather than how well it reads. Holding out the whole report group
# costs about 60 training studies out of 4,407, and the count printed below is the check that no
# group ended up on both sides.

# %%
def bucket(sid: str) -> int:
    h = hashlib.md5(fold(reports.get(sid, sid)).encode()).hexdigest()
    return int(h[:8], 16) % 100

gold_text = set(reports.reindex(gold.index).map(fold))
is_val = np.array([bucket(s) < HOLDOUT * 100 or fold(reports.get(s, "")) in gold_text
                   for s in kept_ids])
Ytr = target.reindex(kept_ids).values.astype(np.float32)

groups = reports.reindex(kept_ids).value_counts()
shared = groups[groups > 1].index
leaked = sum(1 for t in shared
             if len(set(is_val[[i for i, s in enumerate(kept_ids) if reports[s] == t]])) > 1)
facts([("studies used for training", int((~is_val).sum())),
       ("studies held out", int(is_val.sum())),
       ("labelled studies used for training", len(set(np.array(kept_ids)[~is_val]) & set(gold.index))),
       ("labelled studies held out", f"{len(set(np.array(kept_ids)[is_val]) & set(gold.index))}"
                                     f" of {len(gold)}"),
       ("report groups split across the divide", leaked)],
      "The split, and the two checks on it")

# %%
fig, ax = plt.subplots(figsize=(7.4, 2.6))
m = Ytr[~is_val].mean(axis=0)
o = np.argsort(-m)
ax.bar(np.array(LABELS)[o], m[o], color=BLUE)
for i, v in enumerate(m[o]):
    ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=7)
ax.set_ylim(0, m.max() * 1.2); ax.set_ylabel("average target value")
ax.set_title(f"What the model is asked to predict, from the {TARGET_NAME} labels")
ax.tick_params(axis="x", rotation=45)
for t in ax.get_xticklabels():
    t.set_ha("right")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. A small model
#
# A ResNet18 that was trained on ImageNet, with two changes.
#
# The first layer normally takes three colour channels. Here it takes twelve, one per slice, so
# the twelve values at one pixel are the brightness of that spot as the stack passes through the
# joint. The first layer's weights are averaged across the three colours, copied twelve times,
# and divided by four so that the numbers coming out of the layer keep roughly the size the rest
# of the network expects.
#
# The last layer becomes twelve outputs instead of a thousand, one per finding.
#
# The training targets are values between 0 and 1 rather than 0 or 1, because they come from a
# language model that reports how sure it is. Binary cross entropy accepts that directly and
# treats a target of 0.7 as seven parts positive to three parts negative.
#
# There is no horizontal flip in the augmentation. Mirroring an image of a knee turns it into an
# image of the other knee, which relabels medial as lateral, so five of the twelve findings would
# be trained against the wrong answer. Shifting the image and changing its brightness are safe.
#
# The predictions come from the three epochs that scored best on the held out fifth, combined by
# averaging their ranks. Section 1 said the score reads only the order, so ranks are the right
# thing to average, and three epochs of one run cost nothing extra to keep.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.manual_seed(SEED)
np.random.seed(SEED)     # batches() draws from the global generator
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    # A real matrix multiply, so a broken GPU image fails here rather than in an hour.
    _ = (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
log(f"device {dev}  {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")

def build_model(n_ch=N_SLICES, n_out=len(LABELS)):
    m = torchvision.models.resnet18(weights=None)
    # find_file is bounded on purpose. An rglob under /kaggle/input would walk the competition
    # mount and its ~700,000 DICOM files.
    w = find_file("resnet18.pth") or find_file("resnet18-f37072fd.pth")
    if w is not None:
        sd = torch.load(str(w), map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        m.load_state_dict(sd)
        note(f"Loaded ImageNet weights from `{w}`.")
    else:
        note("No ImageNet weights are attached, so the network starts from random values.")
    old = m.conv1.weight.data
    m.conv1 = nn.Conv2d(n_ch, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = old.mean(1, keepdim=True).repeat(1, n_ch, 1, 1) * (3.0 / n_ch)
    m.fc = nn.Linear(512, n_out)
    return m

model = build_model().to(dev)
facts([("architecture", "ResNet18"),
       ("input", f"{N_SLICES} slices at {SIZE} by {SIZE}"),
       ("outputs", len(LABELS)),
       ("parameters", f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} million"),
       ("device", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")],
      "The model")

# %%
def augment(x: torch.Tensor) -> torch.Tensor:
    """Shift by up to 8% of the image, scale the brightness, add a little noise."""
    b = x.shape[0]
    dx, dy = (torch.rand(2, b, device=x.device) * 0.16 - 0.08)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = 1; theta[:, 1, 1] = 1
    theta[:, 0, 2] = dx; theta[:, 1, 2] = dy
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    x = x * (0.9 + 0.2 * torch.rand(b, 1, 1, 1, device=x.device))
    return x + 0.01 * torch.randn_like(x)

def batches(idx, size, shuffle):
    idx = np.array(idx)
    if shuffle:
        idx = idx[np.random.permutation(len(idx))]
    for i in range(0, len(idx), size):
        yield idx[i:i + size]

def to_gpu(X, rows):
    return torch.from_numpy(X[rows]).to(dev).float().div_(255).sub_(0.5).div_(0.25)

@torch.no_grad()
def predict(X, rows, bs=64):
    model.eval()
    out = []
    for b in batches(rows, bs, False):
        with torch.autocast("cuda", enabled=(dev == "cuda")):
            out.append(torch.sigmoid(model(to_gpu(X, b))).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)), np.float32)

tr_rows = np.where(~is_val)[0]
va_rows = np.where(is_val)[0]
Yt = torch.from_numpy(Ytr).to(dev)

head = [p for n, p in model.named_parameters() if n.startswith(("fc", "conv1"))]
body = [p for n, p in model.named_parameters() if not n.startswith(("fc", "conv1"))]
opt = torch.optim.AdamW([{"params": head, "lr": 6e-4}, {"params": body, "lr": 1.5e-4}],
                        weight_decay=1e-4)
steps = max(1, EPOCHS * int(np.ceil(len(tr_rows) / BATCH)))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[6e-4, 1.5e-4], total_steps=steps,
                                            pct_start=0.25)
scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

def safe_auc(y, p):
    """AUC, or nan when the column has no positives or no negatives to compare."""
    y = np.asarray(y)
    if len(y) == 0 or y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, p))

def mean_auc(y, p):
    vals = [safe_auc((y[:, j] > 0.5).astype(int), p[:, j]) for j in range(len(LABELS))]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan

history, snapshots = [], []
done = 0
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for b in batches(tr_rows, BATCH, True):
        if done >= steps:
            break
        x = augment(to_gpu(Xtr, b))
        with torch.autocast("cuda", enabled=(dev == "cuda")):
            loss = F.binary_cross_entropy_with_logits(model(x), Yt[b])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        total += loss.item() * len(b); done += 1
    pv = predict(Xtr, va_rows)
    va = mean_auc(Ytr[va_rows], pv)
    history.append({"epoch": epoch + 1, "loss": total / len(tr_rows), "holdout_auc": va})
    log(f"epoch {epoch + 1}/{EPOCHS}  loss {total / len(tr_rows):.4f}  holdout AUC {va:.4f}")
    snapshots.append((va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
    if time.time() > HARD_DEADLINE - 2400:
        log("stopping training early to leave time for the test set")
        break

assert snapshots, "training did not finish one epoch, so there is nothing to report"
hist = pd.DataFrame(history)

# Section 1 said the score reads only the order, so several models should be combined by
# averaging their ranks rather than their probabilities. The three epochs that scored best on
# the held out fifth are combined that way here. Averaging probabilities instead would let
# whichever epoch happens to be most confident decide the answer.
KEEP = min(3, len(snapshots))
top = sorted(snapshots, key=lambda t: -t[0])[:KEEP]

def predict_top(X, rows):
    """Rank average the predictions of the KEEP best epochs."""
    acc = None
    for _, sd in top:
        model.load_state_dict(sd)
        p = pd.DataFrame(predict(X, rows), columns=LABELS)
        p = p.rank(pct=True) if len(p) > 1 else p
        acc = p if acc is None else acc + p
    return (acc / len(top)).values

model.load_state_dict(top[0][1])   # the CAM section reads one model, so leave the best loaded
kept_epochs = [h["epoch"] for h in history if h["holdout_auc"] in [t[0] for t in top]]
show(hist.assign(**{"combined by rank average": hist.epoch.isin(kept_epochs)})
     .rename(columns={"loss": "training loss", "holdout_auc": "holdout AUC"}),
     "Every epoch, and which three were kept",
     bars=["holdout AUC"], vmin=0.5, vmax=0.85, hide_index=True)
facts([("best single epoch, holdout AUC", f"{top[0][0]:.4f}"),
       ("epochs combined by rank average", ", ".join(str(e) for e in kept_epochs))],
      "What is used for every prediction below")

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 2.8))
a1.plot(hist.epoch, hist.loss, color=BLUE, lw=2, marker="o", ms=4)
a1.set_xlabel("epoch"); a1.set_ylabel("training loss")
a1.set_title("Training loss")
a2.plot(hist.epoch, hist.holdout_auc, color=ORANGE, lw=2, marker="o", ms=4)
a2.axhline(0.5, color=MUTED, ls=":", lw=1)
a2.text(hist.epoch.iloc[0], 0.508, "guessing", color=MUTED, fontsize=7.5)
a2.set_xlabel("epoch"); a2.set_ylabel("average AUC on the held out fifth")
a2.set_title("Held out agreement with the report labels")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 9. What the model learned
#
# There are two different questions here and they need two different measurements.
#
# The first is whether the model agrees with the labels it was trained on. That is measured on
# the held out fifth, it covers hundreds of studies, and it is the number to use when comparing
# two versions of the model.
#
# The second is whether the model agrees with a radiologist looking at the images, which is what
# the competition scores. That can only be measured on the 58 labelled studies, and none of them
# were trained on, so it is a fair measurement of the wrong size. Section 3 showed a single
# finding's AUC on 58 studies moves by 0.2 to 0.3 from resampling alone, so this number says
# whether the model works at all and not much more.

# %%
val_ids = np.array(kept_ids)[va_rows]
pred_val = pd.DataFrame(predict_top(Xtr, va_rows), index=val_ids, columns=LABELS)
single = pd.DataFrame(predict(Xtr, va_rows), index=val_ids, columns=LABELS)

gold_rows = [i for i, s in enumerate(kept_ids) if s in gold.index]
pred_gold = pd.DataFrame(predict_top(Xtr, np.array(gold_rows)),
                         index=[kept_ids[i] for i in gold_rows], columns=LABELS)
g58 = gold.reindex(pred_gold.index)

holdout_auc = pd.Series({c: safe_auc((Ytr[va_rows][:, j] > 0.5).astype(int), pred_val[c])
                         for j, c in enumerate(LABELS)})
gold_auc = pd.Series({c: safe_auc(g58[c], pred_gold[c]) for c in LABELS})
label_auc = score_against_gold(target)

table = pd.DataFrame({
    "positives in holdout": (Ytr[va_rows] > 0.5).sum(axis=0).astype(int),
    "model vs report labels": holdout_auc,
    "model vs radiologist (58)": gold_auc,
    "report labels vs radiologist (58)": label_auc,
})
show(table.sort_values("model vs radiologist (58)", ascending=False),
     "Per finding, three AUC values that mean three different things",
     grad=["model vs report labels", "model vs radiologist (58)",
           "report labels vs radiologist (58)"], vmin=0.5, vmax=1.0)
facts([("best single epoch, against the report labels on the holdout",
        round(mean_auc(Ytr[va_rows], single.values), 3)),
       ("three epochs combined, same holdout", round(holdout_auc.mean(), 3)),
       ("three epochs combined, against the radiologist on the 58", round(gold_auc.mean(), 3)),
       ("the report labels themselves, against the radiologist on the 58",
        round(label_auc.mean(), 3))],
      "Averages over the twelve findings")

# %%
o = gold_auc.sort_values(ascending=False).index
x = np.arange(len(LABELS))
fig, ax = plt.subplots(figsize=(9.5, 3.2))
ax.bar(x - 0.22, holdout_auc[o], 0.44, color=BLUE, label="model against the report labels, holdout")
ax.bar(x + 0.22, gold_auc[o], 0.44, color=ORANGE, label="model against the radiologist, 58 studies")
ax.plot(x, label_auc[o], "o", color=INK, ms=5, label="report labels against the radiologist")
ax.axhline(0.5, color=MUTED, ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels(o, rotation=45, ha="right")
ax.set_ylim(0.3, 1.0); ax.set_ylabel("ROC AUC")
ax.set_title("Where the model works, and what its labels allowed")
ax.legend(fontsize=7.5, loc="lower left")
plt.tight_layout(); plt.show()

# %% [markdown]
# Read the two bars together with the black dot. The blue bar says how well the model reproduces
# the labels it was given. The black dot says how good those labels were in the first place. The
# orange bar is the product of the two, and it cannot beat the dot by much for long.
#
# The submitted version of this notebook scored 0.798 on the public leaderboard against the
# 0.778 in the orange bars. The two agreeing is not a given, because the 58 studies and the test
# set are different samples of different sizes, and section 3 says how much room there is for
# them to disagree.
#
# MCL and Fracture are the two orange bars to be careful with. Both have the fewest positives in
# the holdout, 130 and 77, and both are the findings whose orange bar swung most between runs of
# this notebook. Their black dots are high, at 0.968 and 0.870, so the labels are not the limit
# there. The model is.
#
# ### Does the model tell the findings apart, or does it learn one thing twelve times
#
# A study with one finding often has several, as section 3 showed. So a model can score above 0.5
# on all twelve columns by learning a single idea of how damaged a knee looks. Compare how much
# the predictions move together against how much the labels do.

# %%
pc = pred_val.corr(method="spearman")
lc = pd.DataFrame(Ytr[va_rows], columns=LABELS).corr(method="spearman")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for ax, m, t in [(a1, lc, "The training labels"), (a2, pc, "The model's predictions")]:
    im = ax.imshow(m.loc[LABELS, LABELS].values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=90, fontsize=7)
    ax.set_yticks(x); ax.set_yticklabels(LABELS, fontsize=7)
    ax.set_title(f"{t}\naverage off diagonal correlation "
                 f"{(m.values.sum() - 12) / (144 - 12):.2f}")
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.85)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 10. What the model looks at
#
# A number saying the model works is not the same as knowing what it responded to. The check
# below is a class activation map. It takes the last block of convolutions, asks how much each of
# its 512 feature maps pushed one output up, and adds the maps together with those amounts as
# weights. The result is a rough picture of which parts of the slice raised that finding's score.
#
# Read it as a hint and not as evidence. The map has the resolution of the last block, which is
# 7 by 7 for a 224 pixel input, so it can point at a region and no smaller.

# %%
def class_activation_map(row: int, finding: str):
    """Where in the stack the last block of the network pushed one finding up."""
    model.eval()
    feats, grads = {}, {}
    h1 = model.layer4.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    h2 = model.layer4.register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("g", go[0]))
    x = to_gpu(Xtr, np.array([row]))
    out = model(x)
    model.zero_grad()
    out[0, LABELS.index(finding)].backward()
    h1.remove(); h2.remove()
    a, g = feats["a"][0], grads["g"][0]
    cam = F.relu((a * g.mean(dim=(1, 2), keepdim=True)).sum(0))
    cam = cam / (cam.max() + 1e-6)
    cam = cv2.resize(cam.detach().cpu().numpy(), (SIZE, SIZE))
    return cam, float(torch.sigmoid(out[0, LABELS.index(finding)]))

best_finding = gold_auc.idxmax()
scores = pred_gold[best_finding].sort_values()
picks = [(scores.index[-1], "highest score"), (scores.index[-2], "second highest"),
         (scores.index[1], "second lowest"), (scores.index[0], "lowest score")]

fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
for col, (sid, tag) in enumerate(picks):
    row = kept_ids.index(sid)
    cam, p = class_activation_map(row, best_finding)
    mid = Xtr[row, N_SLICES // 2]
    axes[0, col].imshow(mid, cmap="gray")
    axes[0, col].set_title(f"{tag}\nmodel {p:.2f}, radiologist "
                           f"{int(gold.loc[sid, best_finding])}", fontsize=8)
    axes[1, col].imshow(mid, cmap="gray")
    axes[1, col].imshow(cam, cmap="inferno", alpha=0.45)
    axes[1, col].set_title("where the score came from", fontsize=8)
    for r in (0, 1):
        axes[r, col].axis("off")
fig.suptitle(f"Middle slice and activation map for {best_finding}, "
             f"the finding the model does best on", y=1.0, fontsize=9.5)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### How the scores are spread out
#
# The last check on the predictions is whether they separate at all. A model that returns almost
# the same number for every study has an AUC near 0.5 no matter how the number was computed.

# %%
fig, axes = plt.subplots(3, 4, figsize=(11, 5.4), sharex=True)
for ax, c in zip(axes.ravel(), LABELS):
    y = (Ytr[va_rows][:, LABELS.index(c)] > 0.5)
    ax.hist(pred_val[c][~y], bins=30, color=BLUE, alpha=0.75, label="label says absent")
    ax.hist(pred_val[c][y], bins=30, color=ORANGE, alpha=0.75, label="label says present")
    ax.set_title(f"{c}   AUC {holdout_auc[c]:.2f}", fontsize=8)
    ax.set_yticks([])
axes[0, 0].legend(fontsize=6.5)
fig.suptitle("Predicted score on the held out studies, split by the training label", y=1.0)
plt.tight_layout(); plt.show()

show(pred_val.describe().loc[["min", "50%", "max", "std"]].T
     .rename(columns={"50%": "middle", "std": "spread"}),
     "How far apart the predictions are on the held out studies",
     bars=["spread"], vmin=0, vmax=0.35)

# %% [markdown]
# ## 11. Writing the submission
#
# The test images are read the same way as the training images, the model predicts, and each
# column is replaced by its ranks. In the public run there are only a handful of test studies. In
# the scored run this cell reads a test set nobody has seen.
#
# Any study the reader could not open, or did not reach before the clock ran out, gets 0.5 in
# every column. That is the value of a study the model knows nothing about, and it keeps the file
# valid, which is what the run is graded on.

# %%
chosen_test = pick_series(test_series, COMP / "test_series")
test_ids = [s for s in test.StudyInstanceUID if s in chosen_test.index]
log(f"chose a series for {len(test_ids)} of {len(test)} test studies")

Xte, kept_test = build_set(test_ids, chosen_test, COMP / "test_series",
                           max(600, HARD_DEADLINE - time.time() - 900), "test")

pred_test = pd.DataFrame(predict_top(Xte, np.arange(len(kept_test))),
                         index=kept_test, columns=LABELS)

show(pred_test.describe().loc[["min", "50%", "max"]].T.rename(columns={"50%": "middle"}),
     f"The model's output on the {len(pred_test)} test studies it could read")

sub = write_submission(pred_test)
log(f"wrote submission.csv with {len(sub)} rows")
shown = sub.head(8).copy()
shown.insert(0, "study", shown.StudyInstanceUID.map(short_uid))
show(shown.drop(columns="StudyInstanceUID"),
     "The first rows of submission.csv, after the ranks are taken",
     grad=LABELS, vmin=0, vmax=1, hide_index=True)

# %% [markdown]
# ## 12. What to try next
#
# The measurements above point at four changes, in the order they are worth making.
#
# Better labels come first. Section 5 measured a gap of about 0.17 AUC between a word list and a
# language model reading the same reports, and section 9 showed the model cannot beat its labels
# by much. Reading the reports more carefully is worth more than a bigger network.
#
# Then the other two planes. This model reads one sagittal series. Osteoarthritis in the inner
# and outer compartments is read from the coronal images, and the cartilage behind the kneecap is
# read from the axial ones, so three of the twelve findings are being asked for from a view that
# barely shows them.
#
# Then physical scale. Section 6 showed millimetres per pixel varies between studies. Cropping a
# fixed number of millimetres around the joint and resizing that, instead of resizing the whole
# image, gives every study the same scale and puts a floor on how small a feature can be and
# still survive the resize. A meniscal tear is one to three millimetres across.
#
# Last, train once and predict many times. Nothing requires the scored run to be the run that
# learned the weights. Training in a separate notebook, saving the weights as a dataset, and
# attaching them here would free the whole session for reading and predicting, which is the part
# that genuinely cannot be done in advance.
