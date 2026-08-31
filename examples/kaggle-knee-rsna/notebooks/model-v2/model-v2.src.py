# %% [markdown]
# # 🦵 RSNA Knee: 0.926 LB, CoaTNet + fine-tune blend
#
# **0.926 on the public leaderboard** (top 25% of ~2,700 teams, scored by the two-arm
# version of this notebook), one T4, internet off, about 6.5 hours. This notebook ports the
# public **0.924** CoaTNet checkpoint, verifies it against the 58 radiologist-read studies,
# fine-tunes a second arm from it, adds the checkpoint's SWA twin as a third arm, and blends
# all three under two-pass test-time augmentation.
#
# The checkpoint that made this worth doing: `dreaddevelopment/raptor-knee-widedense` (CC0) —
# a CoAtNet trained on soft language-model labels of the reports, scoring **0.924 as a single
# model** on the public leaderboard, weights and inference notebook fully public. Our first
# notebook trained a small ResNet18 and scored 0.798, so three things happen here with it.
#
# 1. **🔍 Reproduce it.** Port their preprocessing and head exactly, load their checkpoint,
#    score it on the 58 studies that carry radiologist labels, and compare against the two
#    numbers they published for it (0.9167 stored in the checkpoint, 0.9054 in their notebook's
#    comments). We land between them.
# 2. **🎯 Fine-tune it.** Continue training from their checkpoint on a different label set,
#    `yunusgmsoy report_labels_v5`, which is a four-source merge that contains the 58 real
#    annotations. A second arm trained on different labels is the one kind of diversity their
#    own blending did not have; their arms shared labels and architecture, and blending bought
#    them about +0.001 on the live board. Expect a small gain here too, not a second 0.92.
# 3. **🧪 Blend and submit.** Weighted rank-mean of the arms, the weights picked on a
#    held-out fifth of the studies and never on the 58.
# 4. **🧬 Go further.** Eight fine-tune epochs (the 0.926 version stopped at three while the
#    gate was still climbing), the dataset's SWA checkpoint as a third arm, and a second
#    center-shifted pass over the test set — each change measurable against what scored 0.926.
#
# Two honesty rules, both inherited from the first notebook. The v5 label set contains the answers
# for the 58 labelled studies, so whether those 58 may be used to *measure* depends entirely on
# whether they were in training: `INCLUDE_GOLD=False` keeps them out and the 58 stay a
# measurement, `INCLUDE_GOLD=True` puts them in for the final submission run and the 58 stop
# being a measurement. And a mean AUC over 58 studies moves about 0.03 between runs of the same
# recipe, so these numbers validate that a model works; they do not rank models that are close.
#
# Everything runs in this notebook on one T4 with the internet off: reading all 4,407 training
# studies, arm-1 inference, fine-tuning, blending, and writing the submission. About 6.5 hours.

# %%
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gc, glob, hashlib, json, math, pathlib, shutil, time, unicodedata, warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

T0 = time.time()

def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

# One session gets a fixed amount of wall clock, and the scored run reruns the whole notebook
# against a test set nobody has seen. Every stage below checks these numbers, so a slow stage
# shrinks the next one instead of killing the run. The fallback ladder: the notebook writes a
# placeholder submission first, then arm-1 only (about 0.92), then the blend.
HARD_DEADLINE = T0 + 8.5 * 3600     # 30 min of the 9 h session left for Kaggle teardown
PREP_BUDGET_S = 2.5 * 3600          # pass-1 read budget, same number as the first notebook
FT_END_BY = HARD_DEADLINE - 3.0 * 3600
EVAL_END_BY = HARD_DEADLINE - 1.2 * 3600
TEST_START_BY = HARD_DEADLINE - 1.0 * 3600
FINAL_WRITE_BY = HARD_DEADLINE - 0.25 * 3600

# The flag that decides what this run is. False: the 58 radiologist-labelled studies stay out of
# fine-tuning, so every number against them is a measurement. True: they go in (their v5 labels
# are the exact annotations), the run is the submission run, and the 58-study number printed is
# contaminated by construction. Two pushes: first False to measure, then True to submit.
INCLUDE_GOLD = True

HOLDOUT_FRAC = 0.20                 # report-hash holdout, excluded from fine-tune in BOTH modes
W_WINDOWS = 12                      # windows sampled per study per epoch
EPOCHS_FT = 8                       # v2 stopped at 3 while the holdout gate was still climbing
ACCUM = 4                           # studies per optimizer step
LR_FT = 1e-5
WD_FT = 1e-4
WARMUP_FT = 100
SEED = 0
READ_THREADS = 12
K_EVAL = 42                         # windows per study at inference, raptor's scored setting
K_GATE = 12                         # cheaper windows for the per-epoch gate
TTA_TEST = 2                        # test-set passes per arm, center-shifted; gold/holdout stay 1

TIMINGS = {}

# The competition mounts under /kaggle/input/competitions/<slug> on this image, and under
# /kaggle/input/<slug> on older ones.
COMP = next(p for p in map(pathlib.Path, [
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    "/kaggle/input/rsna-knee-abnormality-detection",
    "/tmp/rsnaknee"]) if (p / "train.csv").exists())

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

train = pd.read_csv(COMP / "train.csv")
train_series = pd.read_csv(COMP / "train_series.csv")
test = pd.read_csv(COMP / "test.csv")
test_series = pd.read_csv(COMP / "test_series.csv")
# train.csv / test.csv carry only StudyInstanceUID + Report (+labels); the series CSVs add
# SeriesInstanceUID, so cast per-column rather than assuming both everywhere.
for d in (train, train_series, test, test_series):
    for c in ("StudyInstanceUID", "SeriesInstanceUID"):
        if c in d.columns:
            d[c] = d[c].astype(str)

reports = train.set_index("StudyInstanceUID").Report.fillna("")
gold = train[train[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS].astype(int)

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Scratch space for the memmap stack cache and the epoch checkpoints. On this image /kaggle/tmp
# does not exist until something creates it, and shutil.disk_usage on a missing path raises.
os.makedirs("/kaggle/tmp", exist_ok=True)

log(f"data root        {COMP}")
log(f"train studies    {len(train):,}    train series {len(train_series):,}")
log(f"test studies     {len(test):,}     test series  {len(test_series):,}")
log(f"labelled studies {len(gold)}  ({len(gold) / len(train):.2%} of training)")
log(f"INCLUDE_GOLD     {INCLUDE_GOLD}   ({'the 58 are training rows, not a measurement' if INCLUDE_GOLD else 'the 58 are held out of fine-tuning and stay a measurement'})")
log(f"device {dev} | gpus {torch.cuda.device_count()} | "
    f"torch {torch.__version__} | timm {timm.__version__} | pydicom {pydicom.__version__} | "
    f"cv2 {cv2.__version__}")
if dev == "cuda":
    log(f"gpu {torch.cuda.get_device_name(0)}")
log(f"disk free /kaggle/tmp {shutil.disk_usage('/kaggle/tmp').free / 1e9:.1f} GB | "
    f"RAM {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.1f} GB")

# %% [markdown]
# ## 📝 Write a valid submission before doing anything else
#
# Same rule as the first notebook, for the same reason. When this is submitted, Kaggle runs it
# again against a hidden test set, and a run that fails part way through scores nothing. The
# placeholder of all 0.5 values is written now; two later rungs overwrite it, and neither rung
# ever touches submission.csv unless it has a complete result in hand.

# %%
def write_submission(pred: pd.DataFrame, path="submission.csv") -> pd.DataFrame:
    """Rank inside each column, then write one row per test study.

    The score reads only the order of the values in a column, so replacing the numbers by their
    ranks throws nothing away. Feeding already-ranked values back through is harmless: the second
    ranking preserves the order of the first.
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

# %% [markdown]
# ## 🧰 Shared helpers
#
# Carried over from the first notebook: the HTML table display, the bounded file finder, the
# scorers, and the bootstrap. The bounded finder matters more than it looks: a recursive glob
# under /kaggle/input walks the competition mount and its 700,000 DICOM files, which costs
# minutes for every lookup.

# %%
from html import escape as _esc
from IPython.display import display, HTML, Markdown

TABLE_CSS = [
    {"selector": "caption", "props": [("caption-side", "top"), ("text-align", "left"),
                                      ("font-weight", "600"), ("font-size", "0.95rem"),
                                      ("padding", "0 0 0.45rem 0"), ("color", "#0b0b0b")]},
    {"selector": "th", "props": [("background-color", "#f2f1ec"), ("color", "#0b0b0b"),
                                 ("font-weight", "600"), ("text-align", "right"),
                                 ("padding", "5px 11px"), ("border-bottom", "1px solid #c3c2b7")]},
    {"selector": "th.row_heading", "props": [("text-align", "left"),
                                             ("background-color", "#fcfcfb")]},
    {"selector": "td", "props": [("padding", "5px 11px"), ("text-align", "right"),
                                 ("border-bottom", "1px solid #e1e0d9")]},
    {"selector": "", "props": [("border-collapse", "collapse"), ("font-size", "0.86rem"),
                               ("font-variant-numeric", "tabular-nums"), ("margin", "0.3rem 0")]},
]

def _cell(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{v:,.3f}" if abs(v) < 1e4 else f"{v:,.0f}"
    return str(v)

def show(df, caption="", bars=None, vmin=None, vmax=None, hide_index=False):
    """Display a table as HTML, optionally with in-cell bars (the baseline's shading option is
    dropped: this notebook shades nothing)."""
    st = (pd.DataFrame(df).style
          .set_caption(caption).set_table_styles(TABLE_CSS)
          .format(_cell, na_rep=""))
    if bars is not None:
        st = st.bar(subset=bars, color="#cde2fb", vmin=vmin, vmax=vmax)
    if hide_index:
        st = st.hide(axis="index")
    display(st)

def facts(pairs, caption=""):
    show(pd.DataFrame({"value": [v for _, v in pairs]}, index=[k for k, _ in pairs]), caption)

def note(md):
    display(Markdown(md))

def short_uid(u, n=10):
    u = str(u)
    return u if len(u) <= n else u[:n] + ".."

def fold(s: str) -> str:
    """Lower case, strip accents, map the Turkish dotless i onto a plain i."""
    s = str(s).replace("\u0131", "i").replace("\u0130", "i")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()

SEARCH = [p for p in map(pathlib.Path, ["/kaggle/input/datasets", "/kaggle/input", "/tmp/rsnallm"])
          if p.exists()]

def find_file(name: str):
    """Find an attached dataset file without walking the competition folder."""
    for root in SEARCH:
        for pat in (name, f"*/{name}", f"*/*/{name}", f"*/*/*/{name}"):
            for hit in root.glob(pat):
                if "competitions" not in hit.parts:
                    return hit
    return None

def score_against_gold(pred: pd.DataFrame) -> pd.Series:
    p = pred.reindex(gold.index)
    return pd.Series({c: roc_auc_score(gold[c], p[c].fillna(0.0)) for c in LABELS})

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

def bootstrap_auc(y: pd.DataFrame, p: pd.DataFrame, n_boot=2000, seed=0):
    """95% interval width for each finding's AUC and for the average, from resampling the rows."""
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

# %% [markdown]
# ## 🏷️ The fine-tuning labels
#
# One published set is attached: `yunusgmsoy/rsna-knee-llm-labels-4-source-merged`, file
# `report_labels_v5.csv`. It merges four language-model label sets and, unlike every honest set
# the first notebook audited, it carries the 58 radiologist annotations written in exactly. That
# is the reason this notebook picked it: real labels are the best labels there are. It is also
# the reason the `INCLUDE_GOLD` flag exists, because the same rows cannot be both training data
# and the measuring stick.

# %%
PUBLISHED = {"four sets merged": "report_labels_v5.csv"}
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
assert published, "the v5 label file was not found among the attached datasets"

v5_name = next(iter(published))
v5 = published[v5_name].reindex(train.StudyInstanceUID)
LABEL_FILE = hit.name

check = contains_the_answers(v5.dropna(subset=LABELS, how="all"))
show(pd.DataFrame([check]), "Does v5 contain the 58 answers, and is it soft elsewhere")
facts([("studies covered", int(v5[LABELS].notna().any(axis=1).sum())),
       ("mean target across findings", round(float(v5.stack().mean()), 3)),
       ("share of cells exactly 0 or 1 off the 58", round(check["exactly 0 or 1 elsewhere"], 4))],
      f"The fine-tuning target: {v5_name}")
note(f"Policy for this run: `INCLUDE_GOLD={INCLUDE_GOLD}`. "
     + ("The 58 annotated studies are inside fine-tuning, so every 58-study number printed below "
        "is contaminated by construction and is printed only to show the model saw them."
        if INCLUDE_GOLD else
        "The 58 annotated studies are held out of fine-tuning, so the 58-study numbers below are "
        "honest measurements."))

# %% [markdown]
# ## ✂️ The split
#
# Two dispositions of the 4,407 studies, decided before anything is trained. A fifth of the
# studies go to a holdout by a hash of the report text, which keeps studies sharing a report on
# the same side; the holdout is excluded from fine-tuning in **both** flag modes, so epoch
# selection and the blend weight are always measured out-of-sample. The 58 gold studies are the
# second disposition: out of fine-tuning entirely when `INCLUDE_GOLD=False`, and so is any study
# whose report matches a gold study's report, because the v5 labels of such a twin are the
# gold labels under another name. When `INCLUDE_GOLD=True` the gold rows and their twins go into
# fine-tuning; only the report-hash holdout stays out.

# %%
def bucket(sid: str) -> int:
    h = hashlib.md5(fold(reports.get(sid, sid)).encode()).hexdigest()
    return int(h[:8], 16) % 100

all_train_ids = list(train.StudyInstanceUID)
gold_ids = list(gold.index)
holdout_ids = [s for s in all_train_ids if bucket(s) < HOLDOUT_FRAC * 100]
ft_base = [s for s in all_train_ids if s not in set(holdout_ids)]

gold_text = set(reports.reindex(gold_ids).map(fold))
if INCLUDE_GOLD:
    ft_ids = ft_base
else:
    ft_ids = [s for s in ft_base
              if s not in set(gold_ids) and fold(reports.get(s, "")) not in gold_text]

if INCLUDE_GOLD:
    # The report-hash holdout keeps its 6 gold studies in both modes, so gold-in means
    # every non-holdout gold study, not all 58.
    assert (set(gold_ids) - set(holdout_ids)) <= set(ft_ids), \
        "gold-in mode but holdout-free gold is missing from fine-tuning"
else:
    assert (set(ft_ids) & set(gold_ids)) == set(), "gold rows leaked into an honest fine-tune set"
n_twins = len(ft_base) - len(ft_ids) if not INCLUDE_GOLD else \
    sum(1 for s in ft_base if fold(reports.get(s, "")) in gold_text)
log(f"holdout (both modes)  {len(holdout_ids):,}   gold inside it "
    f"{len(set(holdout_ids) & set(gold_ids))}")
log(f"fine-tune set         {len(ft_ids):,}   ({'gold + report twins included' if INCLUDE_GOLD else 'gold and report twins excluded'})")
log(f"report twins handled  {n_twins}")

# %% [markdown]
# ## 🦾 The raptor arm, ported
#
# Everything in the next two cells is ported from the public inference notebook of
# `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`, as exactly as the code can be
# carried, because the weights were trained against this exact pipeline and any drift shows up
# as a silent score drop. What the pipeline does:
#
# - **Five fixed slots** per study, always in the same order: 18 slices from a sagittal
#   fluid-sensitive series, 14 from a second sagittal series, 12 coronal fluid-sensitive,
#   8 coronal, 12 axial. A study with no series for a slot leaves it as zeros.
# - **Slices spread across 6 to 94 percent** of each series, not the middle: the collateral
#   ligaments and the lateral meniscus live in the peripheral slices.
# - **Series ordered by geometry**, from `ImagePositionPatient` crossed with the in-plane axes,
#   which is the only order that means anything (the first notebook measured filename order at a
#   rank correlation of 0.043 with physical position).
# - **A 140 mm crop** using the pixel spacing, then resize to 336 px, so a knee fills the same
#   fraction of the frame whatever the scanner's resolution.
# - **Three neighbouring slices per window**, 42 windows per study at 384 px, ImageNet norm.
#
# The head pools those windows into twelve scores with **separate attention weights per finding**,
# so a cruciate tear can pick the two sagittal slices that show it while osteoarthritis reads the
# coronal stack.
#
# Documented deltas from their code, all deliberate: the weight-file path points at this
# notebook's dataset mount; `apply_modality_lut` is imported from whichever pydicom location
# provides it (pydicom 3 moved it); a `Fluid_Sensitive` value of NaN is treated as "no
# preference" instead of raising inside `int()` and silently dropping the study to 0.5; and
# `build_study` also returns a per-slice recipe so this notebook can rebuild a study's stack from
# disk without re-decoding headers.

# %%
IMG = 336
CROP_MM = 140.0
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(s[2] for s in SLOTS)                     # 64
NORM = "imagenet"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def build_backbone(arch, pretrained=False):
    # maxvit/maxxvit/coatnet are conv-attention hybrids: NO CLS token, NO interpolatable
    # pos-embed -> avg pool. The "vit" substring in "coatnet"/"maxvit" must NOT route them
    # down the ViT path (mirrors their finetune_raptor.py exactly).
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)

class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))

def load_model(pt_path, device):
    """One model resident at a time is the discipline, not an optimisation.

    Their scored run OOM'd system RAM with DataParallel replicating modules across many studies;
    sequential single-model residency is what graded. Two 293 MB state dicts would not OOM on
    their own, but the discipline costs nothing and matches the scored configuration.
    """
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    bb = build_backbone(ck["arch"], pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    return model, int(ck.get("res", 384)), ck

def _eval_centers(mask, D, k, shift=0):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    if shift:
        idx = np.clip(idx + shift, 0, len(cs) - 1)
    return [cs[i] for i in idx]

def eval_windows(vol, mask, k, res, norm=NORM, shift=0):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k, shift=shift)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x

@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    if device.startswith("cuda"):
        # fp16 conv on T4 is fully cuDNN-supported (bf16 is NOT -> "no engine").
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
        except RuntimeError:
            torch.cuda.empty_cache()
            o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
    o = torch.sigmoid(model(x).float())
    return o[0].cpu().numpy()

def infer_study(model, sid, k, shift=0):
    vol, mask = get_stack(sid)
    return infer_probs(model, eval_windows(vol, mask, k=k, res=RES, shift=shift), dev)

def infer_guarded(model, ids, k, deadline, tag):
    """Evaluate studies one at a time, stopping if the projection crosses the deadline.

    The hidden-rerun reality is that the same notebook runs against an unknown test size on
    the same wall clock, so every eval larger than a handful of studies needs a stop rule, not
    a fixed size. Truncation is safe because callers shuffle `ids` first: what survives is a
    random sample of the set, so its AUC stays unbiased. Returns (ids_used, probs).
    """
    probs, used, t0 = [], [], time.time()
    for i, s in enumerate(ids):
        if i and (time.time() - t0) / i * (len(ids) - i) > max(deadline - time.time(), 0):
            log(f"{tag}: stopped at {i}/{len(ids)} — projection crossed the deadline")
            break
        probs.append(infer_study(model, s, k))
        used.append(s)
        if (i + 1) % 100 == 0:
            log(f"{tag}: {i + 1}/{len(ids)}  {(time.time() - t0) / (i + 1):.1f} s/study")
    return used, (np.stack(probs) if probs else np.zeros((0, 12), np.float32))

def rankpct(x):                                   # per-column percentile rank in [0,1]
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))

def blend_ranks(p2, p1, w):
    """Weighted rank-mean, raptor's recipe: ranks, weighted, never probabilities."""
    return w * rankpct(np.clip(p2, 0, 1)) + (1 - w) * rankpct(np.clip(p1, 0, 1))

def window_centers(mask):
    """Every valid window centre, for training-time sampling."""
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, MAXS))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    return cs or [max(1, min((lo + hi) // 2, MAXS - 2))]

# %% [markdown]
# ### 🖼️ The preprocessing, also ported

# %%
def _fs(row):
    """NaN Fluid_Sensitive means no preference, not a crash.

    Their `_pick_series_for_slot` calls `int(r.get('Fluid_Sensitive', 0) or 0)`, and a NaN is
    truthy, so `int(nan)` raises, the study's per-study try/except catches it, and the study
    silently scores 0.5. Train-series rows carry NaNs, so the fix is part of the port, not an
    embellishment.
    """
    v = row.get("Fluid_Sensitive")
    return 0 if pd.isna(v) else int(v)

def _make_reader():
    import pydicom as _pd
    try:
        from pydicom.pixel_data_handlers.util import apply_modality_lut
        _lut_from = "pydicom.pixel_data_handlers.util"
    except ImportError:
        from pydicom.pixels.utils import apply_modality_lut
        _lut_from = "pydicom.pixels.utils"
    log(f"modality LUT helper from {_lut_from} (pydicom {_pd.__version__})")

    def order_and_meta(sdir):
        fs = list(pathlib.Path(sdir).glob("*.dcm")); recs = []; ps_list = []
        for f in fs:
            try:
                h = _pd.dcmread(str(f), stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                    n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps); recs.append((pos, str(f), ps))
            except Exception:
                recs.append((0.0, str(f), 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = _pd.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize

reader = _make_reader()
order_and_meta, read_px, mm_crop_resize = reader

def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane
             and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if _fs(r) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None

def build_study(sid, ser_records, tsdir, reader):
    """The 64-slot stack, its valid mask, and a recipe to rebuild it without re-decoding."""
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    rec_f = [None] * MAXS; rec_ps = [0.0] * MAXS
    rec_lo = [0.0] * MAXS; rec_hi = [1.0] * MAXS
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        n = len(files); lo, hi = int(n * 0.06), int(n * 0.94) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []; pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for j, (a, ps) in enumerate(zip(arrs, pss)):
            if idx >= MAXS: break
            if a is None:
                idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8)
            rec_f[idx] = files[min(picks[j], n - 1)][0]
            rec_ps[idx] = ps if ps > 0 else med_ps
            rec_lo[idx] = float(loq); rec_hi[idx] = float(hiq)
            idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask, {"files": rec_f, "pss": rec_ps, "loqs": rec_lo, "hiqs": rec_hi}

def find_test_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")

CKPT_FILE = "raptor_ft_coatnet_v4_full.pt"
CKPT_FILE_SWA = "raptor_ft_coatnet_v4_full_swa.pt"   # the dataset's second checkpoint: arm 3

def find_weight_file(fname):
    direct = [f"/kaggle/input/raptor-knee-widedense/{fname}",
              f"/kaggle/input/raptor-knee-widedense/1/{fname}"]
    for p in direct:
        if os.path.exists(p):
            return p
    for d in sorted(glob.glob("/kaggle/input/*/")):
        if "competition" in d.lower():
            continue
        hits = glob.glob(os.path.join(d, "**", fname), recursive=True)
        if hits:
            return hits[0]
    raise RuntimeError(f"{fname} not found under /kaggle/input")

TRAIN_DIR = COMP / "train_series"
SER_TRAIN = {k: v.to_dict("records") for k, v in train_series.groupby("StudyInstanceUID")}

def get_stack(sid):
    """(vol, mask) for any study: memmap row, recipe rebuild, or a fresh decode.

    The recipe path is bit-identical to a fresh decode because the per-slot window bounds are
    stored, not recomputed.
    """
    if sid in idx_of and USE_MM:
        return np.asarray(STACKS[idx_of[sid]]), np.frombuffer(recipe[sid]["mask"], np.uint8)
    r = recipe.get(sid)
    if r is not None:
        vol = np.zeros((MAXS, IMG, IMG), np.uint8)
        for i in range(MAXS):
            if r["files"][i] is None:
                continue
            a = read_px(r["files"][i])
            aw = np.clip((a - r["loqs"][i]) / (r["hiqs"][i] - r["loqs"][i] + 1e-6), 0, 1)
            vol[i] = (mm_crop_resize(aw, r["pss"][i]) * 255).astype(np.uint8)
        return vol, np.frombuffer(r["mask"], np.uint8)
    if sid in SER_TRAIN:
        vol, mask, _ = build_study(sid, SER_TRAIN, str(TRAIN_DIR), reader)
    else:
        vol, mask, _ = build_study(sid, SER_TEST, str(TEST_DIR), reader)
    return vol, mask

# %% [markdown]
# ## 📦 Load the checkpoint
#
# The file is a dictionary, not a bare state dict, and it carries its own provenance: the
# architecture name, the resolution it expects, the label order it was trained with, which epoch
# it came from, and the score its author measured on the 58. Every one of those is checked against
# this notebook's own settings, because a mismatch here degrades silently otherwise.

# %%
t_load = time.time()
weight_path = find_weight_file(CKPT_FILE)
arm1, RES, ck = load_model(weight_path, dev)
ck_meta = {k: v for k, v in ck.items() if k != "model"}
del ck; gc.collect()

facts([( "checkpoint", short_uid(weight_path.split("/")[-1], 40)),
       ("architecture", str(ck_meta.get("arch"))),
       ("resolution", int(ck_meta.get("res", RES))),
       ("trained epoch", ck_meta.get("epoch")),
       ("their gold AUC, stored in the file", round(float(ck_meta.get("gold_auc", np.nan)), 4)),
       ("their gold AUC, in their notebook comment", 0.9054),
       ("parameters", f"{sum(p.numel() for p in arm1.parameters()) / 1e6:.2f} M")],
      "What the checkpoint says about itself")
assert list(ck_meta["lab"]) == LABELS, f"label order drift: {ck_meta['lab']}"
assert RES == 384, f"unexpected model resolution {RES}"
log(f"checkpoint loaded in {time.time() - t_load:.0f}s")

x = torch.randn(1, 4, 3, RES, RES, device=dev)
with torch.no_grad():
    _ = arm1(x)      # warmup: cudnn.benchmark autotunes the first call, do not time it
    if dev == "cuda":
        torch.cuda.synchronize()
    t_s = time.time()
    for _ in range(4):
        _ = arm1(x)
    if dev == "cuda":
        torch.cuda.synchronize()
if dev == "cuda":
    torch.cuda.synchronize()
MS_WINDOW = (time.time() - t_s) / 4 * 1000
del x; gc.collect()
log(f"sanity forward: {MS_WINDOW:.0f} ms/window at {RES}px -> "
    f"~{MS_WINDOW * K_EVAL / 1000:.1f} s/study at {K_EVAL} windows")

# %% [markdown]
# ## 📚 Pass 1: read every training study once
#
# The arithmetic that decides how this is built: 4,407 studies of 64 slices at 336 by 336, one
# byte per pixel, is **31.8 GB**. That cannot live in the RAM of a Kaggle box, so the stacks go to
# a memory-mapped file on the fast local disk instead, and what stays in RAM is a small recipe per
# study: the 64 file paths, the pixel spacing and window bounds each slice was normalised with,
# and the valid window centres. With the recipe, any study can be rebuilt exactly, bounds and all,
# without re-reading a single header; if the disk does not have room, the notebook falls back to
# rebuilding from the recipe for every study and pays the decode time per epoch instead.
#
# The read order is the 58 gold studies first, then a seeded shuffle of the rest, so if the time
# budget truncates the pass, what was read is a random sample rather than the front of the
# alphabetical directory listing.

# %%
rng = np.random.default_rng(SEED)
rest = [s for s in all_train_ids if s not in set(gold_ids)]
decode_order = list(gold_ids) + list(rng.permutation(rest))

MM_PATH = "/kaggle/tmp/stacks.fbm"
DISK_NEED = int(len(all_train_ids) * MAXS * IMG * IMG) + int(2e9)
USE_MM = shutil.disk_usage("/kaggle/tmp").free >= DISK_NEED
idx_of, recipe = {}, {}
if USE_MM:
    STACKS = np.memmap(MM_PATH, dtype=np.uint8, mode="w+",
                       shape=(len(all_train_ids), MAXS, IMG, IMG))
    log(f"memmap {STACKS.shape} at {MM_PATH} ({STACKS.nbytes / 1e9:.1f} GB)")
else:
    STACKS = None
    log("not enough disk for the memmap; every study will be rebuilt from its recipe")

def decode_one(sid):
    vol, mask, rec = build_study(sid, SER_TRAIN, str(TRAIN_DIR), reader)
    return sid, vol, mask, rec

t_p1 = time.time(); n_done = 0; truncated = False
ex = ThreadPoolExecutor(READ_THREADS)
# Submit in bounded batches, never one future per study up front. A Future holds its result
# alive until the Future object is dropped, and one decoded stack is 7.2 MB, so 4,407 queued
# futures retain all 31.8 GB of decoded output in RAM at once and the kernel is killed at
# study ~3,300. ex.map releases each result as it is consumed, so the in-flight footprint
# stays at 2*READ_THREADS stacks.
while True:
    if time.time() - T0 > PREP_BUDGET_S or time.time() > HARD_DEADLINE - 1800:
        truncated = True
        log("pass-1 budget reached; the rest will be decoded on demand")
        break
    batch = decode_order[n_done:n_done + 2 * READ_THREADS]
    if not batch:
        break
    for sid, vol, mask, rec in ex.map(decode_one, batch):
        idx_of[sid] = n_done
        recipe[sid] = {"mask": mask.tobytes(), "centers": window_centers(mask), **rec}
        if USE_MM:
            STACKS[n_done] = vol
        n_done += 1
        if n_done % 250 == 0:
            rate = n_done / (time.time() - t_p1)
            log(f"pass 1: {n_done:,}/{len(decode_order):,}  {rate:.2f} studies/s  "
                f"eta {(len(decode_order) - n_done) / max(rate, 1e-9) / 60:.0f} min")
ex.shutdown(wait=False, cancel_futures=True)
if USE_MM:
    STACKS.flush()
TIMINGS["pass1_read_s"] = round(time.time() - t_p1, 1)
facts([("studies decoded and cached", n_done),
       ("truncated by budget", truncated),
       ("stacks on disk (memmap)", USE_MM),
       ("recipe memory", f"{sum(sum(len(f) for f in r['files'] if f) for r in recipe.values()) / 1e6:.0f} MB of paths")],
      "Pass 1")
log(f"pass 1 done: {n_done:,} studies in {TIMINGS['pass1_read_s']:.0f}s")

# %% [markdown]
# ## 🔍 Reproduction check: arm 1 against the radiologist
#
# The number this whole notebook is anchored to. Their checkpoint stores 0.9167 for these 58
# studies and their notebook's comment says 0.9054 for the same file; one of those two describes
# a different run. Whatever this port prints here is the number that applies to this pipeline.

# %%
TIMINGS["arm1_gold_start"] = round(time.time() - T0, 1)
t_g = time.time()
rows = []
for i, s in enumerate(gold_ids):
    rows.append(infer_study(arm1, s, K_EVAL))
    if (i + 1) % 20 == 0:
        log(f"arm-1 gold: {i + 1}/{len(gold_ids)}")
P1_gold = np.stack(rows)
pred1_gold = pd.DataFrame(P1_gold, index=gold_ids, columns=LABELS)
auc1_gold = score_against_gold(pred1_gold)
w_per1, w_mean1 = bootstrap_auc(gold, pred1_gold)
TIMINGS["arm1_gold_s"] = round(time.time() - t_g, 1)
TIMINGS["s_per_study_eval"] = TIMINGS["arm1_gold_s"] / max(len(gold_ids), 1)
log(f"eval speed: {TIMINGS['s_per_study_eval']:.1f} s/study at {K_EVAL} windows")

show(pd.DataFrame({"AUC on the 58": auc1_gold}).sort_values("AUC on the 58", ascending=False),
     f"Arm 1 (their weights, this pipeline) against the radiologist, "
     f"mean {auc1_gold.mean():.3f} (95% CI width {w_mean1:.3f})",
     bars=["AUC on the 58"], vmin=0.4, vmax=1.0)
repro_ok = auc1_gold.mean() >= 0.88
if repro_ok:
    log(f"REPRODUCTION OK: arm-1 gold mean {auc1_gold.mean():.4f} "
        f"(their file says {float(ck_meta.get('gold_auc', np.nan)):.4f}, "
        f"their comment says 0.9054)")
else:
    log(f"REPRODUCTION FAILED: arm-1 gold mean {auc1_gold.mean():.4f} is far from "
        f"their 0.9054-0.9167. Read the preprocessing before trusting anything below.")

# %% [markdown]
# ## 📈 Arm 1 per finding: where the checkpoint is strong, and where it is not

# %%
order1 = auc1_gold.sort_values()
fig, ax = plt.subplots(figsize=(8, 4.5))
bar_colors = ["#d62728" if v < 0.85 else "#1f77b4" for v in order1.values]
ax.barh(order1.index, order1.values, color=bar_colors)
ax.axvline(float(auc1_gold.mean()), color="k", ls="--", lw=1,
           label=f"mean {auc1_gold.mean():.4f}")
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("AUC on the 58 radiologist studies")
ax.set_title("Arm 1 (their checkpoint, this pipeline) per finding — red is below 0.85")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🚀 Rung 1: arm 1 on the test set
#
# The first real submission, written the moment arm 1 has predicted the test set. On the public
# run that is a handful of studies and takes seconds; on the scored run it is the hidden test, at
# roughly the per-study cost measured here. From this point on a scored run can only get better.

# %%
t_r1 = time.time()
ROOT = find_test_root()
TEST_DIR = ROOT + ("/test_series" if os.path.isdir(ROOT + "/test_series") else "/test_images")
SER_TEST = {k: v.to_dict("records") for k, v in test_series.groupby("StudyInstanceUID")}
log(f"test root {ROOT} | series dir {TEST_DIR} | "
    f"Fluid_Sensitive column present: {'Fluid_Sensitive' in test_series.columns}")

test_ids = list(test.StudyInstanceUID)
arm1_test = np.full((len(test_ids), len(LABELS)), 0.5, np.float32)
test_times = []
n_fallback = 0
for i, s in enumerate(test_ids):
    t_s = time.time()
    try:
        arm1_test[i] = infer_study(arm1, s, K_EVAL)
    except Exception as e:
        n_fallback += 1
        log(f"  study {i} {short_uid(s)} FALLBACK ({type(e).__name__}: {e})")
    test_times.append(time.time() - t_s)
    if (i + 1) % 100 == 0 or i + 1 == len(test_ids):
        log(f"arm-1 test: {i + 1}/{len(test_ids)}  {np.median(test_times):.2f} s/study")
sub1 = write_submission(pd.DataFrame(arm1_test, index=test_ids, columns=LABELS))
TIMINGS["rung1_s"] = round(time.time() - t_r1, 1)
log(f"RUNG 1 written: {len(sub1)} rows, arm 1 only, {n_fallback} fallback rows "
    f"({TIMINGS['rung1_s']:.0f}s)")

# %% [markdown]
# ## 📏 Arm 1 on the holdout
#
# The blend weight later is chosen here, not on the 58, so arm 1 needs scores for the same
# holdout the fine-tuned arm will be measured on. If pass 1 was truncated, the holdout shrinks to
# the studies that were decoded, which is still a random sample of it.

# %%
ho_ids = [s for s in holdout_ids if s in idx_of]
if len(ho_ids) < len(holdout_ids):
    log(f"holdout truncated to {len(ho_ids)} of {len(holdout_ids)} decoded studies")
rng.shuffle(ho_ids)          # a time-guarded truncation below must stay a random sample
t_h = time.time()
# Deadline: the fine-tune owns the clock from FT_END_BY, so this eval must end 1.2 h before
# it. At a pathological 41 s/study, 832 studies would cost 9.5 h — the guard, not the list
# length, decides how many actually run.
ho_ids, P1_ho = infer_guarded(arm1, ho_ids, K_EVAL, FT_END_BY - 1.2 * 3600, "arm-1 holdout") \
    if ho_ids else ([], np.zeros((0, 12), np.float32))
TIMINGS["arm1_holdout_s"] = round(time.time() - t_h, 1)
Y_ho_bin = (v5.reindex(ho_ids).fillna(0.0).values > 0.5).astype(int) if ho_ids else np.zeros((0, 12), int)
arm1_ho_weak = mean_auc(Y_ho_bin, P1_ho) if ho_ids else np.nan
log(f"arm 1 on the holdout: {len(ho_ids)} studies, weak-label mean AUC {arm1_ho_weak:.4f} "
    f"({TIMINGS['arm1_holdout_s']:.0f}s)")
np.savez_compressed("/kaggle/tmp/arm1_holdout.npz", ids=np.array(ho_ids), probs=P1_ho)

# %% [markdown]
# ## 🔁 Arm 1 on the test set, second pass (TTA)
#
# Test-time augmentation, kept cheap: the eval windows are evenly spaced over a study's valid
# range, so a second pass with every center shifted one slice is a genuinely different view of
# the same study, and averaging the two passes is the standard free lunch. Test only — the
# holdout and the 58 stay single-pass so every selection number stays comparable to v2, and
# the pass runs now because arm 1 is freed for the fine-tune right after this.

# %%
arm1_test_b = None
if TTA_TEST > 1 and test_ids and \
        time.time() + len(test_ids) * float(np.median(test_times or [1.7])) < FT_END_BY - 1.2 * 3600:
    t_b = time.time()
    arm1_test_b = np.full((len(test_ids), len(LABELS)), 0.5, np.float32)
    for i, s in enumerate(test_ids):
        try:
            arm1_test_b[i] = infer_study(arm1, s, K_EVAL, shift=1)
        except Exception as e:
            log(f"  study {i} {short_uid(s)} FALLBACK ({type(e).__name__}: {e})")
        if (i + 1) % 100 == 0 or i + 1 == len(test_ids):
            log(f"arm-1 test pass 2: {i + 1}/{len(test_ids)}")
    TIMINGS["arm1_test_tta_s"] = round(time.time() - t_b, 1)
    log(f"arm-1 TTA pass done ({TIMINGS['arm1_test_tta_s']:.0f}s)")
else:
    log("arm-1 TTA pass skipped: TTA_TEST=1, or the clock belongs to the fine-tune")

# %% [markdown]
# ## 🎯 Fine-tune arm 2 from the checkpoint
#
# The second arm starts from their weights and continues on the v5 labels, with three deliberate
# departures from a from-scratch run. The learning rate is 1e-5, flat-ish and low, because the
# head is already trained on a near-identical target and the risk is damage, not underfitting.
# There is no horizontal flip, because a flip turns a left knee into a right knee and relabels
# medial as lateral (the first notebook's finding), and no geometric augmentation at all, because
# the 140 mm crop already fixes scale; the augmentation is which 12 of a study's valid windows
# are sampled this epoch. And the gate that picks the epoch is the report-hash holdout against
# the v5 labels, in both flag modes, so selection is never done on the 58 and, in the
# include-gold run, never on data the model saw.
#
# One study forward at a time (its 12 windows pooled by the attention head), four studies per
# optimizer step. Arm 1 is freed first: one model resident at a time.

# %%
del arm1; gc.collect()
if dev == "cuda":
    torch.cuda.empty_cache()
arm2, RES2, _ = load_model(weight_path, dev)
assert RES2 == RES

v5_ft = v5.reindex(ft_ids).fillna(0.0).astype(np.float32)
Y_ft = torch.from_numpy(v5_ft.values).to(dev)
log(f"fine-tune set {len(ft_ids):,} studies, target mean {v5_ft.values.mean():.4f}")

opt = torch.optim.AdamW(arm2.parameters(), lr=LR_FT, weight_decay=WD_FT)
TOTAL_STEPS = EPOCHS_FT * int(np.ceil(len(ft_ids) / ACCUM))
def lr_lambda(step):
    if step < WARMUP_FT:
        return step / max(1, WARMUP_FT)
    p = (step - WARMUP_FT) / max(1, TOTAL_STEPS - WARMUP_FT)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

_MEAN_D, _STD_D = _MEAN.to(dev), _STD.to(dev)

def win_uint8(vol, centers):
    w = np.empty((len(centers), 3, IMG, IMG), np.uint8)
    for j, c in enumerate(centers):
        c = max(1, min(int(c), MAXS - 2))
        w[j, 0] = vol[c - 1]; w[j, 1] = vol[c]; w[j, 2] = vol[c + 1]
    return w

def windows_to_gpu(w):
    x = torch.from_numpy(w).to(dev).float().div_(255)
    if x.shape[-1] != RES:
        x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
    return (x - _MEAN_D) / _STD_D

def gate_auc(model, ids, y_bin, k):
    if not ids:
        return np.nan, np.zeros((0, len(LABELS)), np.float32)
    P = np.stack([infer_study(model, s, k) for s in ids])
    return mean_auc(y_bin, P), P

g_eval = np.random.default_rng(SEED + 7)
# The gate runs once per epoch, so its size is set from the measured eval speed, not a
# constant: a ~3 min gate per epoch at whatever s/study the box actually delivers.
s_per_eval = TIMINGS.get("s_per_study_eval", 12.0) * K_GATE / K_EVAL
n_gate = int(np.clip(180 / max(s_per_eval, 1e-6), 100, 500))
ho_eval_ids = list(g_eval.choice(ho_ids, min(n_gate, len(ho_ids)), replace=False)) if ho_ids else []
Y_ho_eval = (v5.reindex(ho_eval_ids).fillna(0.0).values > 0.5).astype(int) if ho_eval_ids else np.zeros((0, 12), int)
log(f"epoch gate: {len(ho_eval_ids)} holdout studies at {K_GATE} windows "
    f"({s_per_eval:.1f} s/study measured at {K_EVAL})")

# %%
history, snaps = [], []
TIMINGS["finetune_start"] = round(time.time() - T0, 1)
for epoch in range(EPOCHS_FT):
    if time.time() > FT_END_BY:
        log("fine-tune budget spent before this epoch; keeping what is trained")
        break
    g = np.random.default_rng(SEED * 1000 + epoch)
    order = g.permutation(len(ft_ids))
    arm2.train()
    t_ep = time.time(); run = 0.0; seen = 0
    for n_i, i in enumerate(order):
        if time.time() > FT_END_BY:
            log(f"epoch {epoch}: time guard mid-epoch at study {n_i}")
            break
        if n_i == 50:
            per = (time.time() - t_ep) / 50
            if t_ep + per * len(order) > FT_END_BY:
                log(f"epoch {epoch}: projected {per * len(order) / 60:.0f} min does not fit; "
                    f"stopping fine-tune after {seen} studies")
                break
        sid = ft_ids[i]
        vol, mask = get_stack(sid)
        pool = recipe[sid]["centers"] if sid in recipe else window_centers(mask)
        take = min(W_WINDOWS, len(pool))
        cs = g.choice(pool, size=take, replace=False) if take else [MAXS // 2]
        x = windows_to_gpu(win_uint8(vol, cs))
        with torch.autocast("cuda", enabled=(dev == "cuda")):
            logits = arm2(x[None])
            loss = F.binary_cross_entropy_with_logits(logits[0], Y_ft[i]) / ACCUM
        scaler.scale(loss).backward()
        run += float(loss.item()) * ACCUM; seen += 1
        if seen % ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(arm2.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            opt.zero_grad(set_to_none=True)
    if seen == 0:
        break
    gate_v, _ = gate_auc(arm2, ho_eval_ids, Y_ho_eval, K_GATE)
    gold_v = np.nan
    # Per-epoch gold trend at the cheap K_GATE resolution only — it selects nothing, and the
    # headline arm-2 gold number is measured at K_EVAL in the evaluation cell. 42 windows here
    # would cost ~40 min per epoch for a number this notebook never uses to decide anything.
    if not INCLUDE_GOLD and gold_ids and time.time() < FT_END_BY + 600:
        gold_v, _ = gate_auc(arm2, gold_ids, gold.values.astype(int), K_GATE)
    snaps.append((gate_v, {k: v.detach().cpu().clone() for k, v in arm2.state_dict().items()}))
    history.append({"epoch": epoch + 1, "loss": run / max(seen, 1),
                    "holdout_weak_auc": gate_v, "gold58_auc": gold_v, "studies": seen})
    torch.save(snaps[-1][1], f"/kaggle/tmp/ft_epoch{epoch}.pt")
    log(f"epoch {epoch + 1}/{EPOCHS_FT}  loss {run / max(seen, 1):.4f}  "
        f"holdout-weak AUC {gate_v:.4f}  gold-58 {gold_v if np.isfinite(gold_v) else float('nan'):.4f}  "
        f"({(time.time() - t_ep) / 60:.0f} min, {seen} studies)")
TIMINGS["finetune_s"] = round(time.time() - T0 - TIMINGS["finetune_start"], 1)

hist_df = pd.DataFrame(history)
if len(hist_df):
    show(hist_df, "Fine-tune history (selection column: holdout_weak_auc)")

best_epoch = -1
arm2_ready = False
if snaps:
    best_epoch = int(max(range(len(snaps)), key=lambda i: snaps[i][0]))
    arm2.load_state_dict(snaps[best_epoch][1])
    arm2_ready = True
    log(f"selected epoch {best_epoch + 1} by holdout-weak AUC "
        f"({snaps[best_epoch][0]:.4f} vs last {snaps[-1][0]:.4f})")
else:
    log("no epoch completed; rung 1 stands")

# %% [markdown]
# ## ⚖️ Arm 2 measured, then the blend weight
#
# Arm 2 gets the same full-window measurement arm 1 got. In the honest mode the gold-58 numbers
# are the radiologist-grade check, with the bootstrap interval; in the include-gold mode they sit
# under a contamination banner, printed so the log says plainly what the model saw.
#
# The blend weight comes from the holdout, never from the 58. One caveat is worth saying in
# words: both arms are trained on the same label family, so the holdout measures agreement with
# v5, and a gain measured against v5 is an upper bound on the gain against a radiologist. Their
# own live-board experience was that blending near-identical arms bought about +0.001.

# %%
P2_ho = P2_gold = None
arm2_ho_weak = arm2_gold_mean = np.nan
if arm2_ready and ho_ids and time.time() < EVAL_END_BY:
    t_a2 = time.time()
    used2, P2_ho = infer_guarded(arm2, ho_ids, K_EVAL, EVAL_END_BY, "arm-2 holdout")
    if len(used2) < len(ho_ids):          # keep arm 1 / arm 2 / Y row-aligned for the blend
        ho_ids = used2
        P1_ho = P1_ho[:len(ho_ids)]
        Y_ho_bin = Y_ho_bin[:len(ho_ids)]
    arm2_ho_weak = mean_auc(Y_ho_bin, P2_ho) if ho_ids else np.nan
    log(f"arm 2 on the holdout: weak-label mean AUC {arm2_ho_weak:.4f} "
        f"({time.time() - t_a2:.0f}s)")
    # The 58 at 42 windows costs the same per study as any holdout eval; check the projection
    # before starting it rather than discovering the overrun at the deadline.
    est_gold = len(gold_ids) * TIMINGS.get("s_per_study_eval", 12.0)
    if time.time() + est_gold < EVAL_END_BY:
        rows = []
        for i, s in enumerate(gold_ids):
            rows.append(infer_study(arm2, s, K_EVAL))
        P2_gold = np.stack(rows)
        pred2_gold = pd.DataFrame(P2_gold, index=gold_ids, columns=LABELS)
        auc2_gold = score_against_gold(pred2_gold)
        arm2_gold_mean = float(auc2_gold.mean())
    else:
        log(f"arm-2 gold eval skipped: {est_gold:.0f}s projection past the eval budget")
    if P2_gold is None:
        log("arm-2 gold numbers unavailable (eval budget spent)")
    elif INCLUDE_GOLD:
        note(f"**CONTAMINATED**: these 58 rows were in fine-tuning, so "
             f"{arm2_gold_mean:.3f} is not a measurement.")
    else:
        w2, wm2 = bootstrap_auc(gold, pred2_gold)
        show(pd.DataFrame({"arm 1": auc1_gold, "arm 2": auc2_gold}).sort_values("arm 2"),
             f"The two arms against the radiologist: {auc1_gold.mean():.3f} vs "
             f"{arm2_gold_mean:.3f}", bars=["arm 2"], vmin=0.4, vmax=1.0)
        log(f"arm-2 gold mean {arm2_gold_mean:.4f} (95% CI width {wm2:.3f})")
elif not arm2_ready:
    log("arm 2 skipped: no fine-tune epoch completed")
else:
    log("arm 2 eval skipped: evaluation budget spent")

W_BLEND = 0.0
blend_ho_weak = np.nan
if P2_ho is not None and ho_ids:
    ws = np.round(np.arange(0, 1.001, 0.05), 2)
    scores = [mean_auc(Y_ho_bin, blend_ranks(P2_ho, P1_ho, w)) for w in ws]
    curve = pd.DataFrame({"w (arm 2 weight)": ws, "holdout weak AUC": scores})
    W_BLEND = float(curve.loc[curve["holdout weak AUC"].idxmax(), "w (arm 2 weight)"])
    blend_ho_weak = float(curve["holdout weak AUC"].max())
    show(curve, "Blend weight, chosen on the holdout (never on the 58)",
         bars=["holdout weak AUC"], vmin=min(scores) - 0.005, vmax=max(scores) + 0.005)
    facts([("arm 1 alone (w=0)", f"{scores[0]:.4f}"),
           ("arm 2 alone (w=1)", f"{scores[-1]:.4f}"),
           (f"blend at w={W_BLEND}", f"{blend_ho_weak:.4f}"),
           ("blend gain over the better arm", f"{blend_ho_weak - max(scores[0], scores[-1]):+.4f}")],
          "What blending is worth on this label family")
    note("Both arms share the same labels, architecture and preprocessing, so this gain is an "
         "upper bound on the radiologist-grade gain; the published arms that shared everything "
         "gained about +0.001 on the live board.")
    if not INCLUDE_GOLD and P2_gold is not None:
        G = blend_ranks(P2_gold, P1_gold, W_BLEND)
        blend_gold = score_against_gold(pd.DataFrame(G, index=gold_ids, columns=LABELS))
        log(f"blend gold mean {blend_gold.mean():.4f} (reported, not selected on)")
else:
    log("blend skipped: arm 2 has no scores; W_BLEND stays 0.0")

# %% [markdown]
# ## 📈 Two arms per finding, and what the blend weight is worth

# %%
if np.isfinite(arm2_gold_mean):
    fig, ax_l = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(len(LABELS))
    ax_l.bar(xs - 0.2, auc1_gold.loc[LABELS].values, 0.4,
             label=f"arm 1 (mean {auc1_gold.mean():.4f})")
    ax_l.bar(xs + 0.2, auc2_gold.loc[LABELS].values, 0.4,
             label=f"arm 2 (mean {arm2_gold_mean:.4f})")
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels(LABELS, rotation=45, ha="right")
    ax_l.set_ylim(0.5, 1.0)
    ax_l.set_title("Per-finding AUC on the 58" +
                   (" — arm 2 saw them in training (contaminated)" if INCLUDE_GOLD else ""))
    ax_l.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
if np.isfinite(blend_ho_weak):
    fig, ax_r = plt.subplots(figsize=(6.5, 4))
    ax_r.plot(ws, scores, "o-")
    ax_r.axvline(W_BLEND, color="r", ls="--", lw=1, label=f"w* = {W_BLEND:.2f}")
    ax_r.set_xlabel("arm-2 weight in the rank blend")
    ax_r.set_ylabel("holdout weak-label AUC")
    ax_r.set_title("Blend weight, chosen on the holdout (never on the 58)")
    ax_r.legend()
    plt.tight_layout()
    plt.show()
if not np.isfinite(arm2_gold_mean) and not np.isfinite(blend_ho_weak):
    log("arm-2 measurements unavailable this run; no two-arm charts")

# %% [markdown]
# ## 🏁 Rung 2: the blended submission
#
# All-or-nothing, by the same rule the first notebook used for its model stage: the projected
# cost of arm 2 on the test set is measured from arm 1's per-study time, and if it does not fit
# before the final-write deadline, submission.csv is not touched. A scored run that loses arm 2
# still has rung 1 on disk.

# %%
ran_rung2 = False
if arm2_ready and P2_ho is not None and time.time() < TEST_START_BY:
    per_study = float(np.median(test_times)) if test_times else 1.0
    projected = time.time() + len(test_ids) * per_study * 1.5 + 300
    if projected < FINAL_WRITE_BY:
        t_r2 = time.time()
        arm2_test = np.full((len(test_ids), len(LABELS)), 0.5, np.float32)
        for i, s in enumerate(test_ids):
            try:
                arm2_test[i] = infer_study(arm2, s, K_EVAL)
            except Exception as e:
                log(f"  study {i} {short_uid(s)} FALLBACK ({type(e).__name__}: {e})")
            if (i + 1) % 100 == 0 or i + 1 == len(test_ids):
                log(f"arm-2 test: {i + 1}/{len(test_ids)}")
        ranks = blend_ranks(arm2_test, arm1_test, W_BLEND)
        sub2 = write_submission(pd.DataFrame(ranks, index=test_ids, columns=LABELS))
        assert np.isfinite(sub2[LABELS].values).all(), "non-finite value in the blended submission"
        TIMINGS["rung2_s"] = round(time.time() - t_r2, 1)
        ran_rung2 = True
        log(f"RUNG 2 written: blend at w={W_BLEND}, arm 1 + arm 2 ({TIMINGS['rung2_s']:.0f}s)")
    else:
        log(f"rung 2 skipped: projected finish {projected - T0:.0f}s is past the "
            f"final-write deadline; rung 1 stands")
else:
    log("rung 2 skipped: no fine-tuned arm or too late; rung 1 stands")

# %% [markdown]
# ## 🔁 Arm 2 on the test set, second pass (TTA)
#
# Same shifted-center second pass as arm 1 got, while arm 2 is still resident.

# %%
arm2_test_b = None
if ran_rung2 and TTA_TEST > 1 and test_ids:
    per_study_b = float(np.median(test_times)) if test_times else 1.7
    if time.time() + len(test_ids) * per_study_b * 1.5 + 300 < FINAL_WRITE_BY - 3600:
        t_b2 = time.time()
        arm2_test_b = np.full((len(test_ids), len(LABELS)), 0.5, np.float32)
        for i, s in enumerate(test_ids):
            try:
                arm2_test_b[i] = infer_study(arm2, s, K_EVAL, shift=1)
            except Exception as e:
                log(f"  study {i} {short_uid(s)} FALLBACK ({type(e).__name__}: {e})")
            if (i + 1) % 100 == 0 or i + 1 == len(test_ids):
                log(f"arm-2 test pass 2: {i + 1}/{len(test_ids)}")
        TIMINGS["arm2_test_tta_s"] = round(time.time() - t_b2, 1)
        log(f"arm-2 TTA pass done ({TIMINGS['arm2_test_tta_s']:.0f}s)")
    else:
        log("arm-2 TTA pass skipped: the projection crosses the arm-3 window")
else:
    log("arm-2 TTA pass skipped: rung 2 did not run, or TTA_TEST=1")

# %% [markdown]
# ## 🧬 Arm 3: the SWA checkpoint
#
# The dataset ships a second checkpoint, `raptor_ft_coatnet_v4_full_swa.pt` — stochastic
# weight averaging over the same training run. Same architecture, same label family, a
# different point in weight space: the cheapest diversity there is. It gets no fine-tuning;
# it is a third opinion, measured exactly like the other two.

# %%
P3_ho = P3_gold = None
arm3_ho_weak = arm3_gold_mean = np.nan
weight_path_swa = find_weight_file(CKPT_FILE_SWA)
arm3, RES3, ck3 = load_model(weight_path_swa, dev)
assert RES3 == RES, f"SWA checkpoint res {RES3} != {RES}"
if "lab" in ck3:
    assert list(ck3["lab"]) == LABELS, "SWA checkpoint label order differs"
log(f"arm 3 (SWA) loaded: stored gold_auc {ck3.get('gold_auc')}, epoch {ck3.get('epoch')}")
if ho_ids and time.time() < EVAL_END_BY:
    t_a3 = time.time()
    used3, P3_ho = infer_guarded(arm3, ho_ids, K_EVAL, EVAL_END_BY, "arm-3 holdout")
    if len(used3) < len(ho_ids):          # keep every arm and Y row-aligned for the blend
        ho_ids = used3
        P1_ho = P1_ho[:len(ho_ids)]
        if P2_ho is not None:
            P2_ho = P2_ho[:len(ho_ids)]
        Y_ho_bin = Y_ho_bin[:len(ho_ids)]
    arm3_ho_weak = mean_auc(Y_ho_bin, P3_ho) if ho_ids else np.nan
    log(f"arm 3 on the holdout: weak-label mean AUC {arm3_ho_weak:.4f} "
        f"({time.time() - t_a3:.0f}s)")
    est_gold3 = len(gold_ids) * TIMINGS.get("s_per_study_eval", 12.0)
    if time.time() + est_gold3 < EVAL_END_BY:
        P3_gold = np.stack([infer_study(arm3, s, K_EVAL) for s in gold_ids])
        auc3_gold = score_against_gold(pd.DataFrame(P3_gold, index=gold_ids, columns=LABELS))
        arm3_gold_mean = float(auc3_gold.mean())
        log(f"arm-3 gold mean {arm3_gold_mean:.4f}")
    else:
        log("arm-3 gold eval skipped: projection past the eval budget")
else:
    log("arm 3 evals skipped: no decoded holdout, or the eval budget is spent")

# %% [markdown]
# ## 🤝 The three-way blend, weights chosen on the holdout
#
# A simplex grid over (arm 1, arm 2, arm 3) in steps of 0.1, scored on the same holdout rows
# the two-way weight came from. An arm with no holdout scores is forced to weight zero, so
# the grid degrades to the two-way or one-way case instead of failing.

# %%
W3 = (1.0, 0.0, 0.0)
blend3_ho_weak = np.nan
if ho_ids and (P2_ho is not None or P3_ho is not None):
    R1 = rankpct(np.clip(P1_ho, 0, 1))
    R2 = rankpct(np.clip(P2_ho, 0, 1)) if P2_ho is not None else None
    R3 = rankpct(np.clip(P3_ho, 0, 1)) if P3_ho is not None else None
    grid = []
    for w2i in range(0, 11):
        for w3i in range(0, 11 - w2i):
            w2, w3 = w2i / 10.0, w3i / 10.0
            if (R2 is None and w2 > 0) or (R3 is None and w3 > 0):
                continue
            w1 = round(1.0 - w2 - w3, 1)
            mix = w1 * R1 + (w2 * R2 if R2 is not None else 0.0) + \
                  (w3 * R3 if R3 is not None else 0.0)
            grid.append(((w1, w2, w3), mean_auc(Y_ho_bin, mix)))
    W3, blend3_ho_weak = max(grid, key=lambda t: t[1])
    g_lo = min(g[1] for g in grid) - 0.005
    g_hi = max(g[1] for g in grid) + 0.005
    top5 = sorted(grid, key=lambda t: -t[1])[:5]
    show(pd.DataFrame({"w (arm 1)": [t[0][0] for t in top5],
                       "w (arm 2)": [t[0][1] for t in top5],
                       "w (arm 3)": [t[0][2] for t in top5],
                       "holdout weak AUC": [round(t[1], 4) for t in top5]}),
         f"Best three-way blends on the holdout — w* = {W3}, weak AUC {blend3_ho_weak:.4f}",
         bars=["holdout weak AUC"], vmin=g_lo, vmax=g_hi)
    log(f"3-way blend: w*={W3} holdout weak AUC {blend3_ho_weak:.4f} "
        f"(two-way was {blend_ho_weak:.4f}, arm 1 alone {scores[0] if np.isfinite(blend_ho_weak) else arm1_ho_weak:.4f})")
else:
    log("3-way blend skipped: no second arm has holdout scores; W3 stays (1, 0, 0)")

# %% [markdown]
# ## 🏆 Rung 3: the three-arm submission
#
# All-or-nothing, same rule as rung 2: arm 3 predicts the test set (two center-shifted
# passes when TTA_TEST allows), the three arms blend at the holdout-chosen weights, and if
# the projection does not fit before the final-write deadline, submission.csv is not
# touched and rung 2 stands.

# %%
ran_rung3 = False
if ho_ids and P3_ho is not None and time.time() < TEST_START_BY:
    per_study3 = float(np.median(test_times)) if test_times else 1.7
    projected3 = time.time() + len(test_ids) * per_study3 * TTA_TEST * 1.5 + 300
    if projected3 < FINAL_WRITE_BY:
        t_r3 = time.time()
        arm3_test = np.full((len(test_ids), len(LABELS)), 0.5, np.float32)
        for i, s in enumerate(test_ids):
            try:
                arm3_test[i] = infer_study(arm3, s, K_EVAL)
            except Exception as e:
                log(f"  study {i} {short_uid(s)} FALLBACK ({type(e).__name__}: {e})")
            if (i + 1) % 100 == 0 or i + 1 == len(test_ids):
                log(f"arm-3 test: {i + 1}/{len(test_ids)}")
        arm3_test_b = None
        if TTA_TEST > 1 and time.time() + len(test_ids) * per_study3 * 1.5 + 300 < FINAL_WRITE_BY:
            arm3_test_b = np.full((len(test_ids), len(LABELS)), 0.5, np.float32)
            for i, s in enumerate(test_ids):
                try:
                    arm3_test_b[i] = infer_study(arm3, s, K_EVAL, shift=1)
                except Exception as e:
                    log(f"  study {i} {short_uid(s)} FALLBACK ({type(e).__name__}: {e})")
                if (i + 1) % 100 == 0 or i + 1 == len(test_ids):
                    log(f"arm-3 test pass 2: {i + 1}/{len(test_ids)}")
        del arm3; gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
        r1 = rankpct(np.clip(arm1_test if arm1_test_b is None else (arm1_test + arm1_test_b) / 2, 0, 1))
        r2 = rankpct(np.clip(arm2_test if arm2_test_b is None else (arm2_test + arm2_test_b) / 2, 0, 1)) \
            if ran_rung2 else None
        r3 = rankpct(np.clip(arm3_test if arm3_test_b is None else (arm3_test + arm3_test_b) / 2, 0, 1))
        w1, w2, w3 = W3
        w_sum = w1 + (w2 if r2 is not None else 0.0) + w3
        ranks3 = (w1 * r1 + (w2 * r2 if r2 is not None else 0.0) + w3 * r3) / w_sum
        sub3 = write_submission(pd.DataFrame(ranks3, index=test_ids, columns=LABELS))
        assert np.isfinite(sub3[LABELS].values).all(), "non-finite value in the 3-way submission"
        TIMINGS["rung3_s"] = round(time.time() - t_r3, 1)
        ran_rung3 = True
        log(f"RUNG 3 written: 3-way blend w={W3}, TTA passes={TTA_TEST} "
            f"({TIMINGS['rung3_s']:.0f}s)")
    else:
        log(f"rung 3 skipped: projected finish {projected3 - T0:.0f}s is past the "
            f"final-write deadline; rung 2 stands")
else:
    log("rung 3 skipped: no arm-3 scores or too late; rung 2 stands")

# %% [markdown]
# ## 🖼️ What one study looks like to the model

# %%
if USE_MM and n_done and gold_ids[0] in idx_of:
    vol0 = np.asarray(STACKS[idx_of[gold_ids[0]]])
    slot_names = ["Sagittal FS", "Sagittal", "Coronal FS", "Coronal", "Axial"]
    bounds = np.cumsum([s[2] for s in SLOTS])
    picks = np.linspace(0, MAXS - 1, 16).round().astype(int)
    fig, axes = plt.subplots(4, 4, figsize=(8.5, 8.5))
    for a, i in zip(axes.flat, picks):
        a.imshow(vol0[i], cmap="gray")
        a.set_title(f"slot {i}: {slot_names[int(np.searchsorted(bounds, i, side='right'))]}",
                    fontsize=8)
        a.axis("off")
    fig.suptitle(f"16 of {MAXS} slots, one radiologist-labelled study "
                 f"({short_uid(gold_ids[0])}) — decoded first in pass 1")
    plt.tight_layout()
    plt.show()
else:
    log("no decoded stacks cached this run; skipping the study grid")

# %% [markdown]
# ## 📊 The measurements this run leaves behind

# %%
sub_final = pd.read_csv("submission.csv")
assert list(sub_final.columns) == ["StudyInstanceUID"] + LABELS
assert len(sub_final) == len(test)
assert sub_final[LABELS].notna().all().all()

MEAS = {
    "include_gold": INCLUDE_GOLD,
    "holdout_frac": HOLDOUT_FRAC,
    "n_decoded": n_done,
    "pass1_truncated": truncated,
    "use_memmap": USE_MM,
    "n_holdout_used": len(ho_ids),
    "n_train_ft": len(ft_ids),
    "epochs_run": len(history),
    "windows_per_epoch": W_WINDOWS,
    "batch_studies": ACCUM,
    "lr": LR_FT,
    "selected_epoch": best_epoch + 1 if best_epoch >= 0 else None,
    "ckpt_file": CKPT_FILE,
    "ckpt_gold_auc": float(ck_meta.get("gold_auc", np.nan)),
    "ckpt_epoch": ck_meta.get("epoch"),
    "arch": str(ck_meta.get("arch")),
    "res": RES,
    "label_file": LABEL_FILE,
    "v5_contains_gold": bool(check["contains the answers"]),
    "arm1_gold_mean": float(auc1_gold.mean()),
    "arm1_gold_ci95_width": round(w_mean1, 4),
    "arm1_holdout_weak": None if not np.isfinite(arm1_ho_weak) else round(arm1_ho_weak, 4),
    "arm2_gold_mean": None if not np.isfinite(arm2_gold_mean) else round(arm2_gold_mean, 4),
    "arm2_gold_contaminated": INCLUDE_GOLD,
    "arm2_holdout_weak": None if not np.isfinite(arm2_ho_weak) else round(arm2_ho_weak, 4),
    "blend_w": W_BLEND,
    "blend_holdout_weak": None if not np.isfinite(blend_ho_weak) else round(blend_ho_weak, 4),
    "arm3_swa_file": CKPT_FILE_SWA,
    "arm3_gold_mean": None if not np.isfinite(arm3_gold_mean) else round(arm3_gold_mean, 4),
    "arm3_holdout_weak": None if not np.isfinite(arm3_ho_weak) else round(arm3_ho_weak, 4),
    "blend3_w": W3,
    "blend3_holdout_weak": None if not np.isfinite(blend3_ho_weak) else round(blend3_ho_weak, 4),
    "tta_test_passes": TTA_TEST,
    "rung_written": 3 if ran_rung3 else (2 if ran_rung2 else 1),
    "per_finding_arm1_gold": {c: round(float(v), 4) for c, v in auc1_gold.items()},
    "timings": TIMINGS,
    "seed": SEED,
    "versions": {"torch": torch.__version__, "timm": timm.__version__,
                 "pydicom": pydicom.__version__, "cv2": cv2.__version__},
}
facts([("run type", "submission (gold in fine-tune)" if INCLUDE_GOLD else "measurement (gold held out)"),
       ("studies decoded", n_done),
       ("fine-tune epochs", len(history)),
       ("arm 1 gold mean", round(float(auc1_gold.mean()), 4)),
       ("arm 2 gold mean", None if not np.isfinite(arm2_gold_mean) else round(arm2_gold_mean, 4)),
       ("arm 3 gold mean (SWA)", None if not np.isfinite(arm3_gold_mean) else round(arm3_gold_mean, 4)),
       ("blend weight (2-way)", W_BLEND),
       ("blend weights (3-way)", W3),
       ("submission rung",
        "3 (three-arm blend)" if ran_rung3 else ("2 (blend)" if ran_rung2 else "1 (arm 1 only)"))],
      "This run")
log("MEASUREMENT_JSON " + json.dumps(MEAS, separators=(",", ":"), default=str))
log(f"submission.csv ready: {len(sub_final)} rows x {len(sub_final.columns)} cols")

# %% [markdown]
# ## ⏱️ Where the hours went

# %%
phases = [("pass1_read_s", "pass 1: decode every study"), ("arm1_gold_s", "arm 1 on the 58"),
          ("arm1_holdout_s", "arm 1 on the holdout"), ("finetune_s", "fine-tune arm 2"),
          ("rung1_s", "rung 1: arm 1 on test"), ("arm1_test_tta_s", "arm 1 test, TTA pass 2"),
          ("rung2_s", "rung 2: blend on test"), ("arm2_test_tta_s", "arm 2 test, TTA pass 2"),
          ("rung3_s", "rung 3: three-arm test + blend")]
timed = [(nm, TIMINGS[k] / 60.0) for k, nm in phases if k in TIMINGS]
if timed:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh([t[0] for t in timed], [t[1] for t in timed])
    ax.set_xlabel("minutes")
    ax.set_title(f"one T4, total {(time.time() - T0) / 3600:.1f} h")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 💡 What the runs behind 0.926 settled
#
# - **The reproduction landed between their two published numbers.** This pipeline prints
#   0.9128 on the 58; the checkpoint stores 0.9167 and their notebook's comment says 0.9054
#   for the same file. One of those described a different run.
# - **The 58-study gate predicts the leaderboard.** Blend gate 0.9170 here became 0.926 on
#   the public board; the first notebook's 0.778 became 0.798. Same direction both times,
#   about +0.01–0.02, so the 58 stay the steering instrument — for direction, not for
#   ranking close models.
# - **Putting the 58 into training bought nothing out-of-sample.** Holdout weak AUC 0.9092
#   with them in fine-tuning against 0.9097 with them out. The final submission still comes
#   from the gold-in run: the holdout says it costs nothing, and 52 extra radiologist-read
#   studies cannot hurt on the hidden test.
# - **A bug in the original pipeline, kept visible.** Their `_pick_series_for_slot` does
#   `int(r.get('Fluid_Sensitive', 0) or 0)`; NaN is truthy, `int(nan)` raises, and their
#   per-study try/except turns those studies into silent 0.5 submissions. This port treats
#   NaN as no-preference and logs whether the column exists.
# - **The weak findings are the soft-tissue ones.** Synovitis (~0.78) and patellofemoral OA
#   (~0.82) sit well below the rest; a mean over twelve findings hides that.

# %% [markdown]
# ## 🧭 What to do with this run
#
# Two pushes are planned. This one ran with `INCLUDE_GOLD=False`, so its numbers are measurements:
# the arm-1 reproduction against their two published numbers, arm 2's gain, the blend curve, all
# against both the holdout and the radiologist's 58. The next push flips the flag to `True`, the
# 58 go into fine-tuning as the exact annotations they are, and that run is the submission.
# `MEASUREMENT_JSON` in the log carries `include_gold`, so the two runs stay distinguishable
# afterwards.
#
# After both runs land: `FINAL_RESULTS.md` gets a model-v2 section with the reproduction verdict,
# both arms, the blend, both leaderboard scores and the timings; the README gets the second
# notebook; and the runs decide whether a third idea (their SWA checkpoint as a third arm, or a
# longer fine-tune on better labels) is worth a GPU-hour.
