# %% [markdown]
# # RSNA Knee: masked-loss 2.5D baseline
#
# A deliberately small baseline that applies one finding from
# [58 Labels, 9 Languages, a Leak](https://www.kaggle.com/code/paiky1995/rsna-knee-58-labels-9-languages-a-leak).
#
# A radiology report often says nothing about a finding, and that is not the same as the
# radiologist ruling it out. Measured on the 58 annotated studies, when a language model reading
# the report answers "cannot tell", the finding is present 8.2% of the time on average and 34.1%
# of the time for Synovitis. Every published training notebook trains those entries toward zero.
# This one leaves them out of the loss instead, and weights the rest by how much independent
# labelers agree for that report's language.
#
# The model is a ResNet18 over 16 sagittal slices, trained on about 700 studies out of 4,407. It
# will not trouble the leaderboard. It exists to make the recipe runnable rather than to win.
#
# ### What you get if you fork this
#
# 1. A masked and weighted loss you can drop into a stronger model, in the cell marked `LOSS`.
# 2. No horizontal flip, and the reason why in one line of comment.
# 3. Rank transformed output, which is free because the metric reads only the order.

# %%
import os, pathlib, io, json, warnings
import numpy as np, pandas as pd, pydicom, torch, torch.nn as nn, torchvision
from PIL import Image

warnings.filterwarnings("ignore")
COMP = pathlib.Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
if not COMP.exists():
    COMP = pathlib.Path("/kaggle/input/rsna-knee-abnormality-detection")
WEIGHTS = next(pathlib.Path("/kaggle/input").rglob("model.pt"), None)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    # A real operation, so a mismatched kernel image fails now rather than in an hour.
    _ = (torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).sum().item()
print(f"data     {COMP}")
print(f"weights  {WEIGHTS}")
print(f"device   {dev}")

test = pd.read_csv(COMP / "test.csv")
test_series = pd.read_csv(COMP / "test_series.csv")
print(f"test studies {len(test)}  test series {len(test_series)}")

# %% [markdown]
# ## Picking one series per study
#
# The trained model reads a single sagittal fluid sensitive series, so inference has to choose the
# same kind of series it was trained on. Where a study has no sagittal series the best available
# one is used instead.

# %%
def pick_series(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["rank"] = d.Anatomical_Plane.eq("Sagittal") * 2 + d.Fluid_Sensitive.fillna(0)
    return (d.sort_values("rank", ascending=False)
            .groupby("StudyInstanceUID").SeriesInstanceUID.first().to_dict())

chosen = pick_series(test_series)
print(f"chose a series for {len(chosen)} of {len(test)} studies")

N_SLICES, SIZE = 16, 256

def build_volume(study: str, series: str) -> np.ndarray:
    """Same preprocessing as training. Percentile clip, resize, evenly spaced slices."""
    folder = COMP / "test_series" / study / series
    files = sorted(folder.glob("*.dcm"))
    slices = []
    for f in files:
        try:
            d = pydicom.dcmread(str(f))
            slices.append((float(getattr(d, "InstanceNumber", 0) or 0), d))
        except Exception:
            continue
    vol = np.zeros((N_SLICES, SIZE, SIZE), dtype=np.float32)
    if not slices:
        return vol
    slices.sort(key=lambda t: t[0])
    idx = np.linspace(0, len(slices) - 1, N_SLICES).round().astype(int)
    for j, i in enumerate(idx):
        a = slices[i][1].pixel_array.astype(np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        vol[j] = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                            .resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    return vol

# %% [markdown]
# ## LOSS
#
# This is the cell worth taking. It is not used at inference and is here so the recipe travels
# with the submission.
#
# Two corrections are applied, and they are independent of each other. `weight` says how much
# independent labelers agree for this report's language and finding, and it runs from 0.701 for
# Bulgarian to 0.855 for English. `unstated` says the report does not mention the finding at all,
# which is 27.3% of all study and finding pairs and 70.7% of them for Synovitis. Those entries are
# removed from the loss rather than trained toward zero.

# %%
def masked_weighted_bce(logits, target, weight, unstated):
    """Mean over the entries that survive the mask, not over all twelve."""
    w = weight * (~unstated).float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * w
    return loss.sum() / w.sum().clamp(min=1.0)

# Training used one more rule that costs nothing to state. Five of the twelve findings name a
# side of the knee, and a mirrored left knee reads as a right knee, so a horizontal flip would
# relabel medial as lateral. Shift and brightness only.
print("loss and augmentation rules defined")

# %% [markdown]
# ## Predict
#
# If no trained weights are attached the notebook still produces a valid submission by writing a
# constant, so the run never fails for a missing dataset.

# %%
def make_model(n_ch: int) -> nn.Module:
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(n_ch, 64, 7, 2, 3, bias=False)
    m.fc = nn.Linear(512, len(LABELS))
    return m

if WEIGHTS is None:
    print("no weights attached, writing a constant submission")
    pred = pd.DataFrame(0.5, index=test.StudyInstanceUID, columns=LABELS)
else:
    ck = torch.load(WEIGHTS, map_location=dev, weights_only=False)
    model = make_model(ck.get("n_ch", N_SLICES)).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded weights trained for {ck['cfg']['epochs']} epochs "
          f"on {ck['cfg']['label_file']}")

    rows, ids, BS = [], [], 8
    with torch.no_grad():
        batch = []
        for sid in test.StudyInstanceUID:
            batch.append(build_volume(sid, chosen.get(sid, "")))
            ids.append(sid)
            if len(batch) == BS:
                x = torch.from_numpy(np.stack(batch)).to(dev)
                rows.append(torch.sigmoid(model(x)).float().cpu().numpy())
                batch = []
        if batch:
            x = torch.from_numpy(np.stack(batch)).to(dev)
            rows.append(torch.sigmoid(model(x)).float().cpu().numpy())
    pred = pd.DataFrame(np.concatenate(rows), index=ids, columns=LABELS)

print(pred.describe().round(3).to_string())

# %% [markdown]
# ## Write the submission
#
# The score reads only the order within each column, so a rank transform is free and it removes
# any question about how well the probabilities are spread out.

# %%
sub = pred.rank(pct=True)
sub.index.name = "StudyInstanceUID"
sub = sub.reindex(test.StudyInstanceUID).fillna(0.5).reset_index()
sub = sub[["StudyInstanceUID"] + LABELS]
sub.to_csv("submission.csv", index=False)

assert len(sub) == len(test), f"{len(sub)} rows against {len(test)} test studies"
assert list(sub.columns) == ["StudyInstanceUID"] + LABELS, "column order does not match"
assert sub[LABELS].notna().all().all(), "found a missing prediction"
print(sub.head().to_string(index=False))
print(f"\nwrote submission.csv with {len(sub)} rows")
