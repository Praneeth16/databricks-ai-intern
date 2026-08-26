#!/usr/bin/env python3
"""Train the minimal 2.5D baseline on Databricks serverless GPU.

One model per row of versions.yaml, which is the ledger pattern the S6E8 example uses.

The loss is the point of this script rather than the architecture. Section 6 of the
notebook measured that a report is silent about a finding far more often than it rules
it out, and that for Synovitis about a third of the silent entries are positive. So a
silent entry is masked out of the loss instead of being trained toward zero, and every
remaining entry is weighted by how much independent labelers agree for that study's
language and finding.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _dbx  # noqa: E402

JOB = r'''
import json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

WORK = os.environ["KNEE_WORK"]
CFG = json.loads(os.environ["KNEE_CFG"])
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {dev}  torch {torch.__version__}")
if dev == "cuda":
    # Probe with a real operation so a bad kernel image fails in seconds, not an hour.
    print(f"gpu {torch.cuda.get_device_name(0)}  "
          f"matmul ok {bool((torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().isfinite())}")

studies = sorted(p[:-4] for p in os.listdir(f"{WORK}/arrays") if p.endswith(".npy"))
print(f"ingested studies {len(studies)}")

weak = pd.read_csv(f"{WORK}/{CFG['label_file']}").drop_duplicates("StudyInstanceUID")
weak = weak.set_index("StudyInstanceUID")[LABELS].astype(float)
conf = pd.read_csv(f"{WORK}/label_confidence.csv")
W = conf.pivot(index="StudyInstanceUID", columns="finding", values="weight")[LABELS]
M = conf.pivot(index="StudyInstanceUID", columns="finding", values="unstated")[LABELS]

studies = [s for s in studies if s in weak.index and s in W.index]
print(f"usable studies {len(studies)}")

train_full = pd.read_csv(f"{WORK}/train.csv")
gold = (train_full[train_full[LABELS].notna().any(axis=1)]
        .set_index("StudyInstanceUID")[LABELS].astype(int))
anchor = [s for s in studies if s in gold.index]
print(f"annotated studies inside the pilot: {len(anchor)}")

rng = np.random.default_rng(CFG["seed"])
pool = [s for s in studies if s not in set(anchor)]
rng.shuffle(pool)
n_val = int(len(pool) * CFG["val_frac"])
val_ids, tr_ids = pool[:n_val], pool[n_val:]
print(f"train {len(tr_ids)}  val {len(val_ids)}  anchor {len(anchor)}")

class KneeSet(Dataset):
    def __init__(self, ids, train):
        self.ids, self.train = ids, train
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        sid = self.ids[i]
        x = np.load(f"{WORK}/arrays/{sid}.npy").astype(np.float32)
        if self.train:
            # No horizontal flip. Five of the twelve findings name a side of the knee,
            # and a mirrored left knee reads as a right knee, so a flip would relabel
            # medial as lateral.
            if np.random.rand() < 0.5:
                s = np.random.randint(-12, 13, 2)
                x = np.roll(x, s, axis=(1, 2))
            x = x * np.float32(np.random.uniform(0.9, 1.1))
        y = weak.loc[sid, LABELS].values.astype(np.float32)
        if CFG.get("disable_mask"):
            # The ablation. Every entry counts equally, silent ones included, which is
            # what every published training notebook does today.
            w = np.ones(len(LABELS), dtype=np.float32)
        else:
            w = W.loc[sid, LABELS].values.astype(np.float32)
            w = w * (~M.loc[sid, LABELS].values.astype(bool)).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)

def make_model(n_ch):
    import torchvision
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    old = m.conv1.weight.data
    m.conv1 = nn.Conv2d(n_ch, 64, 7, 2, 3, bias=False)
    # Reuse the pretrained stem by averaging its RGB filters across the new channels.
    m.conv1.weight.data = old.mean(1, keepdim=True).repeat(1, n_ch, 1, 1) * (3.0 / n_ch)
    m.fc = nn.Linear(512, len(LABELS))
    return m

n_ch = np.load(f"{WORK}/arrays/{studies[0]}.npy").shape[0]
model = make_model(n_ch).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
dl_tr = DataLoader(KneeSet(tr_ids, True), batch_size=CFG["bs"], shuffle=True,
                   num_workers=4, drop_last=True)
dl_va = DataLoader(KneeSet(val_ids, False), batch_size=CFG["bs"], num_workers=4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"],
                                            total_steps=CFG["epochs"] * max(len(dl_tr), 1))
bce = nn.BCEWithLogitsLoss(reduction="none")
scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

def evaluate(ids, target):
    model.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(ids), CFG["bs"]):
            xb = torch.stack([torch.from_numpy(
                np.load(f"{WORK}/arrays/{s}.npy").astype(np.float32))
                for s in ids[i:i + CFG["bs"]]]).to(dev)
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                P.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    P = np.concatenate(P)
    aucs = {}
    for j, c in enumerate(LABELS):
        y = target.loc[ids, c].values
        yb = (y > 0.5).astype(int)
        if len(np.unique(yb)) > 1:
            aucs[c] = roc_auc_score(yb, P[:, j])
    return float(np.mean(list(aucs.values()))), aucs, P

hist = []
for ep in range(CFG["epochs"]):
    model.train()
    tot, n, t0 = 0.0, 0, time.time()
    for xb, yb, wb in dl_tr:
        xb, yb, wb = xb.to(dev), yb.to(dev), wb.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
            loss_el = bce(model(xb), yb) * wb
            # Mean over the entries that survive the mask, not over all 12.
            loss = loss_el.sum() / wb.sum().clamp(min=1.0)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        tot += float(loss) * len(xb); n += len(xb)
    va, _, _ = evaluate(val_ids, weak)
    row = {"epoch": ep, "loss": tot / max(n, 1), "val_auc_weak": va,
           "sec": round(time.time() - t0, 1)}
    if anchor:
        row["anchor_auc"] = evaluate(anchor, gold)[0]
    hist.append(row)
    print(json.dumps(row), flush=True)

va, per_label, _ = evaluate(val_ids, weak)
print(f"\nfinal val AUC against the weak labels: {va:.4f}")
print(json.dumps({k: round(v, 4) for k, v in per_label.items()}, indent=2))
if anchor:
    aa, ap, _ = evaluate(anchor, gold)
    print(f"\nAUC against the {len(anchor)} annotated studies in the pilot: {aa:.4f}")
    print("That is 11 or so studies. Treat it as a smoke test, not a measurement.")

out = f"{WORK}/models/{CFG['version']}"
os.makedirs(out, exist_ok=True)
torch.save({"state_dict": model.state_dict(), "n_ch": n_ch, "labels": LABELS,
            "cfg": CFG}, f"{out}/model.pt")
pd.DataFrame(hist).to_csv(f"{out}/history.csv", index=False)
json.dump({"version": CFG["version"], "val_auc_weak": va,
           "anchor_auc": (aa if anchor else None), "n_train": len(tr_ids),
           "per_label": per_label}, open(f"{out}/result.json", "w"), indent=2)
print(f"\nwrote {out}/model.pt")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--gpu", default="GPU_1xA10")
    ap.add_argument("--timeout-min", type=int, default=180)
    ap.add_argument("--ledger", default=None)
    args = ap.parse_args()

    ledger_path = pathlib.Path(args.ledger or (pathlib.Path(__file__).parent.parent / "versions.yaml"))
    rows = yaml.safe_load(ledger_path.read_text())["versions"]
    cfg = next((r for r in rows if r["version"] == args.version), None)
    if cfg is None:
        raise SystemExit(f"{args.version} is not in {ledger_path}; known: "
                         f"{[r['version'] for r in rows]}")
    print(json.dumps(cfg, indent=2))

    wc = _dbx.workspace_client()
    script = (f"import os\nos.environ['KNEE_WORK'] = {_dbx.WORK!r}\n"
              f"os.environ['KNEE_CFG'] = {json.dumps(cfg)!r}\n" + JOB)
    sub = _dbx.submit(wc, script, name=f"knee_train_{args.version}.py", gpu=args.gpu,
                      deps=["torch>=2.4", "torchvision>=0.19", "scikit-learn>=1.4",
                            "pandas>=2.0", "pyarrow>=15.0"],
                      timeout_min=args.timeout_min)
    print(f"submitted run {sub['run_id']}")
    _dbx.wait(wc, sub["run_id"], timeout_min=args.timeout_min)
    print("\n--- run output ---")
    print(_dbx.run_output(wc, sub["run_id"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
