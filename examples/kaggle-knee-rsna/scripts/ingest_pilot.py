#!/usr/bin/env python3
"""Ingest the pilot studies into a UC Volume, on Databricks serverless compute.

The full corpus is about 512 GB across 24,371 series. The audit in audit_labels.py
needs none of it. The model needs some, so this takes one sagittal fluid sensitive
series for each of about 700 studies, which is roughly 15 GB.

The download runs on Databricks rather than a laptop for the bandwidth and for the
parallelism. Headers are pulled in the same pass, because the job already opens every
file and a second walk over 15 GB would buy nothing.

Kaggle credentials are read from a Databricks secret scope and passed as dynamic
secret references, so the key never appears in the job payload or in an environment
variable baked into the request.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _dbx  # noqa: E402

JOB = r'''
import concurrent.futures as cf, io, json, os, time
import numpy as np, pandas as pd, pydicom, requests
from PIL import Image

WORK = os.environ["KNEE_WORK"]
COMP = "rsna-knee-abnormality-detection"
N_STUDIES = int(os.environ["KNEE_N_STUDIES"])
N_SLICES = int(os.environ["KNEE_N_SLICES"])
SIZE = int(os.environ["KNEE_SIZE"])
USER = dbutils.widgets.get("KAGGLE_USERNAME")
KEY = dbutils.widgets.get("KAGGLE_KEY")

manifest = pd.read_csv(f"{WORK}/manifest.csv")
series_meta = pd.read_csv(f"{WORK}/train_series.csv")
print(f"manifest rows {len(manifest):,}  series rows {len(series_meta):,}")

m = manifest[manifest.path.str.startswith("train_series/")].copy()
parts = m.path.str.split("/", expand=True)
m["study"], m["series"] = parts[1], parts[2]

# One series per study. Prefer sagittal and fluid sensitive, which is the plane the
# meniscus and cruciate findings are read on.
sm = series_meta.set_index("SeriesInstanceUID")
m = m.join(sm[["Anatomical_Plane", "Fluid_Sensitive"]], on="series")
m["rank"] = (m.Anatomical_Plane.eq("Sagittal") * 2 + m.Fluid_Sensitive.fillna(0)).astype(int)

chosen, sizes = {}, m.groupby("series").size()
for study, g in m.groupby("study"):
    best = g.sort_values("rank", ascending=False).series.iloc[0]
    if sizes.get(best, 0) >= 8:
        chosen[study] = best
studies = sorted(chosen)[:N_STUDIES]
print(f"studies to ingest: {len(studies)}")

sess = requests.Session()
sess.auth = (USER, KEY)
BASE = "https://www.kaggle.com/api/v1/competitions/data/download"

def fetch(path):
    for attempt in range(4):
        try:
            r = sess.get(f"{BASE}/{COMP}/{path}", timeout=90)
            if r.status_code == 200 and r.content[:32]:
                return r.content
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return None

TAGS = ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "Laterality",
        "SeriesDescription", "SliceThickness", "SpacingBetweenSlices", "RepetitionTime",
        "EchoTime", "Rows", "Columns", "PatientSex", "SoftwareVersions",
        "MRAcquisitionType", "ScanOptions", "PatientID"]

os.makedirs(f"{WORK}/arrays", exist_ok=True)
headers, failed, done = [], [], 0
t0 = time.time()

for study in studies:
    out_npy = f"{WORK}/arrays/{study}.npy"
    if os.path.exists(out_npy):
        done += 1
        continue
    ser = chosen[study]
    paths = m[(m.study == study) & (m.series == ser)].path.tolist()

    with cf.ThreadPoolExecutor(16) as ex:
        blobs = list(ex.map(fetch, paths))

    slices = []
    for b in blobs:
        if b is None:
            continue
        try:
            d = pydicom.dcmread(io.BytesIO(b))
            slices.append((d, float(getattr(d, "InstanceNumber", 0) or 0)))
        except Exception:
            continue
    if len(slices) < 8:
        failed.append(study)
        continue

    slices.sort(key=lambda t: t[1])
    d0 = slices[0][0]
    rec = {"StudyInstanceUID": study, "SeriesInstanceUID": ser, "n_slices": len(slices)}
    for t in TAGS:
        v = getattr(d0, t, None)
        rec[t] = str(v) if v is not None else None
    ps = getattr(d0, "PixelSpacing", None)
    rec["pixel_spacing"] = float(ps[0]) if ps else None
    headers.append(rec)

    # Sample N_SLICES evenly through the stack so every study is the same shape.
    idx = np.linspace(0, len(slices) - 1, N_SLICES).round().astype(int)
    vol = np.zeros((N_SLICES, SIZE, SIZE), dtype=np.float16)
    for j, i in enumerate(idx):
        a = slices[i][0].pixel_array.astype(np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        vol[j] = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                            .resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float16) / 255.0
    np.save(out_npy, vol)
    done += 1
    if done % 25 == 0:
        el = time.time() - t0
        print(f"{done}/{len(studies)} studies  {el/60:.1f} min  "
              f"{el/max(done,1):.1f} s/study", flush=True)

pd.DataFrame(headers).to_parquet(f"{WORK}/dicom_headers.parquet", index=False)
print(f"\ningested {done} studies, {len(failed)} failed")
print(f"headers written for {len(headers)} studies")
if headers:
    h = pd.DataFrame(headers)
    print("\nvendor mix:"); print(h.Manufacturer.value_counts().to_string())
    print("\nlaterality present:", h.Laterality.replace("", None).notna().mean().round(3))
    print("field strength:"); print(h.MagneticFieldStrength.value_counts().to_string())
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=700)
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--scope", default="kaggle")
    ap.add_argument("--manifest", default="/tmp/rsnaknee/manifest.csv")
    ap.add_argument("--series", default="/tmp/rsnaknee/train_series.csv")
    ap.add_argument("--train", default="/tmp/rsnaknee/train.csv")
    ap.add_argument("--confidence", default="/tmp/rsnarun/label_confidence.csv")
    ap.add_argument("--weak",
                    default="/tmp/rsnallm/flight0234_rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--timeout-min", type=int, default=240)
    args = ap.parse_args()

    wc = _dbx.workspace_client()
    print(f"workspace {wc.config.host}")
    print(f"user      {wc.current_user.me().user_name}")
    print(f"work dir  {_dbx.WORK}")

    if not args.skip_upload:
        # The ingest job needs the first two. train.py needs the rest, and staging them
        # here keeps the Volume complete after one command.
        uploads = [(args.manifest, "manifest.csv"), (args.series, "train_series.csv"),
                   (args.train, "train.csv"), (args.confidence, "label_confidence.csv"),
                   (args.weak, "flight_hybrid_labels.csv")]
        for local, name in uploads:
            if not pathlib.Path(local).exists():
                print(f"skipping {name}, {local} not found")
                continue
            print(f"uploading {local} -> {_dbx.WORK}/{name}")
            _dbx.upload_file(wc, local, f"{_dbx.WORK}/{name}")

    script = (
        f"import os\n"
        f"os.environ['KNEE_WORK'] = {_dbx.WORK!r}\n"
        f"os.environ['KNEE_N_STUDIES'] = {str(args.studies)!r}\n"
        f"os.environ['KNEE_N_SLICES'] = {str(args.slices)!r}\n"
        f"os.environ['KNEE_SIZE'] = {str(args.size)!r}\n"
        f"dbutils.widgets.text('KAGGLE_USERNAME', '')\n"
        f"dbutils.widgets.text('KAGGLE_KEY', '')\n" + JOB
    )

    sub = _dbx.submit(
        wc, script, name="knee_ingest_pilot.py",
        deps=["pydicom>=3.0", "pillow>=10.0", "requests>=2.31", "pyarrow>=15.0"],
        timeout_min=args.timeout_min,
        secret_env={"KAGGLE_USERNAME": f"{args.scope}/username",
                    "KAGGLE_KEY": f"{args.scope}/key"},
    )
    print(f"\nsubmitted run {sub['run_id']}")
    print(f"notebook     {sub['notebook_path']}")
    r = _dbx.wait(wc, sub["run_id"], timeout_min=args.timeout_min)
    print(json.dumps(r.get("status", r.get("state", {})), indent=2)[:600])
    print("\n--- run output ---")
    print(_dbx.run_output(wc, sub["run_id"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
