#!/usr/bin/env python3
"""Download and preprocess the pilot studies locally, then upload the arrays.

Why this exists rather than doing the download on Databricks. Kaggle returns 401 for
every authenticated API call made from Databricks serverless compute, using the same key
and the same code that succeeds from a laptop. Verified with a diagnostic run: the
credentials load correctly and Databricks even redacts the username in the job output,
which confirms the secret value is right, and `competitions_list`, `competition_list_files`
and `competition_download_file` all return 401. The same call from a laptop returns the
file. So Kaggle appears to reject API authentication from that network, and no change on
our side fixes it.

The split that does work. Download and decode here, then upload only the preprocessed
arrays. Each study becomes a 16 by 256 by 256 float16 array, which is about 2 MB against
21 MB of DICOM, so the upload is small and the GPU training still runs on the workspace
where it belongs.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _dbx  # noqa: E402

COMP = "rsna-knee-abnormality-detection"
TAGS = ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "Laterality",
        "SeriesDescription", "SliceThickness", "SpacingBetweenSlices", "RepetitionTime",
        "EchoTime", "Rows", "Columns", "PatientSex", "SoftwareVersions",
        "MRAcquisitionType", "ScanOptions", "PatientID"]


def choose_series(manifest: pd.DataFrame, series_meta: pd.DataFrame, min_slices: int
                  ) -> tuple[dict[str, str], pd.DataFrame]:
    m = manifest[manifest.path.str.startswith("train_series/")].copy()
    parts = m.path.str.split("/", expand=True)
    m["study"], m["series"] = parts[1], parts[2]
    sm = series_meta.set_index("SeriesInstanceUID")
    m = m.join(sm[["Anatomical_Plane", "Fluid_Sensitive"]], on="series")
    # Sagittal fluid sensitive first. That is the plane the meniscus and cruciate
    # findings are read on, and it is the one the model is trained for.
    m["rank"] = (m.Anatomical_Plane.eq("Sagittal") * 2 + m.Fluid_Sensitive.fillna(0)).astype(int)
    sizes = m.groupby("series").size()
    chosen = {}
    for study, g in m.groupby("study"):
        best = g.sort_values("rank", ascending=False).series.iloc[0]
        if sizes.get(best, 0) >= min_slices:
            chosen[study] = best
    return chosen, m


def build_volume(folder: str, n_slices: int, size: int) -> tuple[np.ndarray, dict | None]:
    slices = []
    for f in sorted(os.listdir(folder)):
        try:
            d = pydicom.dcmread(os.path.join(folder, f))
            slices.append((float(getattr(d, "InstanceNumber", 0) or 0), d))
        except Exception:
            continue
    if len(slices) < 8:
        return np.zeros((0,)), None
    slices.sort(key=lambda t: t[0])
    d0 = slices[0][1]
    head = {t: (str(getattr(d0, t)) if getattr(d0, t, None) is not None else None) for t in TAGS}
    ps = getattr(d0, "PixelSpacing", None)
    head["pixel_spacing"] = float(ps[0]) if ps else None
    head["n_slices"] = len(slices)

    idx = np.linspace(0, len(slices) - 1, n_slices).round().astype(int)
    vol = np.zeros((n_slices, size, size), dtype=np.float16)
    for j, i in enumerate(idx):
        a = slices[i][1].pixel_array.astype(np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        vol[j] = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                            .resize((size, size), Image.BILINEAR), dtype=np.float16) / 255.0
    return vol, head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=700)
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest", default="/tmp/rsnaknee/manifest_snapshot.csv")
    ap.add_argument("--series", default="/tmp/rsnaknee/train_series.csv")
    ap.add_argument("--out", default="/tmp/rsnaknee/arrays")
    ap.add_argument("--upload", action="store_true", help="push arrays to the UC Volume")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    series_meta = pd.read_csv(args.series)
    chosen, m = choose_series(manifest, series_meta, 8)
    studies = sorted(chosen)[:args.studies]
    print(f"manifest rows {len(manifest):,}  candidate studies {len(chosen):,}")
    print(f"ingesting {len(studies)} studies at {args.slices}x{args.size}x{args.size}")

    def grab(path: str, dest: str) -> bool:
        for attempt in range(4):
            try:
                api.competition_download_file(COMP, path, path=dest, force=True, quiet=True)
                return True
            except Exception:
                time.sleep(2 ** attempt)
        return False

    headers, failed, done, t0 = [], [], 0, time.time()
    hdr_path = out.parent / "dicom_headers.parquet"
    if hdr_path.exists():
        headers = pd.read_parquet(hdr_path).to_dict("records")

    for study in studies:
        target = out / f"{study}.npy"
        if target.exists():
            done += 1
            continue
        tmp = tempfile.mkdtemp(prefix="knee_")
        try:
            paths = m[(m.study == study) & (m.series == chosen[study])].path.tolist()
            with cf.ThreadPoolExecutor(args.workers) as ex:
                list(ex.map(lambda p: grab(p, tmp), paths))
            vol, head = build_volume(tmp, args.slices, args.size)
            if head is None:
                failed.append(study)
                print(f"  {study[-10:]} failed, got {len(os.listdir(tmp))} of {len(paths)} files")
                if done == 0 and len(failed) >= 3:
                    raise SystemExit("first three studies all failed, stopping")
                continue
            np.save(target, vol)
            head["StudyInstanceUID"] = study
            head["SeriesInstanceUID"] = chosen[study]
            headers.append(head)
            done += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        if done % 20 == 0:
            el = time.time() - t0
            pd.DataFrame(headers).to_parquet(hdr_path, index=False)
            print(f"{done}/{len(studies)}  {el/60:.1f} min  {el/max(done,1):.1f} s/study  "
                  f"eta {(len(studies)-done)*el/max(done,1)/60:.0f} min", flush=True)

    pd.DataFrame(headers).to_parquet(hdr_path, index=False)
    print(f"\ningested {done}, failed {len(failed)}")
    if headers:
        h = pd.DataFrame(headers)
        print("\nvendor mix:")
        print(h.Manufacturer.value_counts().head(6).to_string())
        lat = h.Laterality.replace("", None)
        print(f"\nlaterality recorded in the DICOM header: {lat.notna().mean():.1%}")
        print("field strength:")
        print(h.MagneticFieldStrength.value_counts().to_string())

    if args.upload:
        wc = _dbx.workspace_client()
        files = sorted(out.glob("*.npy"))
        print(f"\nuploading {len(files)} arrays to {_dbx.WORK}/arrays")

        def push(f: pathlib.Path) -> bool:
            for attempt in range(3):
                try:
                    _dbx.upload_file(wc, f, f"{_dbx.WORK}/arrays/{f.name}")
                    return True
                except Exception:
                    time.sleep(2 ** attempt)
            return False

        with cf.ThreadPoolExecutor(8) as ex:
            ok = sum(ex.map(push, files))
        print(f"uploaded {ok} of {len(files)}")
        if hdr_path.exists():
            _dbx.upload_file(wc, hdr_path, f"{_dbx.WORK}/dicom_headers.parquet")
            print("uploaded dicom_headers.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
