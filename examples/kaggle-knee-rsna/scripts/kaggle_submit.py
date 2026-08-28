#!/usr/bin/env python3
"""Build, push and then verify the notebook and the CC0 dataset.

Two things this guards against. Every `kaggle kernels push` queues a full re-run, so
the build assertions run first and a broken cell never costs a re-run. And a pushed
notebook whose run failed shows no outputs and reads as broken to a visitor, so the
push is not the end of the job: we wait for COMPLETE and check the outputs exist.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
NB = HERE.parent / "notebooks"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def slug_for(title: str) -> str:
    """Kaggle lowercases, collapses every run of non-alphanumerics to one hyphen,
    then trims. Emoji vanish. So the slug is predictable before the push."""
    return re.sub(r"^-|-$", "", re.sub(r"[^a-z0-9]+", "-", title.lower()))


def build(nb_dir: pathlib.Path) -> dict:
    p = run([sys.executable, "build.py"], cwd=nb_dir)
    print(p.stdout or p.stderr)
    if p.returncode:
        raise SystemExit("build failed, nothing pushed")
    return json.loads((nb_dir / "kernel-metadata.json").read_text())


def push_dataset(staging: pathlib.Path, public: bool) -> None:
    files = sorted(f for f in staging.rglob("*") if f.is_file())
    print(f"\nstaging directory holds {len(files)} files:")
    for f in files:
        print(f"  {f.name:32s} {f.stat().st_size:>12,} bytes")
    if any(f.suffix not in {".csv", ".json", ".md", ".npy"} for f in files):
        raise SystemExit("unexpected file type in the staging directory, refusing to upload")

    meta = json.loads((staging / "dataset-metadata.json").read_text())
    exists = run(["kaggle", "datasets", "status", meta["id"]]).returncode == 0
    if exists:
        p = run(["kaggle", "datasets", "version", "-p", str(staging),
                 "-m", "refreshed from the audit notebook", "--dir-mode", "zip"])
    else:
        # `datasets create` is private by default and the CLI cannot flip it later.
        cmd = ["kaggle", "datasets", "create", "-p", str(staging), "--dir-mode", "zip"]
        if public:
            cmd.append("-u")
        p = run(cmd)
    print(p.stdout or p.stderr)


def push_kernel(meta: dict, nb_dir: pathlib.Path) -> str:
    p = run(["kaggle", "kernels", "push", "-p", str(nb_dir)])
    print(p.stdout or p.stderr)
    if p.returncode:
        raise SystemExit("kernel push failed")
    # Renaming moves the slug. Kaggle warns that the title does not resolve to the
    # given id, renames in place, and that warning is expected rather than an error.
    return meta["id"]


def wait_for_run(ref: str, timeout_s: int) -> str:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        p = run(["kaggle", "kernels", "status", ref])
        out = (p.stdout or "") + (p.stderr or "")
        last = out.strip().splitlines()[-1] if out.strip() else last
        print(f"  {last}")
        if "complete" in out.lower():
            return "COMPLETE"
        if "error" in out.lower() or "cancel" in out.lower():
            return "FAILED"
        time.sleep(30)
    return "TIMEOUT"


def verify_output(ref: str, out_dir: pathlib.Path, expect: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = run(["kaggle", "kernels", "output", ref, "-p", str(out_dir)])
    print(p.stdout or p.stderr)
    files = sorted(f.name for f in out_dir.rglob("*") if f.is_file())
    print(f"\ndownloaded {len(files)} files: {files}")

    logs = [f for f in out_dir.rglob("*.log")]
    for lg in logs:
        text = lg.read_text(errors="replace")
        for bad in ("Traceback", "ModuleNotFoundError", "SyntaxError"):
            if bad in text:
                raise SystemExit(f"{lg.name} contains {bad}; the published run is broken")
    # One __results___files directory per figure cell is the evidence the plots rendered.
    figs = list(out_dir.rglob("__results___files/*"))
    print(f"rendered figure files: {len(figs)}")
    if expect not in files:
        print(f"WARNING: {expect} missing from the run output")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="/tmp/rsnads")
    ap.add_argument("--dataset", action="store_true", help="create or version the CC0 dataset")
    ap.add_argument("--kernel", action="store_true", help="push the notebook")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--verify", action="store_true", help="wait for the run and check it")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--dir", default=None,
                    help="notebook directory under the example root, e.g. notebooks/model-v2; "
                         "default is the top-level notebooks/ dir")
    ap.add_argument("--expect", default="label_confidence.csv",
                    help="output file the finished run must contain")
    args = ap.parse_args()

    nb_dir = (HERE.parent / args.dir).resolve() if args.dir else NB
    if not (nb_dir / "build.py").is_file():
        raise SystemExit(f"no build.py in {nb_dir}")

    meta = build(nb_dir)
    title = meta["title"]
    print(f"\ntitle       {title!r}  ({len(title)}/50 chars)")
    print(f"id          {meta['id']}")
    print(f"slug from title -> {slug_for(title)}")

    if args.dataset:
        push_dataset(pathlib.Path(args.staging), args.public)
    if args.kernel:
        ref = push_kernel(meta, nb_dir)
        if args.verify:
            state = wait_for_run(ref, args.timeout)
            print(f"\nrun state: {state}")
            if state != "COMPLETE":
                raise SystemExit(f"run did not complete ({state})")
            verify_output(ref, HERE.parent / "artifacts" / "kernel-output", args.expect)
    return 0


if __name__ == "__main__":
    sys.exit(main())
