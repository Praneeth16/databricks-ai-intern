#!/usr/bin/env python3
"""Enumerate every file in the competition into a local manifest.

The Kaggle file-listing endpoint caps a page at 200 rows regardless of the
--page-size you ask for, so the ~700k DICOM paths take ~3.5k calls. Selective
download needs exact paths, so there is no way around it. Resumable: the last
page token is checkpointed after every page, so an interrupted run continues
instead of restarting.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time

COMP = "rsna-knee-abnormality-detection"
TOKEN_LINE = "Next Page Token = "


def fetch_page(token: str | None, retries: int = 5) -> tuple[list[str], str | None]:
    cmd = ["kaggle", "competitions", "files", COMP, "--page-size", "200", "--csv"]
    if token:
        cmd += ["--page-token", token]
    for attempt in range(retries):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            break
        time.sleep(2**attempt)
    else:
        raise RuntimeError(f"page failed after {retries} tries: {proc.stderr[:400]}")

    rows, nxt = [], None
    for line in proc.stdout.splitlines():
        if line.startswith(TOKEN_LINE):
            nxt = line[len(TOKEN_LINE) :].strip() or None
        elif line.strip() and not line.startswith("name,"):
            rows.append(line)
    return rows, nxt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/rsnaknee/manifest.csv")
    ap.add_argument("--state", default="/tmp/rsnaknee/manifest.state.json")
    ap.add_argument("--max-pages", type=int, default=100_000)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    state_path = pathlib.Path(args.state)
    out.parent.mkdir(parents=True, exist_ok=True)

    token, pages, seen = None, 0, 0
    if state_path.exists():
        st = json.loads(state_path.read_text())
        token, pages, seen = st.get("token"), st.get("pages", 0), st.get("rows", 0)
        print(f"resuming after {pages} pages / {seen} rows", flush=True)

    mode = "a" if seen else "w"
    with out.open(mode, newline="") as fh:
        w = csv.writer(fh)
        if not seen:
            w.writerow(["path", "size", "created"])
        while pages < args.max_pages:
            rows, token = fetch_page(token)
            for r in rows:
                # paths never contain a comma; size and created are the last two fields
                parts = r.rsplit(",", 2)
                if len(parts) == 3:
                    w.writerow(parts)
                    seen += 1
            pages += 1
            fh.flush()
            state_path.write_text(json.dumps({"token": token, "pages": pages, "rows": seen}))
            if pages % 25 == 0:
                print(f"pages={pages} rows={seen}", flush=True)
            if not token:
                print(f"done: pages={pages} rows={seen}", flush=True)
                return 0
    print(f"stopped at max-pages: pages={pages} rows={seen} token_pending={bool(token)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
