"""Write measured OOF (and optionally LB) scores back into versions.yaml.

versions.yaml is the ledger: every attempt is a row, and `oof_auc` / `lb_auc` on that row
are results rather than settings. Filling them in by hand is how a ledger drifts from the
runs it claims to record, so this copies them from results.json instead.

    python record_results.py                                  # OOF from results.json
    python record_results.py --lb v7_xgb_ident_bag=0.97012     # add a leaderboard score
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/tmp/s6e8work/results.json")
    ap.add_argument("--versions-file", default=str(HERE.parent / "versions.yaml"))
    ap.add_argument("--lb", action="append", default=[], metavar="ID=SCORE",
                    help="record a measured leaderboard score; repeatable")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text()) if Path(args.results).exists() else {}
    oof = {vid: r["oof_auc"] for vid, r in results.items() if r.get("oof_auc") is not None}
    lb = {}
    for pair in args.lb:
        vid, _, score = pair.partition("=")
        lb[vid.strip()] = float(score)

    path = Path(args.versions_file)
    lines = path.read_text().splitlines(keepends=True)
    current: str | None = None
    changed: list[str] = []

    for i, line in enumerate(lines):
        m = re.match(r"^\s*-\s*id:\s*(\S+)\s*$", line)
        if m:
            current = m.group(1)
            continue
        if current is None:
            continue
        for field, table in (("oof_auc", oof), ("lb_auc", lb)):
            m2 = re.match(rf"^(\s*){field}:\s*(\S+)\s*$", line)
            if m2 and current in table:
                value = f"{table[current]:.6f}"
                if m2.group(2) != value:
                    lines[i] = f"{m2.group(1)}{field}: {value}\n"
                    changed.append(f"{current}.{field} {m2.group(2)} -> {value}")

    path.write_text("".join(lines))
    for c in changed:
        print(c)
    print(f"{len(changed)} field(s) updated in {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
