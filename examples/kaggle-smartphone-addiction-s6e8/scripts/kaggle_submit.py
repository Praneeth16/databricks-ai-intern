"""Submit a CSV to the S6E8 leaderboard and poll until Kaggle scores it.

Uses the official `kaggle` client rather than hand-rolled REST calls. Two reasons: the
submission flow is a three-step upload against a generated transport, not stable URL
paths, and the client's auth chain tries `~/.kaggle/access_token` *before* the legacy
`kaggle.json` username/key pair — which matters here, because the legacy key on this
machine returns 401 while the access token works.

    pip install kaggle
    python kaggle_submit.py --file /tmp/s6e8work/submission_v4_xgb_full_fe.csv \
        --message "v4 xgb full-FE, OOF 0.968434"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

COMPETITION = "playground-series-s6e8"
EXPECTED_ROWS = 296_302


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def validate(csv_path: Path, sample_path: Path | None) -> None:
    """Fail before spending one of the day's 10 submissions on a malformed file."""
    import pandas as pd

    sub = pd.read_csv(csv_path)
    if list(sub.columns) != ["id", "addicted_label"]:
        raise SystemExit(f"bad columns {list(sub.columns)}, expected ['id', 'addicted_label']")
    if len(sub) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} rows, got {len(sub)}")
    if sub["addicted_label"].isna().any():
        raise SystemExit("submission contains nulls")
    if sample_path and sample_path.exists():
        sample = pd.read_csv(sample_path)
        if not sub["id"].equals(sample["id"]):
            raise SystemExit("id column does not match sample_submission (order matters)")
    print(f"validated {csv_path.name}: {len(sub)} rows, "
          f"range [{sub.addicted_label.min():.6g}, {sub.addicted_label.max():.6g}]")


def poll(api, competition: str, timeout_s: int = 900) -> str | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        newest = api.competition_submissions(competition)[0]
        status = str(getattr(newest, "status", "")).lower()
        score = getattr(newest, "public_score", None)
        if score not in (None, ""):
            print(f"SCORED  public={score}  ({getattr(newest, 'description', '')})")
            return str(score)
        if "error" in status:
            print(f"ERROR   {getattr(newest, 'error_description', 'unknown')}")
            return None
        print(f"  ...{status.split('.')[-1]}")
        time.sleep(10)
    print("timed out waiting for a score")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--message", required=True)
    ap.add_argument("--competition", default=COMPETITION)
    ap.add_argument("--sample", default="/tmp/s6e8/sample_submission.csv")
    ap.add_argument("--no-poll", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    if not args.skip_validate:
        validate(path, Path(args.sample) if args.sample else None)

    api = _api()
    resp = api.competition_submit(str(path), args.message, args.competition)
    print(f"submitted: {resp}")
    if not args.no_poll:
        poll(api, args.competition)
    return 0


if __name__ == "__main__":
    sys.exit(main())
