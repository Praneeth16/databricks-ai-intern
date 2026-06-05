# Databricks notebook source
"""Submit v4.1 leaner to Kaggle from inside Databricks (where pip works)."""
import os, subprocess, sys, json, time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
KAGGLE_JSON = os.path.expanduser("~/.kaggle/kaggle.json")
with open(KAGGLE_JSON, "w") as f:
    json.dump({"username": "praneeth_paikray",
               "key": "KGAT_d7a0018cbb8131f82d3dd175d2e4cdb6"}, f)
os.chmod(KAGGLE_JSON, 0o600)

COMP = "playground-series-s6e5"
FILE = "/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops/submission_v4_1_leaner.csv"
MSG = "v4.1 Phase 2 leaner: vs_field deltas + position gaps + within-driver rollups, mcw=20 reg_lambda=5, 26 feats, val_auc=0.8988"

print(f"Submitting {FILE}...")
r = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", COMP, "-f", FILE, "-m", MSG],
    capture_output=True, text=True,
)
print("STDOUT:", r.stdout)
print("STDERR:", r.stderr)
print("rc:", r.returncode)

# Poll for score
print("\nPolling for score...")
for i in range(30):
    time.sleep(15)
    rs = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMP],
        capture_output=True, text=True,
    )
    out = rs.stdout
    print(f"--- t={(i+1)*15}s ---")
    print(out[:2000])
    if "submission_v4_1_leaner.csv" in out and ("complete" in out.lower() or "0." in out):
        # Look for the row with our file
        for line in out.split("\n"):
            if "submission_v4_1_leaner.csv" in line:
                print(f"FOUND: {line}")
        break

print("\nDONE")
