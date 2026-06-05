# Databricks notebook source
"""Submit v5.4 blend3-full to Kaggle from inside Databricks."""
import os, subprocess, sys, json, time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
TOK = "KGAT_d7a0018cbb8131f82d3dd175d2e4cdb6"
TOK_PATH = os.path.expanduser("~/.kaggle/access_token")
with open(TOK_PATH, "w") as f:
    f.write(TOK)
os.chmod(TOK_PATH, 0o600)
os.environ["KAGGLE_API_TOKEN"] = TOK

COMP = "playground-series-s6e5"
FILE = "/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops/submission_v5_4_blend3_full.csv"
MSG = ("v5.4 blend3-full: XGB+LGBM+CB each Optuna 20 trials, "
       "Optuna weights xgb=0.468/lgbm=0.368/cb=0.164, 14 feats, "
       "val_auc=0.9054, predicted_lb=0.9494")

print(f"Submitting {FILE}...")
r = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", COMP, "-f", FILE, "-m", MSG],
    capture_output=True, text=True,
)
print("STDOUT:", r.stdout)
print("STDERR:", r.stderr)
print("rc:", r.returncode)

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
    if "submission_v5_4_blend3_full.csv" in out and ("complete" in out.lower() or "0." in out):
        for line in out.split("\n"):
            if "submission_v5_4_blend3_full.csv" in line:
                print(f"FOUND: {line}")
        break

print("\nDONE")
try:
    dbutils.notebook.exit("DONE rc=" + str(r.returncode))
except Exception:
    pass
