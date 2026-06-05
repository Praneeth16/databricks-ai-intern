# Databricks notebook source
"""Submit v7 kfold+seeds to Kaggle."""
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
FILE = "/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops/submission_v7_kfold_seeds.csv"
MSG = ("v7 kfold+seeds: 5-fold StratifiedGroupKFold(by Race) + 3 seeds (42,7,555) "
       "per model, XGB+LGBM+CB frozen mid-range params, Optuna OOF weights "
       "xgb=0.583/lgbm=0.291/cb=0.126, OOF AUC=0.9296")

print(f"Submitting {FILE}...")
r = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", COMP, "-f", FILE, "-m", MSG],
    capture_output=True, text=True,
)
print("STDOUT:", r.stdout)
print("STDERR:", r.stderr)
print("rc:", r.returncode)

print("\nPolling for score...")
final_line = ""
for i in range(30):
    time.sleep(15)
    rs = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMP],
        capture_output=True, text=True,
    )
    out = rs.stdout
    for line in out.split("\n"):
        if "submission_v7_kfold_seeds.csv" in line:
            print(f"t={(i+1)*15}s: {line}")
            final_line = line
            # publicScore is last non-empty token before privateScore (often blank)
            parts = line.split()
            if parts and parts[-1].startswith("0."):
                break
    if final_line and final_line.split()[-1].startswith("0."):
        break

print("\nDONE")
try:
    dbutils.notebook.exit(f"rc={r.returncode}\nfinal_line={final_line}")
except Exception:
    pass
