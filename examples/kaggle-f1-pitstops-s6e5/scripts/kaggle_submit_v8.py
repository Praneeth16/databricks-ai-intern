# Databricks notebook source
"""Submit v8 seed_ensemble to Kaggle."""
import os, subprocess, sys, json, time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
TOK = "KGAT_d7a0018cbb8131f82d3dd175d2e4cdb6"
TP = os.path.expanduser("~/.kaggle/access_token")
with open(TP, "w") as f: f.write(TOK)
os.chmod(TP, 0o600)
os.environ["KAGGLE_API_TOKEN"] = TOK

COMP = "playground-series-s6e5"
FILE = "/Volumes/serverless_lakebase_praneeth_catalog/databricks_ai_intern_test/scratch/f1_pitstops/submission_v8_seed_ensemble.csv"
MSG = ("v8 seed-ensemble: Year=2025 holdout, Optuna 20 trials per model, "
       "5 seeds [42,7,555,2024,13] per model averaged. Optuna weights "
       "xgb=0.32/lgbm=0.55/cb=0.13, val_auc=0.9053, predicted_lb=0.9493")

print(f"Submitting {FILE}...")
r = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", COMP, "-f", FILE, "-m", MSG],
    capture_output=True, text=True,
)
print("STDOUT:", r.stdout); print("STDERR:", r.stderr); print("rc:", r.returncode)

print("\nPolling for score...")
final = ""
for i in range(30):
    time.sleep(15)
    rs = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP],
                        capture_output=True, text=True)
    for line in rs.stdout.split("\n"):
        if "submission_v8_seed_ensemble.csv" in line:
            print(f"t={(i+1)*15}s: {line}")
            final = line
            parts = line.split()
            if parts and parts[-1].startswith("0."):
                break
    if final and final.split()[-1].startswith("0."):
        break

print("\nDONE")
try:
    dbutils.notebook.exit(f"rc={r.returncode}\nfinal_line={final}")
except Exception:
    pass
