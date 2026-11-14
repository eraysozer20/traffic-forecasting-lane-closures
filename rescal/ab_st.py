import subprocess
import sys

PYTHON_EXE = r"C:\Users\erays\anaconda3\python.exe"
SCRIPT = "main3.py"

cmd = [
    PYTHON_EXE,
    SCRIPT,
    "--dataset_choice", "Tyson",
    "--epochs", "20",
    "--model_variant", "wz",
    "--num_runs", "5",
    "--base_seed", "42",
    "--ablation_csv", "ablation_results.csv",
    "--ablation_modes",
    "baseline",
    "rescal_base_only",
    "rescal_base_plus_fold",
    "rescal_base_plus_fold_no_closure",
    "rescal_base_plus_fold_no_weekly",
]

print("Running command:")
print(" ".join(cmd))

result = subprocess.run(cmd)
sys.exit(result.returncode)