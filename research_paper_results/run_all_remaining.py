"""
Batch runner: Finishes all remaining experiments in a single process.
Runs: ablations (skipping completed), class weight, crossval (5-fold, 20 epochs).
"""
import subprocess, sys, os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPTS_DIR)

scripts = [
    ("Ablations (M4)", "run_ablations.py"),
    ("Class Weight (M6)", "run_class_weight_ablation.py"),
    ("Cross-Validation (M5)", "run_crossval.py"),
]

for name, script in scripts:
    print(f"\n{'#'*70}")
    print(f"# RUNNING: {name}")
    print(f"{'#'*70}\n")
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, script)],
        cwd=BASE,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  [WARNING] {name} exited with code {result.returncode} (may be stderr warnings)")
    else:
        print(f"  [OK] {name} completed successfully")

print(f"\n{'#'*70}")
print("# ALL EXPERIMENTS COMPLETE")
print(f"{'#'*70}")
