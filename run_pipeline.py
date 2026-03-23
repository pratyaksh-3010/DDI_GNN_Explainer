"""
Full Pipeline Runner — Run this after fixing all scripts.

Executes:
  1. nodes_edges.py     → layer3_nodes.csv, layer3_edges_weighted.csv, layer3_training_cases.csv
  2. build_drug_lookup.py → structure_name_lookup.csv
  3. enrich_nodes.py     → layer3_nodes_enriched.csv
  4. train_gnn.py        → weighted_gcn_model.pth, model_config.json
  5. Copy files to STREAMLIT_FRONTEND/DATA/
"""

import subprocess
import sys
import shutil
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def run_step(name, script_path):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"Script: {script_path}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE,
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\n❌ FAILED: {name}")
        sys.exit(1)
    else:
        print(f"\n✅ DONE: {name}")


# Step 1: Feature engineering
run_step(
    "Layer 3 — Feature Engineering",
    os.path.join(BASE, "LAYER _3_CODE", "nodes_edges.py")
)

# Step 2: Drug name lookup
run_step(
    "Layer 6 — Drug Name Lookup",
    os.path.join(BASE, "LAYER_6", "build_drug_lookup.py")
)

# Step 3: Enrich nodes
run_step(
    "Layer 6 — Enrich Nodes",
    os.path.join(BASE, "LAYER_6", "enrich_nodes.py")
)

# Step 4: Train GNN
run_step(
    "Layer 4 — Train GNN",
    os.path.join(BASE, "LAYER_4_CODE", "train_gnn.py")
)

# Step 5: Copy files to frontend
print(f"\n{'='*60}")
print("STEP: Copy files to frontend DATA directory")
print(f"{'='*60}\n")

data_dir = os.path.join(BASE, "STREAMLIT_FRONTEND", "DATA")
os.makedirs(data_dir, exist_ok=True)

files_to_copy = [
    "layer3_nodes_enriched.csv",
    "layer3_edges_weighted.csv",
    "layer3_feature_meta.json",
    "weighted_gcn_model.pth",
]

for fname in files_to_copy:
    src = os.path.join(BASE, fname)
    dst = os.path.join(data_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  ✅ Copied {fname}")
    else:
        print(f"  ⚠️  Missing: {fname}")

print(f"\n{'='*60}")
print("PIPELINE COMPLETE!")
print(f"{'='*60}")
print(f"\nTo run the app:")
print(f"  cd {os.path.join(BASE, 'STREAMLIT_FRONTEND')}")
print(f"  streamlit run app.py")
