import pandas as pd
import json
import os

print("Analyzing Side Effects for V2 (Top 1000 Filter)")

# 1. Load the huge DDI edge file
ddi_path = r"DATA_AGGREGATION_CODE\Graph_Data\edges_drug_drug.csv"
print(f"Reading {ddi_path}...")
ddi = pd.read_csv(ddi_path, dtype=str)

# 2. Count side effect frequencies
print("Calculating side effect frequencies...")
se_counts = ddi['side_effect'].value_counts()

# 3. Get Top 1000
top_1000 = se_counts.head(1000)
top_1000_names = set(top_1000.index)

print(f"Total unique side effects originally: {len(se_counts)}")
print(f"Total isolated for V2: {len(top_1000_names)}")
print("\n--- Top 10 Most Frequent ---")
print(top_1000.head(10))
print("\n--- Bottom 10 of the Top 1000 ---")
print(top_1000.tail(10))

# 4. Save the Top 1000 summary for user verification
os.makedirs("V2", exist_ok=True)
summary_file = r"V2\top_1000_side_effects_summary.json"
top_1000.to_json(summary_file, orient="index")
print(f"\nSaved statistical summary to {summary_file}")

# 5. Create V2 Graph_Data directory (leave original alone)
v2_out = r"V2\Graph_Data"
os.makedirs(v2_out, exist_ok=True)

# 6. Filter the DDI edges (This is the only graph file that actually has SE strings directly in it. 
# Wait, edges_drug_side_effect.csv also has mono side effects)
print("\nFiltering DDI Edges...")
filtered_ddi = ddi[ddi['side_effect'].isin(top_1000_names)]
print(f"Filtered DDI edges from {len(ddi)} down to {len(filtered_ddi)}")
filtered_ddi.to_csv(os.path.join(v2_out, "edges_drug_drug.csv"), index=False)

# 7. Filter Mono Side Effects (if any)
mono_path = r"DATA_AGGREGATION_CODE\Graph_Data\edges_drug_side_effect.csv"
if os.path.exists(mono_path):
    print("Filtering Mono Side Effects...")
    mono = pd.read_csv(mono_path, dtype=str)
    filtered_mono = mono[mono['side_effect'].isin(top_1000_names)]
    print(f"Filtered Mono edges from {len(mono)} down to {len(filtered_mono)}")
    filtered_mono.to_csv(os.path.join(v2_out, "edges_drug_side_effect.csv"), index=False)

# 8. Filter Side Effect Nodes
se_nodes = pd.DataFrame({"side_effect": list(top_1000_names)})
se_nodes.to_csv(os.path.join(v2_out, "nodes_side_effect.csv"), index=False)

# 9. Copy over all the other graph components untouched
# (The user wanted to ensure we don't break anything, and the proteins/patients aren't inherently filtered by SE)
import shutil
other_files = [
    "edges_patient_drug.csv", "edges_drug_protein.csv", "edges_protein_protein.csv",
    "nodes_patient.csv", "nodes_drug.csv", "nodes_protein.csv"
]
print("Copying remaining graph structures...")
for f in other_files:
    src = os.path.join(r"DATA_AGGREGATION_CODE\Graph_Data", f)
    dst = os.path.join(v2_out, f)
    if os.path.exists(src):
        shutil.copy(src, dst)

print("\nV2 Top 1000 Filtering Complete! Original data remains untouched.")
