import json
import os

with open('ablation_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Find max values for bolding if needed, but simple output works too
readme_path = 'README.md'
with open(readme_path, 'r', encoding='utf-8') as f:
    readme = f.read()

table_lines = [
    "| Model | Description | Params | AUC | Accuracy | Precision | Recall | F1 |",
    "|---|---|---:|---:|---:|---:|---:|---:|"
]

descriptions = {
    "WeightedGCN (Ours)": "Full model: edge weights + embeddings",
    "StandardGCN (Binary Edges)": "Binary edges (weight=1.0)",
    "GCN (No Embeddings)": "No learnable embeddings (65-dim input only)",
    "GAT": "GATConv (4-head attention, no edge weights)",
    "GIN": "GINConv (MLP-based message passing)",
    "GraphSAGE": "SAGEConv (sampling-based)"
}

for r in results:
    model = r['model']
    desc = descriptions.get(model, "")
    params = f"{r['parameters']:,}"
    auc = f"{r['auc']:.4f}"
    acc = f"{r['accuracy']:.4f}"
    prec = f"{r['precision']:.4f}"
    rec = f"{r['recall']:.4f}"
    f1 = f"{r['f1']:.4f}"
    
    # Simple bolding for ours and best, but just writing the data is fine.
    if model == "WeightedGCN (Ours)":
        model_name = f"**{model}**"
    else:
        model_name = model
        
    row = f"| {model_name} | {desc} | {params} | {auc} | {acc} | {prec} | {rec} | {f1} |"
    table_lines.append(row)

new_table_str = "\n".join(table_lines)

# Find existing table block
start_idx = readme.find("| Model | Description | Params | AUC | Accuracy | Precision | Recall | F1 |")
end_idx = readme.find("### Key Ablation Insights")

if start_idx != -1 and end_idx != -1:
    new_readme = readme[:start_idx] + new_table_str + "\n\n" + readme[end_idx:]
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_readme)
    print("README updated successfully!")
else:
    print("Could not find table block in README")
