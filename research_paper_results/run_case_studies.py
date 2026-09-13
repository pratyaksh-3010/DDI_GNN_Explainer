"""
Case Studies — TP, TN, FP, FN Explainability Analysis

Selects 4-5 cases spanning different prediction outcomes:
  - True Positive (high-risk, correctly predicted)
  - True Negative (low-risk, correctly predicted)
  - False Positive (low-risk, predicted high)
  - False Negative (high-risk, predicted low)

Runs GNNExplainer + gradient attribution on each case.
Saves: case_studies.json, case_studies.txt
"""

import os, sys, io, json, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
nodes_df = pd.read_csv(os.path.join(BASE, "layer3_nodes.csv"))
edges_df = pd.read_csv(os.path.join(BASE, "layer3_edges_weighted.csv"))
cases_df = pd.read_csv(os.path.join(BASE, "layer3_training_cases.csv"))

with open(os.path.join(BASE, "layer3_feature_meta.json"), "r") as f:
    feat_meta = json.load(f)

FEAT_DIM = feat_meta["feat_dim"]
feat_names = feat_meta["feature_names"]
X = torch.tensor(nodes_df[feat_names].values, dtype=torch.float)
node_ids = list(nodes_df["structure_id"].astype(str))
NUM_NODES = len(node_ids)

# Drug name lookup
try:
    name_lookup = pd.read_csv(os.path.join(BASE, "structure_name_lookup.csv"), dtype=str)
    struct_to_name = dict(zip(name_lookup["structure_id"], name_lookup["drug_name"]))
except:
    struct_to_name = {}

def get_drug_name(global_idx):
    sid = node_ids[global_idx]
    return struct_to_name.get(sid, f"Drug_{sid}")

def beautify_feature(name):
    if name.startswith("atc_l1_"):
        letter = name.replace("atc_l1_", "")
        atc_names = {"A": "Alimentary/Metabolism", "B": "Blood", "C": "Cardiovascular",
                     "D": "Dermatologicals", "G": "Genitourinary", "H": "Hormones",
                     "J": "Anti-infectives", "L": "Antineoplastic", "M": "Musculoskeletal",
                     "N": "Nervous System", "P": "Antiparasitic", "R": "Respiratory",
                     "S": "Sensory Organs", "V": "Various"}
        return f"ATC: {atc_names.get(letter, letter)}"
    elif name.startswith("tc_"):
        return f"Target: {name[3:]}"
    elif name.startswith("at_"):
        return f"Action: {name[3:]}"
    elif name == "log_atc_count": return "Drug class diversity"
    elif name == "log_gene_count": return "Target gene count"
    elif name == "log_se_count": return "Known side effects"
    return name

# Build edges + subgraphs
edge_index_list, edge_weight_list = [], []
for _, row in edges_df.iterrows():
    i, j, w = int(row["source"]), int(row["target"]), float(row["weight"])
    edge_index_list.append([i, j]); edge_index_list.append([j, i])
    edge_weight_list.append(w); edge_weight_list.append(w)

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

adj_dict = {}
for i in range(edge_index.shape[1]):
    u, v, w = edge_index[0, i].item(), edge_index[1, i].item(), edge_weight[i].item()
    adj_dict.setdefault(u, []).append((v, w))

data_list = []
for _, row in cases_df.iterrows():
    node_list = ast.literal_eval(row["nodes"])
    label = row["label"]
    node_indices = [int(n) for n in node_list]
    node_set = set(node_indices)
    local_map = {old: i for i, old in enumerate(node_indices)}
    new_edges, new_weights = [], []
    for u in node_indices:
        if u in adj_dict:
            for v, w in adj_dict[u]:
                if v in node_set:
                    new_edges.append([local_map[u], local_map[v]])
                    new_weights.append(w)
    if len(new_edges) == 0:
        continue
    data = Data(
        x=X[node_indices],
        edge_index=torch.tensor(new_edges, dtype=torch.long).t().contiguous(),
        edge_weight=torch.tensor(new_weights, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.float),
        node_id=torch.tensor(node_indices, dtype=torch.long)
    )
    data_list.append(data)

# Train/test split
labels = [d.y.item() for d in data_list]
train_idx, test_idx = train_test_split(range(len(data_list)), test_size=0.2, stratify=labels, random_state=42)
test_data = [data_list[i] for i in test_idx]

# ============================================================
# LOAD MODEL
# ============================================================
EMBED_DIM = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WeightedGCN(nn.Module):
    def __init__(self, num_nodes, feat_dim, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        input_dim = embed_dim + feat_dim
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 1)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None, node_id=None):
        if edge_index is None:
            data = x; x = data.x; edge_index = data.edge_index
            edge_weight = data.edge_weight; batch = getattr(data, "batch", None); node_id = data.node_id
        if node_id is None:
            node_id = torch.arange(x.size(0), device=x.device)
        x = x.float(); edge_weight = edge_weight.float()
        embed = self.embedding(node_id)
        x = torch.cat([embed, x], dim=1)
        x = torch.relu(self.conv1(x, edge_index, edge_weight)); x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index, edge_weight)); x = self.dropout(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x)); x = self.dropout(x)
        return self.lin2(x)

checkpoint = torch.load(os.path.join(BASE, "weighted_gcn_model.pth"), map_location=device)
model = WeightedGCN(
    num_nodes=checkpoint["num_nodes"],
    feat_dim=checkpoint["feat_dim"],
    embed_dim=checkpoint.get("embed_dim", 32)
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded.")

# ============================================================
# FIND OPTIMAL THRESHOLD + CLASSIFY TEST SET
# ============================================================
best_threshold = checkpoint.get("best_threshold", 0.5)

test_predictions = []
for idx, graph in enumerate(test_data):
    graph_dev = graph.to(device)
    with torch.no_grad():
        logit = model(graph_dev).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
    true_label = int(graph.y.item())
    pred_label = 1 if prob > best_threshold else 0

    category = ""
    if true_label == 1 and pred_label == 1:
        category = "TP"
    elif true_label == 0 and pred_label == 0:
        category = "TN"
    elif true_label == 0 and pred_label == 1:
        category = "FP"
    elif true_label == 1 and pred_label == 0:
        category = "FN"

    test_predictions.append({
        "index": idx,
        "prob": prob,
        "true_label": true_label,
        "pred_label": pred_label,
        "category": category,
        "n_drugs": graph.x.shape[0],
        "n_edges": graph.edge_index.shape[1],
    })

# Select best cases (prefer more drugs for better visualization)
tp_cases = sorted([p for p in test_predictions if p["category"] == "TP"], key=lambda x: (-x["n_drugs"], -x["prob"]))
tn_cases = sorted([p for p in test_predictions if p["category"] == "TN"], key=lambda x: (-x["n_drugs"], x["prob"]))
fp_cases = sorted([p for p in test_predictions if p["category"] == "FP"], key=lambda x: (-x["n_drugs"], -x["prob"]))
fn_cases = sorted([p for p in test_predictions if p["category"] == "FN"], key=lambda x: (-x["n_drugs"], x["prob"]))

print(f"\nTest set distribution: TP={len(tp_cases)}, TN={len(tn_cases)}, FP={len(fp_cases)}, FN={len(fn_cases)}")

selected = []
if tp_cases: selected.append(("True Positive (High Risk, Correctly Predicted)", tp_cases[0]))
if tn_cases: selected.append(("True Negative (Low Risk, Correctly Predicted)", tn_cases[0]))
if fp_cases: selected.append(("False Positive (Low Risk, Predicted High)", fp_cases[0]))
if fn_cases: selected.append(("False Negative (High Risk, Predicted Low)", fn_cases[0]))
# Add a second TP if available for breadth
if len(tp_cases) > 1: selected.append(("True Positive Case 2", tp_cases[1]))

print(f"Selected {len(selected)} cases for analysis")

# ============================================================
# EXPLAINABILITY ANALYSIS
# ============================================================
def analyze_case(graph, case_info, title):
    graph = graph.to(device)
    result = {
        "title": title,
        "category": case_info["category"],
        "probability": round(case_info["prob"], 4),
        "true_label": case_info["true_label"],
        "predicted_label": case_info["pred_label"],
        "n_drugs": case_info["n_drugs"],
        "n_edges": case_info["n_edges"],
    }

    # Drug names
    drug_names = []
    for i in range(graph.x.shape[0]):
        global_id = graph.node_id[i].item()
        drug_names.append(get_drug_name(global_id))
    result["drugs"] = drug_names

    # Gradient attribution
    x_input = graph.x.clone().detach().requires_grad_(True)
    logit = model(
        x_input,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        batch=torch.zeros(x_input.size(0), dtype=torch.long).to(device),
        node_id=graph.node_id
    )
    model.zero_grad()
    logit.backward()
    grads = x_input.grad.detach().cpu().numpy()
    x_vals = x_input.detach().cpu().numpy()
    attr = np.abs(grads * x_vals)
    drug_scores = attr.sum(axis=1)
    drug_scores_norm = drug_scores / (drug_scores.sum() + 1e-8)

    # Per-drug importance with feature breakdown
    drug_analysis = []
    ranked = np.argsort(drug_scores_norm)[::-1]
    for rank, idx in enumerate(ranked):
        drug_info = {
            "rank": rank + 1,
            "name": drug_names[idx],
            "importance": round(float(drug_scores_norm[idx]), 4),
            "top_features": []
        }
        drug_attr = attr[idx]
        top_feats = np.argsort(drug_attr)[::-1][:5]
        for fi in top_feats:
            if drug_attr[fi] > 0.001:
                drug_info["top_features"].append({
                    "feature": beautify_feature(feat_names[fi]),
                    "raw_name": feat_names[fi],
                    "score": round(float(drug_attr[fi]), 4),
                })
        drug_analysis.append(drug_info)

    result["drug_importance"] = drug_analysis

    # GNNExplainer
    try:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='binary_classification',
                task_level='graph',
                return_type='raw'
            )
        )
        explanation = explainer(
            graph.x,
            graph.edge_index,
            edge_weight=graph.edge_weight,
            batch=torch.zeros(graph.x.size(0), dtype=torch.long).to(device),
            node_id=graph.node_id
        )

        node_importance = explanation.node_mask.sum(dim=1).detach().cpu().numpy()
        node_importance_norm = node_importance / (node_importance.sum() + 1e-8)

        gnnexplainer_drugs = []
        ranked_gnn = np.argsort(node_importance_norm)[::-1]
        for rank, idx in enumerate(ranked_gnn):
            gnnexplainer_drugs.append({
                "rank": rank + 1,
                "name": drug_names[idx],
                "importance": round(float(node_importance_norm[idx]), 4),
            })
        result["gnnexplainer_importance"] = gnnexplainer_drugs

        # Edge importance
        edge_importance = explanation.edge_mask.detach().cpu().numpy()
        edge_idx_np = graph.edge_index.cpu().numpy()
        edge_ranked = np.argsort(edge_importance)[::-1]
        top_edges = []
        seen_pairs = set()
        for ei in edge_ranked[:5]:
            s, t = edge_idx_np[0, ei], edge_idx_np[1, ei]
            pair = (min(s, t), max(s, t))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            top_edges.append({
                "drug1": drug_names[s],
                "drug2": drug_names[t],
                "importance": round(float(edge_importance[ei]), 4),
            })
        result["top_edge_importance"] = top_edges
    except Exception as e:
        result["gnnexplainer_error"] = str(e)

    return result

# ============================================================
# RUN ANALYSIS
# ============================================================
all_case_studies = []

for title, case_info in selected:
    print(f"\nAnalyzing: {title}")
    graph = test_data[case_info["index"]]
    case_result = analyze_case(graph, case_info, title)
    all_case_studies.append(case_result)
    print(f"  Prob={case_info['prob']:.4f}, Drugs={case_info['n_drugs']}, Category={case_info['category']}")

# Save JSON
with open(os.path.join(OUT_DIR, "case_studies.json"), "w") as f:
    json.dump(all_case_studies, f, indent=2)

# Save human-readable text
with open(os.path.join(OUT_DIR, "case_studies.txt"), "w", encoding="utf-8") as f:
    for case in all_case_studies:
        f.write("=" * 70 + "\n")
        f.write(f"CASE: {case['title']}\n")
        f.write(f"Category: {case['category']}\n")
        f.write(f"Predicted Probability: {case['probability']:.4f}\n")
        f.write(f"True Label: {'HIGH RISK' if case['true_label'] == 1 else 'LOW RISK'}\n")
        f.write(f"Predicted Label: {'HIGH RISK' if case['predicted_label'] == 1 else 'LOW RISK'}\n")
        f.write(f"Drugs: {case['n_drugs']} | Edges: {case['n_edges']}\n")
        f.write(f"Drug Names: {', '.join(case['drugs'])}\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- Gradient Attribution: Per-Drug Importance ---\n")
        for drug in case["drug_importance"]:
            f.write(f"  #{drug['rank']} {drug['name']}: {drug['importance']:.4f}\n")
            for feat in drug["top_features"]:
                f.write(f"      -> {feat['feature']}: {feat['score']:.4f}\n")
        f.write("\n")

        if "gnnexplainer_importance" in case:
            f.write("--- GNNExplainer: Per-Drug Importance ---\n")
            for drug in case["gnnexplainer_importance"]:
                f.write(f"  #{drug['rank']} {drug['name']}: {drug['importance']:.4f}\n")
            f.write("\n")

        if "top_edge_importance" in case:
            f.write("--- Top Edge Importance ---\n")
            for edge in case["top_edge_importance"]:
                f.write(f"  {edge['drug1']} <-> {edge['drug2']}: {edge['importance']:.4f}\n")
        f.write("\n\n")

print(f"\nCase studies saved to case_studies.json and case_studies.txt")
print(f"Total cases analyzed: {len(all_case_studies)}")
