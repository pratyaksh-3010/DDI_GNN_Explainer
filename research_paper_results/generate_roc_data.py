"""
Generate ROC Curve coordinates for the WeightedGCN model.
Outputs FPR/TPR coordinates to roc_curve_data.json and prints them.
Uses the same 80/20 stratified split (random_state=42) as training.
"""

import os, sys, json, ast
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# ── paths (run from research_paper_results/) ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nodes_path   = os.path.join(ROOT, "layer3_nodes.csv")
edges_path   = os.path.join(ROOT, "layer3_edges_weighted.csv")
cases_path   = os.path.join(ROOT, "layer3_training_cases.csv")
meta_path    = os.path.join(ROOT, "layer3_feature_meta.json")
model_path   = os.path.join(ROOT, "weighted_gcn_model.pth")

# ── imports for model ──
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ============================================================
# MODEL DEFINITION  (must match training exactly)
# ============================================================
class WeightedGCN(torch.nn.Module):
    def __init__(self, num_nodes, feat_dim, embed_dim=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embed_dim)
        input_dim = embed_dim + feat_dim
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.dropout = torch.nn.Dropout(0.3)
        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None, node_id=None):
        if edge_index is None:
            data = x
            x = data.x
            edge_index = data.edge_index
            edge_weight = data.edge_weight
            batch = getattr(data, "batch", None)
            node_id = data.node_id

        if node_id is None:
            node_id = torch.arange(x.size(0), device=x.device)

        x = x.float()
        edge_weight = edge_weight.float()
        embed = self.embedding(node_id)
        x = torch.cat([embed, x], dim=1)

        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)

        x = self.conv2(x, edge_index, edge_weight)
        x = torch.relu(x)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)

        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        return x

# ============================================================
# LOAD DATA & BUILD GRAPHS
# ============================================================
print("Loading data...")
nodes_df = pd.read_csv(nodes_path)
edges_df = pd.read_csv(edges_path)
cases_df = pd.read_csv(cases_path)

with open(meta_path, "r") as f:
    feat_meta = json.load(f)

FEAT_DIM = feat_meta["feat_dim"]
feat_names = feat_meta["feature_names"]
X = torch.tensor(nodes_df[feat_names].values, dtype=torch.float)
NUM_NODES = len(nodes_df)

# Build adjacency
adj_dict = {}
for _, row in edges_df.iterrows():
    i, j = int(row["source"]), int(row["target"])
    w = float(row["weight"])
    adj_dict.setdefault(i, []).append((j, w))
    adj_dict.setdefault(j, []).append((i, w))

# Build subgraphs
print("Building subgraphs...")
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

print(f"Total graphs with edges: {len(data_list)}")

# ============================================================
# SAME TRAIN/TEST SPLIT AS TRAINING
# ============================================================
labels = [d.y.item() for d in data_list]
train_idx, test_idx = train_test_split(
    range(len(data_list)), test_size=0.2, stratify=labels, random_state=42
)
test_data = [data_list[i] for i in test_idx]
test_loader = DataLoader(test_data, batch_size=64)
print(f"Test set size: {len(test_data)}")

# ============================================================
# LOAD MODEL
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device)

model = WeightedGCN(
    num_nodes=checkpoint.get("num_nodes", NUM_NODES),
    feat_dim=checkpoint.get("feat_dim", FEAT_DIM),
    embed_dim=checkpoint.get("embed_dim", 32)
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded.")

# ============================================================
# EVALUATE & COMPUTE ROC
# ============================================================
all_probs, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).view(-1)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(batch.y.view(-1).cpu().numpy().tolist())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc_score = roc_auc_score(all_labels, all_probs)

# Downsample to ~50 points for the paper (evenly spaced + key points)
n_points = len(fpr)
if n_points > 50:
    step = max(1, n_points // 50)
    idx = list(range(0, n_points, step))
    if (n_points - 1) not in idx:
        idx.append(n_points - 1)
    fpr_sampled = fpr[idx]
    tpr_sampled = tpr[idx]
    thresholds_sampled = thresholds[idx]
else:
    fpr_sampled = fpr
    tpr_sampled = tpr
    thresholds_sampled = thresholds

# ============================================================
# SAVE & PRINT
# ============================================================
roc_data = {
    "auc": round(float(auc_score), 4),
    "test_samples": len(test_data),
    "n_positive": int(all_labels.sum()),
    "n_negative": int(len(all_labels) - all_labels.sum()),
    "optimal_threshold": round(float(checkpoint.get("best_threshold", 0.5)), 4),
    "coordinates_sampled": [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
        for f, t in zip(fpr_sampled, tpr_sampled)
    ],
    "coordinates_full": [
        {"fpr": round(float(f), 6), "tpr": round(float(t), 6)}
        for f, t in zip(fpr, tpr)
    ],
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roc_curve_data.json")
with open(out_path, "w") as f:
    json.dump(roc_data, f, indent=2)

print(f"\n{'='*60}")
print(f"ROC-AUC: {auc_score:.4f}")
print(f"Test samples: {len(test_data)} ({int(all_labels.sum())} pos, {int(len(all_labels) - all_labels.sum())} neg)")
print(f"\n--- ROC Coordinates (sampled ~50 points) ---")
print(f"{'FPR':>8}  {'TPR':>8}")
print(f"{'-'*8}  {'-'*8}")
for f, t in zip(fpr_sampled, tpr_sampled):
    print(f"{f:8.4f}  {t:8.4f}")

print(f"\nFull coordinates ({len(fpr)} points) saved to: {out_path}")
