"""
M6 — Class Weighting Ablation

Trains WeightedGCN twice with identical setup:
  1. With class weighting (BCEWithLogitsLoss with pos_weight)
  2. Without class weighting (standard BCEWithLogitsLoss)

Saves: class_weight_results.json
"""

import os, sys, json, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

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

labels = [d.y.item() for d in data_list]
train_idx, test_idx = train_test_split(range(len(data_list)), test_size=0.2, stratify=labels, random_state=42)
train_data = [data_list[i] for i in train_idx]
test_data = [data_list[i] for i in test_idx]
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

n_pos = sum(labels)
n_neg = len(labels) - n_pos
pos_weight_val = n_neg / max(n_pos, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Class distribution: {n_pos} positive ({100*n_pos/len(labels):.1f}%), {n_neg} negative ({100*n_neg/len(labels):.1f}%)")
print(f"pos_weight: {pos_weight_val:.4f}")

EMBED_DIM = 32

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


def train_and_evaluate(use_weighting, epochs=20):
    config_name = "Weighted" if use_weighting else "Unweighted"
    print(f"\n{'='*60}")
    print(f"Training: {config_name} BCEWithLogitsLoss")
    print(f"{'='*60}")

    model = WeightedGCN(NUM_NODES, FEAT_DIM, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    if use_weighting:
        pw = torch.tensor([pos_weight_val], dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits.view(-1), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_probs, all_labels_eval = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).view(-1)
                all_probs.extend(probs.cpu().numpy())
                all_labels_eval.extend(batch.y.view(-1).cpu().numpy())

        all_probs_np = np.array(all_probs)
        all_labels_np = np.array(all_labels_eval)

        fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = thresholds[best_idx]
        preds = (all_probs_np > threshold).astype(int)

        auc = roc_auc_score(all_labels_np, all_probs_np)
        acc = accuracy_score(all_labels_np, preds)
        prec = precision_score(all_labels_np, preds, zero_division=0)
        rec = recall_score(all_labels_np, preds, zero_division=0)
        f1 = f1_score(all_labels_np, preds, zero_division=0)
        cm = confusion_matrix(all_labels_np, preds)

        if auc > best_auc:
            best_auc = auc
            best_metrics = {
                "config": config_name,
                "pos_weight": round(float(pos_weight_val), 4) if use_weighting else None,
                "best_epoch": epoch + 1,
                "auc": round(float(auc), 4),
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "threshold": round(float(threshold), 4),
                "confusion_matrix": cm.tolist(),
            }

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}  AUC={auc:.4f}  F1={f1:.4f}")

    print(f"  Best AUC: {best_metrics['auc']}")
    return best_metrics


# Run both
weighted_result = train_and_evaluate(use_weighting=True)
unweighted_result = train_and_evaluate(use_weighting=False)

# Compute differences
f1_diff = weighted_result["f1"] - unweighted_result["f1"]
auc_diff = weighted_result["auc"] - unweighted_result["auc"]

output = {
    "experiment": "Class Weighting Ablation (M6)",
    "class_distribution": {
        "positive": n_pos,
        "negative": n_neg,
        "positive_pct": round(100 * n_pos / len(labels), 2),
        "negative_pct": round(100 * n_neg / len(labels), 2),
        "pos_weight": round(pos_weight_val, 4),
    },
    "weighted": weighted_result,
    "unweighted": unweighted_result,
    "comparison": {
        "f1_difference": round(f1_diff, 4),
        "auc_difference": round(auc_diff, 4),
        "recommendation": "weighted" if f1_diff > 0 else "unweighted",
        "summary": f"Class weighting {'improved' if f1_diff > 0 else 'reduced'} F1 by {abs(f1_diff)*100:.1f} points "
                   f"and {'improved' if auc_diff > 0 else 'reduced'} AUC by {abs(auc_diff)*100:.1f} points."
    }
}

with open(os.path.join(OUT_DIR, "class_weight_results.json"), "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print("CLASS WEIGHTING ABLATION SUMMARY")
print(f"{'='*60}")
print(f"{'Metric':<15} {'Weighted':>12} {'Unweighted':>12} {'Diff':>10}")
print(f"{'-'*50}")
for m in ["auc", "accuracy", "precision", "recall", "f1"]:
    w = weighted_result[m]
    u = unweighted_result[m]
    print(f"{m.upper():<15} {w:>12.4f} {u:>12.4f} {w-u:>+10.4f}")
print(f"{'='*60}")
print(f"\n{output['comparison']['summary']}")
print("Results saved to class_weight_results.json")
