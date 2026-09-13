"""
Polypharmacy GNN — Model Feasibility & Dataset Compatibility Validation

Validates:
  1. All data files exist and are consistent
  2. Node/edge/feature dimensions match across files
  3. Trained model loads and produces valid predictions
  4. Full test-set metrics (AUC, Accuracy, Precision, Recall, F1)
"""

import os
import sys
import json
import ast
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score,
    recall_score, f1_score, confusion_matrix, accuracy_score
)

BASE = os.path.dirname(os.path.abspath(__file__))
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ PASS: {name}")
    else:
        FAIL += 1
        print(f"  ❌ FAIL: {name} — {detail}")
    return condition


# ============================================================
# 1. DATA FILE EXISTENCE
# ============================================================
print("\n" + "=" * 60)
print("SECTION 1: Data File Existence")
print("=" * 60)

required_files = {
    "layer3_nodes.csv": "Node features",
    "layer3_edges_weighted.csv": "Edge list with weights",
    "layer3_training_cases.csv": "Training cases",
    "layer3_feature_meta.json": "Feature metadata",
    "weighted_gcn_model.pth": "Trained GCN model",
    "layer3_nodes_enriched.csv": "Enriched nodes with drug names",
    "structure_name_lookup.csv": "Drug name lookup",
}

for fname, desc in required_files.items():
    path = os.path.join(BASE, fname)
    check(f"{fname} ({desc})", os.path.exists(path), "File not found")

# ============================================================
# 2. DATASET INTEGRITY
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2: Dataset Integrity")
print("=" * 60)

nodes_df = pd.read_csv(os.path.join(BASE, "layer3_nodes.csv"))
edges_df = pd.read_csv(os.path.join(BASE, "layer3_edges_weighted.csv"))
cases_df = pd.read_csv(os.path.join(BASE, "layer3_training_cases.csv"))

with open(os.path.join(BASE, "layer3_feature_meta.json"), "r") as f:
    feat_meta = json.load(f)

FEAT_DIM = feat_meta["feat_dim"]
feat_names = feat_meta["feature_names"]
NUM_NODES = len(nodes_df)

print(f"\n  Dataset Statistics:")
print(f"    Nodes (drugs):      {NUM_NODES}")
print(f"    Edges (interactions): {len(edges_df)}")
print(f"    Training cases:     {len(cases_df)}")
print(f"    Feature dimension:  {FEAT_DIM}")
print(f"    Feature names:      {len(feat_names)}")

check("Feature count matches metadata", len(feat_names) == FEAT_DIM,
      f"Expected {FEAT_DIM}, got {len(feat_names)}")

check("All feature columns exist in nodes_df",
      all(f in nodes_df.columns for f in feat_names),
      "Missing columns: " + str([f for f in feat_names if f not in nodes_df.columns]))

check("Edge sources in valid range",
      edges_df["source"].max() < NUM_NODES and edges_df["source"].min() >= 0,
      f"Range [{edges_df['source'].min()}, {edges_df['source'].max()}] for {NUM_NODES} nodes")

check("Edge targets in valid range",
      edges_df["target"].max() < NUM_NODES and edges_df["target"].min() >= 0,
      f"Range [{edges_df['target'].min()}, {edges_df['target'].max()}] for {NUM_NODES} nodes")

check("No self-loops in edges",
      (edges_df["source"] != edges_df["target"]).all(),
      "Self-loops found")

check("Edge weights are positive",
      (edges_df["weight"] > 0).all(),
      "Non-positive weights found")

# Class distribution
n_pos = cases_df["label"].sum()
n_neg = len(cases_df) - n_pos
print(f"\n  Class Distribution:")
print(f"    High risk (1): {n_pos} ({100*n_pos/len(cases_df):.1f}%)")
print(f"    Low risk  (0): {n_neg} ({100*n_neg/len(cases_df):.1f}%)")

check("Both classes present", n_pos > 0 and n_neg > 0,
      "Missing a class!")

# ============================================================
# 3. MODEL LOADING
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: Model Loading & Architecture")
print("=" * 60)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

checkpoint = torch.load(os.path.join(BASE, "weighted_gcn_model.pth"), map_location=device)

check("Checkpoint has model_state_dict", "model_state_dict" in checkpoint)
check("Checkpoint has num_nodes", "num_nodes" in checkpoint)
check("Checkpoint has feat_dim", "feat_dim" in checkpoint)

ckpt_nodes = checkpoint.get("num_nodes", 0)
ckpt_feat = checkpoint.get("feat_dim", 0)
ckpt_embed = checkpoint.get("embed_dim", 32)

check("num_nodes matches data",
      ckpt_nodes == NUM_NODES,
      f"Model: {ckpt_nodes}, Data: {NUM_NODES}")

check("feat_dim matches data",
      ckpt_feat == FEAT_DIM,
      f"Model: {ckpt_feat}, Data: {FEAT_DIM}")

model = WeightedGCN(
    num_nodes=ckpt_nodes,
    feat_dim=ckpt_feat,
    embed_dim=ckpt_embed
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Model Architecture:")
print(f"    Total parameters:     {total_params:,}")
print(f"    Trainable parameters: {trainable_params:,}")
print(f"    Embedding dim:        {ckpt_embed}")
print(f"    Best AUC (training):  {checkpoint.get('best_auc', 'N/A')}")
print(f"    Best threshold:       {checkpoint.get('best_threshold', 'N/A')}")

check("Model loaded successfully", True)

# ============================================================
# 4. INFERENCE VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: Inference Validation (Full Test Set)")
print("=" * 60)

X = torch.tensor(nodes_df[feat_names].values, dtype=torch.float)

# Build global edge structures
edge_index_list = []
edge_weight_list = []
for _, row in edges_df.iterrows():
    i = int(row["source"])
    j = int(row["target"])
    w = float(row["weight"])
    edge_index_list.append([i, j])
    edge_index_list.append([j, i])
    edge_weight_list.append(w)
    edge_weight_list.append(w)

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

adj_dict = {}
for i in range(edge_index.shape[1]):
    u = edge_index[0, i].item()
    v = edge_index[1, i].item()
    w = edge_weight[i].item()
    if u not in adj_dict:
        adj_dict[u] = []
    adj_dict[u].append((v, w))

# Build subgraphs
data_list = []
skipped = 0
for _, row in cases_df.iterrows():
    node_list = ast.literal_eval(row["nodes"])
    label = row["label"]
    node_indices = [int(n) for n in node_list]
    node_set = set(node_indices)
    local_map = {old: i for i, old in enumerate(node_indices)}

    new_edges = []
    new_weights = []
    for u in node_indices:
        if u in adj_dict:
            for v, w in adj_dict[u]:
                if v in node_set:
                    new_edges.append([local_map[u], local_map[v]])
                    new_weights.append(w)

    if len(new_edges) == 0:
        skipped += 1
        continue

    new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    new_edge_weight = torch.tensor(new_weights, dtype=torch.float)

    data = Data(
        x=X[node_indices],
        edge_index=new_edge_index,
        edge_weight=new_edge_weight,
        y=torch.tensor([label], dtype=torch.float),
        node_id=torch.tensor(node_indices, dtype=torch.long)
    )
    data_list.append(data)

print(f"  Subgraphs built: {len(data_list)} (skipped {skipped} with no edges)")

# Train/test split (same as training)
labels = [data.y.item() for data in data_list]
train_idx, test_idx = train_test_split(
    range(len(data_list)), test_size=0.2, stratify=labels, random_state=42
)

test_data = [data_list[i] for i in test_idx]
test_loader = DataLoader(test_data, batch_size=64)

# Run inference
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).view(-1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch.y.view(-1).cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

check("All predictions are valid probabilities",
      np.all((all_probs >= 0) & (all_probs <= 1)),
      f"Range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")

check("Predictions have variance",
      all_probs.std() > 0.01,
      f"Std: {all_probs.std():.6f} — model may be collapsed")

# Optimal threshold
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

preds = (all_probs > best_threshold).astype(int)

auc = roc_auc_score(all_labels, all_probs)
acc = accuracy_score(all_labels, preds)
prec = precision_score(all_labels, preds, zero_division=0)
rec = recall_score(all_labels, preds, zero_division=0)
f1 = f1_score(all_labels, preds, zero_division=0)
cm = confusion_matrix(all_labels, preds)

print(f"\n  Test Set Results ({len(test_data)} samples):")
print(f"    AUC:       {auc:.4f}")
print(f"    Accuracy:  {acc:.4f}")
print(f"    Precision: {prec:.4f}")
print(f"    Recall:    {rec:.4f}")
print(f"    F1 Score:  {f1:.4f}")
print(f"    Threshold: {best_threshold:.4f}")
print(f"    Confusion Matrix:")
print(f"      {cm}")

check("AUC > 0.5 (better than random)", auc > 0.5, f"AUC={auc:.4f}")
check("F1 > 0 (model produces both predictions)", f1 > 0, f"F1={f1:.4f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print(f"VALIDATION COMPLETE: {PASS} passed, {FAIL} failed")
print("=" * 60)

if FAIL == 0:
    print("\n🎉 All checks passed! Model is feasible and dataset is compatible.")
else:
    print(f"\n⚠️  {FAIL} check(s) failed. Review above for details.")

sys.exit(FAIL)
