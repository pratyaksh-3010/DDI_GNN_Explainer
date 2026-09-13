"""
M4 — GNN Architecture Ablations

Trains 5 model variants + the original WeightedGCN on the SAME data split:
  1. WeightedGCN (original, baseline)
  2. StandardGCN (binary edges, weight=1.0)
  3. NoEmbedGCN (no learnable embeddings, input=65-dim only)
  4. GAT (GATConv replaces GCNConv)
  5. GIN (GINConv replaces GCNConv)
  6. GraphSAGE (SAGEConv replaces GCNConv)

Saves: ablation_results.json, ablation_comparison.png
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
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LOAD DATA (same pipeline as train_gnn.py)
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

# Build global edge index + weight
edge_index_list, edge_weight_list = [], []
for _, row in edges_df.iterrows():
    i, j, w = int(row["source"]), int(row["target"]), float(row["weight"])
    edge_index_list.append([i, j])
    edge_index_list.append([j, i])
    edge_weight_list.append(w)
    edge_weight_list.append(w)

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

# Adjacency dict
adj_dict = {}
for i in range(edge_index.shape[1]):
    u, v, w = edge_index[0, i].item(), edge_index[1, i].item(), edge_weight[i].item()
    adj_dict.setdefault(u, []).append((v, w))

# Build subgraphs
data_list = []
skipped = 0
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
        skipped += 1
        continue

    data = Data(
        x=X[node_indices],
        edge_index=torch.tensor(new_edges, dtype=torch.long).t().contiguous(),
        edge_weight=torch.tensor(new_weights, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.float),
        node_id=torch.tensor(node_indices, dtype=torch.long)
    )
    data_list.append(data)

print(f"Total subgraphs: {len(data_list)} (skipped {skipped})")

# Fixed train/test split
labels = [d.y.item() for d in data_list]
train_idx, test_idx = train_test_split(range(len(data_list)), test_size=0.2, stratify=labels, random_state=42)
train_data = [data_list[i] for i in train_idx]
test_data = [data_list[i] for i in test_idx]
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

n_pos = sum(labels)
n_neg = len(labels) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# MODEL DEFINITIONS
# ============================================================
EMBED_DIM = 32

class WeightedGCN(nn.Module):
    """Original model: GCNConv + learnable embeddings + edge weights."""
    def __init__(self, num_nodes, feat_dim, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        input_dim = embed_dim + feat_dim
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 1)
        self.use_embedding = True
        self.use_edge_weight = True

    def forward(self, x, edge_index=None, edge_weight=None, batch=None, node_id=None):
        if edge_index is None:
            data = x; x = data.x; edge_index = data.edge_index
            edge_weight = data.edge_weight; batch = getattr(data, "batch", None); node_id = data.node_id
        if node_id is None:
            node_id = torch.arange(x.size(0), device=x.device)
        x = x.float(); edge_weight = edge_weight.float()
        if self.use_embedding:
            embed = self.embedding(node_id)
            x = torch.cat([embed, x], dim=1)
        ew = edge_weight if self.use_edge_weight else None
        x = torch.relu(self.conv1(x, edge_index, ew)); x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index, ew)); x = self.dropout(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x)); x = self.dropout(x)
        return self.lin2(x)


class StandardGCN(nn.Module):
    """Ablation 1: Binary edges (weight=1.0), still has embeddings."""
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
        x = x.float()
        embed = self.embedding(node_id)
        x = torch.cat([embed, x], dim=1)
        # Ignore edge weights — use None (binary adjacency)
        x = torch.relu(self.conv1(x, edge_index, None)); x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index, None)); x = self.dropout(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x)); x = self.dropout(x)
        return self.lin2(x)


class NoEmbedGCN(nn.Module):
    """Ablation 2: No learnable embeddings, uses only 65-dim features."""
    def __init__(self, num_nodes, feat_dim, embed_dim=32):
        super().__init__()
        self.conv1 = GCNConv(feat_dim, 128)  # 65-dim input, no embedding
        self.conv2 = GCNConv(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 1)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None, node_id=None):
        if edge_index is None:
            data = x; x = data.x; edge_index = data.edge_index
            edge_weight = data.edge_weight; batch = getattr(data, "batch", None); node_id = data.node_id
        x = x.float(); edge_weight = edge_weight.float()
        x = torch.relu(self.conv1(x, edge_index, edge_weight)); x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index, edge_weight)); x = self.dropout(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x)); x = self.dropout(x)
        return self.lin2(x)


class GATModel(nn.Module):
    """Ablation 3: GAT baseline (GATConv replaces GCNConv)."""
    def __init__(self, num_nodes, feat_dim, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        input_dim = embed_dim + feat_dim
        self.conv1 = GATConv(input_dim, 32, heads=4, concat=True, dropout=0.3)  # 32*4=128
        self.conv2 = GATConv(128, 64, heads=1, concat=False, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 1)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None, node_id=None):
        if edge_index is None:
            data = x; x = data.x; edge_index = data.edge_index
            edge_weight = data.edge_weight; batch = getattr(data, "batch", None); node_id = data.node_id
        if node_id is None:
            node_id = torch.arange(x.size(0), device=x.device)
        x = x.float()
        embed = self.embedding(node_id)
        x = torch.cat([embed, x], dim=1)
        # GAT uses attention, not edge weights
        x = torch.relu(self.conv1(x, edge_index)); x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index)); x = self.dropout(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x)); x = self.dropout(x)
        return self.lin2(x)


class GINModel(nn.Module):
    """Ablation 4: GIN baseline (GINConv replaces GCNConv)."""
    def __init__(self, num_nodes, feat_dim, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        input_dim = embed_dim + feat_dim
        gin_nn1 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128))
        self.conv1 = GINConv(gin_nn1)
        gin_nn2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 64))
        self.conv2 = GINConv(gin_nn2)
        self.dropout = nn.Dropout(0.3)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 1)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None, node_id=None):
        if edge_index is None:
            data = x; x = data.x; edge_index = data.edge_index
            edge_weight = data.edge_weight; batch = getattr(data, "batch", None); node_id = data.node_id
        if node_id is None:
            node_id = torch.arange(x.size(0), device=x.device)
        x = x.float()
        embed = self.embedding(node_id)
        x = torch.cat([embed, x], dim=1)
        x = torch.relu(self.conv1(x, edge_index)); x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index)); x = self.dropout(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x)); x = self.dropout(x)
        return self.lin2(x)


class GraphSAGEModel(nn.Module):
    """Ablation 5: GraphSAGE baseline (SAGEConv replaces GCNConv)."""
    def __init__(self, num_nodes, feat_dim, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        input_dim = embed_dim + feat_dim
        self.conv1 = SAGEConv(input_dim, 128)
        self.conv2 = SAGEConv(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 1)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None, node_id=None):
        if edge_index is None:
            data = x; x = data.x; edge_index = data.edge_index
            edge_weight = data.edge_weight; batch = getattr(data, "batch", None); node_id = data.node_id
        if node_id is None:
            node_id = torch.arange(x.size(0), device=x.device)
        x = x.float()
        embed = self.embedding(node_id)
        x = torch.cat([embed, x], dim=1)
        x = torch.relu(self.conv1(x, edge_index)); x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index)); x = self.dropout(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x)); x = self.dropout(x)
        return self.lin2(x)


# ============================================================
# TRAINING + EVALUATION
# ============================================================
def train_and_evaluate(model_class, model_name, epochs=100):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    model = model_class(num_nodes=NUM_NODES, feat_dim=FEAT_DIM, embed_dim=EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    best_auc = 0
    best_metrics = {}
    best_model_state = None

    for epoch in range(epochs):
        # Train
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

        # Evaluate
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
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "model": model_name,
                "parameters": total_params,
                "best_epoch": epoch + 1,
                "auc": round(float(auc), 4),
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "threshold": round(float(threshold), 4),
                "confusion_matrix": cm.tolist(),
                "final_loss": round(float(avg_loss), 4),
            }

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}  AUC={auc:.4f}  F1={f1:.4f}")

    # Save model checkpoint
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    model_path = os.path.join(OUT_DIR, f"model_{safe_name}.pth")
    torch.save({
        "model_state_dict": best_model_state or model.state_dict(),
        "model_name": model_name,
        "num_nodes": NUM_NODES,
        "feat_dim": FEAT_DIM,
        "embed_dim": EMBED_DIM,
        "best_auc": best_auc,
        "best_threshold": best_metrics.get("threshold", 0.5),
    }, model_path)
    print(f"  Model saved to {model_path}")
    print(f"  Best AUC: {best_metrics['auc']:.4f} at epoch {best_metrics['best_epoch']}")
    return best_metrics


# ============================================================
# RUN ALL ABLATIONS (incremental save after each model)
# ============================================================
models_to_run = [
    (WeightedGCN, "WeightedGCN (Ours)"),
    (StandardGCN, "StandardGCN (Binary Edges)"),
    (NoEmbedGCN, "GCN (No Embeddings)"),
    (GATModel, "GAT"),
    (GINModel, "GIN"),
    (GraphSAGEModel, "GraphSAGE"),
]

results_path = os.path.join(OUT_DIR, "ablation_results.json")

# Load any previously saved results to skip completed models
all_results = []
completed_models = set()
if os.path.exists(results_path):
    try:
        with open(results_path, "r") as f:
            all_results = json.load(f)
        completed_models = {r["model"] for r in all_results}
        print(f"Loaded {len(all_results)} previously completed models: {completed_models}")
    except:
        all_results = []

for model_class, model_name in models_to_run:
    if model_name in completed_models:
        print(f"\n[SKIP] {model_name} already completed.")
        continue
    result = train_and_evaluate(model_class, model_name, epochs=100)
    all_results.append(result)
    # Save incrementally after each model
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  [SAVED] Incremental results saved ({len(all_results)} models done)")

print(f"\nAll results saved to {results_path}")

# ============================================================
# PLOT: Ablation Comparison (Grouped Bar Chart)
# ============================================================
print("Generating ablation comparison plot...")

metrics_to_plot = ["auc", "accuracy", "precision", "recall", "f1"]
metric_labels = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
model_names = [r["model"] for r in all_results]

fig, ax = plt.subplots(figsize=(14, 7))
n_models = len(model_names)
n_metrics = len(metrics_to_plot)
bar_width = 0.13
x = np.arange(n_models)

colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
    values = [r[metric] for r in all_results]
    offset = (i - n_metrics / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, values, bar_width, label=label, color=colors[i], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("M4: GNN Architecture Ablation Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
ax.legend(loc="lower right")
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ablation_comparison.png"), dpi=150)
plt.close(fig)
print("  [OK] Saved ablation_comparison.png")

# Print summary table
print(f"\n{'='*80}")
print(f"{'Model':<30} {'AUC':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Params':>10}")
print(f"{'='*80}")
for r in all_results:
    print(f"{r['model']:<30} {r['auc']:>8.4f} {r['accuracy']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['parameters']:>10,}")
print(f"{'='*80}")
print("\nAblation experiments complete!")
