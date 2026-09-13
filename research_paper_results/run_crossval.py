"""
M5 — 5-Fold Cross-Validation with Early Stopping + Training Curves

Uses StratifiedKFold(n_splits=5) on the WeightedGCN model.
Per fold: 80% train (split into 75% train / 25% val) + 20% test.
Early stopping on validation AUC with patience=15.
Records per-epoch training curves and AUC coordinates.

Saves: crossval_results.json, training_curves.png, auc_coordinates.json
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
from sklearn.model_selection import StratifiedKFold, train_test_split
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

# Build edges
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

# Build subgraphs
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

print(f"Total subgraphs: {len(data_list)}")
labels = np.array([d.y.item() for d in data_list])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# MODEL
# ============================================================
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


def evaluate_loader(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels_eval, total_loss = [], [], 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits.view(-1), batch.y)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).view(-1)
            all_probs.extend(probs.cpu().numpy())
            all_labels_eval.extend(batch.y.view(-1).cpu().numpy())

    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels_eval)
    avg_loss = total_loss / max(len(loader), 1)

    try:
        auc = roc_auc_score(all_labels_np, all_probs_np)
    except ValueError:
        auc = 0.5

    fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    threshold = thresholds[best_idx]
    preds = (all_probs_np > threshold).astype(int)

    return {
        "loss": float(avg_loss),
        "auc": float(auc),
        "accuracy": float(accuracy_score(all_labels_np, preds)),
        "precision": float(precision_score(all_labels_np, preds, zero_division=0)),
        "recall": float(recall_score(all_labels_np, preds, zero_division=0)),
        "f1": float(f1_score(all_labels_np, preds, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(all_labels_np, preds).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "all_probs": all_probs_np.tolist(),
        "all_labels": all_labels_np.tolist(),
    }


# ============================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================
MAX_EPOCHS = 20
PATIENCE = 10

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
all_training_curves = []

for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(range(len(data_list)), labels)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/5")
    print(f"{'='*60}")

    # Further split train_val into train + validation (75/25 of train_val)
    train_val_labels = labels[train_val_idx]
    train_sub_idx, val_sub_idx = train_test_split(
        range(len(train_val_idx)), test_size=0.25, stratify=train_val_labels, random_state=42
    )

    actual_train_idx = train_val_idx[train_sub_idx]
    actual_val_idx = train_val_idx[val_sub_idx]

    train_data = [data_list[i] for i in actual_train_idx]
    val_data = [data_list[i] for i in actual_val_idx]
    test_data = [data_list[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    n_pos = sum(1 for i in actual_train_idx if labels[i] == 1)
    n_neg = len(actual_train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)

    model = WeightedGCN(NUM_NODES, FEAT_DIM, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training with early stopping
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    epoch_curves = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}

    for epoch in range(MAX_EPOCHS):
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
        train_loss = total_loss / len(train_loader)

        # Evaluate train and val
        train_metrics = evaluate_loader(model, train_loader, criterion, device)
        val_metrics = evaluate_loader(model, val_loader, criterion, device)

        epoch_curves["train_loss"].append(train_loss)
        epoch_curves["val_loss"].append(val_metrics["loss"])
        epoch_curves["train_auc"].append(train_metrics["auc"])
        epoch_curves["val_auc"].append(val_metrics["auc"])

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}  "
                  f"Val AUC={val_metrics['auc']:.4f}  patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Load best model and evaluate on test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate_loader(model, test_loader, criterion, device)

    fold_result = {
        "fold": fold_idx + 1,
        "stopped_epoch": epoch + 1,
        "best_val_auc": round(float(best_val_auc), 4),
        "test_auc": round(float(test_metrics["auc"]), 4),
        "test_accuracy": round(float(test_metrics["accuracy"]), 4),
        "test_precision": round(float(test_metrics["precision"]), 4),
        "test_recall": round(float(test_metrics["recall"]), 4),
        "test_f1": round(float(test_metrics["f1"]), 4),
        "test_threshold": round(float(test_metrics["threshold"]), 4),
        "test_confusion_matrix": test_metrics["confusion_matrix"],
    }
    fold_results.append(fold_result)
    all_training_curves.append(epoch_curves)

    print(f"  Test AUC={test_metrics['auc']:.4f}  F1={test_metrics['f1']:.4f}")

# ============================================================
# AGGREGATE RESULTS
# ============================================================
metric_keys = ["test_auc", "test_accuracy", "test_precision", "test_recall", "test_f1"]
summary = {}
for key in metric_keys:
    values = [r[key] for r in fold_results]
    clean_key = key.replace("test_", "")
    summary[clean_key] = {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
        "per_fold": values,
    }

output = {
    "experiment": "5-Fold Cross-Validation with Early Stopping",
    "n_folds": 5,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "fold_results": fold_results,
    "summary": summary,
}

with open(os.path.join(OUT_DIR, "crossval_results.json"), "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to crossval_results.json")

# Save AUC coordinates (from last fold's test evaluation for reference)
auc_coords = {
    "description": "ROC curve coordinates from each fold's test evaluation",
    "folds": []
}

# Re-evaluate each fold to get FPR/TPR coordinates
# We'll use the final fold's data since we can't save all models
# Instead, let's save the last evaluated test metrics which include fpr/tpr
auc_coords["note"] = "AUC coordinates from the final fold's test set evaluation"
auc_coords["fpr"] = test_metrics["fpr"]
auc_coords["tpr"] = test_metrics["tpr"]
auc_coords["test_auc"] = round(float(test_metrics["auc"]), 4)

with open(os.path.join(OUT_DIR, "auc_coordinates.json"), "w") as f:
    json.dump(auc_coords, f, indent=2)
print("AUC coordinates saved to auc_coordinates.json")

# ============================================================
# PLOT: Training Curves (2x2 grid)
# ============================================================
print("Generating training curves plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Colors for each fold
fold_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

for fold_idx, curves in enumerate(all_training_curves):
    epochs = range(1, len(curves["train_loss"]) + 1)
    color = fold_colors[fold_idx]
    label = f"Fold {fold_idx + 1}"

    axes[0, 0].plot(epochs, curves["train_loss"], color=color, alpha=0.7, label=label)
    axes[0, 1].plot(epochs, curves["val_loss"], color=color, alpha=0.7, label=label)
    axes[1, 0].plot(epochs, curves["train_auc"], color=color, alpha=0.7, label=label)
    axes[1, 1].plot(epochs, curves["val_auc"], color=color, alpha=0.7, label=label)

axes[0, 0].set_title("Training Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
axes[0, 1].set_title("Validation Loss"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Loss")
axes[1, 0].set_title("Training AUC"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("AUC")
axes[1, 1].set_title("Validation AUC"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("AUC")

for ax in axes.flat:
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("M5: 5-Fold Cross-Validation Training Curves", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
plt.close(fig)
print("  [OK] Saved training_curves.png")

# Print summary
print(f"\n{'='*60}")
print("CROSS-VALIDATION SUMMARY (mean +/- std)")
print(f"{'='*60}")
for key, val in summary.items():
    print(f"  {key.upper():<12}: {val['mean']:.4f} +/- {val['std']:.4f}")
print(f"{'='*60}")
