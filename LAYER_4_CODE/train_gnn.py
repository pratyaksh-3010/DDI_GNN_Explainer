"""
Layer 4 — GNN Training for Polypharmacy Risk Prediction

Architecture: WeightedGCN with:
  - Learnable node embeddings (32-dim)
  - Rich multi-hot node features (from layer3_feature_meta.json)
  - 2-layer GCN with edge weights
  - Global mean pooling → MLP → logits (raw, no sigmoid)
  - BCEWithLogitsLoss with class weighting
  - Dropout for regularization
"""

import pandas as pd
import numpy as np
import torch
import ast
import json

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, confusion_matrix

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
nodes_df = pd.read_csv("layer3_nodes.csv")
edges_df = pd.read_csv("layer3_edges_weighted.csv")
cases_df = pd.read_csv("layer3_training_cases.csv")

with open("layer3_feature_meta.json", "r") as f:
    feat_meta = json.load(f)

FEAT_DIM = feat_meta["feat_dim"]
feat_names = feat_meta["feature_names"]

print(f"Feature dimension: {FEAT_DIM}")
print(f"Feature names: {feat_names[:5]} ... ({len(feat_names)} total)")

# ============================================================
# BUILD NODE FEATURE MATRIX
# ============================================================
# Extract only the feature columns (skip structure_id, raw count columns)
X = nodes_df[feat_names].values
X = torch.tensor(X, dtype=torch.float)

print(f"Node feature matrix: {X.shape}")

# Node index mapping
node_ids = list(nodes_df["structure_id"].astype(str))
node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
NUM_NODES = len(node_ids)

# ============================================================
# BUILD GLOBAL EDGE INDEX + WEIGHT
# ============================================================
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

print(f"Global edges: {edge_index.shape[1]}")

# ============================================================
# BUILD PER-CASE SUBGRAPHS
# ============================================================
print("Building subgraphs...")

adj_dict = {}
for i in range(edge_index.shape[1]):
    u = edge_index[0, i].item()
    v = edge_index[1, i].item()
    w = edge_weight[i].item()
    if u not in adj_dict:
        adj_dict[u] = []
    adj_dict[u].append((v, w))

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

print(f"Total subgraphs: {len(data_list)} (skipped {skipped} with no edges)")

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
labels = [data.y.item() for data in data_list]

train_idx, test_idx = train_test_split(
    range(len(data_list)),
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_data = [data_list[i] for i in train_idx]
test_data = [data_list[i] for i in test_idx]

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# Compute class weight for imbalanced data
n_pos = sum(labels)
n_neg = len(labels) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)
print(f"Class distribution: {n_pos} positive, {n_neg} negative")
print(f"pos_weight: {pos_weight.item():.2f}")

# ============================================================
# MODEL DEFINITION
# ============================================================
EMBED_DIM = 32

class WeightedGCN(torch.nn.Module):
    """
    GCN that outputs RAW LOGITS (no sigmoid).
    Use BCEWithLogitsLoss for training, apply sigmoid at inference.
    """
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
        # Support both Data object input and explicit tensor input
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
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.dropout(x)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = global_mean_pool(x, batch)

        x = torch.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)  # RAW LOGITS — no sigmoid here

        return x


# ============================================================
# TRAINING
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = WeightedGCN(
    num_nodes=NUM_NODES,
    feat_dim=FEAT_DIM,
    embed_dim=EMBED_DIM
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))


def train_epoch():
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

    return total_loss / len(train_loader)


def evaluate(loader):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)

            probs = torch.sigmoid(logits).view(-1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    preds = (all_probs > best_threshold).astype(int)

    acc = np.mean(preds == all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    cm = confusion_matrix(all_labels, preds)

    return acc, auc, precision, recall, f1, cm, best_threshold


# ============================================================
# TRAIN LOOP
# ============================================================
EPOCHS = 100
best_auc = 0

for epoch in range(EPOCHS):
    loss = train_epoch()
    acc, auc, precision, recall, f1, cm, threshold = evaluate(test_loader)

    if auc > best_auc:
        best_auc = auc
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_nodes": NUM_NODES,
            "feat_dim": FEAT_DIM,
            "embed_dim": EMBED_DIM,
            "best_threshold": threshold,
            "best_auc": best_auc,
        }, "weighted_gcn_model.pth")

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"""
Epoch {epoch+1}/{EPOCHS}
  Loss:      {loss:.4f}
  Accuracy:  {acc:.4f}
  AUC:       {auc:.4f}
  Precision: {precision:.4f}
  Recall:    {recall:.4f}
  F1:        {f1:.4f}
  Threshold: {threshold:.4f}
  Confusion Matrix:
  {cm}""")

print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
print("Model saved to weighted_gcn_model.pth")

# Save model config for frontend
model_config = {
    "num_nodes": NUM_NODES,
    "feat_dim": FEAT_DIM,
    "embed_dim": EMBED_DIM,
}
with open("model_config.json", "w") as f:
    json.dump(model_config, f, indent=2)
print("Model config saved to model_config.json")
