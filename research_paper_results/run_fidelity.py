"""
Fidelity+ Scores — Quantitative Explanation Faithfulness

Computes fidelity+ metric: does removing top-k important nodes change
the model's prediction? Higher fidelity+ = explanations are faithful.

For each test graph:
  1. Get original prediction
  2. Identify top-k important nodes (via gradient attribution)
  3. Mask/zero out those nodes' features
  4. Get masked prediction
  5. fidelity+ = |original_pred - masked_pred|

Reports mean fidelity+ for k=1,2,3 across all test graphs.

Saves: fidelity_results.json
"""

import os, json, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split

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
# FIDELITY+ COMPUTATION
# ============================================================
def get_node_importance(graph):
    """Compute per-node importance via gradient attribution."""
    graph = graph.to(device)
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
    return drug_scores


def compute_fidelity_plus(graph, top_k):
    """
    Compute fidelity+ by masking top-k important nodes.
    Returns |original_pred - masked_pred|.
    """
    graph = graph.to(device)
    n_nodes = graph.x.shape[0]

    if top_k >= n_nodes:
        top_k = n_nodes - 1  # Keep at least one node
    if top_k <= 0:
        return None

    # Original prediction
    with torch.no_grad():
        original_logit = model(graph).item()
        original_prob = torch.sigmoid(torch.tensor(original_logit)).item()

    # Get node importance
    importance = get_node_importance(graph)
    top_k_nodes = np.argsort(importance)[::-1][:top_k]

    # Create masked graph (zero out features of top-k nodes)
    masked_x = graph.x.clone()
    for node_idx in top_k_nodes:
        masked_x[node_idx] = torch.zeros(FEAT_DIM)

    # Masked prediction
    with torch.no_grad():
        masked_logit = model(
            masked_x,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
            batch=torch.zeros(masked_x.size(0), dtype=torch.long).to(device),
            node_id=graph.node_id
        ).item()
        masked_prob = torch.sigmoid(torch.tensor(masked_logit)).item()

    fidelity = abs(original_prob - masked_prob)
    return fidelity


def compute_fidelity_minus(graph, top_k):
    """
    Compute fidelity- by keeping ONLY top-k important nodes.
    Returns |original_pred - kept_pred|.
    Lower fidelity- = better (explanations capture what matters).
    """
    graph = graph.to(device)
    n_nodes = graph.x.shape[0]

    if top_k >= n_nodes:
        top_k = n_nodes - 1
    if top_k <= 0:
        return None

    # Original prediction
    with torch.no_grad():
        original_logit = model(graph).item()
        original_prob = torch.sigmoid(torch.tensor(original_logit)).item()

    # Get node importance — mask everything EXCEPT top-k
    importance = get_node_importance(graph)
    top_k_nodes = set(np.argsort(importance)[::-1][:top_k].tolist())

    masked_x = graph.x.clone()
    for node_idx in range(n_nodes):
        if node_idx not in top_k_nodes:
            masked_x[node_idx] = torch.zeros(FEAT_DIM)

    with torch.no_grad():
        masked_logit = model(
            masked_x,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
            batch=torch.zeros(masked_x.size(0), dtype=torch.long).to(device),
            node_id=graph.node_id
        ).item()
        masked_prob = torch.sigmoid(torch.tensor(masked_logit)).item()

    fidelity = abs(original_prob - masked_prob)
    return fidelity


# ============================================================
# RUN ON ALL TEST GRAPHS
# ============================================================
print(f"Computing fidelity scores on {len(test_data)} test graphs...")

k_values = [1, 2, 3]
fidelity_plus_results = {k: [] for k in k_values}
fidelity_minus_results = {k: [] for k in k_values}

for i, graph in enumerate(test_data):
    if (i + 1) % 100 == 0 or i == 0:
        print(f"  Processing graph {i+1}/{len(test_data)}...")

    n_nodes = graph.x.shape[0]

    for k in k_values:
        if n_nodes > k:
            fp = compute_fidelity_plus(graph, k)
            fm = compute_fidelity_minus(graph, k)
            if fp is not None:
                fidelity_plus_results[k].append(fp)
            if fm is not None:
                fidelity_minus_results[k].append(fm)

# ============================================================
# AGGREGATE
# ============================================================
output = {
    "experiment": "Fidelity Scores (Explanation Faithfulness)",
    "total_test_graphs": len(test_data),
    "k_values": k_values,
    "fidelity_plus": {},
    "fidelity_minus": {},
    "interpretation": {
        "fidelity_plus": "Higher fidelity+ = removing top-k important nodes significantly changes prediction = explanations are faithful",
        "fidelity_minus": "Lower fidelity- = keeping only top-k important nodes preserves prediction = explanations capture what matters",
    },
}

for k in k_values:
    fp_vals = fidelity_plus_results[k]
    fm_vals = fidelity_minus_results[k]

    output["fidelity_plus"][f"k={k}"] = {
        "mean": round(np.mean(fp_vals), 4) if fp_vals else None,
        "std": round(np.std(fp_vals), 4) if fp_vals else None,
        "median": round(np.median(fp_vals), 4) if fp_vals else None,
        "n_samples": len(fp_vals),
    }
    output["fidelity_minus"][f"k={k}"] = {
        "mean": round(np.mean(fm_vals), 4) if fm_vals else None,
        "std": round(np.std(fm_vals), 4) if fm_vals else None,
        "median": round(np.median(fm_vals), 4) if fm_vals else None,
        "n_samples": len(fm_vals),
    }

with open(os.path.join(OUT_DIR, "fidelity_results.json"), "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print("FIDELITY SCORES SUMMARY")
print(f"{'='*60}")
print("\nFidelity+ (higher = better, explanations remove critical info):")
for k in k_values:
    vals = output["fidelity_plus"][f"k={k}"]
    if vals["mean"] is not None:
        print(f"  k={k}: mean={vals['mean']:.4f} +/- {vals['std']:.4f}  (n={vals['n_samples']})")

print("\nFidelity- (lower = better, explanations preserve critical info):")
for k in k_values:
    vals = output["fidelity_minus"][f"k={k}"]
    if vals["mean"] is not None:
        print(f"  k={k}: mean={vals['mean']:.4f} +/- {vals['std']:.4f}  (n={vals['n_samples']})")

print(f"{'='*60}")
print("Results saved to fidelity_results.json")
