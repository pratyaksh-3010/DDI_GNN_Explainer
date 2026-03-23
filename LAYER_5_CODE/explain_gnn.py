"""
Layer 5 — GNN Explainability

Produces per-drug importance scores + rich explanations showing:
  - Which features (ATC, targets, side effects) drive risk
  - Which drug-drug edges are most important
  - Specific reasons (enzyme conflicts, ATC duplication, etc.)
"""

import pandas as pd
import numpy as np
import torch
import ast
import json

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD DATA
# ============================================================
nodes_df = pd.read_csv("layer3_nodes.csv")
edges_df = pd.read_csv("layer3_edges_weighted.csv")
cases_df = pd.read_csv("layer3_training_cases.csv")

with open("layer3_feature_meta.json", "r") as f:
    feat_meta = json.load(f)

FEAT_DIM = feat_meta["feat_dim"]
feat_names = feat_meta["feature_names"]

# Node features
X = nodes_df[feat_names].values
X = torch.tensor(X, dtype=torch.float)

node_ids = list(nodes_df["structure_id"].astype(str))
node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
NUM_NODES = len(node_ids)

# Try to load drug name lookup
try:
    name_lookup = pd.read_csv("structure_name_lookup.csv", dtype=str)
    struct_to_name = dict(zip(name_lookup["structure_id"], name_lookup["drug_name"]))
except:
    struct_to_name = {}

# Build global edge index
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

# ============================================================
# BUILD GRAPH LIST
# ============================================================
adj_dict = {}
for i in range(edge_index.shape[1]):
    u = edge_index[0, i].item()
    v = edge_index[1, i].item()
    w = edge_weight[i].item()
    if u not in adj_dict:
        adj_dict[u] = []
    adj_dict[u].append((v, w))

data_list = []
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

print(f"Total graphs available: {len(data_list)}")

# ============================================================
# MODEL DEFINITION (must match training)
# ============================================================
EMBED_DIM = 32

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

        return x  # RAW LOGITS

# ============================================================
# LOAD TRAINED MODEL
# ============================================================
checkpoint = torch.load("weighted_gcn_model.pth", map_location=device)

model = WeightedGCN(
    num_nodes=checkpoint.get("num_nodes", NUM_NODES),
    feat_dim=checkpoint.get("feat_dim", FEAT_DIM),
    embed_dim=checkpoint.get("embed_dim", EMBED_DIM)
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully.")

# ============================================================
# SELECT SAMPLE CASES
# ============================================================
high_risk_graphs = [g for g in data_list if g.y.item() == 1]
low_risk_graphs = [g for g in data_list if g.y.item() == 0]

print(f"High risk graphs: {len(high_risk_graphs)}")
print(f"Low risk graphs: {len(low_risk_graphs)}")


def get_drug_name(global_idx):
    """Get human-readable drug name from global node index."""
    struct_id = node_ids[global_idx]
    return struct_to_name.get(struct_id, f"Drug_{struct_id}")


def explain_case(graph, title=""):
    """Run full explainability pipeline on a single graph."""
    graph = graph.to(device)

    print(f"\n{'='*60}")
    print(f"CASE: {title}")
    print(f"Drugs in combination: {graph.x.shape[0]}")
    print(f"Edges: {graph.edge_index.shape[1]}")

    # --- Prediction ---
    with torch.no_grad():
        logit = model(graph).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

    print(f"Risk probability: {prob:.4f} ({'HIGH RISK' if prob > 0.5 else 'LOW RISK'})")
    print(f"Ground truth: {'HIGH' if graph.y.item() == 1 else 'LOW'}")

    # --- Drug names ---
    drug_names = []
    for i in range(graph.x.shape[0]):
        global_id = graph.node_id[i].item()
        drug_names.append(get_drug_name(global_id))
    print(f"Drugs: {drug_names}")

    # -----------------------------------------------
    # 1. GNNExplainer
    # -----------------------------------------------
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=ModelConfig(
            mode='binary_classification',
            task_level='graph',
            return_type='raw'  # CORRECT: model returns raw logits
        )
    )

    explanation = explainer(
        graph.x,
        graph.edge_index,
        edge_weight=graph.edge_weight,
        batch=torch.zeros(graph.x.size(0), dtype=torch.long).to(device),
        node_id=graph.node_id
    )

    # Node importance (sum over all feature attributions per node)
    node_importance = explanation.node_mask.sum(dim=1).detach().cpu().numpy()
    node_importance_norm = node_importance / (node_importance.sum() + 1e-8)

    # Feature importance (which features matter most globally)
    feat_importance = explanation.node_mask.mean(dim=0).detach().cpu().numpy()

    # Edge importance
    edge_importance = explanation.edge_mask.detach().cpu().numpy()

    print("\n--- GNNExplainer: Per-Drug Importance ---")
    ranked = np.argsort(node_importance_norm)[::-1]
    for rank, idx in enumerate(ranked):
        global_id = graph.node_id[idx].item()
        name = get_drug_name(global_id)
        score = node_importance_norm[idx]
        print(f"  #{rank+1} {name} (struct {node_ids[global_id]}): {score:.4f}")

        # Show which features contribute for this drug
        node_feat_scores = explanation.node_mask[idx].detach().cpu().numpy()
        top_feats = np.argsort(np.abs(node_feat_scores))[::-1][:5]
        for fi in top_feats:
            if abs(node_feat_scores[fi]) > 0.01:
                print(f"      → {feat_names[fi]}: {node_feat_scores[fi]:.4f}")

    print("\n--- GNNExplainer: Top Feature Importance ---")
    top_feat_idx = np.argsort(np.abs(feat_importance))[::-1][:10]
    for fi in top_feat_idx:
        print(f"  {feat_names[fi]}: {feat_importance[fi]:.4f}")

    print("\n--- GNNExplainer: Edge Importance ---")
    edge_idx_np = graph.edge_index.cpu().numpy()
    edge_ranked = np.argsort(edge_importance)[::-1]
    seen_pairs = set()
    for ei in edge_ranked[:10]:
        s, t = edge_idx_np[0, ei], edge_idx_np[1, ei]
        pair = (min(s, t), max(s, t))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        name_s = drug_names[s]
        name_t = drug_names[t]
        print(f"  {name_s} ↔ {name_t}: importance={edge_importance[ei]:.4f}")

    # -----------------------------------------------
    # 2. Gradient-based Attribution (Integrated Gradients approx)
    # -----------------------------------------------
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

    # Gradient × Input attribution
    attr = np.abs(grads * x_vals)
    node_saliency = attr.sum(axis=1)
    node_saliency_norm = node_saliency / (node_saliency.sum() + 1e-8)

    print("\n--- Gradient Attribution: Per-Drug Importance ---")
    ranked = np.argsort(node_saliency_norm)[::-1]
    for rank, idx in enumerate(ranked):
        global_id = graph.node_id[idx].item()
        name = get_drug_name(global_id)
        score = node_saliency_norm[idx]
        print(f"  #{rank+1} {name}: {score:.4f}")

        # Which features drive the gradient for this drug?
        drug_attr = attr[idx]
        top_feats = np.argsort(drug_attr)[::-1][:5]
        for fi in top_feats:
            if drug_attr[fi] > 0.01:
                print(f"      → {feat_names[fi]}: {drug_attr[fi]:.4f}")


# ============================================================
# RUN EXPLANATIONS
# ============================================================
if high_risk_graphs:
    explain_case(high_risk_graphs[0], "High Risk Sample 1")

if len(high_risk_graphs) > 1:
    explain_case(high_risk_graphs[1], "High Risk Sample 2")

if low_risk_graphs:
    explain_case(low_risk_graphs[0], "Low Risk Sample 1")

print("\n=== Explainability analysis complete ===")