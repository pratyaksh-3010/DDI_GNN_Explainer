"""
Graph builder for Streamlit frontend.
Builds subgraphs for user-specified drug combinations.
Uses node_index (integer) based edges — consistent with training.
"""

import pandas as pd
import torch
import json
import os
from torch_geometric.data import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
nodes_path = os.path.join(BASE_DIR, "DATA", "layer3_nodes_enriched.csv")
edges_path = os.path.join(BASE_DIR, "DATA", "layer3_edges_weighted.csv")
meta_path = os.path.join(BASE_DIR, "DATA", "layer3_feature_meta.json")

nodes = pd.read_csv(nodes_path)
edges = pd.read_csv(edges_path)

with open(meta_path, "r") as f:
    feat_meta = json.load(f)

feat_names = feat_meta["feature_names"]

# Build drug name → node_index lookup (case-insensitive)
name_to_index = {}
for _, row in nodes.iterrows():
    drug = row.get("drug_name")
    if isinstance(drug, str) and drug.strip():
        name_to_index[drug.strip().lower()] = int(row["node_index"])

# Feature matrix using ONLY the feature columns
X = torch.tensor(
    nodes[feat_names].values,
    dtype=torch.float
)

# Precompute adjacency from edges (edges use node_index already)
adj = {}  # node_index → [(neighbor_index, weight), ...]
for _, row in edges.iterrows():
    src = int(row["source"])
    tgt = int(row["target"])
    w = float(row["weight"])

    if src not in adj:
        adj[src] = []
    adj[src].append((tgt, w))

    if tgt not in adj:
        adj[tgt] = []
    adj[tgt].append((src, w))


def get_available_drugs():
    """Return list of all available drug names."""
    return sorted(name_to_index.keys())


def build_graph(drug_names):
    """
    Build a PyG subgraph from a list of drug names.

    Returns:
        data: PyG Data object
        indices: list of global node indices
    """
    indices = []

    for drug in drug_names:
        drug_clean = drug.lower().strip()

        if drug_clean not in name_to_index:
            raise ValueError(
                f"'{drug}' not found in database. "
                f"Available drugs: {', '.join(sorted(list(name_to_index.keys()))[:20])}..."
            )

        indices.append(name_to_index[drug_clean])

    node_set = set(indices)

    # Build subgraph edges using adjacency list (fast!)
    sub_edges = []
    sub_weights = []
    local_map = {idx: i for i, idx in enumerate(indices)}

    for idx in indices:
        if idx in adj:
            for neighbor, w in adj[idx]:
                if neighbor in node_set:
                    s = local_map[idx]
                    t = local_map[neighbor]
                    sub_edges.append([s, t])
                    sub_weights.append(w)

    # If no edges found, add self-loops with low weight
    # (so the GNN can still process the graph)
    if len(sub_edges) == 0:
        for i in range(len(indices)):
            sub_edges.append([i, i])
            sub_weights.append(0.1)

    edge_index = torch.tensor(sub_edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(sub_weights, dtype=torch.float)

    data = Data(
        x=X[indices],
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_id=torch.tensor(indices, dtype=torch.long)
    )

    return data, indices