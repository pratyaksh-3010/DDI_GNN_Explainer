"""
Model utilities for Streamlit frontend.
Loads the trained model, runs predictions, and provides explainability.
"""

import os
import json
import torch
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig


# ============================================================
# MODEL DEFINITION (must match training exactly)
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

        x = x.float()
        edge_weight = edge_weight.float()
        edge_index = edge_index.long()

        if node_id is None:
            node_id = torch.arange(x.size(0), device=x.device)
        node_id = node_id.long()

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
# LOAD MODEL
# ============================================================
def load_model():
    """Load trained model from checkpoint."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "DATA", "weighted_gcn_model.pth")
    meta_path = os.path.join(BASE_DIR, "DATA", "layer3_feature_meta.json")

    # Load feature metadata
    with open(meta_path, "r") as f:
        feat_meta = json.load(f)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    model = WeightedGCN(
        num_nodes=checkpoint["num_nodes"],
        feat_dim=checkpoint["feat_dim"],
        embed_dim=checkpoint.get("embed_dim", 32)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, feat_meta


# ============================================================
# PREDICT
# ============================================================
def predict(model, graph_data):
    """Run prediction and return risk probability."""
    model.eval()

    with torch.no_grad():
        logit = model(graph_data)
        prob = torch.sigmoid(logit).item()

    return prob


# ============================================================
# GRADIENT ATTRIBUTION
# ============================================================
def gradient_attribution(model, data, feat_names):
    """
    Compute per-drug importance via Gradient × Input attribution.

    Returns:
        drug_scores: normalized importance per drug (numpy array)
        feature_attr: per-drug per-feature attribution matrix (numpy array)
    """
    model.eval()

    x_input = data.x.clone().detach().requires_grad_(True)

    logit = model(
        x_input,
        edge_index=data.edge_index,
        edge_weight=data.edge_weight,
        batch=torch.zeros(x_input.size(0), dtype=torch.long),
        node_id=data.node_id
    )

    model.zero_grad()
    logit.backward()

    grads = x_input.grad.detach().cpu().numpy()
    x_vals = x_input.detach().cpu().numpy()

    # Gradient × Input
    attr = np.abs(grads * x_vals)

    # Per-drug importance (sum over features)
    drug_scores = attr.sum(axis=1)
    drug_scores_norm = drug_scores / (drug_scores.sum() + 1e-8)

    return drug_scores_norm, attr


# ============================================================
# GNNEXPLAINER
# ============================================================
def explain_graph(model, data, feat_names):
    """
    Run GNNExplainer on a graph.

    Returns:
        node_imp: per-node importance (numpy array)
        edge_imp: per-edge importance (numpy array)
        feat_imp: per-feature importance (numpy array)
        node_feat_attr: per-node per-feature attribution matrix
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="binary_classification",
            task_level="graph",
            return_type="raw"  # Model returns raw logits
        ),
    )

    explanation = explainer(
        data.x,
        data.edge_index,
        edge_weight=data.edge_weight,
        batch=torch.zeros(data.x.size(0), dtype=torch.long),
        node_id=data.node_id
    )

    node_feat_attr = explanation.node_mask.detach().cpu().numpy()
    node_imp = node_feat_attr.sum(axis=1)
    node_imp_norm = node_imp / (node_imp.sum() + 1e-8)

    edge_imp = explanation.edge_mask.detach().cpu().numpy()

    feat_imp = np.abs(node_feat_attr).mean(axis=0)

    return node_imp_norm, edge_imp, feat_imp, node_feat_attr


# ============================================================
# INTERPRET FEATURES
# ============================================================
def get_top_features(attr_row, feat_names, top_k=5):
    """
    Given a single node's feature attribution row, return the top-k
    contributing features as a list of (feature_name, score) tuples.
    """
    abs_attr = np.abs(attr_row)
    top_idx = np.argsort(abs_attr)[::-1][:top_k]

    results = []
    for idx in top_idx:
        if abs_attr[idx] > 0.001:
            name = feat_names[idx]
            # Make feature names human-readable
            display = _beautify_feature(name)
            results.append((display, float(attr_row[idx])))

    return results


def _beautify_feature(name):
    """Convert feature column names to human-readable labels."""
    if name.startswith("atc_l1_"):
        letter = name.replace("atc_l1_", "")
        atc_names = {
            "A": "Alimentary/Metabolism",
            "B": "Blood",
            "C": "Cardiovascular",
            "D": "Dermatologicals",
            "G": "Genitourinary",
            "H": "Hormones",
            "J": "Anti-infectives",
            "L": "Antineoplastic/Immunomod.",
            "M": "Musculoskeletal",
            "N": "Nervous System",
            "P": "Antiparasitic",
            "R": "Respiratory",
            "S": "Sensory Organs",
            "V": "Various",
        }
        return f"ATC: {atc_names.get(letter, letter)}"
    elif name.startswith("tc_"):
        return f"Target: {name[3:]}"
    elif name.startswith("at_"):
        return f"Action: {name[3:]}"
    elif name == "log_atc_count":
        return "Drug class diversity"
    elif name == "log_gene_count":
        return "Target gene count"
    elif name == "log_se_count":
        return "Known side effects count"
    return name