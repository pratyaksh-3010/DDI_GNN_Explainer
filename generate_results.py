"""
Polypharmacy GNN — Full Results Generation

Generates all metrics, plots, and explainability outputs:
  - ROC Curve
  - Confusion Matrix Heatmap
  - Feature Importance Chart
  - Per-Drug Importance (sample cases)
  - Dataset & Model Summary Statistics
  - Explainability Case Studies
"""

import os
import sys
import io
import json

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import ast
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score,
    recall_score, f1_score, confusion_matrix, accuracy_score
)

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "RESULTS")
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# ============================================================
# MODEL
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
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.dropout(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x


def beautify_feature(name):
    if name.startswith("atc_l1_"):
        letter = name.replace("atc_l1_", "")
        atc_names = {
            "A": "Alimentary/Metabolism", "B": "Blood", "C": "Cardiovascular",
            "D": "Dermatologicals", "G": "Genitourinary", "H": "Hormones",
            "J": "Anti-infectives", "L": "Antineoplastic", "M": "Musculoskeletal",
            "N": "Nervous System", "P": "Antiparasitic", "R": "Respiratory",
            "S": "Sensory Organs", "V": "Various",
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
        return "Known side effects"
    return name


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

# Drug name lookup
try:
    name_lookup = pd.read_csv(os.path.join(BASE, "structure_name_lookup.csv"), dtype=str)
    struct_to_name = dict(zip(name_lookup["structure_id"], name_lookup["drug_name"]))
except:
    struct_to_name = {}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(os.path.join(BASE, "weighted_gcn_model.pth"), map_location=device)

model = WeightedGCN(
    num_nodes=checkpoint["num_nodes"],
    feat_dim=checkpoint["feat_dim"],
    embed_dim=checkpoint.get("embed_dim", 32)
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded.")

# Build edges
edge_index_list = []
edge_weight_list = []
for _, row in edges_df.iterrows():
    i, j, w = int(row["source"]), int(row["target"]), float(row["weight"])
    edge_index_list.append([i, j])
    edge_index_list.append([j, i])
    edge_weight_list.append(w)
    edge_weight_list.append(w)

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

adj_dict = {}
for i in range(edge_index.shape[1]):
    u, v, w = edge_index[0, i].item(), edge_index[1, i].item(), edge_weight[i].item()
    if u not in adj_dict:
        adj_dict[u] = []
    adj_dict[u].append((v, w))

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

# Train/test split
labels = [d.y.item() for d in data_list]
train_idx, test_idx = train_test_split(
    range(len(data_list)), test_size=0.2, stratify=labels, random_state=42
)
test_data = [data_list[i] for i in test_idx]
test_loader = DataLoader(test_data, batch_size=64)

# ============================================================
# EVALUATE
# ============================================================
print("Evaluating model on test set...")
all_probs, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).view(-1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch.y.view(-1).cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

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

print(f"  AUC={auc:.4f}  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

# ============================================================
# PLOT 1: ROC CURVE
# ============================================================
print("Generating ROC Curve...")
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"WeightedGCN (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")
ax.scatter([fpr[best_idx]], [tpr[best_idx]], c="red", s=100, zorder=5,
           label=f"Optimal threshold = {best_threshold:.3f}")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Polypharmacy Risk Prediction")
ax.legend(loc="lower right")
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
plt.close(fig)
print("  [OK] Saved roc_curve.png")

# ============================================================
# PLOT 2: CONFUSION MATRIX
# ============================================================
print("Generating Confusion Matrix...")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"],
            ax=ax, cbar_kws={"label": "Count"})
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix (threshold={best_threshold:.3f})")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close(fig)
print("  [OK] Saved confusion_matrix.png")

# ============================================================
# PLOT 3: FEATURE IMPORTANCE (Gradient-based, averaged)
# ============================================================
print("Computing feature importance (gradient attribution, sampled)...")

# Sample up to 200 test cases for gradient computation
sample_size = min(200, len(test_data))
sample_data = test_data[:sample_size]

all_attr = np.zeros(FEAT_DIM)

for data in sample_data:
    data = data.to(device)
    x_input = data.x.clone().detach().requires_grad_(True)
    logit = model(
        x_input,
        edge_index=data.edge_index,
        edge_weight=data.edge_weight,
        batch=torch.zeros(x_input.size(0), dtype=torch.long).to(device),
        node_id=data.node_id
    )
    model.zero_grad()
    logit.backward()
    grads = x_input.grad.detach().cpu().numpy()
    x_vals = x_input.detach().cpu().numpy()
    attr = np.abs(grads * x_vals).mean(axis=0)
    all_attr += attr

all_attr /= sample_size

# Top 15 features
top_feat_idx = np.argsort(all_attr)[::-1][:15]
top_names = [beautify_feature(feat_names[i]) for i in top_feat_idx]
top_scores = [all_attr[i] for i in top_feat_idx]

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_names)))
bars = ax.barh(range(len(top_names)), top_scores[::-1], color=colors[::-1])
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names[::-1])
ax.set_xlabel("Average Feature Importance (|Gradient x Input|)")
ax.set_title("Top 15 Features Driving Polypharmacy Risk")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"))
plt.close(fig)
print("  [OK] Saved feature_importance.png")

# ============================================================
# PLOT 4: CLASS DISTRIBUTION
# ============================================================
print("Generating class distribution chart...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
n_pos_all = sum(labels)
n_neg_all = len(labels) - n_pos_all
bars = axes[0].bar(["Low Risk", "High Risk"], [n_neg_all, n_pos_all],
                   color=["#4CAF50", "#F44336"], edgecolor="white", linewidth=1.5)
axes[0].set_title("Training Case Class Distribution")
axes[0].set_ylabel("Count")
for bar, val in zip(bars, [n_neg_all, n_pos_all]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{val:,}", ha="center", fontweight="bold")

# Drugs per case histogram
drugs_per_case = [len(ast.literal_eval(row["nodes"])) for _, row in cases_df.iterrows()]
axes[1].hist(drugs_per_case, bins=range(2, max(drugs_per_case)+2),
             color="#2196F3", edgecolor="white", alpha=0.8)
axes[1].set_title("Drugs per Case Distribution")
axes[1].set_xlabel("Number of Drugs")
axes[1].set_ylabel("Count")
axes[1].axvline(np.mean(drugs_per_case), color="red", linestyle="--",
                label=f"Mean = {np.mean(drugs_per_case):.1f}")
axes[1].legend()

fig.suptitle("Dataset Statistics", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "dataset_stats.png"))
plt.close(fig)
print("  [OK] Saved dataset_stats.png")

# ============================================================
# PLOT 5: PER-DRUG IMPORTANCE EXAMPLE
# ============================================================
print("Generating per-drug importance example...")

high_risk_graphs = [g for g in data_list if g.y.item() == 1]
low_risk_graphs = [g for g in data_list if g.y.item() == 0]


def get_drug_name(global_idx):
    sid = node_ids[global_idx]
    return struct_to_name.get(sid, f"Drug_{sid}")


def drug_importance_gradient(graph):
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
    drug_scores_norm = drug_scores / (drug_scores.sum() + 1e-8)
    prob = torch.sigmoid(logit).item()
    return drug_scores_norm, prob


# Select examples with enough drugs
hr_examples = [g for g in high_risk_graphs if g.x.shape[0] >= 3][:2]
lr_examples = [g for g in low_risk_graphs if g.x.shape[0] >= 3][:1]

examples = hr_examples + lr_examples
n_examples = len(examples)

if n_examples > 0:
    fig, axes = plt.subplots(1, n_examples, figsize=(6 * n_examples, 5))
    if n_examples == 1:
        axes = [axes]

    titles = ["High Risk Case 1", "High Risk Case 2", "Low Risk Case 1"]

    for i, (graph, title) in enumerate(zip(examples, titles[:n_examples])):
        scores, prob = drug_importance_gradient(graph)
        names = [get_drug_name(graph.node_id[j].item()) for j in range(graph.x.shape[0])]

        # Truncate long names
        short_names = [n[:20] + "..." if len(n) > 20 else n for n in names]

        ranked = np.argsort(scores)[::-1]
        sorted_names = [short_names[j] for j in ranked]
        sorted_scores = [scores[j] for j in ranked]

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_names)))
        axes[i].barh(range(len(sorted_names)), sorted_scores[::-1], color=colors[::-1])
        axes[i].set_yticks(range(len(sorted_names)))
        axes[i].set_yticklabels(sorted_names[::-1], fontsize=9)
        axes[i].set_xlabel("Importance")
        label = "HIGH" if graph.y.item() == 1 else "LOW"
        axes[i].set_title(f"{title}\nPred: {prob:.1%} ({label} risk)")
        axes[i].grid(axis="x", alpha=0.3)

    fig.suptitle("Per-Drug Risk Contribution (Gradient Attribution)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "drug_importance.png"))
    plt.close(fig)
    print("  [OK] Saved drug_importance.png")

# ============================================================
# PLOT 6: PREDICTION DISTRIBUTION
# ============================================================
print("Generating prediction distribution...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(all_probs[all_labels == 0], bins=30, alpha=0.7, color="#4CAF50",
        label="Low Risk (actual)", density=True)
ax.hist(all_probs[all_labels == 1], bins=30, alpha=0.7, color="#F44336",
        label="High Risk (actual)", density=True)
ax.axvline(best_threshold, color="black", linestyle="--", linewidth=2,
           label=f"Threshold = {best_threshold:.3f}")
ax.set_xlabel("Predicted Risk Probability")
ax.set_ylabel("Density")
ax.set_title("Prediction Distribution by True Label")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "prediction_distribution.png"))
plt.close(fig)
print("  [OK] Saved prediction_distribution.png")

# ============================================================
# SUMMARY STATS FILE
# ============================================================
print("Saving summary statistics...")
with open(os.path.join(RESULTS_DIR, "summary_stats.txt"), "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("POLYPHARMACY GNN — RESULTS SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write("DATASET STATISTICS\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Total drug nodes:          {NUM_NODES}\n")
    f.write(f"  Total drug-drug edges:     {len(edges_df)}\n")
    f.write(f"  Total training cases:      {len(cases_df)}\n")
    f.write(f"  Valid subgraphs:           {len(data_list)}\n")
    f.write(f"  Feature dimension:         {FEAT_DIM}\n")
    f.write(f"  High risk cases:           {n_pos_all} ({100*n_pos_all/len(labels):.1f}%)\n")
    f.write(f"  Low risk cases:            {n_neg_all} ({100*n_neg_all/len(labels):.1f}%)\n")
    f.write(f"  Avg drugs per case:        {np.mean(drugs_per_case):.1f}\n\n")

    f.write("DATA SOURCES\n")
    f.write("-" * 40 + "\n")
    f.write("  FAERS (2023 Q1-Q4):        FDA Adverse Event Reports\n")
    f.write("  TWOSIDES:                  Drug-drug interaction database\n")
    f.write("  SIDER:                     Side effect resource\n")
    f.write("  DrugCentral:               Drug targets, mechanisms, ATC\n")
    f.write("  RxNorm:                    Drug name normalization\n\n")

    f.write("MODEL ARCHITECTURE\n")
    f.write("-" * 40 + "\n")
    total_params = sum(p.numel() for p in model.parameters())
    f.write(f"  Model:                     WeightedGCN\n")
    f.write(f"  Embedding dim:             {checkpoint.get('embed_dim', 32)}\n")
    f.write(f"  GCN Layer 1:               {FEAT_DIM + 32} -> 128\n")
    f.write(f"  GCN Layer 2:               128 -> 64\n")
    f.write(f"  MLP:                       64 -> 32 -> 1\n")
    f.write(f"  Pooling:                   Global Mean\n")
    f.write(f"  Dropout:                   0.3\n")
    f.write(f"  Total parameters:          {total_params:,}\n")
    f.write(f"  Loss:                      BCEWithLogitsLoss (weighted)\n\n")

    f.write("TEST SET RESULTS\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Test samples:              {len(test_data)}\n")
    f.write(f"  AUC:                       {auc:.4f}\n")
    f.write(f"  Accuracy:                  {acc:.4f}\n")
    f.write(f"  Precision:                 {prec:.4f}\n")
    f.write(f"  Recall:                    {rec:.4f}\n")
    f.write(f"  F1 Score:                  {f1:.4f}\n")
    f.write(f"  Optimal Threshold:         {best_threshold:.4f}\n")
    f.write(f"  Confusion Matrix:\n")
    f.write(f"    TN={cm[0][0]}  FP={cm[0][1]}\n")
    f.write(f"    FN={cm[1][0]}  TP={cm[1][1]}\n\n")

    f.write("NODE FEATURES (65 dimensions)\n")
    f.write("-" * 40 + "\n")
    f.write(f"  ATC Level-1 multi-hot:     14 dims (drug classification)\n")
    f.write(f"  Target class multi-hot:    {feat_meta['n_tc']} dims (pharmacological targets)\n")
    f.write(f"  Action type multi-hot:     {feat_meta['n_at']} dims (mechanism of action)\n")
    f.write(f"  Count features:            3 dims (log-scaled ATC, gene, SE counts)\n")

print("  [OK] Saved summary_stats.txt")

# ============================================================
# EXPLAINABILITY SAMPLES
# ============================================================
print("Generating explainability case studies...")
with open(os.path.join(RESULTS_DIR, "explainability_samples.txt"), "w", encoding="utf-8") as f:
    cases_to_explain = []
    if hr_examples:
        cases_to_explain.append(("HIGH RISK CASE", hr_examples[0]))
    if lr_examples:
        cases_to_explain.append(("LOW RISK CASE", lr_examples[0]))

    for title, graph in cases_to_explain:
        graph = graph.to(device)
        scores, prob = drug_importance_gradient(graph)
        names = [get_drug_name(graph.node_id[j].item()) for j in range(graph.x.shape[0])]

        f.write("=" * 60 + "\n")
        f.write(f"CASE: {title}\n")
        f.write(f"Drugs: {len(names)}\n")
        f.write(f"Predicted Risk: {prob:.4f} ({'HIGH' if prob > 0.5 else 'LOW'})\n")
        f.write(f"Ground Truth: {'HIGH' if graph.y.item() == 1 else 'LOW'}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Per-Drug Importance (Gradient Attribution):\n")
        ranked = np.argsort(scores)[::-1]
        for rank, idx in enumerate(ranked):
            f.write(f"  #{rank+1} {names[idx]}: {scores[idx]:.4f}\n")

            # Feature breakdown
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
            drug_attr = np.abs(grads[idx] * x_vals[idx])
            top_feats = np.argsort(drug_attr)[::-1][:5]
            for fi in top_feats:
                if drug_attr[fi] > 0.001:
                    f.write(f"      -> {beautify_feature(feat_names[fi])}: {drug_attr[fi]:.4f}\n")

        f.write("\n")

print("  [OK] Saved explainability_samples.txt")

print(f"\n{'='*60}")
print(f"ALL RESULTS SAVED TO: {RESULTS_DIR}")
print(f"{'='*60}")
print("Files generated:")
for fname in os.listdir(RESULTS_DIR):
    fpath = os.path.join(RESULTS_DIR, fname)
    size = os.path.getsize(fpath)
    print(f"  {fname} ({size:,} bytes)")
