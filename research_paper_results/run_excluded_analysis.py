"""
M7 — Excluded Cases Analysis

Compares cases excluded from GNN training (0 edges in the drug interaction
graph) vs included cases (>0 edges). Reports severity rates, drug counts,
and statistical comparisons.

Saves: excluded_analysis.json
"""

import os, json, ast
import numpy as np
import pandas as pd
from scipy import stats

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

node_ids = list(nodes_df["structure_id"].astype(str))
NUM_NODES = len(node_ids)

# Build adjacency
edge_index_list, edge_weight_list = [], []
for _, row in edges_df.iterrows():
    i, j, w = int(row["source"]), int(row["target"]), float(row["weight"])
    edge_index_list.append([i, j]); edge_index_list.append([j, i])
    edge_weight_list.append(w); edge_weight_list.append(w)

adj_dict = {}
for idx in range(len(edge_index_list)):
    u, v = edge_index_list[idx]
    adj_dict.setdefault(u, set()).add(v)

# ============================================================
# CLASSIFY CASES
# ============================================================
print("Analyzing cases...")

included_cases = []
excluded_cases = []

for _, row in cases_df.iterrows():
    node_list = ast.literal_eval(row["nodes"])
    label = row["label"]
    node_indices = [int(n) for n in node_list]
    node_set = set(node_indices)
    n_drugs = len(node_indices)

    # Count edges in this case's subgraph
    n_edges = 0
    for u in node_indices:
        if u in adj_dict:
            for v in adj_dict[u]:
                if v in node_set and v > u:
                    n_edges += 1

    case_info = {
        "label": label,
        "n_drugs": n_drugs,
        "n_edges": n_edges,
    }

    if n_edges == 0:
        excluded_cases.append(case_info)
    else:
        included_cases.append(case_info)

excluded_df = pd.DataFrame(excluded_cases)
included_df = pd.DataFrame(included_cases)

print(f"\nTotal cases: {len(cases_df)}")
print(f"Included (>0 edges): {len(included_df)}")
print(f"Excluded (0 edges):  {len(excluded_df)}")

# ============================================================
# COMPUTE STATISTICS
# ============================================================
excluded_severity_rate = excluded_df["label"].mean() if len(excluded_df) > 0 else 0
included_severity_rate = included_df["label"].mean() if len(included_df) > 0 else 0

excluded_mean_drugs = excluded_df["n_drugs"].mean() if len(excluded_df) > 0 else 0
included_mean_drugs = included_df["n_drugs"].mean() if len(included_df) > 0 else 0

excluded_median_drugs = excluded_df["n_drugs"].median() if len(excluded_df) > 0 else 0
included_median_drugs = included_df["n_drugs"].median() if len(included_df) > 0 else 0

# Statistical tests
if len(excluded_df) > 0 and len(included_df) > 0:
    # Chi-squared test for severity rate difference
    contingency = np.array([
        [excluded_df["label"].sum(), len(excluded_df) - excluded_df["label"].sum()],
        [included_df["label"].sum(), len(included_df) - included_df["label"].sum()],
    ])
    chi2, p_severity = stats.chi2_contingency(contingency)[:2]

    # Mann-Whitney U test for drug count difference
    u_stat, p_drugs = stats.mannwhitneyu(
        excluded_df["n_drugs"], included_df["n_drugs"], alternative="two-sided"
    )
else:
    chi2, p_severity = None, None
    u_stat, p_drugs = None, None

# Included edge distribution
if len(included_df) > 0:
    included_mean_edges = included_df["n_edges"].mean()
    included_median_edges = included_df["n_edges"].median()
    included_max_edges = included_df["n_edges"].max()
else:
    included_mean_edges = included_median_edges = included_max_edges = 0

# ============================================================
# OUTPUT
# ============================================================
output = {
    "experiment": "Excluded Cases Analysis (M7)",
    "total_cases": len(cases_df),
    "excluded": {
        "count": len(excluded_df),
        "percentage": round(100 * len(excluded_df) / len(cases_df), 2),
        "severity_rate": round(excluded_severity_rate, 4),
        "mean_drug_count": round(excluded_mean_drugs, 2),
        "median_drug_count": round(excluded_median_drugs, 2),
        "drug_count_distribution": excluded_df["n_drugs"].describe().to_dict() if len(excluded_df) > 0 else {},
    },
    "included": {
        "count": len(included_df),
        "percentage": round(100 * len(included_df) / len(cases_df), 2),
        "severity_rate": round(included_severity_rate, 4),
        "mean_drug_count": round(included_mean_drugs, 2),
        "median_drug_count": round(included_median_drugs, 2),
        "mean_edges": round(included_mean_edges, 2),
        "median_edges": round(included_median_edges, 2),
        "max_edges": int(included_max_edges),
        "drug_count_distribution": included_df["n_drugs"].describe().to_dict() if len(included_df) > 0 else {},
    },
    "statistical_tests": {
        "severity_chi2": round(chi2, 4) if chi2 is not None else None,
        "severity_p_value": round(p_severity, 6) if p_severity is not None else None,
        "drug_count_mann_whitney_u": round(u_stat, 4) if u_stat is not None else None,
        "drug_count_p_value": round(p_drugs, 6) if p_drugs is not None else None,
    },
    "narrative": (
        f"Of {len(cases_df)} total cases, {len(excluded_df)} ({100*len(excluded_df)/len(cases_df):.1f}%) "
        f"were excluded due to having no drug-drug interaction edges in the TWOSIDES-derived graph. "
        f"Excluded cases had a severity rate of {excluded_severity_rate:.1%} vs {included_severity_rate:.1%} "
        f"for included cases. Excluded cases had a mean of {excluded_mean_drugs:.1f} drugs per case "
        f"(vs {included_mean_drugs:.1f} for included). "
        f"The severity rate difference was {'statistically significant' if p_severity is not None and p_severity < 0.05 else 'not statistically significant'} "
        f"(chi-squared p={p_severity:.4f})." if p_severity is not None else
        f"Of {len(cases_df)} total cases, {len(excluded_df)} were excluded."
    ),
}

with open(os.path.join(OUT_DIR, "excluded_analysis.json"), "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n{'='*60}")
print("EXCLUDED CASES ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"{'Metric':<30} {'Excluded':>15} {'Included':>15}")
print(f"{'-'*60}")
print(f"{'Count':<30} {len(excluded_df):>15,} {len(included_df):>15,}")
print(f"{'Severity Rate':<30} {excluded_severity_rate:>15.4f} {included_severity_rate:>15.4f}")
print(f"{'Mean Drug Count':<30} {excluded_mean_drugs:>15.2f} {included_mean_drugs:>15.2f}")
print(f"{'Median Drug Count':<30} {excluded_median_drugs:>15.1f} {included_median_drugs:>15.1f}")
if len(included_df) > 0:
    print(f"{'Mean Edges':<30} {'N/A':>15} {included_mean_edges:>15.2f}")
print(f"{'='*60}")
if p_severity is not None:
    print(f"Severity chi2 p-value: {p_severity:.6f}")
if p_drugs is not None:
    print(f"Drug count Mann-Whitney p-value: {p_drugs:.6f}")
print(f"\n{output['narrative']}")
print("\nResults saved to excluded_analysis.json")
