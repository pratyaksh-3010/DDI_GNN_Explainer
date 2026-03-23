import pandas as pd
import numpy as np
import ast
import json

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
faers = pd.read_csv("faers_layer2_final.csv")
twosides = pd.read_csv("twosides_filtered.csv", dtype=str)
mapping = pd.read_csv("rxcui_structure_map.csv", dtype=str)
atc_map = pd.read_csv("structure_atc_map.csv", dtype=str)
mech_map = pd.read_csv("structure_mechanism_map.csv", dtype=str)

rx_to_struct = dict(zip(mapping["rxcui"], mapping["structure_id"]))

# ============================================================
# STEP 1: Collect ALL unique structure IDs
# ============================================================
all_structs = set()

for row in faers["STRUCTURE_ID_LIST"]:
    try:
        sids = ast.literal_eval(row)
        all_structs.update([str(s) for s in sids])
    except:
        continue

print(f"Total drug nodes from FAERS: {len(all_structs)}")

# Create ordered list and index mapping
node_ids = sorted(list(all_structs))
node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

# ============================================================
# STEP 2: Build RICH node features per drug
# ============================================================
print("Building node features...")

# --- ATC Level-1 multi-hot (14 categories: A-V) ---
ATC_L1_CATS = list("ABCDGHJLMNPRSV")
atc_l1_idx = {c: i for i, c in enumerate(ATC_L1_CATS)}

struct_to_atc_l1 = {}
for _, row in atc_map.iterrows():
    sid = str(row["structure_id"])
    code = str(row.get("atc_code", ""))
    if len(code) >= 1:
        l1 = code[0].upper()
        if sid not in struct_to_atc_l1:
            struct_to_atc_l1[sid] = set()
        struct_to_atc_l1[sid].add(l1)

# --- Target class multi-hot ---
TARGET_CLASSES = sorted(mech_map["target_class"].dropna().unique().tolist())
tc_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}

struct_to_tc = {}
for _, row in mech_map.iterrows():
    sid = str(row["struct_id"])
    tc = row.get("target_class")
    if pd.notna(tc):
        if sid not in struct_to_tc:
            struct_to_tc[sid] = set()
        struct_to_tc[sid].add(tc)

# --- Action type multi-hot ---
ACTION_TYPES = sorted(mech_map["action_type"].dropna().unique().tolist())
at_idx = {a: i for i, a in enumerate(ACTION_TYPES)}

struct_to_at = {}
for _, row in mech_map.iterrows():
    sid = str(row["struct_id"])
    at = row.get("action_type")
    if pd.notna(at):
        if sid not in struct_to_at:
            struct_to_at[sid] = set()
        struct_to_at[sid].add(at)

# --- Count features from FAERS aggregation ---
struct_counts = {}  # sid -> {atc_count, gene_count, se_count}

for _, row in faers.iterrows():
    try:
        sids = ast.literal_eval(row["STRUCTURE_ID_LIST"])
        atc = ast.literal_eval(row["ATC_CODES"]) if pd.notna(row["ATC_CODES"]) else []
        genes = ast.literal_eval(row["TARGET_GENES"]) if pd.notna(row["TARGET_GENES"]) else []
        se = ast.literal_eval(row["KNOWN_SIDE_EFFECTS"]) if pd.notna(row["KNOWN_SIDE_EFFECTS"]) else []
    except:
        continue

    for sid in sids:
        sid = str(sid)
        if sid not in struct_counts:
            struct_counts[sid] = {"atc": set(), "genes": set(), "se": set()}
        struct_counts[sid]["atc"].update(atc)
        struct_counts[sid]["genes"].update(genes)
        struct_counts[sid]["se"].update(se)

# --- Assemble feature matrix ---
# Feature layout:
#   [0:14]   ATC L1 multi-hot (14)
#   [14:14+T] target_class multi-hot (T)
#   [14+T:14+T+A] action_type multi-hot (A)
#   [last 3]  count features (atc_count, gene_count, se_count)

n_atc = len(ATC_L1_CATS)
n_tc = len(TARGET_CLASSES)
n_at = len(ACTION_TYPES)
n_counts = 3
FEAT_DIM = n_atc + n_tc + n_at + n_counts

print(f"Feature dimensions: ATC_L1={n_atc}, TargetClass={n_tc}, ActionType={n_at}, Counts=3")
print(f"Total feature dimension: {FEAT_DIM}")

X = np.zeros((len(node_ids), FEAT_DIM), dtype=np.float32)

for i, nid in enumerate(node_ids):
    # ATC L1 multi-hot
    if nid in struct_to_atc_l1:
        for cat in struct_to_atc_l1[nid]:
            if cat in atc_l1_idx:
                X[i, atc_l1_idx[cat]] = 1.0

    # Target class multi-hot
    offset_tc = n_atc
    if nid in struct_to_tc:
        for tc in struct_to_tc[nid]:
            if tc in tc_idx:
                X[i, offset_tc + tc_idx[tc]] = 1.0

    # Action type multi-hot
    offset_at = n_atc + n_tc
    if nid in struct_to_at:
        for at in struct_to_at[nid]:
            if at in at_idx:
                X[i, offset_at + at_idx[at]] = 1.0

    # Count features (log-scaled)
    offset_cnt = n_atc + n_tc + n_at
    data = struct_counts.get(nid, {"atc": [], "genes": [], "se": []})
    X[i, offset_cnt + 0] = np.log1p(len(data["atc"]))
    X[i, offset_cnt + 1] = np.log1p(len(data["genes"]))
    X[i, offset_cnt + 2] = np.log1p(len(data["se"]))

print(f"Node feature matrix shape: {X.shape}")

# ============================================================
# STEP 3: Save nodes CSV
# ============================================================
# Build feature column names for interpretability
feat_names = []
feat_names += [f"atc_l1_{c}" for c in ATC_L1_CATS]
feat_names += [f"tc_{c}" for c in TARGET_CLASSES]
feat_names += [f"at_{a}" for a in ACTION_TYPES]
feat_names += ["log_atc_count", "log_gene_count", "log_se_count"]

nodes_df = pd.DataFrame(X, columns=feat_names)
nodes_df.insert(0, "structure_id", node_ids)
# Also save the raw counts for readability
for i, nid in enumerate(node_ids):
    data = struct_counts.get(nid, {"atc": [], "genes": [], "se": []})
    nodes_df.loc[i, "atc_count"] = len(data["atc"])
    nodes_df.loc[i, "gene_count"] = len(data["genes"])
    nodes_df.loc[i, "side_effect_count"] = len(data["se"])

nodes_df.to_csv("layer3_nodes.csv", index=False)
print(f"Nodes CSV saved. Total nodes: {len(nodes_df)}")

# Save feature metadata for downstream use
feature_meta = {
    "feat_dim": FEAT_DIM,
    "feature_names": feat_names,
    "atc_l1_cats": ATC_L1_CATS,
    "target_classes": TARGET_CLASSES,
    "action_types": ACTION_TYPES,
    "n_atc": n_atc,
    "n_tc": n_tc,
    "n_at": n_at,
}
with open("layer3_feature_meta.json", "w") as f:
    json.dump(feature_meta, f, indent=2)
print("Feature metadata saved to layer3_feature_meta.json")

# ============================================================
# STEP 4: Build weighted edges from TWOSIDES
# ============================================================
print("Building edges from TWOSIDES...")
twosides["PRR"] = pd.to_numeric(twosides["PRR"], errors="coerce")
twosides = twosides.dropna(subset=["PRR"])

pairs = {}  # (sorted pair of node_idx) -> max PRR

for _, row in twosides.iterrows():
    r1 = str(row["drug1_rxcui"])
    r2 = str(row["drug2_rxcui"])

    if r1 in rx_to_struct and r2 in rx_to_struct:
        s1 = rx_to_struct[r1]
        s2 = rx_to_struct[r2]

        if s1 in node_to_idx and s2 in node_to_idx and s1 != s2:
            idx1 = node_to_idx[s1]
            idx2 = node_to_idx[s2]
            pair = (min(idx1, idx2), max(idx1, idx2))

            prr = row["PRR"]
            if pair not in pairs or prr > pairs[pair]:
                pairs[pair] = prr

# Create edges DataFrame with NODE INDICES (not structure_ids!)
edge_rows = []
for (i, j), prr in pairs.items():
    edge_rows.append({
        "source": i,
        "target": j,
        "weight": np.log1p(prr)
    })

edges_df = pd.DataFrame(edge_rows)
edges_df.to_csv("layer3_edges_weighted.csv", index=False)
print(f"Edges CSV saved. Total edges: {len(edges_df)}")

# ============================================================
# STEP 5: Build training cases from FAERS
# ============================================================
print("Building training cases...")

# Precompute adjacency dictionary (using node indices)
adj_dict = {}
for _, row in edges_df.iterrows():
    u = int(row["source"])
    v = int(row["target"])
    w = float(row["weight"])
    if u not in adj_dict:
        adj_dict[u] = []
    adj_dict[u].append((v, w))
    if v not in adj_dict:
        adj_dict[v] = []
    adj_dict[v].append((u, w))

case_data = []

for _, row in faers.iterrows():
    if pd.isna(row["SEVERITY"]):
        continue

    try:
        sids = ast.literal_eval(row["STRUCTURE_ID_LIST"])
    except:
        continue

    label = int(row["SEVERITY"])
    nodes = [node_to_idx[str(sid)] for sid in sids if str(sid) in node_to_idx]

    if len(nodes) >= 2:
        case_data.append({
            "nodes": nodes,
            "label": label
        })

cases_df = pd.DataFrame(case_data)
cases_df.to_csv("layer3_training_cases.csv", index=False)
print(f"Training cases saved. Total cases: {len(cases_df)}")

# ============================================================
# STATS
# ============================================================
if len(cases_df) > 0:
    high = cases_df["label"].sum()
    low = len(cases_df) - high
    print(f"\nClass distribution:")
    print(f"  High risk: {high} ({100*high/len(cases_df):.1f}%)")
    print(f"  Low risk:  {low} ({100*low/len(cases_df):.1f}%)")
print("\n=== Layer 3 complete ===")
