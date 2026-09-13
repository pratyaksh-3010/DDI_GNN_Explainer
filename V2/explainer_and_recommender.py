"""
explainer_and_recommender.py  — V2 HGT Explainer & Counterfactual Recommender
================================================================================
Pipeline:
  1. Load trained HGT model + graph data + mappings.
  2. Given a patient case-ID:
       a. Extract patient subgraph via NeighborLoader.
       b. Run HGT forward pass → severity score + top-K side effects.
       c. Read TRUE attention weights from HGTConv hooks (last conv layer).
       d. Aggregate attention per prescribed drug → risk hierarchy.
  3. Counterfactual recommendation:
       a. Identify the highest-risk drug.
       b. Find ATC-class alternatives.
       c. For each alternative: clone the subgraph, swap the drug edge, re-run model.
       d. Recommend the alternative with the lowest predicted severity.
"""

import copy
import json
import ast
import random
import os

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from hetero_attention_model import ExplainableHeteroGNN

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED = r"V2\Processed"
NUM_SIDE_EFFECTS = 1000

print("=" * 60)
print("  V2 HGT Explainer & Counterfactual Recommender")
print("=" * 60)

# ── 1. Load Data & Mappings ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

print("Loading graph data...")
data = torch.load(os.path.join(PROCESSED, "hetero_graph_data.pt"), weights_only=False)

# ── Mirror the training graph setup exactly ───────────────────────────────────
# 1) Add reverse edges (same as train_hetero.py)
from torch_geometric.utils import degree as pyg_degree

def _add_reverse_edges(d):
    d["drug", "prescribed_by", "patient"].edge_index = \
        d["patient", "prescribed", "drug"].edge_index.flip(0)
    d["protein", "targeted_by", "drug"].edge_index = \
        d["drug", "targets", "protein"].edge_index.flip(0)
    d["side_effect", "caused_by", "drug"].edge_index = \
        d["drug", "causes", "side_effect"].edge_index.flip(0)
    return d

data = _add_reverse_edges(data)

# 2) Replace all-ones patient.x with normalised degree feature
num_patients = data["patient"].num_nodes
pd_edge = data["patient", "prescribed", "drug"].edge_index
deg = pyg_degree(pd_edge[0], num_nodes=num_patients).float()
data["patient"].x = (deg / (deg.max() + 1e-8)).unsqueeze(1)

data = data.to(device)
print(f"  Graph ready. Edge types: {len(data.edge_types)}")



print("Loading node mappings...")
with open(os.path.join(PROCESSED, "node_mappings.json"), "r") as f:
    mappings = json.load(f)

drug_map     = mappings["drug"]      # rxcui  → graph idx
inv_drug_map = {v: k for k, v in drug_map.items()}
patient_map  = mappings["patient"]   # caseid → graph idx
se_map       = mappings.get("side_effect", {})
inv_se_map   = {v: k for k, v in se_map.items()}

# ── 2. Load Model ─────────────────────────────────────────────────────────────
print("Loading model...")
cfg_path = os.path.join(PROCESSED, "training_config.json")
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        CFG = json.load(f)
    print(f"  Loaded config: hidden={CFG['hidden_channels']}, heads={CFG['num_heads']}, layers={CFG['num_layers']}")
else:
    CFG = {"hidden_channels": 128, "num_heads": 4, "num_layers": 2, "num_side_effects": 1000}
    print("  WARNING: training_config.json not found, using defaults.")

model = ExplainableHeteroGNN(
    hidden_channels=CFG["hidden_channels"],
    num_heads=CFG["num_heads"],
    num_layers=CFG["num_layers"],
    node_types=data.node_types,
    metadata=data.metadata(),
    num_side_effects=CFG.get("num_side_effects", NUM_SIDE_EFFECTS),
).to(device)

model_path = os.path.join(PROCESSED, "hgt_model.pth")
try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"  Loaded weights from {model_path}")
except FileNotFoundError:
    print(f"  WARNING: {model_path} not found. Using untrained weights (run train_hetero.py first).")
except RuntimeError as e:
    print(f"  WARNING: Could not load weights ({e}). Using untrained weights.")

model.eval()

# ── 3. Load Top-1000 SE Summary ───────────────────────────────────────────────
se_summary_path = r"V2\top_1000_side_effects_summary.json"
se_names = {}  # idx → human-readable name
if os.path.exists(se_summary_path):
    with open(se_summary_path) as f:
        raw = json.load(f)
    # Build idx → name from the se_map
    for name, idx in se_map.items():
        se_names[idx] = name

# ── 4. ATC Class Lookup ───────────────────────────────────────────────────────
print("\nBuilding ATC class lookup...")
atc_dict: dict[str, set] = {}   # rxcui → {atc_level1, ...}

try:
    faers = pd.read_csv(r"DATA_AGGREGATION_CODE\faers_layer3_final.csv", dtype=str)
    struct_map = pd.read_csv("rxcui_structure_map.csv", dtype=str)
    struct_to_rxcui = dict(zip(struct_map["structure_id"], struct_map["rxcui"]))

    for _, row in faers.iterrows():
        try:
            structs = ast.literal_eval(row["STRUCTURE_ID_LIST"])
            rxcuis  = [struct_to_rxcui[str(s)] for s in structs if str(s) in struct_to_rxcui]
            atcs    = ast.literal_eval(row["ATC_LEVEL1"])
            for rxcui, atc in zip(rxcuis, atcs):
                atc_dict.setdefault(rxcui, set()).add(atc)
        except Exception:
            pass

    print(f"  ATC lookup built for {len(atc_dict):,} drugs.")
except Exception as e:
    print(f"  WARNING: Could not build ATC lookup ({e}). Counterfactual will be limited.")

# Group rxcuis by ATC class for fast candidate lookup
atc_to_rxcuis: dict[str, list] = {}
for rxcui, atc_set in atc_dict.items():
    for atc in atc_set:
        atc_to_rxcuis.setdefault(atc, []).append(rxcui)


# ── 5. Helper: Run Single Patient Subgraph Through Model ──────────────────────
def _run_patient(p_idx: int, full_data: HeteroData):
    """
    Extract a 2-hop subgraph centred on patient p_idx and run the HGT.
    Returns (sev_prob, top_se_indices, batch, sev_logit, attention_dict)
    """
    loader = NeighborLoader(
        full_data,
        num_neighbors=[15, 15],
        batch_size=1,
        input_nodes=("patient", torch.tensor([p_idx], dtype=torch.long)),
    )
    batch = next(iter(loader)).to(device)

    with torch.no_grad():
        sev_logits, se_logits, _ = model(batch.x_dict, batch.edge_index_dict)

    sev_prob = torch.sigmoid(sev_logits[0]).item()

    # Top-10 side effects
    se_probs_all = torch.sigmoid(se_logits[0])           # (1000,)
    top_se_vals, top_se_idx = se_probs_all.topk(10)

    return sev_prob, top_se_idx.tolist(), top_se_vals.tolist(), batch


# ── 6. Explainer ─────────────────────────────────────────────────────────────
def explain_patient_risk(caseid_str: str):
    if caseid_str not in patient_map:
        print(f"  Patient '{caseid_str}' not found in graph.")
        return None

    p_idx = patient_map[caseid_str]
    sev_prob, top_se_idx, top_se_vals, batch = _run_patient(p_idx, data)

    print(f"\n{'='*60}")
    print(f"  CASE: {caseid_str}")
    print(f"  Predicted Severity Risk Score : {sev_prob:.4f}")
    print(f"  Risk Level : ", end="")
    if sev_prob > 0.7:
        print("HIGH ⚠️")
    elif sev_prob > 0.4:
        print("MODERATE")
    else:
        print("LOW ✅")

    # ── Top-K Side Effects ────────────────────────────────────────────────────
    print(f"\n  Top-10 Predicted Side Effects:")
    for rank, (idx, prob) in enumerate(zip(top_se_idx, top_se_vals), 1):
        name = se_names.get(idx, f"SE_{idx}")
        print(f"    {rank:>2}. {name:<40} p={prob:.3f}")

    # ── Drug Risk Hierarchy via Attention Weights ─────────────────────────────
    print("\n  --- RISK HIERARCHY (True HGT Attention) ---")

    # Get edges: which drugs prescribed this patient?
    # NeighborLoader samples INCOMING edges for the seed node, so we look at
    # the REVERSE edge (drug, prescribed_by, patient) to find the patient's drugs.
    pd_edges_fwd = None
    patient_drug_batch_indices = []

    try:
        # Try the reverse edge first (most reliable in NeighborLoader batches)
        rev_edges = batch["drug", "prescribed_by", "patient"].edge_index  # (2, E)
        # patient seed is always at local index 0
        for i in range(rev_edges.shape[1]):
            if rev_edges[1, i].item() == 0:
                patient_drug_batch_indices.append(rev_edges[0, i].item())
        pd_edges_fwd = rev_edges.flip(0)  # reuse as (patient, drug) for attention matching
    except (KeyError, AttributeError):
        pass

    if not patient_drug_batch_indices:
        # Fallback: try the forward edge
        try:
            pd_edges_fwd = batch["patient", "prescribed", "drug"].edge_index
            for i in range(pd_edges_fwd.shape[1]):
                if pd_edges_fwd[0, i].item() == 0:
                    patient_drug_batch_indices.append(pd_edges_fwd[1, i].item())
        except (KeyError, AttributeError):
            pass

    # Map batch indices → global rxcui
    patient_drugs = []
    for d_batch_idx in patient_drug_batch_indices:
        if d_batch_idx < batch["drug"].n_id.shape[0]:
            d_global_idx = batch["drug"].n_id[d_batch_idx].item()
            rxcui = inv_drug_map.get(d_global_idx, f"UNK_{d_global_idx}")
            patient_drugs.append((d_batch_idx, rxcui))

    print(f"  Prescribed drugs: {[r for _, r in patient_drugs]}")

    # Retrieve attention weights from the last HGT layer (most semantically refined)
    attn_scores: dict[str, float] = {}   # rxcui → aggregated attention

    last_layer_idx = max(model.attention_dict.keys()) if model.attention_dict else None
    if last_layer_idx is not None:
        attn_layer = model.attention_dict[last_layer_idx]

        # Look for patient→drug relations (any rel name containing 'prescribed' or 'drug')
        for rel_key, alpha in attn_layer.items():
            # rel_key is a tuple like ('patient', 'prescribed', 'drug') or a string
            rel_str = str(rel_key)
            if "prescribed" in rel_str or ("patient" in rel_str and "drug" in rel_str):
                # alpha shape: (E,) or (E, num_heads) — average over heads if needed
                if alpha.dim() == 2:
                    alpha = alpha.mean(dim=1)          # (E,)
                # Match edge indices from batch
                if pd_edges_fwd is not None and alpha.shape[0] == pd_edges_fwd.shape[1]:
                    for i, (d_batch_idx, rxcui) in enumerate(patient_drugs):
                        # find the edge(s) for this drug
                        mask = (pd_edges_fwd[0] == 0) & (pd_edges_fwd[1] == d_batch_idx)
                        if mask.any():
                            score = alpha[mask].mean().item()
                            attn_scores[rxcui] = score

    # Fallback: embedding dot-product similarity if no attention was captured
    if not attn_scores:
        _, _, x_dict = model(batch.x_dict, batch.edge_index_dict)
        p_embed = x_dict["patient"][0]
        for d_batch_idx, rxcui in patient_drugs:
            if d_batch_idx < x_dict["drug"].shape[0]:
                d_embed = x_dict["drug"][d_batch_idx]
                attn_scores[rxcui] = torch.dot(p_embed, d_embed).item()

    # Sort drugs by risk attribution
    drug_scores = sorted(attn_scores.items(), key=lambda x: x[1], reverse=True)

    for rank, (rxcui, score) in enumerate(drug_scores, 1):
        atcs = list(atc_dict.get(rxcui, {"Unknown"}))
        print(f"    #{rank} RxCUI: {rxcui:<8} | Attention Score: {score:>7.4f} | ATC: {atcs}")

    return drug_scores, sev_prob, [r for _, r in patient_drugs]


# ── 7. Counterfactual Recommender ────────────────────────────────────────────
def recommend_safer_combination(caseid_str: str, drug_scores: list, patient_drugs: list, current_risk: float):
    print(f"\n  --- COUNTERFACTUAL SIMULATION ---")
    if not drug_scores:
        print("  No drug scores available.")
        return

    highest_risk_rxcui = drug_scores[0][0]
    print(f"  Targeting highest-risk drug for substitution: {highest_risk_rxcui}")
    print(f"  Current Risk Score: {current_risk:.4f}")

    # Get ATC candidates
    target_atcs = atc_dict.get(highest_risk_rxcui, set())
    if not target_atcs:
        print("  No ATC class found — cannot suggest therapeutic alternative.")
        return

    candidates = set()
    for atc in target_atcs:
        candidates.update(atc_to_rxcuis.get(atc, []))
    candidates.discard(highest_risk_rxcui)
    candidates = [c for c in candidates if c in drug_map]  # must exist in graph

    if not candidates:
        print("  No alternative drugs found in same ATC class.")
        return

    print(f"  Found {len(candidates)} ATC-class alternatives. Simulating top-20...")

    p_idx = patient_map[caseid_str]

    # ── Build the baseline batch (subgraph for this patient) ──────────────────
    base_loader = NeighborLoader(
        data,
        num_neighbors=[15, 15],
        batch_size=1,
        input_nodes=("patient", torch.tensor([p_idx], dtype=torch.long)),
    )
    base_batch = next(iter(base_loader)).to(device)

    # Identify the edge index to remove (patient → highest-risk drug)
    pd_edges = base_batch.get("patient", "prescribed", "drug") if False else None
    try:
        pd_edges = base_batch["patient", "prescribed", "drug"].edge_index
    except KeyError:
        pd_edges = None

    high_risk_graph_idx = drug_map.get(highest_risk_rxcui)
    high_risk_batch_idx = None
    if pd_edges is not None and high_risk_graph_idx is not None:
        global_ids = base_batch["drug"].n_id.tolist()
        if high_risk_graph_idx in global_ids:
            high_risk_batch_idx = global_ids.index(high_risk_graph_idx)

    best_alternative = None
    best_risk = current_risk
    results = []

    for alt_rxcui in list(candidates)[:20]:
        alt_graph_idx = drug_map.get(alt_rxcui)
        if alt_graph_idx is None:
            continue

        # ── Clone batch and swap the edge ────────────────────────────────────
        cf_batch = copy.deepcopy(base_batch)

        if pd_edges is not None and high_risk_batch_idx is not None:
            cf_edge = cf_batch["patient", "prescribed", "drug"].edge_index.clone()

            # Remove old edge (patient→high_risk_drug)
            mask = ~((cf_edge[0] == 0) & (cf_edge[1] == high_risk_batch_idx))
            cf_edge = cf_edge[:, mask]

            # We need the alternative drug to exist in the batch's drug n_id list.
            # If it doesn't, add a new feature row for it from the full graph.
            alt_global_ids = cf_batch["drug"].n_id.tolist()
            if alt_graph_idx not in alt_global_ids:
                # Append alternative drug node features from full graph
                alt_feats = data["drug"].x[alt_graph_idx].unsqueeze(0).to(device)
                cf_batch["drug"].x = torch.cat([cf_batch["drug"].x, alt_feats], dim=0)
                cf_batch["drug"].n_id = torch.cat(
                    [cf_batch["drug"].n_id, torch.tensor([alt_graph_idx], device=device)]
                )
                new_batch_idx = cf_batch["drug"].x.shape[0] - 1
            else:
                new_batch_idx = alt_global_ids.index(alt_graph_idx)

            # Add new edge (patient→alternative drug)
            new_edge = torch.tensor([[0], [new_batch_idx]], dtype=torch.long, device=device)
            cf_edge = torch.cat([cf_edge, new_edge], dim=1)
            cf_batch["patient", "prescribed", "drug"].edge_index = cf_edge

        # ── Run counterfactual forward pass ───────────────────────────────────
        with torch.no_grad():
            try:
                sev_logits_cf, _, _ = model(cf_batch.x_dict, cf_batch.edge_index_dict)
                cf_risk = torch.sigmoid(sev_logits_cf[0]).item()
            except Exception as e:
                cf_risk = current_risk  # fallback if graph is malformed

        delta = current_risk - cf_risk
        results.append((alt_rxcui, cf_risk, delta))

        if cf_risk < best_risk:
            best_risk = cf_risk
            best_alternative = alt_rxcui

    # ── Report Results ────────────────────────────────────────────────────────
    results.sort(key=lambda x: x[1])
    print(f"\n  {'Alternative RxCUI':<20} | {'New Risk':>8} | {'Delta Risk':>10}")
    print(f"  {'-'*44}")
    for alt, risk, delta in results[:10]:
        indicator = " <- BEST" if alt == best_alternative else ""
        print(f"  {alt:<20} | {risk:>8.4f} | {delta:>+10.4f}{indicator}")

    if best_alternative:
        atcs_orig = list(atc_dict.get(highest_risk_rxcui, {"?"}))
        atcs_alt  = list(atc_dict.get(best_alternative,   {"?"}))
        print(f"\n  +==================================================+")
        print(f"  |  RECOMMENDATION                                  |")
        print(f"  |  Replace : {highest_risk_rxcui:<38}|")
        print(f"  |  With    : {best_alternative:<38}|")
        print(f"  |  Original risk  : {current_risk:.4f}                       |")
        print(f"  |  Projected risk : {best_risk:.4f}                       |")
        print(f"  |  ATC classes    : {str(atcs_orig):<31}|")
        print(f"  +==================================================+")
    else:
        print("  No safer alternative found among the top-20 candidates.")

    return best_alternative, best_risk


# -- 8. CLI Entry Point --------------------------------------------------------
if __name__ == "__main__":
    test_caseid = list(patient_map.keys())[0]
    print(f"\nDemo: running on patient '{test_caseid}'")

    result = explain_patient_risk(test_caseid)
    if result:
        drug_scores, orig_risk, prescribed = result
        recommend_safer_combination(test_caseid, drug_scores, prescribed, orig_risk)
