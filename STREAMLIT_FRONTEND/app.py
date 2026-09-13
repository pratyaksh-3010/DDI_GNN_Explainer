"""
app.py — PolyGuard: Drug Combination Analyser
==============================================
Enter drug names -> get predicted side effects + safer replacements.
Powered by the V2 HGT model trained on 89k FAERS adverse event reports.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "V2"))

import copy, json, torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree as pyg_degree

from hetero_attention_model import ExplainableHeteroGNN

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolyGuard — Drug Combination Analyser",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#0a0e1a; color:#e2e8f0; }
.stApp { background:#0a0e1a; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a 0%,#1a1f35 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
section[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
section[data-testid="stSidebar"] .stMultiSelect > div { background:#1e293b !important; }

.metric-card {
    background:rgba(15,23,42,0.8); border:1px solid rgba(99,102,241,0.3);
    border-radius:16px; padding:18px 22px; backdrop-filter:blur(12px);
    box-shadow:0 8px 32px rgba(0,0,0,0.4); margin-bottom:12px;
}
.metric-label { font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:.1em; margin-bottom:4px; }
.metric-value { font-size:28px; font-weight:700; color:#e2e8f0; line-height:1.1; }
.metric-sub   { font-size:12px; color:#64748b; margin-top:4px; }

.badge-high     { background:rgba(239,68,68,.15);  border:1px solid rgba(239,68,68,.4);  color:#f87171; padding:3px 14px; border-radius:999px; font-size:13px; font-weight:600; }
.badge-moderate { background:rgba(251,146,60,.15); border:1px solid rgba(251,146,60,.4); color:#fb923c; padding:3px 14px; border-radius:999px; font-size:13px; font-weight:600; }
.badge-low      { background:rgba(34,197,94,.15);  border:1px solid rgba(34,197,94,.4);  color:#4ade80; padding:3px 14px; border-radius:999px; font-size:13px; font-weight:600; }

.section-header { font-size:17px; font-weight:600; color:#e2e8f0; border-left:3px solid #6366f1; padding-left:12px; margin:28px 0 14px 0; }

.drug-chip {
    display:inline-block; background:rgba(99,102,241,.15);
    border:1px solid rgba(99,102,241,.35); border-radius:999px;
    padding:3px 12px; font-size:12px; color:#a5b4fc; margin:3px 4px;
}
.se-chip {
    display:inline-block; background:rgba(239,68,68,.10);
    border:1px solid rgba(239,68,68,.25); border-radius:999px;
    padding:3px 12px; font-size:12px; color:#fca5a5; margin:3px 4px;
}
.rec-box {
    background:linear-gradient(135deg,rgba(16,185,129,.08),rgba(99,102,241,.08));
    border:1px solid rgba(16,185,129,.35); border-radius:16px; padding:20px 24px; margin-top:14px;
}
.rec-title { font-size:15px; font-weight:700; color:#34d399; margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "V2", "Processed")


# ── Load everything (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading graph and model …")
def load_everything():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(os.path.join(PROCESSED, "hetero_graph_data.pt"), weights_only=False)

    # Mirror training: reverse edges + degree patient features
    data["drug", "prescribed_by", "patient"].edge_index = \
        data["patient", "prescribed", "drug"].edge_index.flip(0)
    data["protein", "targeted_by", "drug"].edge_index = \
        data["drug", "targets", "protein"].edge_index.flip(0)
    data["side_effect", "caused_by", "drug"].edge_index = \
        data["drug", "causes", "side_effect"].edge_index.flip(0)

    pd_edge = data["patient", "prescribed", "drug"].edge_index
    deg     = pyg_degree(pd_edge[0], num_nodes=data["patient"].num_nodes).float()
    data["patient"].x = (deg / (deg.max() + 1e-8)).unsqueeze(1)
    data = data.to(device)

    # Mappings
    with open(os.path.join(PROCESSED, "node_mappings.json")) as f:
        maps = json.load(f)

    # Model
    with open(os.path.join(PROCESSED, "training_config.json")) as f:
        cfg = json.load(f)
    model = ExplainableHeteroGNN(
        hidden_channels=cfg["hidden_channels"], num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"], node_types=data.node_types,
        metadata=data.metadata(), num_side_effects=cfg.get("num_side_effects", 1000),
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(PROCESSED, "hgt_model.pth"),
                   map_location=device, weights_only=True)
    )
    model.eval()

    # Drug names: rxcui -> display name
    drug_names = {}
    name_file = os.path.join(ROOT, "drug_names.tsv")
    if os.path.exists(name_file):
        df_n = pd.read_csv(name_file, sep="\t", header=None,
                           names=["rxcui", "name"], dtype=str)
        drug_names = dict(zip(df_n["rxcui"].str.strip(), df_n["name"].str.strip()))

    # SE names: idx -> name
    se_names = {v: k for k, v in maps.get("side_effect", {}).items()}

    # ATC lookup
    atc_dict = {}
    try:
        import ast
        faers     = pd.read_csv(os.path.join(ROOT, "DATA_AGGREGATION_CODE",
                                              "faers_layer3_final.csv"), dtype=str)
        struct_map = pd.read_csv(os.path.join(ROOT, "rxcui_structure_map.csv"), dtype=str)
        s2r = dict(zip(struct_map["structure_id"], struct_map["rxcui"]))
        for _, row in faers.iterrows():
            try:
                structs = ast.literal_eval(row["STRUCTURE_ID_LIST"])
                rxcuis  = [s2r[str(s)] for s in structs if str(s) in s2r]
                atcs    = ast.literal_eval(row["ATC_LEVEL1"])
                for rc, atc in zip(rxcuis, atcs):
                    atc_dict.setdefault(rc, set()).add(atc)
            except Exception:
                pass
    except Exception:
        pass

    atc_to_rxcuis = {}
    for rc, atc_set in atc_dict.items():
        for atc in atc_set:
            atc_to_rxcuis.setdefault(atc, []).append(rc)

    # Build: drug_gidx -> [se_gidx, ...] from graph edges
    ds_edge  = data["drug", "causes", "side_effect"].edge_index.cpu()
    drug_to_se = {}
    for i in range(ds_edge.shape[1]):
        d, s = ds_edge[0, i].item(), ds_edge[1, i].item()
        drug_to_se.setdefault(d, []).append(s)

    # Build patient -> [drug_gidx] lookup (on CPU for speed)
    pd_cpu    = data["patient", "prescribed", "drug"].edge_index.cpu()
    pat_drugs = {}  # patient_gidx -> set(drug_gidx)
    for i in range(pd_cpu.shape[1]):
        p, d = pd_cpu[0, i].item(), pd_cpu[1, i].item()
        pat_drugs.setdefault(p, set()).add(d)

    return {
        "device": device, "data": data, "model": model,
        "patient_map": maps["patient"],
        "drug_map": maps["drug"],
        "inv_drug_map": {v: k for k, v in maps["drug"].items()},
        "drug_names": drug_names,
        "se_names": se_names,
        "atc_dict": atc_dict,
        "atc_to_rxcuis": atc_to_rxcuis,
        "drug_to_se": drug_to_se,
        "pat_drugs": pat_drugs,
        "ddi_edge": data["drug", "interacts_with", "drug"].edge_index.cpu(),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def label(rxcui, drug_names):
    n = drug_names.get(str(rxcui), "")
    return f"{n.title()} ({rxcui})" if n else f"RxCUI {rxcui}"


def get_graph_se(drug_rxcuis: list, res: dict, top_k: int = 20):
    """
    Aggregate side effects from the graph edges for the given drug RxCUIs.
    Returns list of (se_name, drug_count) sorted by how many drugs cause it.
    """
    drug_map  = res["drug_map"]
    drug_to_se= res["drug_to_se"]
    se_names  = res["se_names"]

    se_counter = {}
    causing_drugs = {}  # se_idx -> list of rxcui
    for rxcui in drug_rxcuis:
        gidx = drug_map.get(str(rxcui))
        if gidx is None:
            continue
        for se_idx in drug_to_se.get(gidx, []):
            se_counter[se_idx] = se_counter.get(se_idx, 0) + 1
            causing_drugs.setdefault(se_idx, []).append(rxcui)

    top = sorted(se_counter.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"se": se_names.get(si, f"SE_{si}"), "count": cnt,
         "drugs": causing_drugs.get(si, [])}
        for si, cnt in top
    ]


def find_matching_patient(drug_rxcuis: list, res: dict):
    """Find the real patient with maximum overlap with the queried drugs."""
    drug_map  = res["drug_map"]
    pat_drugs = res["pat_drugs"]

    query_gidx = {drug_map[r] for r in drug_rxcuis if r in drug_map}
    if not query_gidx:
        return None

    best_p, best_score = None, -1
    for p, d_set in pat_drugs.items():
        score = len(d_set & query_gidx)
        if score > best_score:
            best_score, best_p = score, p
    return best_p  # graph index of patient


def run_hgt_inference(patient_gidx: int, res: dict):
    """Run HGT on a patient by graph index. Returns (sev_prob, se_probs)."""
    device = res["device"]
    data   = res["data"]
    model  = res["model"]

    loader = NeighborLoader(
        data, num_neighbors=[20, 10], batch_size=1,
        input_nodes=("patient", torch.tensor([patient_gidx], dtype=torch.long)),
        num_workers=0,
    )
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        sev_l, se_l, _ = model(batch.x_dict, batch.edge_index_dict)
    return torch.sigmoid(sev_l[0]).item(), torch.sigmoid(se_l[0]).cpu()


def run_counterfactual(drug_to_replace: str, all_drugs: list, patient_gidx: int,
                       current_risk: float, n_candidates: int, res: dict):
    device        = res["device"]
    data          = res["data"]
    model         = res["model"]
    drug_map      = res["drug_map"]
    atc_dict      = res["atc_dict"]
    atc_to_rxcuis = res["atc_to_rxcuis"]

    target_atcs = atc_dict.get(drug_to_replace, set())
    candidates  = set()
    for atc in target_atcs:
        candidates.update(atc_to_rxcuis.get(atc, []))
    candidates.discard(drug_to_replace)
    candidates = [c for c in candidates if c in drug_map and c not in all_drugs]

    if not candidates:
        return [], None, 0

    loader = NeighborLoader(
        data, num_neighbors=[20, 10], batch_size=1,
        input_nodes=("patient", torch.tensor([patient_gidx], dtype=torch.long)),
        num_workers=0,
    )
    base_batch = next(iter(loader)).to(device)

    old_gidx  = drug_map.get(drug_to_replace)
    old_gids  = base_batch["drug"].n_id.tolist()
    old_bidx  = old_gids.index(old_gidx) if old_gidx in old_gids else None

    results = []
    for alt_rxcui in list(candidates)[:n_candidates]:
        alt_gidx = drug_map[alt_rxcui]
        cf       = copy.deepcopy(base_batch)
        try:
            rev_e = cf["drug", "prescribed_by", "patient"].edge_index.clone()
            if old_bidx is not None:
                keep  = ~((rev_e[0] == old_bidx) & (rev_e[1] == 0))
                rev_e = rev_e[:, keep]
            alt_gids = cf["drug"].n_id.tolist()
            if alt_gidx not in alt_gids:
                cf["drug"].x   = torch.cat([cf["drug"].x, data["drug"].x[alt_gidx].unsqueeze(0)], 0)
                cf["drug"].n_id = torch.cat([cf["drug"].n_id, torch.tensor([alt_gidx], device=device)])
                new_bidx = cf["drug"].x.shape[0] - 1
            else:
                new_bidx = alt_gids.index(alt_gidx)
            new_edge = torch.tensor([[new_bidx], [0]], device=device)
            rev_e    = torch.cat([rev_e, new_edge], 1)
            cf["drug", "prescribed_by", "patient"].edge_index = rev_e
        except Exception:
            pass

        with torch.no_grad():
            try:
                sv, _, _ = model(cf.x_dict, cf.edge_index_dict)
                cf_risk  = torch.sigmoid(sv[0]).item()
            except Exception:
                cf_risk = current_risk
        results.append({"rxcui": alt_rxcui, "new_risk": cf_risk,
                         "delta": current_risk - cf_risk})

    results.sort(key=lambda x: x["new_risk"])
    return results, results[0] if results else None, len(candidates)


# ── Plotly layout ─────────────────────────────────────────────────────────────
PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="Inter", color="#cbd5e1"),
          margin=dict(l=0, r=0, t=30, b=0))

def risk_gauge(prob):
    color = "#ef4444" if prob > .7 else ("#f97316" if prob > .4 else "#22c55e")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 44, "color": "#e2e8f0", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569"},
            "bar":  {"color": color, "thickness": .25},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [{"range": [0,40],  "color": "rgba(34,197,94,.1)"},
                      {"range": [40,70], "color": "rgba(249,115,22,.1)"},
                      {"range": [70,100],"color": "rgba(239,68,68,.1)"}],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": .8, "value": prob * 100},
        },
    ))
    fig.update_layout(**PL, height=250)
    return fig


def se_bar_graph(se_data: list, n_drugs: int):
    """Bar chart of side effects coloured by how many drugs cause them."""
    labels = [d["se"].title()     for d in se_data][::-1]
    counts = [d["count"]          for d in se_data][::-1]
    fracs  = [c / n_drugs         for c in counts]
    colors = [f"rgba(239,68,68,{.3 + .7 * f})" for f in fracs]

    fig = go.Figure(go.Bar(
        x=counts, y=labels, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{c}/{n_drugs} drugs" for c in counts],
        textposition="outside", textfont=dict(size=10, color="#94a3b8"),
    ))
    fig.update_layout(
        **PL, height=max(300, len(labels) * 22),
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, n_drugs * 1.3]),
        yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        bargap=.25,
        title=dict(text="How many of your drugs cause each side effect",
                   font=dict(size=12, color="#94a3b8"), x=0),
    )
    return fig


def ddi_net_fig(rxcuis: list, ddi_edge, drug_map, drug_names):
    gidx_set = {drug_map[r] for r in rxcuis if r in drug_map}
    if len(gidx_set) < 2:
        return None
    src, dst = ddi_edge[0].tolist(), ddi_edge[1].tolist()
    edges = [(s, d) for s, d in zip(src, dst) if s in gidx_set and d in gidx_set and s != d]
    n = len(rxcuis)
    angles = [2 * 3.14159 * i / n for i in range(n)]
    pos = {r: (float(np.cos(a)), float(np.sin(a))) for r, a in zip(rxcuis, angles)}
    inv = {v: k for k, v in drug_map.items()}

    ex, ey = [], []
    for s, d in edges:
        rs, rd = inv.get(s), inv.get(d)
        if rs in pos and rd in pos:
            x0, y0 = pos[rs]; x1, y1 = pos[rd]
            ex += [x0, x1, None]; ey += [y0, y1, None]

    fig = go.Figure()
    if ex:
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
            line=dict(color="rgba(239,68,68,0.45)", width=2.5),
            hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[pos[r][0] for r in rxcuis], y=[pos[r][1] for r in rxcuis],
        mode="markers+text",
        marker=dict(size=30, color="rgba(99,102,241,0.8)",
                    line=dict(color="#818cf8", width=2)),
        text=[label(r, drug_names) for r in rxcuis],
        textposition="top center", textfont=dict(size=11, color="#e2e8f0"),
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        **PL, height=360,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.6, 1.6]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.6, 1.6], scaleanchor="x"),
        title=dict(text=f"Drug Interaction Network  ({len(edges)} DDI edges between your drugs)",
                   font=dict(size=12, color="#94a3b8"), x=0),
    )
    return fig


def cf_bar(results, drug_names):
    df = pd.DataFrame(results)
    colors = ["rgba(34,197,94,.7)" if d > 0 else "rgba(239,68,68,.5)" for d in df["delta"]]
    fig = go.Figure(go.Bar(
        x=df["rxcui"].apply(lambda r: label(r, drug_names)),
        y=(df["delta"] * 100).round(3),
        marker_color=colors, marker_line_width=0,
        text=(df["delta"] * 100).apply(lambda v: f"{v:+.2f}%"),
        textposition="outside", textfont=dict(size=10),
    ))
    fig.update_layout(
        **PL, height=280,
        title=dict(text="Risk Change per Alternative (green = safer)", font=dict(size=12, color="#94a3b8"), x=0),
        xaxis=dict(showgrid=False, tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, ticksuffix="%"),
        bargap=.2,
    )
    return fig


# ── Load ──────────────────────────────────────────────────────────────────────
if "res" not in st.session_state:
    st.session_state.res = load_everything()
res = st.session_state.res

# Build searchable drug list: "Name (rxcui)" -> rxcui
drug_map   = res["drug_map"]
drug_names = res["drug_names"]

known_drugs = {}  # display_label -> rxcui
for rxcui in drug_map.keys():
    known_drugs[label(rxcui, drug_names)] = rxcui
drug_options = sorted(known_drugs.keys())


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 12px;'>
        <div style='font-size:40px;'>💊</div>
        <div style='font-size:20px;font-weight:700;color:#e2e8f0;margin-top:6px;'>PolyGuard</div>
        <div style='font-size:11px;color:#64748b;margin-top:2px;'>Drug Combination Analyser</div>
    </div>
    <hr style='border-color:rgba(99,102,241,.2);margin:0 0 18px;'>
    """, unsafe_allow_html=True)

    st.markdown("**Select Drugs**")
    selected_labels = st.multiselect(
        "Choose drugs (type to search)",
        options=drug_options,
        placeholder="e.g. Aspirin, Warfarin …",
        label_visibility="collapsed",
        max_selections=10,
    )
    selected_rxcuis = [known_drugs[lbl] for lbl in selected_labels]

    st.markdown("<br>", unsafe_allow_html=True)
    n_cf = st.slider("Alternatives to simulate", 5, 40, 20, 5)

    analyse_btn = st.button("Analyse Combination", use_container_width=True, type="primary")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#475569;line-height:1.8;'>
        <b style='color:#64748b;'>How it works</b><br>
        1. Select 2+ drugs<br>
        2. Graph edges show which side effects each drug causes<br>
        3. HGT model estimates overall combination risk<br>
        4. Counterfactual simulation finds safer alternatives<br>
        <br>
        <b style='color:#64748b;'>Model</b><br>
        HGT · 2L · 4H · 128-dim<br>
        Val Accuracy: 68.70%
    </div>
    """, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:28px 0 6px;'>
    <h1 style='font-size:32px;font-weight:700;color:#e2e8f0;margin:0;'>
        Drug Combination Risk Analyser
    </h1>
    <p style='font-size:14px;color:#64748b;margin-top:6px;'>
        Enter your drug combination on the left to get predicted side effects, interaction network, and safer alternatives.
    </p>
</div>
<hr style='border-color:rgba(99,102,241,.15);margin:0 0 20px;'>
""", unsafe_allow_html=True)


# ── Landing state ─────────────────────────────────────────────────────────────
if not analyse_btn and "last_result" not in st.session_state:
    st.markdown("""
    <div style='text-align:center;padding:60px 0;'>
        <div style='font-size:60px;'>🔬</div>
        <h2 style='color:#475569;font-weight:500;margin-top:14px;'>Select drugs to begin</h2>
        <p style='color:#334155;font-size:14px;max-width:480px;margin:10px auto;'>
            Choose 2 or more drug names from the left sidebar and click <b>Analyse Combination</b>.
            PolyGuard will predict shared side effects, estimate risk, and suggest safer alternatives.
        </p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl, sub) in zip(
        [c1, c2, c3, c4],
        [("1,920", "Drugs", "in the knowledge graph"),
         ("1,000", "Side Effects", "SIDER + OFFSIDES"),
         ("174K", "DDI Edges", "drug interaction pairs"),
         ("89k", "Patients", "FAERS adverse event reports")],
    ):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                        f'<div class="metric-value">{val}</div>'
                        f'<div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)
    st.stop()


# ── Run analysis ──────────────────────────────────────────────────────────────
if analyse_btn:
    if len(selected_rxcuis) < 1:
        st.warning("Please select at least 1 drug.")
        st.stop()
    st.session_state.last_result = {
        "rxcuis": selected_rxcuis, "labels": selected_labels,
    }
    st.session_state.cf_state = None

result_meta = st.session_state.last_result
rxcuis      = result_meta["rxcuis"]
labels_used = result_meta["labels"]

# Graph-based SE aggregation
with st.spinner("Aggregating side effects from knowledge graph …"):
    se_data = get_graph_se(rxcuis, res, top_k=20)

# HGT risk score (find best-matching patient)
with st.spinner("Running HGT risk model …"):
    p_gidx   = find_matching_patient(rxcuis, res)
    sev_prob = None
    if p_gidx is not None:
        try:
            sev_prob, _ = run_hgt_inference(p_gidx, res)
        except Exception as e:
            st.warning(f"HGT inference failed: {e}")

# DDI edges between these drugs
ddi_fig = ddi_net_fig(rxcuis, res["ddi_edge"], drug_map, drug_names)


# ── Panel 1: overview strip ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Drug Combination Overview</div>', unsafe_allow_html=True)

chips = "".join(f'<span class="drug-chip">{lbl}</span>' for lbl in labels_used)
st.markdown(f'<div style="margin-bottom:16px;">{chips}</div>', unsafe_allow_html=True)

ov1, ov2, ov3, ov4 = st.columns(4)
with ov1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Drugs Selected</div>'
                f'<div class="metric-value">{len(rxcuis)}</div></div>', unsafe_allow_html=True)
with ov2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Shared Side Effects</div>'
                f'<div class="metric-value">{len(se_data)}</div>'
                f'<div class="metric-sub">from top-20</div></div>', unsafe_allow_html=True)
with ov3:
    if sev_prob is not None:
        badge = ('<span class="badge-high">HIGH</span>' if sev_prob > .7
                 else ('<span class="badge-moderate">MODERATE</span>' if sev_prob > .4
                       else '<span class="badge-low">LOW</span>'))
        st.markdown(f'<div class="metric-card"><div class="metric-label">Combination Risk</div>'
                    f'<div class="metric-value">{sev_prob:.1%}</div>'
                    f'<div class="metric-sub">{badge}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><div class="metric-label">Combination Risk</div>'
                    '<div class="metric-value">N/A</div></div>', unsafe_allow_html=True)
with ov4:
    # Count DDI pairs
    gidx_set = {drug_map.get(r) for r in rxcuis} - {None}
    src_l, dst_l = res["ddi_edge"][0].tolist(), res["ddi_edge"][1].tolist()
    n_ddi = sum(1 for s, d in zip(src_l, dst_l) if s in gidx_set and d in gidx_set and s != d)
    st.markdown(f'<div class="metric-card"><div class="metric-label">Known DDI Pairs</div>'
                f'<div class="metric-value">{n_ddi}</div>'
                f'<div class="metric-sub">in this combination</div></div>', unsafe_allow_html=True)


# ── Panel 2: Risk gauge + SE bar ──────────────────────────────────────────────
col_g, col_se = st.columns([1, 2], gap="large")

with col_g:
    st.markdown('<div class="section-header">HGT Risk Score</div>', unsafe_allow_html=True)
    if sev_prob is not None:
        st.plotly_chart(risk_gauge(sev_prob), use_container_width=True,
                        config={"displayModeBar": False})
        if sev_prob > .7:
            st.markdown('<div style="text-align:center"><span class="badge-high">HIGH RISK</span></div>',
                        unsafe_allow_html=True)
            st.caption("Immediate clinical review recommended.")
        elif sev_prob > .4:
            st.markdown('<div style="text-align:center"><span class="badge-moderate">MODERATE RISK</span></div>',
                        unsafe_allow_html=True)
            st.caption("Monitor closely. Consider drug review.")
        else:
            st.markdown('<div style="text-align:center"><span class="badge-low">LOW RISK</span></div>',
                        unsafe_allow_html=True)
            st.caption("Combination appears relatively safe.")
    else:
        st.info("Risk score unavailable — no matching patient found in graph.")

with col_se:
    st.markdown('<div class="section-header">Predicted Side Effects (Top 20)</div>', unsafe_allow_html=True)
    if se_data:
        st.plotly_chart(se_bar_graph(se_data, len(rxcuis)),
                        use_container_width=True, config={"displayModeBar": False})
        st.caption("Based on direct drug→side-effect edges in the knowledge graph (SIDER + OFFSIDES).")
    else:
        st.info("No side effect associations found for the selected drugs in the graph.")


# ── Panel 3: DDI Network ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Drug Interaction Network</div>', unsafe_allow_html=True)
if ddi_fig:
    st.plotly_chart(ddi_fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("No known drug-drug interaction edges between selected drugs, or fewer than 2 drugs selected.")


# ── Panel 4: Counterfactual ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Safer Drug Alternative Finder</div>', unsafe_allow_html=True)

if len(rxcuis) < 1:
    st.info("Select at least 1 drug to find safer alternatives.")
elif p_gidx is None:
    st.warning("Cannot run counterfactual — no matching patient found in graph.")
else:
    drug_to_replace_lbl = st.selectbox(
        "Which drug would you like to find an alternative for?",
        options=labels_used,
    )
    drug_to_replace_rxcui = known_drugs[drug_to_replace_lbl]

    if st.button("Find Safer Alternatives", type="primary"):
        with st.spinner(f"Simulating {n_cf} alternative drugs via counterfactual …"):
            cf_results, best_cf, total_cands = run_counterfactual(
                drug_to_replace_rxcui, rxcuis, p_gidx,
                sev_prob if sev_prob else 0.5, n_cf, res
            )
        st.session_state.cf_state = {
            "results": cf_results, "best": best_cf, "total": total_cands,
            "replaced": drug_to_replace_rxcui,
            "replaced_lbl": drug_to_replace_lbl,
            "current_risk": sev_prob or 0.5,
        }

    if st.session_state.get("cf_state"):
        cf = st.session_state.cf_state
        if not cf["results"]:
            st.info("No ATC-class alternatives found for this drug in the knowledge graph.")
        else:
            # Table
            df_cf = pd.DataFrame(cf["results"])
            df_cf["Drug Name"]        = df_cf["rxcui"].apply(lambda r: label(r, drug_names))
            df_cf["New Risk (%)"]     = (df_cf["new_risk"] * 100).round(2)
            df_cf["Risk Change (%)"]  = (df_cf["delta"] * 100).round(3)
            df_cf["Safer?"]           = df_cf["delta"].apply(lambda d: "YES" if d > 0 else "No")
            st.dataframe(
                df_cf[["Drug Name", "New Risk (%)", "Risk Change (%)", "Safer?"]],
                use_container_width=True, hide_index=True,
            )

            st.plotly_chart(cf_bar(cf["results"], drug_names),
                            use_container_width=True, config={"displayModeBar": False})

            if cf["best"] and cf["best"]["delta"] > 0:
                best = cf["best"]
                st.markdown(f"""
                <div class="rec-box">
                    <div class="rec-title">Recommendation</div>
                    <table style="width:100%;font-size:14px;color:#cbd5e1;border-collapse:collapse;">
                        <tr><td style="color:#94a3b8;width:160px;padding:4px 0;">Replace</td>
                            <td><b>{cf["replaced_lbl"]}</b></td></tr>
                        <tr><td style="color:#94a3b8;padding:4px 0;">With</td>
                            <td><b>{label(best["rxcui"], drug_names)}</b></td></tr>
                        <tr><td style="color:#94a3b8;padding:4px 0;">Current risk</td>
                            <td>{cf["current_risk"]:.1%}</td></tr>
                        <tr><td style="color:#94a3b8;padding:4px 0;">Projected risk</td>
                            <td><b style="color:#34d399;">{best["new_risk"]:.1%}</b>
                            &nbsp;<span style="font-size:12px;color:#64748b;">
                            ({best["delta"]*100:+.2f}% reduction)</span></td></tr>
                        <tr><td style="color:#94a3b8;padding:4px 0;">Alternatives tested</td>
                            <td>{len(cf["results"])} of {cf["total"]} same-class drugs</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No safer alternative found in the simulated candidates.")


# ── Panel 5: SE detail table ──────────────────────────────────────────────────
if se_data:
    with st.expander("Full Side Effect Details"):
        rows = []
        for d in se_data:
            rows.append({
                "Side Effect": d["se"].title(),
                "Caused by # drugs": d["count"],
                "Out of": len(rxcuis),
                "Drugs": ", ".join(label(r, drug_names) for r in d["drugs"]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)