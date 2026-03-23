"""
Polypharmacy Risk Predictor — Streamlit App

Shows:
  - Risk probability with visual indicator
  - Per-drug importance scores (GNNExplainer + Gradient)
  - Why each drug contributes (feature-level breakdown)
  - Drug-drug interaction importance
"""

import streamlit as st
import torch
import json
import os
import numpy as np

from graph_builder import build_graph, nodes, get_available_drugs
from model_utils import (
    load_model, predict, gradient_attribution,
    explain_graph, get_top_features
)

# ============================================================
# LOAD MODEL & METADATA
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
meta_path = os.path.join(BASE_DIR, "DATA", "layer3_feature_meta.json")

with open(meta_path, "r") as f:
    feat_meta = json.load(f)

feat_names = feat_meta["feature_names"]
model, checkpoint, _ = load_model()

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Polypharmacy Risk Predictor", layout="wide")

st.title("💊 Polypharmacy Risk Predictor")
st.markdown("Predict the risk of adverse drug interactions using Graph Neural Networks with explainability.")

# Drug input
st.subheader("Enter Drug Combination")

available_drugs = get_available_drugs()
st.caption(f"{len(available_drugs)} drugs available in database")

# Multiselect for easier drug selection
selected_drugs = st.multiselect(
    "Select drugs (or type to search):",
    options=[d.title() for d in available_drugs],
    default=None,
    help="Select at least 2 drugs"
)

# Also allow free-text input
user_input = st.text_input(
    "Or enter drug names separated by commas:",
    placeholder="e.g., lamivudine, enfuvirtide, cytarabine"
)

if st.button("🔍 Predict Risk", type="primary"):

    # Merge inputs
    drugs = []
    if selected_drugs:
        drugs.extend([d.strip() for d in selected_drugs])
    if user_input.strip():
        drugs.extend([d.strip() for d in user_input.split(",")])

    drugs = [d for d in drugs if d.strip()]

    if len(drugs) < 2:
        st.error("Please enter at least 2 drugs.")
    else:
        try:
            graph, indices = build_graph(drugs)

            # ----------------------------------------
            # PREDICTION
            # ----------------------------------------
            risk_prob = predict(model, graph)

            st.divider()
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("🎯 Risk Assessment")
                st.metric("Risk Probability", f"{risk_prob:.1%}")

                if risk_prob > 0.7:
                    st.error("⚠️ HIGH RISK — Significant interaction potential")
                elif risk_prob > 0.5:
                    st.warning("⚡ MODERATE-HIGH RISK")
                elif risk_prob > 0.3:
                    st.info("📋 MODERATE-LOW RISK")
                else:
                    st.success("✅ LOW RISK")

                # Edge info
                has_edges = graph.edge_index.shape[1] > 0
                n_edges = graph.edge_index.shape[1] // 2  # undirected
                if has_edges:
                    st.caption(f"Found {n_edges} known drug-drug interaction(s)")
                else:
                    st.caption("⚠️ No known interactions in TWOSIDES database")

            with col2:
                st.subheader("Drug Combination")
                drug_info = []
                for i, drug in enumerate(drugs):
                    idx = indices[i]
                    row = nodes.iloc[idx]
                    drug_info.append({
                        "Drug": drug.title(),
                        "ATC Count": int(row.get("atc_count", 0)),
                        "Gene Targets": int(row.get("gene_count", 0)),
                        "Known Side Effects": int(row.get("side_effect_count", 0)),
                    })
                st.dataframe(drug_info, use_container_width=True, hide_index=True)

            st.divider()

            # ----------------------------------------
            # EXPLAINABILITY
            # ----------------------------------------
            st.subheader("🔬 Explainability Analysis")

            tab1, tab2, tab3 = st.tabs([
                "Drug Importance (GNNExplainer)",
                "Drug Importance (Gradient)",
                "Drug-Drug Interaction Importance"
            ])

            # --- TAB 1: GNNExplainer ---
            with tab1:
                with st.spinner("Running GNNExplainer (200 epochs)..."):
                    node_imp, edge_imp, feat_imp, node_feat_attr = explain_graph(
                        model, graph, feat_names
                    )

                st.markdown("**Per-Drug Risk Contribution (GNNExplainer)**")

                # Sort by importance
                ranked = np.argsort(node_imp)[::-1]
                for rank, idx in enumerate(ranked):
                    drug_name = drugs[idx].title()
                    score = node_imp[idx]

                    # Color bar
                    bar_pct = int(score * 100)
                    st.markdown(f"**#{rank+1} {drug_name}** — importance: `{score:.4f}`")
                    st.progress(float(min(score, 1.0)))

                    # Feature breakdown
                    top_feats = get_top_features(node_feat_attr[idx], feat_names, top_k=5)
                    if top_feats:
                        with st.expander(f"Why {drug_name} contributes to risk"):
                            for feat_name, feat_score in top_feats:
                                direction = "↑ increases" if feat_score > 0 else "↓ decreases"
                                st.markdown(f"- **{feat_name}**: {direction} risk (score: `{feat_score:.4f}`)")

                # Global feature importance
                st.markdown("---")
                st.markdown("**Top Risk Factors (Global Feature Importance)**")
                from model_utils import _beautify_feature
                top_feat_idx = np.argsort(np.abs(feat_imp))[::-1][:8]
                for fi in top_feat_idx:
                    if feat_imp[fi] > 0.001:
                        display = _beautify_feature(feat_names[fi])
                        st.markdown(f"- {display}: `{feat_imp[fi]:.4f}`")

            # --- TAB 2: Gradient Attribution ---
            with tab2:
                drug_scores, feature_attr = gradient_attribution(model, graph, feat_names)

                st.markdown("**Per-Drug Risk Contribution (Gradient × Input)**")

                ranked = np.argsort(drug_scores)[::-1]
                for rank, idx in enumerate(ranked):
                    drug_name = drugs[idx].title()
                    score = drug_scores[idx]

                    st.markdown(f"**#{rank+1} {drug_name}** — importance: `{score:.4f}`")
                    st.progress(float(min(score, 1.0)))

                    top_feats = get_top_features(feature_attr[idx], feat_names, top_k=5)
                    if top_feats:
                        with st.expander(f"Feature breakdown for {drug_name}"):
                            for feat_name, feat_score in top_feats:
                                st.markdown(f"- **{feat_name}**: `{abs(feat_score):.4f}`")

            # --- TAB 3: Edge Importance ---
            with tab3:
                if edge_imp is not None and len(edge_imp) > 0:
                    st.markdown("**Drug–Drug Interaction Importance**")

                    edge_idx_np = graph.edge_index.cpu().numpy()
                    seen_pairs = set()
                    edge_data = []

                    edge_ranked = np.argsort(edge_imp)[::-1]
                    for ei in edge_ranked:
                        s, t = edge_idx_np[0, ei], edge_idx_np[1, ei]
                        pair = (min(s, t), max(s, t))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)

                        drug1 = drugs[s].title()
                        drug2 = drugs[t].title()
                        imp = edge_imp[ei]

                        edge_data.append({
                            "Drug 1": drug1,
                            "Drug 2": drug2,
                            "Interaction Importance": f"{imp:.4f}",
                            "Edge Weight": f"{graph.edge_weight[ei].item():.3f}"
                        })

                    if edge_data:
                        st.dataframe(edge_data, use_container_width=True, hide_index=True)
                    else:
                        st.info("No known drug-drug interactions found between these drugs.")
                else:
                    st.info("No edge importance data available.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ============================================================
# SIDEBAR INFO
# ============================================================
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This app predicts polypharmacy risk using a **Graph Neural Network** "
        "trained on FAERS, TWOSIDES, SIDER, and DrugCentral data."
    )
    st.markdown("### Explainability Methods")
    st.markdown(
        "- **GNNExplainer**: Learns feature masks to identify important node features and edges\n"
        "- **Gradient Attribution**: Uses gradient × input to quantify per-drug contribution\n"
    )
    st.markdown("### Data Sources")
    st.markdown(
        "- **FAERS** — FDA Adverse Event Reports\n"
        "- **TWOSIDES** — Drug-drug interaction database\n"
        "- **SIDER** — Drug side effects\n"
        "- **DrugCentral** — Drug targets & enzymes\n"
        "- **ATC** — Drug classification\n"
    )