"""
Polypharmacy GNN — PowerPoint Presentation Generator

Creates a professional presentation with:
  Slide 1:  Problem Statement
  Slide 2:  Summary of References (1/2) — Databases & Datasets
  Slide 3:  Summary of References (2/2) — GNN Methods & Related Work
  Slide 4:  Data Pipeline & Preprocessing Methodology
  Slide 5:  Graph Construction & Feature Engineering
  Slide 6:  GNN Architecture & Training
  Slide 7:  Results — Model Performance
  Slide 8:  Results — Explainability & Case Studies
  Slide 9:  Results — Visualizations (ROC, Feature Importance)
"""

import os
import json
import numpy as np
import torch
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "RESULTS")
OUTPUT_PATH = os.path.join(BASE, "Polypharmacy_GNN_Presentation.pptx")

# ============================================================
# LOAD METRICS
# ============================================================
checkpoint = torch.load(
    os.path.join(BASE, "weighted_gcn_model.pth"),
    map_location="cpu"
)

with open(os.path.join(BASE, "layer3_feature_meta.json"), "r") as f:
    feat_meta = json.load(f)

with open(os.path.join(BASE, "model_config.json"), "r") as f:
    model_config = json.load(f)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

# Colors
DARK_BG = RGBColor(0x1A, 0x1A, 0x2E)       # Dark navy
ACCENT1 = RGBColor(0x00, 0xD4, 0xFF)        # Cyan
ACCENT2 = RGBColor(0xFF, 0x6B, 0x6B)        # Coral
ACCENT3 = RGBColor(0x4E, 0xC9, 0xB0)        # Teal
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
MED_GRAY = RGBColor(0x99, 0x99, 0x99)
DARK_TEXT = RGBColor(0x33, 0x33, 0x33)
SECTION_BG = RGBColor(0xF5, 0xF7, 0xFA)
BLUE_ACC = RGBColor(0x21, 0x96, 0xF3)
GREEN_ACC = RGBColor(0x4C, 0xAF, 0x50)
RED_ACC = RGBColor(0xF4, 0x43, 0x36)
ORANGE_ACC = RGBColor(0xFF, 0x98, 0x00)
PURPLE_ACC = RGBColor(0x9C, 0x27, 0xB0)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def set_slide_bg(slide, color):
    """Set slide background to solid color."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, font_size=14,
                bold=False, color=WHITE, alignment=PP_ALIGN.LEFT,
                font_name="Calibri"):
    """Add a text box to a slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_bullet_frame(slide, left, top, width, height, items, font_size=13,
                     color=WHITE, spacing=Pt(6), font_name="Calibri"):
    """Add a multi-line bulleted text frame."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()

        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = font_name
        p.space_after = spacing
        p.level = 0

    return txBox


def add_box_shape(slide, left, top, width, height, fill_color, text="",
                  font_size=11, font_color=WHITE, bold=False):
    """Add a rounded rectangle shape with text."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()

    if text:
        tf = shape.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(font_size)
        p.font.color.rgb = font_color
        p.font.bold = bold

    return shape


def add_arrow(slide, left, top, width, height):
    """Add a right-pointing arrow."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT1
    shape.line.fill.background()
    return shape


# ============================================================
# CREATE PRESENTATION
# ============================================================
prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]  # blank

# ============================================================
# SLIDE 1: PROBLEM STATEMENT
# ============================================================
print("Creating Slide 1: Problem Statement...")
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_textbox(slide, Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
            "💊 POLYPHARMACY RISK PREDICTION", 32, True, ACCENT1, PP_ALIGN.LEFT)

add_textbox(slide, Inches(0.8), Inches(1.2), Inches(11), Inches(0.4),
            "Using Graph Neural Networks with Explainability", 20, False, LIGHT_GRAY)

# Divider line
add_box_shape(slide, Inches(0.8), Inches(1.8), Inches(11.5), Inches(0.03), ACCENT1)

# Problem statement content
problem_items = [
    "▸ Polypharmacy (use of 5+ drugs) affects 40-50% of elderly patients",
    "▸ Drug-drug interactions (DDIs) cause ~195,000 hospitalizations/year in the US alone",
    "▸ Traditional rule-based systems cannot capture complex multi-drug interaction patterns",
    "▸ Existing approaches lack explainability — clinicians need to understand WHY a combination is risky",
]
add_bullet_frame(slide, Inches(0.8), Inches(2.2), Inches(5.5), Inches(2.5),
                 problem_items, 14, WHITE, Pt(10))

# Objective box
add_box_shape(slide, Inches(7), Inches(2.2), Inches(5.5), Inches(2.5),
              RGBColor(0x16, 0x3C, 0x6B))
add_textbox(slide, Inches(7.3), Inches(2.3), Inches(5), Inches(0.4),
            "OBJECTIVE", 16, True, ACCENT1)
obj_items = [
    "Build a GNN-based system that:",
    "  1. Predicts polypharmacy risk from drug combinations",
    "  2. Uses real-world adverse event data (FAERS)",
    "  3. Incorporates drug-drug interaction evidence (TWOSIDES)",
    "  4. Provides per-drug explainability (GNNExplainer)",
    "  5. Delivers results via interactive Streamlit UI",
]
add_bullet_frame(slide, Inches(7.3), Inches(2.8), Inches(5), Inches(2),
                 obj_items, 13, WHITE, Pt(6))

# Key innovation boxes
add_textbox(slide, Inches(0.8), Inches(5.0), Inches(11), Inches(0.4),
            "KEY INNOVATIONS", 16, True, ACCENT3)

innovations = [
    ("Multi-Source\nData Fusion", "FAERS + TWOSIDES +\nSIDER + DrugCentral"),
    ("Rich Node\nFeatures", "65-dim: ATC + Targets\n+ Actions + Counts"),
    ("Weighted\nGraph GCN", "Edge weights from\nPRR scores"),
    ("Dual\nExplainability", "GNNExplainer +\nGradient Attribution"),
]

for i, (title, desc) in enumerate(innovations):
    x = Inches(0.8 + i * 3.1)
    add_box_shape(slide, x, Inches(5.5), Inches(2.8), Inches(1.5),
                  RGBColor(0x16, 0x3C, 0x6B))
    add_textbox(slide, x + Inches(0.15), Inches(5.55), Inches(2.5), Inches(0.6),
                title, 13, True, ACCENT1, PP_ALIGN.CENTER)
    add_textbox(slide, x + Inches(0.15), Inches(6.1), Inches(2.5), Inches(0.6),
                desc, 11, False, LIGHT_GRAY, PP_ALIGN.CENTER)


# ============================================================
# SLIDE 2: REFERENCES 1/2 — Databases
# ============================================================
print("Creating Slide 2: References (Databases)...")
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.5),
            "SUMMARY OF REFERENCES (1/2) — Databases & Datasets", 26, True, ACCENT1)
add_box_shape(slide, Inches(0.8), Inches(1.0), Inches(11.5), Inches(0.03), ACCENT1)

databases = [
    {
        "name": "FAERS (FDA Adverse Event Reporting System)",
        "color": BLUE_ACC,
        "desc": "Post-market surveillance database of adverse drug events reported to the FDA",
        "details": [
            "Used: 2023 Q1-Q4 quarterly data files (DEMO, DRUG, OUTC, REAC tables)",
            "Polypharmacy cases: Cases with ≥2 drugs marked as Primary/Secondary Suspect",
            "Severity labels: DE (Death), LT (Life-threatening), HO (Hospitalization), DS (Disability)",
            "Reference: FDA (2023). FAERS Quarterly Data Files. fda.gov/drugs/fda-adverse-event-reporting-system-faers",
        ]
    },
    {
        "name": "TWOSIDES",
        "color": GREEN_ACC,
        "desc": "Comprehensive database of polypharmacy side effects mined from FAERS",
        "details": [
            "Contains ~870M drug-drug-side effect associations with PRR (Proportional Reporting Ratio)",
            "Used as edge weights in the drug interaction graph (log1p-transformed PRR scores)",
            "Filtered to drugs present in our FAERS cohort with PRR > 1 (risk-elevating interactions)",
            "Reference: Tatonetti et al. (2012). Data-Driven Prediction of Drug Effects. Sci Transl Med. 4(125)",
        ]
    },
    {
        "name": "SIDER (Side Effect Resource)",
        "color": ORANGE_ACC,
        "desc": "Database linking marketed drugs to known side effects using MedDRA",
        "details": [
            "Provides per-drug known side effect profiles for feature enrichment",
            "Mapped via STITCH compound IDs → RxNorm → DrugCentral structure_id",
            "Side effect counts used as node features (log-scaled)",
            "Reference: Kuhn et al. (2016). SIDER database of drugs and side effects. Nucleic Acids Res. 44(D1)",
        ]
    },
    {
        "name": "DrugCentral + RxNorm",
        "color": PURPLE_ACC,
        "desc": "Drug information resource with targets, mechanisms, and ATC classification",
        "details": [
            "DrugCentral: Target genes, target classes, action types, ATC codes per drug",
            "RxNorm: Standardized drug name normalization (RXCUI identifiers)",
            "Provides structure_id as canonical drug identifier linking all data sources",
            "References: Ursu et al. (2019). DrugCentral 2019. Nucleic Acids Res; Nelson et al. (2011). RxNorm. AMIA.",
        ]
    },
]

y_start = 1.2
for i, db in enumerate(databases):
    y = Inches(y_start + i * 1.5)
    # Color bar
    add_box_shape(slide, Inches(0.8), y, Inches(0.1), Inches(1.3), db["color"])
    # Title
    add_textbox(slide, Inches(1.1), y, Inches(11), Inches(0.35),
                db["name"], 14, True, WHITE)
    # Description
    add_textbox(slide, Inches(1.1), y + Inches(0.3), Inches(11), Inches(0.25),
                db["desc"], 11, False, ACCENT3)
    # Details
    for j, detail in enumerate(db["details"]):
        add_textbox(slide, Inches(1.3), y + Inches(0.55 + j * 0.2), Inches(10.5), Inches(0.2),
                    f"• {detail}", 10, False, LIGHT_GRAY)


# ============================================================
# SLIDE 3: REFERENCES 2/2 — Methods
# ============================================================
print("Creating Slide 3: References (Methods)...")
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.5),
            "SUMMARY OF REFERENCES (2/2) — GNN Methods & Related Work", 26, True, ACCENT1)
add_box_shape(slide, Inches(0.8), Inches(1.0), Inches(11.5), Inches(0.03), ACCENT1)

methods = [
    {
        "name": "Graph Convolutional Networks (GCN)",
        "color": BLUE_ACC,
        "details": [
            "Kipf & Welling (2017). Semi-Supervised Classification with GCNs. ICLR.",
            "Message-passing framework: aggregates neighbor features via spectral convolution",
            "We use 2-layer GCN with learnable edge weights from TWOSIDES PRR scores",
        ]
    },
    {
        "name": "GNNExplainer",
        "color": GREEN_ACC,
        "details": [
            "Ying et al. (2019). GNNExplainer: Generating Explanations for GNNs. NeurIPS.",
            "Learns soft masks on node features and edges to identify important substructures",
            "Provides per-drug importance scores and per-feature attribution for clinical interpretability",
        ]
    },
    {
        "name": "Polypharmacy Side Effect Prediction",
        "color": ORANGE_ACC,
        "details": [
            "Zitnik et al. (2018). Modeling polypharmacy side effects with GCNs. Bioinformatics.",
            "Decagon model: multirelational link prediction on protein-protein + drug-drug networks",
            "Our approach adapts graph-level classification (case-level prediction) vs link prediction",
        ]
    },
    {
        "name": "Gradient-based Attribution",
        "color": PURPLE_ACC,
        "details": [
            "Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks (Integrated Gradients). ICML.",
            "We use Gradient × Input as a fast approximation for per-drug feature attribution",
            "Dual explainability: complementary to GNNExplainer (structural vs gradient-based)",
        ]
    },
    {
        "name": "Adverse Event Prediction from EHR/FAERS Data",
        "color": RED_ACC,
        "details": [
            "Bates et al. (2003). Detecting Adverse Events Using Information Technology. JAMIA.",
            "Harpaz et al. (2012). Novel Data-Mining Methods for Drug Safety Signal Detection. Clinical Pharmacology.",
            "Our contribution: GNN-based approach with multi-source data fusion + explainability",
        ]
    },
]

y_start = 1.2
for i, m in enumerate(methods):
    y = Inches(y_start + i * 1.2)
    add_box_shape(slide, Inches(0.8), y, Inches(0.1), Inches(1.0), m["color"])
    add_textbox(slide, Inches(1.1), y, Inches(11), Inches(0.3),
                m["name"], 14, True, WHITE)
    for j, detail in enumerate(m["details"]):
        add_textbox(slide, Inches(1.3), y + Inches(0.3 + j * 0.22), Inches(10.5), Inches(0.22),
                    f"• {detail}", 10, False, LIGHT_GRAY)


# ============================================================
# SLIDE 4: DATA PIPELINE & PREPROCESSING
# ============================================================
print("Creating Slide 4: Data Pipeline & Preprocessing...")
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.5),
            "DATA PIPELINE & PREPROCESSING METHODOLOGY", 26, True, ACCENT1)
add_box_shape(slide, Inches(0.8), Inches(1.0), Inches(11.5), Inches(0.03), ACCENT1)

# Pipeline flow boxes
steps = [
    ("1. FAERS\nFiltering", "faers_filtering.py", BLUE_ACC, [
        "Load raw FAERS quarterly data (Q1-Q4 2023)",
        "Retain cases with ≥2 drugs (PS/SS role codes)",
        "Assign severity labels from OUTC table",
        "Output: modifiedfaersQ1-Q4.csv",
    ]),
    ("2. Combine &\nNormalize", "combine_faers.py\nfaers_rxnorm.py", GREEN_ACC, [
        "Merge 4 quarterly files → merged_faers.csv",
        "Load RxNorm RXNCONSO.RRF (English IN/PIN terms)",
        "Normalize drug names (strip dosage, format)",
        "Map FAERS drug names → RXCUI identifiers",
    ]),
    ("3. DrugCentral\nMapping", "faers_drugcentral.py\nstructure_atc.py", ORANGE_ACC, [
        "Map RXCUI → structure_id via rxcui_structure_map",
        "Attach ATC classification codes per case",
        "Extract ATC Level-1 categories (A-V, 14 classes)",
        "Output: faers_with_atc.csv",
    ]),
    ("4. Mechanism &\nSide Effects", "mechanism_mapping.py\nsider_processing.py", PURPLE_ACC, [
        "Map structure_id → target genes, target classes, action types",
        "Map SIDER side effects via STITCH → RxNorm → structure_id",
        "Attach known side effect profiles per drug per case",
        "Output: faers_layer2_final.csv (606 MB, full enriched dataset)",
    ]),
]

for i, (title, script, color, details) in enumerate(steps):
    x = Inches(0.5 + i * 3.15)
    # Box
    add_box_shape(slide, x, Inches(1.3), Inches(2.9), Inches(0.8), color)
    add_textbox(slide, x + Inches(0.1), Inches(1.35), Inches(2.7), Inches(0.7),
                title, 12, True, WHITE, PP_ALIGN.CENTER)

    # Script name
    add_textbox(slide, x + Inches(0.1), Inches(2.15), Inches(2.7), Inches(0.25),
                f"({script})", 9, False, MED_GRAY, PP_ALIGN.CENTER)

    # Details
    for j, detail in enumerate(details):
        add_textbox(slide, x + Inches(0.1), Inches(2.5 + j * 0.22), Inches(2.7), Inches(0.22),
                    f"• {detail}", 9, False, LIGHT_GRAY)

    # Arrow between boxes
    if i < len(steps) - 1:
        add_arrow(slide, x + Inches(2.95), Inches(1.55), Inches(0.2), Inches(0.3))

# TWOSIDES processing
add_textbox(slide, Inches(0.8), Inches(3.8), Inches(11), Inches(0.3),
            "PARALLEL: TWOSIDES Drug-Drug Interaction Processing", 14, True, ACCENT3)

twosides_items = [
    "▸ twosides_processing.py: Load TWOSIDES.csv (~4.3 GB, ~870M DDI associations)",
    "▸ Rename columns: drug_1_rxnorm_id → drug1_rxcui, drug_2_rxnorm_id → drug2_rxcui",
    "▸ Filter to drug pairs present in our FAERS cohort (both drugs must have valid RXCUI in our dataset)",
    "▸ Retain only interactions with PRR > 1 (proportional reporting ratio > baseline risk)",
    "▸ Output: twosides_filtered.csv (~1.2 GB) — used as graph edge source with log1p(PRR) weights",
]
add_bullet_frame(slide, Inches(0.8), Inches(4.2), Inches(11.5), Inches(2.2),
                 twosides_items, 11, LIGHT_GRAY, Pt(5))

# Data flow summary
add_textbox(slide, Inches(0.8), Inches(5.8), Inches(11), Inches(0.3),
            "DATA FLOW SUMMARY", 14, True, ACCENT1)
summary_items = [
    "Raw FAERS (4 quarters) → Filtered polypharmacy cases → RxNorm normalized → DrugCentral structure_id mapped",
    "→ ATC classifications attached → Target genes/mechanisms attached → SIDER side effects attached",
    "→ faers_layer2_final.csv (enriched dataset with all drug metadata per case)",
]
add_bullet_frame(slide, Inches(0.8), Inches(6.2), Inches(11.5), Inches(1.0),
                 summary_items, 11, LIGHT_GRAY, Pt(4))


# ============================================================
# SLIDE 5: GRAPH CONSTRUCTION & FEATURE ENGINEERING
# ============================================================
print("Creating Slide 5: Graph Construction & Feature Engineering...")
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.5),
            "GRAPH CONSTRUCTION & FEATURE ENGINEERING", 26, True, ACCENT1)
add_box_shape(slide, Inches(0.8), Inches(1.0), Inches(11.5), Inches(0.03), ACCENT1)

# Left: Nodes
add_textbox(slide, Inches(0.8), Inches(1.2), Inches(5.5), Inches(0.4),
            "NODE CONSTRUCTION (nodes_edges.py)", 16, True, WHITE)

node_items = [
    f"▸ {model_config['num_nodes']} unique drugs (structure_id from DrugCentral)",
    "▸ Each node = one drug, identified by DrugCentral structure_id",
    "▸ Nodes extracted from all STRUCTURE_ID_LIST entries in FAERS",
    "",
    "65-DIMENSIONAL FEATURE VECTOR PER NODE:",
    f"  [0:14]   ATC Level-1 multi-hot — 14 dims (A through V)",
    f"  [14:{14+feat_meta['n_tc']}]  Target class multi-hot — {feat_meta['n_tc']} dims",
    f"  [{14+feat_meta['n_tc']}:{14+feat_meta['n_tc']+feat_meta['n_at']}]  Action type multi-hot — {feat_meta['n_at']} dims",
    "  [last 3]  Log-scaled counts: ATC diversity, gene targets, side effects",
    "",
    "Feature sources:",
    "  • ATC codes → structure_atc_map.csv (Level-1 one-hot encoding)",
    "  • Target classes → structure_mechanism_map.csv (pharmacological targets)",
    "  • Action types → structure_mechanism_map.csv (agonist, antagonist, etc.)",
    "  • Count features → aggregated from faers_layer2_final.csv",
]
add_bullet_frame(slide, Inches(0.8), Inches(1.7), Inches(5.5), Inches(4.5),
                 node_items, 11, LIGHT_GRAY, Pt(3))

# Right: Edges
add_textbox(slide, Inches(7.0), Inches(1.2), Inches(5.5), Inches(0.4),
            "EDGE CONSTRUCTION", 16, True, WHITE)

import pandas as pd
edges_df = pd.read_csv(os.path.join(BASE, "layer3_edges_weighted.csv"))
cases_df = pd.read_csv(os.path.join(BASE, "layer3_training_cases.csv"))
n_pos = int(cases_df["label"].sum())
n_neg = int(len(cases_df) - n_pos)

edge_items = [
    f"▸ {len(edges_df):,} weighted edges from TWOSIDES database",
    "▸ Edge = known drug-drug interaction between two structure_ids",
    "▸ Weight = log1p(PRR) where PRR = Proportional Reporting Ratio",
    "▸ Self-loops excluded, undirected (stored as bidirectional)",
    "▸ Drug pairs mapped: RXCUI → structure_id via rxcui_structure_map",
    "",
    "TRAINING CASES (from FAERS):",
    f"▸ {len(cases_df):,} polypharmacy cases (≥2 drugs with edges)",
    f"▸ Label 1 (High Risk): {n_pos:,} cases ({100*n_pos/len(cases_df):.1f}%)",
    f"▸ Label 0 (Low Risk):  {n_neg:,} cases ({100*n_neg/len(cases_df):.1f}%)",
    "▸ Severity from FAERS OUTC: DE, LT, HO, DS → High Risk",
    "",
    "PER-CASE SUBGRAPH:",
    "▸ For each FAERS case, extract the subset of drug nodes",
    "▸ Include only edges between drugs in that case",
    "▸ Skip cases with no edges (drugs not in TWOSIDES)",
    "▸ Each subgraph = one training sample for the GNN",
]
add_bullet_frame(slide, Inches(7.0), Inches(1.7), Inches(5.5), Inches(4.5),
                 edge_items, 11, LIGHT_GRAY, Pt(3))

# Bottom: Enrichment
add_textbox(slide, Inches(0.8), Inches(6.0), Inches(11), Inches(0.3),
            "NODE ENRICHMENT (Layer 6)", 14, True, ACCENT3)
enrich_items = [
    "▸ build_drug_lookup.py: Map structure_id → human-readable drug name via RxNorm RXNCONSO (IN/PIN terms)",
    "▸ enrich_nodes.py: Merge drug names into nodes CSV, assign node_index for frontend lookup → layer3_nodes_enriched.csv",
]
add_bullet_frame(slide, Inches(0.8), Inches(6.4), Inches(11.5), Inches(0.8),
                 enrich_items, 11, LIGHT_GRAY, Pt(4))


# ============================================================
# SLIDE 6: GNN ARCHITECTURE & TRAINING
# ============================================================
print("Creating Slide 6: Architecture...")
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.5),
            "GNN ARCHITECTURE & TRAINING METHODOLOGY", 26, True, ACCENT1)
add_box_shape(slide, Inches(0.8), Inches(1.0), Inches(11.5), Inches(0.03), ACCENT1)

# Architecture flow
arch_blocks = [
    ("Drug Node\nFeatures\n(65-dim)", BLUE_ACC),
    ("Learnable\nEmbedding\n(32-dim)", GREEN_ACC),
    ("Concat\n(97-dim)", MED_GRAY),
    ("GCN Layer 1\n97 → 128\n+ ReLU + Dropout", ORANGE_ACC),
    ("GCN Layer 2\n128 → 64\n+ ReLU + Dropout", ORANGE_ACC),
    ("Global Mean\nPooling", PURPLE_ACC),
    ("MLP\n64→32→1\nLogits", RED_ACC),
]

for i, (text, color) in enumerate(arch_blocks):
    x = Inches(0.3 + i * 1.85)
    add_box_shape(slide, x, Inches(1.3), Inches(1.6), Inches(1.1), color,
                  text, 10, WHITE, True)
    if i < len(arch_blocks) - 1:
        add_arrow(slide, x + Inches(1.65), Inches(1.65), Inches(0.2), Inches(0.2))

# Training details
add_textbox(slide, Inches(0.8), Inches(2.8), Inches(5.5), Inches(0.4),
            "TRAINING CONFIGURATION", 16, True, WHITE)

train_items = [
    "▸ Model: WeightedGCN  (custom PyTorch Geometric model)",
    f"▸ Parameters: {sum(p.numel() for p in torch.nn.Embedding(model_config['num_nodes'], 32).parameters()) + 128*(97+1) + 64*(128+1) + 32*(64+1) + 1*(32+1):,}+ total",
    "▸ Loss: BCEWithLogitsLoss with class-weighted pos_weight",
    "▸ Optimizer: Adam (lr=0.001, weight_decay=1e-5)",
    "▸ Epochs: 100 (best model saved by validation AUC)",
    "▸ Batch size: 64 subgraphs per batch",
    "▸ Train/Test split: 80/20 (stratified by severity label)",
    "▸ Dropout: 0.3 (applied after each GCN layer + MLP)",
    "▸ Threshold: Youden's J statistic (optimal from ROC curve)",
]
add_bullet_frame(slide, Inches(0.8), Inches(3.3), Inches(5.5), Inches(3.5),
                 train_items, 12, LIGHT_GRAY, Pt(5))

# Explainability
add_textbox(slide, Inches(7.0), Inches(2.8), Inches(5.5), Inches(0.4),
            "EXPLAINABILITY METHODS", 16, True, WHITE)

explain_items = [
    "Method 1: GNNExplainer (200 epochs)",
    "  • Learns soft masks on features and edges",
    "  • Per-node: which drugs contribute most to risk",
    "  • Per-feature: which properties drive the prediction",
    "  • Per-edge: which drug-drug interactions matter",
    "",
    "Method 2: Gradient × Input Attribution",
    "  • Backpropagate through model to get ∂loss/∂input",
    "  • Multiply gradients by input features",
    "  • Per-drug importance = sum of |grad × input| per node",
    "  • Per-feature breakdown shows specific risk drivers",
    "",
    "Key output per drug:",
    "  → Importance score (0-1 normalized)",
    "  → Top contributing features (ATC, targets, actions)",
    "  → Drug-drug edge importance scores",
]
add_bullet_frame(slide, Inches(7.0), Inches(3.3), Inches(5.5), Inches(3.5),
                 explain_items, 11, LIGHT_GRAY, Pt(3))

# Model architecture note
add_textbox(slide, Inches(0.8), Inches(6.8), Inches(11.5), Inches(0.4),
            "Edge weights from TWOSIDES PRR scores flow through GCN convolution, "
            "allowing the model to learn interaction-strength-aware drug representations.",
            11, False, ACCENT3, PP_ALIGN.CENTER)


# ============================================================
# SLIDE 7: RESULTS — MODEL PERFORMANCE
# ============================================================
print("Creating Slide 7: Results (Performance)...")
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.5),
            "RESULTS — MODEL PERFORMANCE", 26, True, ACCENT1)
add_box_shape(slide, Inches(0.8), Inches(1.0), Inches(11.5), Inches(0.03), ACCENT1)

# Metrics from checkpoint
best_auc = checkpoint.get("best_auc", 0)
best_thresh = checkpoint.get("best_threshold", 0.5)

# Metric boxes
metrics = [
    ("AUC-ROC", f"{best_auc:.4f}", BLUE_ACC),
    ("Best Threshold", f"{best_thresh:.4f}", GREEN_ACC),
    ("Nodes / Edges", f"{model_config['num_nodes']} / {len(edges_df):,}", ORANGE_ACC),
    ("Feature Dim", f"{model_config['feat_dim']}", PURPLE_ACC),
]

for i, (label, value, color) in enumerate(metrics):
    x = Inches(0.8 + i * 3.1)
    add_box_shape(slide, x, Inches(1.3), Inches(2.8), Inches(1.2), color)
    add_textbox(slide, x + Inches(0.1), Inches(1.4), Inches(2.6), Inches(0.3),
                label, 13, False, WHITE, PP_ALIGN.CENTER)
    add_textbox(slide, x + Inches(0.1), Inches(1.75), Inches(2.6), Inches(0.5),
                value, 24, True, WHITE, PP_ALIGN.CENTER)

# Add ROC curve image if available
roc_path = os.path.join(RESULTS_DIR, "roc_curve.png")
if os.path.exists(roc_path):
    slide.shapes.add_picture(roc_path, Inches(0.8), Inches(2.8), Inches(5.5), Inches(4.2))
    add_textbox(slide, Inches(0.8), Inches(7.0), Inches(5.5), Inches(0.3),
                "ROC Curve with Optimal Threshold", 10, False, MED_GRAY, PP_ALIGN.CENTER)

# Confusion matrix
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
if os.path.exists(cm_path):
    slide.shapes.add_picture(cm_path, Inches(6.8), Inches(2.8), Inches(5.5), Inches(4.2))
    add_textbox(slide, Inches(6.8), Inches(7.0), Inches(5.5), Inches(0.3),
                "Confusion Matrix (Test Set)", 10, False, MED_GRAY, PP_ALIGN.CENTER)

if not os.path.exists(roc_path) or not os.path.exists(cm_path):
    add_textbox(slide, Inches(2), Inches(3.5), Inches(9), Inches(1),
                "⚠ Run generate_results.py first to generate plots",
                16, True, ACCENT2, PP_ALIGN.CENTER)


# ============================================================
# SLIDE 8: RESULTS — EXPLAINABILITY
# ============================================================
print("Creating Slide 8: Explainability Results...")
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.5),
            "RESULTS — EXPLAINABILITY & FEATURE ANALYSIS", 26, True, ACCENT1)
add_box_shape(slide, Inches(0.8), Inches(1.0), Inches(11.5), Inches(0.03), ACCENT1)

# Feature importance image
fi_path = os.path.join(RESULTS_DIR, "feature_importance.png")
if os.path.exists(fi_path):
    slide.shapes.add_picture(fi_path, Inches(0.8), Inches(1.3), Inches(6.0), Inches(3.5))

# Drug importance image
di_path = os.path.join(RESULTS_DIR, "drug_importance.png")
if os.path.exists(di_path):
    slide.shapes.add_picture(di_path, Inches(0.8), Inches(4.8), Inches(11.5), Inches(2.5))

# Prediction distribution
pd_path = os.path.join(RESULTS_DIR, "prediction_distribution.png")
if os.path.exists(pd_path):
    slide.shapes.add_picture(pd_path, Inches(7.2), Inches(1.3), Inches(5.2), Inches(3.5))

if not os.path.exists(fi_path):
    add_textbox(slide, Inches(2), Inches(3.5), Inches(9), Inches(1),
                "⚠ Run generate_results.py first to generate plots",
                16, True, ACCENT2, PP_ALIGN.CENTER)

# ============================================================
# SLIDE 9: RESULTS — DATASET STATS & CONCLUSION
# ============================================================
print("Creating Slide 9: Dataset Stats & Conclusion...")
slide = prs.slides.add_slide(blank_layout)
set_slide_bg(slide, DARK_BG)

add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.5),
            "DATASET STATISTICS & CONCLUSION", 26, True, ACCENT1)
add_box_shape(slide, Inches(0.8), Inches(1.0), Inches(11.5), Inches(0.03), ACCENT1)

# Dataset stats image
ds_path = os.path.join(RESULTS_DIR, "dataset_stats.png")
if os.path.exists(ds_path):
    slide.shapes.add_picture(ds_path, Inches(0.8), Inches(1.3), Inches(6.5), Inches(3.0))

# Conclusion
add_textbox(slide, Inches(7.8), Inches(1.3), Inches(5), Inches(0.4),
            "KEY FINDINGS", 16, True, WHITE)

conclusion_items = [
    f"✓ Model achieves AUC of {best_auc:.4f} on polypharmacy risk prediction",
    "✓ Multi-source data fusion (FAERS + TWOSIDES + SIDER + DrugCentral) provides rich drug representations",
    "✓ 65-dimensional node features capture pharmacological properties across multiple ontologies",
    "✓ GNNExplainer provides clinically interpretable per-drug risk attribution",
    "✓ Gradient attribution method offers complementary feature-level explanations",
    "✓ Interactive Streamlit frontend enables real-time drug combination risk assessment",
]
add_bullet_frame(slide, Inches(7.8), Inches(1.8), Inches(4.8), Inches(2.5),
                 conclusion_items, 11, LIGHT_GRAY, Pt(5))

# Future work
add_textbox(slide, Inches(7.8), Inches(4.5), Inches(5), Inches(0.4),
            "FUTURE WORK", 16, True, ACCENT3)

future_items = [
    "▸ Incorporate temporal patterns from sequential prescriptions",
    "▸ Add patient-level features (age, comorbidities, genetics)",
    "▸ Expand to multi-label side effect prediction (per-side-effect)",
    "▸ Integrate with electronic health records for real-time alerts",
    "▸ Validate on external datasets (UK MHRA, EudraVigilance)",
]
add_bullet_frame(slide, Inches(7.8), Inches(5.0), Inches(4.8), Inches(2.2),
                 future_items, 11, LIGHT_GRAY, Pt(5))

if not os.path.exists(ds_path):
    add_textbox(slide, Inches(1), Inches(5.0), Inches(5), Inches(1),
                "⚠ Run generate_results.py first for dataset stats plot",
                14, True, ACCENT2, PP_ALIGN.CENTER)


# ============================================================
# SAVE
# ============================================================
prs.save(OUTPUT_PATH)
print(f"\n{'='*60}")
print(f"Presentation saved: {OUTPUT_PATH}")
print(f"Total slides: {len(prs.slides)}")
print(f"{'='*60}")
