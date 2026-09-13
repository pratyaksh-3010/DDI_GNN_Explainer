# Polypharmacy Risk Reduction — HGT-Based Clinical Decision Support

> **Status:** Training active · Epoch ~29/50 · Val Accuracy 67.8% and rising

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Datasets](#2-datasets)
3. [Data Aggregation Pipeline](#3-data-aggregation-pipeline)
4. [Heterogeneous Knowledge Graph](#4-heterogeneous-knowledge-graph)
5. [Model Architecture](#5-model-architecture)
6. [Training Details](#6-training-details)
7. [What Has Been Built](#7-what-has-been-built)
8. [What Will Be Done Next](#8-what-will-be-done-next)
9. [Repository Structure](#9-repository-structure)
10. [How to Run](#10-how-to-run)

---

## 1. Project Overview

Polypharmacy — the concurrent use of multiple medications — is one of the leading causes of preventable hospital admissions. Patients on 5+ drugs face exponentially increasing risks of:
- **Adverse Drug Interactions (ADIs)** — drugs chemically interfering with each other
- **Side-effect amplification** — overlapping pharmacological mechanisms
- **Severity escalation** — unrecognised combinations tipping patients into critical states

This project builds an **AI-powered clinical decision support system** that:
1. Models patients, drugs, proteins, and side effects as a **Heterogeneous Knowledge Graph**
2. Trains a **Heterogeneous Graph Transformer (HGT)** to learn rich embeddings of each entity
3. Predicts **polypharmacy risk severity** (binary: high / low) for each patient
4. Predicts **which side effects** a patient's drug combination is likely to cause (multi-label, top-1000 SEs)
5. Provides **explainability** via HGT attention weights (which drug-drug edges drive risk)
6. Recommends **safer drug alternatives** via counterfactual simulation

---

## 2. Datasets

### 2.1 FAERS (FDA Adverse Event Reporting System)
- **Source:** US FDA Adverse Event Reporting System (FAERS)
- **What it provides:** Real adverse event report cases containing patient cases (`caseid`), severe outcomes (`SEVERITY`), structure ID lists (`STRUCTURE_ID_LIST`), drug combinations, and ATC level classification.
- **Scale used:** 89,118 unique patient adverse event cases with prescription/drug records.
- **Key fields used:**
  - Patient case ID -> graph node (`patient`)
  - Drug prescriptions -> `(patient, prescribed, drug)` edges (311,373 total)
  - Severe outcome label -> binary `patient.y` (high risk / severe outcome = 1, low risk = 0)
  - Note: NaN-safe BCE loss with `valid_mask` handles missing label values.

### 2.2 Bio-Decagon & TWOSIDES Datasets
- **Source:** Stanford Decagon Project (Marinka Zitnik et al.) & TWOSIDES
- **What it provides:**
  - `bio-decagon-targets.csv`: Drug to target gene mappings.
  - `bio-decagon-ppi.csv`: Human protein-protein interaction (PPI) network.
  - `bio-decagon-combo.csv` & `twosides_filtered.csv`: Polypharmacy drug-drug-side-effect combination interactions with PRR (Proportional Reporting Ratio) scores.
  - `bio-decagon-mono.csv`: Single-drug side effect profiles (mono-therapy).
- **Scale used:** Merged into 174,959 polypharmacy interaction edges and mono-therapy side effect associations.

### 2.3 STRING & MyGene Databases
- **Source:** STRING v11/v12 & MyGene.info API
- **What it provides:** Protein-protein interaction network mapped to HUGO Gene Symbols.
- **Scale used:** 2,020 unique target proteins; 22,023 PPI edges.

### 2.4 SIDER / OFFSIDES & DrugCentral Vocabulary Crosswalk
- **Source:** DrugCentral SQL Dump (`drugcentral.dump.11012023.sql`), RxNorm `rxcui_structure_map.csv`, SIDER, OFFSIDES.
- **What it provides:** Unified mapping between PubChem CIDs, Structure IDs, and RxCUI codes to harmonize identifiers across datasets.
- **Scale used:** Top 1,000 side effect categories; 1,920 unique drugs mapped to Morgan ECFP fingerprints / RxCUI identifiers.

---

## 3. Data Aggregation Pipeline

All raw data processing scripts live in `DATA_AGGREGATION_CODE/`:

```
Raw Sources (FAERS, Decagon, SIDER/OFFSIDES, TWOSIDES, DrugCentral, STRING)
    |
    +-- 1_build_vocab_crosswalk.py
    |       Parses drugcentral SQL dump & rxcui_structure_map.
    |       Maps PubChem CIDs to RxCUIs via Structure IDs.
    |       Output: DATA_AGGREGATION_CODE/stitch_to_rxcui_map.csv
    |
    +-- 2_merge_targets_and_ppi.py
    |       Queries MyGene API for Entrez ID -> HUGO Gene Symbol mapping.
    |       Merges Decagon drug targets with FAERS mechanism targets and PPI network.
    |       Outputs: DATA_AGGREGATION_CODE/faers_with_combined_targets.csv
    |                DATA_AGGREGATION_CODE/filtered_ppi.csv
    |
    +-- 3_merge_side_effects.py
    |       Unions TWOSIDES & Decagon polypharmacy combination edges with PRR scores.
    |       Merges single-drug (mono) side effects into FAERS records.
    |       Outputs: DATA_AGGREGATION_CODE/faers_layer3_final.csv
    |                DATA_AGGREGATION_CODE/combined_polypharmacy_edges.csv
    |
    +-- 4_construct_hetero_graph.py
            Extracts patient, drug, protein, and side_effect nodes and edge lists.
            Constructs and serialises PyTorch Geometric HeteroData graph.
            Output: V2/Processed/hetero_graph_data.pt
```

### Side-Effect Filtering
`V2/filter_top_1000_se.py` reduces 4,000+ raw side effect classes down to the **top-1000** by frequency across all drugs, keeping the graph tractable while retaining clinical relevance.

---

## 4. Heterogeneous Knowledge Graph

The final graph (`hetero_graph_data.pt`) schema:

### Node Types

| Node Type    | Count  | Feature Dim | Description |
|-------------|--------|-------------|-------------|
| `patient`   | 89,118 | 1           | Normalised drug-prescription degree (number of drugs / max drugs) |
| `drug`      | 1,920  | 1,024       | Morgan ECFP4 molecular fingerprint derived from SMILES |
| `protein`   | 2,020  | 1           | Degree feature (placeholder; STRING has no universal embedding) |
| `side_effect` | 1,000 | 1          | Degree feature (placeholder) |

### Edge Types — Original (5) + Reverse (3) = 8 Total

| Relation | Count | Description |
|----------|-------|-------------|
| `(patient, prescribed, drug)` | 311,373 | Patient was prescribed this drug |
| `(drug, prescribed_by, patient)` | 311,373 | **Reverse** — lets drugs propagate info back to patients |
| `(drug, targets, protein)` | 315,745 | Drug binds to this protein target |
| `(protein, targeted_by, drug)` | 315,745 | **Reverse** — protein informs drug embeddings |
| `(protein, interacts_with, protein)` | 22,023 | PPI network (bidirectional already) |
| `(drug, interacts_with, drug)` | 174,959 | DDI network with 2-dim interaction type attributes |
| `(drug, causes, side_effect)` | 792,546 | Drug causes this side effect (with severity score attr) |
| `(side_effect, caused_by, drug)` | 792,546 | **Reverse** — SE informs drug embeddings |

> **Critical design decision — Reverse Edges:**
> HGTConv only computes output embeddings for nodes that appear as *destinations* in at least one edge type. Patient nodes only appeared as *sources* (patient->drug), meaning HGT never updated patient embeddings — they stayed as their initial 1-dim degree feature projection, and the model defaulted to predicting the majority class (57.83%). Adding reverse edges `(drug, prescribed_by, patient)` allows drug embeddings — which encode protein targets and side-effect profiles — to flow back into patient representations via attention-weighted message passing. This single fix raised accuracy from 57.83% -> 67%+.

---

## 5. Model Architecture

**File:** [`V2/hetero_attention_model.py`](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/V2/hetero_attention_model.py)

```
ExplainableHeteroGNN (290,905 trainable parameters)
|
+-- Input Projections (LazyLinear, one per node type)
|     patient     :    1 -> 128-dim + ReLU
|     drug        : 1024 -> 128-dim + ReLU
|     protein     :    1 -> 128-dim + ReLU
|     side_effect :    1 -> 128-dim + ReLU
|
+-- HGT Layer 1  (HGTConv, 128-dim hidden, 4 attention heads)
|     - Multi-head attention across all 8 relation types simultaneously
|     - Each head learns different relation-specific attention patterns
|     - Residual fallback: if a node type is not a destination in any edge,
|       its pre-layer features are preserved (prevents KeyError for patient nodes)
|
+-- HGT Layer 2  (HGTConv, 128-dim hidden, 4 attention heads)
|     - Second-order neighbourhood aggregation
|     - Patient embeddings now encode 2-hop info: patient->drug->protein/SE
|
+-- Task Head 1: Severity Prediction (binary classification)
|     Linear(128->64) -> ReLU -> Dropout(0.3) -> Linear(64->1)
|     Training: BCEWithLogitsLoss, NaN labels masked via valid_mask
|     Inference: sigmoid > 0.5 -> high risk
|
+-- Task Head 2: Side-Effect Multi-Label (1000 classes)
      Linear(128->128) -> ReLU -> Dropout(0.3) -> Linear(128->1000)
      Training: BCEWithLogitsLoss with pos_weight=0.09
                (corrects 92% positive-label density imbalance)
      Inference: sigmoid per class -> top-k SEs predicted
```

### Explainability via Attention Hooks
- `register_forward_hook` is attached to each `HGTConv` layer during `__init__`
- After every forward pass, `module._alpha` (shape: `{rel_type: (E, num_heads)}`) is captured
- Stored in `model.attention_dict[layer_idx]` — available without modifying HGTConv internals
- The `explainer_and_recommender.py` reads these weights to rank which drug-drug or drug-protein edges most influenced each patient's risk prediction

---

## 6. Training Details

**File:** [`V2/train_hetero.py`](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/V2/train_hetero.py)

| Hyperparameter | Value | Rationale |
|---|---|---|
| Hidden channels | 128 | Sufficient for 4-type HG with 8 relation types |
| HGT layers | 2 | Covers 2-hop patient->drug->protein/SE neighbourhood |
| Attention heads | 4 | Different heads specialise in different relation types |
| Learning rate | 1e-3 (Adam) | With CosineAnnealingLR decay to 1e-5 over 50 epochs |
| Batch size | 256 | NeighborLoader with [15, 5] fanout per hop |
| Epochs | 50 | Model checkpoint saved at best validation loss |
| Severity loss weight | 1.0 | Primary clinical task |
| SE loss weight | 0.1 | Downweighted: SE labels are 92% positive |
| SE pos_weight | 0.09 | ~(1-0.92)/0.92, corrects label imbalance |
| Gradient clipping | 1.0 (norm) | Prevents gradient explosion |
| num_workers | 0 | Windows: multiprocessing spawn re-imports script without `__main__` guard |

### Bugs Found and Fixed During Development

| Bug | Symptom | Root Cause | Fix |
|-----|---------|-----------|-----|
| `AttributeError: 'EdgeStorage' has no attribute 'edge_index'` | Crash at label building | Accessing non-existent edge type creates empty EdgeStorage, breaking NeighborLoader CSC builder | Check `edge_type in data.edge_types` before accessing |
| `UnicodeEncodeError` on `->` arrows | Crash on Windows terminal | Windows cp1252 console can't encode Unicode `->` | Replaced all `->` with ASCII `->` |
| Script runs 5x in parallel | Windows multiprocessing crash | `num_workers > 0` spawns new processes that re-import the full script | Wrap all execution in `if __name__ == '__main__'`; set `num_workers=0` |
| `ImportError: 'NeighborSampler' requires pyg-lib or torch-sparse` | Crash at loader creation | Missing optional PyG C++ backend | `pip install torch-scatter torch-sparse` from PyG wheels |
| `KeyError: 'patient'` in HGTConv | Crash at first forward pass | Patient is source-only; HGTConv drops it from output dict | Residual fallback loop after each conv layer |
| `loss=nan` every epoch | No learning at all | `patient.x` all-ones (zero signal); `patient.y` contains NaN entries | Replace x with degree features; mask NaN y with `valid_mask` |
| Accuracy stuck at 57.83% (majority class) | No learning above baseline | Patient nodes never receive messages (source-only) | Add 3 reverse edge types for bidirectional message passing |

### Training Progress

```
Run 1 (no reverse edges):
  Epoch 1-30: loss=nan every epoch -> fixed patient.x + NaN masking
  Epoch 1-30: loss~0.689, accuracy=57.83% stuck -> needed reverse edges

Run 2 (with reverse edges, current):
Epoch | Train Loss | Val Loss  | Val Acc
------+------------+-----------+---------
  1   |   ~0.68    |  ~0.68    |  ~60.0%
  5   |   ~0.65    |  ~0.65    |  ~63.0%
 10   |   0.6199   |  0.6255   |  65.35%
 16   |   0.6073   |  0.6160   |  66.65%  <- saved
 18   |   0.6023   |  0.6121   |  66.81%  <- saved
 24   |   0.5876   |  0.6064   |  67.60%  <- saved
 28   |   0.5818   |  0.6040   |  67.83%  <- saved
 ... (training ongoing, target: 70%+)
```

---

## 7. What Has Been Built

### Data Pipeline
- [x] Vocabulary crosswalk unifying DrugBank / STRING / SIDER identifiers
- [x] Drug-protein target edge construction
- [x] Protein-protein interaction edge construction
- [x] Drug-side-effect edge construction (SIDER + OFFSIDES)
- [x] FAERS patient adverse event report integration
- [x] Heterogeneous graph construction and serialisation (`hetero_graph_data.pt`)
- [x] Top-1000 side-effect filtering (`filter_top_1000_se.py`)
- [x] Drug SMILES -> Morgan fingerprint extraction (`extract_smiles.py`)

### Model
- [x] `ExplainableHeteroGNN` with dual prediction heads (severity + 1000-class side effects)
- [x] Attention weight capture via forward hooks (no HGTConv source modification)
- [x] Residual fallback for source-only node types across HGT layers
- [x] Reverse edges enabling bidirectional message passing

### Training Infrastructure
- [x] CUDA GPU training (RTX 4060 Laptop, 8GB VRAM)
- [x] NaN-safe BCE loss with `valid_mask` patient masking
- [x] SE loss with `pos_weight` for 92% label density correction
- [x] Gradient clipping (norm=1.0)
- [x] CosineAnnealingLR scheduler
- [x] Best-epoch model checkpointing -> `hgt_model.pth`
- [x] All Windows compatibility fixes (multiprocessing, encoding, PyG deps)

### Explainer & Recommender (framework)
- [x] `explainer_and_recommender.py` — loads model + graph, extracts attention weights
- [x] Counterfactual simulation — clone subgraph, swap drug, re-run HGT, compare risk scores
- [x] Drug safety scoring based on DDI edge attributes

---

## 8. What Will Be Done Next

### Step A — Complete Training (In Progress ~Epoch 28/50)
- [ ] Wait for 50-epoch run to finish
- [ ] Evaluate final Val Accuracy and Val Loss (target: Acc >= 70%, Loss < 0.60)
- [ ] If plateau reached: continue training with lower LR or richer protein/SE node features

### Step B — Testing & Evaluation
- [ ] Run `explainer_and_recommender.py` with saved `hgt_model.pth`
- [ ] Validate attention extraction on 10 sample patients
- [ ] Compute AUROC and F1 for severity head
- [ ] Compute precision@10 and AUROC for side-effect multi-label head
- [ ] Verify: counterfactual drug swap lowers predicted severity score

### Step C — Streamlit Frontend Overhaul
**File:** `STREAMLIT_FRONTEND/app.py` — complete rewrite using HGT model only

Planned UI panels:

| Panel | Description |
|-------|-------------|
| Patient Search | Search by ID; shows drug list, severity score, confidence |
| Risk Dashboard | HGT severity probability gauge; risk tier (low/medium/high) |
| Drug Network | Interactive force-directed graph of prescribed drugs + DDI edges coloured by interaction type |
| Side-Effect Radar | Top-20 predicted SEs with probability bars; filter by organ system |
| Attention Heatmap | Bar chart of top-10 drug-drug and drug-protein edges ranked by attention weight |
| Drug Swap Recommender | Select a drug to remove/replace; shows alternative drugs and predicted new risk score |

### Step D — End-to-End Integration
- [ ] Load `hgt_model.pth` in Streamlit with `@st.cache_resource`
- [ ] Cache graph object in session state
- [ ] Build patient-lookup index from graph node IDs
- [ ] Wire all UI panels to live model inference

### Step E — Final Validation & Documentation
- [ ] Confusion matrix and ROC curve plots
- [ ] Side-effect precision-recall curves
- [ ] Explainability case studies (2-3 real patients)
- [ ] Final README update with benchmark numbers

---

## 9. Repository Structure

```
HEALTHCARE/
|
+-- README.md                             <- This file
+-- polypharmacy_risk_reduction_PRD.md    <- Full product requirements document
|
+-- DATA_AGGREGATION_CODE/
|   +-- 1_build_vocab_crosswalk.py        <- Harmonise drug/protein/SE identifiers
|   +-- 2_merge_targets_and_ppi.py        <- Drug-protein + PPI edges
|   +-- 3_merge_side_effects.py           <- Drug-SE edges (SIDER + OFFSIDES)
|   +-- 4_construct_hetero_graph.py       <- Build final HeteroData .pt file
|   +-- Graph_Data/
|       +-- edges_drug_protein.csv
|       +-- edges_protein_protein.csv
|       +-- edges_drug_side_effect.csv
|       +-- (other intermediate CSVs)
|
+-- V2/
|   +-- hetero_attention_model.py         <- ExplainableHeteroGNN (HGT architecture)
|   +-- train_hetero.py                   <- CUDA training script (main entry point)
|   +-- explainer_and_recommender.py      <- Attention extraction + counterfactuals
|   +-- dataset_builder.py               <- Dataset loading utilities
|   +-- filter_top_1000_se.py            <- Filter to top-1000 side effects
|   +-- extract_smiles.py               <- DrugBank SMILES -> Morgan fingerprint
|   +-- Processed/
|       +-- hetero_graph_data.pt          <- Final heterogeneous knowledge graph
|       +-- hgt_model.pth                <- Best model weights (auto-saved during training)
|       +-- training_config.json         <- Hyperparameters log
|
+-- STREAMLIT_FRONTEND/
    +-- app.py                            <- Clinical UI (to be overhauled in Step C)
```

---

## 10. How to Run

### Prerequisites
```bash
# PyTorch + CUDA (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# PyTorch Geometric
pip install torch_geometric

# PyG C++ backends (required for NeighborSampler)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

# Other
pip install streamlit pandas numpy scikit-learn rdkit
```

### Step 1 — Build the Graph (skip if `hetero_graph_data.pt` exists)
```bash
cd DATA_AGGREGATION_CODE
python 1_build_vocab_crosswalk.py
python 2_merge_targets_and_ppi.py
python 3_merge_side_effects.py
python 4_construct_hetero_graph.py
cd ..
python V2/filter_top_1000_se.py
```

### Step 2 — Train the HGT Model
```bash
# Windows: must be run as a script (not interactively) due to multiprocessing
python V2/train_hetero.py
# Saves best model to: V2/Processed/hgt_model.pth
# Saves config to:     V2/Processed/training_config.json
```

### Step 3 — Run Explainer & Recommender
```bash
python V2/explainer_and_recommender.py
```

### Step 4 — Launch the Streamlit UI
```bash
streamlit run STREAMLIT_FRONTEND/app.py
```

---

## Hardware Requirements
| Component | Minimum | Tested On |
|-----------|---------|-----------|
| GPU | 6 GB VRAM, CUDA 12.x | NVIDIA RTX 4060 Laptop (8 GB) |
| RAM | 16 GB | 16 GB |
| Storage | 5 GB | SSD recommended |
| Python | 3.11 | 3.11.x |
| PyTorch | 2.x | 2.11.0+cu128 |
| PyG | 2.x | 2.7.0 |

---

*Last updated: September 2026 | HGT v2 | PyTorch 2.11.0+cu128 | PyG 2.7.0*
