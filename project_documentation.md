# Polypharmacy Risk Prediction Using Graph Neural Networks — Complete Project Documentation

---

## 1. Project Overview

### 1.1 What Is This Project?

This project builds an **end-to-end machine learning pipeline** that predicts the **risk of adverse drug interactions** (polypharmacy risk) when a patient takes multiple medications simultaneously. It uses a **Graph Neural Network (GNN)** trained on real-world pharmacological data, and provides **explainable AI (XAI)** outputs so clinicians can understand *why* a particular drug combination is risky.

### 1.2 The Problem: Polypharmacy

**Polypharmacy** — the concurrent use of 2+ medications — is extremely common, especially in elderly patients with multiple chronic conditions. When drugs are taken together, they can interact in unpredictable ways, leading to:
- **Adverse Drug Reactions (ADRs)**: Nausea, organ damage, hospitalizations
- **Drug-Drug Interactions (DDIs)**: One drug amplifying or blocking another
- **Life-threatening outcomes**: Death, disability, hospitalization

Existing clinical tools (e.g., drug interaction checkers) rely on **pairwise** lookups — they check Drug A vs Drug B. But when a patient takes 5+ drugs, the **combinatorial explosion** of possible interactions makes pairwise checking impractical. This project uses **graph-based deep learning** to analyze the *entire combination* holistically.

### 1.3 Why Graph Neural Networks?

Drugs and their interactions naturally form a **graph**:
- **Nodes** = Individual drugs (with pharmacological features)
- **Edges** = Known drug-drug interactions (with severity weights)

GNNs can learn from this graph structure to:
1. Propagate information between connected drugs (message passing)
2. Capture higher-order interactions (Drug A affects Drug B, which changes how Drug C behaves)
3. Pool information across all drugs in a combination to predict overall risk

This is fundamentally more powerful than pairwise lookups because the GNN considers the **entire neighborhood** of interactions simultaneously.

---

## 2. Data Sources — In Depth

### 2.1 FAERS (FDA Adverse Event Reporting System)

| Attribute | Detail |
|-----------|--------|
| **Source** | FDA's public database of adverse event reports |
| **Quarters Used** | 2023 Q1, Q2, Q3, Q4 |
| **Format** | `$`-delimited text files (DEMO, DRUG, OUTC, REAC tables) |
| **Purpose** | Provides real-world cases of patients taking multiple drugs and their outcomes |

**How it's used:**
- The **DRUG** table lists which drugs each patient (case) was taking.
- The **OUTC** table lists outcomes (death, hospitalization, disability, etc.)
- We filter for **polypharmacy cases** (2+ drugs per case) and create a binary **severity label**:
  - `SEVERITY = 1` (HIGH RISK): if any outcome is Death (DE), Life-Threatening (LT), Hospitalization (HO), or Disability (DS)
  - `SEVERITY = 0` (LOW RISK): otherwise

**Why FAERS?** It's the gold standard for post-market drug safety surveillance. It contains millions of real patient cases with actual outcomes, making it ideal for supervised learning.

### 2.2 RxNorm (Drug Name Normalization)

| Attribute | Detail |
|-----------|--------|
| **Source** | NIH National Library of Medicine |
| **File** | `RXNCONSO.RRF` from `RxNorm_full_prescribe_02022026` |
| **Purpose** | Normalize messy drug names to standardized RxCUI identifiers |

**Why it's needed:** FAERS drug names are free-text and extremely messy:
- "ASPIRIN 325MG TABLET" → RxCUI for Aspirin
- "aspirin" → Same RxCUI
- "ASA" → Same RxCUI

The normalization process:
1. Load RxNorm's `RXNCONSO.RRF` file (only English, RxNorm source, Ingredient/Precise Ingredient term types)
2. Apply string normalization: uppercase, strip dosage forms (MG, ML, TABLET, etc.), collapse whitespace
3. Build a lookup dictionary mapping normalized strings → RxCUI
4. Map each FAERS drug name to its RxCUI

### 2.3 DrugCentral (Drug Targets & Mechanisms)

| Attribute | Detail |
|-----------|--------|
| **Source** | DrugCentral database |
| **Files** | [rxcui_structure_map.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/rxcui_structure_map.csv), [structure_atc_map.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/structure_atc_map.csv), [structure_mechanism_map.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/structure_mechanism_map.csv) |
| **Purpose** | Map drugs to their ATC classifications, gene targets, and mechanisms of action |

The mapping chain: `RxCUI → structure_id → ATC codes, target genes, target classes, action types`

**Why DrugCentral?** It provides the **pharmacological features** for each drug — what therapeutic class it belongs to, what proteins it targets, and how it acts on those targets. These features are critical for the GNN to learn *why* certain drug combinations are dangerous.

### 2.4 SIDER (Side Effect Resource)

| Attribute | Detail |
|-----------|--------|
| **Source** | SIDER database (side effects of drugs) |
| **Files** | `meddra_all_se.tsv`, [drug_names.tsv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/drug_names.tsv) |
| **Purpose** | Known side effects for each individual drug |

**Processing:**
1. Load SIDER's side effect data (MedDRA Preferred Terms only)
2. Map SIDER drug names → RxCUI using the same RxNorm normalization
3. Map RxCUI → structure_id via DrugCentral
4. Attach known side effects to each FAERS case based on which drugs are present

**Why SIDER?** Drugs with many known side effects individually are more likely to cause problems in combination. The count of known side effects becomes a node feature.

### 2.5 TWOSIDES (Drug-Drug Interaction Database)

| Attribute | Detail |
|-----------|--------|
| **Source** | TWOSIDES database |
| **Original Size** | ~4.3 GB CSV |
| **Key Metric** | PRR (Proportional Reporting Ratio) |
| **Purpose** | Provides the **edges** of the drug interaction graph |

**Processing:**
1. Load the full TWOSIDES dataset with columns: `drug1_rxcui`, `drug2_rxcui`, `side_effect`, `PRR`
2. Filter to keep only drug pairs where **both** drugs appear in our FAERS dataset
3. Filter for `PRR > 1` (interaction is reported more often than expected by chance)
4. The **maximum PRR** across all side effects for each drug pair becomes the **edge weight**

**Why TWOSIDES?** It's the largest database of **observed** drug-drug interactions with quantitative severity scores. The PRR tells us how much more frequently a side effect occurs with the drug pair than expected — higher PRR = more dangerous interaction.

---

## 3. Architecture: The 6-Layer Pipeline

The project is organized into 6 logical **layers**, each handling a distinct stage of the pipeline.

### 3.1 Layer 1-2: Data Preprocessing (`DATA_PREPROCESSING_CODE/`)

This layer transforms raw data into a unified, enriched dataset. It consists of **8 scripts** run sequentially:

#### Script 1: [faers_filtering.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/DATA_PREPROCESSING_CODE/faers_filtering.py) (×4 quarters)
- **Input**: Raw FAERS quarterly files (DEMO, DRUG, OUTC, REAC)
- **Process**:
  1. Filter drugs to only Primary Suspect (PS) and Secondary Suspect (SS) roles
  2. Remove cases with unnamed drugs
  3. Keep only **polypharmacy cases** (≥2 unique drugs)
  4. Compute severity label from outcome codes: `{DE, LT, HO, DS}` → severe
  5. Group drugs per case into a list
- **Output**: `modifiedfaersqN.csv` (one per quarter)
- **Why PS/SS only?** Concomitant medications are less likely to be causally related to the adverse event.

#### Script 2: [combine_faers.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/DATA_PREPROCESSING_CODE/combine_faers.py)
- **Process**: Concatenates all 4 quarterly CSVs
- **Output**: [merged_faers.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/merged_faers.csv) (~46 MB)

#### Script 3: [faers_rxnorm_normalisation.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/DATA_PREPROCESSING_CODE/faers_rxnorm_normalisation.py)
- **Process**: Maps free-text drug names → RxCUI using RxNorm
- **Key Design Decisions**:
  - Only keep Ingredient (IN) and Precise Ingredient (PIN) term types to avoid duplicates from different formulations
  - Normalize strings by removing dosage forms, uppercasing, stripping special characters
  - Drop cases with <2 mappable drugs (they're no longer polypharmacy)
- **Output**: [faers_rxnorm.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/faers_rxnorm.csv)

#### Script 4: [faers_drugcentral_mapping.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/DATA_PREPROCESSING_CODE/faers_drugcentral_mapping.py)
- **Process**: Maps RxCUI → DrugCentral structure_id
- **Output**: [faers_drugcentral.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/faers_drugcentral.csv)
- **Why structure_id?** It's DrugCentral's canonical drug identifier, needed to access ATC, target, and mechanism data.

#### Script 5: [structure_atc_mapping.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/DATA_PREPROCESSING_CODE/structure_atc_mapping.py)
- **Process**: Attaches ATC codes and Level-1 ATC categories to each case
- **ATC Level-1**: Single letter (A-V) indicating therapeutic area (e.g., N=Nervous System, C=Cardiovascular)
- **Output**: [faers_with_atc.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/faers_with_atc.csv)

#### Script 6: [Structure_id_mechanism_mapping.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/DATA_PREPROCESSING_CODE/Structure_id_mechanism_mapping.py)
- **Process**: Attaches target genes and target classes to each case
- **Output**: [faers_with_mechanism.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/faers_with_mechanism.csv)

#### Script 7: [sider_processing.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/DATA_PREPROCESSING_CODE/sider_processing.py)
- **Process**: Maps SIDER side effects to structure_ids and attaches known side effects to each FAERS case
- **Output**: [faers_layer2_final.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/faers_layer2_final.csv) (~606 MB — the complete enriched dataset)

#### Script 8: [twosides_processing.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/DATA_PREPROCESSING_CODE/twosides_processing.py)
- **Process**: Filters TWOSIDES to drugs in our FAERS dataset, keeps PRR > 1
- **Output**: [twosides_filtered.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/twosides_filtered.csv) (~1.2 GB)

### 3.2 Layer 3: Graph Construction ([LAYER _3_CODE/nodes_edges.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/LAYER%20_3_CODE/nodes_edges.py))

This is the core **feature engineering** step that transforms tabular data into a graph suitable for GNN training.

**Node Construction (1,198 drug nodes):**

Each drug node has a **65-dimensional feature vector**:

| Feature Group | Dimensions | Encoding | Description |
|---------------|-----------|----------|-------------|
| ATC Level-1 | 14 | Multi-hot | Therapeutic classification (A-V) |
| Target Class | 25 | Multi-hot | Pharmacological target type (GPCR, Enzyme, Ion channel, etc.) |
| Action Type | 23 | Multi-hot | Mechanism of action (AGONIST, ANTAGONIST, INHIBITOR, etc.) |
| Count Features | 3 | Log-scaled | log1p(ATC count), log1p(gene count), log1p(side effect count) |

**Why multi-hot encoding?** A single drug can belong to multiple ATC categories, target multiple protein classes, and have multiple action types. Multi-hot captures this multiplicity.

**Why log-scaling counts?** Drug counts spanning orders of magnitude (1 to 1000+). Log scaling prevents high-count drugs from dominating the feature space.

**Edge Construction (32,582 edges):**
- Source: TWOSIDES drug pairs mapped to structure_ids
- Weight: `log1p(max_PRR)` — log-scaled maximum Proportional Reporting Ratio
- Edges are **undirected** (stored once, expanded to bidirectional during training)
- Self-loops are excluded

**Training Cases (92,510 total):**
- Each case = a list of node indices (the drugs the patient was taking) + severity label
- Only cases where ≥2 drugs have known edges are kept as valid subgraphs → **67,959 valid subgraphs**

**Class Distribution:**
- High risk: 38,868 (57.2%)
- Low risk: 29,091 (42.8%)

**Outputs:**
- [layer3_nodes.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/layer3_nodes.csv) — Node feature matrix
- [layer3_edges_weighted.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/layer3_edges_weighted.csv) — Edge list with weights
- [layer3_training_cases.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/layer3_training_cases.csv) — Per-case drug lists and labels
- [layer3_feature_meta.json](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/layer3_feature_meta.json) — Feature metadata (names, dimensions, categories)

### 3.3 Layer 4: GNN Training ([LAYER_4_CODE/train_gnn.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/LAYER_4_CODE/train_gnn.py))

#### Model Architecture: WeightedGCN

```
Input: Node features (65-dim) + Learnable embeddings (32-dim) = 97-dim per node
  ↓
GCNConv Layer 1: 97 → 128 (with edge weights) + ReLU + Dropout(0.3)
  ↓
GCNConv Layer 2: 128 → 64 (with edge weights) + ReLU + Dropout(0.3)
  ↓
Global Mean Pooling (aggregate all node representations)
  ↓
Linear: 64 → 32 + ReLU + Dropout(0.3)
  ↓
Linear: 32 → 1 (raw logits, NO sigmoid)
```

**Total Parameters: 61,249**

**Key Design Decisions:**

1. **Learnable Node Embeddings (32-dim)**: Each of the 1,198 drugs gets its own learnable 32-dimensional embedding vector. This allows the model to learn drug-specific representations beyond what the hand-crafted features capture. The embedding is concatenated with the 65 feature dimensions → 97-dim input.

2. **GCNConv with Edge Weights**: Standard GCN propagation uses the adjacency matrix. By passing `edge_weight` (log-scaled PRR), the model learns that strongly interacting drug pairs (high PRR) should exchange more information during message passing.

3. **Raw Logits Output**: The model outputs raw logits (not probabilities). This is required by `BCEWithLogitsLoss` which applies sigmoid internally for numerical stability (avoids log(sigmoid(x)) precision issues).

4. **BCEWithLogitsLoss with Class Weighting**: Since high-risk cases outnumber low-risk cases (57% vs 43%), the loss function uses `pos_weight = n_neg / n_pos` to give more importance to the minority class during training.

5. **Subgraph Construction**: For each FAERS case, a **subgraph** is extracted from the global drug graph containing only the drugs in that case and the edges between them. Cases with no edges between their drugs are skipped (the GNN needs at least some structure to learn from).

6. **Optimal Threshold (Youden's J)**: Instead of using 0.5 as the classification threshold, the model uses Youden's J statistic (`max(TPR - FPR)`) on the ROC curve to find the optimal threshold. Final optimal threshold: **0.5032**.

**Training Configuration:**
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Epochs: 100
- Batch size: 64
- Train/Test split: 80/20 (stratified)
- Device: CUDA if available, else CPU

**Test Set Results (13,592 samples):**

| Metric | Value |
|--------|-------|
| **AUC** | **0.8159** |
| **Accuracy** | 0.7219 |
| **Precision** | 0.8082 |
| **Recall** | 0.6737 |
| **F1 Score** | 0.7348 |

**Confusion Matrix:**
```
              Predicted
              Low    High
Actual Low    4575   1243
Actual High   2537   5237
```

### 3.4 Layer 5: Explainability ([LAYER_5_CODE/explain_gnn.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/LAYER_5_CODE/explain_gnn.py))

This is a critical layer that makes the model's predictions **interpretable** using two complementary XAI methods:

#### Method 1: GNNExplainer

**What it does**: GNNExplainer learns **soft masks** over node features and edges that maximize the mutual information between the masked subgraph and the model's prediction. In simpler terms, it finds the minimal set of features and connections that explain why the model made its prediction.

**Configuration**:
```python
Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',      # Learn which features matter
    edge_mask_type='object',           # Learn which edges matter
    model_config=ModelConfig(
        mode='binary_classification',
        task_level='graph',
        return_type='raw'              # Model outputs raw logits
    )
)
```

**Outputs**:
- **Per-drug importance**: Sum of feature mask values per node → normalized to show each drug's contribution percentage
- **Per-feature importance**: Which features (ATC class, target type, action type) drive the prediction
- **Edge importance**: Which drug-drug interaction edges are most influential

#### Method 2: Gradient × Input Attribution

**What it does**: Computes the gradient of the model's output with respect to the input features, then multiplies by the input values. This gives a first-order approximation of how much each input feature contributes to the output.

**Process**:
1. Clone input features with `requires_grad=True`
2. Forward pass through the model
3. Backpropagate
4. Compute `|gradient × input|` for each feature of each drug
5. Sum over features per drug → per-drug importance
6. Normalize to get relative contributions

**Why two methods?** GNNExplainer and Gradient Attribution capture different aspects of importance. GNNExplainer is optimization-based (learns what to mask), while Gradient Attribution is analytical (computes sensitivity). Agreement between methods increases confidence in explanations.

### 3.5 Layer 6: Utility Scripts

#### [build_drug_lookup.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/LAYER_6/build_drug_lookup.py)
Maps structure_id → human-readable drug names via the chain: `structure_id → rxcui → RxNorm canonical name`. Creates [structure_name_lookup.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/structure_name_lookup.csv).

#### [enrich_nodes.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/LAYER_6/enrich_nodes.py)
Merges drug names into [layer3_nodes.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/layer3_nodes.csv) and adds `node_index` column for frontend use. Creates [layer3_nodes_enriched.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/layer3_nodes_enriched.csv).

#### [find_example_pairs.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/LAYER_6/find_example_pairs.py)
Uses NetworkX to find cliques (≥3 drugs all mutually interacting) in the drug graph — useful for finding interesting polypharmacy test cases.

---

## 4. Streamlit Frontend

### 4.1 Architecture

The frontend consists of 3 files:

| File | Purpose |
|------|---------|
| [app.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/STREAMLIT_FRONTEND/app.py) | Main Streamlit UI — drug selection, prediction display, explainability tabs |
| [model_utils.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/STREAMLIT_FRONTEND/model_utils.py) | Model loading, prediction, GNNExplainer, gradient attribution |
| [graph_builder.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/STREAMLIT_FRONTEND/graph_builder.py) | Builds PyG subgraphs from user-selected drug names |

### 4.2 How It Works

1. **Drug Selection**: User selects drugs via multiselect dropdown (1,198 drugs available) or free-text comma-separated input
2. **Graph Building** ([graph_builder.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/STREAMLIT_FRONTEND/graph_builder.py)):
   - Look up each drug name → node_index (case-insensitive)
   - Extract subgraph: features for selected nodes + edges between them from adjacency list
   - If no edges exist, add self-loops with weight 0.1 (so GNN can still process)
3. **Prediction** ([model_utils.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/STREAMLIT_FRONTEND/model_utils.py)):
   - Forward pass through loaded WeightedGCN model
   - Apply sigmoid to get probability
   - Display risk level with color coding (>0.7 HIGH, >0.5 MODERATE-HIGH, >0.3 MODERATE-LOW, else LOW)
4. **Explainability** (3 tabs):
   - **Tab 1**: GNNExplainer per-drug importance with feature breakdowns
   - **Tab 2**: Gradient attribution per-drug importance with feature breakdowns
   - **Tab 3**: Drug-drug interaction (edge) importance table

### 4.3 Feature Beautification

Raw feature names like `atc_l1_N` are converted to human-readable labels like `ATC: Nervous System` using [_beautify_feature()](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/STREAMLIT_FRONTEND/model_utils.py#216-248). All 14 ATC categories, target classes, and action types have readable names.

---

## 5. Pipeline Orchestration & Validation

### 5.1 [run_pipeline.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/run_pipeline.py)
Runs the full pipeline in order:
1. Layer 3 feature engineering → graph files
2. Drug name lookup → [structure_name_lookup.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/structure_name_lookup.csv)
3. Node enrichment → [layer3_nodes_enriched.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/layer3_nodes_enriched.csv)
4. GNN training → [weighted_gcn_model.pth](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/weighted_gcn_model.pth)
5. Copy all required files to `STREAMLIT_FRONTEND/DATA/`

### 5.2 `validate_pipeline.py`
Comprehensive validation with 15+ automated checks:
- All required data files exist
- Feature dimensions match across files
- Edge indices are in valid range, no self-loops, positive weights
- Both classes present in training data
- Model checkpoint contains required keys
- Model `num_nodes` and `feat_dim` match data
- All predictions are valid probabilities (0-1)
- Predictions have variance (model hasn't collapsed)
- AUC > 0.5 (better than random)
- F1 > 0 (model produces both predictions)

### 5.3 `generate_results.py`
Generates 6 publication-quality visualizations and 2 text reports:
1. **ROC Curve** with optimal threshold marker
2. **Confusion Matrix** heatmap
3. **Feature Importance** bar chart (top 15 features by gradient attribution)
4. **Dataset Statistics** (class distribution + drugs-per-case histogram)
5. **Per-Drug Importance** example cases
6. **Prediction Distribution** histogram by true label

---

## 6. Problems Faced & Solutions

### 6.1 Drug Name Normalization Chaos
**Problem**: FAERS drug names are notoriously messy — misspellings, varied capitalizations, dosage forms embedded in names, brand names instead of generics.
**Solution**: Aggressive string normalization (uppercase, strip dosage forms like MG/ML/TABLET, regex cleanup) + RxNorm's canonical ingredient mapping. Even so, many drugs fail to map, reducing dataset size.

### 6.2 Multi-Database ID Mapping
**Problem**: Each database uses different identifiers — FAERS uses free-text names, RxNorm uses RxCUI, DrugCentral uses structure_id, SIDER uses STITCH IDs, TWOSIDES uses RxNorm IDs.
**Solution**: Built a multi-hop mapping chain: `drug_name → RxCUI → structure_id → ATC/targets/mechanisms`. Each hop loses some drugs that can't be mapped, which is why we start with hundreds of thousands of FAERS cases but end with ~92K training cases.

### 6.3 Class Imbalance
**Problem**: High-risk cases (57.2%) outnumber low-risk cases (42.8%). While not severe, this can bias the model toward predicting high risk.
**Solution**: Used `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos` to upweight the minority class during training. Also used Youden's J statistic instead of a fixed 0.5 threshold for optimal classification.

### 6.4 Subgraphs with No Edges
**Problem**: Many FAERS cases contain drugs that have no known interactions in TWOSIDES. These cases produce subgraphs with nodes but no edges, which GCNConv can't meaningfully process.
**Solution**: During training, skip cases with no edges (67,959 valid out of 92,510 total). In the frontend, add self-loops with low weight (0.1) so the model can still produce a prediction based on node features alone.

### 6.5 Model Outputting Sigmoid vs Raw Logits
**Problem**: Initially there was confusion about whether the model should output probabilities (with sigmoid) or raw logits. `BCEWithLogitsLoss` expects raw logits, but `GNNExplainer` needs to know the output type.
**Solution**: Model always outputs raw logits. Sigmoid is applied: (a) inside `BCEWithLogitsLoss` automatically during training, (b) explicitly during inference with `torch.sigmoid()`, (c) `GNNExplainer` configured with `return_type='raw'`.

### 6.6 GNNExplainer Configuration
**Problem**: GNNExplainer's API requires specifying `ModelConfig` with the correct `return_type`. If set to `'probs'` when the model returns logits (or vice versa), the explanations are meaningless.
**Solution**: Set `return_type='raw'` in ModelConfig since WeightedGCN outputs raw logits. The explainer handles the sigmoid internally.

### 6.7 Node Embedding Index Mismatch
**Problem**: The model uses `torch.nn.Embedding(num_nodes, embed_dim)` where `node_id` must be a valid index into the embedding table. If `num_nodes` in the saved model doesn't match the current data, loading fails.
**Solution**: Save `num_nodes`, `feat_dim`, and `embed_dim` in the checkpoint. During loading, reconstruct the model with the exact same dimensions. The validation script explicitly checks this match.

### 6.8 Edge Index Format Errors
**Problem**: PyTorch Geometric expects edge indices as a `[2, num_edges]` LongTensor. Accidentally passing `[num_edges, 2]` or FloatTensor causes cryptic errors deep in the C++ backend.
**Solution**: Always use `.t().contiguous()` after creating edge tensors, and explicitly cast to `torch.long`.

### 6.9 Large Dataset Processing
**Problem**: TWOSIDES CSV is 4.3 GB, `faers_layer2_final.csv` is 606 MB. Loading and processing these files is slow and memory-intensive.
**Solution**: Use pandas with `dtype=str` to avoid type inference overhead. Process in chunks where possible. Use set-based lookups instead of DataFrame merges for speed.

### 6.10 Windows Console Encoding
**Problem**: The results generation script uses Unicode characters (arrows, checkmarks) that fail on Windows console's default encoding.
**Solution**: Wrap stdout/stderr with `io.TextIOWrapper(..., encoding='utf-8', errors='replace')` at script startup.

---

## 7. Key Concepts Explained

### 7.1 Graph Convolutional Networks (GCN)
GCN performs **message passing**: each node updates its representation by aggregating information from its neighbors. With edge weights, more strongly interacting neighbors contribute more. After 2 layers, each node's representation captures information from its 2-hop neighborhood.

### 7.2 Global Mean Pooling
After GCN layers, each drug has a 64-dim representation. To make a prediction about the *entire combination*, we need a single vector. Global mean pooling averages all drug representations into one 64-dim vector representing the combination as a whole.

### 7.3 Proportional Reporting Ratio (PRR)
PRR measures how much more frequently a side effect is reported for a drug pair compared to what's expected. `PRR > 1` means the side effect occurs more often than chance. Higher PRR = stronger evidence of a real interaction. We use `log1p(PRR)` as edge weights to compress the scale.

### 7.4 Youden's J Statistic
The optimal classification threshold is found by maximizing `J = TPR - FPR` across all possible thresholds on the ROC curve. This balances sensitivity and specificity, giving a threshold that's optimal regardless of class distribution.

### 7.5 BCEWithLogitsLoss
Combines sigmoid activation and binary cross-entropy loss in a single, numerically stable operation. Using raw logits instead of pre-sigmoid probabilities avoids floating-point precision issues near 0 and 1.

---

## 8. Complete File Reference

### Data Files

| File | Size | Description |
|------|------|-------------|
| `TWOSIDES.csv` | 4.3 GB | Raw drug-drug interaction database |
| `twosides_filtered.csv` | 1.2 GB | Filtered to FAERS drugs, PRR > 1 |
| `faers_layer2_final.csv` | 606 MB | Fully enriched FAERS dataset |
| `merged_faers.csv` | 46 MB | Combined quarterly FAERS data |
| `faers_with_mechanism.csv` | 43 MB | FAERS + ATC + targets + mechanisms |
| `faers_with_atc.csv` | 14 MB | FAERS + ATC classifications |
| `faers_rxnorm.csv` | 11 MB | FAERS mapped to RxCUI |
| `faers_drugcentral.csv` | 4.3 MB | FAERS mapped to structure_ids |
| `layer3_training_cases.csv` | 2.3 MB | 92,510 training cases |
| `layer3_edges_weighted.csv` | 888 KB | 32,582 drug-drug interaction edges |
| `layer3_nodes_enriched.csv` | 382 KB | 1,198 drug nodes with names |
| `layer3_nodes.csv` | 360 KB | 1,198 drug nodes (features only) |
| `weighted_gcn_model.pth` | 249 KB | Trained model checkpoint |
| `layer3_feature_meta.json` | 2.9 KB | Feature names and dimensions |
| `model_config.json` | 64 B | Model architecture config |

### Source Code Files

| Layer | File | Lines | Purpose |
|-------|------|-------|---------|
| 1-2 | `faers_filtering.py` | 44 | Filter FAERS quarterly data |
| 1-2 | `combine_faers.py` | 13 | Merge quarterly files |
| 1-2 | `faers_rxnorm_normalisation.py` | 79 | RxNorm drug name mapping |
| 1-2 | `faers_drugcentral_mapping.py` | 32 | DrugCentral structure_id mapping |
| 1-2 | `structure_atc_mapping.py` | 38 | ATC classification attachment |
| 1-2 | `Structure_id_mechanism_mapping.py` | 44 | Target gene/class attachment |
| 1-2 | `sider_processing.py` | 172 | SIDER side effect integration |
| 1-2 | `twosides_processing.py` | 48 | TWOSIDES filtering |
| 3 | `nodes_edges.py` | 277 | Graph construction & feature engineering |
| 4 | `train_gnn.py` | 336 | GNN model training |
| 4 | `class_check.py` | 14 | Class distribution & GPU check |
| 5 | `explain_gnn.py` | 345 | GNNExplainer + gradient attribution |
| 6 | `build_drug_lookup.py` | 60 | Drug name lookup table |
| 6 | `enrich_nodes.py` | 30 | Enrich nodes with names |
| 6 | `find_example_pairs.py` | 38 | Find polypharmacy cliques |
| — | `app.py` | 255 | Streamlit frontend UI |
| — | `model_utils.py` | 247 | Model utilities for frontend |
| — | `graph_builder.py` | 116 | Graph builder for frontend |
| — | `run_pipeline.py` | 92 | Pipeline orchestration |
| — | `validate_pipeline.py` | 351 | Model & data validation |
| — | `generate_results.py` | 587 | Results & visualization generation |

---

## 9. Results Summary

The trained WeightedGCN model achieves **AUC = 0.8159** on the held-out test set, demonstrating strong discriminative ability between high-risk and low-risk polypharmacy cases. The model's explainability outputs reveal clinically meaningful patterns — for example, in a high-risk case of rituximab + prednisolone + methylprednisolone, the model correctly identifies rituximab (an antineoplastic antibody) as the highest-risk contributor, driven by its drug class diversity and antibody binding mechanism.

---

*Document generated from complete source code analysis of the HEALTHCARE project, covering all 20+ source files, 15+ data files, and the full processing pipeline.*
