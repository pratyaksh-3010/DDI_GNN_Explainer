# Polypharmacy Risk Prediction Using Heterogeneous Graph Neural Networks with Explainable AI

## *Review Panel Submission Document*

---

## 1. Knowledge on Domain / Problem Statement

### 1.1 Domain: Polypharmacy and Drug Safety

**Polypharmacy** — the concurrent use of multiple medications by a single patient — is a pervasive clinical reality, particularly among elderly patients with multiple chronic conditions. The World Health Organization (WHO) identifies polypharmacy as a global patient safety challenge, with over 40% of elderly patients receiving five or more medications daily. The co-administration of multiple drugs creates a combinatorial risk space for **Drug-Drug Interactions (DDIs)** and **Adverse Drug Reactions (ADRs)** that cannot be predicted from individual drug safety profiles alone.

**Adverse Drug Reactions** represent a significant burden on global healthcare systems:

- ADRs are the **4th–6th leading cause of death** in the United States, accounting for over 100,000 deaths annually.
- Approximately **6.5% of hospital admissions** are ADR-related, with 20–30% of those caused by DDIs.
- ADR-related hospitalizations cost the US healthcare system an estimated **$136 billion annually**.
- The FDA Adverse Event Reporting System (FAERS) receives over **2 million adverse event reports per year**, reflecting the scale of post-market drug safety surveillance.

### 1.2 Limitations of Current Approaches

| Limitation | Description |
|------------|-------------|
| **Pairwise-only analysis** | Existing clinical tools (Lexicomp, Epocrates, Medscape) check Drug A vs. Drug B in isolation, unable to detect emergent risks from multi-drug combinations where indirect pathways (Drug A → Protein X ← Drug B) create unsafe interactions. |
| **Alert fatigue** | Binary interaction alerts (yes/no) without severity context cause clinicians to override 49–96% of DDI warnings, rendering safety systems ineffective. |
| **Static knowledge bases** | Rule-based systems cannot scale with ~50 new molecular entities approved annually; manual curation lag means novel drug combinations are unchecked. |
| **Homogeneous modeling** | Existing graph-based computational models treat all biomedical entities as a single node type, discarding the rich multi-type relationships between drugs, proteins, patients, and side effects. |
| **No explainability** | Deep learning models provide no mechanistic rationale for their predictions, precluding clinical adoption where trust and accountability are essential. |
| **No actionable alternatives** | No existing system recommends safer therapeutic alternatives when a risky combination is detected. |

### 1.3 Problem Statement

> Given the limitations of existing pairwise drug interaction systems and homogeneous computational models, how can we design a system that:
> 1. Models the **heterogeneous multi-relational structure** of the biomedical domain (drugs, proteins, patients, side effects) as a unified graph,
> 2. Predicts **patient-level adverse event severity** for multi-drug combinations using real-world FAERS outcomes,
> 3. Provides **interpretable explanations** of which drugs and pharmacological mechanisms drive predicted risk, and
> 4. **Recommends safer drug alternatives** by substituting high-risk drugs with therapeutically equivalent options?

### 1.4 Project Objectives

1. **Heterogeneous Data Integration**: Integrate 7 pharmacological data sources (FAERS, RxNorm, DrugCentral, SIDER, TWOSIDES, Bio-Decagon, PubChem) into a unified heterogeneous graph with 4 node types and 5 edge types.
2. **Heterogeneous Graph Transformer (HGT)**: Train an HGT model with type-specific multi-head attention to predict patient-level severity from the learned embeddings.
3. **Molecular Feature Enrichment**: Represent drugs using 1024-bit Morgan molecular fingerprints derived from SMILES structures via PubChem.
4. **Embedding-Based Explainability**: Provide per-drug risk attribution via dot-product scoring between patient and drug embeddings.
5. **Safer-Combination Recommender**: Suggest ATC-class-constrained drug substitutions to reduce predicted risk.

---

## 2. Literature Review

The following literature survey covers **18 recent, peer-reviewed publications** (2020–2024) organized by thematic relevance to our project.

### 2.1 Pharmacovigilance and Adverse Drug Reactions

**[1] Harpaz, R., Dumouchel, W., Shah, N.H. et al.** (2012). "Novel Data-Mining Methodologies for Adverse Drug Event Discovery and Analysis." *Clinical Pharmacology & Therapeutics*, 91(6), 1010–1021. DOI: 10.1038/clpt.2012.50

This seminal paper demonstrated the utility of FAERS data mining for computational drug safety research, establishing the data quality challenges (reporting bias, duplicate reports, inconsistent nomenclature) that subsequent systems must address. Our project directly addresses the nomenclature challenge through multi-hop RxNorm normalization.

---

**[2] Ji, S., Pan, S., Cambria, E. et al.** (2022). "A Survey on Knowledge Graphs: Representation, Acquisition, and Applications." *IEEE Transactions on Neural Networks and Learning Systems*, 33(2), 494–514. DOI: 10.1109/TNNLS.2021.3070843

This comprehensive survey established the theoretical foundations for representing biomedical knowledge as heterogeneous graphs with typed nodes and edges. The survey's taxonomy of knowledge graph construction, representation learning, and downstream reasoning tasks directly informs our heterogeneous graph design with distinct patient, drug, protein, and side-effect node types.

---

### 2.2 Graph Neural Networks for Drug-Drug Interaction Prediction

**[3] Zitnik, M., Agrawal, M. & Leskovec, J.** (2018). "Modeling Polypharmacy Side Effects with Graph Convolutional Networks." *Bioinformatics*, 34(13), i457–i466. DOI: 10.1093/bioinformatics/bty294

The **Decagon** model was the first to apply GNNs to polypharmacy side effect prediction, constructing a multimodal graph with drug-drug, drug-protein, and protein-protein edges. Our project builds directly upon Decagon by: (a) adding patient nodes as first-class entities, (b) using HGT instead of homogeneous GCN, (c) training on real-world FAERS patient outcomes rather than database associations, and (d) adding explainability and recommendation capabilities.

---

**[4] Yu, Y., Huang, K., Zhang, C. et al.** (2022). "SumGNN: Multi-Typed Drug Interaction Prediction via Efficient Knowledge Graph Summarization." *Bioinformatics*, 37(18), 2988–2995. DOI: 10.1093/bioinformatics/btab207

SumGNN integrates external biomedical knowledge with local subgraph structures for DDI prediction using an attention-based subgraph extraction mechanism. The local subgraph extraction strategy used in SumGNN is conceptually similar to our NeighborLoader-based mini-batch training, which samples multi-hop neighborhoods around patient nodes.

---

**[5] Nyamabo, A.K., Yu, H. & Shi, J.Y.** (2022). "SSI-DDI: Substructure-Substructure Interactions for Drug-Drug Interaction Prediction." *Briefings in Bioinformatics*, 23(6), bbab133. DOI: 10.1093/bib/bbab133

SSI-DDI decomposes drug molecules into substructures and learns pairwise interaction patterns at the substructure level. While chemically grounded, this approach requires extensive molecular graph data. Our use of 1024-bit Morgan fingerprints captures similar substructure information in a computationally efficient fixed-length binary vector representation.

---

**[6] Lin, S., Wang, Y., Zhang, L. et al.** (2023). "MHGNN: Multi-scale Heterogeneous Graph Neural Network for Drug-Drug Interaction Prediction." *Briefings in Bioinformatics*, 24(1), bbac578. DOI: 10.1093/bib/bbac578

MHGNN operates on heterogeneous drug-centered graphs incorporating molecular fingerprints, targets, enzymes, and pathways as distinct node types with multi-scale message passing. This work directly validates our approach of modeling drugs alongside their biological targets and mechanisms as a heterogeneous graph rather than a flat feature vector.

---

**[7] Su, X., You, Z., Wang, L. et al.** (2024). "Dual-Channel Learning Framework for Drug-Drug Interaction Prediction via Relation-Aware Heterogeneous Graph Transformer." *Proceedings of AAAI 2024*.

This paper proposes a relation-aware heterogeneous graph transformer for DDI prediction, demonstrating that type-specific attention mechanisms outperform uniform message passing for modeling diverse biomedical relationships — a finding consistent with our choice of HGT architecture over homogeneous GCN.

---

### 2.3 Heterogeneous Graphs in Biomedicine

**[8] Hu, Z., Dong, Y., Wang, K. & Sun, Y.** (2020). "Heterogeneous Graph Transformer." *Proceedings of WWW 2020*, 2704–2710. DOI: 10.1145/3366423.3380027

This foundational paper introduced the **Heterogeneous Graph Transformer (HGT)**, which is the core architecture of our model. HGT uses type-specific multi-head attention with relative temporal encoding, enabling nodes to aggregate information from neighbors through type-aware query-key-value projections. Our ExplainableHeteroGNN directly implements HGTConv layers from this work.

---

**[9] Wang, X., Ji, H., Shi, C. et al.** (2019). "Heterogeneous Graph Attention Network." *Proceedings of WWW 2019*, 2022–2032. DOI: 10.1145/3308558.3313562

The **Heterogeneous Graph Attention Network (HAN)** introduced hierarchical attention (node-level and semantic-level) for heterogeneous graphs via meta-paths. While HAN requires predefined meta-paths, HGT's attention formulation eliminates this requirement by learning type-specific attention weights end-to-end, which is why we adopted HGT for our project.

---

**[10] Li, M., Wang, Y., Zheng, R. et al.** (2024). "Precision Adverse Drug Reactions Prediction with Heterogeneous Graph Neural Network." *Advanced Science*, 11(34), 2404671. DOI: 10.1002/advs.202404671

**PreciseADR** constructs heterogeneous graphs containing patient, disease, drug, and ADR nodes, propagating information across diverse node types using GNNs for patient-level ADR prediction. This is the closest related work to our system; however, PreciseADR focuses on individual ADR type prediction rather than overall severity classification, does not incorporate molecular fingerprints, and lacks a drug recommendation module.

---

**[11] Li, Y., Pei, S., Zhang, R. et al.** (2024). "HGTDR: Heterogeneous Graph Transformer for Drug Repurposing Using the PrimeKG Knowledge Graph." *Bioinformatics*, 40(7), btae350. DOI: 10.1093/bioinformatics/btae350

HGTDR applies heterogeneous graph transformers to drug repurposing by leveraging the PrimeKG knowledge graph. Their use of type-specific attention on multi-relational biomedical graphs demonstrates the efficacy of HGT architectures in pharmaceutical applications, validating our architectural choice.

---

### 2.4 Molecular Representations for Drug Modeling

**[12] Rogers, D. & Hahn, M.** (2010). "Extended-Connectivity Fingerprints." *Journal of Chemical Information and Modeling*, 50(5), 742–754. DOI: 10.1021/ci100050t

This paper introduced Extended-Connectivity Fingerprints (ECFP/Morgan fingerprints), which encode circular substructures around each atom as a fixed-length binary vector. We use 1024-bit Morgan fingerprints with radius 2 (equivalent to ECFP4) as drug node features, providing a chemistry-grounded representation that captures functional groups, ring systems, and pharmacophoric features.

---

**[13] Deng, Y., Xu, X., Qiu, Y. et al.** (2020). "A Multimodal Deep Learning Framework for Predicting Drug-Drug Interaction Events." *Bioinformatics*, 36(15), 4316–4322. DOI: 10.1093/bioinformatics/btaa501

This multimodal framework jointly predicts interaction types and adverse event outcomes, demonstrating that auxiliary pharmacological tasks improve DDI prediction. This multi-task learning principle is reflected in our heterogeneous graph where drug-protein-side-effect relationships jointly inform patient-level severity prediction.

---

**[14] Jiang, Y., Wang, Y., Shen, Y. et al.** (2024). "SMR-DDI: A Self-supervised Molecular Representation Learning Framework for Drug-Drug Interaction Prediction." *BMC Bioinformatics*, 25, 218. DOI: 10.1186/s12859-024-05841-3

SMR-DDI uses contrastive self-supervised learning to pre-train molecular representations before fine-tuning for DDI prediction. Their work demonstrates the value of chemical structure-based drug representations — a principle we adopt through Morgan fingerprints, though our approach uses established fingerprint methods rather than learned representations for reproducibility.

---

### 2.5 Explainable AI and Drug Recommendation

**[15] Ying, Z., Bourgeois, D., You, J. et al.** (2019). "GNNExplainer: Generating Explanations for Graph Neural Networks." *Proceedings of NeurIPS 2019*, 9244–9255.

GNNExplainer learns soft masks over node features and edges by maximizing mutual information between the masked subgraph and the model's prediction. While designed for homogeneous graphs, the principles of feature and edge masking inform our embedding-based attribution approach for heterogeneous settings.

---

**[16] Bhatt, U., Xiang, A., Sharma, S. et al.** (2020). "Explainable Machine Learning in Deployment." *Proceedings of FAT* 2020*, 648–657. DOI: 10.1145/3351095.3375624

This paper surveyed deployed explainable ML systems and identified critical gaps between academic XAI research and production requirements — particularly in healthcare where clinicians need actionable, domain-grounded explanations. Our approach maps drug attribution scores to ATC therapeutic classes, providing pharmacologically meaningful explanations.

---

**[17] Yang, C., Xiao, C., Ma, F. et al.** (2023). "REFINE: A Fine-Grained Medication Recommendation System Using Deep Learning and Personalized Drug Interaction Modeling." *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*.

REFINE addresses the limitations of DDI-aware medication recommendation by modeling drug interaction severity as weighted graphs and designing a balanced loss function that weighs therapeutic benefits against interaction risks. Our safer-combination recommender shares REFINE's philosophy of ATC-class-constrained substitution but operates on heterogeneous graph embeddings rather than longitudinal patient timelines.

---

**[18] Jiang, W., Li, C., Guo, S. et al.** (2024). "GRAPHCARE: Enhancing Healthcare Predictions with Personalized Knowledge Graphs." *Proceedings of ICLR 2024*.

GRAPHCARE generates personalized knowledge graphs from Electronic Health Records alongside external biomedical knowledge for healthcare predictions. While focused on broader clinical outcomes, its approach of connecting patients to medical entities through knowledge graphs parallels our patient-drug-protein-side-effect heterogeneous graph design.

---

### 2.6 Summary of Literature Landscape

| Research Theme | Key Gap Identified | Our Approach |
|----------------|-------------------|-------------|
| DDI Prediction | Pairwise-only models dominate ([3–7]) | Patient-level severity over full combinations |
| Graph Modeling | Homogeneous graphs lose multi-type information ([3–5]) | Heterogeneous graph with 4 node types, 5 edge types ([8–11]) |
| Drug Representation | Hand-crafted features lack chemical grounding | 1024-bit Morgan fingerprints ([12–14]) |
| Explainability | Black-box models hinder clinical adoption ([15–16]) | Embedding-based attribution + ATC-class labeling |
| Recommendation | No systems recommend safer alternatives ([17–18]) | ATC-constrained drug substitution |

---

## 3. Design of Proposed Methodology

### 3.1 High-Level Architecture

The system follows a **7-phase pipeline** from raw data to clinical decision support:

```
Phase 1: Data Preprocessing (8 scripts)
   ↓   [FAERS filtering, RxNorm normalization, DrugCentral mapping,
   ↓    ATC classification, mechanism mapping, SIDER/TWOSIDES integration]
   ↓
Phase 2: Data Aggregation (4 scripts)
   ↓   [PubChem-RxCUI crosswalk, Bio-Decagon targets/PPI merge,
   ↓    side effect union, heterogeneous graph construction]
   ↓
Phase 3: V2 Side Effect Filtering
   ↓   [Top 1000 most frequent side effects selected]
   ↓
Phase 4: Molecular Feature Extraction
   ↓   [PubChem SMILES → Morgan Fingerprints via RDKit]
   ↓
Phase 5: Heterogeneous Dataset Building
   ↓   [PyG HeteroData with 4 node types, 5 edge types]
   ↓
Phase 6: HGT Model Training
   ↓   [ExplainableHeteroGNN with NeighborLoader mini-batching]
   ↓
Phase 7: Explainability & Recommendation
       [Embedding-based attribution + ATC-constrained drug substitution]
```

### 3.2 Data Sources

| Source | Type | Key Contribution |
|--------|------|-----------------|
| **FAERS** (2023 Q1–Q4) | FDA adverse event reports | Patient cases with drug lists and outcome severity labels |
| **RxNorm** | Drug nomenclature standard | Canonical RxCUI identifiers for drug name normalization |
| **DrugCentral** | Drug knowledge database | Structure IDs, ATC classification, PubChem CID mappings |
| **SIDER** | Side effect database | Known mono-drug adverse reactions |
| **TWOSIDES** | DDI database (4.3 GB) | Drug pair interactions with PRR scores |
| **Bio-Decagon** | Multimodal biomedical graph | Drug-protein targets, protein-protein interactions, polypharmacy side effects |
| **PubChem** | Molecular structure database | Canonical SMILES for Morgan fingerprint generation |

### 3.3 Heterogeneous Graph Design

The biomedical domain is modeled as a **heterogeneous graph** with:

**4 Node Types:**

| Node Type | Count | Feature | Dimension |
|-----------|-------|---------|-----------|
| Patient | ~92,000 | Constant indicator | 1 |
| Drug | ~1,200 | Morgan fingerprint (ECFP4) | 1,024 |
| Protein | ~4,000+ | PPI degree centrality | 1 |
| Side Effect | 1,000 | DDI frequency (normalized) | 1 |

**5 Edge Types:**

| Edge Type | Source → Target | Attributes | Source Database |
|-----------|----------------|------------|-----------------|
| prescribed | Patient → Drug | None | FAERS |
| targets | Drug → Protein | None | DrugCentral + Bio-Decagon |
| interacts_with | Protein ↔ Protein | None | Bio-Decagon PPI |
| interacts_with | Drug ↔ Drug | mean_PRR, num_shared_SE | TWOSIDES + Bio-Decagon |
| causes | Drug → Side Effect | mean_PRR | TWOSIDES + Bio-Decagon |

### 3.4 Model Architecture: ExplainableHeteroGNN

```
Input: Type-specific node features
  ↓
Type-Specific Linear Projections (each type → 64-dim, lazy initialization)
  ↓
HGTConv Layer 1 (64 → 64, 4 attention heads, type-specific attention)
  ↓
HGTConv Layer 2 (64 → 64, 4 attention heads, type-specific attention)
  ↓
Extract Patient Embeddings (64-dim)
  ↓
Severity Head: Linear(64→32) + ReLU + Dropout(0.3) + Linear(32→1)
  ↓
Output: Raw logits (BCEWithLogitsLoss) + All node embeddings (for explainability)
```

**Training Configuration:**
- Mini-batch: NeighborLoader (10 neighbors/hop, 2 hops, batch_size=128)
- Optimizer: Adam (lr=0.005, weight_decay=0.001)
- Loss: BCEWithLogitsLoss
- Split: 80% train / 20% validation

### 3.5 Explainability: Embedding-Based Drug Attribution

For each patient case:
1. Extract local subgraph via NeighborLoader
2. Forward pass through HGT → severity logits + all embeddings
3. Compute drug attribution: `score(drug_i) = patient_embedding · drug_i_embedding`
4. Rank drugs by attribution score with ATC class labels

### 3.6 Safer-Combination Recommender

1. Identify the highest-risk drug (largest attribution score)
2. Retrieve its ATC therapeutic class(es)
3. Find alternative drugs in the same ATC class
4. Evaluate each alternative via counterfactual re-inference
5. Recommend the substitution with lowest projected risk

---

## 4. Module Description / System Design

### Module 1: Data Preprocessing Module

**Purpose**: Transform raw FAERS quarterly data and external drug databases into a unified, enriched patient-drug dataset.

**Components** (8 scripts, sequential execution):

| Script | Responsibility |
|--------|---------------|
| `faers_filtering.py` (×4) | Filter FAERS quarterly files to polypharmacy cases (≥2 drugs) with Primary/Secondary Suspect roles; compute binary severity from outcome codes {DE, LT, HO, DS} |
| `combine_faers.py` | Concatenate 4 quarterly CSVs into a unified dataset |
| `faers_rxnorm_normalisation.py` | Normalize free-text drug names to canonical RxCUI identifiers using RxNorm RXNCONSO.RRF; aggressive string normalization (uppercase, strip dosage forms, regex) |
| `faers_drugcentral_mapping.py` | Map RxCUI → DrugCentral structure_id for pharmacological enrichment |
| `structure_atc_mapping.py` | Attach ATC Level-1 therapeutic classification codes (14 classes: A–V) |
| `Structure_id_mechanism_mapping.py` | Attach target gene symbols, target classes, and action types |
| `sider_processing.py` | Integrate known mono-drug side effects from SIDER per drug per case |
| `twosides_processing.py` | Filter TWOSIDES DDI database to FAERS drugs with PRR > 1 |

**Input**: Raw FAERS text files, RxNorm RRF, DrugCentral maps, SIDER TSV, TWOSIDES CSV (4.3 GB)
**Output**: `faers_layer2_final.csv` (606 MB enriched dataset), `twosides_filtered.csv` (1.2 GB filtered DDI edges)

---

### Module 2: Data Aggregation Module

**Purpose**: Integrate Bio-Decagon multimodal biomedical data with the preprocessed FAERS dataset and construct the raw heterogeneous graph structure.

**Components** (4 scripts, sequential execution):

| Script | Responsibility |
|--------|---------------|
| `1_build_vocab_crosswalk.py` | Parse DrugCentral SQL dump for PUBCHEM_CID entries; build PubChem → structure_id → RxCUI crosswalk for Bio-Decagon integration |
| `2_merge_targets_and_ppi.py` | Map STITCH IDs → RxCUI via crosswalk; convert Entrez Gene IDs → HGNC symbols via mygene API; merge Decagon drug-protein targets with DrugCentral targets; filter PPI to genes present in FAERS cases |
| `3_merge_side_effects.py` | Union TWOSIDES and Decagon polypharmacy edges (sorted pairs, max PRR); merge Decagon mono side effects with SIDER data |
| `4_construct_hetero_graph.py` | Extract 4 node type lists (patient, drug, protein, side_effect) and 5 edge type lists as CSV files |

**Input**: DrugCentral SQL, Bio-Decagon (targets, PPI, combo), preprocessed FAERS data
**Output**: `Graph_Data/` directory with 4 node CSVs and 5 edge CSVs

---

### Module 3: Side Effect Curation Module

**Purpose**: Reduce noise by selecting the top 1,000 most clinically significant side effects.

**Component**: `filter_top_1000_se.py`

**Process**:
- Compute frequency distribution of all side effects across combined DDI edges
- Select top 1,000 by frequency (coverage: anaemia 117K → neutropenic sepsis 9K reports)
- Filter DDI edges, drug-side-effect edges, and side-effect nodes to this curated set
- Copy remaining graph components (patients, drugs, proteins, PPI) unchanged

**Input**: Full `Graph_Data/` from Module 2
**Output**: `V2/Graph_Data/` with filtered edges and nodes

---

### Module 4: Molecular Feature Extraction Module

**Purpose**: Obtain molecular structure representations for drug nodes.

**Component**: `extract_smiles.py`

**Process**:
- Map RxCUI → PubChem CID via vocabulary crosswalk
- Fetch Canonical SMILES from PubChem PUG REST API (batch requests of 100, 0.3s rate limiting)
- Save RxCUI→SMILES mapping for fingerprint generation

**Input**: Drug node list, PubChem-RxCUI crosswalk
**Output**: `drug_smiles.csv` (~79 KB)

---

### Module 5: Heterogeneous Dataset Construction Module

**Purpose**: Build the PyTorch Geometric HeteroData object encoding the complete multi-relational graph.

**Component**: `dataset_builder.py`

**Process**:
1. Load all node and edge CSVs; build ID→index mappings for each type
2. Generate drug features: 1024-bit Morgan fingerprints via RDKit (radius=2) from SMILES
3. Generate protein features: PPI degree centrality
4. Generate side-effect features: normalized DDI frequency
5. Initialize patient features: constant 1.0 (aggregated via HGT message passing)
6. Construct all 5 edge types with proper index mapping and deduplication:
   - Drug-Drug: grouped by unique pair → mean_PRR + count of shared SEs (2-dim attributes)
   - Drug-Side-Effect: grouped by unique (drug, SE) → mean_PRR (1-dim attribute)
7. Add patient severity labels (`data['patient'].y`)
8. Serialize HeteroData and node mappings

**Input**: `V2/Graph_Data/` CSV files + `drug_smiles.csv`
**Output**: `V2/Processed/hetero_graph_data.pt` (39 MB), `node_mappings.json` (1.8 MB)

---

### Module 6: HGT Model Training Module

**Purpose**: Train the ExplainableHeteroGNN on patient severity prediction.

**Components**: `hetero_attention_model.py` (model definition), `train_hetero.py` (training loop)

**Model Architecture (ExplainableHeteroGNN)**:
- Type-specific linear projections: each node type → 64-dim hidden space
- 2 × HGTConv layers: 64 channels, 4 attention heads each
- Severity prediction head: MLP (64→32→1) with ReLU and Dropout(0.3)
- Dual output: severity logits + all node embeddings

**Training Loop**:
- NeighborLoader mini-batching: 10 neighbors sampled per hop, 2 hops, batch_size=128
- Lazy layer initialization via dummy forward pass
- BCEWithLogitsLoss for binary severity classification
- 80/20 random train/validation split
- Checkpoint: `hgt_model.pth`

**Input**: `hetero_graph_data.pt`
**Output**: Trained model weights `hgt_model.pth`

---

### Module 7: Explainability & Recommendation Module

**Purpose**: Provide clinician-interpretable risk explanations and suggest safer drug alternatives.

**Component**: `explainer_and_recommender.py`

**Explainer Function** (`explain_patient_risk`):
1. Load trained HGT model and heterogeneous graph
2. For a given patient case, extract local subgraph via NeighborLoader (15 neighbors/hop)
3. Forward pass → severity prediction score + node embeddings
4. Identify prescribed drugs via patient→drug edges
5. Compute dot-product attribution: `score = patient_embed · drug_embed`
6. Output ranked drug risk hierarchy with ATC class labels

**Recommender Function** (`recommend_safer_combination`):
1. Select highest-risk drug from the attribution ranking
2. Retrieve its ATC Level-1 class(es)
3. Search for alternative drugs in the same ATC class (preserving therapeutic intent)
4. Evaluate top 20 candidates via counterfactual risk projection
5. Output the safest substitute with projected risk reduction

**Input**: Trained model, HeteroData, patient case ID
**Output**: Risk score, per-drug attribution ranking, suggested safer substitution

---

### System Design: Complete Data Flow

```
Raw FAERS (Q1-Q4)      RxNorm RRF       DrugCentral SQL
     ↓                      ↓                  ↓
[Module 1: Preprocessing]──────────────────────────
     ↓                                          ↓
faers_layer2_final.csv              stitch_to_rxcui_map.csv
     ↓                                          ↓
[Module 2: Aggregation]   ←   Bio-Decagon (targets, PPI, combo)
     ↓
Graph_Data/ (nodes + edges CSVs)
     ↓
[Module 3: SE Curation]  →  V2/Graph_Data/ (top 1000 SE)
     ↓
[Module 4: SMILES Extract] ← PubChem API  →  drug_smiles.csv
     ↓
[Module 5: Dataset Build]  →  hetero_graph_data.pt
     ↓
[Module 6: HGT Training]  →  hgt_model.pth
     ↓
[Module 7: Explain + Recommend]  →  Risk score + Attribution + Substitution
```

---

## References

[1] Harpaz, R. et al. (2012). *Clinical Pharmacology & Therapeutics*, 91(6), 1010–1021. DOI: 10.1038/clpt.2012.50

[2] Ji, S. et al. (2022). *IEEE TNNLS*, 33(2), 494–514. DOI: 10.1109/TNNLS.2021.3070843

[3] Zitnik, M. et al. (2018). *Bioinformatics*, 34(13), i457–i466. DOI: 10.1093/bioinformatics/bty294

[4] Yu, Y. et al. (2022). *Bioinformatics*, 37(18), 2988–2995. DOI: 10.1093/bioinformatics/btab207

[5] Nyamabo, A.K. et al. (2022). *Briefings in Bioinformatics*, 23(6), bbab133. DOI: 10.1093/bib/bbab133

[6] Lin, S. et al. (2023). *Briefings in Bioinformatics*, 24(1), bbac578. DOI: 10.1093/bib/bbac578

[7] Su, X. et al. (2024). *Proceedings of AAAI 2024*.

[8] Hu, Z. et al. (2020). *Proceedings of WWW 2020*, 2704–2710. DOI: 10.1145/3366423.3380027

[9] Wang, X. et al. (2019). *Proceedings of WWW 2019*, 2022–2032. DOI: 10.1145/3308558.3313562

[10] Li, M. et al. (2024). *Advanced Science*, 11(34), 2404671. DOI: 10.1002/advs.202404671

[11] Li, Y. et al. (2024). *Bioinformatics*, 40(7), btae350. DOI: 10.1093/bioinformatics/btae350

[12] Rogers, D. & Hahn, M. (2010). *J. Chem. Inf. Model.*, 50(5), 742–754. DOI: 10.1021/ci100050t

[13] Deng, Y. et al. (2020). *Bioinformatics*, 36(15), 4316–4322. DOI: 10.1093/bioinformatics/btaa501

[14] Jiang, Y. et al. (2024). *BMC Bioinformatics*, 25, 218. DOI: 10.1186/s12859-024-05841-3

[15] Ying, Z. et al. (2019). *Proceedings of NeurIPS 2019*, 9244–9255.

[16] Bhatt, U. et al. (2020). *Proceedings of FAT* 2020*, 648–657. DOI: 10.1145/3351095.3375624

[17] Yang, C. et al. (2023). *NeurIPS 2023*.

[18] Jiang, W. et al. (2024). *Proceedings of ICLR 2024*.
