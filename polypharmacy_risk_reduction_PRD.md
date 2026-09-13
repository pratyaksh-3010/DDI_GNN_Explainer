# Product Requirements Document (PRD)
## Explainable Heterogeneous GNNs for Counterfactual Polypharmacy Risk Reduction

**Document Version:** 1.0  
**Status:** Proposed / Implementation-Ready PRD  
**Project Type:** Healthcare AI / Clinical Decision Support Research Prototype  
**Primary ML Architecture:** Heterogeneous Graph Transformer (HGT)  
**Primary Framework:** PyTorch Geometric  
**Proposed UI:** Streamlit  
**Data Sources:** FAERS, TWOSIDES, RxNorm, ATC, Drug–Target, PPI  

> **Source basis:** This PRD is based primarily on the updated project workflow supplied for this project. It preserves the workflow's stated architecture and methodology. Where the workflow does not specify an implementation detail, the item is marked as proposed or TBD rather than treated as finalized.

---

# 1. Executive Summary

## 1.1 Product Vision

Develop an **explainable AI system for polypharmacy risk assessment and counterfactual risk reduction**.

Given a patient's medication combination, the system should:

1. Construct the patient's heterogeneous clinical/biological subgraph.
2. Predict the overall severity of the medication regimen.
3. Predict the most likely adverse side effects.
4. Extract HGT attention information.
5. Rank medications according to their contribution to predicted risk.
6. Identify the highest-risk medication.
7. Find therapeutically appropriate alternatives using ATC classification.
8. Replace the risky medication with candidate alternatives.
9. Re-run the HGT model for each counterfactual regimen.
10. Recommend the alternative regimen with the lowest predicted severity.
11. Explain why the original regimen was considered risky and how the recommendation changes the prediction.

The project workflow is:

```text
Input Patient Drug Combination
        ↓
Extract Patient Subgraph
        ↓
Heterogeneous Graph Transformer
        ↓
Predict Severity + Side Effects
        ↓
Extract Attention Weights
        ↓
Identify Highest-Risk Drug
        ↓
Counterfactual Swapping with ATC Alternatives
        ↓
Simulate New Combinations
        ↓
Safest Recommended Regimen
```

---

# 2. Problem Statement

Polypharmacy creates a complex interaction environment in which a patient's risk may depend on:

- multiple simultaneous medications,
- drug-drug relationships,
- drug-target relationships,
- protein-protein relationships,
- adverse effects,
- patient/case characteristics,
- and the overall medication context.

Traditional pairwise drug-interaction models are insufficient for representing the complete structure.

The project's problem is:

> **How can we model a patient's medication regimen as a heterogeneous clinical and biological graph, predict its adverse-event severity, explain which drug contributes most to the predicted risk, and evaluate therapeutically valid alternative regimens using counterfactual reasoning?**

---

# 3. Product Goal

The final prototype must answer four questions.

## Q1. Is this regimen risky?

```text
Patient + Drugs
      ↓
HGT
      ↓
Severity prediction
```

## Q2. What adverse effects are likely?

```text
Patient + Drugs
      ↓
HGT
      ↓
Top-K side effects
```

## Q3. Which drug is contributing most to the predicted risk?

```text
HGT attention
      ↓
Aggregation
      ↓
Drug Risk Hierarchy
```

## Q4. Can we reduce the predicted risk?

```text
Highest-risk drug
       ↓
ATC alternatives
       ↓
Counterfactual regimens
       ↓
HGT evaluation
       ↓
Lowest predicted severity
```

---

# 4. Product Scope

## 4.1 In Scope

- FAERS preprocessing
- TWOSIDES integration
- RxNorm drug normalization
- ATC classification
- Drug-target integration
- PPI integration
- Heterogeneous graph construction
- PyTorch Geometric `HeteroData`
- HGT implementation
- Severity prediction
- Multi-label side-effect prediction
- HGT attention extraction
- Drug risk ranking
- Counterfactual drug substitution
- ATC-constrained alternatives
- Counterfactual HGT inference
- Evaluation on severe FAERS cases
- Streamlit interface
- Risk hierarchy visualization
- Counterfactual recommendation visualization

---

## 4.2 Out of Scope

The first version must not:

- prescribe medication to real patients,
- autonomously make clinical decisions,
- claim biological causality from attention weights,
- replace a physician or pharmacist,
- automatically change prescriptions,
- guarantee that a recommended drug is clinically appropriate for a particular individual,
- claim FDA or clinical approval.

The system is a **research/decision-support prototype**, not an autonomous prescribing system.

---

# 5. Target Users

## 5.1 Primary User — Researcher / Clinical AI Researcher

Needs to:

- input medication combinations,
- assess predicted risk,
- understand model reasoning,
- identify risky drugs,
- evaluate alternatives,
- inspect model performance.

## 5.2 Secondary User — Clinician/Pharmacist for Prototype Evaluation

Potential use cases:

- review predicted regimen severity,
- inspect predicted adverse effects,
- examine model-ranked risky medications,
- inspect candidate alternatives.

The output must be clearly identified as model-generated research/decision-support information.

---

# 6. High-Level System Architecture

```text
                         ┌─────────────┐
                         │    FAERS    │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  TWOSIDES   │
                         └──────┬──────┘
                                │
                    ┌───────────▼───────────┐
                    │ Data Filtering        │
                    │ & Polypharmacy        │
                    │ Selection             │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ RxNorm + ATC Mapping  │
                    └───────────┬───────────┘
                                │
                ┌───────────────┼────────────────┐
                │               │                │
         Drug Targets          PPI         Side Effects
                │               │                │
                └───────────────┼────────────────┘
                                ↓
                    ┌────────────────────┐
                    │ Graph Structure    │
                    │ Builder             │
                    └─────────┬──────────┘
                              ↓
                    ┌────────────────────┐
                    │ PyG HeteroData     │
                    └─────────┬──────────┘
                              ↓
                    ┌────────────────────┐
                    │ Patient Subgraph   │
                    └─────────┬──────────┘
                              ↓
                    ┌────────────────────┐
                    │ HGT Layer 1        │
                    └─────────┬──────────┘
                              ↓
                    ┌────────────────────┐
                    │ HGT Layer 2        │
                    └─────────┬──────────┘
                              ↓
                     Node Representations
                         /            \
                        /              \
                       ↓                ↓
               Severity Head      Side Effect Head
                       │                │
                       ↓                ↓
                 Severity         Top-K Effects
                       │
                       ↓
                Attention Extraction
                       ↓
                Drug Risk Hierarchy
                       ↓
                Highest-Risk Drug
                       ↓
                  ATC Alternatives
                       ↓
              Counterfactual Graphs
                       ↓
                  HGT Inference
                       ↓
                Risk Comparison
                       ↓
              Recommended Regimen
```

---

# 7. Data Requirements

## 7.1 FAERS

### Purpose

FAERS provides the clinical/adverse-event foundation.

Use FAERS to:

- identify medication cases,
- identify polypharmacy cases,
- identify reported adverse events,
- categorize severe clinical outcomes,
- generate training/evaluation examples.

---

## 7.2 TWOSIDES

### Purpose

TWOSIDES contributes drug-drug/adverse-event association information.

Conceptually:

```text
Drug A + Drug B
       ↓
Drug interaction
       ↓
Potential adverse event
```

This information is integrated into the heterogeneous graph.

---

## 7.3 RxNorm

### Purpose

Normalize drug names into standard identifiers.

```text
Raw Drug Name
      ↓
Normalization
      ↓
RxNorm Mapping
      ↓
Standard Drug Identifier
```

---

## 7.4 ATC

### Purpose

ATC provides therapeutic classification.

Primary use:

```text
Risky Drug
    ↓
ATC Class
    ↓
Candidate Alternatives
```

Counterfactual candidates should be constrained to the same therapeutic class according to the project's methodology.

---

## 7.5 Drug-Target Dataset

### Purpose

Connect drugs to proteins.

```text
Drug
  │
  └── targets → Protein
```

---

## 7.6 PPI Dataset

### Purpose

Connect proteins to one another.

```text
Protein P1 ── interacts ── Protein P2
```

This adds biological context to the clinical graph.

---

# 8. Data Preprocessing Requirements

## 8.1 Raw Data Cleaning

Each source should undergo:

```text
Raw Data
   ↓
Schema validation
   ↓
Missing-value handling
   ↓
Duplicate handling
   ↓
Entity normalization
   ↓
Identifier mapping
   ↓
Quality checks
```

---

## 8.2 Drug Normalization

```text
Raw Drug Name
      ↓
Normalize text
      ↓
RxNorm mapping
      ↓
Standard drug ID
      ↓
ATC mapping
```

Unmapped drugs must be explicitly tracked rather than silently discarded.

---

## 8.3 Polypharmacy Filtering

The project focuses on polypharmacy cases.

The implementation must explicitly define the minimum number of medications required for a case to qualify.

```text
number of medications >= N
```

**Status:** `N` is TBD and must be finalized based on the research definition and dataset distribution.

---

## 8.4 Severity Label Creation

Each clinical case requires a binary target:

```text
severity ∈ {0, 1}

0 = non-severe
1 = severe
```

The exact FAERS outcome-to-label mapping must be formally documented.

---

## 8.5 Side-Effect Labels

Each case can contain multiple side effects.

Example:

```text
Patient P1:

[SideEffect_A,
 SideEffect_B,
 SideEffect_C]
```

Represent as a multi-hot target:

```text
[1, 0, 1, 0, 1, ...]
```

---

# 9. Canonical Graph Schema

## 9.1 Node Types

The initial graph contains:

```text
patient
drug
protein
side_effect
```

---

## 9.2 Patient Node

Conceptually:

```text
patient_id
features
```

### Identity requirement

FAERS does not necessarily provide a conventional persistent longitudinal patient identity. Therefore, the implementation must explicitly define what a `patient` node represents.

A safer technical representation may be:

```text
patient_case
```

where the node corresponds to a clinical report/case.

---

## 9.3 Drug Node

```text
drug_id
rxnorm_id
atc_code
features
```

Potential features:

- drug embedding,
- molecular characteristics,
- interaction-derived features,
- target-derived features.

**Exact feature set:** TBD.

---

## 9.4 Protein Node

```text
protein_id
features
```

Potential features:

- protein identity embedding,
- biological/network features.

**Exact feature set:** TBD.

---

## 9.5 Side Effect Node

```text
side_effect_id
features
```

The exact feature representation is TBD.

---

# 10. Canonical Edge Schema

The graph should distinguish semantic relations.

## Core relationships

```text
(patient, takes, drug)

(drug, targets, protein)

(protein, interacts, protein)

(patient, experiences, side_effect)

(drug, interacts, drug)
```

The exact relation names for side-effect and drug-interaction relationships should be standardized during implementation.

---

# 11. Graph Construction

The graph builder converts processed datasets into a unified heterogeneous graph.

## Step 1 — Create Entity Dictionaries

Example:

```text
Drug:

Aspirin → 0
Drug B  → 1
Drug C  → 2
```

```text
Protein:

P100 → 0
P101 → 1
P102 → 2
```

```text
Patient:

Case_001 → 0
Case_002 → 1
```

---

## Step 2 — Create Node Feature Matrices

Create:

```text
X_patient
X_drug
X_protein
X_side_effect
```

Each row corresponds to one node of that type.

---

## Step 3 — Create Edge Indices

For:

```text
patient → takes → drug
```

create:

```text
edge_index_takes
```

For:

```text
drug → targets → protein
```

create:

```text
edge_index_targets
```

For:

```text
protein → interacts → protein
```

create:

```text
edge_index_ppi
```

and equivalent structures for the other relations.

---

## Step 4 — Add Edge Attributes

Where the source provides meaningful numerical relationship information, retain it as edge attributes.

Example:

```text
Drug A ── interaction ── Drug B
             │
             └── interaction strength
```

The exact edge attributes are TBD and depend on the selected source fields.

---

## Step 5 — Build PyG HeteroData

Conceptually:

```python
data = HeteroData()
```

Node features:

```python
data["patient"].x
data["drug"].x
data["protein"].x
data["side_effect"].x
```

Edges:

```python
data["patient", "takes", "drug"].edge_index

data["drug", "targets", "protein"].edge_index

data["protein", "interacts", "protein"].edge_index

data["patient", "experiences", "side_effect"].edge_index

data["drug", "interacts", "drug"].edge_index
```

---

# 12. Graph Validation

Before model training, automated checks must verify:

## Node Checks

- No invalid IDs
- No unexpected node types
- No NaN feature values
- Correct feature dimensions

## Edge Checks

- Every source ID exists
- Every destination ID exists
- No invalid edge indices
- Expected relation types exist

## Data Checks

- No label leakage
- Correct case construction
- Correct severity labels
- Correct side-effect labels

---

# 13. Patient Subgraph Extraction

The complete heterogeneous graph may be very large.

For inference, extract a patient-specific subgraph.

Starting point:

```text
Patient P1
   ↓
Drugs taken by P1
```

Expand through relevant relationships:

```text
Patient
 ↓
Drugs
 ↓
Targets
 ↓
Proteins
 ↓
PPI
```

and:

```text
Drugs
 ↓
Side Effects
```

and relevant drug-drug interactions.

**Hop count:** TBD.

---

# 14. Example Patient Subgraph

```text
                         Protein P1
                              ↑
                           targets
                              │
Patient P1 ──takes──> Drug A ────── Drug B
    │                   │             │
    │                   │             │
    │                targets       targets
    │                   │             │
    │                   ↓             ↓
    │                Protein P2    Protein P3
    │
    ├──takes──> Drug B
    │
    └──takes──> Drug C
          │
          └──── associated side effect
```

This subgraph is the input to HGT.

---

# 15. HGT Model Requirements

## 15.1 Architecture

The initial architecture follows:

```text
Heterogeneous Subgraph
          ↓
      HGT Layer 1
          ↓
      HGT Layer 2
          ↓
     Node Embeddings
        /       \
       /         \
Severity Head   Side Effect Head
```

---

## 15.2 What HGT Does

At each layer, information is propagated between connected nodes while considering:

- source node type,
- target node type,
- edge/relation type,
- attention.

For example:

```text
Patient → Drug
```

is semantically different from:

```text
Drug → Protein
```

and:

```text
Protein → Protein
```

HGT is designed to model these heterogeneous relationships.

---

# 16. HGT Message Passing — Conceptual Flow

```text
Input node representations
          ↓
Identify neighboring nodes
          ↓
Identify node types
          ↓
Identify edge/relation types
          ↓
Create relation-aware representations
          ↓
Calculate attention
          ↓
Aggregate messages
          ↓
Update node representation
```

Repeated over two HGT layers, this produces contextualized node embeddings.

---

# 17. Multi-Head Attention

HGT uses multiple attention heads.

Conceptually:

```text
             HGT
              │
       ┌──────┼──────┐
       ↓      ↓      ↓
     Head 1 Head 2 Head 3
       │      │      │
       └──────┼──────┘
              ↓
       Aggregated representation
```

The project uses these learned attention values as an explainability signal.

Do not automatically interpret individual heads as biologically specialized unless experiments demonstrate such specialization.

---

# 18. Severity Prediction Head

The first head performs binary classification.

```text
Patient representation
       ↓
Linear layer
       ↓
Severity logit/probability
```

Output:

```text
P(severe)
```

The final classification threshold is TBD and should be selected using validation data.

---

# 19. Side-Effect Prediction Head

The second head performs multi-label classification.

```text
Patient representation
       ↓
Linear layer
       ↓
One score per side effect
       ↓
Probabilities
       ↓
Top-K
```

Example:

```text
Nausea       0.83
Dizziness    0.74
Bleeding     0.61
Headache     0.42
Rash         0.12
```

Top-K:

```text
Nausea
Dizziness
Bleeding
```

---

# 20. Multi-Task Training

```text
                 HGT
                  ↓
            Node Embeddings
             /          \
            ↓            ↓
       Severity       Side Effects
          Loss            Loss
             \            /
              \          /
                Total Loss
                    ↓
             Backpropagation
```

The model learns both tasks simultaneously.

---

# 21. Proposed Loss Design

This is a proposed implementation, not finalized by the source workflow.

```text
L_total =
λ1 L_severity
+
λ2 L_side_effect
```

where:

```text
L_severity = binary classification loss

L_side_effect = multi-label classification loss
```

`λ1` and `λ2` should be tuned on validation data.

---

# 22. Training Pipeline

```text
Processed datasets
       ↓
Graph construction
       ↓
Train/Validation/Test split
       ↓
Training HeteroData
       ↓
HGT
       ↓
Severity prediction
       ↓
Side-effect prediction
       ↓
Loss
       ↓
Backpropagation
       ↓
Parameter update
       ↓
Validation
       ↓
Best model checkpoint
```

---

# 23. Data Splitting Requirement

Splitting should happen at an appropriate case/patient level.

Avoid:

```text
Same clinical case
    ↓
partly in train
partly in test
```

because this can cause information leakage.

A possible initial split is:

```text
70% train
15% validation
15% test
```

but this is **proposed only** and should be finalized after inspecting the dataset.

---

# 24. Class Imbalance

The system must measure:

```text
Severe cases
Non-severe cases
```

If severe cases are underrepresented, investigate:

- weighted loss,
- sampling,
- threshold optimization.

The final approach should be experimentally selected.

---

# 25. Explainability Module

The explainability module captures HGT attention during inference.

```text
HGT forward pass
       ↓
Attention values
       ↓
Filter patient-relevant edges
       ↓
Aggregate attention
       ↓
Map attention back to drugs
       ↓
Drug scores
       ↓
Risk ranking
```

---

# 26. Drug Risk Hierarchy

Example:

```text
Patient P1

Drug A → 0.12
Drug B → 0.61
Drug C → 0.18
Drug D → 0.09
```

Rank:

```text
1. Drug B
2. Drug C
3. Drug A
4. Drug D
```

This is the project's **Risk Hierarchy**.

---

# 27. Attention Interpretation Constraint

The system should say:

> **“Drug B received the highest model-derived attention/contribution score.”**

It should not say:

> **“Drug B biologically caused the adverse event.”**

Attention is being used as an interpretability signal for model behavior, not as proof of biological causality.

---

# 28. Attention Aggregation

The exact aggregation strategy is not specified by the source and must be finalized experimentally.

Potential strategies:

```text
mean
sum
weighted mean
layer-weighted aggregation
head-weighted aggregation
```

The chosen approach should be validated.

---

# 29. Counterfactual Recommender

The counterfactual module is:

```text
Original regimen
       ↓
Predict risk
       ↓
Identify highest-risk drug
       ↓
Find ATC alternatives
       ↓
Generate candidate regimens
       ↓
Run HGT
       ↓
Compare predicted severity
       ↓
Select lowest-risk candidate
```

---

# 30. Counterfactual Algorithm

### Input

```text
Patient P
Drug set D = {D1, D2, ..., Dn}
```

### Step 1

Calculate:

```text
Risk_original
```

### Step 2

Calculate drug risk hierarchy:

```text
R(D1), R(D2), ..., R(Dn)
```

### Step 3

Select:

```text
D_risky = argmax R(Di)
```

### Step 4

Find ATC alternatives:

```text
Alternatives = ATC_candidates(D_risky)
```

### Step 5

For every candidate:

```text
D' = D - {D_risky} + {candidate}
```

### Step 6

Construct the counterfactual subgraph.

### Step 7

Run HGT.

### Step 8

Calculate:

```text
Risk_candidate
```

### Step 9

Select:

```text
candidate* = argmin Risk_candidate
```

subject to all defined constraints.

---

# 31. Counterfactual Graph Reconstruction

If:

```text
Original drug = B
Replacement = Y
```

do not merely change the drug ID.

Original:

```text
Patient → B
B → Protein P1
B → Protein P2
B → Side Effect X
B → Drug C
```

Counterfactual:

```text
Patient → Y
Y → Protein P7
Y → Protein P8
Y → Side Effect Z
Y → Drug C
```

The replacement drug must bring its own relevant features and relationships.

---

# 32. Recommendation Constraints

## Constraint 1 — Therapeutic compatibility

```text
ATC(candidate) compatible with ATC(original)
```

## Constraint 2 — Preserve other medications

All other medications remain unchanged in the first version.

## Constraint 3 — Valid graph representation

Candidate must have enough information to construct its relevant graph neighborhood.

## Constraint 4 — Data coverage

Candidates with insufficient graph/data coverage should be flagged rather than treated as equally reliable.

---

# 33. Recommendation Output

The system returns:

```text
Original Regimen
↓
Predicted Severity
↓
Top-K Side Effects
↓
Highest-Risk Drug
↓
ATC Class
↓
Candidate Alternatives
↓
Predicted Severity for Each
↓
Recommended Counterfactual
```

---

# 34. Evaluation Strategy

Evaluation must occur at three levels.

## 34.1 Severity Prediction

Metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC

Recall for severe cases should receive particular attention.

---

## 34.2 Side-Effect Prediction

Potential metrics:

- Micro-F1
- Macro-F1
- Precision@K
- Recall@K
- Hamming loss
- Average Precision / suitable ranking metrics

For the Top-K objective, Precision@K and Recall@K should be prominent.

---

## 34.3 Counterfactual Evaluation

For severe cases:

```text
Original severity
        ↓
Counterfactual severity
```

Measure:

```text
ΔRisk =
Risk_original - Risk_counterfactual
```

A positive value indicates a reduction in predicted risk.

---

# 35. Counterfactual Metrics

Proposed metrics:

## Risk Reduction

```text
RRR =
(R_original - R_counterfactual)
/
R_original
```

## Successful Reduction Rate

```text
count(R_counterfactual < R_original)
/
count(test cases)
```

## Average Risk Reduction

```text
mean(R_original - R_counterfactual)
```

## Recommendation Coverage

Percentage of cases where:

```text
risky drug identified
AND
ATC alternative found
AND
counterfactual graph constructed
```

---

# 36. Baseline Models

The project foundation includes a homogeneous GCN baseline.

Recommended comparison:

```text
Baseline:
GCN

Additional baseline:
R-GCN (optional)

Proposed:
HGT
```

Compare:

- severity performance,
- side-effect performance,
- explainability capability,
- counterfactual performance.

---

# 37. Ablation Studies

## Experiment A — Molecular Network

```text
HGT without molecular network
vs
HGT + Drug Target + PPI
```

## Experiment B — TWOSIDES

```text
HGT without TWOSIDES
vs
HGT + TWOSIDES
```

## Experiment C — Multi-Task Learning

```text
Severity only
vs
Severity + Side Effects
```

## Experiment D — ATC Constraint

```text
Counterfactual without ATC constraint
vs
ATC-constrained counterfactual
```

These experiments help determine whether each component contributes useful information.

---

# 38. Explainability Evaluation

The explainability component should be evaluated rather than only visualized.

Potential questions:

```text
Does the highest-ranked drug correlate with
known drug-event relationships?

Does changing/removing the highest-ranked drug
substantially change the model prediction?
```

A useful analysis:

```text
Original graph
     ↓
Risk = R

Modify top-ranked drug
     ↓
Risk = R'
```

---

# 39. Proposed Technology Stack

```text
Python
PyTorch
PyTorch Geometric
Pandas
NumPy
Scikit-learn
Streamlit
```

Optional:

```text
CUDA
GPU
```

for model training.

---

# 40. Recommended Repository Structure

```text
polypharmacy-risk/
│
├── data/
│   ├── raw/
│   │   ├── faers/
│   │   ├── twosides/
│   │   ├── rxnorm/
│   │   ├── atc/
│   │   ├── drug_target/
│   │   └── ppi/
│   │
│   ├── processed/
│   └── mappings/
│
├── src/
│   ├── preprocessing/
│   │   ├── faers.py
│   │   ├── twosides.py
│   │   ├── rxnorm.py
│   │   └── atc.py
│   │
│   ├── graph/
│   │   ├── schema.py
│   │   ├── builder.py
│   │   ├── features.py
│   │   └── subgraph.py
│   │
│   ├── models/
│   │   ├── hgt.py
│   │   ├── heads.py
│   │   └── losses.py
│   │
│   ├── explainability/
│   │   ├── attention.py
│   │   └── risk_hierarchy.py
│   │
│   ├── counterfactual/
│   │   ├── alternatives.py
│   │   ├── graph_swap.py
│   │   └── recommender.py
│   │
│   ├── evaluation/
│   │   ├── severity.py
│   │   ├── side_effects.py
│   │   └── counterfactual.py
│   │
│   └── utils/
│
├── notebooks/
├── models/
├── configs/
│
├── app/
│   └── streamlit_app.py
│
├── tests/
├── requirements.txt
└── README.md
```

---

# 41. Configuration Management

Hyperparameters should not be hard-coded.

Example:

```yaml
model:
  hidden_dim: TBD
  num_heads: TBD
  num_layers: 2
  dropout: TBD

training:
  batch_size: TBD
  learning_rate: TBD
  epochs: TBD

prediction:
  top_k_side_effects: TBD

counterfactual:
  max_candidates: TBD
```

The architecture currently specifies two HGT layers; other hyperparameters are TBD.

---

# 42. Model Checkpointing

Save:

```text
best_model.pt
```

along with:

```text
model configuration
feature schema
node mappings
edge schema
training metadata
```

This ensures that the UI uses exactly the same graph encoding and model configuration as training.

---

# 43. Inference Pipeline

```text
User enters drugs
        ↓
Normalize drug names
        ↓
RxNorm mapping
        ↓
ATC mapping
        ↓
Retrieve graph entities
        ↓
Construct patient subgraph
        ↓
HGT inference
        ↓
Severity
+
Side effects
        ↓
Attention extraction
        ↓
Risk hierarchy
        ↓
Highest-risk drug
        ↓
ATC alternatives
        ↓
Counterfactual graph generation
        ↓
HGT for each candidate
        ↓
Risk comparison
        ↓
Recommendation
```

---

# 44. Streamlit UI

## Section 1 — Input

```text
POLYPHARMACY RISK ANALYZER

Patient/Case Information

Drug 1: [________]
Drug 2: [________]
Drug 3: [________]
Drug 4: [________]

[ ANALYZE ]
```

---

## Section 2 — Risk Assessment

```text
Overall Severity
HIGH

Predicted probability:
82%
```

Top-K side effects:

```text
1. Side Effect A
2. Side Effect B
3. Side Effect C
```

---

## Section 3 — Explainability

```text
Drug Risk Hierarchy

1. Drug B     ██████████
2. Drug C     ████
3. Drug A     ███
4. Drug D     ██
```

Display:

```text
Highest-risk drug:
Drug B
```

Label the ranking as:

> Model-derived risk contribution / attention ranking

---

## Section 4 — Counterfactual Recommendation

```text
Highest-risk drug:
Drug B

Therapeutic class:
ATC-X

Alternatives:

Drug X     Risk = 0.44
Drug Y     Risk = 0.31  ← Recommended
Drug Z     Risk = 0.56
```

Then:

```text
Original:
A + B + C + D

Counterfactual:
A + Y + C + D

Predicted severity:
0.87 → 0.31
```

---

# 45. UI Safety Disclaimer

Display prominently:

> **Research prototype:** Predictions and counterfactual recommendations are generated by an AI model and are intended for research/decision-support evaluation. They do not constitute medical advice or an autonomous prescribing recommendation. Clinical decisions must be made by qualified healthcare professionals.

---

# 46. Optional API Architecture

For a future production-style system:

```text
Streamlit
    ↓
FastAPI
    ↓
Inference Service
    ↓
HGT Model
    ↓
Graph/Data Layer
```

For the first research prototype:

```text
Streamlit → Python inference
```

is sufficient.

---

# 47. Logging

An inference record can contain:

```text
timestamp
input drug IDs
normalized IDs
model version
predicted severity
top-K side effects
risk hierarchy
counterfactual candidates
selected recommendation
```

Avoid storing unnecessary sensitive patient information.

---

# 48. Reproducibility

Record:

```text
Random seed
Dataset version
Preprocessing version
Model version
Hyperparameters
Feature schema
Graph schema
Train/validation/test split
```

---

# 49. Testing Strategy

## 49.1 Unit Tests

Test:

```text
Drug normalization
ATC mapping
Node mapping
Edge creation
HeteroData creation
Subgraph extraction
Counterfactual swapping
```

---

## 49.2 Graph Tests

Verify:

```text
Every edge source exists
Every edge target exists
No invalid node indices
Expected node types exist
Expected edge types exist
```

---

## 49.3 Model Tests

Verify:

```text
Forward pass succeeds
Correct tensor shapes
No NaNs
Severity output shape correct
Side-effect output shape correct
Attention extraction works
```

---

## 49.4 End-to-End Test

Input:

```text
Patient P1
Drugs A, B, C
```

Expected pipeline:

```text
subgraph
 ↓
HGT
 ↓
severity
 ↓
side effects
 ↓
attention
 ↓
risk hierarchy
 ↓
ATC alternatives
 ↓
counterfactual predictions
 ↓
recommendation
```

---

# 50. Data Quality Dashboard

Before training, generate:

```text
Number of cases
Number of polypharmacy cases
Number of drugs
Number of proteins
Number of side effects
Number of edges per relation
Severity distribution
Side-effect distribution
RxNorm mapping rate
ATC mapping rate
Drug-target coverage
PPI coverage
```

---

# 51. Final ETL Data Flow

```text
FAERS
  ↓
clean
  ↓
filter
  ↓
polypharmacy
  ↓
clinical labels
  │
  ├───────────────┐
  ↓               ↓
RxNorm          Side Effects
  ↓
ATC
  │
  ├───────────────┐
  ↓               ↓
Drugs          Therapeutic class
  │
  ↓
Drug Targets
  ↓
Proteins
  ↓
PPI
  ↓
Unified Graph
```

---

# 52. Project Milestones

## Milestone 0 — Project Definition

Deliverables:

- PRD
- graph schema
- dataset inventory
- target definition
- evaluation plan

---

## Milestone 1 — Data Pipeline

Deliverables:

- FAERS cleaned
- TWOSIDES cleaned
- RxNorm mapping
- ATC mapping
- Drug-target integration
- PPI integration

---

## Milestone 2 — Graph Construction

Deliverables:

- node dictionaries
- edge dictionaries
- feature matrices
- HeteroData
- graph validation
- subgraph extraction

---

## Milestone 3 — Baseline

Implement:

```text
GCN / existing baseline
```

Evaluate:

```text
Severity
Side effects
```

---

## Milestone 4 — HGT

Implement:

```text
HGT Layer 1
HGT Layer 2
Severity Head
Side-effect Head
Training
Validation
Checkpointing
```

Deliverable:

```text
trained_hgt.pt
```

---

## Milestone 5 — Explainability

Implement:

```text
Attention extraction
Attention aggregation
Drug scoring
Risk hierarchy
Visualization
```

---

## Milestone 6 — Counterfactual Recommender

Implement:

```text
Highest-risk drug selection
ATC candidate retrieval
Graph replacement
Counterfactual inference
Risk ranking
Best candidate selection
```

---

## Milestone 7 — Evaluation

Run:

```text
Severity evaluation
Side-effect evaluation
Counterfactual evaluation
Ablations
Baseline comparison
```

---

## Milestone 8 — Streamlit

Build:

```text
Input UI
Risk UI
Side-effect UI
Risk hierarchy UI
Counterfactual UI
Graph visualization
```

---

## Milestone 9 — Final Integration

```text
DATA
 ↓
GRAPH
 ↓
HGT
 ↓
EXPLANATION
 ↓
COUNTERFACTUAL
 ↓
EVALUATION
 ↓
UI
```

---

## Milestone 10 — Research Documentation

Produce:

```text
System architecture
Dataset methodology
Graph construction methodology
HGT methodology
Explainability methodology
Counterfactual methodology
Experimental results
Limitations
Future work
```

---

# 53. Definition of Done

## Data

- [ ] FAERS preprocessing works.
- [ ] TWOSIDES integration works.
- [ ] RxNorm normalization works.
- [ ] ATC mapping works.
- [ ] Drug-target data integrated.
- [ ] PPI data integrated.

## Graph

- [ ] All node types implemented.
- [ ] All approved edge types implemented.
- [ ] PyG HeteroData successfully generated.
- [ ] Graph validation passes.
- [ ] Patient subgraph extraction works.

## Model

- [ ] HGT implemented.
- [ ] Two HGT layers implemented.
- [ ] Severity head implemented.
- [ ] Multi-label side-effect head implemented.
- [ ] Training pipeline works.
- [ ] Validation pipeline works.
- [ ] Checkpointing works.

## Explainability

- [ ] HGT attention extraction works.
- [ ] Attention aggregation works.
- [ ] Drug risk ranking works.
- [ ] Risk hierarchy generated.

## Counterfactual

- [ ] Highest-risk drug identified.
- [ ] ATC alternatives retrieved.
- [ ] Candidate graph generated.
- [ ] HGT rerun for candidates.
- [ ] Candidate severity compared.
- [ ] Best candidate selected.

## Evaluation

- [ ] Baseline comparison completed.
- [ ] Severity metrics calculated.
- [ ] Side-effect metrics calculated.
- [ ] Counterfactual metrics calculated.
- [ ] Ablation experiments completed.

## UI

- [ ] Drug input works.
- [ ] Severity displayed.
- [ ] Side effects displayed.
- [ ] Risk hierarchy displayed.
- [ ] Counterfactual alternatives displayed.
- [ ] Recommendation displayed.
- [ ] Safety disclaimer displayed.

---

# 54. Critical Research Risks

## Risk 1 — Data Leakage

Potential problem:

```text
Same case information
appears in train and test
```

Mitigation:

```text
Case-level splitting
```

---

## Risk 2 — Incomplete Drug Mappings

Some drugs may not map cleanly to RxNorm/ATC.

Mitigation:

```text
Mapping coverage report
```

and explicit handling of unmapped entities.

---

## Risk 3 — Sparse Biological Information

Some drugs may lack target information.

Mitigation:

```text
Coverage analysis
Missing-data strategy
```

---

## Risk 4 — Attention Misinterpretation

Problem:

> Attention is not automatically causal evidence.

Mitigation:

Use:

```text
Model-derived contribution/risk ranking
```

rather than:

```text
Causal drug effect
```

---

## Risk 5 — Counterfactual Model Bias

A candidate may appear lower-risk because it has sparse or poorly represented data.

Mitigation:

Consider:

```text
Candidate coverage
Prediction confidence
Graph completeness
```

---

## Risk 6 — Clinically Invalid Recommendation

Same-ATC alternatives are not automatically appropriate for every patient.

Therefore:

```text
Model output
≠
Final clinical prescription
```

---

## Risk 7 — Label Definition

FAERS severity is not necessarily a perfect representation of true clinical severity.

Document:

```text
How severe outcomes were defined
How labels were generated
```

and discuss limitations.

---

# 55. Security & Privacy

The system should:

- avoid storing unnecessary patient information,
- use anonymized research IDs,
- avoid exposing raw case identifiers in the UI,
- restrict access to raw datasets,
- separate development and production data,
- avoid transmitting sensitive data unnecessarily.

---

# 56. Performance Requirements

## Training

```text
GPU acceleration preferred
```

## UI Inference

Target:

```text
Original regimen → response within a few seconds
```

Exact performance target should be measured after graph size and model complexity are known.

Counterfactual inference will naturally require additional computation:

```text
N alternatives
×
HGT inference
```

---

# 57. Counterfactual Performance Optimization

Instead of rebuilding the entire global graph for every candidate:

```text
Global graph
      ↓
Patient subgraph
      ↓
Clone relevant subgraph
      ↓
Swap drug
      ↓
Run HGT
```

Cache reusable information such as:

```text
Drug features
Drug-target relationships
ATC mappings
Protein neighborhoods
```

---

# 58. Versioning

Every experiment/model should record:

```text
Model Version
Dataset Version
Graph Schema Version
Feature Version
```

Example:

```text
HGT-v1
GraphSchema-v1
Dataset-v1
Features-v1
```

---

# 59. Final Research Architecture

```text
                  ┌─────────────────┐
                  │   FAERS         │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │   TWOSIDES      │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │ PREPROCESSING   │
                  └────────┬────────┘
                           │
                 ┌─────────▼─────────┐
                 │ RxNorm + ATC      │
                 └─────────┬─────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
      Drug Targets       PPI       Side Effects
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                 ┌──────────────────┐
                 │ HETEROGENEOUS    │
                 │ GRAPH BUILDER    │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ PyG HeteroData   │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ Patient Subgraph │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │     HGT #1       │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │     HGT #2       │
                 └────────┬─────────┘
                          ▼
                  Node Representations
                     /          \
                    /            \
                   ▼              ▼
             Severity Head   Side Effect Head
                   │              │
                   ▼              ▼
                Severity       Top-K Effects
                   │
                   ▼
            Attention Extraction
                   │
                   ▼
            Drug Risk Hierarchy
                   │
                   ▼
             Highest-Risk Drug
                   │
                   ▼
              ATC Alternatives
                   │
                   ▼
          Counterfactual Graphs
             /      |       \
            /       |        \
           ▼        ▼         ▼
        HGT-X     HGT-Y     HGT-Z
           │        │         │
           ▼        ▼         ▼
        Risk-X    Risk-Y    Risk-Z
            \       |        /
             \      |       /
              └─────┼──────┘
                    ▼
             Lowest Predicted
                 Severity
                    │
                    ▼
          Recommended Regimen
                    │
                    ▼
             Streamlit UI
```

---

# 60. Complete End-to-End Example

Suppose the user enters:

```text
Patient P1

Drug A
Drug B
Drug C
Drug D
```

## Step 1 — Normalize

```text
Drug names
   ↓
RxNorm IDs
```

## Step 2 — Retrieve metadata

```text
ATC information
Drug targets
PPI relationships
Side-effect relationships
Drug interactions
```

## Step 3 — Build graph

```text
Patient → Drugs → Proteins → Side Effects
```

## Step 4 — Extract subgraph

```text
Patient-specific heterogeneous subgraph
```

## Step 5 — HGT

```text
HGT Layer 1
      ↓
HGT Layer 2
      ↓
Patient representation
```

## Step 6 — Prediction

```text
Severity = 0.87
```

## Step 7 — Side Effects

```text
Effect A = 0.82
Effect B = 0.71
Effect C = 0.63
```

## Step 8 — Attention

```text
Drug A = 0.12
Drug B = 0.61
Drug C = 0.18
Drug D = 0.09
```

Therefore:

```text
Highest-risk drug = Drug B
```

## Step 9 — ATC alternatives

```text
Drug B
 ↓
ATC class
 ↓
X
Y
Z
```

## Step 10 — Counterfactual simulation

```text
A+B+C+D → 0.87

A+X+C+D → 0.44

A+Y+C+D → 0.31

A+Z+C+D → 0.56
```

## Step 11 — Recommendation

```text
Original:
A + B + C + D

Recommended counterfactual:
A + Y + C + D

Predicted severity:
0.87 → 0.31
```

---

# 61. Core Research Contribution

The project's contribution can be summarized as:

```text
Clinical Data
     +
Molecular Data
     ↓
Heterogeneous Graph
     ↓
HGT
     ↓
Multi-task Prediction
     ↓
┌───────────────────────────┐
│ Severity                  │
│ Side Effects              │
└───────────────────────────┘
     ↓
Attention-based Explanation
     ↓
Risk Hierarchy
     ↓
Highest-risk Drug
     ↓
ATC-constrained Counterfactuals
     ↓
Repeated HGT Evaluation
     ↓
Risk-minimizing Alternative
```

## One-sentence project definition

> **The project transforms real-world polypharmacy cases and molecular drug/target networks into a heterogeneous graph, uses an HGT-based multi-task predictor to estimate severity and adverse effects, uses attention as a model-level interpretability signal to rank drugs, and performs ATC-constrained counterfactual drug substitutions to identify a regimen with lower predicted severity.**

---

# 62. Implementation Priority

The implementation should proceed in this order:

## Priority 1 — Graph Schema

```text
CSV
 ↓
Entities
 ↓
IDs
 ↓
Nodes
 ↓
Edges
 ↓
Features
 ↓
HeteroData
```

## Priority 2 — HGT

```text
HeteroData
 ↓
HGT
 ↓
Embeddings
 ↓
Severity + Side Effects
 ↓
Attention
```

## Priority 3 — Counterfactual Engine

```text
Attention
 ↓
Risky Drug
 ↓
ATC Candidates
 ↓
Graph Swap
 ↓
HGT
 ↓
Risk Ranking
```

## Priority 4 — Evaluation

```text
Baseline
+
HGT
+
Ablation
+
Counterfactual evaluation
```

## Priority 5 — UI

```text
Streamlit
 ↓
Input
 ↓
Prediction
 ↓
Explanation
 ↓
Counterfactual Recommendation
```

---

# 63. Final Definition of the System

At completion, the system should behave as:

```text
                    USER
                     │
                     ▼
             Drug Combination
                     │
                     ▼
             Drug Normalization
                     │
                     ▼
              Patient Subgraph
                     │
                     ▼
          ┌─────────────────────┐
          │ Heterogeneous Graph │
          │ Transformer (HGT)   │
          └──────────┬──────────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
         Severity       Side Effects
              │             │
              └──────┬──────┘
                     ▼
              Attention Analysis
                     │
                     ▼
              Risk Hierarchy
                     │
                     ▼
              Highest-Risk Drug
                     │
                     ▼
                ATC Lookup
                     │
                     ▼
             Alternative Drugs
                     │
                     ▼
           Counterfactual Graphs
                     │
                     ▼
                HGT Scoring
                     │
                     ▼
             Risk Comparison
                     │
                     ▼
            Recommended Regimen
                     │
                     ▼
                  STREAMLIT
```

**End state:** a reproducible research prototype that goes from raw clinical/molecular data to a heterogeneous graph, HGT prediction, explainable drug-risk ranking, and ATC-constrained counterfactual risk-reduction analysis.
