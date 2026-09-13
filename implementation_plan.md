# Explainable Heterogeneous GNN Implementation Plan

## Goal Description
You want to upgrade your model to utilize the new heterogeneous graph data. The new architecture needs to not only predict severity but also provide **explainability** (which drugs/edges caused the risk), **counterfactual recommendations** (suggesting safer alternative drug combinations), and **side-effect prediction**.

To achieve this, we will migrate from the standard `GCNConv` (which is homogeneous and non-explainable) to a **Heterogeneous Graph Transformer (HGT)** or **Heterogeneous Graph Attention Network (HAN)**. These architectures natively compute *attention weights* over edges. By inspecting these attention weights, we can explain exactly *why* a prediction was made.

## User Review Required

> [!WARNING]
> **Major Architectural Shift**
> Moving from a Homogeneous GCN to a Heterogeneous Attention Network requires a complete rewrite of your PyTorch Geometric (PyG) dataset loader and training loop. We will create a new directory `LAYER_6/` to contain this advanced architecture so we don't break your existing Layer 4 and Layer 5 code.

## Open Questions

> [!IMPORTANT]
> 1. **Side Effect Prediction:** There are 13,000+ unique side effects. Predicting all 13k simultaneously as a multi-label classification task will result in a massive output layer and sparse gradients. I propose we filter the target side effects down to the **top 100 or 500 most clinically severe side effects** (e.g., Anaphylaxis, Myocardial Infarction, etc.). Do you approve of filtering the prediction targets?
> 2. **Safer Combinations (Counterfactuals):** To find "safer" combinations, the model will test substituting drugs. Should it only search for substitutes within the same ATC Class (therapeutic equivalents), or should it blindly test all 1,920 drugs? (I recommend restricting to the same ATC level 3/4 class to ensure the alternative actually treats the patient's underlying condition).

## Proposed Changes

We will create a new modeling pipeline in `LAYER_6/`.

### 1. `LAYER_6/dataset_builder.py`
This script will convert the CSVs in `DATA_AGGREGATION_CODE/Graph_Data/` into PyG `HeteroData` objects.
- It will extract patient-specific subgraphs on the fly.
- It will assign multi-label tensors for the target side effects (for Feature 4).

### 2. `LAYER_6/hetero_attention_model.py`
Defines the `ExplainableHeteroGNN` using `torch_geometric.nn.HGTConv`.
- The model will have two prediction heads:
  - **Head 1:** Binary classification for overall patient `Severity`.
  - **Head 2:** Multi-label classification for `Side Effects`.
- It will explicitly return the `attention_weights` tensor alongside the predictions.

### 3. `LAYER_6/explainer_and_recommender.py`
This script fulfills your core analytical requirements:
- **Risk Hierarchy (Requirement 1 & 4):** It runs a patient subgraph through the model, extracts the `attention_weights` for the `(drug)-[targets]->(protein)` and `(drug)-[interacts]->(drug)` edges, and ranks the drugs by their total attention score. The drug with the highest attention is the primary driver of the predicted risk/side effect.
- **Safer Combinations (Requirement 2):** It takes the highest-risk drug identified above, swaps it out for other drugs in the same ATC class, generates new candidate subgraphs, and passes them through the model. It then returns the combination that yields the lowest predicted severity score.

## Verification Plan

### Automated Tests
1. Verify that `HeteroData` objects successfully load without dimensional mismatches.
2. Verify that the dual-head HGT model computes both Loss(Severity) and Loss(SideEffects) and that gradients backpropagate.

### Manual Verification
1. We will run the `explainer_and_recommender.py` on a known "Severe" polypharmacy case from your dataset.
2. We will visually inspect the output to ensure the explanation (the hierarchical list of risky drugs/targets) makes logical clinical sense.
3. We will review the recommended "safer combinations" to ensure the predicted severity drops as expected.
