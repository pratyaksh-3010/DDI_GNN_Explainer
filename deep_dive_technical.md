# Deep-Dive Technical Documentation — Graph Construction, PyTorch Geometric, Embeddings & Explainability

---

## Part 1: Graph Construction — From Raw Data to a Drug Interaction Graph

### 1.1 Why a Graph? The Fundamental Insight

Traditional machine learning treats each patient case as a **flat feature vector** — e.g., "Patient took Drug A, Drug B, Drug C → risk = high/low." This loses crucial information: *how do these drugs interact with each other?*

A graph captures **relational structure**:
- Drug A and Drug B might interact strongly (shared enzyme targets)
- Drug B and Drug C might interact weakly
- Drug A and Drug C might not interact at all

By modeling this as a graph, we preserve the topology of interactions. The GNN can then reason about **paths** through the graph — for example, even if Drug A and Drug C don't directly interact, they might both interact with Drug B, creating an indirect risk pathway.

### 1.2 Node Construction — Step by Step

Each drug in our graph is a **node** with a 65-dimensional feature vector. Here's exactly how that vector is built:

#### Step 1: Identify All Unique Drugs

In [nodes_edges.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/LAYER%20_3_CODE/nodes_edges.py#L21-L34), we iterate over every row of [faers_layer2_final.csv](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/faers_layer2_final.csv) and collect all unique [structure_id](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/DATA_PREPROCESSING_CODE/faers_drugcentral_mapping.py#11-15) values:

```python
all_structs = set()
for row in faers["STRUCTURE_ID_LIST"]:
    sids = ast.literal_eval(row)       # Convert string "[1234, 5678]" to list
    all_structs.update([str(s) for s in sids])
```

This yields **1,198 unique drugs**. We sort them and assign each a sequential integer index (0 to 1197). This index is critical — it becomes the node's ID everywhere in the system.

#### Step 2: Build ATC Level-1 Multi-Hot Features (14 dimensions)

The ATC (Anatomical Therapeutic Chemical) classification has 14 top-level categories:

| Letter | Meaning | Example Drugs |
|--------|---------|---------------|
| A | Alimentary tract & Metabolism | Insulin, Metformin |
| B | Blood & Blood-forming organs | Warfarin, Heparin |
| C | Cardiovascular system | Atenolol, Lisinopril |
| D | Dermatologicals | Hydrocortisone cream |
| G | Genitourinary & Sex hormones | Estradiol |
| H | Systemic Hormones | Levothyroxine, Prednisone |
| J | Anti-infectives | Amoxicillin, Ciprofloxacin |
| L | Antineoplastic & Immunomodulating | Rituximab, Methotrexate |
| M | Musculoskeletal | Ibuprofen, Naproxen |
| N | Nervous system | Morphine, Diazepam, SSRIs |
| P | Antiparasitic products | Chloroquine |
| R | Respiratory system | Albuterol, Fluticasone |
| S | Sensory organs | Eye drops, Ear drops |
| V | Various | Contrast agents |

For each drug, we create a **14-dimensional binary vector** where a `1` in position `i` means the drug belongs to ATC category `i`. A drug can belong to multiple categories (e.g., a drug that treats both cardiovascular and metabolic conditions would have `1`s in positions C and A).

**Why multi-hot instead of one-hot?** Because ~15% of drugs have ATC codes spanning multiple Level-1 categories. One-hot would force us to pick just one, losing information.

#### Step 3: Build Target Class Multi-Hot Features (25 dimensions)

From DrugCentral's mechanism data, each drug is mapped to the **types of proteins it targets**. There are 25 unique target classes:

`Adhesion, Antibody, CD molecules, Cytokine, Cytosolic other, Enzyme, GPCR, Glycoprotein, Ion channel, Kinase, Membrane other, Membrane receptor, Nuclear hormone receptor, Nuclear other, Polyprotein, RNA, Ribosomal protein, Secreted, Structural, Surface antigen, Transcription factor, Transporter, Tumour-associated antigen, Unclassified, Viral envelope protein`

For each drug, we create a **25-dimensional binary vector** indicating which target classes it affects. For example:
- **Morphine** targets GPCRs (opioid receptors) → `tc_GPCR = 1`
- **Ibuprofen** targets Enzymes (COX-1/2) → `tc_Enzyme = 1`
- **Rituximab** targets Surface antigens (CD20) → `tc_Surface antigen = 1`

**Why this matters for polypharmacy:** Two drugs targeting the **same** protein class can compete, amplify effects, or cause toxicity. The GNN learns that combinations with overlapping target classes are riskier.

#### Step 4: Build Action Type Multi-Hot Features (23 dimensions)

Each drug has a *mechanism of action* — how it interacts with its targets. 23 unique action types:

`ACTIVATOR, AGONIST, ALLOSTERIC ANTAGONIST, ALLOSTERIC MODULATOR, ANTAGONIST, ANTIBODY BINDING, ANTISENSE INHIBITOR, BINDING AGENT, BLOCKER, GATING INHIBITOR, INHIBITOR, INVERSE AGONIST, MODULATOR, NEGATIVE ALLOSTERIC MODULATOR, NEGATIVE MODULATOR, OPENER, OTHER, PARTIAL AGONIST, PHARMACOLOGICAL CHAPERONE, POSITIVE ALLOSTERIC MODULATOR, POSITIVE MODULATOR, RELEASING AGENT, SUBSTRATE`

**Why this matters:** An AGONIST + ANTAGONIST for the same receptor cancel each other out. Two INHIBITORS of the same enzyme can cause excessive inhibition. The GNN learns these interaction patterns.

#### Step 5: Build Count Features (3 dimensions)

Three scalar features, **log-transformed**:

```python
X[i, offset + 0] = np.log1p(len(data["atc"]))    # Drug class diversity
X[i, offset + 1] = np.log1p(len(data["genes"]))   # Number of gene targets
X[i, offset + 2] = np.log1p(len(data["se"]))      # Number of known side effects
```

**Why `log1p` instead of raw counts?**
- Raw counts: Drug A has 2 side effects, Drug B has 500. The 500 completely dominates.
- `log1p(2) = 1.10`, `log1p(500) = 6.22`. Now the ratio is 1:5.6 instead of 1:250.
- `log1p(x) = log(1 + x)` ensures that 0 maps to 0 (not -infinity like plain log).

#### Step 6: Assemble the Final Feature Matrix

```
Feature vector layout for each drug (65 dims total):
┌──────────────┬───────────────┬──────────────┬──────────────┐
│ ATC L1 (14)  │ Target (25)   │ Action (23)  │ Counts (3)   │
│ positions    │ positions     │ positions    │ positions    │
│ [0:14]       │ [14:39]       │ [39:62]      │ [62:65]      │
└──────────────┴───────────────┴──────────────┴──────────────┘
```

The result is a matrix `X` of shape [(1198, 65)](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/DATA_PREPROCESSING_CODE/sider_processing.py#33-39) — 1198 drugs, each with 65 features.

### 1.3 Edge Construction — Drug-Drug Interactions

Edges represent **known drug-drug interactions** from the TWOSIDES database.

#### The PRR (Proportional Reporting Ratio)

For each drug pair (A, B) and side effect S, TWOSIDES provides a PRR:

```
PRR = (reports of S with A+B together) / (expected reports of S given base rates)
```

- `PRR = 1`: Side effect occurs at expected rate — no interaction signal
- `PRR > 1`: Side effect occurs **more** than expected — evidence of interaction
- `PRR = 5`: Side effect is 5× more frequent with this drug pair than expected

We filter for `PRR > 1` only, keeping interactions with evidence.

#### Edge Weight Computation

For each drug pair, there may be hundreds of side effects with different PRR values. We take the **maximum PRR** (worst-case interaction) and apply `log1p`:

```python
weight = np.log1p(max_PRR)
```

This means the edge weight represents the **severity of the worst known interaction** between two drugs. Higher weight = more dangerous interaction = stronger message passing in the GNN.

#### Why Undirected Edges?

Drug interactions are symmetric — if Drug A interacts with Drug B, then Drug B interacts with Drug A. We store each edge once (as the sorted pair) in the CSV, but expand to bidirectional during training:

```python
edge_index_list.append([i, j])  # A → B
edge_index_list.append([j, i])  # B → A
```

### 1.4 Training Case (Subgraph) Construction

Each FAERS patient case becomes a **subgraph** of the global drug graph:

1. **Identify the drugs**: Parse the `nodes` column → list of global node indices
2. **Extract subgraph edges**: For every pair of drugs in the case, check if they have an edge in the global graph
3. **Remap to local indices**: The subgraph has its own 0-based indexing
4. **Skip if no edges**: Cases with no known interactions between their drugs are discarded (the GCN needs edges for message passing)

**Example:**
```
Patient case: [Drug_42, Drug_107, Drug_891]
Global edges: 42↔107 (weight 2.3), 42↔891 (weight 1.1)
No edge between 107 and 891

Subgraph:
  Local node 0 = Global node 42
  Local node 1 = Global node 107
  Local node 2 = Global node 891
  Edges: [0,1], [1,0], [0,2], [2,0] with weights [2.3, 2.3, 1.1, 1.1]
  Label: 1 (high risk) or 0 (low risk)
```

---

## Part 2: PyTorch Geometric — Why It Was Chosen and How It Works

### 2.1 What Is PyTorch Geometric (PyG)?

PyTorch Geometric is a library built on top of PyTorch specifically designed for **deep learning on graphs and irregular structures**. It provides:

1. **Efficient graph data structures** (`Data`, `Batch`)
2. **Message passing layers** (`GCNConv`, `GATConv`, `SAGEConv`, etc.)
3. **Pooling operations** (`global_mean_pool`, `global_max_pool`, `TopKPooling`)
4. **Mini-batching** via `DataLoader` and `Batch`
5. **Explainability tools** (`GNNExplainer`, `Explainer`)

### 2.2 Why PyG Over Alternatives?

| Alternative | Why We Didn't Use It |
|-------------|---------------------|
| **DGL (Deep Graph Library)** | More verbose API, less integrated with PyTorch ecosystem |
| **NetworkX** | Pure Python, no GPU support, no gradient computation — only useful for analysis (we do use it in [find_example_pairs.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/LAYER_6/find_example_pairs.py)) |
| **Custom from scratch** | Implementing message passing, sparse operations, and batching correctly is error-prone and slow |
| **Spektral (TensorFlow)** | We're in the PyTorch ecosystem; mixing frameworks adds complexity |

**PyG advantages for this project specifically:**
1. **`GCNConv` natively supports edge weights** — critical for our PRR-weighted interactions
2. **`DataLoader` handles variable-sized graphs** — each patient case has a different number of drugs
3. **`GNNExplainer` is built-in** — no need for a separate explainability library
4. **`global_mean_pool` handles batched graphs** — automatically knows which nodes belong to which graph in a batch

### 2.3 The `Data` Object — PyG's Core Abstraction

Every graph in PyG is a `Data` object:

```python
data = Data(
    x=tensor,           # Node features: shape [num_nodes, num_features]
    edge_index=tensor,   # Edge list: shape [2, num_edges]
    edge_weight=tensor,  # Edge weights: shape [num_edges]
    y=tensor,           # Graph label: shape [1]
    node_id=tensor      # Global node IDs: shape [num_nodes]
)
```

**Key design decision — `edge_index` format:**

PyG uses a **COO (Coordinate) format** for edges. `edge_index` is a `[2, num_edges]` LongTensor where:
- Row 0 = source node indices
- Row 1 = target node indices

```
edge_index = [[0, 1, 0, 2],   ← sources
              [1, 0, 2, 0]]   ← targets
```

This means: edge from 0→1, edge from 1→0, edge from 0→2, edge from 2→0.

**Why COO format?** It's the most efficient for sparse graphs (which drug interaction graphs are). Memory usage is O(edges) not O(nodes²).

### 2.4 Mini-Batching Variable-Sized Graphs

A fundamental challenge:each patient case has a different number of drugs (2 to 20+). You can't just stack them into a regular tensor. PyG solves this with `Batch`:

```python
from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
```

The `DataLoader` creates a `Batch` object that merges 64 separate graphs into a single **disconnected** super-graph:

```
Graph 1: nodes [0,1,2], edges [0→1, 1→2]
Graph 2: nodes [0,1], edges [0→1]

Batched: nodes [0,1,2,3,4], edges [0→1, 1→2, 3→4]
batch vector: [0,0,0,1,1]  ← tells us which graph each node belongs to
```

The `batch` vector is what makes `global_mean_pool` work — it knows to average nodes 0,1,2 separately from nodes 3,4.

This is far more efficient than processing graphs one at a time because it allows full GPU parallelism across the entire batch.

### 2.5 GCNConv — Message Passing with Edge Weights

The GCN (Graph Convolutional Network) layer from [Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907) performs:

```
h_i^{(l+1)} = σ( Σ_{j∈N(i)} (w_ij / √(d_i · d_j)) · W^{(l)} · h_j^{(l)} + b^{(l)} )
```

In plain English:
1. For each node `i`, look at all its neighbors `j`
2. Take neighbor `j`'s features, multiply by learnable weight matrix `W`
3. Scale by the edge weight `w_ij` and normalize by node degrees
4. Sum all neighbor contributions
5. Apply non-linearity (ReLU)

**In our project, the edge weight `w_ij` is `log1p(PRR)`** — so drug pairs with stronger known interactions exchange more information during message passing. This is how the model learns that high-PRR drug pairs are more dangerous.

After **2 GCN layers**, each node's representation contains information from its **2-hop neighborhood**. If Drug A interacts with Drug B, and Drug B interacts with Drug C, then after 2 layers, Drug A's representation contains indirect information about Drug C.

---

## Part 3: Embeddings — The Learnable Drug Identity

### 3.1 What Are Node Embeddings?

An **embedding** is a dense, learnable vector that represents an entity's identity. In our model:

```python
self.embedding = torch.nn.Embedding(num_nodes=1198, embedding_dim=32)
```

This creates a lookup table of 1198 × 32 = **38,336 learnable parameters** — one 32-dimensional vector for each drug in the graph.

### 3.2 Why Are Embeddings Needed?

Our hand-crafted features (65 dims) capture **pharmacological properties** — what class a drug belongs to, what targets it hits, how it acts. But two drugs can have identical feature vectors while being very different. For example:

- **Morphine** and **Fentanyl** are both `ATC: N` (Nervous System), `tc_GPCR = 1`, `at_AGONIST = 1`
- But Fentanyl is 50-100× more potent than Morphine

The hand-crafted features can't capture this because potency isn't in our feature set. The **learnable embedding** fills this gap — during training, the model learns a unique 32-dimensional "identity vector" for each drug that captures whatever information is useful for prediction but missing from the hand-crafted features.

### 3.3 How Embeddings Integrate with Features

In the [forward pass](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/LAYER_4_CODE/train_gnn.py#L181-L217):

```python
def forward(self, x, edge_index, edge_weight, batch, node_id):
    # x = hand-crafted features (shape: [num_nodes, 65])
    # node_id = global drug indices (shape: [num_nodes])
    
    embed = self.embedding(node_id)    # shape: [num_nodes, 32]
    x = torch.cat([embed, x], dim=1)   # shape: [num_nodes, 97]
    
    # Now x has 97 dimensions per drug: 32 learned + 65 hand-crafted
    x = self.conv1(x, edge_index, edge_weight)  # 97 → 128
    ...
```

The concatenation `[embed | features]` gives the model **both** types of information:
- **embed**: Who is this drug? (learned identity)
- **features**: What type of drug is it? (pharmacological properties)

### 3.4 Embedding Training Dynamics

During backpropagation, gradients flow back through the GCN layers into the embedding table. This means:
- If Drug X frequently appears in high-risk cases, its embedding learns to encode "risky drug" information
- If two drugs often co-occur in risky combinations, their embeddings learn to produce representations that, when pooled together, push the prediction toward high risk
- The 32 dimensions are latent — we don't assign meaning to individual dimensions; the model discovers useful patterns automatically

### 3.5 Why 32 Dimensions?

This is a hyperparameter choice balancing:
- **Too few (e.g., 8)**: Not enough capacity to distinguish 1198 drugs
- **Too many (e.g., 256)**: Overfitting risk — each drug only appears in a subset of training cases, so very high-dimensional embeddings won't have enough gradient signal to learn well
- **32**: A standard sweet spot used in many drug/molecule embedding papers

### 3.6 The `node_id` Requirement

A subtle but critical detail: when building subgraphs, the embedding layer needs to know the **global** node ID, not the local subgraph index. This is why every `Data` object stores `node_id`:

```python
data = Data(
    x=X[node_indices],                              # Local features
    node_id=torch.tensor(node_indices, dtype=torch.long)  # Global IDs for embedding lookup
)
```

Without `node_id`, the embedding layer would look up indices 0, 1, 2 (local) instead of 42, 107, 891 (global), returning the wrong drug embeddings entirely.

---

## Part 4: Explainability — Making the Black Box Transparent

### 4.1 Why Explainability Matters in Healthcare

A model that says "this drug combination has 73% risk" is useless if a doctor can't understand *why*. Explainability serves three purposes:

1. **Clinical trust**: Doctors won't use a tool they can't understand
2. **Actionability**: Knowing *which drug* contributes most to risk lets doctors consider removing or substituting it
3. **Validation**: If the model says Drug A is risky because of its "Nervous System" classification and the doctor knows Drug A is a Nervous System drug, this increases confidence the model is reasoning correctly

### 4.2 Method 1: GNNExplainer — Deep Dive

#### The Core Idea

GNNExplainer (Ying et al., 2019) asks: "What is the **smallest subgraph and subset of features** that explains this prediction?" It learns two types of masks:

1. **Node feature mask** `M_f ∈ R^{num_nodes × num_features}`: For each node, which features matter?
2. **Edge mask** `M_e ∈ R^{num_edges}`: Which edges matter?

#### How It Works — Optimization

GNNExplainer formulates explanation as an **optimization problem**:

```
Maximize: MI(Y, (G_s, X_s))
Subject to: G_s ⊆ G (subgraph of original graph)
```

Where MI = Mutual Information between the prediction Y and the masked subgraph (G_s, X_s).

In practice, this becomes a **200-epoch optimization loop** where:
1. Initialize masks as learnable parameters (all ones)
2. Apply masks to features and edges: `x_masked = x ⊙ sigmoid(M_f)`, `edge_weight_masked = edge_weight ⊙ sigmoid(M_e)`
3. Forward pass with masked inputs
4. Compute loss = KL divergence between original prediction and masked prediction + mask size regularization
5. Update masks via gradient descent
6. After 200 epochs, the mask values indicate importance

#### Configuration in Our Project

```python
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',           # Explain model behavior, not data patterns
    node_mask_type='attributes',        # Mask individual features, not entire nodes
    edge_mask_type='object',            # Mask individual edges
    model_config=ModelConfig(
        mode='binary_classification',
        task_level='graph',             # Whole-graph prediction (not node-level)
        return_type='raw'               # Model outputs raw logits, not probabilities
    )
)
```

**Why `return_type='raw'`?** This is crucial. GNNExplainer internally applies sigmoid to convert logits to probabilities before computing its loss. If we said `return_type='probs'`, it would apply sigmoid twice, producing meaningless explanations. This was one of the bugs we fixed.

#### Extracting Explanations

```python
explanation = explainer(x, edge_index, edge_weight=..., batch=..., node_id=...)

# Per-node importance: sum feature mask values across all features
node_importance = explanation.node_mask.sum(dim=1)  # shape: [num_drugs]
node_importance_norm = node_importance / (node_importance.sum() + 1e-8)

# Per-feature importance: average across all nodes
feat_importance = explanation.node_mask.mean(dim=0)  # shape: [65]

# Per-edge importance: direct mask values
edge_importance = explanation.edge_mask  # shape: [num_edges]
```

**Normalization**: We divide by `sum + 1e-8` (epsilon prevents division by zero) to get relative percentages. So if Drug A has importance 0.45, Drug B has 0.35, Drug C has 0.20, the model is saying "Drug A contributes 45% to the risk prediction."

### 4.3 Method 2: Gradient × Input Attribution — Deep Dive

#### The Core Idea

Gradient × Input (Shrikumar et al., 2017) is a **first-order Taylor approximation** of how much each input feature contributes to the output:

```
attribution_i = |∂f/∂x_i · x_i|
```

Where `f` is the model output and `x_i` is the i-th input feature.

#### Intuition

- `∂f/∂x_i` (gradient): "How sensitive is the output to changes in this feature?"
- `x_i` (input): "How large is this feature actually?"
- Product: "How much does this feature actually contribute given its current value?"

A feature might have a large gradient (high sensitivity) but a zero value (the drug doesn't have this property), in which case it contributes nothing. Conversely, a feature might be active (value = 1) but have near-zero gradient (the model doesn't care about it), also contributing nothing.

#### Implementation

```python
# Clone input and enable gradient tracking
x_input = data.x.clone().detach().requires_grad_(True)

# Forward pass
logit = model(x_input, edge_index=..., edge_weight=..., batch=..., node_id=...)

# Backward pass — compute gradients of output w.r.t. input
model.zero_grad()
logit.backward()

# Extract attributions
grads = x_input.grad.detach().cpu().numpy()      # shape: [num_drugs, 65]
x_vals = x_input.detach().cpu().numpy()           # shape: [num_drugs, 65]
attr = np.abs(grads * x_vals)                     # shape: [num_drugs, 65]

# Per-drug importance: sum across features
drug_scores = attr.sum(axis=1)                    # shape: [num_drugs]
drug_scores_norm = drug_scores / (drug_scores.sum() + 1e-8)
```

**Why `abs`?** Both positive and negative gradients indicate importance. A negative gradient means "increasing this feature decreases risk" — which is still information about the feature's importance. We take absolute value to capture both directions.

### 4.4 GNNExplainer vs. Gradient Attribution — Comparison

| Aspect | GNNExplainer | Gradient × Input |
|--------|-------------|-----------------|
| **Type** | Optimization-based (200 epochs) | Analytical (single backward pass) |
| **Speed** | Slow (~2-5 seconds per graph) | Fast (~50ms per graph) |
| **What it explains** | Features + Edges | Features only |
| **Faithfulness** | High (directly optimizes explanation) | Medium (first-order approximation) |
| **Edge importance** | ✅ Yes | ❌ No |
| **Per-drug breakdown** | ✅ Yes | ✅ Yes |
| **Per-feature breakdown** | ✅ Yes | ✅ Yes |

**Why use both?** They capture different aspects:
- GNNExplainer finds the **minimal sufficient explanation** — if you remove everything except the highlighted parts, you'd get the same prediction
- Gradient Attribution measures **local sensitivity** — which features, if perturbed slightly, would change the prediction most

When both methods agree (e.g., both say Drug A is the most important), we have high confidence. When they disagree, it may indicate the model is using complex nonlinear interactions that neither method fully captures.

### 4.5 Integration with the Streamlit Frontend

The explainability methods are integrated through the [model_utils.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/STREAMLIT_FRONTEND/model_utils.py) module and displayed via three tabs in the frontend:

#### Tab 1: GNNExplainer Drug Importance

For each drug in the combination:
1. Show drug name + normalized importance score
2. Display a progress bar proportional to importance
3. Expand to see **which features** of that drug drive the risk:
   - e.g., "ATC: Antineoplastic: increases risk (score: 0.0336)"
   - e.g., "Target: Surface antigen: increases risk (score: 0.0151)"

Feature names are **beautified** from raw codes to human-readable labels:
```python
"atc_l1_L"  →  "ATC: Antineoplastic/Immunomod."
"tc_GPCR"   →  "Target: GPCR"
"at_INHIBITOR" → "Action: INHIBITOR"
"log_se_count" → "Known side effects count"
```

#### Tab 2: Gradient Attribution Drug Importance

Same per-drug ranking but computed via gradients. Allows the clinician to cross-validate findings between methods.

#### Tab 3: Drug-Drug Interaction Importance (Edge Importance)

Shows a table of drug pairs ranked by how important their **interaction** is to the prediction:

| Drug 1 | Drug 2 | Interaction Importance | Edge Weight |
|--------|--------|----------------------|-------------|
| Rituximab | Prednisolone | 0.8234 | 2.341 |
| Prednisolone | Methylprednisolone | 0.4521 | 1.876 |

This is **unique to GNNExplainer** — gradient attribution cannot provide edge-level importance. The edge importance tells clinicians which specific drug *pair* is the most problematic, not just which individual drug.

### 4.6 Explainability Output Example (Real Data)

From [explainability_samples.txt](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/RESULTS/explainability_samples.txt):

**High Risk Case: Rituximab + Prednisolone + Methylprednisolone**

```
Predicted Risk: 0.5462 (HIGH) | Ground Truth: HIGH

#1 Rituximab:         41.75% importance
    → Drug class diversity:     0.0815
    → ATC: Antineoplastic:      0.0336
    → Target: Surface antigen:  0.0151
    → Action: ANTIBODY BINDING: 0.0148
    → Known side effects:       0.0139

#2 Prednisolone:      34.00% importance
    → Drug class diversity:     0.0823
    → Target gene count:        0.0336
    → Known side effects:       0.0235

#3 Methylprednisolone: 24.25% importance
    → Drug class diversity:     0.0621
    → Known side effects:       0.0201
    → Target gene count:        0.0173
```

**Clinical interpretation**: Rituximab (a monoclonal antibody used in cancer/autoimmune disease) is flagged as the highest-risk contributor because:
1. It has high drug class diversity (belongs to multiple ATC categories)
2. It's an antineoplastic agent (known for severe side effects)
3. It targets surface antigens via antibody binding (a potent mechanism)
4. Combined with two corticosteroids (prednisolone + methylprednisolone), which both have many known side effects and gene targets, the total combination risk is elevated

---

## Part 5: End-to-End Flow — How Everything Connects

### The Complete Data Flow

```
 FAERS (raw text)          RxNorm (RRF)         DrugCentral (SQL dump)
       │                       │                        │
       ▼                       ▼                        ▼
 faers_filtering.py ──→ combine_faers.py ──→ faers_rxnorm_normalisation.py
                                                        │
                           ┌────────────────────────────┤
                           ▼                            ▼
                  faers_drugcentral_mapping.py    rxcui_structure_map.csv
                           │
              ┌────────────┼─────────────┐
              ▼            ▼             ▼
     structure_atc_    Structure_id_   sider_processing.py
     mapping.py        mechanism_         │
              │        mapping.py         │
              └────────────┼─────────────┘
                           ▼
                  faers_layer2_final.csv (enriched dataset)
                           │
              ┌────────────┼─────────────┐
              ▼                          ▼
    twosides_processing.py         nodes_edges.py (Layer 3)
              │                          │
              ▼                    ┌─────┼──────┐
    twosides_filtered.csv          ▼     ▼      ▼
                              nodes.csv edges.csv cases.csv
                                   │     │      │
                                   └─────┼──────┘
                                         ▼
                                   train_gnn.py (Layer 4)
                                         │
                                         ▼
                                 weighted_gcn_model.pth
                                    │           │
                              ┌─────┘           └──────┐
                              ▼                        ▼
                     explain_gnn.py (Layer 5)    Streamlit Frontend
                              │                   (app.py + model_utils.py
                              ▼                    + graph_builder.py)
                     Explainability analysis         │
                              │                      ▼
                              └───────────→  User-facing predictions
                                             with explanations
```

### How a User Prediction Works (Runtime)

1. User types "rituximab, prednisolone" in the Streamlit app
2. [graph_builder.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/STREAMLIT_FRONTEND/graph_builder.py) looks up "rituximab" → node_index 847, "prednisolone" → node_index 623
3. Gets features: `X[847]` and `X[623]` (65-dim each)
4. Checks adjacency list: Is there an edge between 847 and 623? If yes, include it with its weight
5. Creates a PyG `Data` object with 2 nodes, their features, edge(s), and `node_id=[847, 623]`
6. [model_utils.py](file:///c:/Users/praty/OneDrive/Documents/HEALTHCARE/STREAMLIT_FRONTEND/model_utils.py) loads the WeightedGCN model from checkpoint
7. Forward pass:
   - Embedding lookup: `embedding(847)` → 32-dim, `embedding(623)` → 32-dim
   - Concat with features: 97-dim per drug
   - GCNConv1: message passing with edge weights → 128-dim per drug
   - GCNConv2: second round of message passing → 64-dim per drug
   - Global mean pool: average the 2 drug representations → 64-dim
   - MLP: 64 → 32 → 1 → raw logit
   - Sigmoid: logit → probability
8. Display: "Risk: 54.6% — MODERATE-HIGH RISK"
9. Run GNNExplainer (200 optimization epochs) → drug importance + feature importance + edge importance
10. Run Gradient Attribution (single backward pass) → drug importance + feature importance
11. Display all explainability results in 3 tabs

---

*This document provides exhaustive technical detail on graph construction, PyTorch Geometric usage, embedding design, and explainability integration for the Polypharmacy GNN project.*
