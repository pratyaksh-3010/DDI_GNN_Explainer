import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from rdkit import Chem
from rdkit.Chem import AllChem
import json
import os

print("Building Heterogeneous Graph Dataset (V2)...")

v2_in = r"V2\Graph_Data"
out_dir = r"V2\Processed"
os.makedirs(out_dir, exist_ok=True)

# 1. Load Node Lists
print("Loading nodes...")
patients_df = pd.read_csv(os.path.join(v2_in, "nodes_patient.csv"), dtype=str)
drugs_df = pd.read_csv(os.path.join(v2_in, "nodes_drug.csv"), dtype=str)
proteins_df = pd.read_csv(os.path.join(v2_in, "nodes_protein.csv"), dtype=str)
se_df = pd.read_csv(os.path.join(v2_in, "nodes_side_effect.csv"), dtype=str)

# Map IDs to consecutive integers
patient_mapping = {caseid: i for i, caseid in enumerate(patients_df['caseid'])}
drug_mapping = {rxcui: i for i, rxcui in enumerate(drugs_df['rxcui'])}
protein_mapping = {prot: i for i, prot in enumerate(proteins_df['protein'])}
se_mapping = {se: i for i, se in enumerate(se_df['side_effect'])}

# 2. Extract Drug Features (Morgan Fingerprints)
print("Extracting Morgan Fingerprints...")
try:
    smiles_df = pd.read_csv(os.path.join(v2_in, "drug_smiles.csv"), dtype=str)
    smiles_dict = dict(zip(smiles_df['rxcui'], smiles_df['smiles']))
except FileNotFoundError:
    print("Warning: drug_smiles.csv not found. Using random embeddings for now.")
    smiles_dict = {}

fps = []
for rxcui in drugs_df['rxcui']:
    smiles = smiles_dict.get(rxcui)
    fp_arr = np.zeros(1024)
    if pd.notna(smiles) and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp_arr = np.array(fp)
        except:
            pass
    fps.append(fp_arr)
drug_x = torch.tensor(np.array(fps), dtype=torch.float)

# 3. Extract Protein Features (Degree Centrality)
print("Extracting Protein Degree Centrality...")
ppi_df = pd.read_csv(os.path.join(v2_in, "edges_protein_protein.csv"), dtype=str)
prot_degrees = pd.concat([ppi_df['symbol_1'], ppi_df['symbol_2']]).value_counts()
prot_x = []
for p in proteins_df['protein']:
    deg = prot_degrees.get(p, 0)
    prot_x.append([deg])
protein_x = torch.tensor(prot_x, dtype=torch.float)

# 4. Extract Side Effect Features (Frequency)
print("Extracting Side Effect Frequencies...")
ddi_df = pd.read_csv(os.path.join(v2_in, "edges_drug_drug.csv"), dtype=str)
se_freqs = ddi_df['side_effect'].value_counts(normalize=True)
se_x = []
for se in se_df['side_effect']:
    freq = se_freqs.get(se, 0.0)
    se_x.append([freq])
side_effect_x = torch.tensor(se_x, dtype=torch.float)

# 5. Extract Patient Features (Random Initialization since demographics are missing)
# We will just use an empty feature vector and let the HGT model aggregate neighbor features
print("Initializing Patient nodes...")
patient_x = torch.ones((len(patients_df), 1), dtype=torch.float)

# 6. Build PyG HeteroData Object
print("\nBuilding PyG HeteroData object...")
data = HeteroData()

data['patient'].x = patient_x
data['drug'].x = drug_x
data['protein'].x = protein_x
data['side_effect'].x = side_effect_x

# Add Labels
data['patient'].y = torch.tensor(pd.to_numeric(patients_df['severity']).values, dtype=torch.float)

# 7. Add Edges
print("Adding edges...")
# Patient -> Drug
pd_df = pd.read_csv(os.path.join(v2_in, "edges_patient_drug.csv"), dtype=str)
# Filter valid
pd_df = pd_df[pd_df['caseid'].isin(patient_mapping) & pd_df['rxcui'].isin(drug_mapping)]
pd_src = [patient_mapping[c] for c in pd_df['caseid']]
pd_dst = [drug_mapping[r] for r in pd_df['rxcui']]
data['patient', 'prescribed', 'drug'].edge_index = torch.tensor([pd_src, pd_dst], dtype=torch.long)

# Drug -> Protein
dp_df = pd.read_csv(os.path.join(v2_in, "edges_drug_protein.csv"), dtype=str)
dp_df = dp_df[dp_df['rxcui'].isin(drug_mapping) & dp_df['protein'].isin(protein_mapping)]
dp_src = [drug_mapping[r] for r in dp_df['rxcui']]
dp_dst = [protein_mapping[p] for p in dp_df['protein']]
data['drug', 'targets', 'protein'].edge_index = torch.tensor([dp_src, dp_dst], dtype=torch.long)

# Protein -> Protein
ppi_df = ppi_df[ppi_df['symbol_1'].isin(protein_mapping) & ppi_df['symbol_2'].isin(protein_mapping)]
ppi_src = [protein_mapping[p] for p in ppi_df['symbol_1']]
ppi_dst = [protein_mapping[p] for p in ppi_df['symbol_2']]
data['protein', 'interacts_with', 'protein'].edge_index = torch.tensor([ppi_src, ppi_dst], dtype=torch.long)

# Drug -> Drug (DEDUPLICATED: one edge per unique drug pair, weight = mean PRR)
print("Deduplicating Drug-Drug edges...")
ddi_df = ddi_df[ddi_df['drug1_rxcui'].astype(str).isin(drug_mapping) & 
                ddi_df['drug2_rxcui'].astype(str).isin(drug_mapping) & 
                ddi_df['side_effect'].astype(str).isin(se_mapping)]
ddi_df['PRR_numeric'] = pd.to_numeric(ddi_df['PRR'], errors='coerce').fillna(1.0)

# Group by unique drug pairs, aggregate PRR with mean
ddi_dedup = ddi_df.groupby(['drug1_rxcui', 'drug2_rxcui'], as_index=False).agg(
    mean_prr=('PRR_numeric', 'mean'),
    num_shared_se=('side_effect', 'count')
)
print(f"  DDI edges: {len(ddi_df)} raw -> {len(ddi_dedup)} unique pairs")

ddi_src = [drug_mapping[str(r)] for r in ddi_dedup['drug1_rxcui']]
ddi_dst = [drug_mapping[str(r)] for r in ddi_dedup['drug2_rxcui']]
prr_weights = ddi_dedup['mean_prr'].values

data['drug', 'interacts_with', 'drug'].edge_index = torch.tensor([ddi_src, ddi_dst], dtype=torch.long)
data['drug', 'interacts_with', 'drug'].edge_attr = torch.tensor(
    np.column_stack([prr_weights, ddi_dedup['num_shared_se'].values]), 
    dtype=torch.float
)  # 2-dim edge features: [mean_PRR, num_shared_side_effects]

# Drug -> Side Effect (DEDUPLICATED: one edge per unique (drug, side_effect), weight = mean PRR)
print("Deduplicating Drug->Side Effect edges...")
# Combine drug1->SE and drug2->SE, then deduplicate
d1_se = ddi_df[['drug1_rxcui', 'side_effect', 'PRR_numeric']].rename(columns={'drug1_rxcui': 'rxcui'})
d2_se = ddi_df[['drug2_rxcui', 'side_effect', 'PRR_numeric']].rename(columns={'drug2_rxcui': 'rxcui'})
all_drug_se = pd.concat([d1_se, d2_se], ignore_index=True)

# Group by unique (drug, side_effect) pairs
drug_se_dedup = all_drug_se.groupby(['rxcui', 'side_effect'], as_index=False).agg(
    mean_prr=('PRR_numeric', 'mean')
)
# Filter to valid mappings
drug_se_dedup = drug_se_dedup[drug_se_dedup['rxcui'].isin(drug_mapping) & 
                               drug_se_dedup['side_effect'].isin(se_mapping)]
print(f"  Drug->SE edges: {len(all_drug_se)} raw -> {len(drug_se_dedup)} unique pairs")

dse_src = [drug_mapping[r] for r in drug_se_dedup['rxcui']]
dse_dst = [se_mapping[se] for se in drug_se_dedup['side_effect']]
dse_weights = drug_se_dedup['mean_prr'].values

data['drug', 'causes', 'side_effect'].edge_index = torch.tensor([dse_src, dse_dst], dtype=torch.long)
data['drug', 'causes', 'side_effect'].edge_attr = torch.tensor(dse_weights, dtype=torch.float).view(-1, 1)

print("\nDataset Construction Complete!")
print(data)

# Save the HeteroData
torch.save(data, os.path.join(out_dir, "hetero_graph_data.pt"))

# Save the mappings for inference
with open(os.path.join(out_dir, "node_mappings.json"), "w") as f:
    json.dump({
        "patient": patient_mapping,
        "drug": drug_mapping,
        "protein": protein_mapping,
        "side_effect": se_mapping
    }, f)

print(f"Saved PyG HeteroData and mappings to {out_dir}")
