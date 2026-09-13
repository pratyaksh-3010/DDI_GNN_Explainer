import pandas as pd
import ast
import os

print("Constructing Heterogeneous Graph Data Structures...")
out_dir = r"DATA_AGGREGATION_CODE\Graph_Data"
os.makedirs(out_dir, exist_ok=True)

# 1. Load Data
faers = pd.read_csv(r"DATA_AGGREGATION_CODE\faers_layer3_final.csv")
ppi = pd.read_csv(r"DATA_AGGREGATION_CODE\filtered_ppi.csv")
ddi = pd.read_csv(r"DATA_AGGREGATION_CODE\combined_polypharmacy_edges.csv")

patient_nodes = faers[['caseid', 'SEVERITY']].rename(columns={'SEVERITY': 'severity'}).drop_duplicates()

print("Extracting FAERS edges...")
patient_drug_edges = []
drug_protein_edges = []
drug_se_edges = []

print("Loading rxcui mapping for patient nodes...")
struct_map = pd.read_csv("rxcui_structure_map.csv", dtype=str)
struct_to_rxcui = dict(zip(struct_map['structure_id'], struct_map['rxcui']))

for _, row in faers.iterrows():
    caseid = row['caseid']
    try:
        structs = ast.literal_eval(row['STRUCTURE_ID_LIST'])
        rxcuis = [struct_to_rxcui[s] for s in structs if s in struct_to_rxcui]
    except:
        rxcuis = []
    
    try:
        targets = ast.literal_eval(row['COMBINED_TARGET_GENES'])
    except:
        targets = []
        
    try:
        mono_se = ast.literal_eval(row['COMBINED_MONO_SIDE_EFFECTS'])
    except:
        mono_se = []
        
    for r in rxcuis:
        r_str = str(r)
        patient_drug_edges.append({'caseid': caseid, 'rxcui': r_str})
        for t in targets:
            drug_protein_edges.append({'rxcui': r_str, 'protein': t})
        for se in mono_se:
            drug_se_edges.append({'rxcui': r_str, 'side_effect': se.lower()})

if not patient_drug_edges:
    print("Warning: patient_drug_edges is empty. Structs might not be mapping.")
    df_pd = pd.DataFrame(columns=['caseid', 'rxcui'])
else:
    df_pd = pd.DataFrame(patient_drug_edges).drop_duplicates()
    
if not drug_protein_edges:
    df_dp = pd.DataFrame(columns=['rxcui', 'protein'])
else:
    df_dp = pd.DataFrame(drug_protein_edges).drop_duplicates()
    
if not drug_se_edges:
    df_dse = pd.DataFrame(columns=['rxcui', 'side_effect'])
else:
    df_dse = pd.DataFrame(drug_se_edges).drop_duplicates()

print("Extracting Node Sets using Vectorization...")
# Drug nodes
faers_drugs = pd.Series(df_pd['rxcui'].unique())
ddi_drugs = pd.concat([ddi['drug1_rxcui'], ddi['drug2_rxcui']])
drug_nodes = pd.concat([faers_drugs, ddi_drugs]).dropna().astype(str).unique()

# Protein nodes
faers_proteins = pd.Series(df_dp['protein'].unique())
ppi_proteins = pd.concat([ppi['symbol_1'], ppi['symbol_2']])
protein_nodes = pd.concat([faers_proteins, ppi_proteins]).dropna().astype(str).unique()

# Side Effect nodes
faers_se = pd.Series(df_dse['side_effect'].unique())
ddi_se = ddi['side_effect'].dropna().astype(str).str.lower()
se_nodes = pd.concat([faers_se, ddi_se]).unique()

print("Saving Node Lists...")
patient_nodes.to_csv(os.path.join(out_dir, "nodes_patient.csv"), index=False)
pd.DataFrame({"rxcui": drug_nodes}).to_csv(os.path.join(out_dir, "nodes_drug.csv"), index=False)
pd.DataFrame({"protein": protein_nodes}).to_csv(os.path.join(out_dir, "nodes_protein.csv"), index=False)
pd.DataFrame({"side_effect": se_nodes}).to_csv(os.path.join(out_dir, "nodes_side_effect.csv"), index=False)

print("Saving Edge Lists...")
df_pd.to_csv(os.path.join(out_dir, "edges_patient_drug.csv"), index=False)
df_dp.to_csv(os.path.join(out_dir, "edges_drug_protein.csv"), index=False)
ppi.to_csv(os.path.join(out_dir, "edges_protein_protein.csv"), index=False)
ddi.to_csv(os.path.join(out_dir, "edges_drug_drug.csv"), index=False)
df_dse.to_csv(os.path.join(out_dir, "edges_drug_side_effect.csv"), index=False)

print("Heterogeneous Graph Construction Complete!")
print(f"Total Patients: {len(patient_nodes)}")
print(f"Total Drugs: {len(drug_nodes)}")
print(f"Total Proteins: {len(protein_nodes)}")
print(f"Total Side Effects: {len(se_nodes)}")
