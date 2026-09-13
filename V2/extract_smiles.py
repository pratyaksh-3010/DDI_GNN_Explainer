import pandas as pd
import requests
import time
import os

print("Fetching SMILES from PubChem API...")

# 1. Load drug nodes
rxcui_df = pd.read_csv(r"V2\Graph_Data\nodes_drug.csv", dtype=str)
rxcui_set = set(rxcui_df['rxcui'].tolist())

# 2. Map RxCUI to PubChem CID
crosswalk = pd.read_csv(r"DATA_AGGREGATION_CODE\stitch_to_rxcui_map.csv", dtype=str)
rxcui_to_pubchem = dict(zip(crosswalk['rxcui'], crosswalk['pubchem_cid']))

# Some rxcuis might map to multiple pubchems, but we just need one valid smiles
pubchem_cids = []
rxcui_for_cid = []
for rxcui in rxcui_set:
    if rxcui in rxcui_to_pubchem:
        pubchem_cids.append(rxcui_to_pubchem[rxcui])
        rxcui_for_cid.append(rxcui)

print(f"Mapped {len(pubchem_cids)} RxCUIs to PubChem CIDs.")

# 3. Fetch SMILES in batches (PubChem PUG REST API supports up to 100 per request)
# URL format: https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/1,2,3/property/CanonicalSMILES/CSV
batch_size = 100
smiles_data = []

for i in range(0, len(pubchem_cids), batch_size):
    batch = pubchem_cids[i:i+batch_size]
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{','.join(batch)}/property/CanonicalSMILES/CSV"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            if len(lines) > 1:
                # Skip header
                for line in lines[1:]:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        cid = parts[0].strip('"')
                        smiles = parts[1].strip('"')
                        smiles_data.append({"pubchem_cid": cid, "smiles": smiles})
    except Exception as e:
        print(f"Error fetching batch {i}: {e}")
    time.sleep(0.3) # Rate limiting

smiles_df = pd.DataFrame(smiles_data)
# Merge back to RxCUI
final_df = pd.DataFrame({"rxcui": rxcui_for_cid, "pubchem_cid": pubchem_cids})
final_df = final_df.merge(smiles_df, on="pubchem_cid", how="inner")
final_df = final_df[['rxcui', 'smiles']].drop_duplicates(subset=['rxcui'])

final_df.to_csv(r"V2\Graph_Data\drug_smiles.csv", index=False)
print(f"Successfully fetched {len(final_df)} SMILES strings and saved to V2\Graph_Data\drug_smiles.csv")
