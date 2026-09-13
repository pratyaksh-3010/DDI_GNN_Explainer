import pandas as pd
import os

print("Parsing drugcentral.dump.11012023.sql for PUBCHEM_CID mappings...")
sql_file = r"drugcentral.dump.11012023.sql\drugcentral.dump.11012023.sql"

pubchem_to_struct = {}

with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if '\tPUBCHEM_CID\t' in line:
            parts = line.split('\t')
            # Format: id \t identifier \t id_type \t struct_id \t parent_match
            if len(parts) >= 4:
                pubchem_cid = parts[1].strip()
                struct_id = parts[3].strip()
                pubchem_to_struct[pubchem_cid] = struct_id

print(f"Extracted {len(pubchem_to_struct)} PubChem-to-Structure mappings.")

df_pubchem = pd.DataFrame(list(pubchem_to_struct.items()), columns=['pubchem_cid', 'structure_id'])

# Load existing rxcui to structure map
print("Loading rxcui_structure_map.csv...")
rx_map = pd.read_csv("rxcui_structure_map.csv", dtype=str)

# Merge
merged = df_pubchem.merge(rx_map, on="structure_id", how="inner")
print(f"Mapped {len(merged)} PubChem CIDs to RxCUIs via Structure ID.")

# Save mapping
os.makedirs("DATA_AGGREGATION_CODE", exist_ok=True)
merged.to_csv(r"DATA_AGGREGATION_CODE\stitch_to_rxcui_map.csv", index=False)
print("Saved crosswalk to DATA_AGGREGATION_CODE\\stitch_to_rxcui_map.csv")
