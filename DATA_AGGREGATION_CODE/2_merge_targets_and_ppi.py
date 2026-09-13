import pandas as pd
import ast
import re
import os
import mygene

print("Loading vocabulary crosswalk...")
crosswalk = pd.read_csv(r"DATA_AGGREGATION_CODE\stitch_to_rxcui_map.csv", dtype=str)
pubchem_to_rxcui = dict(zip(crosswalk['pubchem_cid'], crosswalk['rxcui']))

def clean_stitch_id(stitch_id):
    num_str = re.sub(r'[^0-9]', '', str(stitch_id))
    return str(int(num_str)) if num_str else None

print("Processing bio-decagon-targets.csv...")
targets = pd.read_csv(r"bio-decagon-targets\bio-decagon-targets.csv", dtype=str)
targets['pubchem_cid'] = targets['STITCH'].apply(clean_stitch_id)
targets['rxcui'] = targets['pubchem_cid'].map(pubchem_to_rxcui)

print("Processing bio-decagon-ppi.csv...")
ppi = pd.read_csv(r"bio-decagon-ppi\bio-decagon-ppi.csv", dtype=str)

all_entrez = list(set(targets['Gene'].dropna()).union(set(ppi['Gene 1'].dropna())).union(set(ppi['Gene 2'].dropna())))

print(f"Querying mygene for {len(all_entrez)} Entrez IDs...")
mg = mygene.MyGeneInfo()
try:
    res = mg.querymany(all_entrez, scopes='entrezgene', fields='symbol', species='human', as_dataframe=True)
    entrez_to_symbol = res['symbol'].dropna().to_dict()
    print(f"Successfully mapped {len(entrez_to_symbol)} Entrez IDs to Symbols.")
except Exception as e:
    print(f"Error querying mygene: {e}")
    entrez_to_symbol = {}

targets['symbol'] = targets['Gene'].map(entrez_to_symbol)
mapped_targets = targets.dropna(subset=['rxcui', 'symbol'])
print(f"Mapped {len(mapped_targets)} out of {len(targets)} Decagon targets to RxCUI and Symbol.")

decagon_drug_targets = (
    mapped_targets.groupby("rxcui")["symbol"]
    .apply(lambda x: list(set(x)))
    .to_dict()
)

print("Loading faers_with_mechanism.csv...")
faers = pd.read_csv("faers_with_mechanism.csv")

def combine_targets(row):
    try:
        rxcuis = ast.literal_eval(row['RXCUI_LIST'])
    except:
        rxcuis = []
        
    try:
        existing_targets = ast.literal_eval(row['TARGET_GENES'])
    except:
        existing_targets = []
        
    all_targets = set(existing_targets)
    
    for r in rxcuis:
        r_str = str(r)
        if r_str in decagon_drug_targets:
            all_targets.update(decagon_drug_targets[r_str])
            
    return list(all_targets)

faers['COMBINED_TARGET_GENES'] = faers.apply(combine_targets, axis=1)

all_genes_in_cases = set()
for g_list in faers['COMBINED_TARGET_GENES']:
    all_genes_in_cases.update(g_list)

print(f"Total unique target genes across all cases: {len(all_genes_in_cases)}")

ppi['symbol_1'] = ppi['Gene 1'].map(entrez_to_symbol)
ppi['symbol_2'] = ppi['Gene 2'].map(entrez_to_symbol)

ppi = ppi.dropna(subset=['symbol_1', 'symbol_2'])

ppi_filtered = ppi[
    (ppi['symbol_1'].isin(all_genes_in_cases)) & 
    (ppi['symbol_2'].isin(all_genes_in_cases))
]
print(f"Filtered PPI network from {len(ppi)} edges to {len(ppi_filtered)} edges.")

faers.to_csv(r"DATA_AGGREGATION_CODE\faers_with_combined_targets.csv", index=False)
ppi_filtered.to_csv(r"DATA_AGGREGATION_CODE\filtered_ppi.csv", index=False)
print("Saved faers_with_combined_targets.csv and filtered_ppi.csv")
