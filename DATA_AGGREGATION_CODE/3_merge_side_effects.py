import pandas as pd
import ast
import re

print("Loading vocabulary crosswalk...")
crosswalk = pd.read_csv(r"DATA_AGGREGATION_CODE\stitch_to_rxcui_map.csv", dtype=str)
pubchem_to_rxcui = dict(zip(crosswalk['pubchem_cid'], crosswalk['rxcui']))

def clean_stitch_id(stitch_id):
    num_str = re.sub(r'[^0-9]', '', str(stitch_id))
    return str(int(num_str)) if num_str else None

print("Processing bio-decagon-combo.csv...")
combo = pd.read_csv(r"bio-decagon-combo\bio-decagon-combo.csv", dtype=str)
combo['pubchem_1'] = combo['STITCH 1'].apply(clean_stitch_id)
combo['pubchem_2'] = combo['STITCH 2'].apply(clean_stitch_id)

combo['rxcui_1'] = combo['pubchem_1'].map(pubchem_to_rxcui)
combo['rxcui_2'] = combo['pubchem_2'].map(pubchem_to_rxcui)

mapped_combo = combo.dropna(subset=['rxcui_1', 'rxcui_2'])
print(f"Mapped {len(mapped_combo)} out of {len(combo)} Decagon combinations to RxCUI.")

# Prepare for merging with TWOSIDES
mapped_combo = mapped_combo.rename(columns={
    'rxcui_1': 'drug1_rxcui',
    'rxcui_2': 'drug2_rxcui',
    'Side Effect Name': 'side_effect'
})
# TWOSIDES format: drug1_rxcui, drug2_rxcui, side_effect, PRR
decagon_ddi = mapped_combo[['drug1_rxcui', 'drug2_rxcui', 'side_effect']].copy()
decagon_ddi['PRR'] = 1.0 # Default weight for Decagon-exclusive edges

print("Loading twosides_filtered.csv...")
twosides = pd.read_csv("twosides_filtered.csv", dtype=str)
twosides['PRR'] = twosides['PRR'].astype(float)

print("Unioning Polypharmacy Edges...")
# Sort RxCUIs so we don't treat A->B and B->A as different combinations
def sort_pair(row):
    d1, d2 = str(row['drug1_rxcui']), str(row['drug2_rxcui'])
    if d1 > d2:
        return d2, d1
    return d1, d2

for df in [decagon_ddi, twosides]:
    sorted_pairs = df.apply(sort_pair, axis=1)
    df['drug1_rxcui'] = [p[0] for p in sorted_pairs]
    df['drug2_rxcui'] = [p[1] for p in sorted_pairs]
    df['side_effect'] = df['side_effect'].str.lower()

# Combine and keep max PRR for duplicates
combined_ddi = pd.concat([twosides, decagon_ddi])
# Group by drug pair and side effect, take max PRR
combined_ddi = combined_ddi.groupby(['drug1_rxcui', 'drug2_rxcui', 'side_effect'])['PRR'].max().reset_index()

print(f"Total Combined DDI Edges: {len(combined_ddi)}")

print("Processing bio-decagon-mono.csv...")
mono = pd.read_csv(r"bio-decagon-mono\bio-decagon-mono.csv", dtype=str)
mono['pubchem_cid'] = mono['STITCH'].apply(clean_stitch_id)
mono['rxcui'] = mono['pubchem_cid'].map(pubchem_to_rxcui)
mapped_mono = mono.dropna(subset=['rxcui'])

mono_dict = (
    mapped_mono.groupby("rxcui")["Side Effect Name"]
    .apply(lambda x: list(set(x)))
    .to_dict()
)

print("Loading faers_with_combined_targets.csv...")
faers = pd.read_csv(r"DATA_AGGREGATION_CODE\faers_with_combined_targets.csv")

def combine_mono_se(row):
    try:
        rxcuis = ast.literal_eval(row['RXCUI_LIST'])
    except:
        rxcuis = []
        
    try:
        existing_se = ast.literal_eval(row['KNOWN_SIDE_EFFECTS'])
    except:
        existing_se = []
        
    all_se = set(existing_se)
    
    for r in rxcuis:
        r_str = str(r)
        if r_str in mono_dict:
            all_se.update(mono_dict[r_str])
            
    return list(all_se)

faers['COMBINED_MONO_SIDE_EFFECTS'] = faers.apply(combine_mono_se, axis=1)

faers.to_csv(r"DATA_AGGREGATION_CODE\faers_layer3_final.csv", index=False)
combined_ddi.to_csv(r"DATA_AGGREGATION_CODE\combined_polypharmacy_edges.csv", index=False)
print("Saved faers_layer3_final.csv and combined_polypharmacy_edges.csv")
