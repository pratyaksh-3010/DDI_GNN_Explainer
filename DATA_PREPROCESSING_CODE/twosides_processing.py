import pandas as pd

twosides = pd.read_csv("TWOSIDES.csv", dtype=str)


twosides.rename(columns={
    "drug_1_rxnorn_id": "drug1_rxcui",
    "drug_2_rxnorm_id": "drug2_rxcui",
    "condition_concept_name": "side_effect"
}, inplace=True)
#remove unnecessary columns 

twosides = twosides[[
    "drug1_rxcui",
    "drug2_rxcui",
    "side_effect",
    "PRR"
]]


twosides["PRR"] = pd.to_numeric(twosides["PRR"], errors="coerce")
twosides = twosides.dropna(subset=["PRR"])

import ast

faers = pd.read_csv("faers_rxnorm.csv")

all_rxcuis = set()

for row in faers["RXCUI_LIST"]:
    rxs = ast.literal_eval(row)
    all_rxcuis.update(rxs)

twosides_filtered = twosides[
    (twosides["drug1_rxcui"].isin(all_rxcuis)) &
    (twosides["drug2_rxcui"].isin(all_rxcuis))
]



twosides_filtered = twosides_filtered[twosides_filtered["PRR"] > 1]


twosides_filtered.to_csv("twosides_filtered.csv", index=False)

print("twosides_filtered.csv saved")
print("Remaining interactions:", len(twosides_filtered))
