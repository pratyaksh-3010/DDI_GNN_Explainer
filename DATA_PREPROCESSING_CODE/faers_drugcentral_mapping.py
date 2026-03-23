import pandas as pd
import ast

faers = pd.read_csv("faers_rxnorm.csv")
mapping = pd.read_csv("rxcui_structure_map.csv", dtype=str)

# Convert mapping to dictionary
rx_to_struct = dict(zip(mapping["rxcui"], mapping["structure_id"]))

#mapping
def map_structure_ids(rxcui_list):
    rxcuis = ast.literal_eval(rxcui_list)
    mapped = [rx_to_struct[r] for r in rxcuis if r in rx_to_struct]
    return list(set(mapped))

faers["STRUCTURE_ID_LIST"] = faers["RXCUI_LIST"].apply(map_structure_ids)


#Keep valid rows only

faers = faers[faers["STRUCTURE_ID_LIST"].apply(len) >= 2]

#Final

faers[["caseid", "STRUCTURE_ID_LIST", "SEVERITY"]].to_csv(
    "faers_drugcentral.csv",
    index=False
)

print("faers_drugcentral.csv saved")
print("Cases:", len(faers))
