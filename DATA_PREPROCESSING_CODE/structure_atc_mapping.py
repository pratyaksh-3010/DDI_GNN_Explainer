import pandas as pd
import ast

faers = pd.read_csv("faers_drugcentral.csv")
atc_map = pd.read_csv("structure_atc_map.csv", dtype=str)

# Build structure_id → ATC list mapping
struct_to_atc = (
    atc_map.groupby("structure_id")["atc_code"]
    .apply(list)
    .to_dict()
)
#attach atc to each case

def map_atc(struct_list):
    struct_ids = ast.literal_eval(struct_list)
    atcs = []
    for sid in struct_ids:
        if sid in struct_to_atc:
            atcs.extend(struct_to_atc[sid])
    return list(set(atcs))

faers["ATC_CODES"] = faers["STRUCTURE_ID_LIST"].apply(map_atc)

#removed number kept only letter for neural ,musculoskeleta etc

faers["ATC_LEVEL1"] = faers["ATC_CODES"].apply(
    lambda codes: list(set([c[0] for c in codes])) if isinstance(codes, list) else []
)


#final

faers.to_csv("faers_with_atc.csv", index=False)

print("faers_with_atc.csv saved")
print("Total cases:", len(faers))
