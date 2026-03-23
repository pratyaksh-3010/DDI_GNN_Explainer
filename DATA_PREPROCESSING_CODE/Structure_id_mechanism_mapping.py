import pandas as pd
import ast

faers = pd.read_csv("faers_with_atc.csv")
mech = pd.read_csv("structure_mechanism_map.csv", dtype=str)


struct_to_targets = (
    mech.groupby("struct_id")["gene"]
    .apply(lambda x: list(set(x.dropna())))
    .to_dict()
)

struct_to_target_class = (
    mech.groupby("struct_id")["target_class"]
    .apply(lambda x: list(set(x.dropna())))
    .to_dict()
)


def map_mechanisms(struct_list):
    struct_ids = ast.literal_eval(struct_list)
    
    genes = []
    classes = []
    
    for sid in struct_ids:
        if sid in struct_to_targets:
            genes.extend(struct_to_targets[sid])
        if sid in struct_to_target_class:
            classes.extend(struct_to_target_class[sid])
    
    return list(set(genes)), list(set(classes))

faers[["TARGET_GENES", "TARGET_CLASSES"]] = faers["STRUCTURE_ID_LIST"].apply(
    lambda x: pd.Series(map_mechanisms(x))
)


faers.to_csv("faers_with_mechanism.csv", index=False)

print("faers_with_mechanism.csv saved")
print("Total cases:", len(faers))
