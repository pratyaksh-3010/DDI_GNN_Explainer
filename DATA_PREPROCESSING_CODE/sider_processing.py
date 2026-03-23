import pandas as pd
import re
import ast

# Load modified FAERS
faers = pd.read_csv("merged_faers.csv")

# -----------------------------
# Load RxNorm concepts (CORRECT columns)
# -----------------------------
rxnconso = pd.read_csv(
    r"RxNorm_full_prescribe_02022026\rrf\RXNCONSO.RRF",
    sep="|",
    header=None,
    dtype=str,
    usecols=[0, 1, 11, 12, 14]  # RXCUI, LAT, SAB, TTY, STR
)

rxnconso.columns = ["RXCUI", "LAT", "SAB", "TTY", "STR"]

# -----------------------------
# Keep English ingredient-level RxNorm terms
# -----------------------------
rx = rxnconso[
    (rxnconso["LAT"] == "ENG") &
    (rxnconso["SAB"] == "RXNORM") &
    (rxnconso["TTY"].isin(["IN", "PIN"]))  # Ingredient / Precise Ingredient
][["RXCUI", "STR"]].dropna()

# -----------------------------
# Normalize strings
# -----------------------------
def norm(s):
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\b(MG|ML|MCG|TABLET|CAPSULE|INJECTION|ORAL|IV)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

rx["STR_NORM"] = rx["STR"].apply(norm)

# Remove duplicates safely
rx = rx.drop_duplicates(subset=["STR_NORM"])

# Build lookup dictionary
rx_lookup = dict(zip(rx["STR_NORM"], rx["RXCUI"]))


#STEP7: ACTUALLY BEGINS



# Load side effects
sider = pd.read_csv(
    r"meddra_all_se.tsv\meddra_all_se.tsv",
    sep="\t",
    header=None,
    names=[
        "stitch_flat_id",
        "stitch_stereo_id",
        "umls_id",
        "meddra_type",
        "meddra_concept_id",
        "side_effect_name"
    ],
    dtype=str
)

# Load drug names
drug_names = pd.read_csv(
    r"drug_names.tsv",
    sep="\t",
    header=None,
    names=["stitch_flat_id", "drug_name"],
    dtype=str
)


sider = sider[sider["meddra_type"] == "PT"]



sider = sider.merge(drug_names, on="stitch_flat_id", how="left")




def norm(s):
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

sider["drug_norm"] = sider["drug_name"].apply(norm)



sider["RXCUI_LIST"] = sider["drug_norm"].map(rx_lookup)

sider = sider[sider["RXCUI_LIST"].notna()]


sider = sider.explode("RXCUI_LIST")



drug_to_se = (
    sider.groupby("RXCUI_LIST")["side_effect_name"]
    .apply(lambda x: list(set(x)))
    .to_dict()
)



#CHECKING COVERAGE

faers = pd.read_csv("faers_rxnorm.csv")


import ast

all_faers_rxcuis = set()
for row in faers["RXCUI_LIST"]:
    all_faers_rxcuis.update(ast.literal_eval(row))

mapped_sider_rxcuis = set(sider["RXCUI_LIST"])

print("FAERS drugs:", len(all_faers_rxcuis))
print("SIDER mapped drugs:", len(mapped_sider_rxcuis))
print("Overlap:", len(all_faers_rxcuis & mapped_sider_rxcuis))


mapping = pd.read_csv("rxcui_structure_map.csv", dtype=str)
rx_to_struct = dict(zip(mapping["rxcui"], mapping["structure_id"]))

structure_to_se = {}

for rxcui, effects in drug_to_se.items():
    if rxcui in rx_to_struct:
        struct_id = rx_to_struct[rxcui]
        if struct_id not in structure_to_se:
            structure_to_se[struct_id] = set()
        structure_to_se[struct_id].update(effects)

# Convert sets to lists
structure_to_se = {k: list(v) for k, v in structure_to_se.items()}


import ast
faers = pd.read_csv("faers_with_mechanism.csv")


def attach_sider_struct(struct_list):
    try:
        sids = ast.literal_eval(struct_list)
    except:
        return []
    
    effects = []
    for sid in sids:
        if sid in structure_to_se:
            effects.extend(structure_to_se[sid])
    
    return list(set(effects))

faers["KNOWN_SIDE_EFFECTS"] = faers["STRUCTURE_ID_LIST"].apply(attach_sider_struct)


faers.to_csv("faers_layer2_final.csv", index=False)

print("Layer 2 final dataset saved.")
