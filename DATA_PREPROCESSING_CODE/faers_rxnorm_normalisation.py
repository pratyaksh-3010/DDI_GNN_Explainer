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

# -----------------------------
# Map FAERS drug names to RxCUI
# -----------------------------
def map_to_rxcui(drug_list):
    drugs = ast.literal_eval(drug_list)  # convert string to list
    out = []

    for d in drugs:
        key = norm(d)
        if key in rx_lookup:
            out.append(rx_lookup[key])

    return list(set(out))

faers["RXCUI_LIST"] = faers["DRUG_LIST"].apply(map_to_rxcui)

# -----------------------------
# Keep valid polypharmacy cases
# -----------------------------
faers = faers[faers["RXCUI_LIST"].apply(len) >= 2]

# -----------------------------
# Save output
# -----------------------------
faers[["caseid", "RXCUI_LIST", "SEVERITY"]].to_csv(
    "faers_rxnorm.csv",
    index=False
)

print("faers_rxnorm.csv saved successfully")
print("Total mapped cases:", len(faers))
