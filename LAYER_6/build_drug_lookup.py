"""
Build structure_id → drug_name lookup using the RxNorm mapping chain.

Fast approach: use rxcui_structure_map + RxNorm RXNCONSO directly,
  no need to iterate over the large merged_faers file.
"""

import pandas as pd
import re

# --------------------------------------------------
# 1. Load  rxcui → structure_id  mapping
# --------------------------------------------------
rx_struct = pd.read_csv("rxcui_structure_map.csv", dtype=str)
rx_to_struct = dict(zip(rx_struct["rxcui"], rx_struct["structure_id"]))

# --------------------------------------------------
# 2. Load RxNorm for canonical drug names
# --------------------------------------------------
print("Loading RxNorm...")
rxnconso = pd.read_csv(
    r"RxNorm_full_prescribe_02022026\rrf\RXNCONSO.RRF",
    sep="|",
    header=None,
    dtype=str,
    usecols=[0, 1, 11, 12, 14]
)
rxnconso.columns = ["RXCUI", "LAT", "SAB", "TTY", "STR"]

rx = rxnconso[
    (rxnconso["LAT"] == "ENG") &
    (rxnconso["SAB"] == "RXNORM") &
    (rxnconso["TTY"].isin(["IN", "PIN"]))
][["RXCUI", "STR"]].dropna().drop_duplicates(subset=["RXCUI"])

# rxcui → canonical name
rxcui_to_name = dict(zip(rx["RXCUI"], rx["STR"]))

# --------------------------------------------------
# 3. Map structure_id → drug name via rxcui
# --------------------------------------------------
print("Building lookup...")
records = []

for rxcui, struct_id in rx_to_struct.items():
    if rxcui in rxcui_to_name:
        records.append({
            "structure_id": struct_id,
            "drug_name": rxcui_to_name[rxcui]
        })

lookup_df = pd.DataFrame(records)

# If multiple names per structure_id, keep the first (they're usually the same)
lookup_df = lookup_df.drop_duplicates(subset=["structure_id"], keep="first")

lookup_df.to_csv("structure_name_lookup.csv", index=False)

print(f"Lookup file created: {len(lookup_df)} drugs mapped")
print(lookup_df.head(10))