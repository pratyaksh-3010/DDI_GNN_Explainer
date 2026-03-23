import pandas as pd

demo = pd.read_csv(r"faers_ascii_2023q4\ASCII\DEMO23Q4.txt", sep="$", dtype=str)
drug = pd.read_csv(r"faers_ascii_2023q4\ASCII\DRUG23Q4.txt", sep="$", dtype=str)
outc = pd.read_csv(r"faers_ascii_2023q4\ASCII\OUTC23Q4.txt", sep="$", dtype=str)
reac = pd.read_csv(r"faers_ascii_2023q4\ASCII\REAC23Q4.txt", sep="$", dtype=str)

drug = drug[drug["role_cod"].isin(["PS", "SS"])]
drug = drug[drug["drugname"].notna()]


drug_counts = drug.groupby("caseid")["drugname"].nunique()
poly_cases = drug_counts[drug_counts >= 2].index


demo = demo[demo["caseid"].isin(poly_cases)]
drug = drug[drug["caseid"].isin(poly_cases)]
outc = outc[outc["caseid"].isin(poly_cases)]
reac = reac[reac["caseid"].isin(poly_cases)]


severe_codes = {"DE", "LT", "HO", "DS"}

case_severity = (
    outc.groupby("caseid")["outc_cod"]
    .apply(lambda x: int(any(code in severe_codes for code in x)))
    .reset_index(name="SEVERITY")
)


case_drugs = (
    drug.groupby("caseid")["drugname"]
    .apply(list)
    .reset_index(name="DRUG_LIST")
)


faers_step1 = case_drugs.merge(case_severity, on="caseid", how="left")

faers_step1.to_csv("modifiedfaersq4.csv", index=False)

print("modifiedfaersq4.csv saved successfully")
print("Total cases:", len(faers_step1))
