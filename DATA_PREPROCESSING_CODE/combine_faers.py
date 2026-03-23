import pandas as pd

# Read the CSV files
df4 = pd.read_csv(r"modifiedfaersq4.csv")
df3 = pd.read_csv(r"modifiedfaersq3.csv")
df2 = pd.read_csv(r"modifiedfaersq2.csv")
df1 = pd.read_csv(r"modifiedfaersq1.csv")

# Concatenate the DataFrames
merged_df = pd.concat([df1, df2,df3,df4], ignore_index=True)

# Save the result to a new CSV file
merged_df.to_csv('merged_faers.csv', index=False)   