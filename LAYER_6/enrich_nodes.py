"""
Enrich layer3_nodes.csv with human-readable drug names from the lookup table.
Also assigns node_index for frontend use.
"""
import pandas as pd

nodes = pd.read_csv("layer3_nodes.csv")
lookup = pd.read_csv("structure_name_lookup.csv", dtype=str)

# Ensure structure_id types match
nodes["structure_id"] = nodes["structure_id"].astype(str)
lookup["structure_id"] = lookup["structure_id"].astype(str)

# Merge drug names
nodes = nodes.merge(lookup, on="structure_id", how="left")

# Assign node_index (must match the row position used in edges)
nodes["node_index"] = nodes.index

# Fill missing drug names with structure_id
nodes["drug_name"] = nodes["drug_name"].fillna(
    nodes["structure_id"].apply(lambda x: f"Drug_{x}")
)

nodes.to_csv("layer3_nodes_enriched.csv", index=False)

print(f"Nodes enriched: {len(nodes)} drugs")
print(f"Named drugs: {nodes['drug_name'].notna().sum()}")
print(f"Sample:")
print(nodes[["structure_id", "drug_name", "node_index"]].head(10))