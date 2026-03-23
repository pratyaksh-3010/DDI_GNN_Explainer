import pandas as pd
import networkx as nx

nodes = pd.read_csv("STREAMLIT_FRONTEND/DATA/layer3_nodes_enriched.csv")
edges = pd.read_csv("STREAMLIT_FRONTEND/DATA/layer3_edges_weighted.csv")

id_to_name = {
    row["node_index"]: row["drug_name"]
    for _, row in nodes.iterrows()
    if isinstance(row["drug_name"], str)
}

# Build graph
G = nx.Graph()

for _, row in edges.iterrows():

    s = row["source"]
    t = row["target"]

    if s in id_to_name and t in id_to_name:
        G.add_edge(id_to_name[s], id_to_name[t])

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Find polypharmacy combinations
poly_combos = []

for clique in nx.find_cliques(G):

    if len(clique) >= 3:   # polypharmacy = 3+ drugs
        poly_combos.append(clique)

print("Total polypharmacy combinations:", len(poly_combos))

for c in poly_combos[:10]:
    print(c)