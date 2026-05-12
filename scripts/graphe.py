import pandas as pd
import numpy as np
import networkx as nx

df = pd.read_csv("../data/diabetes_data_normalized.csv")

poids = np.array([0.10, 0.15, 0.25, 0.25, 0.15, 0.05, 0.05])
X = df.values
n = len(X)

similarite = np.zeros((n, n))

for i in range(n):
    for j in range(i+1, n):
        diff = X[i] - X[j]
        distance = np.sqrt(np.sum(poids * diff**2))
        sim = 1 / (1 + distance)
        similarite[i][j] = sim
        similarite[j][i] = sim



seuil = np.percentile(similarite[similarite > 0], 75)


G = nx.Graph()
G.add_nodes_from(range(n))
for i in range(n):
    for j in range(i+1, n):
        if similarite[i][j] >= seuil:
            G.add_edge(i, j, weight=float(round(similarite[i][j], 6)))


nx.write_gml(G, "../outputs/graphe.gml")
