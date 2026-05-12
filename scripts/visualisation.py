import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUT_DIR  = os.path.join(BASE_DIR, '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_data_normalized.csv'))

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

seuil = np.percentile(similarite[similarite > 0], 95)

G = nx.Graph()
G.add_nodes_from(range(n))
for i in range(n):
    for j in range(i+1, n):
        if similarite[i][j] >= seuil:
            G.add_edge(i, j, weight=float(round(similarite[i][j], 6)))

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(G, pos, node_size=15, alpha=0.7)
nx.draw_networkx_edges(G, pos, alpha=0.05, width=0.5)

plt.title("Graphe de similarité — 1879 patients", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'visualisation.png'), dpi=150)