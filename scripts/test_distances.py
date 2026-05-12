import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as Patch
from scipy.spatial.distance import cdist
import networkx as nx
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUT_DIR  = os.path.join(BASE_DIR, '..', 'outputs')
FIG_DIR  = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

df_norm = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_data_normalized.csv'))
df_raw  = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_data_cleaned.csv'))

COLS_PROJET = [
    'Age', 'BMI', 'FastingBloodSugar', 'HbA1c',
    'SystolicBP', 'CholesterolTotal', 'SleepQuality'
]
POIDS = np.array([0.10, 0.15, 0.25, 0.25, 0.15, 0.05, 0.05])

X = df_norm.values



N_TEST = 50
X_test      = X[:N_TEST]
labels_test = df_raw['Diagnosis'].values[:N_TEST]

def distance_ponderee_euclidienne(X, poids):
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            diff = X[i] - X[j]
            d = np.sqrt(np.sum(poids * diff**2))
            D[i][j] = D[j][i] = d
    return 1 / (1 + D)

def distance_manhattan_ponderee(X, poids):
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.sum(poids * np.abs(X[i] - X[j]))
            D[i][j] = D[j][i] = d
    return 1 / (1 + D)

def similarite_cosine(X):
    return 1 - cdist(X, X, metric='cosine')

sim_euc = distance_ponderee_euclidienne(X_test, POIDS)
sim_man = distance_manhattan_ponderee(X_test, POIDS)

X_weighted = X_test * np.sqrt(POIDS)
sim_cos = np.clip(similarite_cosine(X_weighted), 0, 1)
np.fill_diagonal(sim_cos, 1.0)

def get_vals(S):
    return S[np.triu_indices(len(S), k=1)]

vals_euc = get_vals(sim_euc)
vals_man = get_vals(sim_man)
vals_cos = get_vals(sim_cos)

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

noms     = ["Euclidienne pondérée", "Manhattan pondérée", "Cosine pondérée"]
valeurs  = [vals_euc, vals_man, vals_cos]
matrices = [sim_euc, sim_man, sim_cos]
couleurs = ['#2ecc71', '#3498db', '#e67e22']

for k, (nom, vals, coul) in enumerate(zip(noms, valeurs, couleurs)):
    ax = fig.add_subplot(gs[0, k])
    ax.hist(vals, bins=30, color=coul, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axvline(vals.mean(), color='black', linestyle='--', linewidth=1.5, label=f'μ={vals.mean():.3f}')
    ax.set_title(nom, fontsize=10, fontweight='bold')
    ax.set_xlabel("Similarité", fontsize=9)
    ax.set_ylabel("Fréquence", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

for k, (nom, mat, cmap) in enumerate(zip(noms, matrices, ['Greens', 'Blues', 'Oranges'])):
    ax = fig.add_subplot(gs[1, k])
    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax.set_title(f"Matrice {nom}\n(50 patients)", fontsize=9, fontweight='bold')
    ax.set_xlabel("Patient", fontsize=8)
    ax.set_ylabel("Patient", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Comparaison des métriques de distance — Similarité entre patients",
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(os.path.join(FIG_DIR, 'comparaison_distances.png'), dpi=90, bbox_inches='tight')
plt.close()

r_em = np.corrcoef(vals_euc, vals_man)[0, 1]
r_ec = np.corrcoef(vals_euc, vals_cos)[0, 1]
r_mc = np.corrcoef(vals_man, vals_cos)[0, 1]

N_PROTO      = 100
X_proto      = X[:N_PROTO]
labels_proto = df_raw['Diagnosis'].values[:N_PROTO]

sim_proto = distance_ponderee_euclidienne(X_proto, POIDS)
seuil_p75 = np.percentile(sim_proto[sim_proto > 0], 75)

G_proto = nx.Graph()
G_proto.add_nodes_from(range(N_PROTO))
for i in range(N_PROTO):
    for j in range(i+1, N_PROTO):
        if sim_proto[i][j] >= seuil_p75:
            G_proto.add_edge(i, j, weight=float(sim_proto[i][j]))

fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G_proto, seed=42, k=0.5)
colors = ['#e74c3c' if labels_proto[n] == 1 else '#3498db' for n in G_proto.nodes()]
nx.draw_networkx_nodes(G_proto, pos, node_color=colors, node_size=60, alpha=0.85, ax=ax)
nx.draw_networkx_edges(G_proto, pos, alpha=0.12, ax=ax)

from matplotlib.patches import Patch as MPatch
legend = [MPatch(facecolor='#e74c3c', label='Diabétique'),
          MPatch(facecolor='#3498db', label='Non-Diabétique')]
ax.legend(handles=legend, loc='upper left', fontsize=10)
ax.set_title("Graphe prototype — 100 patients (seuil P75, distance Euclidienne pondérée)",
             fontsize=11, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'graphe_prototype.png'), dpi=90, bbox_inches='tight')
plt.close()