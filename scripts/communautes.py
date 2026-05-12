import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm

warnings.filterwarnings('ignore')
os.makedirs("../outputs", exist_ok=True)

# Données
df_raw  = pd.read_csv("../data/diabetes_data_cleaned.csv")
df_norm = pd.read_csv("../data/diabetes_data_normalized.csv")

COLS  = ['Age', 'BMI', 'FastingBloodSugar', 'HbA1c', 'SystolicBP', 'CholesterolTotal', 'SleepQuality']
POIDS = np.array([0.10, 0.15, 0.25, 0.25, 0.15, 0.05, 0.05])

X      = df_norm[COLS].values
labels = df_raw['Diagnosis'].values
n      = len(X)

# Matrice de similarité (distance euclidienne pondérée : sim = 1 / (1 + d))
matrix_path = "../outputs/similarite_matrix.npy"

if os.path.exists(matrix_path):
    similarite = np.load(matrix_path)
else:
    X_w        = X * np.sqrt(POIDS)
    similarite = np.zeros((n, n), dtype=np.float32)
    BLOC       = 200
    for i in range(0, n, BLOC):
        for j in range(i, n, BLOC):
            diff = X_w[i:i+BLOC, None, :] - X_w[None, j:j+BLOC, :]
            s    = 1.0 / (1.0 + np.sqrt((diff**2).sum(axis=2)))
            similarite[i:i+BLOC, j:j+BLOC] = s
            if i != j:
                similarite[j:j+BLOC, i:i+BLOC] = s.T
    np.save(matrix_path, similarite)

# Graphe de similarité (seuil P75)
seuil = float(np.percentile(similarite[np.triu_indices(n, k=1)], 85))
rows, cols_ = np.where(similarite >= seuil)
mask        = rows < cols_

G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from(
    [(int(r), int(c), {'weight': float(similarite[r, c])})
     for r, c in zip(rows[mask], cols_[mask])]
)

# Détection de communautés — Louvain (résolution=1.0, seed=42)
communities = nx_comm.louvain_communities(G, seed=42, resolution=1.05)
Q           = nx_comm.modularity(G, communities)

partition = {node: comm_id
             for comm_id, comm_set in enumerate(communities)
             for node in comm_set}

print(f"{len(communities)} communautés détectées — Modularité Q = {Q:.4f}")

# Profils par communauté
df_raw['communaute'] = [partition.get(i, -1) for i in range(n)]

profils = []
for comm_id, comm_set in enumerate(communities):
    if len(comm_set) < 20:
        continue
    patients = df_raw.iloc[list(comm_set)]
    taux     = patients['Diagnosis'].mean()
    fbs      = patients['FastingBloodSugar'].mean()

    if taux >= 0.7:
        profil = "Diabétique sévère"
    elif taux >= 0.4:
        profil = "Risque modéré"
    elif taux >= 0.2:
        profil = "Hyperglycémie isolée"
    else:
        profil = "Faible prévalence diabétique"

    row = {
        'communaute':     comm_id,
        'taille':         len(comm_set),
        'taux_diabete_%': round(taux * 100, 1),
        'profil':         profil,
    }

    
    for col in COLS:
        row[f'moy_{col}'] = round(patients[col].mean(), 3)
        row[f'std_{col}'] = round(patients[col].std(),  3)

    profils.append(row)

profils_df = pd.DataFrame(profils).sort_values('taux_diabete_%', ascending=False)

# Sauvegarde
profils_df.to_csv("../outputs/profils.csv", index=False, encoding='utf-8')

pd.DataFrame({
    'patient_idx': range(n),
    'communaute':  [partition.get(i, -1) for i in range(n)],
    'diagnosis':   labels,
}).to_csv("../outputs/partition.csv", index=False, encoding='utf-8')

print("Fichiers sauvegardés : outputs/profils.csv · outputs/partition.csv")