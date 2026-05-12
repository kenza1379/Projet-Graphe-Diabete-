import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random

random.seed(42)
np.random.seed(42)

PATH_GML       = "../outputs/graphe.gml"
PATH_PARTITION = "../outputs/partition.csv"
PATH_OUT_1     = "../outputs/figures/graphe_sous_ensemble.png"
PATH_OUT_2     = "../outputs/figures/graphe_complet_p95.png"

PALETTE = {
    3: "#E24B4A",
    1: "#E09020",
    0: "#185FA5",
    2: "#0F6E56",
}
LABELS = {
    3: "Comm 3 — Diabétique Sévère (94.5%)",
    1: "Comm 1 — Risque Modéré (24.7%)",
    0: "Comm 0 — Hyperglycémie isolée (21.6%)",
    2: "Comm 2 — Profil Sain (7.9%)",
}

G = nx.read_gml(PATH_GML)

df_part = pd.read_csv(PATH_PARTITION)
partition = dict(zip(df_part["patient_idx"], df_part["communaute"]))

nodes_gml = list(G.nodes())
if isinstance(nodes_gml[0], str):
    partition_mapped = {str(k): v for k, v in partition.items()}
else:
    partition_mapped = {int(k): v for k, v in partition.items()}

def node_color(n):
    comm = partition_mapped.get(n, -1)
    return PALETTE.get(comm, "#888888")

comms = df_part["communaute"].unique()
sample_nodes = []
total_sample = 400
for c in comms:
    members = df_part[df_part["communaute"] == c]["patient_idx"].tolist()
    n_pick = max(1, round(total_sample * len(members) / len(df_part)))
    picked = random.sample(members, min(n_pick, len(members)))
    sample_nodes.extend(picked)

if isinstance(nodes_gml[0], str):
    sample_nodes = [str(n) for n in sample_nodes]
else:
    sample_nodes = [int(n) for n in sample_nodes]

sample_nodes = [n for n in sample_nodes if n in G]
G_sub = G.subgraph(sample_nodes).copy()

pos_sub = nx.spring_layout(G_sub, seed=42, k=0.4, iterations=60)

colors_sub  = [node_color(n) for n in G_sub.nodes()]
edge_colors = []
for u, v in G_sub.edges():
    cu = partition_mapped.get(u, -1)
    cv = partition_mapped.get(v, -1)
    if cu == cv:
        edge_colors.append(PALETTE.get(cu, "#aaaaaa") + "55")
    else:
        edge_colors.append("#cccccc44")

fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

nx.draw_networkx_edges(G_sub, pos_sub, ax=ax, edge_color=edge_colors, width=0.3, alpha=None)
nx.draw_networkx_nodes(G_sub, pos_sub, ax=ax, node_color=colors_sub, node_size=35, linewidths=0.3, edgecolors="#ffffff33")

handles = [mpatches.Patch(color=PALETTE[c], label=LABELS[c]) for c in [3, 1, 0, 2]]
ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.85, facecolor="#1a1f2e", edgecolor="#444", labelcolor="white")
ax.set_title(
    f"Graphe de similarité — {G_sub.number_of_nodes()} patients (échantillon stratifié)\n"
    f"Arêtes : {G_sub.number_of_edges():,}  |  Seuil P75  |  Couleur = communauté Louvain (Q = 0.307)",
    color="white", fontsize=11, pad=14
)
ax.axis("off")
plt.tight_layout()
plt.savefig(PATH_OUT_1, dpi=90, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

all_weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]

if len(set(all_weights)) == 1:
    all_edges = list(G.edges())
    n_keep = max(1, int(0.05 * len(all_edges)))
    kept_edges = random.sample(all_edges, n_keep)
    G_p95 = nx.Graph()
    G_p95.add_nodes_from(G.nodes())
    G_p95.add_edges_from(kept_edges)
else:
    threshold_p95 = np.percentile(all_weights, 95)
    G_p95 = nx.Graph()
    G_p95.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if d.get("weight", 1.0) >= threshold_p95:
            G_p95.add_edge(u, v, **d)

pos_full = nx.spring_layout(G_p95, seed=42, k=0.15, iterations=50)

colors_full = [node_color(n) for n in G_p95.nodes()]
edge_colors_full = []
for u, v in G_p95.edges():
    cu = partition_mapped.get(u, -1)
    cv = partition_mapped.get(v, -1)
    if cu == cv:
        edge_colors_full.append(PALETTE.get(cu, "#aaaaaa") + "40")
    else:
        edge_colors_full.append("#cccccc22")

fig2, ax2 = plt.subplots(figsize=(16, 13))
fig2.patch.set_facecolor("#0d1117")
ax2.set_facecolor("#0d1117")

nx.draw_networkx_edges(G_p95, pos_full, ax=ax2, edge_color=edge_colors_full, width=0.2, alpha=None)
nx.draw_networkx_nodes(G_p95, pos_full, ax=ax2, node_color=colors_full, node_size=12, linewidths=0.0)

handles2 = [mpatches.Patch(color=PALETTE[c], label=LABELS[c]) for c in [3, 1, 0, 2]]
ax2.legend(handles=handles2, loc="lower left", fontsize=9, framealpha=0.85, facecolor="#1a1f2e", edgecolor="#444", labelcolor="white")
ax2.set_title(
    f"Graphe de similarité — {G_p95.number_of_nodes()} patients (graphe complet)\n"
    f"Arêtes filtrées P95 : {G_p95.number_of_edges():,}  |  Couleur = communauté Louvain (Q = 0.307)",
    color="white", fontsize=11, pad=14
)
ax2.axis("off")
plt.tight_layout()
plt.savefig(PATH_OUT_2, dpi=90, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close()