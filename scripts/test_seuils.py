import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import os

os.makedirs("../outputs/figures", exist_ok=True)

similarite = np.load("../outputs/similarite_matrix.npy")

idx_upper = np.triu_indices(len(similarite), k=1)
vals_all  = similarite[idx_upper]

seuils = {
    "P70": float(np.percentile(vals_all, 70)),
    "P75": float(np.percentile(vals_all, 75)),
    "P80": float(np.percentile(vals_all, 80)),
}

resultats = []

for nom, seuil in seuils.items():
    n = len(similarite)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(similarite >= seuil)
    mask = rows < cols
    G.add_edges_from(zip(rows[mask].tolist(), cols[mask].tolist()))

    densite   = nx.density(G)
    degre_moy = np.mean([d for _, d in G.degree()])

    communities = nx_comm.louvain_communities(G, seed=42, resolution=1.0)
    n_comm      = len(communities)
    modularite  = nx_comm.modularity(G, communities)

    resultats.append({
        "seuil":      nom,
        "valeur":     seuil,
        "aretes":     G.number_of_edges(),
        "densite":    densite,
        "degre_moy":  degre_moy,
        "n_comm":     n_comm,
        "modularite": modularite,
    })

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("#0d1117")

metrics = [
    ("aretes",     "Nombre d'arêtes", "#4FC3F7"),
    ("degre_moy",  "Degré moyen",     "#FFB74D"),
    ("modularite", "Modularité Q",    "#81C784"),
]

for ax, (metric, label, color) in zip(axes, metrics):
    ax.set_facecolor("#1a1f2e")
    vals  = [r[metric] for r in resultats]
    noms  = [r["seuil"] for r in resultats]
    bars  = ax.bar(noms, vals, color=color, alpha=0.85, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.02,
            f"{val:,.3f}" if metric == "modularite" else f"{val:,.0f}",
            ha="center", va="bottom", color="white", fontsize=10, fontweight="bold"
        )
    ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Seuil", color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.15)

fig.suptitle(
    "Comparaison des graphes selon le seuil de similarité (P70 / P75 / P80)",
    color="white", fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("../outputs/figures/comparaison_seuils.png", dpi=90, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

PALETTE = {
    0: "#E24B4A", 1: "#E09020", 2: "#185FA5", 3: "#0F6E56",
}

df_part = pd.read_csv("../outputs/partition.csv")
partition_base = dict(zip(df_part["patient_idx"], df_part["communaute"]))

fig3, axes3 = plt.subplots(1, 3, figsize=(21, 7))
fig3.patch.set_facecolor("#0d1117")

for ax, r, seuil_val in zip(axes3, resultats, seuils.values()):
    n = len(similarite)
    G_visu = nx.Graph()
    G_visu.add_nodes_from(range(n))
    rows_idx, cols_idx = np.where(similarite >= seuil_val)
    mask = rows_idx < cols_idx
    edges_all = list(zip(rows_idx[mask].tolist(), cols_idx[mask].tolist()))

    weights   = [float(similarite[u, v]) for u, v in edges_all]
    p95       = np.percentile(weights, 95) if weights else 0
    edges_p95 = [(u, v) for (u, v), w in zip(edges_all, weights) if w >= p95]
    G_visu.add_edges_from(edges_p95)

    pos = nx.spring_layout(G_visu, seed=42, k=0.15, iterations=40)

    node_colors = [PALETTE.get(partition_base.get(n, 0), "#888888") for n in G_visu.nodes()]

    ax.set_facecolor("#0d1117")
    nx.draw_networkx_nodes(G_visu, pos, ax=ax, node_color=node_colors, node_size=8, linewidths=0)
    nx.draw_networkx_edges(G_visu, pos, ax=ax, alpha=0.08, width=0.2, edge_color="#aaaaaa")

    ax.set_title(
        f"Seuil {r['seuil']} ({seuil_val:.4f})\n{r['aretes']:,} arêtes | {r['n_comm']} comm. | Q={r['modularite']:.3f}",
        color="white", fontsize=10, fontweight="bold", pad=10
    )
    ax.axis("off")

fig3.suptitle(
    "Graphes de similarité — Comparaison visuelle P70 / P75 / P80",
    color="white", fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("../outputs/figures/graphes_seuils_compares.png", dpi=90, bbox_inches="tight", facecolor=fig3.get_facecolor())
plt.close()