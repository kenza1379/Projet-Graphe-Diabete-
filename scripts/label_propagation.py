import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUT_DIR  = os.path.join(BASE_DIR, '..', 'outputs')
FIG_DIR  = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)



similarite   = np.load(os.path.join(OUT_DIR, 'similarite_matrix.npy'))
partition_df = pd.read_csv(os.path.join(OUT_DIR, 'partition.csv'))
df_raw       = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_data_cleaned.csv'))



n         = len(similarite)
idx_upper = np.triu_indices(n, k=1)
vals_all  = similarite[idx_upper]
seuil     = float(np.percentile(vals_all, 75))


G = nx.Graph()
G.add_nodes_from(range(n))
rows, cols = np.where(similarite >= seuil)
mask = rows < cols
G.add_edges_from(zip(rows[mask].tolist(), cols[mask].tolist()))


NOMS_COMMUNAUTES = {
    3: "Diabète déclaré\n(profil sévère)",
    1: "Profil à risque\n(prédiabète)",
    0: "Hyperglycémie isolée\n(sans complication)",
    2: "Profil sain\n(faible risque)"
}


t0           = time.time()
comm_louvain = nx_comm.louvain_communities(G, seed=42, resolution=1.0)
q_louvain    = nx_comm.modularity(G, comm_louvain)
t_louvain    = time.time() - t0



t0      = time.time()
comm_lp = list(nx_comm.asyn_lpa_communities(G, seed=42))
q_lp    = nx_comm.modularity(G, comm_lp)
t_lp    = time.time() - t0



t0      = time.time()
comm_gm = list(nx_comm.greedy_modularity_communities(G))
q_gm    = nx_comm.modularity(G, comm_gm)
t_gm    = time.time() - t0




partition_lv = {}
for i, comm in enumerate(comm_louvain):
    for node in comm:
        partition_lv[node] = i

partition_lp = {}
for i, comm in enumerate(comm_lp):
    for node in comm:
        partition_lp[node] = i

partition_gm = {}
for i, comm in enumerate(comm_gm):
    for node in comm:
        partition_gm[node] = i


largest_cc = max(nx.connected_components(G), key=len)
G_sub      = G.subgraph(largest_cc).copy()
pos        = nx.spring_layout(G_sub, seed=42, k=0.4)



cmap_lv = cm.get_cmap("tab10", len(comm_louvain))
cmap_lp = cm.get_cmap("tab20", len(comm_lp))
cmap_gm = cm.get_cmap("Set2",  len(comm_gm))

colors_lv = [cmap_lv(partition_lv.get(node, 0)) for node in G_sub.nodes()]
colors_lp = [cmap_lp(partition_lp.get(node, 0)) for node in G_sub.nodes()]
colors_gm = [cmap_gm(partition_gm.get(node, 0)) for node in G_sub.nodes()]


fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#0d1117")
fig.suptitle(
    "Comparaison des algorithmes de détection de communautés\nPatients diabétiques — Graphe de similarité (seuil P75)",
    color="white", fontsize=13, fontweight="bold", y=0.99
)


ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

EDGE_ALPHA = 0.05
NODE_SIZE  = 25


ax1.set_facecolor("#1a1f2e")
ax1.set_title(
    f"Louvain\n{len(comm_louvain)} communautés | Q = {q_louvain:.4f} | {t_louvain:.1f}s",
    color="white", fontsize=10, fontweight="bold"
)
nx.draw_networkx_nodes(G_sub, pos, ax=ax1, node_color=colors_lv, node_size=NODE_SIZE, alpha=0.9)
nx.draw_networkx_edges(G_sub, pos, ax=ax1, edge_color="white", alpha=EDGE_ALPHA, width=0.4)


patches_lv = []
for comm_id in sorted(set(partition_lv.values())):
    nom = NOMS_COMMUNAUTES.get(comm_id, f"Comm {comm_id}")
    taille = sum(1 for v in partition_lv.values() if v == comm_id)
    patches_lv.append(mpatches.Patch(
        color=cmap_lv(comm_id),
        label=f"C{comm_id} — {nom.replace(chr(10), ' ')} ({taille} pts)"
    ))
ax1.legend(handles=patches_lv, fontsize=6, loc="lower left",
           facecolor="#1a1f2e", edgecolor="#444", labelcolor="white")
ax1.axis("off")


ax2.set_facecolor("#1a1f2e")
ax2.set_title(
    f"Label Propagation\n{len(comm_lp)} communautés | Q = {q_lp:.4f} | {t_lp:.1f}s",
    color="white", fontsize=10, fontweight="bold"
)
nx.draw_networkx_nodes(G_sub, pos, ax=ax2, node_color=colors_lp, node_size=NODE_SIZE, alpha=0.9)
nx.draw_networkx_edges(G_sub, pos, ax=ax2, edge_color="white", alpha=EDGE_ALPHA, width=0.4)
ax2.axis("off")


ax3.set_facecolor("#1a1f2e")
ax3.set_title(
    f"Greedy Modularity\n{len(comm_gm)} communautés | Q = {q_gm:.4f} | {t_gm:.1f}s",
    color="white", fontsize=10, fontweight="bold"
)
nx.draw_networkx_nodes(G_sub, pos, ax=ax3, node_color=colors_gm, node_size=NODE_SIZE, alpha=0.9)
nx.draw_networkx_edges(G_sub, pos, ax=ax3, edge_color="white", alpha=EDGE_ALPHA, width=0.4)
ax3.axis("off")


algos    = ["Louvain", "Label\nPropagation", "Greedy\nModularity"]
moduls   = [q_louvain, q_lp, q_gm]
n_comms  = [len(comm_louvain), len(comm_lp), len(comm_gm)]
temps    = [t_louvain, t_lp, t_gm]
couleurs = ["#4FC3F7", "#81C784", "#FFB74D"]

ax4.set_facecolor("#1a1f2e")
bars = ax4.bar(algos, moduls, color=couleurs, alpha=0.85, width=0.5)
for bar, val in zip(bars, moduls):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.4f}", ha="center", va="bottom",
        color="white", fontsize=10, fontweight="bold"
    )
ax4.axhline(0.3, color="#e74c3c", linestyle="--", linewidth=1, label="Seuil Q = 0.3")
ax4.set_title("Modularité Q", color="white", fontsize=10, fontweight="bold")
ax4.tick_params(colors="white")
ax4.set_ylim(0, max(moduls) * 1.25)
for spine in ["top", "right"]:
    ax4.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax4.spines[spine].set_color("#444")
ax4.legend(fontsize=8, labelcolor="white", facecolor="#1a1f2e", edgecolor="#444")


ax5.set_facecolor("#1a1f2e")
bars2 = ax5.bar(algos, n_comms, color=couleurs, alpha=0.85, width=0.5)
for bar, val in zip(bars2, n_comms):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        str(val), ha="center", va="bottom",
        color="white", fontsize=10, fontweight="bold"
    )
ax5.set_title("Nombre de communautés", color="white", fontsize=10, fontweight="bold")
ax5.tick_params(colors="white")
ax5.set_ylim(0, max(n_comms) * 1.3)
for spine in ["top", "right"]:
    ax5.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax5.spines[spine].set_color("#444")


ax6.set_facecolor("#1a1f2e")
bars3 = ax6.bar(algos, temps, color=couleurs, alpha=0.85, width=0.5)
for bar, val in zip(bars3, temps):
    ax6.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:.1f}s", ha="center", va="bottom",
        color="white", fontsize=10, fontweight="bold"
    )
ax6.set_title("Temps d'exécution", color="white", fontsize=10, fontweight="bold")
ax6.tick_params(colors="white")
ax6.set_ylim(0, max(temps) * 1.3)
for spine in ["top", "right"]:
    ax6.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax6.spines[spine].set_color("#444")


meilleur = max(zip(["Louvain", "Label Propagation", "Greedy Modularity"], moduls), key=lambda x: x[1])
fig.text(
    0.5, 0.01,
    f"→ Algorithme retenu : {meilleur[0]} (Q = {meilleur[1]:.4f})"
    f"  |  Granularité clinique optimale : {len(comm_louvain)} groupes de patients identifiés",
    ha="center", color="#4FC3F7", fontsize=10, fontweight="bold"
)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(os.path.join(FIG_DIR, 'comparaison_algorithmes.png'),
            dpi=90, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
