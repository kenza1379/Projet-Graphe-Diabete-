import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# CHEMINS
PATH_PROFILS = "../outputs/profils.csv"
PATH_OUT     = "../outputs/figures/heatmap_profils_communautes.png"

# NOMS ET COULEURS DES COMMU
NOMS = {
    3: "Comm 3\nDiabétique Sévère",
    1: "Comm 1\nRisque Modéré",
    0: "Comm 0\nHyperglycémie isolée",
    2: "Comm 2\nProfil Sain",
}
COULEURS = {
    3: "#E24B4A",
    1: "#E09020",
    0: "#185FA5",
    2: "#0F6E56",
}

# VARIABLES À AFFICHER
VARIABLES = {
    "moy_FastingBloodSugar": "Glycémie à jeun\n(mg/dL)",
    "moy_HbA1c":             "HbA1c\n(%)",
    "moy_BMI":               "IMC\n(kg/m²)",
    "moy_Age":               "Âge\n(ans)",
}

# VALEURS DE REF NORMALES
NORMALES = {
    "moy_FastingBloodSugar": "Normal < 100",
    "moy_HbA1c":             "Normal < 5.7%",
    "moy_BMI":               "Normal 18.5–24.9",
    "moy_Age":               "",
}

df = pd.read_csv(PATH_PROFILS)

ordre = [3, 1, 0, 2]
df = df.set_index("communaute").loc[ordre].reset_index()

# MATRICE POUR HEATMAP
cols = list(VARIABLES.keys())
matrix = df[cols].values.astype(float)  

# NORMALISATION PAR COULEURS
matrix_norm = np.zeros_like(matrix)
for j in range(matrix.shape[1]):
    col = matrix[:, j]
    mn, mx = col.min(), col.max()
    if mx > mn:
        matrix_norm[:, j] = (col - mn) / (mx - mn)
    else:
        matrix_norm[:, j] = 0.5

# FIGURES
fig = plt.figure(figsize=(13, 7))
fig.patch.set_facecolor("#0d1117")

gs = GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.04)
ax_heat = fig.add_subplot(gs[0])
ax_info = fig.add_subplot(gs[1])

# HEATMAP PRINCIPALE

cmap = plt.cm.RdYlBu_r

im = ax_heat.imshow(matrix_norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

# VALEURS RÉELLES
for i in range(4):
    for j in range(4):
        val = matrix[i, j]
        norm_val = matrix_norm[i, j]
        text_color = "white" if norm_val > 0.6 or norm_val < 0.25 else "#111"
        ax_heat.text(
            j, i, f"{val:.1f}",
            ha="center", va="center",
            fontsize=13, fontweight="bold",
            color=text_color
        )

# AXES
ax_heat.set_xticks(range(len(cols)))
ax_heat.set_xticklabels(
    [VARIABLES[c] for c in cols],
    fontsize=10, color="white", ha="center"
)
ax_heat.set_yticks(range(4))
ax_heat.set_yticklabels(
    [NOMS[c] for c in ordre],
    fontsize=10, color="white"
)


for tick, comm in zip(ax_heat.get_yticklabels(), ordre):
    tick.set_color(COULEURS[comm])
    tick.set_fontweight("bold")


for i, comm in enumerate(ordre):
    ax_heat.add_patch(plt.Rectangle(
        (-0.5, i - 0.5), len(cols), 1,
        fill=False,
        edgecolor=COULEURS[comm],
        linewidth=1.5,
        clip_on=False
    ))


ax_heat.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
ax_heat.set_yticks(np.arange(-0.5, 4, 1), minor=True)
ax_heat.grid(which="minor", color="#333", linewidth=0.8)
ax_heat.tick_params(which="minor", bottom=False, left=False)

ax_heat.set_title(
    "Profils cliniques moyens par communauté Louvain",
    color="white", fontsize=13, fontweight="bold", pad=14
)
ax_heat.tick_params(colors="white", labelsize=10)
for spine in ax_heat.spines.values():
    spine.set_edgecolor("#444")


cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
cbar.ax.tick_params(colors="white", labelsize=8)
cbar.set_label("Intensité relative (min→max par variable)", color="white", fontsize=8)
cbar.outline.set_edgecolor("#444")


ax_info.set_facecolor("#0d1117")
ax_info.axis("off")

y = 0.97
ax_info.text(0.05, y, "Effectifs & diabète", color="white",
             fontsize=10, fontweight="bold", transform=ax_info.transAxes, va="top")
y -= 0.06

for comm in ordre:
    row = df[df["communaute"] == comm].iloc[0]
    n    = int(row["taille"])
    pct  = row["taux_diabete_%"]
    ax_info.text(
        0.05, y,
        f"● {n} patients",
        color=COULEURS[comm], fontsize=9, fontweight="bold",
        transform=ax_info.transAxes, va="top"
    )
    y -= 0.05
    ax_info.text(
        0.08, y,
        f"{pct}% diabétiques",
        color="#aaaaaa", fontsize=8,
        transform=ax_info.transAxes, va="top"
    )
    y -= 0.07


y -= 0.04
ax_info.text(0.05, y, "Valeurs normales", color="white",
             fontsize=10, fontweight="bold", transform=ax_info.transAxes, va="top")
y -= 0.06
for col, ref in NORMALES.items():
    if ref:
        ax_info.text(
            0.05, y, ref,
            color="#888888", fontsize=8,
            transform=ax_info.transAxes, va="top"
        )
        y -= 0.055


y -= 0.04
ax_info.text(
    0.05, y,
    "Les couleurs reflètent\nl'intensité relative\npar variable\n(normalisation min-max).",
    color="#666666", fontsize=7.5, style="italic",
    transform=ax_info.transAxes, va="top", linespacing=1.6
)


plt.savefig(PATH_OUT, dpi=90, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

