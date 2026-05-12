import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

df = pd.read_csv("../data/diabetes_data_cleaned.csv")

COLS_PROJET = [
    'Age', 'BMI', 'FastingBloodSugar', 'HbA1c',
    'SystolicBP', 'CholesterolTotal', 'SleepQuality'
]

POIDS = {
    'Age': 0.10,
    'BMI': 0.15,
    'FastingBloodSugar': 0.25,
    'HbA1c': 0.25,
    'SystolicBP': 0.15,
    'CholesterolTotal': 0.05,
    'SleepQuality': 0.05
}

os.makedirs("figures", exist_ok=True)



# STAT DESCRIPTIVES AVANCÉES


desc = df[COLS_PROJET].describe().round(3)
print(desc.to_string())

# Skewness & Kurtosis
for col in COLS_PROJET:
    sk = df[col].skew()
    ku = df[col].kurt()
    print(f"  {col:<22} | skew={sk:+.3f}  kurt={ku:+.3f}")


# MATRICE DE CORRÉLATION + HEATMAP

corr = df[COLS_PROJET + ['Diagnosis']].corr().round(3)


fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    corr, ax=ax, annot=True, fmt=".2f", cmap="RdYlBu_r",
    linewidths=0.5, linecolor='white',
    annot_kws={"size": 10},
    vmin=-1, vmax=1,
    square=True
)
ax.set_title("Matrice de corrélation — 7 variables cliniques + Diagnostic",
             fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("figures/correlation_heatmap.png", dpi=100, bbox_inches='tight')
plt.close()

# ANALYSE DE LA VARIANCE (AVEC ANOVA)

diag0 = df[df['Diagnosis'] == 0]
diag1 = df[df['Diagnosis'] == 1]

resultats_anova = []
for col in COLS_PROJET:
    f_stat, p_val = stats.f_oneway(diag0[col], diag1[col])
    eta2 = (f_stat * 1) / (f_stat * 1 + (len(df) - 2))  
    m0 = diag0[col].mean()
    m1 = diag1[col].mean()
    resultats_anova.append({
        'Variable': col,
        'Moy. Non-Diabétique': round(m0, 2),
        'Moy. Diabétique': round(m1, 2),
        'Différence': round(m1 - m0, 2),
        'F-stat': round(f_stat, 2),
        'p-value': round(p_val, 5),
        'Significatif': '✓' if p_val < 0.05 else '✗',
        'eta²': round(eta2, 4)
    })
    

anova_df = pd.DataFrame(resultats_anova)

# CORRÉLATION AVEC LE DIAG

corr_diag = df[COLS_PROJET].corrwith(df['Diagnosis']).abs().sort_values(ascending=False)

for col, val in corr_diag.items():
    bar = "█" * int(val * 40)
    print(f"  {col:<22} | {val:.3f} {bar}")

# DISTRIBUTION PAR GROUPE 
    
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

couleurs = ['#3498db', '#e74c3c']
labels = ['Non-Diabétique', 'Diabétique']

for idx, col in enumerate(COLS_PROJET[:6]):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    for g, (grp, label, c) in enumerate(zip([diag0, diag1], labels, couleurs)):
        ax.hist(grp[col], bins=25, alpha=0.6, color=c, label=label, edgecolor='none')
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_ylabel("Patients", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle("Distributions par groupe diagnostic — 6 variables cliniques",
             fontsize=13, fontweight='bold', y=1.01)
plt.savefig("figures/distributions_par_groupe.png", dpi=90, bbox_inches='tight')
plt.close()


# BOXPLOTS PAR DIAGNOSTIC

fig, axes = plt.subplots(1, len(COLS_PROJET), figsize=(18, 5))
for ax, col in zip(axes, COLS_PROJET):
    data_plot = [diag0[col].values, diag1[col].values]
    bp = ax.boxplot(data_plot, labels=['Non-D', 'Diab.'], patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_title(col, fontsize=8, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

fig.suptitle("Boxplots par groupe diagnostic", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/boxplots_diagnostic.png", dpi=90, bbox_inches='tight')
plt.close()


