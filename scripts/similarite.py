import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUT_DIR  = os.path.join(BASE_DIR, '..', 'outputs')
FIG_DIR  = os.path.join(BASE_DIR, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

df_norm = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_data_normalized.csv'))
df_raw  = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_data_cleaned.csv'))


COLS_PROJET = [
    'Age', 'BMI', 'FastingBloodSugar', 'HbA1c',
    'SystolicBP', 'CholesterolTotal', 'SleepQuality'
]

POIDS = np.array([0.10, 0.15, 0.25, 0.25, 0.15, 0.05, 0.05])

X      = df_norm.values         
labels = df_raw['Diagnosis'].values

os.makedirs("figures", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

n = len(X)
sqrt_poids = np.sqrt(POIDS)     
t0 = time.time()

X_w = X * sqrt_poids             

BLOC = 200                       
similarite = np.zeros((n, n), dtype=np.float32)

for i in range(0, n, BLOC):
    for j in range(i, n, BLOC):
        xi = X_w[i:i+BLOC]      
        xj = X_w[j:j+BLOC]      

        diff = xi[:, None, :] - xj[None, :, :]   
        d = np.sqrt((diff ** 2).sum(axis=2))      
        s = 1.0 / (1.0 + d)
        similarite[i:i+BLOC, j:j+BLOC] = s
        if i != j:
            similarite[j:j+BLOC, i:i+BLOC] = s.T

elapsed = time.time() - t0

idx_upper = np.triu_indices(n, k=1)
vals = similarite[idx_upper]

SEUIL_P75 = np.percentile(vals, 75)
SEUIL_P90 = np.percentile(vals, 90)


diab_idx   = np.where(labels == 1)[0]
non_diab_idx = np.where(labels == 0)[0]

def sim_intra(grp):
    idx = np.triu_indices(len(grp), k=1)
    submat = similarite[np.ix_(grp, grp)]
    return submat[idx]

sim_dd  = sim_intra(diab_idx)       
sim_nn  = sim_intra(non_diab_idx)   
sim_inter = similarite[np.ix_(diab_idx, non_diab_idx)].flatten()

ratio = ((sim_dd.mean() + sim_nn.mean()) / 2) / sim_inter.mean()

scenarios = {
    "(référence)": np.array([0.10, 0.15, 0.25, 0.25, 0.15, 0.05, 0.05]),
    "Biomarqueurs renforcés":np.array([0.05, 0.10, 0.35, 0.35, 0.05, 0.05, 0.05]),
    "Poids uniformes":       np.array([1/7]*7),
    "Facteurs risque++":     np.array([0.15, 0.20, 0.20, 0.20, 0.15, 0.05, 0.05]),
}

N_TEST = 300
X_test = X[:N_TEST]
sqrt_poids_ref = np.sqrt(POIDS)

resultats_sens = {}
for nom, w in scenarios.items():
    X_w_test = X_test * np.sqrt(w)
    diff = X_w_test[:, None, :] - X_w_test[None, :, :]
    d = np.sqrt((diff**2).sum(axis=2))
    s = 1.0 / (1.0 + d)
    vals_test = s[np.triu_indices(N_TEST, k=1)]
    resultats_sens[nom] = vals_test
    seuil_t = np.percentile(vals_test, 75)
    nb_aretes = (vals_test >= seuil_t).sum() * 2


fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)


ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(vals, bins=60, color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=0.3)
ax1.axvline(vals.mean(), color='black', linestyle='--', lw=1.8, label=f'μ = {vals.mean():.4f}')
ax1.axvline(SEUIL_P75, color='#e74c3c', linestyle='-.', lw=1.8, label=f'Seuil P75 = {SEUIL_P75:.4f}')
ax1.axvline(SEUIL_P90, color='#e67e22', linestyle=':', lw=1.8, label=f'Seuil P90 = {SEUIL_P90:.4f}')
ax1.set_xlabel("Similarité", fontsize=10)
ax1.set_ylabel("Nombre de paires", fontsize=10)
ax1.set_title("Distribution de la similarité pondérée — 1879 patients (toutes paires)", fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)


ax2 = fig.add_subplot(gs[0, 2])
sub50 = similarite[:50, :50]
im = ax2.imshow(sub50, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=0.8)
ax2.set_title("Matrice de sim. (50×50)", fontsize=10, fontweight='bold')
ax2.set_xlabel("Patient", fontsize=8)
ax2.set_ylabel("Patient", fontsize=8)
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)


ax3 = fig.add_subplot(gs[1, :2])
ax3.hist(sim_nn, bins=50, alpha=0.6, color='#3498db', label=f'Intra Non-Diab. μ={sim_nn.mean():.4f}', density=True)
ax3.hist(sim_dd, bins=50, alpha=0.6, color='#e74c3c', label=f'Intra Diabétiques μ={sim_dd.mean():.4f}', density=True)
ax3.hist(sim_inter, bins=50, alpha=0.5, color='#95a5a6', label=f'Inter-groupes μ={sim_inter.mean():.4f}', density=True)
ax3.set_xlabel("Similarité", fontsize=10)
ax3.set_ylabel("Densité", fontsize=10)
ax3.set_title("Similarité intra vs inter groupes diagnostiques", fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3)


ax4 = fig.add_subplot(gs[1, 2])
data_box = [v for v in resultats_sens.values()]
labels_box = [k.replace(' ', '\n') for k in resultats_sens.keys()]
bp = ax4.boxplot(data_box, labels=labels_box, patch_artist=True,
                 medianprops=dict(color='black', linewidth=2))
colors_box = ['#2ecc71', '#3498db', '#e67e22', '#9b59b6']
for patch, c in zip(bp['boxes'], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax4.set_title("Sensibilité aux scénarios\nde pondération (300 patients)", fontsize=9, fontweight='bold')
ax4.set_ylabel("Similarité", fontsize=9)
ax4.tick_params(axis='x', labelsize=6)
ax4.grid(axis='y', alpha=0.3)

fig.suptitle("Analyse de la Matrice de Similarité Pondérée — Kenza", fontsize=13, fontweight='bold')
plt.savefig(os.path.join(FIG_DIR, 'similarite_analyse.png'), dpi=90, bbox_inches='tight')
plt.close()

np.save(os.path.join(OUT_DIR, 'similarite_matrix.npy'), similarite)

