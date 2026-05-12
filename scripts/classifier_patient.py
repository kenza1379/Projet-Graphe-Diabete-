import pandas as pd
import numpy as np
 
 
COLS = ['Age', 'BMI', 'FastingBloodSugar', 'HbA1c',
        'SystolicBP', 'CholesterolTotal', 'SleepQuality']
 
POIDS = np.array([0.10, 0.15, 0.25, 0.25, 0.15, 0.05, 0.05])
 
PATH_NORM      = "../data/diabetes_data_normalized.csv"
PATH_RAW       = "../data/diabetes_data_cleaned.csv"
PATH_PARTITION = "../outputs/partition.csv"
 
NOMS = {
    3: "Diabète déclaré — profil métabolique sévère     (94.5% diabétiques — glycémie ~168, HbA1c ~8.3%)",
    1: "Risque modéré — prédiabète                      (24.7% diabétiques — glycémie ~102, HbA1c ~8.3%)",
    0: "Hyperglycémie isolée — sans complication         (21.6% diabétiques — glycémie ~164, HbA1c ~5.3%)",
    2: "Profil sain — faible risque métabolique          ( 7.9% diabétiques — glycémie ~100, HbA1c ~5.4%)",
}
 
df_raw  = pd.read_csv(PATH_RAW)
df_norm = pd.read_csv(PATH_NORM)
df_part = pd.read_csv(PATH_PARTITION)
 
X = df_norm[COLS].values
partition = dict(zip(df_part["patient_idx"], df_part["communaute"]))
 
means = df_raw[COLS].mean().values
stds  = df_raw[COLS].std().values
 
defaults = {
    'Age':               55,
    'BMI':               27.5,
    'FastingBloodSugar': 145,
    'HbA1c':             7.8,
    'SystolicBP':        128,
    'CholesterolTotal':  195,
    'SleepQuality':      6,
}
 
unites = {
    'Age':               'ans',
    'BMI':               'kg/m²',
    'FastingBloodSugar': 'mg/dL  (normal < 100)',
    'HbA1c':             '%      (normal < 5.7%)',
    'SystolicBP':        'mmHg',
    'CholesterolTotal':  'mg/dL',
    'SleepQuality':      '/ 10',
}
 
nouveau = {}
for col in COLS:
    default = defaults[col]
    unite   = unites[col]
    while True:
        try:
            saisie = input(f"  {col:<22} [{default}] {unite} : ").strip()
            nouveau[col] = float(saisie) if saisie else float(default)
            break
        except ValueError:
            print(f"    ⚠ Valeur invalide, entrez un nombre.")
 
vals_brutes  = np.array([nouveau[c] for c in COLS])
vals_normees = (vals_brutes - means) / stds
 
sqrt_poids = np.sqrt(POIDS)
x_new_w    = vals_normees * sqrt_poids
X_w        = X * sqrt_poids
distances   = np.sqrt(((X_w - x_new_w) ** 2).sum(axis=1))
similarites = 1.0 / (1.0 + distances)
 
K = 20
idx_top = np.argsort(similarites)[::-1][:K]
 
votes = {}
for idx in idx_top:
    comm = partition.get(idx, -1)
    votes[comm] = votes.get(comm, 0) + 1
 
comm_predite    = max(votes, key=votes.get)
score_confiance = votes[comm_predite] / K * 100
 
print(f"\n  ➤ Communauté prédite : Comm {comm_predite}")
print(f"    {NOMS[comm_predite]}")
print(f"\n  ➤ Confiance : {score_confiance:.0f}% ({votes[comm_predite]}/{K} voisins)")
 
print(f"\n  Répartition des {K} voisins :")
for comm, nb in sorted(votes.items(), key=lambda x: -x[1]):
    barre = "█" * nb + "░" * (K - nb)
    print(f"    Comm {comm} : {barre} {nb}/{K}")
 
print(f"\n  Similarité du voisin le plus proche : {similarites[idx_top[0]]:.4f}")
print(f"  Similarité moyenne des {K} voisins  : {similarites[idx_top].mean():.4f}")

