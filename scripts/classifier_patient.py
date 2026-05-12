import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


COLS = ['Age', 'BMI', 'FastingBloodSugar', 'HbA1c',
        'SystolicBP', 'CholesterolTotal', 'SleepQuality']

POIDS = np.array([0.10, 0.15, 0.25, 0.25, 0.15, 0.05, 0.05])


PATH_NORM      = os.path.join(BASE_DIR, '..', 'data', 'diabetes_data_normalized.csv')
PATH_RAW       = os.path.join(BASE_DIR, '..', 'data', 'diabetes_data_cleaned.csv')
PATH_PARTITION = os.path.join(BASE_DIR, '..', 'outputs', 'partition.csv')


NOMS = {
    3: "Diabète déclaré — profil métabolique sévère     (94.5% diabétiques — glycémie ~168, HbA1c ~8.3%)",
    1: "Risque modéré — prédiabète                      (24.7% diabétiques — glycémie ~102, HbA1c ~8.3%)",
    0: "Hyperglycémie isolée — sans complication         (21.6% diabétiques — glycémie ~164, HbA1c ~5.3%)",
    2: "Profil sain — faible risque métabolique          ( 7.9% diabétiques — glycémie ~100, HbA1c ~5.4%)",
}


print("=" * 65)
print("  CLASSIFICATION D'UN NOUVEAU PATIENT PAR VOISINAGE")
print("=" * 65)


df_raw  = pd.read_csv(PATH_RAW)
df_norm = pd.read_csv(PATH_NORM)
df_part = pd.read_csv(PATH_PARTITION)

X = df_norm[COLS].values                          
partition = dict(zip(df_part["patient_idx"], df_part["communaute"]))


means = df_raw[COLS].mean().values
stds  = df_raw[COLS].std().values

print(f"✓ {len(X)} patients chargés")
print(f"✓ Partition Louvain : {df_part['communaute'].nunique()} communautés\n")


print("─" * 65)
print("  Entrez les valeurs du nouveau patient")
print("  (appuyez Entrée pour utiliser la valeur entre crochets)")
print("─" * 65)


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
            val = float(saisie) if saisie else float(default)
            nouveau[col] = val
            break
        except ValueError:
            print(f"    ⚠ Valeur invalide, entrez un nombre.")


vals_brutes  = np.array([nouveau[c] for c in COLS])
vals_normees = (vals_brutes - means) / stds         

print("\n─" * 33)
print("  Valeurs normalisées (Z-score) :")
for c, z in zip(COLS, vals_normees):
    print(f"    {c:<22} : {z:+.3f}")


print("\n  Calcul de la similarité avec les 1879 patients...")

sqrt_poids = np.sqrt(POIDS)
x_new_w    = vals_normees * sqrt_poids               
X_w        = X * sqrt_poids                         

diff       = X_w - x_new_w                          
distances  = np.sqrt((diff ** 2).sum(axis=1))        
similarites = 1.0 / (1.0 + distances)              


K = 20   
idx_top = np.argsort(similarites)[::-1][:K]

print(f"  ✓ Similarité calculée — {K} voisins les plus proches retenus\n")


votes = {}
for idx in idx_top:
    comm = partition.get(idx, -1)
    votes[comm] = votes.get(comm, 0) + 1

comm_predite = max(votes, key=votes.get)
score_confiance = votes[comm_predite] / K * 100

#Resultat
print("=" * 65)
print("  RÉSULTAT DE LA CLASSIFICATION")
print("=" * 65)

print(f"\n  ➤ Communauté prédite : Comm {comm_predite}")
print(f"    {NOMS[comm_predite]}")
print(f"\n  ➤ Confiance : {score_confiance:.0f}% ({votes[comm_predite]}/{K} voisins)")

print(f"\n  Répartition des {K} voisins :")
for comm, nb in sorted(votes.items(), key=lambda x: -x[1]):
    barre = "█" * nb + "░" * (K - nb)
    print(f"    Comm {comm} : {barre} {nb}/{K}")


sim_max = similarites[idx_top[0]]
print(f"\n  Similarité du voisin le plus proche : {sim_max:.4f}")
print(f"  Similarité moyenne des {K} voisins  : {similarites[idx_top].mean():.4f}")


print("\n─" * 33)
print("  Récapitulatif des valeurs saisies :")
normales = {
    'FastingBloodSugar': (0, 100),
    'HbA1c':             (0, 5.7),
    'BMI':               (18.5, 24.9),
}
for col in COLS:
    val = nouveau[col]
    flag = ""
    if col in normales:
        lo, hi = normales[col]
        flag = "  ⚠ ÉLEVÉ" if val > hi else "  ✓ normal"
    print(f"    {col:<22} : {val}{flag}")

print("\n" + "=" * 65)
print("Classification terminée")
print("=" * 65)