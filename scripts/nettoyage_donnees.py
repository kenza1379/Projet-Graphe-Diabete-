import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
FIG_DIR  = os.path.join(BASE_DIR, '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

fichier = os.path.join(DATA_DIR, 'diabetes_data.csv')
df = pd.read_csv(fichier)

# VALEURS MANQUANTES ET DOUBLONS
valeurs_manquantes = df.isnull().sum()
nb_doublons = df.duplicated().sum()


# NETTOYAGE DES DONNÉES

df_clean = df.copy()
lignes_avant = len(df_clean)


if nb_doublons > 0:
    df_clean = df_clean.drop_duplicates()


if valeurs_manquantes.sum() > 0:
    colonnes_numeriques = df_clean.select_dtypes(include=[np.number]).columns
    for col in colonnes_numeriques:
        if df_clean[col].isnull().sum() > 0:
            moyenne = df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(moyenne)
    

lignes_avant_aberrantes = len(df_clean)


if 'Age' in df_clean.columns:
    df_clean = df_clean[(df_clean['Age'] >= 18) & (df_clean['Age'] <= 100)]


if 'BMI' in df_clean.columns:
    df_clean = df_clean[(df_clean['BMI'] >= 10) & (df_clean['BMI'] <= 70)]


if 'FastingBloodSugar' in df_clean.columns:
    df_clean = df_clean[(df_clean['FastingBloodSugar'] >= 50) & (df_clean['FastingBloodSugar'] <= 400)]

lignes_supprimees = lignes_avant_aberrantes - len(df_clean)
lignes_apres = len(df_clean)


# SAUVEGARDE

fichier_nettoye = os.path.join(DATA_DIR, 'diabetes_data_cleaned.csv')
df_clean.to_csv(fichier_nettoye, index=False)

# VISUALISATIONS 

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(['Avant', 'Après'], [lignes_avant, lignes_apres], color=['#ff6b6b', '#4ecdc4'])
axes[0].set_ylabel('Nombre de lignes')
axes[0].set_title('Nombre de patients avant/après nettoyage')
axes[0].grid(axis='y', alpha=0.3)


for i, (label, value) in enumerate([('Avant', lignes_avant), ('Après', lignes_apres)]):
    axes[0].text(i, value + 20, str(value), ha='center', fontweight='bold')


if 'Age' in df_clean.columns:
    axes[1].hist(df_clean['Age'], bins=30, color='#4ecdc4', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Âge')
    axes[1].set_ylabel('Nombre de patients')
    axes[1].set_title('Distribution de l\'âge (après nettoyage)')
    axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'nettoyage_resume.png'), dpi=300, bbox_inches='tight')

plt.close()


