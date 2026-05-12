# Graphe de Similarité entre Patients Diabétiques
 
Projet réalisé dans le cadre du module **Graphes et OpenData** — Licence MIAGE, Université Paris Nanterre (2025–2026).
 
**Auteurs :** Kenza CHANAZ & Fatoumata DEMBELE  
**Application interactive :** [https://diabgraphe.streamlit.app/](https://diabgraphe.streamlit.app/)
 
---
 
## Description
 
Ce projet construit et analyse un **graphe de similarité entre 1 879 patients diabétiques**
à partir de données cliniques réelles. Chaque nœud représente un patient, chaque arête
exprime la proximité clinique entre deux individus selon 7 variables pondérées.
L'algorithme de Louvain permet de regrouper automatiquement les patients en 4 profils médicaux cohérents.
 
---
 
## Structure du projet
 
```
Projet/
├── data/
│   ├── diabetes_data.csv                 # Données brutes (1 879 patients, 46 colonnes)
│   ├── diabetes_data_cleaned.csv         # Données nettoyées
│   └── diabetes_data_normalized.csv      # Données normalisées (Z-score)
│
├── scripts/
│   ├── nettoyage_donnees.py              # Étape 1 — Nettoyage
│   ├── normalisation.py                  # Étape 2 — Normalisation Z-score
│   ├── exploration_avancee.py            # Étape 3 — Exploration statistique
│   ├── similarite.py                     # Étape 4 — Matrice de similarité
│   ├── graphe.py                         # Étape 5 — Construction du graphe
│   ├── communautes.py                    # Étape 6 — Détection Louvain
│   ├── label_propagation.py              # Étape 7 — Comparaison algorithmes
│   ├── test_seuils.py                    # Étape 8 — Test des seuils
│   ├── test_distances.py                 # Étape 9 — Test des distances
│   ├── visualisation_graphe.py           # Étape 10 — Visualisation du graphe
│   ├── visualisation.py                  # Étape 11 — Visualisations diverses
│   ├── heatmap_profils.py                # Étape 12 — Heatmap des profils
│   ├── classifier_patient.py             # Étape 13 — Classification nouveau patient
│   └── export_gephi.py                   # Export .gml pour Gephi
│
├── outputs/
│   ├── figures/
│   │   ├── boxplots_diagnostic.png
│   │   ├── communautes_profils.png
│   │   ├── communautes_radar.png
│   │   ├── comparaison_algorithmes.png
│   │   ├── comparaison_distances.png
│   │   ├── comparaison_seuils.png
│   │   ├── correlation_heatmap.png
│   │   ├── distributions_par_groupe.png
│   │   ├── graphe_complet_p95.png
│   │   ├── graphe_sous_ensemble.png
│   │   ├── graphes_seuils_compares.png
│   │   ├── heatmap_profils_communautes.png
│   │   └── nettoyage_resume.png
│   ├── graphe.gml
│   ├── profils.csv
│   ├── partition.csv
│   ├── similarite_matrix.npy
│   ├── rapport_communautes.txt
│   └── visualisation.png
```
 
---
 
## Variables cliniques utilisées
 
| Variable | Description | Poids |
|---|---|---|
| FastingBloodSugar | Glycémie à jeun | 25% |
| HbA1c | Hémoglobine glyquée | 25% |
| SystolicBP | Pression artérielle systolique | 15% |
| BMI | Indice de masse corporelle | 15% |
| Age | Âge | 10% |
| CholesterolTotal | Cholestérol total | 5% |
| SleepQuality | Qualité du sommeil | 5% |
 
---
 
## Ordre d'exécution des scripts
 
```bash
cd scripts/
python3 nettoyage_donnees.py
python3 normalisation.py
python3 similarite.py
python3 graphe.py
python3 communautes.py
python3 visualisation_graphe.py
python3 heatmap_profils.py
python3 classifier_patient.py
```
 
---
 
## Résultats principaux
 
- **4 communautés** détectées par Louvain (modularité Q ≈ 0.52)
- **Communauté 3** — Diabétique sévère (94.5% diabétiques)
- **Communauté 1** — Risque modéré (24.7%)
- **Communauté 0** — Hyperglycémie isolée (21.6%)
- **Communauté 2** — Profil sain (7.9%)