# Eco-Smart Classifier

## Description
oki oki
Ce projet consiste à développer un pipeline complet de Machine Learning permettant de classifier des déchets et d’estimer leur valeur de revente.

Le projet couvre :

- Le nettoyage et prétraitement des données
- L’analyse non supervisée (clustering)
- L’extraction d’informations à partir de texte (NLP)
- La préparation pour des modèles ML
- Le déploiement MLOps et monitoring de production

---

# Structure du Projet

ECOML/

├── api/
│ └── main.py
│
├── data/
│ ├── raw/
│ │ └── dataset_ProjetML_2026.csv
│ └── processed/
│ └── dataset_clean.csv
│
├── models/
│ ├── modele_classification.pkl
│ ├── modele_regression.pkl
│ └── modele_nlp_tfidf_linearsvc1.pkl
│
├── src/
│ ├── prepare_data.py
│ ├── train.py
│ ├── evaluate.py
│ └── monitor.py
│
├── tests/
│ ├── test_api.py
│ ├── test_data.py
│ ├── test_model.py
│ └── test_nlp.py
│
├── dvc.yaml
├── Dockerfile
├── prometheus.yml
├── requirements.txt
└── README.md

---

# Dataset

Le dataset contient 9 colonnes :

- Poids
- Volume
- Conductivite
- Opacite
- Rigidite
- Prix_Revente
- Source (catégorielle)
- Rapport_Collecte (texte)
- Categorie (target)

## Problèmes détectés

- Valeurs manquantes (NaN)
- Outliers
- Données textuelles non structurées

---

# Prétraitement des données

Les étapes réalisées :

- Suppression des doublons
- Gestion des valeurs manquantes
- Remplacement par la médiane
- Utilisation du texte (NLP) pour compléter certaines valeurs
- Traitement des outliers
- Normalisation avec StandardScaler
- Encodage des variables catégorielles

---

# Modules

## Module 1 : Data Engineering

- Nettoyage des données
- Gestion des NaN
- KNN Imputer
- Iterative Imputer
- Traitement des outliers
- Normalisation

---

## Module 2 : Machine Learning Supervisé

### Classification

Objectif :

Prédire la catégorie du déchet.

Modèles testés :

- RandomForest
- GradientBoosting
- LogisticRegression
- SVM
- KNN

### Régression

Objectif :

Prédire le prix de revente.

Modèles testés :

- CatBoost
- XGBoost
- RandomForestRegressor
- GradientBoostingRegressor
- KNN Regressor

---

## Module 3 : Clustering

### Objectif

Segmenter les données sans utiliser la variable cible (Categorie).

### Étapes

1. Sélection des variables numériques :
   - Poids
   - Volume
   - Conductivite
   - Opacite
   - Rigidite

2. Gestion des valeurs manquantes

3. Normalisation avec StandardScaler

4. Détermination du nombre optimal de clusters :
   - Méthode du coude

5. Application de KMeans

6. Évaluation :
   - Silhouette Score

7. Visualisation :
   - PCA 2D

### Résultats

- Nombre de clusters : 3
- Algorithme : KMeans
- Visualisation : PCA

### Interprétation des clusters

- Cluster 0 : matériaux lourds
- Cluster 1 : matériaux légers
- Cluster 2 : matériaux intermédiaires

---

# Module 4 : NLP

Le texte de Rapport_Collecte a été utilisé pour :

- Extraire des informations utiles
- Nettoyer les descriptions
- Compléter certaines valeurs manquantes
- Construire un pipeline NLP

## Étapes NLP

- Lowercase
- Suppression ponctuation
- Nettoyage Regex
- TF-IDF Vectorization
- Classification NLP

---

# Module 5 : Pipeline Multimodal

Fusion des données :

- numériques
- textuelles

Techniques utilisées :

- TF-IDF
- hstack
- Pipeline sklearn
- Fusion sparse multimodale

---

# Module 6 : MLOps

## Fonctionnalités réalisées

- Versionnement Git + DVC
- Pipeline DVC automatisé
- Tracking MLflow
- API REST FastAPI
- Dockerisation
- Tests automatisés
- Monitoring drift
- Prometheus
- Grafana
- GitHub Actions

---

# Rejouer le Pipeline en 3 Commandes

## 1. Installer les dépendances


pip install -r requirements.txt

## 2. Exécuter le pipeline DVC

dvc repro

## 3.Lancer l’API FastAPI

uvicorn api.main:app --reload

# Tests Automatisés

pytest --cov=src --cov=api tests/

# API REST FastAPI

| Endpoint                | Description           |
| ----------------------- | --------------------- |
| /                       | Test API              |
| /predict/classification | Classification        |
| /predict/regression     | Régression            |
| /predict/nlp            | NLP                   |
| /metrics                | Monitoring Prometheus |

# Docker

- Construire image Docker
docker build -t eco-smart-api .

-Lancer le container

docker run -p 8000:8000 eco-smart-api

# Monitoring Drift — Evidently AI

python src/monitor.py

Le rapport HTML permet de visualiser :

les colonnes driftées
les distributions statistiques
les scores de drift
les comparaisons référence / production

# Prometheus + Grafana

Le projet utilise :

Prometheus pour collecter les métriques
Grafana pour afficher des dashboards temps réel
Endpoint Prometheus

http://127.0.0.1:8000/metrics

Les métriques surveillées :

nombre de requêtes API
activité du système
monitoring temps réel

Grafana permet ensuite de visualiser :

dashboards interactifs
courbes temps réel
observabilité du système

# GitHub Actions — CI/CD

Le pipeline CI/CD exécute automatiquement :

Black
isort
flake8
Pytest
Coverage
Docker Build

à chaque push GitHub.

L’intégration continue permet :

la validation automatique du code
l’exécution des tests
la détection rapide des erreurs
l’automatisation du déploiement

# Technologies utilisées

Python
Pandas
NumPy
Scikit-learn
MLflow
DVC
FastAPI
Docker
Pytest
Evidently AI
Prometheus
Grafana
GitHub Actions
Matplotlib
Regex
NLP

# Conclusion

Ce projet met en œuvre une démarche MLOps complète permettant :

la reproductibilité des expériences
l’automatisation du pipeline
le suivi des modèles
le monitoring de production
le déploiement via API REST
et l’observabilité temps réel

L’intégration de Git, DVC, MLflow, FastAPI, Docker, Evidently AI, Prometheus et Grafana rapproche le projet d’une architecture Machine Learning de production industrielle.