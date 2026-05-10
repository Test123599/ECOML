# Eco-Smart Classifier

## Description

**Eco-Smart Classifier** est un projet complet de Machine Learning et MLOps permettant :

* la classification intelligente des déchets,
* la prédiction du prix de revente,
* l’analyse NLP des descriptions textuelles,
* le clustering non supervisé,
* le déploiement d’une API REST,
* et le monitoring en production.

Le projet a été conçu comme une architecture IA industrialisable intégrant :

* Data Engineering,
* Machine Learning,
* NLP,
* Pipeline Multimodal,
* MLOps,
* Monitoring,
* CI/CD,
* et Déploiement Docker.

---

# Fonctionnalités Principales

## Classification des déchets

Prédire automatiquement la catégorie d’un déchet à partir :

* des données numériques,
* des caractéristiques physiques,
* et des descriptions textuelles.

## Régression

Estimer automatiquement le prix de revente d’un déchet recyclable.

## NLP

Analyser les descriptions textuelles des déchets avec :

* TF-IDF,
* Bag of Words,
* LinearSVC,
* NLP preprocessing.

## Clustering

Segmenter les déchets avec :

* KMeans,
* PCA,
* Elbow Method.

## MLOps

Pipeline complet avec :

* Git,
* DVC,
* MLflow,
* FastAPI,
* Docker,
* Pytest,
* GitHub Actions,
* Prometheus,
* Grafana,
* Evidently AI.

---

# Structure du Projet

```text
ECOML/
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── dataset_ProjetML_2026.csv
│   │
│   └── processed/
│       └── dataset_clean.csv
│
├── models/
│   ├── modele_classification.pkl
│   ├── modele_regression.pkl
│   └── modele_nlp_tfidf_linearsvc1.pkl
│
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── monitor.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_nlp.py
│   └── test_train.py
│
├── logs/
│   └── predictions.jsonl
│
├── .github/workflows/
│   └── ci.yml
│
├── dvc.yaml
├── Dockerfile
├── prometheus.yml
├── requirements.txt
└── README.md
```

---

# Dataset

Le dataset contient les colonnes suivantes :

| Colonne          | Description              |
| ---------------- | ------------------------ |
| Poids            | Poids du déchet          |
| Volume           | Volume du déchet         |
| Conductivite     | Conductivité du matériau |
| Opacite          | Niveau d’opacité         |
| Rigidite         | Rigidité du matériau     |
| Prix_Revente     | Prix de revente          |
| Source           | Source du déchet         |
| Rapport_Collecte | Description textuelle    |
| Categorie        | Variable cible           |

---

# Prétraitement des Données

Les étapes réalisées :

* suppression des doublons,
* gestion des valeurs manquantes,
* comparaison des méthodes d’imputation,
* détection et traitement des outliers,
* normalisation des données,
* encodage des variables catégorielles,
* préparation NLP.

## Méthodes utilisées

### Imputation

* Median Imputation
* KNNImputer
* IterativeImputer

### Scaling

* StandardScaler
* MinMaxScaler

### Encodage

* Label Encoding
* One-Hot Encoding

---
# Module 1 — Exploration, Nettoyage et Analyse des Données

## Objectif

Cette première étape consiste à comprendre en profondeur le dataset, identifier ses problèmes et préparer les données pour les étapes de Machine Learning.

---

## 1. Exploration des données

* Analyse de la structure du dataset
* Identification des types de variables (numériques et catégorielles)
* Étude des distributions des variables
* Analyse des corrélations entre features
* Détection des déséquilibres dans les données

---

## 2. Nettoyage des données

Les opérations réalisées sont :

* Suppression des doublons
* Détection et traitement des valeurs manquantes (NaN)
* Analyse des types de données manquantes :

  * MCAR (Missing Completely At Random)
  * MAR (Missing At Random)
  * MNAR (Missing Not At Random)

---

## 3. Gestion des valeurs manquantes

Comparaison de plusieurs méthodes :

* Imputation par la médiane
* KNNImputer
* IterativeImputer

L’objectif est de choisir la méthode la plus adaptée selon la distribution des données.

---

## 4. Détection des outliers

* Utilisation de la méthode IQR (Interquartile Range)
* Analyse des valeurs extrêmes
* Traitement ou suppression des anomalies

---

## 5. Normalisation et encodage

### Normalisation

* StandardScaler
* MinMaxScaler

### Encodage des variables catégorielles

* Label Encoding
* One-Hot Encoding

---

## Conclusion de ce module

Cette étape garantit :

* la qualité des données,
* la réduction du bruit,
* l’amélioration des performances des modèles,
* et la stabilité du pipeline ML.

Elle constitue une base essentielle pour les étapes suivantes :
Machine Learning, NLP et clustering.




# Module 2 — Machine Learning Supervisé

## Classification

Objectif :
Prédire la catégorie du déchet.

### Modèles comparés

* LogisticRegression
* RandomForest
* GradientBoosting
* SVM
* KNN

### Métriques

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Régression

Objectif :
Prédire le prix de revente.

### Modèles comparés

* CatBoost
* XGBoost
* RandomForestRegressor
* GradientBoostingRegressor
* KNN Regressor

### Métriques

* RMSE
* MAE
* R2
* MAPE

---

# Module 3 — Clustering

## Objectif

Segmenter automatiquement les déchets sans utiliser les labels.

## Étapes réalisées

1. Sélection des variables numériques
2. Normalisation
3. Elbow Method
4. KMeans
5. PCA
6. Visualisation 2D

## Résultats

| Élément                  | Résultat |
| ------------------------ | -------- |
| Algorithme               | KMeans   |
| Nombre de clusters       | 6        |
| Réduction dimensionnelle | PCA      |

---

# Module 4 — NLP

Le texte de `Rapport_Collecte` a été utilisé pour :

* nettoyer les descriptions,
* extraire des informations,
* classifier les déchets,
* enrichir le pipeline multimodal.

## Prétraitement NLP

* lowercase,
* suppression ponctuation,
* regex cleaning,
* tokenization,
* suppression stopwords,
* TF-IDF.

## Approches comparées

* Bag of Words
* TF-IDF
* Word2Vec
* FastText
* CamemBERT (bonus)

## Modèles NLP

* Naive Bayes
* Logistic Regression
* RandomForest
* LinearSVC

---

# Module 5 — Pipeline Multimodal

Fusion :

* données numériques,
* données textuelles.

## Techniques utilisées

* TF-IDF
* ColumnTransformer
* hstack
* sklearn Pipeline
* sparse matrix fusion

Objectif :
utiliser plusieurs types de données dans un pipeline unique.

---

# Module 6 — MLOps

## Fonctionnalités Implémentées

* Versionnement Git
* Versionnement DVC
* MLflow Tracking
* Pipeline DVC DAG
* FastAPI
* Docker
* GitHub Actions
* Pytest
* Evidently AI
* Prometheus
* Grafana

---

# Pipeline Global

```text
Raw Data
↓
Cleaning & Preprocessing
↓
Feature Engineering
↓
NLP + Données Numériques
↓
Fusion Multimodale
↓
Training
↓
MLflow Tracking
↓
Pipeline DVC
↓
Tests Automatisés
↓
FastAPI
↓
Docker
↓
Production
↓
Monitoring
↓
Retraining
```

---

# Installation

## 1. Cloner le Projet

```bash
git clone https://github.com/USERNAME/ECOML.git
cd ECOML
```

---

## 2. Créer un Environnement Virtuel

### Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Installer les Dépendances

```bash
pip install -r requirements.txt
```

---

# Rejouer le Pipeline en 3 Commandes

## 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 2. Exécuter le Pipeline DVC

```bash
dvc repro
```

## 3. Lancer l’API FastAPI

```bash
uvicorn api.main:app --reload
```

---

# Pipeline DVC

Le DAG DVC automatise :

```text
prepare_data.py
→ train.py
→ evaluate.py
→ monitor.py
```

Commande :

```bash
dvc repro
```

---

# MLflow Tracking

## Lancer MLflow

```bash
mlflow ui
```

Interface :

```text
http://127.0.0.1:5000
```

## Expériences Comparées

| Expérience | Modèle             |
| ---------- | ------------------ |
| Exp 1      | RandomForest       |
| Exp 2      | LogisticRegression |
| Exp 3      | SVM                |
| Exp 4      | GradientBoosting   |
| Exp 5      | XGBoost            |

---

# Tests Automatisés

## Exécution des Tests

```bash
pytest --cov=src --cov=api tests/
```

## Vérifications

* schéma dataset,
* gestion NaN,
* pipeline NLP,
* vectorisation TF-IDF,
* accuracy minimale,
* endpoints API.

## Couverture

```text
Coverage ≥ 70%
```

---

# API REST — FastAPI

## Lancer l’API

```bash
uvicorn api.main:app --reload
```

## Documentation Swagger

```text
http://127.0.0.1:8000/docs
```

## Endpoints

| Endpoint                | Description           |
| ----------------------- | --------------------- |
| /                       | Test API              |
| /predict/classification | Classification        |
| /predict/regression     | Régression            |
| /predict/nlp            | NLP                   |
| /dashboard/clusters     | Dashboard PCA         |
| /metrics                | Monitoring Prometheus |

---

# Application Web

L’application web intelligente contient :

## Dashboard Data

* visualisation dataset,
* PCA,
* clustering KMeans.

## Prédiction Manuelle

* sliders interactifs,
* prédiction temps réel,
* estimation du prix.

## Assistant Intelligent NLP

* saisie texte,
* classification automatique.

---

# Docker

## Construire l’Image

```bash
docker build -t eco-smart-api .
```

## Lancer le Container

```bash
docker run -p 8000:8000 eco-smart-api
```

## Vérifier les Containers

```bash
docker ps
```

---

# Monitoring Drift — Evidently AI

## Générer le Rapport

```bash
python src/monitor.py
```

Le rapport HTML permet :

* détection de drift,
* comparaison datasets,
* distributions statistiques,
* scores de dérive.

---

# Monitoring — Prometheus & Grafana

## Prometheus

Endpoint :

```text
http://127.0.0.1:8000/metrics
```

Interface :

```text
http://localhost:9090
```

## Grafana

Interface :

```text
http://localhost:3000
```

## Métriques surveillées

* nombre de requêtes,
* activité API,
* statistiques système,
* monitoring temps réel.

---

# GitHub Actions — CI/CD

Le pipeline CI/CD exécute automatiquement :

* Black,
* isort,
* flake8,
* Pytest,
* Coverage,
* Docker Build.

Objectifs :

* validation automatique,
* détection rapide des erreurs,
* automatisation du déploiement.

---

# Logging JSON

Toutes les prédictions API sont sauvegardées dans :

```text
logs/predictions.jsonl
```

Informations enregistrées :

* timestamp,
* endpoint,
* données utilisateur,
* prédiction,
* méthode utilisée.

---

# Technologies Utilisées

* Python
* Pandas
* NumPy
* Scikit-learn
* MLflow
* DVC
* FastAPI
* Docker
* Pytest
* Evidently AI
* Prometheus
* Grafana
* GitHub Actions
* Matplotlib
* Regex
* NLP

---

# Conclusion

Le projet **Eco-Smart Classifier** met en œuvre un pipeline MLOps complet permettant :

* la reproductibilité,
* l’automatisation,
* le suivi des expériences,
* le déploiement,
* et le monitoring des modèles.

L’intégration de Git, DVC, MLflow, FastAPI, Docker, Evidently AI, Prometheus et Grafana transforme ce projet en une solution Machine Learning industrialisable et prête pour la production.
