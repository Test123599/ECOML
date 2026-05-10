# PROMPTS.md
## Projet : Eco-Smart Classifier
### Réalisé par : Emna Moalla & Chayma Mkaouar

---

# Introduction

Ce document présente les principaux axes de réflexion et les idées utilisées lors du développement du projet **Eco-Smart Classifier**.
Les outils d’intelligence artificielle ont été utilisés comme support d’aide à la compréhension, à la structuration des idées et à l’exploration de certaines approches techniques.

Ils ont permis notamment :
- d’explorer différentes méthodes de preprocessing,
- de comprendre certains concepts de Machine Learning et NLP,
- d’aider à structurer le pipeline global du projet,
- et de clarifier certaines étapes du MLOps.

Toutes les implémentations, les choix finaux et les intégrations ont été réalisés par les membres du projet.

---

# 1. Génération et compréhension du Dataset

- Explorer les caractéristiques d’un dataset de classification des déchets.
- Comprendre les variables physiques (poids, volume, conductivité, etc.).
- Étudier les problèmes possibles : valeurs manquantes, bruit, incohérences.
- Simuler des cas réalistes pour comprendre le comportement des données.

---

# 2. Prétraitement des Données

- Comprendre les étapes de nettoyage des données.
- Étudier la gestion des valeurs manquantes (NaN).
- Comparer différentes méthodes d’imputation (médiane, KNN, iterative).
- Comprendre l’impact du scaling (StandardScaler, MinMaxScaler).
- Explorer la détection des outliers (IQR, IsolationForest).
- Structurer un pipeline de preprocessing avec sklearn.

---

# 3. Analyse Exploratoire des Données (EDA)

- Visualiser la distribution des variables.
- Étudier les relations entre les features.
- Comprendre les corrélations entre variables.
- Explorer la réduction de dimension avec PCA.
- Observer la structure globale des données.

---

# 4. Machine Learning Supervisé

## Classification
- Étudier différents algorithmes de classification :
  RandomForest, SVM, LogisticRegression, XGBoost, KNN.
- Comprendre les métriques d’évaluation :
  Accuracy, Precision, Recall, F1-score.
- Analyser les matrices de confusion.

## Régression
- Explorer les modèles de régression :
  LinearRegression, RandomForestRegressor, XGBoost, CatBoost.
- Comprendre les métriques :
  R2, RMSE, MAE, MAPE.
- Comparer les performances des modèles.

---

# 5. NLP (Traitement du texte)

- Comprendre le nettoyage de texte (lowercase, regex, stopwords).
- Étudier la vectorisation TF-IDF.
- Explorer Bag of Words et autres représentations textuelles.
- Tester des modèles de classification textuelle (LinearSVC, Naive Bayes).
- Analyser les performances du pipeline NLP.

---

# 6. Pipeline Multimodal

- Comprendre la fusion entre données numériques et textuelles.
- Étudier l’utilisation de TF-IDF avec features numériques.
- Explorer ColumnTransformer et hstack.
- Analyser l’intérêt d’un modèle multimodal.

---

# 7. SHAP & Explicabilité

- Comprendre l’interprétation des modèles ML.
- Étudier l’importance des features avec SHAP.
- Visualiser les contributions des variables.
- Analyser les décisions des modèles.

---

# 8. FastAPI

- Comprendre la création d’une API REST.
- Structurer des endpoints pour classification, régression et NLP.
- Charger des modèles pré-entraînés.
- Tester les requêtes API.

---

# 9. Docker

- Comprendre la conteneurisation d’une application.
- Créer un environnement reproductible avec Docker.
- Tester le déploiement local de l’API.

---

# 10. Git & DVC

- Comprendre le versionnement du code avec Git.
- Explorer le rôle de DVC pour les données et modèles.
- Structurer un pipeline reproductible.
- Comprendre le concept de DAG dans MLOps.

---

# 11. MLflow

- Comprendre le suivi des expériences ML.
- Enregistrer les métriques et paramètres.
- Comparer différents modèles.
- Analyser les résultats des expérimentations.

---

# 12. Tests & Pytest

- Comprendre l’importance des tests dans un projet ML.
- Tester les données et les modèles.
- Vérifier le bon fonctionnement de l’API.
- Assurer la stabilité du pipeline.

---

# 13. CI/CD (GitHub Actions)

- Comprendre l’automatisation des tests.
- Exécuter le linting et la validation du code.
- Automatiser la vérification du projet.

---

# 14. Monitoring & Observabilité

- Comprendre le suivi des modèles en production.
- Explorer le data drift et text drift.
- Étudier Prometheus et Grafana.
- Observer les performances du système.

---

# Conclusion

Les outils d’intelligence artificielle ont été utilisés comme support pédagogique afin d’aider à :
- comprendre certaines notions techniques,
- structurer les idées du projet,
- explorer différentes approches méthodologiques.

Les implémentations finales, les choix techniques et l’intégration du système ont été réalisés par les membres du projet Emna et Chayma de manière autonome.