# PROMPTS.md
## Projet : Eco-Smart Classifier
### Réalisé par : Emna Moalla & Chayma Mkaouar

---

# Introduction

Ce document contient l’ensemble des prompts utilisés durant le développement du projet Eco-Smart Classifier.

Les outils d’intelligence artificielle ont été utilisés pour :
- la génération de datasets,
- le preprocessing,
- la création des modèles ML,
- le NLP,
- le pipeline multimodal,
- l’explicabilité SHAP,
- le MLOps,
- le déploiement,
- la documentation,
- et les visualisations.

---

# 1. Génération Dataset

## Prompt 1

Générer une dataset réaliste pour un projet Eco-Smart Classifier contenant :
Poids, Volume, Conductivite, Opacite, Rigidite, Source, Categorie.
Ajouter des NaN, des outliers et des valeurs bruitées mais cohérentes.

Régénérer cette dataset avec des valeurs plus réalistes pour la classification et la régression.

Corriger la colonne Conductivite et rendre les données plus cohérentes.

Créer une dataset écologique réaliste pour le recyclage intelligent.

Créer une dataset contenant des anomalies, valeurs manquantes et bruit.

# 2. Prétraitement des Données

Donner un pipeline complet de nettoyage des données :
- gestion des NaN,
- suppression des doublons,
- traitement des outliers,
- encodage,
- normalisation.


Expliquer StandardScaler et MinMaxScaler avec exemples Python.

Créer un pipeline sklearn de preprocessing complet.

Détecter les outliers avec IQR et IsolationForest.

Créer un preprocessing pipeline avec ColumnTransformer.

# 3.Analyse Exploratoire des Données (EDA)

Créer une analyse exploratoire complète avec matplotlib et seaborn.
Créer les heatmaps, histogrammes, boxplots et pairplots.
Analyser les corrélations entre les variables.
Créer une PCA pour visualiser les données.
Créer une analyse clustering avec KMeans.

# 4.Classification Machine Learning

Créer un modèle RandomForestClassifier complet avec évaluation.
Créer un modèle XGBoost pour classification.
Créer un modèle CatBoostClassifier.
Créer un modèle CatBoostClassifier.
Comparer Accuracy, Precision, Recall et F1-score.
Créer une matrice de confusion avec visualisation.
Créer une courbe ROC et expliquer son rôle.
Créer une courbe ROC et expliquer son rôle.

# 5. Régression Machine Learning

Créer un modèle LinearRegression complet.

Créer un modèle RandomForestRegressor.

Créer un modèle GradientBoostingRegressor.

Créer un modèle XGBRegressor.

Créer un modèle CatBoostRegressor.

Comparer :
- R2
- MAE
- RMSE
- MAPE
- MEDAE

Créer des graphiques comparatifs pour les modèles de régression.

Expliquer les résultats du modèle CatBoost.

---

# 6. NLP

Créer un pipeline NLP avec TF-IDF + LinearSVC.

Nettoyer des données textuelles avec NLP.

Créer une vectorisation TF-IDF complète.

Créer un pipeline NLP sklearn complet.

Classifier des descriptions de déchets avec NLP.

Évaluer le modèle NLP avec :
- Accuracy
- Recall
- Precision
- F1-score

---

# 7. Pipeline Multimodal

Fusionner les modèles :
- régression
- NLP

dans un pipeline multimodal.

Créer un pipeline multimodal intelligent pour Eco-Smart Classifier.

Créer des graphiques pour analyser les performances multimodales.

Créer des visualisations matplotlib pour le pipeline multimodal.

---

# 8. SHAP & Explicabilité

Créer une analyse SHAP complète pour le modèle CatBoost.

Créer un SHAP summary plot.

Créer un SHAP force plot.

Créer un SHAP dependence plot.

Expliquer l’importance des features avec SHAP.

Interpréter les résultats SHAP du pipeline multimodal.

---

# 9. FastAPI

Créer une API FastAPI complète pour les modèles ML.

Créer un endpoint de classification.

Créer un endpoint de régression.

Créer un endpoint NLP.

Charger des modèles .pkl avec joblib dans FastAPI.

Créer une documentation Swagger automatique.

---

# 10. Docker

Créer un Dockerfile pour FastAPI.

Créer un docker-compose.yml complet.

Expliquer les commandes Docker :
- build
- run
- logs
- inspect

Corriger les erreurs Docker sous Windows.

Corriger les problèmes de volumes Docker.

Corriger les conflits de ports Docker.

---

# 11. Git & DVC

Expliquer le rôle de DVC dans un pipeline MLOps.

Créer un pipeline DVC complet.

Créer un fichier dvc.yaml.

Créer un DAG DVC reproductible.

Utiliser Git et DVC ensemble sans stocker les datasets dans Git.

Créer un .gitignore adapté à un projet ML.

Corriger les erreurs Git liées aux fichiers déjà trackés.

---

# 12. MLflow

Expliquer le rôle de MLflow.

Créer plusieurs expériences MLflow.

Logger les métriques et paramètres avec MLflow.

Utiliser MLflow Model Registry.

Créer un tracking MLflow complet.

---

# 13. Pytest & Tests

Créer des tests Pytest pour les modèles ML.

Tester le schéma des données.

Tester la qualité des données après preprocessing.

Tester les prédictions du modèle.

Tester l’API FastAPI.

Créer un rapport de couverture Pytest.

---

# 14. GitHub Actions & CI/CD

Créer un pipeline GitHub Actions pour MLOps.

Ajouter Black, Flake8 et Isort dans GitHub Actions.

Automatiser les tests avec GitHub Actions.

Automatiser le build Docker.

Créer un workflow CI/CD complet.

---

# 15. Monitoring & Observabilité

Configurer Prometheus avec FastAPI.

Configurer Grafana pour le monitoring ML.

Créer des métriques Prometheus personnalisées.

Créer un dashboard Grafana.



# Conclusion

Les outils d’intelligence artificielle ont été utilisés comme assistance durant :
- la génération de code,
- le preprocessing,
- le Machine Learning,
- le NLP,
- le pipeline multimodal,
- le MLOps,
- le déploiement,
- les visualisations,
- et la documentation.

Toutes les validations finales, décisions techniques et intégrations ont été réalisées manuellement par les membres du projet.