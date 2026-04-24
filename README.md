#  Eco-Smart Classifier

## Description

Ce projet consiste à développer un pipeline complet de Machine Learning permettant de classifier des déchets et d’estimer leur valeur de revente.

Le projet couvre :

* Le nettoyage et prétraitement des données
* L’analyse non supervisée (clustering)
* L’extraction d’informations à partir de texte (NLP)
* La préparation pour des modèles ML

---

## Dataset

Le dataset contient 9 colonnes :

* **Poids**
* **Volume**
* **Conductivite**
* **Opacite**
* **Rigidite**
* **Prix_Revente**
* **Source** (catégorielle)
* **Rapport_Collecte** (texte)
* **Categorie** (target)

### Problèmes détectés :

* Valeurs manquantes (NaN)
* Outliers
* Données textuelles non structurées

---

## Prétraitement des données

Les étapes réalisées :

* Suppression des doublons
* Gestion des valeurs manquantes :

  * Remplacement par la médiane
  * Utilisation du texte (NLP) pour compléter certaines valeurs
* Traitement des outliers
* Normalisation avec **StandardScaler**
* Encodage des variables catégorielles

---
## ⚙️ Modules

### 🧹 Module 1 : Data Engineering
- Nettoyage des données
- Gestion des NaN (médiane, KNN Imputer, Iterative Imputer)
- Traitement des outliers
- Normalisation des données

### 🤖 Module 2 : Machine Learning Supervisé
- Classification de la catégorie
- Régression du prix de revente

### 🔍 Module 3 : Clustering
- KMeans pour segmentation des données
- Méthode du coude (Elbow Method)
- Visualisation avec PCA
- Évaluation avec Silhouette Score

### 🧠 Module 4 : NLP
- Nettoyage du texte
- Extraction d’informations depuis Rapport_Collecte
- Vectorisation (TF-IDF)

### 🔗 Module 5 : Pipeline Multimodal
- Fusion des données numériques et textuelles

### 🚀 Module 6 : MLOps
- Versioning avec DVC
- Tracking avec MLflow
- API avec FastAPI


## Clustering (Module 3)

### Objectif

Segmenter les données sans utiliser la variable cible (`Categorie`).

---

### Étapes

1. Sélection des variables numériques :

   * Poids
   * Volume
   * Conductivite
   * Opacite
   * Rigidite

2. Gestion des valeurs manquantes :

   * Remplacement par la médiane

3. Normalisation :

   * StandardScaler

4. Détermination du nombre optimal de clusters :

   * Méthode du coude (Elbow Method)

5. Application de KMeans

6. Évaluation :

   * Silhouette Score

7. Visualisation :

   * Réduction de dimension avec PCA

---

###  Résultats

* Nombre de clusters choisi : **3**
* Méthode utilisée : **KMeans**
* Visualisation : **PCA (2D)**

---

###  Interprétation des clusters

* **Cluster 0** : matériaux lourds (ex: verre)
* **Cluster 1** : matériaux légers (ex: plastique)
* **Cluster 2** : matériaux intermédiaires (ex: papier)

---

##  NLP (Rapport_Collecte)

Le texte a été exploité pour :

* Extraire des informations (volume, type de déchet)
* Compléter certaines valeurs manquantes
* Nettoyer le texte (minuscule, suppression ponctuation)

---

## Technologies utilisées

* Python
* Pandas / NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Regex (NLP simple)

---

## Exécution

```bash
pip install -r requirements.txt
python main.py
```

Ou via Jupyter Notebook :

```bash
jupyter notebook
```

---

## Résultats globaux

* Clustering cohérent avec les caractéristiques physiques des matériaux
* Amélioration de la qualité des données grâce au NLP
* Pipeline prêt pour extension vers classification

---

## Auteur

* **Chayma et Emna **
* DSIR

---
