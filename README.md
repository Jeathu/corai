# CorAI - Prédiction de Maladies Cardiaques par Machine Learning

**Auteur :** Jeathusan 
**Date :** 04 Décembre 2025
**GitHub :** [github.com/Jeathu/corai](https://github.com/Jeathu/corai)

---

## 1. Introduction

### Contexte

Les maladies cardiovasculaires sont la **première cause de mortalité mondiale** (17.9 millions de décès/an). Le diagnostic repose sur des examens coûteux et l'expertise de spécialistes peu disponibles.

### Objectif

Développer un système ML de prédiction du risque cardiaque à partir de 16 variables cliniques simples.

| Objectif | Cible | Résultat |
|----------|-------|----------|
| Accuracy | ≥ 85% | ✅ **99%** |
| F1-Score | ≥ 0.85 | ✅ **0.99** |
| Architecture extensible | Factory Pattern | ✅ |

---

## 2. Les Données

**Fichier :** `data/raw/heart_disease_dataset.csv`  
**Taille :** 1000 patients × 17 colonnes

### Variables

| Type | Nombre | Exemples |
|------|--------|----------|
| **Numériques** | 7 | Age, Cholesterol, Blood Pressure, Heart Rate |
| **Catégorielles** | 9 | Gender, Smoking, Diabetes, Chest Pain Type |
| **Target** | 1 | Heart Disease (0/1) |

### Équilibre des Classes

```
Sains   : 608 (60.8%)
Malades : 392 (39.2%)
Ratio 1.55:1 → Équilibré 😁 (pas besoin de SMOTE)
```


---

## 3. Prétraitement

### Pourquoi c'est crucial ?

| Modèle | Sans prétraitement | Avec prétraitement |
|--------|-------------------|-------------------|
| SVM | 65% | **85%** (+20%) |
| Logistic Regression | 78% | **86%** (+8%) |

### Architecture en 3 Modules


![Architecture du preprocessing](../corai/corai/preprocessing/doc/image/preprocess.png)


**Intérêts :**
- **Maintenabilité** : modifier une étape sans toucher aux autres
- **Testabilité** : tests unitaires par module
- **Scalabilité** : ajout facile de nouvelles transformations

### Transformations Appliquées

#### Encodage : Transformation des variables catégorielles en variables numériques (One-Hot Encoding scikit-learn)

```python
# Avec encodage One-Hot : Male→[1,0], Female→[0,1]
```


#### Mise à l'échelle : Standardisation des caractéristiques numériques

```
X_normalisé = (X - moyenne) / écart-type
```

| Variable | Avant | Après |
|----------|-------|-------|
| Age = 45 | 45 | -0.85 |
| Cholesterol = 200 | 200 | -0.71 |

### Data Leakage - Précaution Critique

```python
# ❌ ERREUR : scaler.fit(X_complet)
# ✅ CORRECT : scaler.fit(X_train) puis transform(X_test)
```

---

## 4. Modèles

![Architecture du modèle](../corai/corai/preprocessing/doc/image/models_sc.png)

### Stratégie Multi-Modèles

| Modèle | Type | Force |
|--------|------|-------|
| Logistic Regression | Linéaire | Interprétable, baseline |
| Random Forest | Bagging | Robuste, feature importance |
| Gradient Boosting | Boosting | Très performant |
| SVM | Kernel | Haute dimension |

### Random Forest - Choix Principal

**Pourquoi ?**
1. Vote de 100 arbres → robuste
2. Feature importance → interprétable
3. Gère les interactions non-linéaires

### Validation Croisée Stratifiée 5-Fold

```
Fold 1: [VAL][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2: [TRAIN][VAL][TRAIN][TRAIN][TRAIN]
...
→ 5 évaluations, score = moyenne
```

Préserve le ratio 60/40 dans chaque fold.

---

## 5. Résultats

### Performances

| Modèle | Accuracy | F1-Score | ROC AUC |
|--------|----------|----------|---------|
| Logistic Regression | 86% | 0.859 | 0.951 |
| SVM | 85% | 0.850 | 0.920 |
| **Random Forest** | **99%** | **0.990** | **1.000** |
| Gradient Boosting | 100% | 1.000 | 1.000 |

### Tableau comparatif des modèles testés :
![Architecture du modèle](../corai/corai/preprocessing/doc/image/tableau_comp.png)

### Matrice de Confusion (Random Forest)

```
              Prédit 0  Prédit 1
Réel 0          122        0
Réel 1            2       76

Erreurs : 2/200 (1%)
```

### Features Importantes

```
1. Cholesterol      (0.18)
2. Age              (0.15)
3. Blood Pressure   (0.13)
4. Exercise Hours   (0.11)
5. Chest Pain Type  (0.09)
```

→ Conforme à la littérature médicale ✓

### Pourquoi Random Forest et pas Gradient Boosting (100%) ?

- 100% sur 1000 échantillons = suspect (surapprentissage)
- Random Forest : meilleure généralisation

---

## 6. Architecture de fichier source 

```
corai/
├── preprocessing/
│   ├── data_loader.py           # Chargement
│   ├── feature_transformer.py   # Transformation
│   └── preprocessing_pipeline.py # Orchestration
│
├── modeling/
│   ├── abstraite_base_model.py  # Classe abstraite
│   ├── model_factory.py         # Factory Pattern
│   ├── train.py / predict.py / evaluate.py
│   └── classifiers/             # RF, GB, LR, SVM
│
├── analytics/                   # EDA, visualisations
├── pipeline_complete.py         # Pipeline end-to-end
└── config.py                    # Configuration centralisée
```

### Factory Pattern

```python
model = ModelFactory.create("random_forest", n_estimators=200)
model = ModelFactory.create("gradient_boosting")
# → Extensible : ajouter XGBoost = 1 fichier + 1 ligne
```

---

## 7. Utilisation

### Option 1 : Pipeline Complet (Recommandé)

```bash
# Installation
git clone https://github.com/Jeathu/corai.git
cd corai && pip install -r requirements.txt

# Prétraitement des données
python -m corai.preprocessing.preprocessing_pipeline
# Input: data/raw/heart_disease_dataset.csv
# Output: data/processed/processed_heart_disease_v0.csv

# Exécution du pipeline complet
python -m corai.pipeline_complete
```

### Option 2 : Entraînement, Prédiction et Évaluation Séparés

> ⚠️ **Prérequis :** Vous devez être dans l'environnement virtuel (suivre les étapes de `instruction_conf.txt`)

#### Entraînement du modèle

```bash
# Random Forest (défaut)
python -m corai.modeling.train

# Autres modèles disponibles
python -m corai.modeling.train --model-type gradient_boosting
python -m corai.modeling.train --model-type logistic_regression
python -m corai.modeling.train --model-type svm
```

- **Input :** `data/processed/processed_heart_disease_v0.csv`
- **Output :** `models/heart_disease_model.pkl`

#### Prédiction

```bash
python -m corai.modeling.predict
```

- **Input :** `data/processed/test_features.csv`, `models/heart_disease_model.pkl`
- **Output :** `models/predictions/test_predictions.csv`

> ⚠️ **Note :** Cette commande nécessite que vous ayez d'abord exécuté le pipeline complet.

#### Évaluation

```bash
python -m corai.modeling.evaluate
```

- **Input :** `data/processed/test_predictions.csv`, `data/processed/test_labels.csv`
- **Output :** `reports/evaluation_metrics.json`

---

## 8. Conclusion

### Réalisations

| Objectif | Statut |
|----------|--------|
| Accuracy ≥ 85% | ✅ 99% |
| Architecture extensible | ✅ Factory + Pipeline |
| Validation robuste | ✅ CV 5-fold |

### Points Forts

- **99% accuracy** avec 2 erreurs sur 200 patients
- **Architecture modulaire** (responsabilité unique, design patterns)
- **Feature importance** cohérente avec la médecine

### Améliorations Futures

- Ajouter XGBoost, SHAP
- API REST (FastAPI)
- Docker

---

**04 Décembre 2025 - Module d'Apprentissage Artificiel**