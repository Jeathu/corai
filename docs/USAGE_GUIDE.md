# 🚀 Guide d'Utilisation du Projet CorAI

## 📋 Vue d'Ensemble

Projet de prédiction de maladies cardiaques avec Machine Learning.

---

## 🎯 Workflows Disponibles

### **Option 1: Pipeline Complet (RECOMMANDÉ) 🌟**

Execute tout le processus en une seule commande:

```bash
# Pipeline basique avec Random Forest
python -m corai.pipeline_complete



# Avec Logistic Regression
python -m corai.pipeline_complete --model-type logistic_regression

# Avec optimisation d'hyperparamètres (plus lent mais meilleur)
python -m corai.pipeline_complete --use-grid-search

# Avec tous les paramètres
python -m corai.pipeline_complete \
    --model-type logistic_regression \
    --use-grid-search \
    --test-size 0.25 \
    --cv-folds 10
```

**Ce que fait le pipeline complet:**
1. Charge les données brutes
2. Prétraitement (encodage, normalisation)
3. Séparation train/test
4. Entraînement du modèle
5. Prédictions sur le test
6. Évaluation des performances
7. Sauvegarde de tous les résultats

---

### **Option 2: Démonstration Rapide 🎬**

Teste le modèle déjà entraîné:

```bash
python -m corai.demo_model
```

**Prérequis:** Avoir déjà exécuté le pipeline ou l'entraînement une fois.

---

### **Option 3: Étape par Étape 🔧**

Pour plus de contrôle, executez chaque étape séparément:

#### **1️⃣ Prétraitement des données**
```bash
python -m corai.preprocessing.preprocessing_pipeline
```
- Input: `data/raw/heart_disease_dataset.csv`
- Output: `data/processed/processed_heart_disease_v0.csv`

#### **2️⃣ Entraînement du modèle**
```bash
# Random Forest (défaut)
python -m corai.modeling.train

# Avec recherche d'hyperparamètres
python -m corai.modeling.train --use-grid-search

# Autres modèles disponibles
python -m corai.modeling.train --model-type gradient_boosting
python -m corai.modeling.train --model-type logistic_regression
python -m corai.modeling.train --model-type svm
```
- Input: `data/processed/processed_heart_disease_v0.csv`
- Output: `models/heart_disease_model.pkl`

#### **3️⃣ Prédiction (nécessite les fichiers du pipeline complet)**
```bash
python -m corai.modeling.predict
```
- Input: `data/processed/test_features.csv`, `models/heart_disease_model.pkl`
- Output: `data/processed/test_predictions.csv`

 **Note:** Cette commande nécessite que vous ayez d'abord exécuté le pipeline complet.

#### **4️⃣ Évaluation (nécessite les fichiers du pipeline complet)**
```bash
python -m corai.modeling.evaluate
```
- Input: `data/processed/test_predictions.csv`, `data/processed/test_labels.csv`
- Output: `reports/evaluation_metrics.json`

---

## 📊 Analyse Exploratoire des Données

### **Visualisations complètes**
```bash
python -m corai.analytics.visualizations.raw_data_visualizations
```
- Génère des graphiques PNG
- Crée un rapport HTML
- Output: `reports/figures/raw_data_png/raw_data_report.html`

### **Analyse statistique**
```bash
python -m corai.analytics.synthese_variables
```

### **EDA complète**
```bash
python -m corai.analytics.eda
```
- Output: `reports/eda_report.txt`

---

## 📁 Fichiers Générés

### **Après le pipeline complet:**

```
data/processed/
  ├── processed_heart_disease_complete.csv  # Données prétraitées complètes
  ├── train_features.csv                    # Features d'entraînement
  ├── train_labels.csv                      # Labels d'entraînement
  ├── test_features.csv                     # Features de test
  ├── test_labels.csv                       # Labels de test
  └── test_predictions.csv                  # Prédictions sur le test

models/
  └── random_forest_heart_disease.pkl       # Modèle entraîné

reports/
  ├── random_forest_evaluation_metrics.json # Métriques du modèle
  └── demo_results/
      ├── predictions_demo.csv              # Résultats de la démo
      └── metrics_demo.json                 # Métriques de la démo
```

---

## 🎓 Exemples de Workflows

### **Workflow 1: Démarrage Rapide**
```bash
# 1. Exécuter le pipeline complet
python -m corai.pipeline_complete

# 2. Voir les visualisations
python -m corai.analytics.visualizations.raw_data_visualizations

# 3. Ouvrir le rapport HTML
start reports/figures/raw_data_png/raw_data_report.html  # Windows
```

### **Workflow 2: Comparaison de Modèles**
```bash
# Random Forest
python -m corai.pipeline_complete --model-type random_forest

# Gradient Boosting
python -m corai.pipeline_complete --model-type gradient_boosting

# Logistic Regression
python -m corai.pipeline_complete --model-type logistic_regression

# Comparer les métriques dans reports/
```

### **Workflow 3: Optimisation**
```bash
# Avec GridSearchCV pour trouver les meilleurs hyperparamètres
python -m corai.pipeline_complete --use-grid-search --cv-folds 10

# Attention: peut prendre plusieurs minutes
```

---

## 🔍 Commandes de Diagnostic

### **Vérifier les données**
```powershell
# Voir les données brutes
python -c "import pandas as pd; print(pd.read_csv('data/raw/heart_disease_dataset.csv').info())"

# Voir les données prétraitées
python -c "import pandas as pd; print(pd.read_csv('data/processed/processed_heart_disease_v0.csv').info())"

# Vérifier le modèle
python -c "import pickle; m = pickle.load(open('models/heart_disease_model.pkl', 'rb')); print(type(m))"
```

### **Vérifier l'installation**
```powershell
# Packages Python
pip list | findstr "pandas scikit-learn"

# Version Python
python --version
```

---

## 🐛 Résolution de Problèmes

### **Erreur: "No such file or directory: test_features.csv"**
**Solution:** Utilisez le pipeline complet ou la démo:
```bash
python -m corai.pipeline_complete
# OU
python -m corai.demo_model
```

### **Erreur: "Module not found"**
**Solution:** Installez les dépendances:
```bash
pip install -r requirements.txt
```

### **Erreur: "Model not found"**
**Solution:** Entraînez d'abord un modèle:
```bash
python -m corai.modeling.train
```

---

## 📊 Métriques d'Évaluation

Le projet calcule:
- **Accuracy**: Précision globale
- **Precision**: Précision par classe
- **Recall**: Rappel par classe
- **F1 Score**: Moyenne harmonique de précision et rappel
- **ROC AUC**: Aire sous la courbe ROC
- **Confusion Matrix**: Matrice de confusion

---

## 🎯 Modèles Disponibles

| Modèle | Commande | Vitesse | Précision |
|--------|----------|---------|-----------|
| Random Forest | `--model-type random_forest` | ⚡⚡ | ⭐⭐⭐ |
| Gradient Boosting | `--model-type gradient_boosting` | ⚡ | ⭐⭐⭐⭐ |
| Logistic Regression | `--model-type logistic_regression` | ⚡⚡⚡ | ⭐⭐ |
| SVM | `--model-type svm` | ⚡ | ⭐⭐⭐ |

---

## 💡 Conseils

1. **Première utilisation:** Commencez par le pipeline complet
2. **Expérimentation:** Testez différents modèles avec `--model-type`
3. **Optimisation:** Utilisez `--use-grid-search` pour les meilleurs résultats
4. **Visualisation:** Toujours vérifier les données avec les outils d'analyse

---

## 📚 Documentation Complète

- **Analyse du projet:** Voir `PROJECT_ANALYSIS.md`
- **Configuration:** Voir `corai/config.py`

---

## ✅ Checklist Rapide

- [ ] Installer les dépendances: `pip install -r requirements.txt`
- [ ] Vérifier les données: `data/raw/heart_disease_dataset.csv` existe
- [ ] Exécuter le pipeline: `python -m corai.pipeline_complete`
- [ ] Vérifier les résultats: Fichiers dans `models/` et `reports/`
- [ ] Tester le modèle: `python -m corai.demo_model`
