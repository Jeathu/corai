"""
Script de démonstration pour tester le modèle existant.
Charge le modèle déjà entraîné et fait des prédictions sur les données traitées.
"""

from pathlib import Path
import pickle
import joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from loguru import logger
import typer

from corai.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR,
    DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE, TARGET_COLUMN
)
from corai.preprocessing.data_loader import load_data, split_features_target
from corai.modeling.evaluate import ModelEvaluator

app = typer.Typer()


@app.command()
def demo(
    data_path: Path = PROCESSED_DATA_DIR / "processed_heart_disease_v0.csv",
    model_path: Path = MODELS_DIR / "heart_disease_model.pkl",
    test_size: float = None,
    random_state: int = None,
):
    """
    Démonstration simple: charge le modèle existant et fait des prédictions.
    
    Args:
        data_path: Chemin vers les données prétraitées
        model_path: Chemin vers le modèle entraîné
        test_size: Proportion pour le test (None = utilise DEFAULT_TEST_SIZE)
        random_state: Graine aléatoire (None = utilise DEFAULT_RANDOM_STATE)
    """
    test_size = test_size or DEFAULT_TEST_SIZE
    random_state = random_state or DEFAULT_RANDOM_STATE
    
    logger.info("=" * 80)
    logger.info("🎬 DÉMONSTRATION DU MODÈLE EXISTANT")
    logger.info("=" * 80)
    
    # 1. Charger les données
    logger.info("Chargement des données prétraitées...")
    df = load_data(data_path)
    logger.info(f"Données chargées: {df.shape}")
    
    # 2. Séparer features et target
    logger.info("Séparation features/target...")
    X, y = split_features_target(df, target=TARGET_COLUMN)
    logger.info(f"Features: {X.shape}, Target: {y.shape}")
    
    # 3. Split train/test
    logger.info(f"Séparation train/test ({int((1-test_size)*100)}/{int(test_size*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 4. Charger le modèle
    logger.info(f"Chargement du modèle depuis: {model_path}")
    
    if not model_path.exists():
        logger.error(f"Modèle non trouvé: {model_path}")
        logger.info("Exécutez d'abord: python -m corai.modeling.train")
        return
    
    # Essayer de charger avec joblib d'abord (nouveau format BaseModel)
    try:
        model_data = joblib.load(model_path)
        
        # Format BaseModel: dict avec 'model', 'metadata', 'model_type'
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            model_type = model_data.get('model_type', 'unknown')
            metadata = model_data.get('metadata', {})
            logger.info(f"Modèle BaseModel chargé: {model_type} (entraîné le {metadata.get('training_date', 'N/A')})")
        else:
            model = model_data
            logger.info("Modèle sklearn chargé (joblib)")
    
    except Exception as e_joblib:
        # Essayer pickle (ancien format)
        logger.debug(f"Erreur joblib: {e_joblib}, tentative avec pickle...")
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                model = model_data.get("model")
                model_type = model_data.get("model_type", "unknown")
                logger.info(f"Modèle chargé (pickle): {model_type}")
            else:
                model = model_data
                logger.info("Modèle chargé (format ancien pickle)")
        except Exception as e_pickle:
            logger.error(f"Impossible de charger le modèle avec joblib ou pickle")
            raise RuntimeError(f"Erreur de chargement: joblib={e_joblib}, pickle={e_pickle}")
    
    # 5. Faire des prédictions
    logger.info("Prédiction sur le test set...")
    y_pred = model.predict(X_test)
    
    # Calculer les probabilités si disponible
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        logger.info("Probabilités calculées")
    
    # 6. Évaluer les performances
    logger.info("=" * 80)
    logger.info("📊 RÉSULTATS DE L'ÉVALUATION")
    logger.info("=" * 80)
    
    evaluator = ModelEvaluator(task_type="classification")
    metrics = evaluator.evaluate_classification(
        y_true=y_test.values,
        y_pred=y_pred,
        y_proba=y_proba
    )
    
    evaluator.print_metrics()
    
    # 7. Afficher quelques exemples de prédictions
    logger.info("\n" + "=" * 80)
    logger.info("🔍 EXEMPLES DE PRÉDICTIONS (10 premiers)")
    logger.info("=" * 80)
    
    results_df = pd.DataFrame({
        "Vrai": y_test.values[:10],
        "Prédit": y_pred[:10],
        "Correct": y_test.values[:10] == y_pred[:10]
    })
    
    if y_proba is not None:
        results_df["Prob_0"] = y_proba[:10, 0]
        results_df["Prob_1"] = y_proba[:10, 1]
    
    logger.info("\n" + results_df.to_string())
    
    # 8. Sauvegarder les résultats
    output_dir = REPORTS_DIR / "demo_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les prédictions complètes
    full_results = pd.DataFrame({
        "true_label": y_test.values,
        "predicted_label": y_pred,
        "correct": y_test.values == y_pred
    })
    
    if y_proba is not None:
        full_results["probability_0"] = y_proba[:, 0]
        full_results["probability_1"] = y_proba[:, 1]
    
    results_path = output_dir / "predictions_demo.csv"
    full_results.to_csv(results_path, index=False)
    logger.info(f"\nPrédictions sauvegardées: {results_path}")
    
    # Sauvegarder les métriques
    metrics_path = output_dir / "metrics_demo.json"
    evaluator.save_metrics(metrics_path)
    
    # 9. Résumé final
    logger.info("\n" + "=" * 80)
    logger.info("✅ RÉSUMÉ")
    logger.info("=" * 80)
    logger.info(f"Précision globale: {metrics['accuracy']:.2%}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    
    correct = (y_test.values == y_pred).sum()
    incorrect = len(y_test) - correct
    logger.info(f"Prédictions correctes: {correct}/{len(y_test)}")
    logger.info(f"Prédictions incorrectes: {incorrect}/{len(y_test)}")
    
    typer.echo("\n✅ Démonstration terminée!")


if __name__ == "__main__":
    app()


# Usage:
# python -m corai.demo_model
