"""
Pipeline complet end-to-end pour le projet CorAI.
Orchestre tout le workflow: prétraitement → entraînement → prédiction → évaluation.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger
import typer

from corai.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR,
    DEFAULT_MODEL_TYPE, DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE, DEFAULT_CV_FOLDS, TARGET_COLUMN
)
from corai.preprocessing.data_loader import load_data, split_features_target
from corai.preprocessing.preprocessing_pipeline import DataDiagnosticsPreprocessor
from corai.modeling.train import ModelTrainer
from corai.modeling.evaluate import ModelEvaluator

app = typer.Typer()


class CompletePipeline:
    """Pipeline complet pour le projet de prédiction de maladies cardiaques."""

    def __init__(
        self,
        raw_data_path: Path,
        model_type: str = None,
        test_size: float = None,
        random_state: int = None
    ):
        """
        Initialise le pipeline complet.
        
        Args:
            raw_data_path: Chemin vers les données brutes
            model_type: Type de modèle à entraîner (None = utilise DEFAULT_MODEL_TYPE)
            test_size: Proportion des données pour le test (None = utilise DEFAULT_TEST_SIZE)
            random_state: Graine aléatoire (None = utilise DEFAULT_RANDOM_STATE)
        """
        self.raw_data_path = raw_data_path
        self.model_type = model_type or DEFAULT_MODEL_TYPE
        self.test_size = test_size or DEFAULT_TEST_SIZE
        self.random_state = random_state or DEFAULT_RANDOM_STATE
        
        # Données
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        
        # Modèle
        self.trainer: Optional[ModelTrainer] = None
        self.model = None
        
        # Résultats
        self.predictions: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None
        self.metrics: Dict[str, Any] = {}

    def step1_load_and_preprocess(self) -> pd.DataFrame:
        """
        Étape 1: Charger et prétraiter les données.
        
        Returns:
            DataFrame prétraité
        """
        logger.info("=" * 80)
        logger.info("ÉTAPE 1: CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES")
        logger.info("=" * 80)
        
        # Créer le pipeline de prétraitement
        pipeline = DataDiagnosticsPreprocessor(target_column=TARGET_COLUMN)
        
        # Charger les données
        logger.info(f"Chargement depuis: {self.raw_data_path}")
        pipeline.load(self.raw_data_path)
        self.df_raw = pipeline.df
        logger.info(f"Données brutes chargées: {self.df_raw.shape}")
        
        # Supprimer les doublons
        removed = pipeline.remove_duplicates()
        logger.info(f"Doublons supprimés: {removed}")
        
        # Appliquer le prétraitement
        logger.info("Application du prétraitement...")
        pipeline.fit_transform_preprocessor()
        
        # Récupérer les données transformées
        X_transformed = pipeline.X_transformed
        y_transformed = pipeline.y_arr
        
        # Combiner en un seul DataFrame
        self.df_processed = X_transformed.copy()
        self.df_processed[TARGET_COLUMN] = y_transformed
        
        logger.success(f"Données prétraitées: {self.df_processed.shape}")
        
        # Sauvegarder les données prétraitées
        processed_path = PROCESSED_DATA_DIR / "processed_heart_disease_complete.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        self.df_processed.to_csv(processed_path, index=False)
        logger.info(f"Données prétraitées sauvegardées: {processed_path}")
        
        return self.df_processed

    def step2_split_data(self):
        """Étape 2: Séparer les données en train/test."""
        logger.info("=" * 80)
        logger.info("ÉTAPE 2: SÉPARATION TRAIN/TEST")
        logger.info("=" * 80)
        
        # Séparer features et target
        X, y = split_features_target(self.df_processed, target=TARGET_COLUMN)
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"Train labels distribution: {self.y_train.value_counts().to_dict()}")
        logger.info(f"Test labels distribution: {self.y_test.value_counts().to_dict()}")
        
        # Sauvegarder les datasets
        train_features_path = PROCESSED_DATA_DIR / "train_features.csv"
        train_labels_path = PROCESSED_DATA_DIR / "train_labels.csv"
        test_features_path = PROCESSED_DATA_DIR / "test_features.csv"
        test_labels_path = PROCESSED_DATA_DIR / "test_labels.csv"
        
        self.X_train.to_csv(train_features_path, index=False)
        self.y_train.to_csv(train_labels_path, index=False)
        self.X_test.to_csv(test_features_path, index=False)
        self.y_test.to_csv(test_labels_path, index=False)
        
        logger.success("Datasets train/test sauvegardés")

    def step3_train_model(self, use_grid_search: bool = False, cv_folds: int = None):
        """
        Étape 3: Entraîner le modèle.
        
        Args:
            use_grid_search: Utiliser GridSearchCV
            cv_folds: Nombre de folds pour la validation croisée
        """
        logger.info("=" * 80)
        logger.info("ÉTAPE 3: ENTRAÎNEMENT DU MODÈLE")
        logger.info("=" * 80)
        
        cv_folds = cv_folds or DEFAULT_CV_FOLDS
        
        self.trainer = ModelTrainer(
            model_type=self.model_type,
            random_state=self.random_state
        )
        
        self.trainer.train(
            self.X_train,
            self.y_train,
            use_grid_search=use_grid_search,
            cv_folds=cv_folds
        )
        
        self.model = self.trainer.model
        
        # Sauvegarder le modèle
        model_path = MODELS_DIR / f"{self.model_type}_heart_disease.pkl"
        self.trainer.save_model(model_path)
        
        logger.success(f"Modèle entraîné et sauvegardé: {model_path}")

    def step4_predict(self):
        """Étape 4: Faire des prédictions sur le test set."""
        logger.info("=" * 80)
        logger.info("ÉTAPE 4: PRÉDICTIONS SUR LE TEST SET")
        logger.info("=" * 80)
        
        if self.model is None:
            raise RuntimeError("Le modèle n'est pas entraîné")
        
        # Prédictions
        self.predictions = self.model.predict(self.X_test)
        logger.info(f"Prédictions effectuées: {len(self.predictions)} échantillons")
        
        # Probabilités
        if hasattr(self.model, "predict_proba"):
            self.probabilities = self.model.predict_proba(self.X_test)
            logger.info("Probabilités calculées")
        
        # Sauvegarder les prédictions
        predictions_df = pd.DataFrame({
            "predictions": self.predictions
        })
        
        if self.probabilities is not None:
            for i in range(self.probabilities.shape[1]):
                predictions_df[f"proba_{i}"] = self.probabilities[:, i]
        
        predictions_path = PROCESSED_DATA_DIR / "test_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.success(f"Prédictions sauvegardées: {predictions_path}")

    def step5_evaluate(self):
        """Étape 5: Évaluer les performances du modèle."""
        logger.info("=" * 80)
        logger.info("ÉTAPE 5: ÉVALUATION DU MODÈLE")
        logger.info("=" * 80)
        
        evaluator = ModelEvaluator(task_type="classification")
        
        self.metrics = evaluator.evaluate_classification(
            y_true=self.y_test.values,
            y_pred=self.predictions,
            y_proba=self.probabilities
        )
        
        evaluator.print_metrics()
        
        # Sauvegarder les métriques
        metrics_path = REPORTS_DIR / f"{self.model_type}_evaluation_metrics.json"
        evaluator.save_metrics(metrics_path)
        
        logger.success(f"Métriques sauvegardées: {metrics_path}")

    def run_complete_pipeline(
        self,
        use_grid_search: bool = False,
        cv_folds: int = None
    ) -> Dict[str, Any]:
        """
        Execute le pipeline complet.
        
        Args:
            use_grid_search: Utiliser GridSearchCV
            cv_folds: Nombre de folds pour la validation croisée
        
        Returns:
            Dictionnaire avec les résultats
        """
        logger.info("🚀 DÉMARRAGE DU PIPELINE COMPLET")
        logger.info("=" * 80)
        
        # Étape 1: Prétraitement
        self.step1_load_and_preprocess()
        
        # Étape 2: Split
        self.step2_split_data()
        
        # Étape 3: Entraînement
        self.step3_train_model(use_grid_search=use_grid_search, cv_folds=cv_folds)
        
        # Étape 4: Prédiction
        self.step4_predict()
        
        # Étape 5: Évaluation
        self.step5_evaluate()
        
        logger.info("=" * 80)
        logger.success("✅ PIPELINE COMPLET TERMINÉ AVEC SUCCÈS!")
        logger.info("=" * 80)
        
        # Résumé
        results = {
            "model_type": self.model_type,
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "metrics": self.metrics,
            "best_params": self.trainer.best_params if self.trainer else None
        }
        
        return results


@app.command()
def main(
    raw_data_path: Path = RAW_DATA_DIR / "heart_disease_dataset.csv",
    model_type: str = None,
    test_size: float = None,
    use_grid_search: bool = False,
    cv_folds: int = None,
    random_state: int = None,
):
    """
    Execute le pipeline complet de bout en bout.
    
    Args:
        raw_data_path: Chemin vers les données brutes
        model_type: Type de modèle (None = utilise DEFAULT_MODEL_TYPE)
        test_size: Proportion des données pour le test (None = utilise DEFAULT_TEST_SIZE)
        use_grid_search: Utiliser GridSearchCV pour optimiser les hyperparamètres
        cv_folds: Nombre de folds pour la validation croisée (None = utilise DEFAULT_CV_FOLDS)
        random_state: Graine aléatoire (None = utilise DEFAULT_RANDOM_STATE)
    """
    pipeline = CompletePipeline(
        raw_data_path=raw_data_path,
        model_type=model_type,
        test_size=test_size,
        random_state=random_state
    )
    
    results = pipeline.run_complete_pipeline(
        use_grid_search=use_grid_search,
        cv_folds=cv_folds
    )
    
    # Afficher le résumé final
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ DES RÉSULTATS")
    logger.info("=" * 80)
    logger.info(f"Modèle: {results['model_type']}")
    logger.info(f"Train size: {results['train_size']}")
    logger.info(f"Test size: {results['test_size']}")
    logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    
    if results['best_params']:
        logger.info(f"Meilleurs paramètres: {results['best_params']}")
    
    typer.echo("\n✅ Pipeline terminé avec succès!")


if __name__ == "__main__":
    app()


# Exemples d'utilisation:
# 
# Pipeline basique avec Random Forest:
#   python -m corai.pipeline_complete
#
# Avec Gradient Boosting:
#   python -m corai.pipeline_complete --model-type gradient_boosting
#
# Avec optimisation d'hyperparamètres:
#   python -m corai.pipeline_complete --use-grid-search
#
# Tous les paramètres:
#   python -m corai.pipeline_complete --model-type gradient_boosting --use-grid-search --test-size 0.25
