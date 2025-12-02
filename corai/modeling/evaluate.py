"""
Module d'évaluation des modèles.
Contient les fonctions pour évaluer les performances des modèles de machine learning.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from loguru import logger
import typer


from corai.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()




class ModelEvaluator:
    """Classe pour évaluer les performances d'un modèle."""

    def __init__(self, task_type: str = "classification"):
        """
           Initialise l'évaluateur.

           Args:
              task_type: Type de tâche ('classification' ou 'regression')
        """
        self.task_type = task_type
        self.metrics: Dict[str, Any] = {}



    def evaluate_classification(self,y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Évalue un modèle de classification.

        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions du modèle
            y_proba: Probabilités prédites (optionnel, pour ROC AUC)

        Returns:
            Dictionnaire contenant les métriques d'évaluation
        """
        metrics = {}

        # Métriques de base
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # ROC AUC si les probabilités sont fournies
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binaire
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-classe
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
            except Exception as e:
                logger.warning(f"Impossible de calculer ROC AUC: {e}")

        # Matrice de confusion
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        # Rapport de classification
        metrics["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        self.metrics = metrics
        return metrics



    def evaluate_regression(self,y_true: np.ndarray,y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Évalue un modèle de régression.
            Args:
               y_true: Vraies valeurs
               y_pred: Prédictions du modèle

            Returns:
               Dictionnaire contenant les métriques d'évaluation
        """

        metrics = {}
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2_score"] = r2_score(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0
        if mask.any():
            metrics["mape"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        self.metrics = metrics
        return metrics



    def save_metrics(self, output_path: Path) -> None:
        """
          Sauvegarde les métriques dans un fichier JSON.

          Args:
            output_path: Chemin du fichier de sortie
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=4, ensure_ascii=False)
        logger.success(f"Métriques sauvegardées: {output_path}")



    def print_metrics(self) -> None:
        """Affiche les métriques dans la console."""
        logger.info("=" * 80)
        logger.info("MÉTRIQUES D'ÉVALUATION DU MODÈLE")
        logger.info("=" * 80)
        if self.task_type == "classification":
            logger.info(f"Accuracy: {self.metrics.get('accuracy', 0):.4f}")
            logger.info(f"Precision: {self.metrics.get('precision', 0):.4f}")
            logger.info(f"Recall: {self.metrics.get('recall', 0):.4f}")
            logger.info(f"F1 Score: {self.metrics.get('f1_score', 0):.4f}")

            if "roc_auc" in self.metrics:
                logger.info(f"ROC AUC: {self.metrics['roc_auc']:.4f}")

            logger.info("\nMatrice de confusion:")
            logger.info(f"{self.metrics.get('confusion_matrix', [])}")

        elif self.task_type == "regression":
            logger.info(f"MSE: {self.metrics.get('mse', 0):.4f}")
            logger.info(f"RMSE: {self.metrics.get('rmse', 0):.4f}")
            logger.info(f"MAE: {self.metrics.get('mae', 0):.4f}")
            logger.info(f"R² Score: {self.metrics.get('r2_score', 0):.4f}")

            if "mape" in self.metrics:
                logger.info(f"MAPE: {self.metrics['mape']:.2f}%")



def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task_type: str = "classification",
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Évalue un modèle et retourne les métriques.

    Args:
        y_true: Vraies valeurs/étiquettes
        y_pred: Prédictions du modèle
        y_proba: Probabilités prédites (pour classification)
        task_type: Type de tâche ('classification' ou 'regression')
        output_path: Chemin optionnel pour sauvegarder les métriques

    Returns:
        Dictionnaire des métriques
    """
    evaluator = ModelEvaluator(task_type=task_type)

    if task_type == "classification":
        metrics = evaluator.evaluate_classification(y_true, y_pred, y_proba)
    elif task_type == "regression":
        metrics = evaluator.evaluate_regression(y_true, y_pred)
    else:
        raise ValueError(f"Type de tâche non supporté: {task_type}")
    evaluator.print_metrics()

    if output_path:
        evaluator.save_metrics(output_path)
    return metrics




@app.command()
def main(
    predictions_path: Path = MODELS_DIR / "predictions/test_predictions.csv",
    metrics_output: Path = REPORTS_DIR / "evaluation_metrics.json",
    task_type: str = "classification",
    predictions_col: str = "predictions",
    actual_col: str = "actual",
) -> None:
    """
    Évalue un modèle à partir d'un fichier de prédictions.

    Args:
        predictions_path: Chemin vers le fichier des prédictions (doit contenir 'predictions' et 'actual')
        metrics_output: Chemin de sortie pour les métriques
        task_type: Type de tâche ('classification' ou 'regression')
        predictions_col: Nom de la colonne des prédictions
        actual_col: Nom de la colonne des vraies valeurs
    """
    logger.info("Chargement des données...")

    # Charger les prédictions
    predictions_df = pd.read_csv(predictions_path)

    # Vérifier que les colonnes nécessaires existent
    if predictions_col not in predictions_df.columns:
        raise ValueError(f"Colonne '{predictions_col}' non trouvée dans {predictions_path}")

    if actual_col not in predictions_df.columns:
        raise ValueError(f"Colonne '{actual_col}' non trouvée dans {predictions_path}. "
                        f"Colonnes disponibles: {list(predictions_df.columns)}")

    y_pred = predictions_df[predictions_col].values
    y_true = predictions_df[actual_col].values

    # Probabilités si disponibles
    y_proba = None
    proba_cols = [col for col in predictions_df.columns if col.startswith("proba_")]
    if proba_cols:
        y_proba = predictions_df[proba_cols].values

    # Évaluer le modèle
    evaluate_model(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        task_type=task_type,
        output_path=metrics_output
    )
    typer.echo(f"Évaluation terminée. Métriques sauvegardées: {metrics_output}")




if __name__ == "__main__":
    app()
