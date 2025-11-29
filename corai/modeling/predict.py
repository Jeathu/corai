"""
Module de prédiction avec les modèles entraînés.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pickle

import pandas as pd
import numpy as np
from loguru import logger
import typer

from corai.config import MODELS_DIR, PROCESSED_DATA_DIR
from corai.preprocessing.data_loader import load_data

app = typer.Typer()


class ModelPredictor:
    """Classe pour charger un modèle et faire des prédictions."""

    def __init__(self, model_path: Path):
        """
        Initialise le prédicteur.
        
        Args:
            model_path: Chemin vers le modèle sauvegardé
        """
        self.model_path = model_path
        self.model = None
        self.model_type: Optional[str] = None
        self.model_metadata: Dict[str, Any] = {}
        self.load_model()

    def load_model(self):
        """Charge le modèle depuis le fichier pickle."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
        
        logger.info(f"Chargement du modèle depuis: {self.model_path}")
        
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict):
            self.model = model_data.get("model")
            self.model_type = model_data.get("model_type")
            self.model_metadata = {
                "best_params": model_data.get("best_params"),
                "cv_scores": model_data.get("cv_scores")
            }
        else:
            # Ancien format: juste le modèle
            self.model = model_data
        
        logger.success(f"Modèle chargé avec succès (type: {self.model_type})")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions sur les données.
        
        Args:
            X: Features pour la prédiction
        
        Returns:
            Array des prédictions
        """
        if self.model is None:
            raise RuntimeError("Modèle non chargé")
        
        logger.info(f"Prédiction sur {X.shape[0]} échantillons...")
        predictions = self.model.predict(X)
        logger.success("Prédictions terminées")
        
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Retourne les probabilités de prédiction (si disponible).
        
        Args:
            X: Features pour la prédiction
        
        Returns:
            Array des probabilités ou None si non disponible
        """
        if self.model is None:
            raise RuntimeError("Modèle non chargé")
        
        if not hasattr(self.model, "predict_proba"):
            logger.warning("Le modèle ne supporte pas predict_proba")
            return None
        
        logger.info("Calcul des probabilités...")
        probabilities = self.model.predict_proba(X)
        
        return probabilities

    def save_predictions(
        self,
        predictions: np.ndarray,
        output_path: Path,
        probabilities: Optional[np.ndarray] = None,
        include_features: Optional[pd.DataFrame] = None
    ):
        """
        Sauvegarde les prédictions dans un fichier CSV.
        
        Args:
            predictions: Array des prédictions
            output_path: Chemin de sortie
            probabilities: Probabilités optionnelles
            include_features: Features à inclure dans le fichier
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_df = pd.DataFrame()
        
        # Ajouter les features si fournies
        if include_features is not None:
            result_df = include_features.copy()
        
        # Ajouter les prédictions
        result_df["predictions"] = predictions
        
        # Ajouter les probabilités si disponibles
        if probabilities is not None:
            n_classes = probabilities.shape[1]
            for i in range(n_classes):
                result_df[f"proba_{i}"] = probabilities[:, i]
        
        result_df.to_csv(output_path, index=False)
        logger.success(f"Prédictions sauvegardées: {output_path}")


def predict_from_file(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    include_probabilities: bool = True,
    save_features: bool = False
) -> pd.DataFrame:
    """
    Fait des prédictions à partir de fichiers.
    
    Args:
        model_path: Chemin vers le modèle
        data_path: Chemin vers les données
        output_path: Chemin de sortie
        include_probabilities: Inclure les probabilités
        save_features: Sauvegarder aussi les features dans le fichier
    
    Returns:
        DataFrame avec les prédictions
    """
    # Charger les données
    logger.info("Chargement des données...")
    X = load_data(data_path)
    
    # Créer le prédicteur
    predictor = ModelPredictor(model_path)
    
    # Faire les prédictions
    predictions = predictor.predict(X)
    
    # Calculer les probabilités si demandé
    probabilities = None
    if include_probabilities:
        probabilities = predictor.predict_proba(X)
    
    # Sauvegarder
    features_to_save = X if save_features else None
    predictor.save_predictions(
        predictions=predictions,
        output_path=output_path,
        probabilities=probabilities,
        include_features=features_to_save
    )
    
    # Retourner un DataFrame avec les résultats
    result_df = pd.DataFrame({"predictions": predictions})
    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            result_df[f"proba_{i}"] = probabilities[:, i]
    
    return result_df


@app.command()
def main(
    model_path: Path = MODELS_DIR / "heart_disease_model.pkl",
    data_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    predictions_output: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    include_probabilities: bool = True,
    save_features: bool = False,
):
    """
    Fait des prédictions avec un modèle entraîné.
    
    Args:
        model_path: Chemin vers le modèle entraîné
        data_path: Chemin vers les données de test
        predictions_output: Chemin de sortie pour les prédictions
        include_probabilities: Inclure les probabilités dans le fichier de sortie
        save_features: Sauvegarder aussi les features dans le fichier de sortie
    """
    predict_from_file(
        model_path=model_path,
        data_path=data_path,
        output_path=predictions_output,
        include_probabilities=include_probabilities,
        save_features=save_features
    )
    
    logger.success("Prédictions terminées avec succès!")
    typer.echo(f"Prédictions sauvegardées: {predictions_output}")


if __name__ == "__main__":
    app()
