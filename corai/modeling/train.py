"""
Module d'entraînement des modèles de machine learning.
Utilise l'architecture BaseModel pour une approche générique.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from loguru import logger
import typer

from corai.config import (
    MODELS_DIR, 
    PROCESSED_DATA_DIR,
    DEFAULT_MODEL_TYPE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_CV_FOLDS,
    TARGET_COLUMN
)
from corai.preprocessing.data_loader import load_data, split_features_target
from corai.modeling.model_factory import ModelFactory

app = typer.Typer()


class ModelTrainer:
    """Classe pour entraîner et sauvegarder des modèles de machine learning."""

    def __init__(
        self,
        model_type: str = None,
        random_state: int = None
    ):
        """
        Initialise le trainer.
        
        Args:
            model_type: Type de modèle à entraîner (utilise DEFAULT_MODEL_TYPE si None)
            random_state: Graine aléatoire (utilise DEFAULT_RANDOM_STATE si None)
        """
        self.model_type = model_type or DEFAULT_MODEL_TYPE
        self.random_state = random_state or DEFAULT_RANDOM_STATE
        self.model = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_scores: Optional[np.ndarray] = None

    def get_model(self):
        """Retourne une instance du modèle selon le type spécifié."""
        return ModelFactory.create_model(
            self.model_type,
            random_state=self.random_state
        )

    def get_param_grid(self) -> Dict[str, list]:
        """Retourne la grille de paramètres pour GridSearchCV."""
        return ModelFactory.get_param_grid(self.model_type)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid_search: bool = False,
        cv_folds: int = None
    ):
        """
        Entraîne le modèle.
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            use_grid_search: Si True, utilise GridSearchCV pour optimiser les hyperparamètres
            cv_folds: Nombre de folds pour la validation croisée (utilise DEFAULT_CV_FOLDS si None)
        """
        if cv_folds is None:
            cv_folds = DEFAULT_CV_FOLDS
            
        logger.info(f"Entraînement du modèle: {self.model_type}")
        
        if use_grid_search:
            logger.info("Recherche d'hyperparamètres avec GridSearchCV...")
            base_model = self.get_model()
            param_grid = self.get_param_grid()
            
            grid_search = GridSearchCV(
                base_model.model,  # Utilise le modèle sklearn interne
                param_grid,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Créer un nouveau modèle avec les meilleurs paramètres
            self.model = self.get_model()
            self.model.set_params(**grid_search.best_params_)
            self.model.fit(X_train, y_train)
            
            self.best_params = grid_search.best_params_
            
            logger.info(f"Meilleurs paramètres: {self.best_params}")
            logger.info(f"Meilleur score CV: {grid_search.best_score_:.4f}")
        else:
            self.model = self.get_model()
            self.model.fit(X_train, y_train)
        
        # Validation croisée
        logger.info("Validation croisée en cours...")
        self.cv_scores = cross_val_score(
            self.model.model, X_train, y_train, cv=cv_folds, scoring="accuracy"
        )
        
        logger.info(f"Scores CV: {self.cv_scores}")
        logger.info(f"Score CV moyen: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std() * 2:.4f})")
        logger.success("Entraînement terminé")

    def save_model(self, output_path: Path):
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            output_path: Chemin de sortie pour le modèle
        """
        if self.model is None:
            raise RuntimeError("Aucun modèle entraîné à sauvegarder")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Utilise la méthode save de BaseModel
        self.model.save(filepath=output_path)
        
        logger.success(f"Modèle sauvegardé: {output_path}")

    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Évalue le modèle sur un ensemble de test.
        
        Args:
            X_test: Features de test
            y_test: Labels de test
        
        Returns:
            Score d'accuracy sur le test
        """
        if self.model is None:
            raise RuntimeError("Le modèle doit être entraîné avant l'évaluation")
        
        predictions = self.model.predict(X_test)
        score = (predictions == y_test.values).mean()
        logger.info(f"Score sur le test: {score:.4f}")
        return score


@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "processed_heart_disease_v0.csv",
    model_output: Path = MODELS_DIR / "heart_disease_model.pkl",
    model_type: str = None,
    target_column: str = None,
    test_size: float = None,
    use_grid_search: bool = False,
    cv_folds: int = None,
    random_state: int = None,
):
    """
    Entraîne un modèle de machine learning.
    
    Args:
        data_path: Chemin vers les données prétraitées
        model_output: Chemin de sortie pour le modèle entraîné
        model_type: Type de modèle (None = utilise DEFAULT_MODEL_TYPE)
        target_column: Nom de la colonne cible (None = utilise TARGET_COLUMN)
        test_size: Proportion des données pour le test (None = utilise DEFAULT_TEST_SIZE)
        use_grid_search: Utiliser GridSearchCV pour optimiser les hyperparamètres
        cv_folds: Nombre de folds pour la validation croisée (None = utilise DEFAULT_CV_FOLDS)
        random_state: Graine aléatoire (None = utilise DEFAULT_RANDOM_STATE)
    """
    # Utiliser les valeurs par défaut du config si non spécifiées
    model_type = model_type or DEFAULT_MODEL_TYPE
    target_column = target_column or TARGET_COLUMN
    test_size = test_size or DEFAULT_TEST_SIZE
    random_state = random_state or DEFAULT_RANDOM_STATE
    
    logger.info(f"Configuration: model={model_type}, test_size={test_size}, random_state={random_state}")
    logger.info("Chargement des données...")
    df = load_data(data_path)
    
    logger.info("Séparation features/target...")
    X, y = split_features_target(df, target=target_column)
    
    logger.info("Séparation train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Taille train: {X_train.shape}, Taille test: {X_test.shape}")
    
    # Entraînement
    trainer = ModelTrainer(model_type=model_type, random_state=random_state)
    trainer.train(X_train, y_train, use_grid_search=use_grid_search, cv_folds=cv_folds)
    
    # Évaluation sur le test
    trainer.evaluate_on_test(X_test, y_test)
    
    # Sauvegarde
    trainer.save_model(model_output)
    
    logger.success("Entraînement du modèle terminé avec succès!")
    typer.echo(f"Modèle sauvegardé: {model_output}")


if __name__ == "__main__":
    app()
