"""
Module d'entraînement des modèles de machine learning.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from loguru import logger
import typer

from corai.config import MODELS_DIR, PROCESSED_DATA_DIR
from corai.preprocessing.data_loader import load_data, split_features_target

app = typer.Typer()


class ModelTrainer:
    """Classe pour entraîner et sauvegarder des modèles de machine learning."""

    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = 42
    ):
        """
        Initialise le trainer.
        
        Args:
            model_type: Type de modèle à entraîner
                       ('random_forest', 'gradient_boosting', 'logistic_regression', 'svm')
            random_state: Graine aléatoire pour la reproductibilité
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_scores: Optional[np.ndarray] = None

    def get_model(self):
        """Retourne une instance du modèle selon le type spécifié."""
        models = {
            "random_forest": RandomForestClassifier(random_state=self.random_state),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.random_state),
            "logistic_regression": LogisticRegression(random_state=self.random_state, max_iter=1000),
            "svm": SVC(random_state=self.random_state, probability=True)
        }
        
        if self.model_type not in models:
            raise ValueError(
                f"Type de modèle non supporté: {self.model_type}. "
                f"Choisir parmi: {list(models.keys())}"
            )
        
        return models[self.model_type]

    def get_param_grid(self) -> Dict[str, list]:
        """Retourne la grille de paramètres pour GridSearchCV."""
        param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5]
            },
            "logistic_regression": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"]
            },
            "svm": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            }
        }
        
        return param_grids.get(self.model_type, {})

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid_search: bool = False,
        cv_folds: int = 5
    ):
        """
        Entraîne le modèle.
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            use_grid_search: Si True, utilise GridSearchCV pour optimiser les hyperparamètres
            cv_folds: Nombre de folds pour la validation croisée
        """
        logger.info(f"Entraînement du modèle: {self.model_type}")
        
        if use_grid_search:
            logger.info("Recherche d'hyperparamètres avec GridSearchCV...")
            base_model = self.get_model()
            param_grid = self.get_param_grid()
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            logger.info(f"Meilleurs paramètres: {self.best_params}")
            logger.info(f"Meilleur score CV: {grid_search.best_score_:.4f}")
        else:
            self.model = self.get_model()
            self.model.fit(X_train, y_train)
        
        # Validation croisée
        logger.info("Validation croisée en cours...")
        self.cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=cv_folds, scoring="accuracy"
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
        
        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "best_params": self.best_params,
            "cv_scores": self.cv_scores
        }
        
        with open(output_path, "wb") as f:
            pickle.dump(model_data, f)
        
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
        
        score = self.model.score(X_test, y_test)
        logger.info(f"Score sur le test: {score:.4f}")
        return score


@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "processed_heart_disease_v0.csv",
    model_output: Path = MODELS_DIR / "heart_disease_model.pkl",
    model_type: str = "random_forest",
    target_column: str = "Heart Disease",
    test_size: float = 0.2,
    use_grid_search: bool = False,
    cv_folds: int = 5,
    random_state: int = 42,
):
    """
    Entraîne un modèle de machine learning.
    
    Args:
        data_path: Chemin vers les données prétraitées
        model_output: Chemin de sortie pour le modèle entraîné
        model_type: Type de modèle ('random_forest', 'gradient_boosting', 'logistic_regression', 'svm')
        target_column: Nom de la colonne cible
        test_size: Proportion des données pour le test
        use_grid_search: Utiliser GridSearchCV pour optimiser les hyperparamètres
        cv_folds: Nombre de folds pour la validation croisée
        random_state: Graine aléatoire
    """
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
