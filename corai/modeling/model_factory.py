"""
Factory pour créer des modèles de manière générique
"""

from typing import Dict, Any
from pathlib import Path

from corai.modeling.abstraite_base_model import BaseModel
from corai.modeling.classifiers import (
    LogisticRegression,
    RandomForest,
    GradientBoosting,
    SVM
)


class ModelFactory:
    """Factory pour créer des instances de modèles"""

    # Mapping des noms de modèles vers leurs classes
    MODEL_REGISTRY = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForest,
        'gradient_boosting': GradientBoosting,
        'svm': SVM
    }

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        Crée une instance de modèle
        
        Args:
            model_type: Type de modèle ('logistic_regression', 'random_forest', etc.)
            **kwargs: Hyperparamètres à passer au modèle
            
        Returns:
            Instance du modèle
            
        Raises:
            ValueError: Si le type de modèle n'est pas supporté
        """
        model_type = model_type.lower()
        
        if model_type not in cls.MODEL_REGISTRY:
            available = list(cls.MODEL_REGISTRY.keys())
            raise ValueError(
                f"Type de modèle '{model_type}' non supporté. "
                f"Modèles disponibles: {available}"
            )
        
        model_class = cls.MODEL_REGISTRY[model_type]
        return model_class(**kwargs)

    @classmethod
    def list_available_models(cls) -> list:
        """Retourne la liste des modèles disponibles"""
        return list(cls.MODEL_REGISTRY.keys())

    @classmethod
    def get_param_grid(cls, model_type: str) -> Dict[str, list]:
        """
        Retourne la grille de paramètres pour GridSearchCV
        
        Args:
            model_type: Type de modèle
            
        Returns:
            Dictionnaire avec les paramètres à tester
        """
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
        
        return param_grids.get(model_type, {})
