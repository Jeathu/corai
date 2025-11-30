"""
Classe abstraite de base pour tous les modèles
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


from corai.config import MODELS_DIR




class BaseModel(ABC):
    """
    Classe abstraite définissant l'interface pour tous les modèles
    Chaque modèle doit hériter de cette classe
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_trained = False
        self.hyperparameters = kwargs
        self.training_metadata = {}
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_model()




    @abstractmethod
    def _initialize_model(self):
        """Initialise le modèle sklearn - à implémenter par chaque sous-classe"""
        pass




    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Retourne les hyperparamètres par défaut - À implémenter"""
        pass




    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BaseModel':
        """
        Entraîne le modèle

        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement

        Returns:
            self
        """
        print(f"Entraînement: {self.name}...")

        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        end_time = datetime.now()

        self.is_trained = True
        self.training_metadata = {
            'training_date': start_time.isoformat(),
            'training_duration_seconds': (end_time - start_time).total_seconds(),
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'feature_names': list(X_train.columns)
        }

        print(f"{self.name} entraîné en {self.training_metadata['training_duration_seconds']:.2f}s")
        return self




    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Effectue des prédictions

        Args:
            X: Features

        Returns:
            Prédictions
        """
        self._check_is_trained()
        return self.model.predict(X)




    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retourne les probabilités de prédiction

        Args:
            X: Features

        Returns:
            Probabilités [P(classe_0), P(classe_1)]
        """
        self._check_is_trained()
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.name} ne supporte pas predict_proba")




    def _check_is_trained(self):
        """Vérifie que le modèle est entraîné"""
        if not self.is_trained:
            raise RuntimeError(f"Le modèle {self.name} n'est pas encore entraîné. Appelez fit() d'abord.")




    def get_params(self) -> Dict[str, Any]:
        """Retourne les hyperparamètres actuels"""
        return self.model.get_params() if self.model else self.hyperparameters




    def set_params(self, **params):
        """Met à jour les hyperparamètres"""
        if self.model:
            self.model.set_params(**params)
        self.hyperparameters.update(params)
        return self




    def save(self, version: str = None, filepath: Path = None) -> Path:
        """
        Sauvegarde le modèle
            Args:
               version: Version du modèle
               filepath: Chemin personnalisé (optionnel)
            Returns:
               Chemin du fichier sauvegardé
        """
        self._check_is_trained()
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        if filepath is None:
            filepath = self.models_dir / f"{self.name}_v{version}.joblib"

        # Sauvegarder le modèle et métadonnées
        model_data = {
            'model': self.model,
            'name': self.name,
            'hyperparameters': self.hyperparameters,
            'training_metadata': self.training_metadata,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Modèle sauvegardé: {filepath}")
        return filepath




    @classmethod
    def load(cls, filepath: Path) -> 'BaseModel':
        """
        Charge un modèle depuis un fichier
            Args:
               filepath: Chemin du fichier
            Returns:
               Instance du modèle
        """
        print(f"Chargement: {filepath}")
        model_data = joblib.load(filepath)

        # Créer une nouvelle instance
        instance = cls.__new__(cls)
        instance.model = model_data['model']
        instance.name = model_data['name']
        instance.hyperparameters = model_data['hyperparameters']
        instance.training_metadata = model_data['training_metadata']
        instance.is_trained = model_data['is_trained']
        instance.models_dir = MODELS_DIR
        return instance




    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Retourne l'importance des features (si disponible)
            Returns:
               Series avec l'importance des features ou None
        """

        self._check_is_trained()
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.training_metadata.get('feature_names', [])
            return pd.Series(
                self.model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)
        elif hasattr(self.model, 'coef_'):
            feature_names = self.training_metadata.get('feature_names', [])
            return pd.Series(
                np.abs(self.model.coef_[0]),
                index=feature_names
            ).sort_values(ascending=False)
        else:
            return None





    def __repr__(self) -> str:
        status = "Entraîné" if self.is_trained else "⏳ Non entraîné"
        return f"{self.name} ({status})"




    def __str__(self) -> str:
        return self.__repr__()