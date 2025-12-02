"""
Modèle SVM (Support Vector Machine)
"""

from sklearn.svm import SVC as SKLearnSVC
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from abstraite_base_model import BaseModel


class SVM(BaseModel):
    """
    Modèle SVM pour la classification
    """

    def __init__(self, **kwargs):
        """
        Initialise le modèle SVM

        Args:
            **kwargs: Hyperparamètres (C, kernel, gamma, etc.)
        """
        super().__init__(name="SVM", **kwargs)



    def _initialize_model(self):
        """Initialise le modèle sklearn"""
        params = self.get_default_params()
        params.update(self.hyperparameters)
        self.model = SKLearnSVC(**params)



    def get_default_params(self) -> Dict[str, Any]:
        """Retourne les hyperparamètres par défaut"""
        return {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
