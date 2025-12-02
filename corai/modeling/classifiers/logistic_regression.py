"""
Modèle Logistic Regression
"""

from sklearn.linear_model import LogisticRegression as SKLearnLogisticRegression
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from abstraite_base_model import BaseModel


class LogisticRegression(BaseModel):
    """
    Modèle de régression logistique pour la classification
    """

    def __init__(self, **kwargs):
        """
        Initialise le modèle Logistic Regression

        Args:
            **kwargs: Hyperparamètres (C, penalty, solver, max_iter, etc.)
        """
        super().__init__(name="LogisticRegression", **kwargs)



    def _initialize_model(self):
        """Initialise le modèle sklearn"""
        params = self.get_default_params()
        params.update(self.hyperparameters)
        self.model = SKLearnLogisticRegression(**params)



    def get_default_params(self) -> Dict[str, Any]:
        """Retourne les hyperparamètres par défaut"""
        return {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42
        }
