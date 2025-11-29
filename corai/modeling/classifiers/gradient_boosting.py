"""
Modèle Gradient Boosting
"""

from sklearn.ensemble import GradientBoostingClassifier as SKLearnGradientBoosting
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from abstraite_base_model import BaseModel


class GradientBoosting(BaseModel):
    """
    Modèle Gradient Boosting pour la classification
    """

    def __init__(self, **kwargs):
        """
        Initialise le modèle Gradient Boosting
        
        Args:
            **kwargs: Hyperparamètres (n_estimators, learning_rate, max_depth, etc.)
        """
        super().__init__(name="GradientBoosting", **kwargs)

    def _initialize_model(self):
        """Initialise le modèle sklearn"""
        params = self.get_default_params()
        params.update(self.hyperparameters)
        self.model = SKLearnGradientBoosting(**params)

    def get_default_params(self) -> Dict[str, Any]:
        """Retourne les hyperparamètres par défaut"""
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'random_state': 42
        }
