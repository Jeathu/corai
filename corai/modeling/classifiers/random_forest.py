"""
Modèle Random Forest
"""

from sklearn.ensemble import RandomForestClassifier as SKLearnRandomForest
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from abstraite_base_model import BaseModel




class RandomForest(BaseModel):
    """
    Modèle Random Forest pour la classification
    """

    def __init__(self, **kwargs):
        """
        Initialise le modèle Random Forest

        Args:
            **kwargs: Hyperparamètres (n_estimators, max_depth, etc.)
        """
        super().__init__(name="RandomForest", **kwargs)




    def _initialize_model(self):
        """Initialise le modèle sklearn"""
        params = self.get_default_params()
        params.update(self.hyperparameters)
        self.model = SKLearnRandomForest(**params)




    def get_default_params(self) -> Dict[str, Any]:
        """Retourne les hyperparamètres par défaut"""
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
