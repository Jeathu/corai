"""
Classifiers - Modèles de classification
"""

from corai.modeling.classifiers.logistic_regression import LogisticRegression
from corai.modeling.classifiers.random_forest import RandomForest
from corai.modeling.classifiers.gradient_boosting import GradientBoosting
from corai.modeling.classifiers.svm import SVM

__all__ = [
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "SVM"
]
