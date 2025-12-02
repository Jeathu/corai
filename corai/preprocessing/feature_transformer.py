import pandas as pd
from typing import Optional, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class HeartDiseasePreprocessor(BaseEstimator, TransformerMixin):
    """
    Préprocesseur pour les données de maladies cardiaques.
    Effectue encodage one-hot et normalisation.
    """

    def __init__(self):
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self._fitted = False
        self.output_feature_names_: List[str] = []
        self._scaler: Optional[StandardScaler] = None
        self._encoder: Optional[OneHotEncoder] = None



    # Méthode pour détecter les colonnes numériques et catégorielles
    def _detecter_colonnes(self, df: pd.DataFrame) -> None:
        """Détecte les colonnes numériques et catégorielles automatiquement."""
        self.numerical_columns = list(df.select_dtypes(include=["int64", "float64"]).columns)
        self.categorical_columns = list(df.select_dtypes(include=["object", "category"]).columns)



    # Méthode pour ajuster et transformer les données
    def fit(self, X: pd.DataFrame, y=None) -> "HeartDiseasePreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X doit être un DataFrame pandas")

        df = X.copy()
        self._detecter_colonnes(df)

        if self.numerical_columns:
            self._scaler = StandardScaler()
            self._scaler.fit(df[self.numerical_columns])

        if self.categorical_columns:
            self._encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self._encoder.fit(df[self.categorical_columns])

        numeric_features = self.numerical_columns
        categorical_features = (
            list(self._encoder.get_feature_names_out(self.categorical_columns))
            if self.categorical_columns else []
        )
        self.output_feature_names_ = numeric_features + categorical_features

        self._fitted = True
        return self



    # Méthode pour transformer les données
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Le préprocesseur doit être ajusté avant transform().")

        df = X.copy()
        fit_cols = self.numerical_columns + self.categorical_columns
        missing = set(fit_cols) - set(df.columns)
        if missing:
            raise KeyError(f"Colonnes manquantes pour la transformation: {missing}")

        result = pd.DataFrame(index=df.index)

        if self.numerical_columns and self._scaler:
            result[self.numerical_columns] = self._scaler.transform(df[self.numerical_columns])

        if self.categorical_columns and self._encoder:
            encoded = self._encoder.transform(df[self.categorical_columns])
            feature_names = self._encoder.get_feature_names_out(self.categorical_columns)
            result[list(feature_names)] = encoded

        return result



    # Méthode pour ajuster et transformer en une seule opération
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Ajuste et transforme en une seule opération."""
        return self.fit(X, y).transform(X)