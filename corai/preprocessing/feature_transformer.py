import pandas as pd
from typing import Optional, List, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler




class HeartDiseasePreprocessor(BaseEstimator, TransformerMixin):
    """
    Préprocesseur pour les données de maladies cardiaques.

    Effectue:
        - Encodage one-hot des variables catégorielles
        - Normalisation des variables numériques
        - fit, transform et fit_transform methods
    """


    def __init__(
        self,
        target_column: str = "Heart Disease",
        target_mapping: Optional[Dict[str, int]] = None
    ):
        self.target_column = target_column
        self.target_mapping = target_mapping or {"Yes": 1, "No": 0}
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self._fitted = False
        self.output_feature_names_: List[str] = []
        self._scaler: Optional[StandardScaler] = None
        self._encoder: Optional[OneHotEncoder] = None




    def _detecter_colonnes(self, df: pd.DataFrame) -> None:
        """Détecte les colonnes numériques et catégorielles automatiquement."""
        cols_numeriques = list(df.select_dtypes(include=["int64", "float64"]).columns)
        cols_categorielles = list(df.select_dtypes(include=["object", "category"]).columns)

        # Retirer la colonne cible si présente
        for col_list in [cols_numeriques, cols_categorielles]:
            if self.target_column in col_list:
                col_list.remove(self.target_column)

        self.numerical_columns = cols_numeriques
        self.categorical_columns = cols_categorielles




    def fit(self, X: pd.DataFrame, y=None) -> "HeartDiseasePreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X doit être un DataFrame pandas")

        df = X.copy()
        self._detecter_colonnes(df)

        # Initialiser et ajuster les préprocesseurs
        if self.numerical_columns:
            self._scaler = StandardScaler()
            self._scaler.fit(df[self.numerical_columns])

        if self.categorical_columns:
            self._encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self._encoder.fit(df[self.categorical_columns])

        # Obtenir les noms des features
        numeric_features = self.numerical_columns
        categorical_features = (
            list(self._encoder.get_feature_names_out(self.categorical_columns))
            if self.categorical_columns else []
        )
        self.output_feature_names_ = numeric_features + categorical_features

        self._fitted = True
        return self




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

        # Gérer la colonne cible
        if self.target_column in df.columns:
            result[self.target_column] = df[self.target_column].map(self.target_mapping)
            mask = result[self.target_column].isna()
            if mask.any():
                result.loc[mask, self.target_column] = df.loc[mask, self.target_column]

        return result




    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Ajuste et transforme en une seule opération."""
        return self.fit(X, y).transform(X)