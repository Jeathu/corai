import pandas as pd
from typing import Optional, List, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class HeartDiseasePreprocessor(BaseEstimator, TransformerMixin):
    """
    Préprocesseur pour les données de maladies cardiaques.

        Effectue:
            - Imputation des valeurs manquantes
            - Encodage one-hot des variables catégorielles
            - Normalisation des variables numériques
    """


    # Constructor
    def __init__(
        self, 
        target_column: str = "Heart Disease Status",
        target_mapping: Optional[Dict[str, int]] = None
    ):
        self.target_column = target_column
        self.target_mapping = target_mapping or {"Yes": 1, "No": 0}
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self._preprocessor: Optional[ColumnTransformer] = None
        self._fitted = False
        self.output_feature_names_: List[str] = []




    # Détecte les colonnes numériques et catégorielles automatiquement
    def _infer_columns(self, df: pd.DataFrame) -> None:
        num_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)
        cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)

        # Retirer la colonne cible si présente
        for col_list in [num_cols, cat_cols]:
            if self.target_column in col_list:
                col_list.remove(self.target_column)

        self.numerical_columns = num_cols
        self.categorical_columns = cat_cols




    # Ajuste le préprocesseur sur les données d'entrée
    def fit(self, X: pd.DataFrame, y=None) -> "HeartDiseasePreprocessor":
        """Ajuste le préprocesseur sur les données d'entrée."""
        df = X.copy()
        self._infer_columns(df)

        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self._preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_columns),
                ('cat', categorical_pipeline, self.categorical_columns)
            ],
            remainder="drop"
        )

        fit_cols = self.numerical_columns + self.categorical_columns
        self._preprocessor.fit(df[fit_cols])

        # Récupération des noms de features
        try:
            self.output_feature_names_ = list(
                self._preprocessor.get_feature_names_out(input_features=fit_cols)
            )
        except AttributeError:
            # Fallback pour anciennes versions de sklearn
            transformed = self._preprocessor.transform(df[fit_cols])
            self.output_feature_names_ = [f"feature_{i}" for i in range(transformed.shape[1])]

        self._fitted = True
        return self




    # Transforme les données d'entrée en utilisant le préprocesseur ajusté
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforme les données avec le préprocesseur ajusté."""
        if not self._fitted or self._preprocessor is None:
            raise RuntimeError("Le préprocesseur doit être ajusté avant transform().")

        df = X.copy()
        fit_cols = self.numerical_columns + self.categorical_columns

        missing = set(fit_cols) - set(df.columns)
        if missing:
            raise KeyError(f"Colonnes manquantes pour la transformation: {missing}")

        transformed = self._preprocessor.transform(df[fit_cols])

        result = pd.DataFrame(
            transformed,
            columns=self.output_feature_names_,
            index=df.index
        )

        # Ajout colonne cible si présente
        if self.target_column in df.columns:
            result[self.target_column] = df[self.target_column].map(self.target_mapping)
            mask = result[self.target_column].isna()
            result.loc[mask, self.target_column] = df.loc[mask, self.target_column]

        return result




    # Ajuste et transforme en une seule opération
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Ajuste et transforme en une seule opération."""
        return self.fit(X, y).transform(X)