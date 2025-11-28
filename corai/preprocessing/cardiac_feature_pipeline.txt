import joblib
from typing import List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



""" Class de prétraitement des données pour l'apprentissage """
class HeartDiseasePreprocessor(BaseEstimator, TransformerMixin):
    # Initialisation du préprocesseur en séparant les colonnes numériques et catégorielles
    def __init__(self):
        self.target_column = "Heart Disease Status"
        self.numerical_columns = [
            "Age",
            "Blood Pressure",
            "Cholesterol Level",
            "BMI",
            "Sleep Hours",
            "Triglyceride Level",
            "Fasting Blood Sugar",
            "CRP Level",
            "Homocysteine Level",
        ]
        self.categorical_columns = [
            "Gender",
            "Exercise Habits",
            "Smoking",
            "Family Heart Disease",
            "Diabetes",
            "High Blood Pressure",
            "Low HDL Cholesterol",
            "High LDL Cholesterol",
            "Alcohol Consumption",
            "Stress Level",
            "Sugar Consumption",
        ]
        self.engineered_categoricals = ["age_group", "bp_category"]

        self._preprocessor: Optional[ColumnTransformer] = None
        self._fitted = False
        self.output_feature_names_: List[str] = []




    """ Ajuste le préprocesseur sur les données d'entraînement """
    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df = self._create_features(df)

        # Séparation des colonnes numériques et catégorielles 
        num_cols = [c for c in self.numerical_columns if c in df.columns]
        cat_cols = [c for c in self.categorical_columns if c in df.columns] + [
            c for c in self.engineered_categoricals if c in df.columns
        ]

        # Normalisation et encodage
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])


        self._preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols)
            ],
            remainder="drop",
        )

        fit_cols = num_cols + cat_cols
        if len(fit_cols) == 0:
            raise ValueError("Aucune colonne valide trouvée pour le fit du préprocesseur.")
        self._preprocessor.fit(df[fit_cols])

        try:
            self.output_feature_names_ = list(self._preprocessor.get_feature_names_out(input_features=fit_cols))
        except Exception:
            transformed = self._preprocessor.transform(df[fit_cols])
            self.output_feature_names_ = [f"f{i}" for i in range(transformed.shape[1])]

        self._fitted = True
        return self




    # Transformation des nouvelles données avec le préprocesseur entraîné
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self._preprocessor is None:
            raise RuntimeError("Le préprocesseur doit être fit avant transform.")

        df = X.copy()
        df = self._create_features(df)

        fit_cols = [c for c in self.numerical_columns if c in df.columns]
        fit_cols += [c for c in self.categorical_columns if c in df.columns]
        fit_cols += [c for c in self.engineered_categoricals if c in df.columns]
        fit_cols = [c for c in fit_cols if c in df.columns]

        transformed = self._preprocessor.transform(df[fit_cols])

        try:
            cols = list(self._preprocessor.get_feature_names_out(input_features=fit_cols))
        except Exception:
            cols = self.output_feature_names_ if self.output_feature_names_ else [f"f{i}" for i in range(transformed.shape[1])]

        result = pd.DataFrame(transformed, columns=cols, index=df.index)

        if self.target_column in df.columns:
            values = df[self.target_column].copy()
            if values.dtype == object:
                mapped = values.map({"Yes": 1, "No": 0})
                if mapped.notna().all():
                    result[self.target_column] = mapped.astype(int).values
                else:
                    result[self.target_column] = values.values
            else:
                result[self.target_column] = values.values

        return result




    #  Variables dérivées sans modifier l'original
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Age" in df.columns and "age_group" not in df.columns:
            df["age_group"] = pd.cut(df["Age"], bins=[0, 40, 60, 200], labels=["young", "middle", "senior"])
        if "Blood Pressure" in df.columns and "bp_category" not in df.columns:
            df["bp_category"] = pd.cut(df["Blood Pressure"], bins=[0, 120, 140, 1000], labels=["normal", "high", "very_high"])
        return df




    # Combinaison des étapes fit et transform
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
