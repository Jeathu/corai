from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger
import typer
from imblearn.over_sampling import SMOTE

from corai.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from corai.preprocessing.dataset import load_data
from corai.preprocessing.cardiac_feature_pipeline import HeartDiseasePreprocessor

app = typer.Typer()



class DataDiagnosticsPreprocessor:
    """
    Orchestrateur du pipeline de prétraitement complet et 
    produit un CSV prêt pour entraîner des modèles.
    """


    # Constructor
    def __init__(self, target_column: str = "Heart Disease"): # colonne cible par défaut
        self.target_column = target_column
        self.preprocessor = HeartDiseasePreprocessor(target_column=self.target_column)
        self.df: Optional[pd.DataFrame] = None
        self.X_transformed: Optional[pd.DataFrame] = None
        self.y_arr: Optional[pd.Series] = None




    # Méthode pour charger les données qui vient de dataset.py
    def load(self, path: Path) -> pd.DataFrame:
        """Charge les données depuis un fichier CSV."""
        self.df = load_data(path)
        logger.info(f"Loaded {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df




    # Méthode pour supprimer les doublons
    def remove_duplicates(self) -> int:
        """Supprime les lignes dupliquées du DataFrame."""
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        before = self.df.shape[0]
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        after = self.df.shape[0]
        removed = before - after
        logger.info(f"Removed {removed} duplicate rows ({before} -> {after})")
        return removed




    # Méthode pour ajuster et transformer les données avec le préprocesseur
    def fit_transform_preprocessor(self) -> pd.DataFrame:
        """Applique le préprocesseur HeartDiseasePreprocessor."""
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        processed = self.preprocessor.fit_transform(self.df)

        if self.target_column in processed.columns:
            self.X_transformed = processed.drop(columns=[self.target_column])
            # Convertir y en int (0 ou 1)
            y_raw = processed[self.target_column]
            self.y_arr = y_raw.astype(int).values  # ← FIX ICI
        else:
            self.X_transformed = processed
            self.y_arr = None

        logger.info(f"Transformed: X={self.X_transformed.shape}")
        return self.X_transformed




    # Méthode pour appliquer SMOTE pour rééquilibrer les classes
    def apply_smote(self, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """Applique SMOTE pour rééquilibrer les classes."""
        if self.X_transformed is None or self.y_arr is None:
            raise RuntimeError("Data not transformed. Call fit_transform_preprocessor() first.")

        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(self.X_transformed.values, self.y_arr)

        X_res_df = pd.DataFrame(X_res, columns=self.X_transformed.columns)
        y_res_series = pd.Series(y_res, name=self.target_column)

        logger.info(f"After SMOTE: X={X_res_df.shape}, y={y_res_series.shape}")
        return X_res_df, y_res_series




    # Méthode pour exporter les données traitées vers un fichier CSV
    def export_processed(self, X_df: pd.DataFrame, y_series: pd.Series, output_path: Path) -> Path:
        """Exporte les données traitées vers un fichier CSV."""
        out_df = X_df.copy()
        out_df[self.target_column] = y_series.values
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        logger.success(f"Saved processed dataset to {output_path}")
        return output_path








@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "heart_disease_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_heart_disease_v0.csv",
    apply_smote: bool = False, # false par défaut pour ne pas forcer l'utilisation de SMOTE
) -> None:
    """Prétraitement des données pour entraînement de modèles."""
    pipeline = DataDiagnosticsPreprocessor()
    pipeline.load(input_path)
    pipeline.remove_duplicates()
    pipeline.fit_transform_preprocessor()

    if apply_smote:
        X_res, y_res = pipeline.apply_smote()
    else:
        X_res = pipeline.X_transformed
        y_res = pd.Series(pipeline.y_arr, name=pipeline.target_column)

    pipeline.export_processed(X_res, y_res, output_path)
    typer.echo(f"CSV prêt pour entraînement: {output_path}")


if __name__ == "__main__":
    app()




# python -m corai.preprocessing.data_diagnostics