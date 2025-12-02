from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from corai.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from corai.preprocessing.data_loader import load_data, duplicate_dataframe, split_features_target
from corai.preprocessing.feature_transformer import HeartDiseasePreprocessor


app = typer.Typer()


class DataDiagnosticsPreprocessor:
    """
    Orchestrateur du pipeline de prétraitement complet et 
    produit un CSV prêt pour entraîner des modèles.
    """


    def __init__(self, target_column: str = "Heart Disease"):
        self.target_column = target_column
        self.preprocessor = HeartDiseasePreprocessor()
        self.df: Optional[pd.DataFrame] = None
        self.X_transformed: Optional[pd.DataFrame] = None
        self.y_arr: Optional[pd.Series] = None



    # wrapper pour load_data
    def load(self, path: Path) -> pd.DataFrame:
        """Charge les données depuis un fichier CSV."""
        self.df = load_data(path)
        return self.df



    # wrapper pour duplicate_dataframe
    def remove_duplicates(self) -> int:
        """ 
        Wrapper: Supprime les lignes dupliquées du DataFrame en appelant 
        la fonction duplicate_dataframe (data_loder.py).
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        before = self.df.shape[0]
        self.df = duplicate_dataframe(self.df)
        after = self.df.shape[0]
        removed = before - after
        return removed



    # wrapper pour fit_transform du préprocesseur
    def fit_transform_preprocessor(self) -> pd.DataFrame:
        """Ajuste et transforme les données avec le préprocesseur."""
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        # Séparer les caractéristiques et la cible
        try:
            X_raw,y_raw = split_features_target(self.df, target=self.target_column)
        except KeyError:
            X_raw, y_raw = self.df.copy(), None

        # Appliquer le préprocesseur (HeartDiseasePreprocessor)
        self.preprocessor.fit(X_raw)
        self.X_transformed = self.preprocessor.transform(X_raw)

        if y_raw is not None:
                mapping = { "Yes": 1, "No": 0 }
                self.y_arr = y_raw.map(mapping).fillna(y_raw).astype(int)
        else:
            self.y_arr = None
        return self.X_transformed



    # méthode pour exporter les données traitées
    def export_processed(self, X_df: pd.DataFrame, y_series: pd.Series, output_path: Path) -> Path:
        """Exporte les données traitées vers un fichier CSV."""
        out_df = X_df.copy()
        out_df[self.target_column] = y_series.values
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        return output_path





@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "heart_disease_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_heart_disease_v0.csv",
) -> None:
    """Prétraitement des données pour entraînement de modèles."""
    pipeline = DataDiagnosticsPreprocessor()
    pipeline.load(input_path)
    pipeline.remove_duplicates()
    pipeline.fit_transform_preprocessor()

    # Utiliser les données transformées
    X_res = pipeline.X_transformed
    y_res = pipeline.y_arr

    pipeline.export_processed(X_res, y_res, output_path)
    typer.echo(f"CSV prêt pour entraînement: {output_path}")


if __name__ == "__main__":
    app()


# python -m corai.preprocessing.preprocessing_pipeline 