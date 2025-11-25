from pathlib import Path
import pandas as pd

from loguru import logger
from tqdm import tqdm
import typer

from corai.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from corai.preprocessing.cardiac_feature_pipeline import HeartDiseasePreprocessor 



app = typer.Typer()



# Méthode pour charger les données
def load_data(path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Loaded data is not a pandas DataFrame")
    return df




@app.command()
def main(
    # ---- IMPORTATION DE DONNÉES ----
    input_path: Path = RAW_DATA_DIR / "heart_disease.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_heart_disease.csv",
):
    df = load_data(input_path)
    preprocessor = HeartDiseasePreprocessor()
    df = preprocessor.fit_transform(df)

    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    typer.echo("Data preprocessing completed successfully.")




if __name__ == "__main__":
    app()




# Pour exécuter ce script:
    # python -m corai.dataset