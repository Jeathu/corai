from pathlib import Path
import pandas as pd
from loguru import logger


import typer

from corai.config import PROCESSED_DATA_DIR, RAW_DATA_DIR




app = typer.Typer()



# Méthode pour charger les données
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("le fichier chargé n'est pas un DataFrame pandas")
    return df




def split_features_target(df: pd.DataFrame, target: str = "Heart Disease Status") -> tuple[pd.DataFrame, pd.Series]:
    """
    Séparation le DataFrame en caractéristiques (X)features  et cible (y)

        Exceptions:
            KeyError: Si la colonne cible n'est pas présente dans le DataFrame
            TypeError: Si df n'est pas un DataFrame pandas
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("le dataframe fourni n'est pas un DataFrame pandas")

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    X = df.loc[:, df.columns != target].copy()
    y = df.loc[:, target].copy()
    return X, y
