from pathlib import Path
import pandas as pd
from loguru import logger


import typer
app = typer.Typer()


# Méthode pour charger les données
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("le fichier chargé n'est pas un DataFrame pandas")
    return df



# Méthode pour séparer features et target
def split_features_target(df: pd.DataFrame, target: str = "Heart Disease") -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare le DataFrame en features (X) et cible (y).

    Exceptions:
        KeyError: Si la colonne cible n'est pas présente
        TypeError: Si df n'est pas un DataFrame pandas
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Le dataframe fourni n'est pas un DataFrame pandas")

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Available: {list(df.columns)}")

    X = df.drop(columns=[target])
    y = df[target].copy()
    return X, y



# Méthode pour supprimer les doublons
def duplicate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes dupliquées du DataFrame."""
    initial_shape = df.shape
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Removed {initial_shape[0] - df_cleaned.shape[0]} duplicate rows.")
    return df_cleaned


