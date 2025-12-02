"""
Module d'analyse exploratoire des données (EDA).
Génère des statistiques descriptives et des analyses des données brutes.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import typer
from loguru import logger

from corai.config import RAW_DATA_DIR, REPORTS_DIR
from corai.preprocessing.data_loader import load_data

app = typer.Typer()


class ExploratoryDataAnalysis:
    """Classe pour effectuer l'analyse exploratoire des données."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialise l'analyse exploratoire.

        Args:
            df: DataFrame pandas à analyser
        """
        self.df = df
        self.report: Dict[str, Any] = {}



    def basic_info(self) -> Dict[str, Any]:
        """Retourne les informations de base du dataset."""
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
        }
        self.report["basic_info"] = info
        return info



    def missing_values(self) -> pd.DataFrame:
        """Analyse les valeurs manquantes."""
        missing = pd.DataFrame({
            "missing_count": self.df.isnull().sum(),
            "missing_percentage": (self.df.isnull().sum() / len(self.df)) * 100
        })
        missing = missing[missing["missing_count"] > 0].sort_values(
            "missing_count", ascending=False
        )
        self.report["missing_values"] = missing
        return missing



    def duplicates_info(self) -> Dict[str, Any]:
        """Analyse les doublons."""
        dup_info = {
            "total_duplicates": self.df.duplicated().sum(),
            "percentage": (self.df.duplicated().sum() / len(self.df)) * 100
        }
        self.report["duplicates"] = dup_info
        return dup_info



    def numerical_summary(self) -> pd.DataFrame:
        """Statistiques descriptives pour les colonnes numériques."""
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])
        if numeric_df.empty:
            logger.warning("Aucune colonne numérique trouvée")
            return pd.DataFrame()

        summary = numeric_df.describe()
        self.report["numerical_summary"] = summary
        return summary



    def categorical_summary(self) -> Dict[str, pd.Series]:
        """Analyse des colonnes catégorielles."""
        cat_df = self.df.select_dtypes(include=["object", "category"])
        if cat_df.empty:
            logger.warning("Aucune colonne catégorielle trouvée")
            return {}

        cat_summary = {}
        for col in cat_df.columns:
            cat_summary[col] = self.df[col].value_counts()

        self.report["categorical_summary"] = cat_summary
        return cat_summary



    def correlations(self, threshold: float = 0.5) -> Optional[pd.DataFrame]:
        """
        Calcule la matrice de corrélation pour les variables numériques.

        Args:
            threshold: Seuil de corrélation à afficher
        """
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            logger.warning("Pas assez de colonnes numériques pour calculer les corrélations")
            return None

        corr_matrix = numeric_df.corr()

        # Filtrer les corrélations importantes
        mask = (corr_matrix.abs() >= threshold) & (corr_matrix != 1.0)
        high_corr = corr_matrix[mask].stack().sort_values(ascending=False)

        self.report["correlations"] = {
            "matrix": corr_matrix,
            "high_correlations": high_corr
        }
        return corr_matrix



    def generate_full_report(self) -> Dict[str, Any]:
        """Génère un rapport complet d'analyse exploratoire."""
        logger.info("Génération du rapport d'analyse exploratoire...")

        self.basic_info()
        self.missing_values()
        self.duplicates_info()
        self.numerical_summary()
        self.categorical_summary()
        self.correlations()

        logger.success("Rapport d'analyse exploratoire généré avec succès")
        return self.report



    def save_report(self, output_path: Path) -> None:
        """Sauvegarde le rapport dans un fichier texte."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT D'ANALYSE EXPLORATOIRE DES DONNÉES\n")
            f.write("=" * 80 + "\n\n")

            # Informations de base
            if "basic_info" in self.report:
                f.write("1. INFORMATIONS DE BASE\n")
                f.write("-" * 80 + "\n")
                info = self.report["basic_info"]
                f.write(f"Shape: {info['shape']}\n")
                f.write(f"Memory usage: {info['memory_usage_mb']:.2f} MB\n\n")

            # Valeurs manquantes
            if "missing_values" in self.report and not self.report["missing_values"].empty:
                f.write("2. VALEURS MANQUANTES\n")
                f.write("-" * 80 + "\n")
                f.write(self.report["missing_values"].to_string())
                f.write("\n\n")

            # Doublons
            if "duplicates" in self.report:
                f.write("3. DOUBLONS\n")
                f.write("-" * 80 + "\n")
                dup = self.report["duplicates"]
                f.write(f"Total duplicates: {dup['total_duplicates']}\n")
                f.write(f"Percentage: {dup['percentage']:.2f}%\n\n")

            # Résumé numérique
            if "numerical_summary" in self.report and not self.report["numerical_summary"].empty:
                f.write("4. STATISTIQUES DESCRIPTIVES (VARIABLES NUMÉRIQUES)\n")
                f.write("-" * 80 + "\n")
                f.write(self.report["numerical_summary"].to_string())
                f.write("\n\n")

            # Résumé catégoriel
            if "categorical_summary" in self.report:
                f.write("5. DISTRIBUTION DES VARIABLES CATÉGORIELLES\n")
                f.write("-" * 80 + "\n")
                for col, counts in self.report["categorical_summary"].items():
                    f.write(f"\n{col}:\n")
                    f.write(counts.to_string())
                    f.write("\n")
        logger.success(f"Rapport sauvegardé: {output_path}")



def perform_eda(
    data_path: Path,
    output_path: Optional[Path] = None
) -> ExploratoryDataAnalysis:
    """
    Effectue une analyse exploratoire complète des données.

    Args:
        data_path: Chemin vers le fichier de données
        output_path: Chemin optionnel pour sauvegarder le rapport

    Returns:
        Instance de ExploratoryDataAnalysis avec le rapport généré
    """
    df = load_data(data_path)
    eda = ExploratoryDataAnalysis(df)
    eda.generate_full_report()

    if output_path:
        eda.save_report(output_path)
    return eda




@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "heart_disease_dataset.csv",
    output_path: Path = REPORTS_DIR / "eda_report.txt",
) -> None:
    """Execute l'analyse exploratoire des données et sauvegarde le rapport."""
    perform_eda(input_path, output_path)
    typer.echo(f"Analyse exploratoire terminée. Rapport: {output_path}")


if __name__ == "__main__":
    app()
