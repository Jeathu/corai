import typer
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger


from corai.config import PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()



"""Classe pour effectuer une analyse exploratoire des données"""
class process_data_analytics:

     #Initialise l'analyseur
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()




    # Vérification des valeurs manquantes dans le DataFrame
    def check_missing_values(self) -> pd.DataFrame:
        logger.info("Checking for missing values...")
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum().values / len(self.df) * 100).round(2)
        })
        return missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)




    # Calcule les statistiques descriptives pour les colonnes numériques
    def get_statistics(self) -> pd.DataFrame:
        logger.info("Calculating descriptive statistics...")
        return self.df[self.numerical_cols].describe().T


    # Calcule la matrice de corrélation pour les colonnes numériques
    def get_correlations(self) -> pd.DataFrame:
        logger.info("Calculating correlations...")
        return self.df[self.numerical_cols].corr()




    # Résume les colonnes catégoriques
    def get_categorical_summary(self) -> dict:
        logger.info("Summarizing categorical columns...")
        summary = {}
        for col in self.categorical_cols:
            summary[col] = self.df[col].value_counts().to_dict()
        return summary





    #  Génère un rapport complet d'analyse
    def generate_analysis_report(self) -> dict:
        logger.info("Generating analysis report...")
        report = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'missing_values': self.check_missing_values(),
            'statistics': self.get_statistics(),
            'correlations': self.get_correlations(),
            'categorical_summary': self.get_categorical_summary()
        }
        return report







@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "processed_heart_disease_v0.csv",
    output_path: Path = REPORTS_DIR / "features_summary.csv",
):
    """Fonction principale pour l'analyse des données."""
    logger.info(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # initialiser l'analyseur de données
    analyzer = process_data_analytics(df)

    # Générer le rapport d'analyse
    logger.info("Performing exploratory data analysis...")
    report = analyzer.generate_analysis_report()

    # Afficher les résultats
    logger.info(f"Dataset shape: {report['shape']}")
    logger.info(f"Columns: {report['columns']}")
    logger.info(f"Missing values:\n{report['missing_values']}")
    logger.info(f"Statistics:\n{report['statistics']}")

    # Sauvegarder les statistiques
    report['statistics'].to_csv(output_path)
    logger.success(f"Features summary saved to {output_path}")


if __name__ == "__main__":
    app()


# python run_analysis.py