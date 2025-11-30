import typer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from tqdm import tqdm



from corai.config import FIGURES_DIR    # report/figures/raw_data_png

app = typer.Typer()



# Chemin vers les données brutes
RAW_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"



# Configuration de style matplotlib
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10





"""Classe pour créer des visualisations avec Matplotlib."""
class VisualisateurMatplotlib:

    # Initialise le visualiseur avec raw DataFrame brut
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Visualiseur initialisé — {len(self.numerical_cols)} numériques, {len(self.categorical_cols)} catégoriques")




    # Crée un histogramme pour la distribution d'une colonne numérique
    def tracer_distribution_numerique(self, colonne: str) -> plt.Figure:
        logger.info(f"Création du graphique de distribution pour {colonne}...")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.hist(self.df[colonne], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f"Distribution de {colonne}", fontsize=14, fontweight='bold')
        ax.set_xlabel(colonne, fontsize=12)
        ax.set_ylabel('Fréquence', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig




    # Crée un box plot pour une colonne numérique
    def tracer_boite(self, colonne: str) -> plt.Figure:
        logger.info(f"Création du box plot pour {colonne}...")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.boxplot(self.df[colonne], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_title(f"Box Plot de {colonne}", fontsize=14, fontweight='bold')
        ax.set_ylabel(colonne, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig




    # Crée un bar chart pour la distribution d'une colonne catégorique
    def tracer_distribution_categorique(self, colonne: str) -> plt.Figure:
        logger.info(f"Création du graphique de distribution pour {colonne}...")

        fig, ax = plt.subplots(figsize=(12, 5))

        comptages = self.df[colonne].value_counts()
        ax.bar(comptages.index, comptages.values, color='coral', edgecolor='black', alpha=0.7)
        ax.set_title(f"Distribution de {colonne}", fontsize=14, fontweight='bold')
        ax.set_xlabel(colonne, fontsize=12)
        ax.set_ylabel('Nombre', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(comptages.values):
            ax.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        return fig




    # Crée un graphique des valeurs manquantes
    def tracer_valeurs_manquantes(self) -> plt.Figure:
        logger.info("Création du graphique des valeurs manquantes...")

        donnees_manquantes = self.df.isnull().sum()
        donnees_manquantes = donnees_manquantes[donnees_manquantes > 0].sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))

        if len(donnees_manquantes) > 0:
            ax.bar(range(len(donnees_manquantes)), donnees_manquantes.values, 
                   color='salmon', edgecolor='black', alpha=0.7)
            ax.set_xticks(range(len(donnees_manquantes)))
            ax.set_xticklabels(donnees_manquantes.index, rotation=45, ha='right')
            ax.set_title("Valeurs Manquantes par Colonne", fontsize=14, fontweight='bold')
            ax.set_ylabel('Nombre de valeurs manquantes', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            for i, v in enumerate(donnees_manquantes.values):
                ax.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Aucune valeur manquante détectée', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        return fig




    # Crée une heatmap des corrélations
    def tracer_heatmap_correlation(self) -> plt.Figure:
        logger.info("Création de la heatmap de corrélation...")

        fig, ax = plt.subplots(figsize=(12, 10))

        correlation_matrix = self.df[self.numerical_cols].corr()

        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, cbar_kws={'label': 'Corrélation'},
                   square=True, linewidths=0.5, ax=ax)
        ax.set_title("Heatmap de Corrélation", fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig




    # Crée un scatter plot entre deux colonnes
    def tracer_nuage_points(self, col_x: str, col_y: str) -> plt.Figure:
        logger.info(f"Création du nuage de points: {col_x} vs {col_y}...")

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.scatter(self.df[col_x], self.df[col_y], alpha=0.5, color='steelblue', edgecolors='k')
        ax.set_xlabel(col_x, fontsize=12)
        ax.set_ylabel(col_y, fontsize=12)
        ax.set_title(f"{col_x} vs {col_y}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        z = np.polyfit(self.df[col_x].dropna(), self.df[col_y].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(self.df[col_x].sort_values(), p(self.df[col_x].sort_values()), 
               "r--", linewidth=2, label='Tendance')
        ax.legend()

        plt.tight_layout()
        return fig




    # Crée un tableau avec les statistiques descriptives
    def tracer_statistiques_descriptives(self) -> plt.Figure:
        logger.info("Création des statistiques descriptives...")

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')

        stats = self.df[self.numerical_cols].describe().T
        stats = stats.round(2)

        # Créer un tableau
        table_data = []
        table_data.append(['Colonne'] + list(stats.columns))
        for idx, row in stats.iterrows():
            table_data.append([idx] + [str(val) for val in row.values])

        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.15] * len(table_data[0]))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Colorer l'en-tête
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title("Statistiques Descriptives (Données Brutes)", 
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig




    # Crée une matrice de paires pour les 4 premières colonnes numériques
    def tracer_paires(self) -> plt.Figure:
        logger.info("Création de la matrice de paires...")

        colonnes = self.numerical_cols[:4]
        n = len(colonnes)

        fig, axes = plt.subplots(n, n, figsize=(14, 12))

        for i, col_y in enumerate(colonnes):
            for j, col_x in enumerate(colonnes):
                ax = axes[i, j]

                if i == j:
                    # Diagonale : histogramme
                    ax.hist(self.df[col_x], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                    ax.set_ylabel('Fréquence' if j == 0 else '')
                else:
                    # Scatter plot
                    ax.scatter(self.df[col_x], self.df[col_y], alpha=0.5, color='steelblue', s=20)

                ax.set_xlabel(col_x if i == n-1 else '')
                ax.set_ylabel(col_y if j == 0 else '')
                ax.grid(True, alpha=0.3)

        fig.suptitle("Matrice de Paires (Données Brutes)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig




    # Génère toutes les visualisations et les sauvegarde en PNG
    def generer_toutes_visualisations(self, chemin_sortie: Path):
        """Génère et sauvegarde toutes les visualisations."""
        logger.info("Génération de toutes les visualisations avec Matplotlib...")

        # Créer le dossier de sortie
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

        # Valeurs manquantes
        logger.info("Traitement du graphique des valeurs manquantes...")
        fig = self.tracer_valeurs_manquantes()
        fig.savefig(chemin_sortie.parent / "01_valeurs_manquantes.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Statistiques descriptives
        logger.info("Traitement des statistiques descriptives...")
        fig = self.tracer_statistiques_descriptives()
        fig.savefig(chemin_sortie.parent / "02_statistiques_descriptives.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Distributions numériques
        logger.info("Traitement des distributions numériques...")
        for idx, col in enumerate(tqdm(self.numerical_cols[:5], desc="Distributions numériques")):
            fig = self.tracer_distribution_numerique(col)
            fig.savefig(chemin_sortie.parent / f"03_distribution_{idx:02d}_{col}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

        #  Heatmap de corrélation
        logger.info("Traitement de la heatmap de corrélation...")
        fig = self.tracer_heatmap_correlation()
        fig.savefig(chemin_sortie.parent / "04_heatmap_correlation.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Distributions catégoriques
        if len(self.categorical_cols) > 0:
            logger.info("Traitement des distributions catégoriques...")
            for idx, col in enumerate(tqdm(self.categorical_cols[:4], desc="Distributions catégoriques")):
                fig = self.tracer_distribution_categorique(col)
                fig.savefig(chemin_sortie.parent / f"05_distribution_cat_{idx:02d}_{col}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)

        # Matrice de paires
        logger.info("Traitement de la matrice de paires...")
        fig = self.tracer_paires()
        fig.savefig(chemin_sortie.parent / "06_matrice_paires.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Générer un HTML pour afficher les images
        logger.info("Génération du rapport HTML...")
        self._generer_rapport_html(chemin_sortie)

        logger.success(f"Toutes les visualisations sauvegardées dans {chemin_sortie.parent}")





    def _generer_rapport_html(self, chemin_sortie: Path):
        """Génère un rapport HTML avec les images PNG."""
        images = sorted([f for f in chemin_sortie.parent.glob("*.png")])

        contenu_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Rapport d'Analyse - Matplotlib</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    text-align: center;
                    border-bottom: 3px solid #007bff;
                    padding-bottom: 15px;
                }
                .info-box {
                    background-color: #e7f3ff;
                    border-left: 4px solid #007bff;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 4px;
                }
                .figure {
                    margin-bottom: 40px;
                    text-align: center;
                }
                .figure img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h2 {
                    color: #666;
                    border-bottom: 2px solid #007bff;
                    padding-bottom: 10px;
                    margin-top: 30px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Rapport d'Analyse des Données Brutes (Matplotlib)</h1>
                <div class="info-box">
                    <strong>Note:</strong> Ce rapport analyse les données brutes (non prétraitées) du fichier heart_disease.csv avec Matplotlib
                </div>
        """

        for idx, img_path in enumerate(images):
            img_name = img_path.name
            contenu_html += f"""
                <div class="figure">
                    <h2>{img_name}</h2>
                    <img src="{img_name}" alt="{img_name}">
                </div>
            """

        contenu_html += """
            </div>
        </body>
        </html>
        """

        with open(chemin_sortie, "w", encoding='utf-8') as f:
            f.write(contenu_html)

        logger.success(f"Rapport HTML généré: {chemin_sortie}")







@app.command()
def principal(
    chemin_entree: Path = RAW_DATA_DIR / "heart_disease.csv",
    chemin_sortie: Path = FIGURES_DIR / "raw_data_png/raw_data_report.html",
):
    """Fonction principale pour générer les visualisations avec Matplotlib."""
    logger.info(f"Chargement du dataset brut depuis {chemin_entree}...")

    try:
        df = pd.read_csv(chemin_entree)
        logger.info(f"Dataset brut chargé avec succès. Forme: {df.shape}")

    except FileNotFoundError:
        logger.error(f"Fichier non trouvé: {chemin_entree}")
        return
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        return

    # Créer le visualiseur
    visualiseur = VisualisateurMatplotlib(df)

    # Générer toutes les visualisations
    visualiseur.generer_toutes_visualisations(chemin_sortie)




if __name__ == "__main__":
    app()


# python -m corai.analytics.visualizations.raw_data_visualizations
# start reports/figures/raw_data_png/raw_data_report.html