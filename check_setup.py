"""
Script de vérification de l'installation et de l'environnement du projet CorAI.
"""

import sys
from pathlib import Path
from loguru import logger

def check_python_version():
    """Vérifie la version de Python."""
    logger.info("Vérification de la version Python...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        logger.success(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        logger.warning(f"⚠️  Python {version.major}.{version.minor}.{version.micro} (recommandé: 3.11+)")
        return False

def check_packages():
    """Vérifie les packages importants."""
    logger.info("Vérification des packages...")
    packages = [
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "loguru",
        "typer"
    ]
    
    missing = []
    for package in packages:
        try:
            __import__(package)
            logger.success(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package} - MANQUANT")
            missing.append(package)
    
    return len(missing) == 0

def check_data_files():
    """Vérifie la présence des fichiers de données."""
    logger.info("Vérification des fichiers de données...")
    
    files_to_check = [
        ("data/raw/heart_disease_dataset.csv", True),  # Requis
        ("data/processed/processed_heart_disease_v0.csv", False),  # Optionnel
        ("models/heart_disease_model.pkl", False),  # Optionnel
    ]
    
    all_ok = True
    for file_path, required in files_to_check:
        path = Path(file_path)
        if path.exists():
            logger.success(f"✅ {file_path}")
        else:
            if required:
                logger.error(f"❌ {file_path} - REQUIS")
                all_ok = False
            else:
                logger.info(f"ℹ️  {file_path} - Non trouvé (sera créé)")
    
    return all_ok

def check_project_structure():
    """Vérifie la structure du projet."""
    logger.info("Vérification de la structure du projet...")
    
    required_dirs = [
        "corai",
        "corai/preprocessing",
        "corai/modeling",
        "corai/analytics",
        "data",
        "data/raw",
        "data/processed",
        "models",
        "reports"
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            logger.success(f"✅ {dir_path}/")
        else:
            logger.error(f"❌ {dir_path}/ - MANQUANT")
            all_ok = False
    
    return all_ok

def check_modules():
    """Vérifie que les modules du projet sont importables."""
    logger.info("Vérification des modules du projet...")
    
    modules = [
        "corai.config",
        "corai.preprocessing.data_loader",
        "corai.preprocessing.feature_transformer",
        "corai.modeling.train",
        "corai.modeling.predict",
        "corai.modeling.evaluate",
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            logger.success(f"✅ {module}")
        except ImportError as e:
            logger.error(f"❌ {module} - {str(e)}")
            all_ok = False
    
    return all_ok

def main():
    """Fonction principale de vérification."""
    logger.info("=" * 80)
    logger.info("🔍 VÉRIFICATION DE L'ENVIRONNEMENT CORAI")
    logger.info("=" * 80)
    
    checks = {
        "Version Python": check_python_version(),
        "Packages Python": check_packages(),
        "Structure du projet": check_project_structure(),
        "Fichiers de données": check_data_files(),
        "Modules du projet": check_modules(),
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ")
    logger.info("=" * 80)
    
    for check_name, result in checks.items():
        if result:
            logger.success(f"✅ {check_name}")
        else:
            logger.warning(f"⚠️  {check_name}")
    
    all_passed = all(checks.values())
    
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.success("✅ TOUTES LES VÉRIFICATIONS SONT PASSÉES!")
        logger.info("\nVous pouvez commencer:")
        logger.info("  python -m corai.demo_model")
        logger.info("  python -m corai.pipeline_complete")
    else:
        logger.warning("⚠️  CERTAINES VÉRIFICATIONS ONT ÉCHOUÉ")
        logger.info("\nActions recommandées:")
        logger.info("  1. Installer les packages: pip install -r requirements.txt")
        logger.info("  2. Vérifier les fichiers de données dans data/raw/")
        logger.info("  3. Consulter USAGE_GUIDE.md pour plus d'informations")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
