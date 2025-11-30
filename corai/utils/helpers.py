"""
Fonctions utilitaires pour le projet corai.
"""

import time
import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
from functools import wraps

from loguru import logger



def ensure_dir(path: Path) -> Path:
    """
      Crée un répertoire s'il n'existe pas.

      Args:
        path: Chemin du répertoire

      Returns:
        Le chemin du répertoire
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path




def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
      Sauvegarde un dictionnaire dans un fichier JSON.

      Args:
        data: Données à sauvegarder
        filepath: Chemin du fichier de sortie
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"JSON sauvegardé: {filepath}")




def load_json(filepath: Path) -> Dict[str, Any]:
    """
       Charge un fichier JSON.

       Args:
        filepath: Chemin du fichier JSON

       Returns:
        Dictionnaire contenant les données
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"JSON chargé: {filepath}")
    return data




def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """
      Retourne un timestamp formaté.

      Args:
        format: Format du timestamp (défaut: YYYYMMDD_HHMMSS)

      Returns:
        Timestamp formaté
    """
    return datetime.now().strftime(format)




def timer(func):
    """
       Décorateur pour mesurer le temps d'exécution d'une fonction.

       Usage:
         @timer
         def my_function():
             pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Début de {func.__name__}...")

        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"{func.__name__} terminé en {elapsed_time:.2f} secondes")
        return result
    return wrapper




def format_size(size_bytes: int) -> str:
    """
      Formate une taille en bytes en format lisible.

      Args:
        size_bytes: Taille en bytes

      Returns:
        Taille formatée (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"




def validate_file_exists(filepath: Path) -> bool:
    """
       Vérifie qu'un fichier existe.

       Args:
           filepath: Chemin du fichier

       Returns:
            True si le fichier existe, False sinon
    """
    filepath = Path(filepath)
    exists = filepath.exists() and filepath.is_file()

    if not exists:
        logger.warning(f"Fichier non trouvé: {filepath}")
    return exists




def validate_dir_exists(dirpath: Path) -> bool:
    """
       Vérifie qu'un répertoire existe.

       Args:
          dirpath: Chemin du répertoire

       Returns:
          True si le répertoire existe, False sinon
    """
    dirpath = Path(dirpath)
    exists = dirpath.exists() and dirpath.is_dir()

    if not exists:
        logger.warning(f"Répertoire non trouvé: {dirpath}")
    return exists




def get_file_size(filepath: Path) -> int:
    """
       Retourne la taille d'un fichier en bytes.

       Args:
          filepath: Chemin du fichier

       Returns:
          Taille du fichier en bytes
    """
    filepath = Path(filepath)

    if not validate_file_exists(filepath):
        return 0
    return filepath.stat().st_size




def count_files_in_dir(dirpath: Path, pattern: str = "*") -> int:
    """
       Compte le nombre de fichiers dans un répertoire.

       Args:
          dirpath: Chemin du répertoire
          pattern: Pattern pour filtrer les fichiers (défaut: tous)

       Returns:
          Nombre de fichiers
    """
    dirpath = Path(dirpath)
    if not validate_dir_exists(dirpath):
        return 0
    return len(list(dirpath.glob(pattern)))
