#from corai import config  # noqa: F401

"""CoRAI - Heart Disease Prediction Project"""

__version__ = "0.1.0"

from corai.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from corai.preprocessing.heart_disease_preprocessor import HeartDiseasePreprocessor

__all__ = ["PROCESSED_DATA_DIR", "RAW_DATA_DIR", "HeartDiseasePreprocessor"]