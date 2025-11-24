import pandas as pd
import numpy as np
from pathlib import Path
import pytest
from corai.preprocessing.heart_disease_preprocessor import HeartDiseasePreprocessor

class TestHeartDiseasePreprocessor:
    
    @pytest.fixture
    def sample_data(self):
        """Crée un dataset de test"""
        return pd.DataFrame({
            "Age": [56.0, 69.0, 46.0, 32.0, 60.0],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
            "Blood Pressure": [153.0, 146.0, 126.0, 122.0, 166.0],
            "Cholesterol Level": [155.0, 286.0, 216.0, 293.0, 242.0],
            "Exercise Habits": ["High", "High", "Low", "High", "Low"],
            "Smoking": ["Yes", "No", "No", "Yes", "Yes"],
            "Family Heart Disease": ["Yes", "Yes", "No", "Yes", "Yes"],
            "Diabetes": ["No", "Yes", "No", "No", "Yes"],
            "BMI": [24.99, 25.22, 29.86, 24.13, 20.49],
            "High Blood Pressure": ["Yes", "No", "No", "Yes", "Yes"],
            "Low HDL Cholesterol": ["No", "No", "Yes", "Yes", "No"],
            "High LDL Cholesterol": ["No", "No", "Yes", "Yes", "No"],
            "Alcohol Consumption": ["High", "Medium", "Low", "Low", "Low"],
            "Stress Level": ["Medium", "High", "Low", "High", "High"],
            "Sleep Hours": [7.63, 8.74, 4.44, 5.25, 7.03],
            "Sugar Consumption": ["Medium", "Medium", "Low", "High", "High"],
            "Triglyceride Level": [342.0, 133.0, 393.0, 293.0, 263.0],
            "Fasting Blood Sugar": [np.nan, 157.0, 92.0, 94.0, 154.0],
            "CRP Level": [12.97, 9.36, 12.71, 12.51, 10.38],
            "Homocysteine Level": [12.39, 19.30, 11.23, 5.96, 8.15],
            "Heart Disease Status": ["No", "No", "No", "No", "No"],
        })
    
    def test_fit(self, sample_data):
        """Test que le fit fonctionne correctement"""
        preprocessor = HeartDiseasePreprocessor()
        result = preprocessor.fit(sample_data)
        
        assert preprocessor._fitted is True
        assert result is preprocessor  # Vérifie que fit retourne self
    
    def test_transform(self, sample_data):
        """Test que transform fonctionne après fit"""
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(sample_data)
        transformed = preprocessor.transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert "Heart Disease Status" in transformed.columns
        assert len(transformed) == len(sample_data)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform en une seule étape"""
        preprocessor = HeartDiseasePreprocessor()
        transformed = preprocessor.fit_transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_data)
        assert "Heart Disease Status" in transformed.columns
    
    def test_feature_engineering(self, sample_data):
        """Test que les features engineered sont créées"""
        preprocessor = HeartDiseasePreprocessor()
        transformed = preprocessor.fit_transform(sample_data)
        
        # Vérifier qu'il y a des colonnes liées aux features engineered
        feature_names = transformed.columns.tolist()
        assert any("age_group" in f for f in feature_names)
        assert any("bp_category" in f for f in feature_names)
    
    def test_target_mapping(self, sample_data):
        """Test la conversion Yes/No en 1/0 pour le target"""
        preprocessor = HeartDiseasePreprocessor()
        transformed = preprocessor.fit_transform(sample_data)
        
        target_values = transformed["Heart Disease Status"].values
        assert all(v in [0, 1] for v in target_values)
    
    def test_missing_values_handling(self):
        """Test la gestion des valeurs manquantes"""
        data = pd.DataFrame({
            "Age": [56.0, np.nan, 46.0],
            "Gender": ["Male", "Female", np.nan],
            "Blood Pressure": [153.0, 146.0, 126.0],
            "Cholesterol Level": [155.0, 286.0, 216.0],
            "Exercise Habits": ["High", "High", "Low"],
            "Smoking": ["Yes", "No", "No"],
            "Family Heart Disease": ["Yes", "Yes", "No"],
            "Diabetes": ["No", "Yes", "No"],
            "BMI": [24.99, 25.22, 29.86],
            "High Blood Pressure": ["Yes", "No", "No"],
            "Low HDL Cholesterol": ["No", "No", "Yes"],
            "High LDL Cholesterol": ["No", "No", "Yes"],
            "Alcohol Consumption": ["High", "Medium", "Low"],
            "Stress Level": ["Medium", "High", "Low"],
            "Sleep Hours": [7.63, 8.74, 4.44],
            "Sugar Consumption": ["Medium", "Medium", "Low"],
            "Triglyceride Level": [342.0, 133.0, 393.0],
            "Fasting Blood Sugar": [np.nan, 157.0, 92.0],
            "CRP Level": [12.97, 9.36, 12.71],
            "Homocysteine Level": [12.39, 19.30, 11.23],
            "Heart Disease Status": ["No", "No", "No"],
        })
        
        preprocessor = HeartDiseasePreprocessor()
        transformed = preprocessor.fit_transform(data)
        
        # Pas de NaN dans le résultat final
        assert not transformed.isna().any().any()








# pytest tests/test_heart_disease_preprocessor.py -v