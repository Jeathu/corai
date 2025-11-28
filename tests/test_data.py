




import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import pytest
from corai.preprocessing.dataset import load_data, main

class TestDataset:
    
    @pytest.fixture
    def sample_csv(self):
        """Crée un CSV temporaire pour les tests"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data = pd.DataFrame({
                "Age": [56.0, 69.0, 46.0],
                "Gender": ["Male", "Female", "Male"],
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
            data.to_csv(f, index=False)
            return Path(f.name)
    
    def test_load_data(self, sample_csv):
        """Test le chargement des données"""
        df = load_data(sample_csv)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "Heart Disease Status" in df.columns
    
    def test_load_data_invalid_path(self):
        """Test le chargement avec un chemin invalide"""
        with pytest.raises(FileNotFoundError):
            load_data(Path("invalid/path/data.csv"))





'''
  Il faut être dans le (venv)
'''
# pytest tests/test_data.py -v

