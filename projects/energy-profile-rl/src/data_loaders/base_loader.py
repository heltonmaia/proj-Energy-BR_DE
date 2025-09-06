"""
Shared data loading utilities for energy projects.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any
import json

class BaseDataLoader:
    """Base data loader for energy time series data."""
    
    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.data = None
    
    def load_csv(self, **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        self.data = pd.read_csv(self.data_path, **kwargs)
        return self.data
    
    def load_json(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        return self.data
    
    def validate_time_series(self, time_col: str = 'timestamp') -> bool:
        """Validate time series data format."""
        if self.data is None:
            return False
        
        if time_col not in self.data.columns:
            return False
        
        # Check for missing values in critical columns
        if self.data[time_col].isna().any():
            return False
        
        return True
    
    def resample_data(self, freq: str, time_col: str = 'timestamp') -> pd.DataFrame:
        """Resample time series data to specified frequency."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        data_copy = self.data.copy()
        data_copy[time_col] = pd.to_datetime(data_copy[time_col])
        data_copy.set_index(time_col, inplace=True)
        
        return data_copy.resample(freq).mean()

class EnergyProfileLoader(BaseDataLoader):
    """Specialized loader for energy profile data."""
    
    def normalize_power_data(self, power_cols: list, max_power: float) -> pd.DataFrame:
        """Normalize power columns to [0, 1] range."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        data_copy = self.data.copy()
        for col in power_cols:
            if col in data_copy.columns:
                data_copy[col] = data_copy[col] / max_power
        
        return data_copy