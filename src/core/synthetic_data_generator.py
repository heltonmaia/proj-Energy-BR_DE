# src/core/synthetic_data_generator.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from typing import List, Optional
from .energy_profile_config import SourceConfig, SimulationConfig

class HistoricalPatternLoader:
    """Loads and processes real historical data to create normalized weekly patterns."""
    def __init__(self, historical_fpath: Path):
        print(f"Loading historical data from: {historical_fpath}")
        df = pd.read_json(historical_fpath)
        
        column_map = {
            'DATA_INICIO': 'date_start', 'ANO': 'year',
            'SUDESTE': 'SOUTHEAST', 'SUL': 'SOUTH', 
            'NORDESTE': 'NORTHEAST', 'NORTE': 'NORTH'
        }
        df.rename(columns=column_map, inplace=True)
        
        df['date'] = pd.to_datetime(df['date_start'], unit='ms')
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df.set_index('date', inplace=True)
        self.df = df
        self.regions = ['SOUTHEAST', 'SOUTH', 'NORTHEAST', 'NORTH']

    def get_available_years(self) -> List[int]:
        """Returns a list of unique years available in the data."""
        return sorted(self.df['year'].unique().tolist())

    def calculate_pattern_for_region(self, region: str, base_year: Optional[int] = None) -> np.ndarray:
        """
        Calculates a normalized weekly price pattern for a SINGLE specified region.
        """
        if region not in self.regions:
            raise ValueError(f"Region '{region}' not found. Available: {self.regions}")
            
        if base_year:
            df_filtered = self.df[self.df['year'] == base_year].copy()
            if df_filtered.empty:
                raise ValueError(f"Year {base_year} not found in historical data.")
        else:
            df_filtered = self.df.copy()

        if base_year:
            annual_means = df_filtered[region].mean() 
        else:
            annual_means = df_filtered.groupby('year')[region].transform('mean')
        
        df_filtered[f'norm_{region}'] = df_filtered[region] / annual_means
        weekly_pattern_series = df_filtered.groupby('week_of_year')[f'norm_{region}'].mean()

        full_week_df = pd.DataFrame(index=pd.RangeIndex(start=1, stop=54, name='week_of_year'))
        merged_pattern = pd.merge(full_week_df, weekly_pattern_series, how='left', left_index=True, right_index=True)
        
        filled_pattern = merged_pattern.ffill().bfill()
        
        return filled_pattern[f'norm_{region}'].values


class ContractDataGenerator:
    """
    Synthetic contract data generator.
    Driven by a SINGLE historical pattern provided for all sources.
    """
    def __init__(self, sources: List[SourceConfig], sim_config: SimulationConfig, historical_pattern: np.ndarray):
        self.sources = {src.name: src for src in sources}
        self.sim_config = sim_config
        self.historical_pattern = historical_pattern
        if self.sim_config.random_seed is not None:
            np.random.seed(self.sim_config.random_seed)

    def generate_contract_profile(self) -> pd.DataFrame:
        n_days = self.sim_config.duration_days
        date_range = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_days, freq='D'))
        
        df_data = {
            'day': np.arange(1, n_days + 1),
            'week_of_year': date_range.isocalendar().week.values,
            'month': date_range.month
        }

        seasonal_multiplier = self.historical_pattern[df_data['week_of_year'] - 1]

        for source_name, source_config in self.sources.items():
            base_price = source_config.base_price
            
            market_vol = {'grid': 0.08}.get(source_name, 0.04)
            noise = np.random.normal(0, market_vol, n_days)
            
            price = base_price * (seasonal_multiplier + noise)
            price = np.maximum(price, base_price * 0.7)
            df_data[f'price_{source_name}'] = price
            
            base_avail = {'hydropower': 0.95, 'wind': 0.85, 'solar': 0.90, 'biomass': 0.92, 'biogas': 0.88, 'grid': 0.98}.get(source_name, 0.90)
            event_noise = np.random.normal(0, 0.04, n_days)
            availability = np.clip(base_avail + event_noise, 0.7, 1.0)
            df_data[f'availability_{source_name}'] = availability

        return pd.DataFrame(df_data)

    def save_data(self, df: pd.DataFrame, output_dir: str = "data/synthetic"):
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.sim_config.experiment_name}.json"
        output_path = output_dir / filename
        
        contracts_metadata = {
            src.name: {'quantity_kw': src.quantity_kw, 'base_price': src.base_price}
            for src in self.sources.values()
        }
        
        metadata = {
            'simulation_config': asdict(self.sim_config),
            'contracts': contracts_metadata,
            'generated_at': 'synthetic_contract_with_historical_pattern',
            'num_points': len(df)
        }
        
        if 'country' in metadata['simulation_config'] and hasattr(metadata['simulation_config']['country'], 'value'):
            metadata['simulation_config']['country'] = metadata['simulation_config']['country'].value
        
        stats = {
            'total_contracted_kw': sum(src.quantity_kw for src in self.sources.values()),
            'avg_prices': {src: float(df[f'price_{src}'].mean()) for src in self.sources},
            'avg_availability': {src: float(df[f'availability_{src}'].mean()) for src in self.sources}
        }
        
        output_json = {
            'metadata': metadata, 'statistics': stats, 'data': df.to_dict(orient='records')
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_json, f, indent=2)