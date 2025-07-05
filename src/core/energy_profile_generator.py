# src/core/energy_profile_generator.py

import numpy as np
import pandas as pd
from .energy_profile_config import IndustrialConfig, OnSiteGenerationConfig, SimulationConfig

class EnergyProfileGenerator:
    def __init__(self, sim_config: SimulationConfig, industrial_config: IndustrialConfig, generation_config: OnSiteGenerationConfig):
        self.sim_config = sim_config
        self.industrial_config = industrial_config
        self.generation_config = generation_config
        if self.sim_config.random_seed is not None:
            np.random.seed(self.sim_config.random_seed)

    def generate_profiles(self) -> pd.DataFrame:
        """Generates the detailed energy profile DataFrame."""
        n_points = int(self.sim_config.duration_days * 24 * (60 / self.sim_config.time_resolution_minutes))
        timestamps = pd.to_datetime(pd.date_range(
            start='2023-01-01', periods=n_points,
            freq=f'{self.sim_config.time_resolution_minutes}min'
        ))
        
        df = pd.DataFrame({'timestamp': timestamps})
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear

        # 1. Generate Industrial Consumption
        df['industrial_consumption_kw'] = self._generate_industrial_load(df)

        # 2. Generate Solar Power
        df['solar_generation_kw'] = self._generate_solar_power(df)

        # 3. Generate Wind Power
        df['wind_generation_kw'] = self._generate_wind_power(df)

        # 4. Energy Balance (initial step for decision making)
        df['energy_balance_kw'] = (df['solar_generation_kw'] + df['wind_generation_kw']) - df['industrial_consumption_kw']
        df['grid_needed_kw'] = -df['energy_balance_kw'].clip(upper=0)
        df['surplus_kw'] = df['energy_balance_kw'].clip(lower=0)
        
        return df

    def _generate_industrial_load(self, df: pd.DataFrame) -> np.ndarray:
        """Models the factory's energy consumption."""
        cfg = self.industrial_config
        
        is_work_hour = (df['hour'] >= cfg.work_start_hour) & (df['hour'] < cfg.work_end_hour)
        is_work_day = df['day_of_week'].isin(cfg.work_days)
        work_load = (is_work_day & is_work_hour) * cfg.work_shift_load_kw
        
        noise = np.random.normal(0, cfg.base_load_kw * 0.05, len(df))
        
        total_load = cfg.base_load_kw + work_load + noise
        return total_load.clip(lower=0)

    def _generate_solar_power(self, df: pd.DataFrame) -> np.ndarray:
        """Models solar power generation based on a daily sine wave."""
        cfg = self.generation_config
        hours = df['hour'] + df['timestamp'].dt.minute / 60.0
        rad_factor = np.sin((hours - 6) * np.pi / 12)
        rad_factor = rad_factor.clip(lower=0)
        
        daily_variation_factors = 1 - (np.random.uniform(0, 0.4, self.sim_config.duration_days))
        daily_variation = daily_variation_factors[df['day_of_year'] - 1]
        
        solar_power = cfg.solar_installed_kw * rad_factor * daily_variation
        return solar_power.clip(lower=0)

    def _generate_wind_power(self, df: pd.DataFrame) -> np.ndarray:
        """Models wind power generation using smoothed random noise."""
        cfg = self.generation_config
        random_noise = np.random.rand(len(df))
        window_size = int(24 * (60 / self.sim_config.time_resolution_minutes) / 4)
        wind_factor = pd.Series(random_noise).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()
        
        wind_power = cfg.wind_installed_kw * wind_factor
        return wind_power.clip(min=0)