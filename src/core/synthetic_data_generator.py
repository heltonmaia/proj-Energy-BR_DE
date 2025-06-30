# src/core/synthetic_data_generator.py

"""
Synthetic Data Generator for Energy Sources
==========================================

Generates synthetic energy generation profiles for solar, wind, and grid sources
based on realistic physical and statistical models defined in energy_profiles.py.

This module is driven by configuration objects, not its own defaults.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Dict
import os
from .energy_profile_config import SourceConfig, IndustrialConfig, SimulationConfig

class SyntheticDataGenerator:
    """Synthetic data generator for multi-source energy systems (novo modelo simplificado)."""
    def __init__(self, sources: Dict[str, SourceConfig], industrial_config: IndustrialConfig, sim_config: SimulationConfig):
        self.sources = sources
        self.industrial_config = industrial_config
        self.sim_config = sim_config
        if self.sim_config.random_seed is not None:
            np.random.seed(self.sim_config.random_seed)

    def _generate_timestamps(self) -> pd.DatetimeIndex:
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        num_points = (self.sim_config.duration_days * 24 * 60) // self.sim_config.time_resolution_minutes
        return pd.date_range(start=start_time, periods=num_points, freq=f"{self.sim_config.time_resolution_minutes}min")

    def generate_complete_profile(self) -> pd.DataFrame:
        timestamps = self._generate_timestamps()
        n = len(timestamps)
        
        # Calcular potência média em kW a partir de GWh/mês
        national_avg_kws = {}
        for src, cfg in self.sources.items():
            monthly_kwh = cfg.monthly_generation_gwh * 1e6  # GWh -> kWh
            avg_kw = monthly_kwh / (30 * 24)  # kWh/mês -> kW médio
            national_avg_kws[src] = avg_kw
        
        # Normalizar geração para não exceder muito o consumo industrial
        total_national_avg = sum(national_avg_kws.values())
        max_total_gen = max(3 * self.industrial_config.total_consumption_kw, 1)
        norm_factor = max_total_gen / total_national_avg if total_national_avg > 0 else 1.0
        
        power_profiles = {}
        for src, cfg in self.sources.items():
            avg_kw = national_avg_kws[src] * norm_factor
            hours = np.arange(n) * self.sim_config.time_resolution_minutes / 60.0
            daily_cycle = 0.15 * np.sin(2 * np.pi * (hours % 24) / 24)
            noise = np.random.normal(0, 0.03, n)
            profile = avg_kw * (1 + daily_cycle + noise)
            profile = np.maximum(0, profile)
            power_profiles[f'{src}_power_kw'] = profile
        
        # Consumo industrial total (soma exata = total_consumption_kwh)
        total_consumption_kwh = self.industrial_config.total_consumption_kw
        time_res_h = self.sim_config.time_resolution_minutes / 60.0
        # Gera uma curva oscilante (média 1.0, soma = n)
        hours = np.arange(n) * self.sim_config.time_resolution_minutes / 60.0
        daily_cycle = 0.05 * np.sin(2 * np.pi * (hours % 24) / 24)
        noise = np.random.normal(0, 0.02, n)
        base_curve = 1.0 + daily_cycle + noise
        base_curve = np.clip(base_curve, 0.9, 1.1)
        # Normalizar para soma = 1.0
        base_curve = base_curve / base_curve.sum()
        # Multiplicar pelo total desejado em kWh e dividir pelo tempo de cada ponto (h) para obter kW instantâneo
        total_consumption = (base_curve * total_consumption_kwh) / time_res_h
        
        # Consumo por fonte (mantém proporção do usuário, soma exata)
        used_profiles = {}
        shares = self.industrial_config.shares
        share_sum = sum(shares.values())
        for src, share in shares.items():
            used_profiles[f'{src}_used_kw'] = total_consumption * (share / share_sum)
        
        # Custo médio ponderado (por MWh)
        avg_cost = sum(self.sources[src].avg_cost * shares.get(src, 0) for src in self.sources)
        cost = total_consumption * time_res_h * (avg_cost / 1000)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            **power_profiles,
            'industrial_consumption_kw': total_consumption,
            **used_profiles,
            f'energy_cost_({self.sim_config.country.value.upper()})': cost
        })
        
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        return df

    def save_data(self, df: pd.DataFrame, output_dir: str = "data/synthetic"):
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.sim_config.experiment_name}.json"
        output_path = output_dir / filename
        
        # Preparar metadados
        metadata = {
            'simulation_config': asdict(self.sim_config),
            'sources': {k: asdict(v) for k, v in self.sources.items()},
            'industrial_config': asdict(self.industrial_config),
            'generated_at': datetime.now().isoformat(),
            'num_points': len(df)
        }
        
        # Corrigir enums para string
        if 'country' in metadata['simulation_config'] and hasattr(metadata['simulation_config']['country'], 'value'):
            metadata['simulation_config']['country'] = metadata['simulation_config']['country'].value
        
        # Calcular estatísticas
        time_res_h = self.sim_config.time_resolution_minutes / 60.0
        stats = {
            'total_energy_generated_kwh': {},
            'total_industrial_consumption_kwh': 0.0,
            'total_cost': 0.0
        }
        
        # Calcular energia gerada por fonte
        for src in self.sources:
            col_name = f'{src}_power_kw'
            if col_name in df.columns:
                values = df[col_name].fillna(0).astype(float)
                stats['total_energy_generated_kwh'][src] = float(np.nansum(values) * time_res_h)
        
        # Calcular consumo total
        if 'industrial_consumption_kw' in df.columns:
            values = df['industrial_consumption_kw'].fillna(0).astype(float)
            stats['total_industrial_consumption_kwh'] = float(np.nansum(values) * time_res_h)
        
        # Calcular custo total
        cost_col = f'energy_cost_({self.sim_config.country.value.upper()})'
        if cost_col in df.columns:
            values = df[cost_col].fillna(0).astype(float)
            stats['total_cost'] = float(np.nansum(values))
        
        # Preparar dados para JSON
        df_copy = df.copy()
        
        # Garantir que timestamp seja string ISO
        if 'timestamp' in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            else:
                # Se já é string, garantir formato correto
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Converter para registros
        data_records = df_copy.to_dict(orient='records')
        
        # Preparar JSON final
        output_json = {
            'metadata': metadata,
            'statistics': stats,
            'data': data_records
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(output_json, f, indent=2)
            print(f"Data successfully saved to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save file: {e}")
            # Tentar salvar sem metadados complexos se houver erro
            try:
                simple_output = {
                    'data': data_records,
                    'experiment_name': self.sim_config.experiment_name,
                    'country': self.sim_config.country.value,
                    'num_points': len(df)
                }
                with open(output_path, 'w') as f:
                    json.dump(simple_output, f, indent=2)
                print(f"Data saved with simplified metadata to: {output_path}")
            except Exception as e2:
                print(f"[CRITICAL ERROR] Could not save file: {e2}")