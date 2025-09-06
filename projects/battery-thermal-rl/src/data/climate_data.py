"""
Dados climáticos brasileiros para simulação térmica.

Este módulo fornece dados de temperatura e outras variáveis climáticas
para diferentes regiões do Brasil, considerando sazonalidade e variações diárias.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class ClimateProfile:
    """Perfil climático de uma região brasileira"""
    region_name: str
    avg_temp_summer: float  # Temperatura média no verão (°C)
    avg_temp_winter: float  # Temperatura média no inverno (°C)
    daily_temp_variation: float  # Variação diária típica (°C)
    humidity_avg: float  # Umidade média (%)
    solar_irradiance_peak: float  # Irradiância solar de pico (kWh/m²/dia)


class BrazilianClimateData:
    """
    Gerador de dados climáticos para diferentes regiões brasileiras.
    
    Baseado em dados climatológicos médios do INMET e considerando
    as principais diferenças regionais do Brasil.
    """
    
    # Regional climate profiles based on historical data
    REGIONAL_PROFILES = {
        'southeast_sp': ClimateProfile(
            region_name="São Paulo - Southeast",
            avg_temp_summer=24.5,
            avg_temp_winter=16.8,
            daily_temp_variation=8.0,
            humidity_avg=75.0,
            solar_irradiance_peak=5.5
        ),
        'northeast_rn': ClimateProfile(
            region_name="Rio Grande do Norte - Northeast",
            avg_temp_summer=29.5,
            avg_temp_winter=26.2,
            daily_temp_variation=7.0,
            humidity_avg=68.0,
            solar_irradiance_peak=7.2
        ),
        'south_rs': ClimateProfile(
            region_name="Rio Grande do Sul - South",
            avg_temp_summer=25.1,
            avg_temp_winter=13.2,
            daily_temp_variation=9.2,
            humidity_avg=78.0,
            solar_irradiance_peak=4.8
        ),
        'central_west_mt': ClimateProfile(
            region_name="Mato Grosso - Central-West",
            avg_temp_summer=27.8,
            avg_temp_winter=22.1,
            daily_temp_variation=12.0,
            humidity_avg=65.0,
            solar_irradiance_peak=6.2
        ),
        'north_am': ClimateProfile(
            region_name="Amazonas - North",
            avg_temp_summer=27.5,
            avg_temp_winter=26.8,
            daily_temp_variation=4.5,
            humidity_avg=85.0,
            solar_irradiance_peak=5.0
        )
    }
    
    def __init__(self, region: str = 'southeast_sp'):
        """
        Initializes climate data generator for specific region.
        
        Args:
            region: Regional climate profile key
        """
        if region not in self.REGIONAL_PROFILES:
            raise ValueError(f"Region '{region}' not found. Options: {list(self.REGIONAL_PROFILES.keys())}")
        
        self.profile = self.REGIONAL_PROFILES[region]
        self.current_date = datetime(2024, 1, 1)
    
    def get_seasonal_temp_avg(self, month: int) -> float:
        """
        Calcula temperatura média sazonal para um mês específico.
        
        Args:
            month: Mês (1-12)
            
        Returns:
            Temperatura média do mês (°C)
        """
        # Verão: Dez, Jan, Feb, Mar (meses 12, 1, 2, 3)
        # Inverno: Jun, Jul, Aug, Set (meses 6, 7, 8, 9)
        
        if month in [12, 1, 2, 3]:  # Verão
            return self.profile.avg_temp_summer
        elif month in [6, 7, 8, 9]:  # Inverno
            return self.profile.avg_temp_winter
        else:  # Transição (outono/primavera)
            # Interpolação linear entre verão e inverno
            if month in [4, 5]:  # Outono
                weight = (month - 3) / 3  # 4->0.33, 5->0.67
                return (self.profile.avg_temp_summer * (1 - weight) + 
                       self.profile.avg_temp_winter * weight)
            else:  # Primavera (10, 11)
                weight = (month - 9) / 3  # 10->0.33, 11->0.67
                return (self.profile.avg_temp_winter * (1 - weight) + 
                       self.profile.avg_temp_summer * weight)
    
    def generate_daily_temperature(self, date: datetime, hour: int) -> float:
        """
        Gera temperatura para hora específica considerando ciclo diário.
        
        Args:
            date: Data de referência
            hour: Hora do dia (0-23)
            
        Returns:
            Temperatura em °C
        """
        seasonal_avg = self.get_seasonal_temp_avg(date.month)
        
        # Ciclo diário: mínima às 6h, máxima às 14h
        hour_radians = (hour - 6) * 2 * np.pi / 24
        daily_variation = self.profile.daily_temp_variation * np.sin(hour_radians) / 2
        
        # Adiciona variação aleatória pequena para realismo
        random_variation = np.random.normal(0, 1.0)
        
        temperature = seasonal_avg + daily_variation + random_variation
        
        # Limita temperaturas extremas não realistas
        return np.clip(temperature, -5, 50)
    
    def generate_solar_irradiance(self, date: datetime, hour: int) -> float:
        """
        Gera irradiância solar para hora específica.
        
        Args:
            date: Data de referência
            hour: Hora do dia (0-23)
            
        Returns:
            Irradiância em kW/m²
        """
        # Solar apenas durante o dia (6h às 18h)
        if hour < 6 or hour > 18:
            return 0.0
        
        # Curva solar parabólica com pico ao meio-dia
        day_progress = (hour - 6) / 12  # 0 a 1 durante o dia
        solar_curve = 4 * day_progress * (1 - day_progress)  # Parábola 0->1->0
        
        # Variação sazonal
        seasonal_factor = 0.8 + 0.4 * np.sin((date.month - 6) * np.pi / 6)
        
        # Variação estocástica para simular nuvens
        cloud_factor = max(0.1, np.random.normal(0.7, 0.3))
        
        irradiance = (self.profile.solar_irradiance_peak * solar_curve * 
                     seasonal_factor * cloud_factor / 24)  # Converter para kW/m²
        
        return max(0, irradiance)
    
    def generate_hourly_data(self, start_date: datetime, days: int) -> pd.DataFrame:
        """
        Gera dados climáticos horários para período específico.
        
        Args:
            start_date: Data de início
            days: Número de dias para gerar
            
        Returns:
            DataFrame com dados horários
        """
        timestamps = []
        temperatures = []
        solar_irradiances = []
        humidities = []
        
        current_date = start_date
        
        for day in range(days):
            for hour in range(24):
                timestamp = current_date.replace(hour=hour)
                timestamps.append(timestamp)
                
                # Temperatura
                temp = self.generate_daily_temperature(current_date, hour)
                temperatures.append(temp)
                
                # Irradiância solar
                solar = self.generate_solar_irradiance(current_date, hour)
                solar_irradiances.append(solar)
                
                # Umidade (varia inversamente com temperatura)
                humidity_variation = np.random.normal(0, 5)
                humidity = self.profile.humidity_avg - (temp - 20) * 1.5 + humidity_variation
                humidities.append(np.clip(humidity, 30, 95))
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'temperature_c': temperatures,
            'solar_irradiance_kw_m2': solar_irradiances,
            'humidity_percent': humidities,
            'region': [self.profile.region_name] * len(timestamps)
        })
    
    def get_extreme_conditions(self) -> Dict[str, float]:
        """Retorna condições climáticas extremas para a região"""
        summer_max = self.profile.avg_temp_summer + self.profile.daily_temp_variation/2 + 5
        winter_min = self.profile.avg_temp_winter - self.profile.daily_temp_variation/2 - 3
        
        return {
            'max_temperature': summer_max,
            'min_temperature': winter_min,
            'max_solar_irradiance': self.profile.solar_irradiance_peak / 12,  # Pico horário
            'avg_humidity': self.profile.humidity_avg
        }


def save_climate_data(region: str, start_date: datetime, days: int, filepath: str):
    """Generate and save climate data to CSV file"""
    climate_gen = BrazilianClimateData(region)
    data = climate_gen.generate_hourly_data(start_date, days)
    data.to_csv(filepath, index=False)
    print(f"Climate data saved to: {filepath}")


def get_available_regions() -> List[str]:
    """Returns list of available regions"""
    return list(BrazilianClimateData.REGIONAL_PROFILES.keys())