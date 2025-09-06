"""
Sistema industrial com painéis solares e demanda energética.

Este módulo modela:
- Perfis de consumo industrial típicos
- Geração solar fotovoltaica 
- Integração com sistema de baterias
- Economia de energia e custos
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class IndustrialProfile:
    """Perfil de uma indústria de médio porte"""
    name: str
    peak_demand_kw: float  # Demanda de pico (kW)
    base_load_kw: float    # Carga base (kW)
    operating_hours_start: int  # Início operação (hora)
    operating_hours_end: int    # Fim operação (hora)
    weekend_factor: float  # Fator de redução fins de semana (0-1)
    seasonal_variation: float  # Variação sazonal (0-1)


@dataclass
class SolarSystemSpecs:
    """Especificações do sistema solar fotovoltaico"""
    installed_capacity_kw: float  # Capacidade instalada (kWp)
    panel_efficiency: float = 0.20  # Eficiência dos painéis (20%)
    system_efficiency: float = 0.85  # Eficiência do sistema (inversor, cabos, etc.)
    degradation_rate: float = 0.005  # Taxa anual de degradação (0.5%)


class IndustrialEnergySystem:
    """
    Sistema integrado industrial com solar + bateria.
    
    Modela uma indústria de médio porte com:
    - Consumo variável por horário/dia/estação
    - Sistema solar fotovoltaico
    - Tarifação diferenciada (horário de ponta/fora ponta)
    """
    
    # Typical Brazilian industrial profiles
    INDUSTRIAL_PROFILES = {
        'medium_metallurgy': IndustrialProfile(
            name="Medium Metallurgy",
            peak_demand_kw=800.0,
            base_load_kw=200.0,
            operating_hours_start=6,
            operating_hours_end=22,
            weekend_factor=0.3,
            seasonal_variation=0.15
        ),
        'medium_textile': IndustrialProfile(
            name="Medium Textile",
            peak_demand_kw=600.0,
            base_load_kw=150.0,
            operating_hours_start=7,
            operating_hours_end=19,
            weekend_factor=0.2,
            seasonal_variation=0.25
        ),
        'medium_food': IndustrialProfile(
            name="Medium Food Processing",
            peak_demand_kw=500.0,
            base_load_kw=180.0,
            operating_hours_start=5,
            operating_hours_end=21,
            weekend_factor=0.6,  # Higher activity on weekends
            seasonal_variation=0.20
        ),
        'medium_chemical': IndustrialProfile(
            name="Medium Chemical",
            peak_demand_kw=1200.0,
            base_load_kw=400.0,
            operating_hours_start=0,  # Continuous operation
            operating_hours_end=24,
            weekend_factor=0.9,
            seasonal_variation=0.10
        )
    }
    
    def __init__(self, 
                 industrial_profile: str,
                 solar_specs: SolarSystemSpecs,
                 electricity_tariff: Dict[str, float] = None):
        """
        Inicializa sistema industrial.
        
        Args:
            industrial_profile: Tipo de indústria
            solar_specs: Especificações do sistema solar
            electricity_tariff: Tarifas de energia (R$/kWh)
        """
        if industrial_profile not in self.INDUSTRIAL_PROFILES:
            raise ValueError(f"Perfil '{industrial_profile}' não encontrado.")
            
        self.profile = self.INDUSTRIAL_PROFILES[industrial_profile]
        self.solar_specs = solar_specs
        
        # Typical Brazilian tariffs (R$/kWh) - approximate 2024 values
        self.tariff = electricity_tariff or {
            'peak': 0.85,           # 6pm-9pm weekdays
            'intermediate': 0.65,   # 5pm-6pm and 9pm-10pm weekdays  
            'off_peak': 0.45,       # Other hours
            'weekend': 0.35         # Weekends and holidays
        }
        
        self.system_age_years = 0  # Para degradação dos painéis
    
    def get_demand_profile(self, timestamp: datetime) -> float:
        """
        Calcula demanda energética para timestamp específico.
        
        Args:
            timestamp: Momento para calcular demanda
            
        Returns:
            Demanda em kW
        """
        hour = timestamp.hour
        month = timestamp.month
        is_weekend = timestamp.weekday() >= 5  # Sábado=5, Domingo=6
        
        # Carga base sempre presente
        base_demand = self.profile.base_load_kw
        
        # Variação sazonal (maior no verão para refrigeração)
        seasonal_factor = 1.0 + self.profile.seasonal_variation * np.sin((month - 1) * np.pi / 6)
        
        # Perfil horário durante operação
        if (self.profile.operating_hours_start <= hour < self.profile.operating_hours_end or 
            self.profile.operating_hours_start == 0):  # Operação contínua
            
            # Curva de demanda típica industrial
            if 6 <= hour <= 8 or 17 <= hour <= 19:  # Picos início/fim expediente
                operational_factor = 0.9
            elif 9 <= hour <= 16:  # Período produtivo normal
                operational_factor = 0.8
            else:  # Outros horários operacionais
                operational_factor = 0.6
                
            operational_demand = (self.profile.peak_demand_kw - base_demand) * operational_factor
        else:
            operational_demand = 0
        
        # Redução fins de semana
        weekend_factor = self.profile.weekend_factor if is_weekend else 1.0
        
        # Variação estocástica pequena (+/- 10%)
        random_factor = np.random.normal(1.0, 0.05)
        
        total_demand = (base_demand + operational_demand) * seasonal_factor * weekend_factor * random_factor
        
        return max(0, total_demand)
    
    def calculate_solar_generation(self, solar_irradiance_kw_m2: float, 
                                 temperature_c: float) -> float:
        """
        Calcula geração solar considerando irradiância e temperatura.
        
        Args:
            solar_irradiance_kw_m2: Irradiância solar (kW/m²)
            temperature_c: Temperatura ambiente (°C)
            
        Returns:
            Geração solar em kW
        """
        if solar_irradiance_kw_m2 <= 0:
            return 0.0
        
        # Geração base
        base_generation = (self.solar_specs.installed_capacity_kw * 
                          solar_irradiance_kw_m2 * 
                          self.solar_specs.panel_efficiency)
        
        # Correção por temperatura (painéis perdem eficiência com calor)
        # Coeficiente típico: -0.4%/°C acima de 25°C
        temp_coefficient = -0.004
        temp_correction = 1 + temp_coefficient * (temperature_c - 25.0)
        temp_correction = max(0.5, temp_correction)  # Limitação mínima
        
        # Eficiência do sistema (inversores, cabos, sujeira, etc.)
        system_efficiency = self.solar_specs.system_efficiency
        
        # Degradação dos painéis ao longo do tempo
        degradation_factor = (1 - self.solar_specs.degradation_rate) ** self.system_age_years
        
        total_generation = (base_generation * temp_correction * 
                          system_efficiency * degradation_factor)
        
        return max(0, total_generation)
    
    def get_electricity_price(self, timestamp: datetime) -> float:
        """
        Retorna preço da eletricidade para timestamp específico.
        
        Args:
            timestamp: Momento para consultar preço
            
        Returns:
            Preço em R$/kWh
        """
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        if is_weekend:
            return self.tariff['weekend']
        
        # Tarifação por horário em dias úteis
        if 18 <= hour < 21:  # Peak hours
            return self.tariff['peak']
        elif hour == 17 or 21 <= hour < 22:  # Intermediate
            return self.tariff['intermediate']
        else:  # Off-peak
            return self.tariff['off_peak']
    
    def calculate_energy_balance(self, 
                               demand_kw: float,
                               solar_generation_kw: float,
                               battery_charge_kw: float = 0,
                               battery_discharge_kw: float = 0) -> Dict[str, float]:
        """
        Calcula balanço energético do sistema.
        
        Args:
            demand_kw: Demanda industrial (kW)
            solar_generation_kw: Geração solar (kW)  
            battery_charge_kw: Energia para carregar bateria (kW)
            battery_discharge_kw: Energia da bateria (kW)
            
        Returns:
            Dict com balanço energético
        """
        # Energia disponível (geração + descarga bateria)
        available_energy = solar_generation_kw + battery_discharge_kw
        
        # Energia total necessária (demanda + carga bateria)
        total_demand = demand_kw + battery_charge_kw
        
        # Balanço
        if available_energy >= total_demand:
            # Excesso de energia
            grid_import = 0
            excess_energy = available_energy - total_demand
            grid_export = excess_energy  # Pode injetar na rede
        else:
            # Déficit - precisa importar da rede
            grid_import = total_demand - available_energy
            grid_export = 0
            excess_energy = 0
        
        return {
            'demand_kw': demand_kw,
            'solar_generation_kw': solar_generation_kw,
            'battery_charge_kw': battery_charge_kw,
            'battery_discharge_kw': battery_discharge_kw,
            'grid_import_kw': grid_import,
            'grid_export_kw': grid_export,
            'excess_energy_kw': excess_energy,
            'self_consumption_kw': min(demand_kw, solar_generation_kw + battery_discharge_kw)
        }
    
    def calculate_costs(self, 
                       energy_balance: Dict[str, float],
                       electricity_price: float,
                       export_price_factor: float = 0.7) -> Dict[str, float]:
        """
        Calcula custos e economias do sistema.
        
        Args:
            energy_balance: Balanço energético
            electricity_price: Preço da eletricidade (R$/kWh)
            export_price_factor: Fator de remuneração para energia exportada
            
        Returns:
            Dict com custos e economias
        """
        # Custo de importação da rede
        import_cost = energy_balance['grid_import_kw'] * electricity_price
        
        # Receita de exportação (geralmente menor que preço de importação)
        export_revenue = (energy_balance['grid_export_kw'] * 
                         electricity_price * export_price_factor)
        
        # Economia com autoconsumo
        self_consumption_savings = (energy_balance['self_consumption_kw'] * 
                                   electricity_price)
        
        # Custo líquido
        net_cost = import_cost - export_revenue
        
        return {
            'import_cost_brl': import_cost,
            'export_revenue_brl': export_revenue,
            'self_consumption_savings_brl': self_consumption_savings,
            'net_cost_brl': net_cost,
            'electricity_price_brl_kwh': electricity_price
        }
    
    def simulate_day(self, 
                     date: datetime,
                     climate_data: pd.DataFrame,
                     battery_actions: List[Tuple[str, float]] = None) -> pd.DataFrame:
        """
        Simula operação do sistema por um dia completo.
        
        Args:
            date: Data da simulação
            climate_data: DataFrame com dados climáticos horários
            battery_actions: Lista de ações da bateria [(ação, potência_kw)]
            
        Returns:
            DataFrame com dados horários do dia
        """
        results = []
        battery_actions = battery_actions or [('hold', 0)] * 24
        
        for hour in range(24):
            timestamp = date.replace(hour=hour)
            
            # Dados climáticos
            climate_row = climate_data[
                (climate_data['timestamp'].dt.date == date.date()) & 
                (climate_data['timestamp'].dt.hour == hour)
            ]
            
            if climate_row.empty:
                # Dados padrão se não encontrar
                temperature = 25.0
                irradiance = 0.0
            else:
                temperature = climate_row.iloc[0]['temperature_c']
                irradiance = climate_row.iloc[0]['solar_irradiance_kw_m2']
            
            # Demanda e geração
            demand = self.get_demand_profile(timestamp)
            solar_gen = self.calculate_solar_generation(irradiance, temperature)
            
            # Ação da bateria
            battery_action, battery_power = battery_actions[hour]
            battery_charge = battery_power if battery_action == 'charge' else 0
            battery_discharge = battery_power if battery_action == 'discharge' else 0
            
            # Balanço energético
            balance = self.calculate_energy_balance(
                demand, solar_gen, battery_charge, battery_discharge
            )
            
            # Custos
            price = self.get_electricity_price(timestamp)
            costs = self.calculate_costs(balance, price)
            
            # Resultado horário
            result = {
                'timestamp': timestamp,
                'temperature_c': temperature,
                'solar_irradiance_kw_m2': irradiance,
                **balance,
                **costs,
                'battery_action': battery_action,
                'battery_power_kw': battery_power
            }
            
            results.append(result)
        
        return pd.DataFrame(results)


# Configurações de sistemas típicos
TYPICAL_SOLAR_SYSTEMS = {
    'small_500kw': SolarSystemSpecs(installed_capacity_kw=500),
    'medium_1000kw': SolarSystemSpecs(installed_capacity_kw=1000),
    'large_2000kw': SolarSystemSpecs(installed_capacity_kw=2000),
}