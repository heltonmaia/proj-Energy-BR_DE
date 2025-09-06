"""
Modelos de bateria para sistemas industriais com gestão térmica.

Este módulo implementa modelos de baterias reais considerando:
- Capacidade e degradação
- Temperatura operacional ótima 
- Eficiência de carga/descarga
- Modelos térmicos baseados em dados reais
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class BatterySpecifications:
    """Especificações técnicas da bateria"""
    capacity_kwh: float  # Capacidade nominal em kWh
    max_charge_rate_kw: float  # Taxa máxima de carga em kW
    max_discharge_rate_kw: float  # Taxa máxima de descarga em kW
    min_soc: float = 0.1  # Estado mínimo de carga (10%)
    max_soc: float = 0.9  # Estado máximo de carga (90%)
    optimal_temp_min: float = 15.0  # Temperatura ótima mínima (°C)
    optimal_temp_max: float = 25.0  # Temperatura ótima máxima (°C)
    critical_temp_max: float = 45.0  # Temperatura crítica máxima (°C)
    cycle_life: int = 6000  # Ciclos de vida esperados


class BatteryModel:
    """
    Modelo de bateria com gestão térmica para aplicações industriais.
    
    Considera:
    - Li-ion: Alta densidade energética, sensível à temperatura
    - Degradação baseada em ciclos e temperatura
    - Eficiência variável com temperatura
    """
    
    def __init__(self, specs: BatterySpecifications):
        self.specs = specs
        self.soc = 0.5  # Estado inicial de carga (50%)
        self.temperature = 25.0  # Temperatura inicial (°C)
        self.cycles_count = 0
        self.degradation_factor = 1.0
        
    def get_efficiency(self, temperature: float, operation: str) -> float:
        """
        Calcula eficiência baseada na temperatura e tipo de operação.
        
        Args:
            temperature: Temperatura atual (°C)
            operation: 'charge' ou 'discharge'
            
        Returns:
            Eficiência entre 0 e 1
        """
        # Eficiência ótima na faixa ideal de temperatura
        if self.specs.optimal_temp_min <= temperature <= self.specs.optimal_temp_max:
            base_efficiency = 0.95 if operation == 'charge' else 0.90
        else:
            # Redução de eficiência fora da faixa ótima
            temp_deviation = min(
                abs(temperature - self.specs.optimal_temp_min),
                abs(temperature - self.specs.optimal_temp_max)
            )
            efficiency_loss = temp_deviation * 0.005  # 0.5% por grau de desvio
            base_efficiency = (0.95 if operation == 'charge' else 0.90) - efficiency_loss
            
        # Consideração da degradação
        return max(0.3, base_efficiency * self.degradation_factor)
    
    def can_operate_safely(self, temperature: float) -> bool:
        """Verifica se a bateria pode operar com segurança na temperatura atual"""
        return temperature <= self.specs.critical_temp_max
    
    def charge(self, power_kw: float, temperature: float, duration_hours: float) -> Tuple[float, float]:
        """
        Simula carga da bateria considerando temperatura.
        
        Returns:
            Tuple[energia_carregada_kwh, temperatura_final]
        """
        if not self.can_operate_safely(temperature):
            return 0.0, temperature
            
        max_power = min(power_kw, self.specs.max_charge_rate_kw)
        efficiency = self.get_efficiency(temperature, 'charge')
        
        # Energia que pode ser armazenada
        available_capacity = (self.specs.max_soc - self.soc) * self.specs.capacity_kwh
        energy_to_charge = min(max_power * duration_hours * efficiency, available_capacity)
        
        # Atualização do SOC
        self.soc += energy_to_charge / self.specs.capacity_kwh
        
        # Aquecimento devido ao processo de carga (estimativa simplificada)
        heat_generated = max_power * duration_hours * (1 - efficiency) * 0.1
        temp_increase = heat_generated / 10.0  # Modelo térmico simplificado
        final_temperature = temperature + temp_increase
        
        return energy_to_charge, final_temperature
    
    def discharge(self, power_kw: float, temperature: float, duration_hours: float) -> Tuple[float, float]:
        """
        Simula descarga da bateria considerando temperatura.
        
        Returns:
            Tuple[energia_fornecida_kwh, temperatura_final]
        """
        if not self.can_operate_safely(temperature):
            return 0.0, temperature
            
        max_power = min(power_kw, self.specs.max_discharge_rate_kw)
        efficiency = self.get_efficiency(temperature, 'discharge')
        
        # Energia disponível para descarga
        available_energy = (self.soc - self.specs.min_soc) * self.specs.capacity_kwh
        energy_to_discharge = min(max_power * duration_hours, available_energy)
        
        # Atualização do SOC
        self.soc -= energy_to_discharge / self.specs.capacity_kwh
        
        # Aquecimento devido ao processo de descarga
        heat_generated = energy_to_discharge * (1 - efficiency) * 0.05
        temp_increase = heat_generated / 10.0
        final_temperature = temperature + temp_increase
        
        return energy_to_discharge * efficiency, final_temperature
    
    def update_degradation(self):
        """Atualiza fator de degradação baseado em ciclos e temperatura"""
        if self.cycles_count > 0:
            degradation_per_cycle = 1.0 / self.specs.cycle_life
            self.degradation_factor = max(0.7, 1.0 - degradation_per_cycle * self.cycles_count)
    
    def get_state(self) -> Dict:
        """Retorna estado atual da bateria"""
        return {
            'soc': self.soc,
            'temperature': self.temperature,
            'degradation_factor': self.degradation_factor,
            'cycles_count': self.cycles_count,
            'available_capacity_kwh': self.soc * self.specs.capacity_kwh,
            'can_charge': self.soc < self.specs.max_soc,
            'can_discharge': self.soc > self.specs.min_soc
        }


# Configurações de baterias industriais comuns
INDUSTRIAL_BATTERY_CONFIGS = {
    'li_ion_100kwh': BatterySpecifications(
        capacity_kwh=100.0,
        max_charge_rate_kw=50.0,
        max_discharge_rate_kw=50.0,
        optimal_temp_min=15.0,
        optimal_temp_max=25.0,
        cycle_life=6000
    ),
    'li_ion_500kwh': BatterySpecifications(
        capacity_kwh=500.0,
        max_charge_rate_kw=200.0,
        max_discharge_rate_kw=200.0,
        optimal_temp_min=15.0,
        optimal_temp_max=25.0,
        cycle_life=6000
    ),
    'na_ion_200kwh': BatterySpecifications(
        capacity_kwh=200.0,
        max_charge_rate_kw=80.0,
        max_discharge_rate_kw=80.0,
        optimal_temp_min=10.0,
        optimal_temp_max=35.0,  # Na-ion mais tolerante ao calor
        cycle_life=8000
    )
}