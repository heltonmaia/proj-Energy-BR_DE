"""
Ambiente de Aprendizado por Reforço para otimização de bateria com gestão térmica.

Este ambiente integra:
- Sistema industrial com demanda variável
- Geração solar fotovoltaica  
- Bateria com modelo térmico
- Dados climáticos brasileiros
- Otimização de custos e durabilidade da bateria
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional
from datetime import datetime, timedelta

from ..models.battery import BatteryModel, BatterySpecifications
from ..models.industrial_system import IndustrialEnergySystem, SolarSystemSpecs
from ..data.climate_data import BrazilianClimateData


class BatteryThermalEnv(gym.Env):
    """
    Ambiente de RL para otimização de sistema bateria + solar + industrial.
    
    O agente deve decidir quando carregar/descarregar a bateria considerando:
    - Preços de energia variáveis por horário
    - Temperatura ambiente e seu impacto na bateria
    - Demanda industrial e geração solar
    - Degradação da bateria (custo de longo prazo)
    
    Action Space:
        - Ação contínua: potência de carga/descarga [-1, 1]
          -1: descarga máxima, 0: sem ação, +1: carga máxima
          
    Observation Space:
        - Estado da bateria: SOC, temperatura, degradação
        - Condições climáticas: temperatura ambiente, irradiância solar
        - Sistema: demanda atual, geração solar, preço energia
        - Temporais: hora do dia, dia da semana, mês
    """
    
    def __init__(self,
                 battery_specs: BatterySpecifications,
                 industrial_profile: str = 'medium_metallurgy',
                 solar_specs: SolarSystemSpecs = None,
                 climate_region: str = 'southeast_sp',
                 simulation_days: int = 30,
                 reward_weights: Dict[str, float] = None):
        """
        Initialize RL environment.
        
        Args:
            battery_specs: Battery specifications
            industrial_profile: Industry type
            solar_specs: Solar system specifications
            climate_region: Brazilian climate region
            simulation_days: Days to simulate
            reward_weights: Weights for reward components
        """
        super().__init__()
        
        # Componentes do sistema
        self.battery = BatteryModel(battery_specs)
        self.industrial_system = IndustrialEnergySystem(
            industrial_profile=industrial_profile,
            solar_specs=solar_specs or SolarSystemSpecs(installed_capacity_kw=1000)
        )
        self.climate_generator = BrazilianClimateData(climate_region)
        
        # Configuração da simulação
        self.simulation_days = simulation_days
        self.hours_per_episode = simulation_days * 24
        
        # Pesos da função de recompensa
        self.reward_weights = reward_weights or {
            'cost_reduction': 1.0,      # Redução de custos energéticos
            'battery_health': 0.3,      # Preservação da bateria
            'grid_stability': 0.2,      # Estabilidade da rede
            'temperature_penalty': 0.5  # Penalização por temperatura inadequada
        }
        
        # Espaços de ação e observação
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # 15 features de observação
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(15,), dtype=np.float32
        )
        
        # Estado interno
        self.reset()
    
    def _generate_climate_data(self, start_date: datetime) -> pd.DataFrame:
        """Gera dados climáticos para o episódio"""
        return self.climate_generator.generate_hourly_data(
            start_date, self.simulation_days
        )
    
    def _normalize_observation(self, raw_obs: Dict) -> np.ndarray:
        """
        Normaliza observações para rede neural.
        
        Features:
        0: SOC da bateria (0-1)
        1: Temperatura da bateria normalizada (0-50°C -> 0-1)
        2: Degradação da bateria (0-1)
        3: Temperatura ambiente normalizada (-5-50°C -> 0-1)  
        4: Irradiância solar normalizada (0-1)
        5: Demanda industrial normalizada (0-peak -> 0-1)
        6: Geração solar normalizada (0-max -> 0-1)
        7: Preço energia normalizado (0.3-1.0 R$/kWh -> 0-1)
        8: Hora do dia (0-23 -> 0-1)
        9: Dia da semana (0-6 -> 0-1)
        10: Mês (1-12 -> 0-1)
        11: Pode carregar bateria (0 ou 1)
        12: Pode descarregar bateria (0 ou 1)
        13: Excesso solar (0-1)
        14: Déficit energético (0-1)
        """
        obs = np.zeros(15, dtype=np.float32)
        
        # Estado da bateria
        obs[0] = raw_obs['battery_soc']
        obs[1] = np.clip(raw_obs['battery_temperature'] / 50.0, 0, 1)
        obs[2] = raw_obs['battery_degradation']
        
        # Condições climáticas
        obs[3] = np.clip((raw_obs['ambient_temperature'] + 5) / 55.0, 0, 1)
        obs[4] = np.clip(raw_obs['solar_irradiance'] * 12, 0, 1)  # Assumindo max ~0.083 kW/m²
        
        # Sistema energético  
        obs[5] = raw_obs['demand_kw'] / self.industrial_system.profile.peak_demand_kw
        obs[6] = raw_obs['solar_generation_kw'] / self.industrial_system.solar_specs.installed_capacity_kw
        obs[7] = (raw_obs['electricity_price'] - 0.3) / 0.7  # 0.3-1.0 -> 0-1
        
        # Temporais
        obs[8] = raw_obs['hour'] / 23.0
        obs[9] = raw_obs['weekday'] / 6.0
        obs[10] = (raw_obs['month'] - 1) / 11.0
        
        # Capacidades da bateria
        obs[11] = 1.0 if raw_obs['can_charge'] else 0.0
        obs[12] = 1.0 if raw_obs['can_discharge'] else 0.0
        
        # Balanço energético
        excess = max(0, raw_obs['solar_generation_kw'] - raw_obs['demand_kw'])
        obs[13] = np.clip(excess / 500, 0, 1)  # Normalizar excesso
        
        deficit = max(0, raw_obs['demand_kw'] - raw_obs['solar_generation_kw'])
        obs[14] = np.clip(deficit / self.industrial_system.profile.peak_demand_kw, 0, 1)
        
        return obs
    
    def _get_raw_observation(self) -> Dict:
        """Coleta observações brutas do sistema"""
        timestamp = self.current_date
        hour_idx = self.step_count % 24
        
        # Dados climáticos atuais
        climate_row = self.climate_data.iloc[self.step_count]
        
        # Estado da bateria
        battery_state = self.battery.get_state()
        
        # Demanda e geração
        demand = self.industrial_system.get_demand_profile(timestamp)
        solar_gen = self.industrial_system.calculate_solar_generation(
            climate_row['solar_irradiance_kw_m2'],
            climate_row['temperature_c']
        )
        
        # Preço da energia
        electricity_price = self.industrial_system.get_electricity_price(timestamp)
        
        return {
            # Bateria
            'battery_soc': battery_state['soc'],
            'battery_temperature': battery_state['temperature'],
            'battery_degradation': battery_state['degradation_factor'],
            'can_charge': battery_state['can_charge'],
            'can_discharge': battery_state['can_discharge'],
            
            # Clima
            'ambient_temperature': climate_row['temperature_c'],
            'solar_irradiance': climate_row['solar_irradiance_kw_m2'],
            
            # Sistema
            'demand_kw': demand,
            'solar_generation_kw': solar_gen,
            'electricity_price': electricity_price,
            
            # Temporal
            'hour': timestamp.hour,
            'weekday': timestamp.weekday(),
            'month': timestamp.month,
            'timestamp': timestamp
        }
    
    def _calculate_reward(self, 
                         action: float, 
                         raw_obs: Dict,
                         energy_balance: Dict,
                         costs: Dict) -> float:
        """
        Calcula recompensa multi-objetivo.
        
        Componentes:
        1. Economia de custos energéticos
        2. Preservação da saúde da bateria
        3. Estabilidade da rede
        4. Penalizações por temperatura inadequada
        """
        reward = 0.0
        
        # 1. Economia de custos (principal objetivo)
        # Recompensa por reduzir custos de importação e aumentar autoconsumo
        baseline_cost = raw_obs['demand_kw'] * raw_obs['electricity_price']  # Custo sem bateria
        actual_cost = costs['net_cost_brl']
        cost_savings = baseline_cost - actual_cost
        reward += self.reward_weights['cost_reduction'] * cost_savings * 10  # Amplificação
        
        # 2. Preservação da bateria
        # Penalização por operar fora da temperatura ótima
        temp_penalty = 0
        if (raw_obs['ambient_temperature'] < self.battery.specs.optimal_temp_min or 
            raw_obs['ambient_temperature'] > self.battery.specs.optimal_temp_max):
            temp_deviation = min(
                abs(raw_obs['ambient_temperature'] - self.battery.specs.optimal_temp_min),
                abs(raw_obs['ambient_temperature'] - self.battery.specs.optimal_temp_max)
            )
            temp_penalty = temp_deviation * 0.05
        
        # Penalização por degradação acelerada
        degradation_penalty = (1.0 - self.battery.degradation_factor) * 10
        
        battery_health_reward = -(temp_penalty + degradation_penalty)
        reward += self.reward_weights['battery_health'] * battery_health_reward
        
        # 3. Estabilidade da rede
        # Recompensa por reduzir importação nos horários de ponta
        if 18 <= raw_obs['hour'] < 21:  # Horário de ponta
            grid_import_penalty = energy_balance['grid_import_kw'] * 0.01
            reward -= self.reward_weights['grid_stability'] * grid_import_penalty
        
        # 4. Penalização por temperatura crítica
        if raw_obs['ambient_temperature'] > self.battery.specs.critical_temp_max:
            reward -= self.reward_weights['temperature_penalty'] * 50  # Penalização alta
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executa um passo no ambiente.
        
        Args:
            action: Ação do agente [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        action_value = np.clip(action[0], -1.0, 1.0)
        
        # Observações atuais
        raw_obs = self._get_raw_observation()
        
        # Interpretar ação
        if action_value > 0.05:  # Carregar
            battery_action = 'charge'
            power_kw = action_value * self.battery.specs.max_charge_rate_kw
        elif action_value < -0.05:  # Descarregar
            battery_action = 'discharge' 
            power_kw = abs(action_value) * self.battery.specs.max_discharge_rate_kw
        else:  # Sem ação
            battery_action = 'hold'
            power_kw = 0
        
        # Executar ação na bateria
        if battery_action == 'charge':
            energy_used, new_temp = self.battery.charge(
                power_kw, raw_obs['ambient_temperature'], 1.0  # 1 hora
            )
            battery_charge_kw = power_kw
            battery_discharge_kw = 0
        elif battery_action == 'discharge':
            energy_provided, new_temp = self.battery.discharge(
                power_kw, raw_obs['ambient_temperature'], 1.0
            )
            battery_charge_kw = 0
            battery_discharge_kw = power_kw
        else:
            battery_charge_kw = 0
            battery_discharge_kw = 0
            new_temp = raw_obs['ambient_temperature']
        
        self.battery.temperature = new_temp
        
        # Calcular balanço energético
        energy_balance = self.industrial_system.calculate_energy_balance(
            demand_kw=raw_obs['demand_kw'],
            solar_generation_kw=raw_obs['solar_generation_kw'],
            battery_charge_kw=battery_charge_kw,
            battery_discharge_kw=battery_discharge_kw
        )
        
        # Calcular custos
        costs = self.industrial_system.calculate_costs(
            energy_balance, raw_obs['electricity_price']
        )
        
        # Calcular recompensa
        reward = self._calculate_reward(action_value, raw_obs, energy_balance, costs)
        
        # Avançar tempo
        self.step_count += 1
        self.current_date += timedelta(hours=1)
        
        # Verificar se episódio terminou
        terminated = self.step_count >= self.hours_per_episode
        truncated = False
        
        # Atualizar degradação da bateria a cada dia
        if self.step_count % 24 == 0:
            self.battery.cycles_count += 0.5  # Meio ciclo por dia (estimativa)
            self.battery.update_degradation()
        
        # Nova observação
        if not terminated:
            next_obs = self._normalize_observation(self._get_raw_observation())
        else:
            next_obs = np.zeros_like(self.observation_space.sample())
        
        # Informações adicionais
        info = {
            'energy_balance': energy_balance,
            'costs': costs,
            'battery_soc': self.battery.soc,
            'battery_temperature': self.battery.temperature,
            'battery_degradation': self.battery.degradation_factor,
            'action_interpreted': battery_action,
            'power_kw': power_kw
        }
        
        return next_obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reinicia o ambiente.
        
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Reiniciar componentes
        self.battery = BatteryModel(self.battery.specs)
        self.step_count = 0
        
        # Data inicial aleatória
        start_month = self.np_random.integers(1, 13)
        start_day = self.np_random.integers(1, 28)  # Evitar problemas com fevereiro
        self.current_date = datetime(2024, start_month, start_day, 0, 0)
        
        # Gerar dados climáticos para o episódio
        self.climate_data = self._generate_climate_data(self.current_date)
        
        # Estado inicial da bateria (aleatório)
        self.battery.soc = self.np_random.uniform(0.2, 0.8)
        
        # Primeira observação
        initial_obs = self._normalize_observation(self._get_raw_observation())
        
        info = {'start_date': self.current_date}
        
        return initial_obs, info
    
    def render(self, mode='human'):
        """Basic rendering of current state"""
        raw_obs = self._get_raw_observation()
        battery_state = self.battery.get_state()
        
        print(f"\n=== Battery Thermal RL Environment ===")
        print(f"Timestamp: {raw_obs['timestamp']}")
        print(f"Battery SOC: {battery_state['soc']:.2f}")
        print(f"Battery Temp: {self.battery.temperature:.1f}°C")
        print(f"Ambient Temp: {raw_obs['ambient_temperature']:.1f}°C") 
        print(f"Demand: {raw_obs['demand_kw']:.1f} kW")
        print(f"Solar Gen: {raw_obs['solar_generation_kw']:.1f} kW")
        print(f"Electricity Price: R${raw_obs['electricity_price']:.3f}/kWh")
        print(f"Degradation: {battery_state['degradation_factor']:.3f}")


def create_env(env_config: Dict = None) -> BatteryThermalEnv:
    """
    Factory function to create environment with default configuration.
    
    Args:
        env_config: Custom environment configuration
        
    Returns:
        Configured environment instance
    """
    from ..models.battery import INDUSTRIAL_BATTERY_CONFIGS
    from ..models.industrial_system import TYPICAL_SOLAR_SYSTEMS
    
    config = env_config or {}
    
    battery_specs = INDUSTRIAL_BATTERY_CONFIGS.get(
        config.get('battery_type', 'li_ion_500kwh')
    )
    
    solar_specs = TYPICAL_SOLAR_SYSTEMS.get(
        config.get('solar_system', 'medium_1000kw')
    )
    
    return BatteryThermalEnv(
        battery_specs=battery_specs,
        industrial_profile=config.get('industrial_profile', 'medium_metallurgy'),
        solar_specs=solar_specs,
        climate_region=config.get('climate_region', 'southeast_sp'),
        simulation_days=config.get('simulation_days', 30),
        reward_weights=config.get('reward_weights', None)
    )