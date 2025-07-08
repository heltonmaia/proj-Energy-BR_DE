# src/core/rl_dess_env.py

# ================================================================= #
#  MUDANÇA PRINCIPAL: Trocar 'gym' por 'gymnasium'                  #
#  Isso alinha o ambiente com o padrão moderno usado pela SB3.      #
# ================================================================= #
import gymnasium as gym
from gymnasium import spaces
# ----------------------------------------------------------------- #

import numpy as np
import pandas as pd
from .dess_system import DESS
from .energy_profile_config import SimulationConfig

class DessEnv(gym.Env):
    """
    Ambiente de Aprendizado por Reforço para gerenciar um 
    Sistema de Suprimento de Energia Descentralizado (DESS).
    
    O agente deve aprender a minimizar os custos operacionais e garantir o 
    suprimento de energia para uma planta industrial.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, profile_data_path: str, sim_config: SimulationConfig):
        super(DessEnv, self).__init__()
        
        # Carrega o cenário pré-gerado
        self.df = pd.read_csv(profile_data_path)
        self.sim_config = sim_config
        
        # Instancia o sistema de armazenamento de energia
        time_step_h = self.sim_config.time_resolution_minutes / 60.0
        self.dess = DESS(sim_config.dess_config, time_step_h)
        
        # --- ESPAÇO DE AÇÃO: O que o agente pode fazer? ---
        # Todas as ações são normalizadas entre -1 e 1 ou 0 e 1.
        # Ação 1: Potência para/da bateria (kW). Negativo=descarregar, Positivo=carregar. [-1, 1]
        # Ação 2: Potência para o eletrolisador (kW). [0, 1]
        # Ação 3: Potência da célula de combustível (kW). [0, 1]
        self.action_space = spaces.Box(low=np.array([-1, 0, 0]), 
                                       high=np.array([1, 1, 1]), 
                                       dtype=np.float32)

        # --- ESPAÇO DE OBSERVAÇÃO: O que o agente vê? ---
        # As observações devem ser normalizadas ou ter uma escala consistente.
        # 1. Hora do dia (normalizada 0-1)
        # 2. Dia da semana (normalizado 0-1)
        # 3. Demanda industrial (kW)
        # 4. Geração solar (kW)
        # 5. Geração eólica (kW)
        # 6. Preço da rede (BRL/MWh)
        # 7. SoC da Bateria (normalizado 0-1)
        # 8. Nível do Tanque de H2 (normalizado 0-1)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)

        self.current_step = 0

    def _get_observation(self):
        """Monta o vetor de observação para o passo atual."""
        # Se a simulação acabar, retorna uma observação de zeros.
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape)

        row = self.df.iloc[self.current_step]
        dess_state = self.dess.get_state()
        
        obs = np.array([
            row['hour'] / 23.0,
            row['day_of_week'] / 6.0,
            row['industrial_consumption_kw'],
            row['solar_generation_kw'],
            row['wind_generation_kw'],
            row['grid_spot_price_brl_per_mwh'],
            dess_state[0], # SoC da bateria já está normalizado
            dess_state[1]  # Nível de H2 já está normalizado
        ])
        return obs

    def reset(self, seed=None, options=None):
        """Reseta o ambiente para um novo episódio."""
        super().reset(seed=seed) # Necessário para compatibilidade com Gymnasium
        self.current_step = 0
        
        # Cria uma nova instância do DESS para começar com tanques e baterias vazios
        time_step_h = self.sim_config.time_resolution_minutes / 60.0
        self.dess = DESS(self.sim_config.dess_config, time_step_h)
        
        # Retorna a observação inicial e um dicionário de informações (padrão do Gymnasium)
        return self._get_observation(), {}

    def step(self, action):
        """Executa um passo no ambiente."""
        # 1. Mapear a ação normalizada para valores de potência reais
        cfg = self.dess.config
        power_to_battery_kw = action[0] * (cfg.battery_max_charge_kw if action[0] > 0 else cfg.battery_max_discharge_kw)
        power_to_electrolyzer_kw = action[1] * cfg.electrolyzer_capacity_kw
        power_from_fuel_cell_kw = action[2] * cfg.fuel_cell_capacity_kw
        
        # 2. Obter dados do cenário para o passo atual
        row = self.df.iloc[self.current_step]
        industrial_demand_kw = row['industrial_consumption_kw']
        on_site_generation_kw = row['solar_generation_kw'] + row['wind_generation_kw']
        grid_price_brl_mwh = row['grid_spot_price_brl_per_mwh']

        # 3. Simular o DESS com as ações do agente
        net_power_dess = self.dess.step(power_to_battery_kw, power_to_electrolyzer_kw, power_from_fuel_cell_kw)
        
        # 4. Fazer o balanço de energia
        # Energia total disponível = (Geração Própria) - (O que foi para o DESS)
        total_available_power = on_site_generation_kw - net_power_dess
        
        # Quanto falta para atender a demanda da fábrica?
        power_deficit = industrial_demand_kw - total_available_power
        
        # Comprar da rede se houver déficit.
        power_from_grid_kw = max(0, power_deficit)
        
        # Energia não atendida (penalidade máxima!)
        unmet_demand_kw = max(0, power_deficit - power_from_grid_kw)

        # 5. Calcular CUSTO (componente da Recompensa)
        time_h = self.sim_config.time_resolution_minutes / 60.0
        
        # Custo da energia comprada da rede
        cost_grid = power_from_grid_kw * (grid_price_brl_mwh / 1000) * time_h
        
        # Custo de degradação/operacional (simplificado) por usar os equipamentos
        cost_op_dess = abs(net_power_dess) * 0.005 # Custo muito pequeno por kW movimentado
        total_cost = cost_grid + cost_op_dess

        # 6. Calculate REWARD
        # Objective: Strongly prioritize resilience (avoid deficit), encourage sustainability and strategic hydrogen use, and still consider cost.
        resilience_score = 0
        # Penalty for unmet demand (resilience)
        resilience_score -= unmet_demand_kw * 200
        # Penalize if battery is always full (>95%)
        battery_soc = self.dess.get_state()[0]
        h2_level = self.dess.get_state()[1]
        if battery_soc > 0.95:
            resilience_score -= 1  # mild penalty for always full
        # Bonus for healthy SoC range (30-70%)
        if 0.3 < battery_soc < 0.7:
            resilience_score += 2

        # Strategic hydrogen use: bonus only when grid price is high
        grid_price = row['grid_spot_price_brl_per_mwh']
        high_price_threshold = 400  # adjust as needed
        if grid_price > high_price_threshold and 0.2 < h2_level < 0.8:
            resilience_score += 4  # strong bonus only when grid price is high
        # Penalize if H2 is high when grid price is low (to avoid unnecessary storage)
        if grid_price <= high_price_threshold and h2_level > 0.5:
            resilience_score -= 2  # mild penalty
        # Penalize if H2 is always low (<10%)
        if h2_level < 0.1:
            resilience_score -= 2

        # Sustainability: reward for using renewables (bonus for >80% renewables)
        total_consumption = industrial_demand_kw
        renewables_used = min(on_site_generation_kw, total_consumption)
        sustainability_score = 0
        if total_consumption > 0:
            frac_renew = renewables_used / total_consumption
            sustainability_score = frac_renew * 2  # base reward
            if frac_renew > 0.8:
                sustainability_score += 3  # extra bonus for high renewable share

        # Weighted reward
        alpha = 1.0   # cost weight
        beta = 20.0   # resilience weight
        gamma = 10.0  # sustainability weight
        reward = -alpha * total_cost + beta * resilience_score + gamma * sustainability_score

        # Save individual scores for analysis/plotting
        info = {
            'total_cost': total_cost,
            'cost_grid': cost_grid,
            'unmet_demand_kw': unmet_demand_kw,
            'power_from_grid_kw': power_from_grid_kw,
            'battery_soc': self.dess.get_state()[0],
            'h2_storage_level': self.dess.get_state()[1],
            'cost_score': -alpha * total_cost,
            'resilience_score': beta * resilience_score,
            'sustainability_score': gamma * sustainability_score
        }
        
        # 7. Lógica de Fim de Episódio
        self.current_step += 1
        done = self.current_step >= len(self.df)
        
        # Padrão de retorno do Gymnasium: observação, recompensa, terminado, truncado, info
        return self._get_observation(), reward, done, False, info

    def close(self):
        """Limpeza de recursos, se necessário."""
        pass