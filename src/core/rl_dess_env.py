# src/core/rl_dess_env.py

# ================================================================= #
#  MAIN CHANGE: Replace 'gym' with 'gymnasium'                      #
#  This aligns the environment with the modern standard used by SB3.#
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
    Reinforcement Learning Environment for managing a 
    Decentralized Energy Supply System (DESS).
    
    The agent must learn to minimize operational costs and ensure 
    energy supply for an industrial plant.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, profile_data_path: str, sim_config: SimulationConfig):
        super(DessEnv, self).__init__()
        
        # Load the pre-generated scenario
        self.df = pd.read_csv(profile_data_path)
        self.sim_config = sim_config
        
        # Instantiate the energy storage system
        time_step_h = self.sim_config.time_resolution_minutes / 60.0
        self.dess = DESS(sim_config.dess_config, time_step_h)
        
        # --- ACTION SPACE: What can the agent do? ---
        # All actions are normalized between -1 and 1 or 0 and 1.
        # Action 1: Power to/from battery (kW). Negative=discharge, Positive=charge. [-1, 1]
        # Action 2: Power to electrolyzer (kW). [0, 1]
        # Action 3: Power from fuel cell (kW). [0, 1]
        self.action_space = spaces.Box(low=np.array([-1, 0, 0]), 
                                       high=np.array([1, 1, 1]), 
                                       dtype=np.float32)

        # --- OBSERVATION SPACE: What does the agent see? ---
        # Observations should be normalized or have a consistent scale.
        # 1. Hour of day (normalized 0-1)
        # 2. Day of week (normalized 0-1)
        # 3. Industrial demand (kW)
        # 4. Solar generation (kW)
        # 5. Wind generation (kW)
        # 6. Grid price (BRL/MWh)
        # 7. Battery SoC (normalized 0-1)
        # 8. H2 tank level (normalized 0-1)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)

        self.current_step = 0

    def _get_observation(self):
        """Builds the observation vector for the current step."""
        # If the simulation is over, return a zero observation.
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
            dess_state[0], # Battery SoC already normalized
            dess_state[1]  # H2 level already normalized
        ])
        return obs

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed) # Required for Gymnasium compatibility
        self.current_step = 0
        
        # Create a new DESS instance to start with empty tanks and batteries
        time_step_h = self.sim_config.time_resolution_minutes / 60.0
        self.dess = DESS(self.sim_config.dess_config, time_step_h)
        
        # Return the initial observation and an info dictionary (Gymnasium standard)
        return self._get_observation(), {}

    def step(self, action):
        """Performs a step in the environment."""
        # 1. Map the normalized action to real power values
        cfg = self.dess.config
        power_to_battery_kw = action[0] * (cfg.battery_max_charge_kw if action[0] > 0 else cfg.battery_max_discharge_kw)
        power_to_electrolyzer_kw = action[1] * cfg.electrolyzer_capacity_kw
        power_from_fuel_cell_kw = action[2] * cfg.fuel_cell_capacity_kw
        
        # 2. Get scenario data for the current step
        row = self.df.iloc[self.current_step]
        industrial_demand_kw = row['industrial_consumption_kw']
        on_site_generation_kw = row['solar_generation_kw'] + row['wind_generation_kw']
        grid_price_brl_mwh = row['grid_spot_price_brl_per_mwh']

        # 3. Simulate the DESS with the agent's actions
        net_power_dess = self.dess.step(power_to_battery_kw, power_to_electrolyzer_kw, power_from_fuel_cell_kw)
        
        # 4. Energy balance
        # Total available energy = (On-site generation) - (What went to DESS)
        total_available_power = on_site_generation_kw - net_power_dess
        
        # How much is missing to meet the factory demand?
        power_deficit = industrial_demand_kw - total_available_power
        
        # Buy from the grid if there is a deficit.
        power_from_grid_kw = max(0, power_deficit)
        
        # Unmet demand (maximum penalty!)
        unmet_demand_kw = max(0, power_deficit - power_from_grid_kw)

        # 5. Calculate COST (component of the reward)
        time_h = self.sim_config.time_resolution_minutes / 60.0
        
        # Cost of energy purchased from the grid
        cost_grid = power_from_grid_kw * (grid_price_brl_mwh / 1000) * time_h
        
        # Degradation/operational cost (simplified) for using the equipment
        cost_op_dess = abs(net_power_dess) * 0.005 # Very small cost per kW moved
        total_cost = cost_grid + cost_op_dess

        # 6. Calculate REWARD
        # Objective: Strongly prioritize resilience (avoid deficit), encourage sustainability and strategic hydrogen use, and still consider cost.
        resilience_score = 0
        # Define battery_soc and h2_level before using them
        battery_soc = self.dess.get_state()[0]
        h2_level = self.dess.get_state()[1]

        # Strong incentive for battery in healthy range
        if 0.3 < battery_soc < 0.8:
            resilience_score += 5  # strong bonus for healthy SoC range
        if battery_soc < 0.05:
            resilience_score -= 10  # strong penalty if battery is empty
        if battery_soc > 0.95:
            resilience_score -= 2  # penalty if battery is always full

        # Strategic hydrogen use: dynamic threshold based on moving average of recent price history
        window = 96 * 7  # 1 week of 15-min steps
        recent_prices = self.df['grid_spot_price_brl_per_mwh'].iloc[max(0, self.current_step-window):self.current_step+1]
        if len(recent_prices) > 0:
            moving_avg = np.mean(recent_prices)
        else:
            moving_avg = 400  # fallback if not enough history

        # Explicit bonus for using H2 via fuel cell when grid price is high
        fuel_cell_used = abs(power_from_fuel_cell_kw) * self.sim_config.time_resolution_minutes / 60.0  # kWh generated in the step
        if grid_price_brl_mwh > moving_avg and fuel_cell_used > 0:
            resilience_score += 5 * fuel_cell_used  # bonus proportional to H2 use via fuel cell when price is high
        if grid_price_brl_mwh <= moving_avg and h2_level > 0.5:
            resilience_score -= 2  # light penalty to avoid excess H2 when not needed

        # Penalize if H2 is always low (<10%)
        if h2_level < 0.1:
            resilience_score -= 2

        # Bonus for battery discharge during high price
        battery_discharge = max(0, -power_to_battery_kw) * self.sim_config.time_resolution_minutes / 60.0  # kWh discharged
        if grid_price_brl_mwh > moving_avg and battery_discharge > 0:
            resilience_score += 5 * battery_discharge  # bonus proportional for battery discharge

        # Bonus for keeping battery and H2 at intermediate levels (20%-80%)
        if 0.2 < battery_soc < 0.8:
            resilience_score += 3  # bonus for healthy SoC
        if 0.2 < h2_level < 0.8:
            resilience_score += 3  # bonus for healthy H2

        # Penalty for very low stocks (<10%)
        if battery_soc < 0.1:
            resilience_score -= 4  # penalty for nearly empty battery
        if h2_level < 0.1:
            resilience_score -= 4  # penalty for nearly empty H2

        # Bonus for battery discharge during high price (adjusted)
        if grid_price_brl_mwh > moving_avg and battery_discharge > 0:
            resilience_score += 3 * battery_discharge  # adjusted bonus

        # Even higher bonus for H2 use via fuel cell during high price
        if grid_price_brl_mwh > moving_avg and fuel_cell_used > 0:
            resilience_score += 6 * fuel_cell_used  # adjusted bonus

        # Penalty if battery or H2 stay full for too long
        if battery_soc > 0.9:
            resilience_score -= 2  # penalty for full battery
        if h2_level > 0.9:
            resilience_score -= 2  # penalty for full H2

        # Penalty if price is high and storage is not used
        if grid_price_brl_mwh > moving_avg and battery_discharge == 0 and fuel_cell_used == 0:
            resilience_score -= 10  # strong penalty for not using storage during high price

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
        
        # 7. End of episode logic
        self.current_step += 1
        done = self.current_step >= len(self.df)
        
        # Gymnasium return pattern: observation, reward, terminated, truncated, info
        return self._get_observation(), reward, done, False, info

    def close(self):
        """Resource cleanup, if needed."""
        pass