# src/core/rl_env.py

import json
import numpy as np

class ContractEnergyEnv:
    def __init__(self, json_fpath):
        with open(json_fpath, 'r') as f:
            self.data = json.load(f)
        self.sources = self.data['metadata']['contracts']
        self.df = self.data['data']
        self.n_sources = len(self.sources)
        self.n_steps = len(self.df)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        prices = [self.df[self.current_step][f'price_{s}'] for s in self.sources]
        avail = [self.df[self.current_step][f'availability_{s}'] for s in self.sources]
        return np.array(prices + avail)

    def step(self, action):
        # Action is a normalized allocation for each source
        action = np.array(action)
        action /= np.sum(action) if np.sum(action) > 0 else 1

        total_demand_kw = 1000  # Assume a constant demand for this simple env
        allocation_kw = action * total_demand_kw

        # Calculate cost
        prices = np.array([self.df[self.current_step][f'price_{s}'] for s in self.sources])
        cost = np.sum(allocation_kw * prices / 1000) # Price is per MWh

        # Calculate sustainability score (example)
        sust_factors = {'solar': 10, 'wind': 9, 'hydropower': 7, 'biomass': 5, 'biogas': 4, 'grid': 2}
        sust_score = np.sum(action * np.array([sust_factors.get(s, 0) for s in self.sources]))

        # Reward: inverse of cost, plus sustainability bonus
        reward = (1 / (cost + 1e-6)) + sust_score * 0.1

        self.current_step += 1
        done = self.current_step >= self.n_steps
        next_state = self._get_state() if not done else np.zeros_like(self._get_state())
        
        return next_state, reward, done, {'cost': cost, 'sust_score': sust_score, 'allocation_kw': allocation_kw}

    def run_random_policy(self, save_path=None):
        self.reset()
        done = False
        history = []
        while not done:
            action = np.random.rand(self.n_sources)
            _, reward, done, info = self.step(action)
            
            history.append({
                'step': self.current_step,
                'reward': reward,
                'cost': info['cost'],
                'sust_score': info['sust_score'],
                'allocation_kw': info['allocation_kw'].tolist()
            })
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(history, f, indent=2)