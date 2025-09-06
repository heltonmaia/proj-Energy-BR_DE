# Criar um novo arquivo: src/train.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from core.rl_dess_env import DessEnv
from core.energy_profile_config import SimulationConfig, DESSConfig

# 1. Configurações
sim_config = SimulationConfig(
    duration_days=30, # Comece com um período curto para treinar mais rápido
    time_resolution_minutes=60,
    dess_config=DESSConfig()
)
profile_path = "data/synthetic/brazil_12m_sol536k_wind157k_profile.csv" # Use um perfil já gerado

# 2. Criar e verificar o ambiente
env = DessEnv(profile_data_path=profile_path, sim_config=sim_config)
# check_env(env) # Bom para depurar o ambiente

# 3. Definir o modelo de RL
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_dess_tensorboard/")

# 4. Treinar o modelo
total_timesteps = 100000
model.learn(total_timesteps=total_timesteps)

# 5. Salvar o modelo
models_dir = "models/PPO"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
model.save(f"{models_dir}/dess_ppo_model_{total_timesteps}")

# Para rodar o tensorboard: tensorboard --logdir ./ppo_dess_tensorboard/