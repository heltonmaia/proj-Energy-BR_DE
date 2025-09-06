# src/core/evaluate.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path para garantir que 'stable_baselines3' seja encontrado
# e para consistência, embora os imports abaixo agora sejam relativos.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO

# --- Imports Relativos ---
# Como estamos dentro de 'core', usamos '.' para importar de arquivos no mesmo diretório.
from .rl_dess_env import DessEnv
from .energy_profile_config import SimulationConfig, DESSConfig

def plot_evaluation_results(history_df, output_path, title):
    """
    Plota os resultados detalhados de uma simulação de avaliação.
    """
    print("Gerando gráfico de avaliação...")
    fig, axs = plt.subplots(5, 1, figsize=(20, 20), sharex=True,
                           gridspec_kw={'height_ratios': [3, 2, 2, 2, 1.5]})

    fig.suptitle(title, fontsize=16, y=0.99)

    # --- Energy Balance Plot (kW) ---
    axs[0].plot(history_df.index, history_df['industrial_demand_kw'], label='Industrial Demand', color='black', linewidth=2.5, zorder=5)
    axs[0].plot(history_df.index, history_df['on_site_generation_kw'], label='On-site Generation (Solar+Wind)', color='green', linestyle='--', zorder=4)
    axs[0].plot(history_df.index, history_df['power_from_grid_kw'], label='Grid Purchase', color='red', alpha=0.8, zorder=3)
    axs[0].fill_between(history_df.index, history_df['power_from_grid_kw'], 0, color='red', alpha=0.2, label='Deficit (Purchased from Grid)')
    axs[0].set_ylabel('Power (kW)')
    axs[0].set_title('Energy Balance')
    axs[0].legend()
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Storage Levels Plot ---
    ax2_twin = axs[1].twinx()
    axs[1].plot(history_df.index, history_df['battery_soc'] * 100, label='Battery SoC (%)', color='blue')
    ax2_twin.plot(history_df.index, history_df['h2_storage_level'] * 100, label='Hydrogen Level (%)', color='magenta')
    axs[1].set_ylabel('Battery (%)', color='blue')
    ax2_twin.set_ylabel('Hydrogen (%)', color='magenta')
    axs[1].tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='magenta')
    axs[1].set_title('Storage Levels')
    axs[1].set_ylim(-5, 105)
    ax2_twin.set_ylim(-5, 105)
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # --- Agent Actions (DESS Control Decisions) ---
    axs[2].plot(history_df.index, history_df['action_battery_kw'], label='Battery (Charge/Discharge)', color='blue')
    axs[2].plot(history_df.index, history_df['action_electrolyzer_kw'], label='Electrolyzer (H2 Production)', color='orange')
    axs[2].plot(history_df.index, history_df['action_fuel_cell_kw'], label='Fuel Cell (Generation)', color='magenta')
    axs[2].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axs[2].set_ylabel('Power (kW)')
    axs[2].set_title('Agent Actions (DESS Control Decisions)')
    axs[2].legend()
    axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Cost and Grid Price Analysis ---
    ax4_twin = axs[3].twinx()
    axs[3].plot(history_df.index, history_df['cumulative_cost'], label='Cumulative Cost (BRL)', color='red')
    ax4_twin.plot(history_df.index, history_df['grid_price'], label='Grid Price (BRL/MWh)', color='purple', alpha=0.5)
    axs[3].set_ylabel('Cumulative Cost (BRL)', color='red')
    ax4_twin.set_ylabel('Grid Price (BRL/MWh)', color='purple')
    axs[3].tick_params(axis='y', labelcolor='red')
    ax4_twin.tick_params(axis='y', labelcolor='purple')
    axs[3].set_title('Cost vs. Grid Price Analysis')
    axs[3].set_xlabel('Time Step (15 min resolution)')
    axs[3].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # --- Objective Scores Over Time ---
    axs[4].plot(history_df.index, history_df['cost_score'], label='Cost Score', color='red')
    axs[4].plot(history_df.index, history_df['resilience_score'], label='Resilience Score', color='blue')
    axs[4].plot(history_df.index, history_df['sustainability_score'], label='Sustainability Score', color='green')
    axs[4].legend()
    axs[4].set_title('Objective Scores Over Time')
    axs[4].set_ylabel('Score')
    axs[4].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[4].set_xlabel('Time Step (15 min resolution)')
    
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Gráfico de avaliação salvo em: {output_path}")

# --- MUDANÇA PRINCIPAL: FUNÇÃO RENOMEADA DE 'main' PARA 'run_evaluation' ---
def run_evaluation():
    """
    Função principal para carregar um agente treinado e avaliar seu desempenho.
    """
    # Path agora é relativo ao local do script (src/core)
    project_root = Path(__file__).resolve().parents[2]

    # --- 1. Selecionar o Modelo Treinado (.zip) ---
    models_dir = project_root / "models"
    if not models_dir.exists():
        print(f"[ERROR] Pasta de modelos não encontrada: {models_dir}")
        return
    
    zip_files = sorted(list(models_dir.glob("**/*.zip")))
    if not zip_files:
        print("[ERROR] Nenhum modelo .zip treinado encontrado na pasta 'models'.")
        return

    print("\nModelos treinados disponíveis:")
    for i, f in enumerate(zip_files):
        print(f"  {i+1}. {f.name}")

    try:
        choice = int(input("Escolha o modelo para avaliar: ").strip())
        model_path = zip_files[choice - 1]
    except (ValueError, IndexError):
        print("Seleção inválida. Abortando.")
        return

    # --- 2. Encontrar o Perfil .csv correspondente ---
    try:
        parts = model_path.stem.split('_')
        profile_name = "_".join(parts[1:-1])
        profile_path = project_root / "data" / "synthetic" / f"{profile_name}.csv"
        if not profile_path.exists():
            raise FileNotFoundError
    except (IndexError, FileNotFoundError):
        print(f"[ERROR] Não foi possível encontrar o perfil '{profile_name}.csv' correspondente.")
        csv_files = sorted([f for f in (project_root / "data" / "synthetic").glob("*.csv")])
        if not csv_files:
            print("[ERROR] Nenhum perfil .csv encontrado. Abortando.")
            return
        print("\nPor favor, selecione o arquivo de cenário manualmente:")
        for i, f in enumerate(csv_files):
            print(f"  {i+1}. {f.name}")
        try:
            csv_choice = int(input(f"Escolha o perfil .csv: ").strip())
            profile_path = csv_files[csv_choice - 1]
        except (ValueError, IndexError):
            print("Seleção inválida. Abortando.")
            return

    print(f"\nAvaliando modelo '{model_path.name}' com o cenário '{profile_path.name}'...")

    # --- 3. Configurar e Criar o Ambiente ---
    sim_config = SimulationConfig(
        dess_config=DESSConfig()
    )
    env = DessEnv(profile_data_path=str(profile_path), sim_config=sim_config)

    # --- 4. Carregar o Modelo ---
    model = PPO.load(model_path, env=env)

    # --- 5. Rodar a Simulação de Avaliação ---
    obs, info = env.reset()
    done = False
    history = []
    
    print("Iniciando simulação de avaliação...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        current_data = env.df.iloc[env.current_step - 1]
        
        cfg = env.dess.config
        action_battery_kw = action[0] * (cfg.battery_max_charge_kw if action[0] > 0 else cfg.battery_max_discharge_kw)
        action_electrolyzer_kw = action[1] * cfg.electrolyzer_capacity_kw
        action_fuel_cell_kw = action[2] * cfg.fuel_cell_capacity_kw
        
        log_entry = {
            'industrial_demand_kw': current_data['industrial_consumption_kw'],
            'on_site_generation_kw': current_data['solar_generation_kw'] + current_data['wind_generation_kw'],
            'power_from_grid_kw': info['power_from_grid_kw'],
            'battery_soc': info['battery_soc'],
            'h2_storage_level': info['h2_storage_level'],
            'action_battery_kw': action_battery_kw,
            'action_electrolyzer_kw': action_electrolyzer_kw,
            'action_fuel_cell_kw': action_fuel_cell_kw,
            'total_cost': info['total_cost'],
            'grid_price': current_data['grid_spot_price_brl_per_mwh'],
            'cost_score': info['cost_score'],
            'resilience_score': info['resilience_score'],
            'sustainability_score': info['sustainability_score']
        }
        history.append(log_entry)
    
    print("Simulação concluída.")

    # --- 6. Processar e Plotar Resultados ---
    history_df = pd.DataFrame(history)
    history_df['cumulative_cost'] = history_df['total_cost'].cumsum()

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    csv_output_path = results_dir / f"evaluation_{model_path.stem}.csv"
    history_df.to_csv(csv_output_path, index=False)
    print(f"\nDados detalhados da avaliação salvos em: {csv_output_path}")

    plot_output_path_detail = results_dir / f"evaluation_{model_path.stem}_detail_14_days.png"
    title_detail = f"Desempenho do Agente (Detalhe de 14 dias)\nModelo: {model_path.name}"
    plot_evaluation_results(history_df.head(96 * 14), plot_output_path_detail, title_detail)
    
    plot_output_path_full = results_dir / f"evaluation_{model_path.stem}_full_period.png"
    title_full = f"Desempenho do Agente (Período Completo)\nModelo: {model_path.name}"
    plot_evaluation_results(history_df, plot_output_path_full, title_full)

# Este bloco permite que o script seja executável por si só, se necessário.
if __name__ == "__main__":
    run_evaluation() # <-- CHAMADA À FUNÇÃO RENOMEADA