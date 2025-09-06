0# src/app_cli.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import locale
import os
import glob
import shutil

# Adiciona o diretório raiz ao path para garantir que todos os módulos sejam encontrados
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

# --- Imports para Treinamento e Avaliação ---
from stable_baselines3 import PPO
from core.dess_system import DESS 
from core.rl_dess_env import DessEnv
from core.evaluate import run_evaluation # Importa a função de avaliação de 'core'

# --- Imports para Geração de Dados e Configuração ---
from core.energy_profile_config import (
    get_configs_for_country, SimulationConfig, Country,
    IndustrialConfig, OnSiteGenerationConfig, DESSConfig
)
from core.synthetic_data_generator import ContractDataGenerator, HistoricalPatternLoader
from core.energy_profile_generator import EnergyProfileGenerator

# --- Imports para Plotagem ---
from utils.plot import plot_energy_profiles, plot_monthly_consumption_summary, plot_real_pld, plot_monthly_cost_summary

# Configura o locale para formatação de moeda
try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except locale.Error:
    print("Warning: Brazilian locale 'pt_BR.UTF-8' not found. Using default for currency formatting.")
    locale.setlocale(locale.LC_TIME, 'C')


def run_contract_price_generation():
    """Generates synthetic PRICE data and a validation plot with REAL historical data."""
    print("\n--- Mode: Contract Price Generation (Historical Base) ---")
    
    project_root = Path(__file__).resolve().parent.parent
    historical_data_path = project_root / "data" / "real" / "Historico_do_Preco_Medio_Semanal_-_30_de_junho_de_2001_a_30_de_maio_de_2025.json"

    if not historical_data_path.exists():
        print(f"\n[CRITICAL ERROR] Historical data file not found at: {historical_data_path}\n")
        return

    pattern_loader = HistoricalPatternLoader(historical_data_path)
    regions = pattern_loader.regions
    print("\nAvailable Regions for Price Patterns:")
    for i, region in enumerate(regions):
        print(f"  {i+1}. {region}")
    
    try:
        region_choice = int(input(f"Select a region by number (default: 1 - {regions[0]}): ") or 1)
        selected_region = regions[region_choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Using default region (SOUTHEAST).")
        selected_region = "SOUTHEAST"

    # Pergunta o ano ao usuário
    try:
        year_choice = input("Enter the year to extract the price pattern (leave blank for all years): ").strip()
        year_choice = int(year_choice) if year_choice else None
    except ValueError:
        print("Invalid year. Using all available years.")
        year_choice = None

    sim_config = SimulationConfig(
        duration_days=365
    )
    if year_choice:
        pattern = pattern_loader.calculate_pattern_for_region(selected_region, base_year=year_choice)
    else:
        pattern = pattern_loader.calculate_pattern_for_region(selected_region)

    # Defina o sufixo do ano para o nome do arquivo
    year_suffix = f"_{year_choice}" if year_choice else ""
    sim_config.experiment_name = f"exp{year_suffix}"

    generator = ContractDataGenerator(get_configs_for_country(sim_config.country), sim_config, pattern)
    price_df = generator.generate_contract_profile()
    
    output_dir = project_root / "data" / "synthetic"
    generator.save_data(price_df, output_dir)
    print(f"Data saved to: {output_dir / (sim_config.experiment_name + '.json')}")

    # Debug: Verifique se o DataFrame tem dados e colunas de preço
    print("\n[DEBUG] price_df head:")
    print(price_df.head())
    print("[DEBUG] price_df columns:")
    print(price_df.columns)
    if price_df.empty:
        print("[WARNING] price_df is empty, nothing to plot!")
    else:
        any_plotted = False
        plt.figure(figsize=(14, 6))
        for col in price_df.columns:
            if col.startswith('price_'):
                print(f"[DEBUG] Plotting column: {col}")
                plt.plot(price_df.index, price_df[col], label=col.replace('price_', '').capitalize())
                any_plotted = True
        if not any_plotted:
            print("[WARNING] No price_ columns found to plot!")
        plt.title(f"Synthetic Contract Prices ({sim_config.experiment_name})")
        plt.xlabel("Day")
        plt.ylabel("Price (BRL/MWh)")
        plt.legend()
        plt.tight_layout()
        plot_path = output_dir / f"{sim_config.experiment_name}_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Validation plot saved to: {plot_path}")

def run_full_energy_profile_generation():
    """Generates a complete energy profile based on user-defined ENERGY targets."""
    print("\n--- Mode: Generate Full Energy Profile (Industry) ---")
    
    # --- Fase 1: Coleta de Dados do Usuário (Baseado em Energia) ---
    try:
        months = int(input("Enter number of months to simulate (default: 12): ").strip() or 12)
        total_consumption_target_mwh = float(input("Enter total planned energy consumption for the period in MWh (e.g., 2000): ").strip())
        solar_share_percentage = float(input("Enter the desired % of consumption met by SOLAR (e.g., 40): ").strip()) / 100.0
        wind_share_percentage = float(input("Enter the desired % of consumption met by WIND (e.g., 20): ").strip()) / 100.0
    except ValueError:
        print("Invalid input. Aborting simulation.")
        return

    sim_config = SimulationConfig(
        duration_days=months * 30,
        time_resolution_minutes=15,
        country=Country.BRAZIL,
        random_seed=42,
    )
    # ... (o resto desta função permanece o mesmo que nas versões anteriores) ...
    # O código foi omitido para brevidade, mas deve ser incluído aqui.
    print("\nStep 1: Calculating required power capacity to meet energy targets...")
    unit_gen = EnergyProfileGenerator(sim_config, IndustrialConfig(), OnSiteGenerationConfig(solar_installed_kw=1, wind_installed_kw=1))
    unit_df = unit_gen.generate_profiles()
    time_step_h = sim_config.time_resolution_minutes / 60.0
    kwh_per_kw_solar = (unit_df['solar_generation_kw'] * time_step_h).sum()
    kwh_per_kw_wind = (unit_df['wind_generation_kw'] * time_step_h).sum()

    target_solar_kwh = total_consumption_target_mwh * 1000 * solar_share_percentage
    target_wind_kwh = total_consumption_target_mwh * 1000 * wind_share_percentage
    
    required_solar_kw = target_solar_kwh / kwh_per_kw_solar if kwh_per_kw_solar > 0 else 0
    required_wind_kw = target_wind_kwh / kwh_per_kw_wind if kwh_per_kw_wind > 0 else 0
    
    print(f" -> To meet targets, you need ~{required_solar_kw:.0f} kW of Solar and ~{required_wind_kw:.0f} kW of Wind.")

    # --- NOVO: Perguntar qual arquivo de contratos usar ---
    project_root = Path(__file__).resolve().parent.parent
    contract_jsons = sorted(glob.glob(str(project_root / 'data' / 'synthetic' / 'exp_*.json')))
    if not contract_jsons:
        print("[ERROR] No contract JSON files found. Please run option 1 first.")
        return
    print("\nAvailable contract JSON files:")
    for i, f in enumerate(contract_jsons):
        print(f"  {i+1}. {os.path.basename(f)}")
    try:
        contract_choice = int(input(f"Select contract JSON by number (default: 1): ") or 1)
        contract_json_path = contract_jsons[contract_choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Using first contract JSON.")
        contract_json_path = contract_jsons[0]
    print(f"Using contract file: {os.path.basename(contract_json_path)}")
    contract_base = os.path.splitext(os.path.basename(contract_json_path))[0]
    with open(contract_json_path, 'r') as f:
        contract_data = json.load(f)
    contract_df = pd.DataFrame(contract_data['data'])

    # Ajuste o nome dos arquivos gerados para incluir o prefixo do contrato
    sim_config.experiment_name = f"{contract_base}_exp_{sim_config.country.value}_{months}m_sol{int(required_solar_kw)}k_wind{int(required_wind_kw)}k"

    print("\nStep 2: Generating price, generation, and consumption profiles...")
    historical_data_path = project_root / "data" / "real" / "Historico_do_Preco_Medio_Semanal_-_30_de_junho_de_2001_a_30_de_maio_de_2025.json"
    pattern_loader = HistoricalPatternLoader(historical_data_path)
    grid_pattern = pattern_loader.calculate_pattern_for_region("SOUTHEAST")
    contract_gen = ContractDataGenerator(get_configs_for_country(sim_config.country), sim_config, grid_pattern)
    price_df = contract_gen.generate_contract_profile()
    
    industrial_cfg = IndustrialConfig()
    generation_cfg = OnSiteGenerationConfig(solar_installed_kw=required_solar_kw, wind_installed_kw=required_wind_kw)
    profile_gen = EnergyProfileGenerator(sim_config, industrial_cfg, generation_cfg)
    profile_df = profile_gen.generate_profiles()

    print("\nStep 3: Adjusting consumption and calculating detailed costs...")
    current_total_consumption_kwh = (profile_df['industrial_consumption_kw'] * time_step_h).sum()
    scaling_factor = (total_consumption_target_mwh * 1000) / current_total_consumption_kwh
    profile_df['industrial_consumption_kw'] *= scaling_factor

    for month_num in range(1, months + 1):
        monthly_variation = 1 + np.random.uniform(-0.05, 0.05)
        profile_df.loc[profile_df['timestamp'].dt.month == month_num, 'industrial_consumption_kw'] *= monthly_variation
    
    full_df = pd.merge(profile_df, contract_df[['day', 'price_grid']], left_on='day_of_year', right_on='day', how='left').ffill()
    full_df.rename(columns={'price_grid': 'grid_spot_price_brl_per_mwh'}, inplace=True)
    full_df.drop(columns=['day', 'day_of_year'], inplace=True)
    
    full_df['solar_used_kw'] = np.minimum(full_df['solar_generation_kw'], full_df['industrial_consumption_kw'])
    remaining_demand = full_df['industrial_consumption_kw'] - full_df['solar_used_kw']
    full_df['wind_used_kw'] = np.minimum(full_df['wind_generation_kw'], remaining_demand)
    remaining_demand -= full_df['wind_used_kw']
    full_df['grid_used_kw'] = remaining_demand.clip(lower=0)

    full_df['cost_solar'] = full_df['solar_used_kw'] * (generation_cfg.solar_lcoe_brl_per_mwh / 1000) * time_step_h
    full_df['cost_wind'] = full_df['wind_used_kw'] * (generation_cfg.wind_lcoe_brl_per_mwh / 1000) * time_step_h
    
    grid_contract_used = np.minimum(full_df['grid_used_kw'], industrial_cfg.grid_contract_volume_kw)
    grid_spot_used = (full_df['grid_used_kw'] - industrial_cfg.grid_contract_volume_kw).clip(lower=0)
    
    full_df['cost_grid_contract'] = grid_contract_used * (industrial_cfg.grid_contract_price_brl_per_mwh / 1000) * time_step_h
    full_df['cost_grid_spot'] = grid_spot_used * (full_df['grid_spot_price_brl_per_mwh'] / 1000) * time_step_h

    print("\nStep 4: Saving data and generating plots...")
    output_dir = project_root / "data" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{sim_config.experiment_name}.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"\nFull energy profile with costs saved to: {csv_path}")

    plot1_path = output_dir / f"{sim_config.experiment_name}_weekly_detail.png"
    title = f"Energy Profile Detail (First Week)\nTarget: {total_consumption_target_mwh} MWh, Solar: {solar_share_percentage:.0%}, Wind: {wind_share_percentage:.0%}"
    plot_energy_profiles(full_df.head(96 * 7), title, save_path=plot1_path)
    print(f"Didactic weekly plot saved to: {plot1_path}")

    plot2_path = output_dir / f"{sim_config.experiment_name}_monthly_consumption.png"
    plot_monthly_consumption_summary(full_df, sim_config, save_path=plot2_path)
    print(f"Monthly consumption plot saved to: {plot2_path}")
    
    plot3_path = output_dir / f"{sim_config.experiment_name}_monthly_costs.png"
    plot_monthly_cost_summary(full_df, sim_config, save_path=plot3_path)
    print(f"Monthly cost analysis plot saved to: {plot3_path}")

    print("\n" + "="*50)
    print("Simulation complete.")
    print("="*50 + "\n")

def run_training_session():
    """Handles the RL agent training process."""
    print("\n--- Mode: Train DESS Management Agent (RL) ---")
    project_root = Path(__file__).resolve().parent.parent

    # 1. Selecionar o cenário (arquivo .csv)
    profile_dir = project_root / "data" / "synthetic"
    if not profile_dir.exists():
        print(f"[ERROR] Directory not found: {profile_dir}")
        print("Please generate a profile using Option 2 first.")
        return

    csv_files = sorted([f for f in profile_dir.glob("*.csv")])
    if not csv_files:
        print("[ERROR] No .csv profiles found in 'data/synthetic'.")
        print("Please generate a profile using Option 2 first.")
        return

    print("\nAvailable profiles for training:")
    for i, f in enumerate(csv_files):
        print(f"  {i+1}. {f.name}")

    try:
        choice = int(input(f"Choose the profile by number: ").strip())
        profile_path = csv_files[choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Aborting.")
        return

    # 2. Obter parâmetros de treinamento do usuário
    try:
        total_timesteps = int(input("Enter total training timesteps (e.g., 100000): ").strip())
    except ValueError:
        print("Invalid number. Aborting.")
        return

    # 3. Configurar e criar o ambiente
    sim_config = SimulationConfig(
        dess_config=DESSConfig()
    )
    # ... (o resto desta função permanece o mesmo que nas versões anteriores) ...
    # O código foi omitido para brevidade, mas deve ser incluído aqui.
    print("\nCreating the RL environment...")
    env = DessEnv(profile_data_path=str(profile_path), sim_config=sim_config)
    print("Environment created successfully.")

    model_name = f"PPO_{profile_path.stem}_{total_timesteps}steps"
    models_dir = project_root / "models"
    log_dir = project_root / "logs"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(log_dir))

    print(f"\n--- Starting Training ---")
    print(f"Model: {model_name}")
    print(f"Environment Profile: {profile_path.name}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print("-" * 25)

    try:
        model.learn(total_timesteps=total_timesteps, tb_log_name=model_name)
    except Exception as e:
        print("\n[FATAL ERROR] Training failed.")
        traceback.print_exc()
        return

    print("\n--- Training Complete ---")

    model_path = models_dir / f"{model_name}.zip"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    print("\nTo visualize the training results, run this command in a new terminal:")
    print(f"  tensorboard --logdir {log_dir}")
    print("=" * 50 + "\n")

def clean_pycache_folders():
    """Recursively delete all __pycache__ folders in the project."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    removed = 0
    for dirpath, dirnames, filenames in os.walk(project_root):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed: {pycache_path}")
                removed += 1
            except Exception as e:
                print(f"[ERROR] Could not remove {pycache_path}: {e}")
    if removed == 0:
        print("No __pycache__ folders found.")
    else:
        print(f"Total __pycache__ folders removed: {removed}")


def main():
    """Main function that displays the menu and directs to other functions."""
    while True:
        print("\n" + "="*50)
        print("{:^50}".format("Energy & RL Simulation CLI"))
        print("="*50)
        print("{:^50}".format("Main Menu"))
        print("-"*50)
        print("  1. Generate Contract Prices & Validation Plot")
        print("  2. Generate Full Industry Profile (Energy-based)")
        print("  3. Train DESS Management Agent (RL)")
        print("  4. Evaluate Trained Agent")
        print("  5. Clean all __pycache__ folders")
        print("  0. Exit")
        print("-"*50)
        choice = input("Select an option: ").strip()

        if choice == '1':
            try:
                run_contract_price_generation()
            except Exception as e:
                print(f"\n[ERROR] Contract price generation failed: {e}")
                traceback.print_exc()
        elif choice == '2':
            try:
                run_full_energy_profile_generation()
            except Exception as e:
                print(f"\n[ERROR] Full energy profile generation failed: {e}")
                traceback.print_exc()
        elif choice == '3':
            try:
                run_training_session()
            except Exception as e:
                print(f"\n[ERROR] Training session failed: {e}")
                traceback.print_exc()
        elif choice == '4':
            try:
                run_evaluation()
            except Exception as e:
                print(f"\n[ERROR] Evaluation failed: {e}")
                traceback.print_exc()
        elif choice == '5':
            clean_pycache_folders()
        elif choice == '0':
            print("Exiting.")
            break
        else:
            print("Invalid option. Please enter 1, 2, 3, 4, 5 or 0.")

if __name__ == "__main__":
    main()