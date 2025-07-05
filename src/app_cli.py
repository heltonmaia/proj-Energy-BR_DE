# src/app_cli.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import locale

# Add root directory to path for correct imports
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

# Project imports
from core.energy_profile_config import (
    get_configs_for_country, SimulationConfig, Country, 
    IndustrialConfig, OnSiteGenerationConfig
)
from core.synthetic_data_generator import ContractDataGenerator, HistoricalPatternLoader
from core.energy_profile_generator import EnergyProfileGenerator
from core.rl_env import ContractEnergyEnv
# Updated plot functions with a new cost plot
from utils.plot import plot_rl_results, plot_energy_profiles, plot_monthly_consumption_summary, plot_real_pld, plot_monthly_cost_summary

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

    print("\nGeneration Modes:")
    print("  1. Generate for a single specific year.")
    print("  2. Generate for ALL available years.")
    mode_choice = input(f"Choose mode for region {selected_region} (1 or 2, default: 1): ").strip() or '1'

    available_years = pattern_loader.get_available_years()
    years_to_process = []
    if mode_choice == '1':
        print("\nAvailable base years:", ", ".join(map(str, available_years)))
        year_input = input("Enter a base year (or leave blank for all-time average): ").strip()
        years_to_process.append(int(year_input) if year_input else None)
    elif mode_choice == '2':
        years_to_process = available_years
    else:
        print("Invalid mode. Aborting.")
        return

    country = Country.BRAZIL
    source_configs = get_configs_for_country(country)
    
    for year in years_to_process:
        base_year_str = str(year) if year else "avg_all_years"
        print(f"\n--- Processing for Region: {selected_region}, Base Year: {base_year_str} ---")

        real_pattern = pattern_loader.calculate_pattern_for_region(selected_region, base_year=year)
        
        sim_config = SimulationConfig(
            duration_days=12 * 30,
            country=country,
            random_seed=42,
            experiment_name=f"{country.value}_12m_{selected_region.lower()}_base_{base_year_str}",
            experiment_description=f"Contract prices for region: {selected_region}, based on year: {base_year_str}"
        )

        generator = ContractDataGenerator(source_configs, sim_config, real_pattern)
        data_df = generator.generate_contract_profile()

        output_dir = project_root / "data" / "synthetic"
        generator.save_data(data_df, output_dir)
        print(f"Data saved to: {output_dir / (sim_config.experiment_name + '.json')}")

        # --- PLOT 1: Synthetic Data (PNG) ---
        fig_synthetic_path = output_dir / f"{sim_config.experiment_name}_synthetic_prices.png"
        fig = plt.figure(figsize=(16, 8))
        for src_name, src_config in generator.sources.items():
            if f'price_{src_name}' in data_df.columns:
                plt.plot(data_df['day'], data_df[f'price_{src_name}'], label=f'{src_name.capitalize()} Price', linestyle='-')
        plt.ylabel('Price (BRL/MWh)')
        plt.xlabel('Day of Simulation (12 months)')
        plt.title(f'Synthetic Prices (Pattern: {selected_region} / Year: {base_year_str})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_synthetic_path, dpi=150)
        plt.close(fig)
        print(f"Synthetic prices plot saved to: {fig_synthetic_path}")

        # --- PLOT 2: Real Historical Data (PNG) ---
        if year: # Only plot real data if a specific year was chosen
            fig_real_path = output_dir / f"{sim_config.experiment_name}_real_pld.png"
            plot_real_pld(pattern_loader, selected_region, year, save_path=fig_real_path)
            print(f"Real historical PLD plot saved to: {fig_real_path}")


    print("\n" + "="*50)
    print(f"Contract price generation complete.")
    print("="*50 + "\n")

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

    print("\nStep 1: Calculating required power capacity to meet energy targets...")
    # --- Fase 2: Cálculo Inverso da Potência Instalada ---
    # Gera perfis de 1 kW para saber a produção de energia por kW instalado (fator de capacidade)
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

    # --- Fase 3: Geração do Perfil Principal ---
    sim_config.experiment_name = f"{sim_config.country.value}_{months}m_sol{int(required_solar_kw)}k_wind{int(required_wind_kw)}k_profile"
    project_root = Path(__file__).resolve().parent.parent

    print("\nStep 2: Generating price, generation, and consumption profiles...")
    # Gera o preço dinâmico da rede (PLD)
    historical_data_path = project_root / "data" / "real" / "Historico_do_Preco_Medio_Semanal_-_30_de_junho_de_2001_a_30_de_maio_de_2025.json"
    pattern_loader = HistoricalPatternLoader(historical_data_path)
    grid_pattern = pattern_loader.calculate_pattern_for_region("SOUTHEAST")
    contract_gen = ContractDataGenerator(get_configs_for_country(sim_config.country), sim_config, grid_pattern)
    price_df = contract_gen.generate_contract_profile()
    
    # Gera os perfis de consumo e geração no local com a potência calculada
    industrial_cfg = IndustrialConfig()
    generation_cfg = OnSiteGenerationConfig(solar_installed_kw=required_solar_kw, wind_installed_kw=required_wind_kw)
    profile_gen = EnergyProfileGenerator(sim_config, industrial_cfg, generation_cfg)
    profile_df = profile_gen.generate_profiles()

    # --- Fase 4: Ajuste do Consumo e Cálculo de Custos ---
    print("\nStep 3: Adjusting consumption and calculating detailed costs...")
    # Escala o consumo para corresponder exatamente à meta do usuário
    current_total_consumption_kwh = (profile_df['industrial_consumption_kw'] * time_step_h).sum()
    scaling_factor = (total_consumption_target_mwh * 1000) / current_total_consumption_kwh
    profile_df['industrial_consumption_kw'] *= scaling_factor

    # Adiciona variação mensal ao consumo
    for month_num in range(1, months + 1):
        monthly_variation = 1 + np.random.uniform(-0.05, 0.05)
        profile_df.loc[profile_df['timestamp'].dt.month == month_num, 'industrial_consumption_kw'] *= monthly_variation
    
    # Merge e lógica de despacho
    full_df = pd.merge(profile_df, price_df[['day', 'price_grid']], left_on='day_of_year', right_on='day', how='left').ffill()
    full_df.rename(columns={'price_grid': 'grid_spot_price_brl_per_mwh'}, inplace=True)
    full_df.drop(columns=['day', 'day_of_year'], inplace=True)
    
    full_df['solar_used_kw'] = np.minimum(full_df['solar_generation_kw'], full_df['industrial_consumption_kw'])
    remaining_demand = full_df['industrial_consumption_kw'] - full_df['solar_used_kw']
    full_df['wind_used_kw'] = np.minimum(full_df['wind_generation_kw'], remaining_demand)
    remaining_demand -= full_df['wind_used_kw']
    full_df['grid_used_kw'] = remaining_demand.clip(lower=0)

    # Lógica de Custo Detalhada
    full_df['cost_solar'] = full_df['solar_used_kw'] * (generation_cfg.solar_lcoe_brl_per_mwh / 1000) * time_step_h
    full_df['cost_wind'] = full_df['wind_used_kw'] * (generation_cfg.wind_lcoe_brl_per_mwh / 1000) * time_step_h
    
    grid_contract_used = np.minimum(full_df['grid_used_kw'], industrial_cfg.grid_contract_volume_kw)
    grid_spot_used = (full_df['grid_used_kw'] - industrial_cfg.grid_contract_volume_kw).clip(lower=0)
    
    full_df['cost_grid_contract'] = grid_contract_used * (industrial_cfg.grid_contract_price_brl_per_mwh / 1000) * time_step_h
    full_df['cost_grid_spot'] = grid_spot_used * (full_df['grid_spot_price_brl_per_mwh'] / 1000) * time_step_h

    # --- Fase 5: Salvamento e Plotagem ---
    print("\nStep 4: Saving data and generating plots...")
    output_dir = project_root / "data" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{sim_config.experiment_name}.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"\nFull energy profile with costs saved to: {csv_path}")

    # Plot 1: Perfil detalhado
    plot1_path = output_dir / f"{sim_config.experiment_name}_weekly_detail.png"
    title = f"Energy Profile Detail (First Week)\nTarget: {total_consumption_target_mwh} MWh, Solar: {solar_share_percentage:.0%}, Wind: {wind_share_percentage:.0%}"
    plot_energy_profiles(full_df.head(96 * 7), title, save_path=plot1_path)
    print(f"Didactic weekly plot saved to: {plot1_path}")

    # Plot 2: Resumo de consumo em kWh
    plot2_path = output_dir / f"{sim_config.experiment_name}_monthly_consumption.png"
    plot_monthly_consumption_summary(full_df, sim_config, save_path=plot2_path)
    print(f"Monthly consumption plot saved to: {plot2_path}")
    
    # Plot 3: Resumo de CUSTOS em BRL
    plot3_path = output_dir / f"{sim_config.experiment_name}_monthly_costs.png"
    plot_monthly_cost_summary(full_df, sim_config, save_path=plot3_path)
    print(f"Monthly cost analysis plot saved to: {plot3_path}")

    print("\n" + "="*50)
    print("Simulation complete.")
    print("="*50 + "\n")

def run_rl_simulation():
    """Runs an RL simulation using a generated contract data file."""
    print("\n--- Mode: Reinforcement Learning Simulation ---")
    project_root = Path(__file__).parent.parent.resolve()
    json_dir = project_root / "data" / "synthetic"
    
    if not json_dir.exists():
        print(f"Directory not found: {json_dir}")
        return

    json_files = sorted([f for f in json_dir.glob("*.json") if f.is_file()])
    if not json_files:
        print("No .json files found in 'data/synthetic'. Generate contract data first (Option 1).")
        return
    else:
        print("Available contract files:")
        for i, f in enumerate(json_files):
            print(f"  {i+1}. {f.name}")
        
        try:
            choice = int(input(f"Choose the file by number: ").strip())
            json_path = json_files[choice - 1]
        except (ValueError, IndexError):
            print("Invalid selection. Aborting.")
            return

    try:
        env = ContractEnergyEnv(str(json_path))
        print(f"\nRunning random policy in environment with sources: {list(env.sources.keys())}")
        save_path = json_path.with_suffix('.rl_result.json')
        env.run_random_policy(save_path=str(save_path))
        print(f"RL simulation complete. Results saved to: {save_path}")

        view = input("Do you want to plot the RL simulation results now? (y/n): ").strip().lower()
        if view == 'y':
            pdf_path = save_path.with_suffix('.pdf')
            plot_rl_results(str(save_path), save_path=str(pdf_path))
            print(f"Plot of results saved to: {pdf_path}")
    except Exception as e:
        print(f"An error occurred during the RL simulation: {e}")
        traceback.print_exc()

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
        print("  3. Run RL Simulation (on contract data)")
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
            run_rl_simulation()
        elif choice == '0':
            print("Exiting.")
            break
        else:
            print("Invalid option. Please enter 1, 2, 3 or 0.")

if __name__ == "__main__":
    main()