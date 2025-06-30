# src/app_cli.py

import argparse
from core.energy_profile_config import CountryProfileManager, Region, SimulationConfig, Country
from core.synthetic_data_generator import SyntheticDataGenerator
import pandas as pd
from utils.plot import plot_energy_profiles, plot_dual_energy_figures
import json
from pathlib import Path
import os
import numpy as np

def run_simulation():
    print("\nAvailable Country Energy Profiles:")
    print("="*50)
    country_list = [Country.BRAZIL, Country.GERMANY]
    for idx, country in enumerate(country_list, 1):
        print(f"  {idx}. {country.value.capitalize()}")

    # --- Select country ---
    while True:
        try:
            country_choice = int(input("\nSelect a country by number: "))
            if 1 <= country_choice <= len(country_list):
                country = country_list[country_choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(country_list)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # --- Duration in months ---
    default_months = 1
    months_input = input(f"Enter experiment duration in months [default: {default_months}]: ")
    try:
        months = int(months_input) if months_input.strip() else default_months
    except ValueError:
        print("Invalid input. Using default duration.")
        months = default_months
    duration_days = months * 30

    # --- Get sources and ask for consumption and shares ---
    sources, _ = CountryProfileManager.get_profile(country)
    print("\nAvailable energy sources for this country:")
    for src in sources:
        print(f"  - {src}")
    while True:
        try:
            total_consumption_kwh = float(input("Informe o total de energia a ser consumida no mês (kWh): "))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    # Sorteio dos percentuais com viés proporcional ao share real
    print("\nAssigning source shares for your consumption (proportional to real mix)...")
    real_shares = np.array([sources[src].share for src in sources])
    random_weights = np.random.dirichlet(real_shares * 10)  # Viés forte para o real
    shares = {src: float(f"{w:.3f}") for src, w in zip(sources, random_weights)}
    print("Suggested shares (%):")
    for src, pct in shares.items():
        print(f"  {src.capitalize()}: {pct*100:.1f}%")
    confirm = input("Use these shares? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Enter the percentage of consumption for each source (must sum to 100%):")
        shares = {}
        total_share = 0.0
        for src in sources:
            while True:
                try:
                    pct = float(input(f"  {src.capitalize()} (%): ")) / 100.0
                    if 0 <= pct <= 1:
                        shares[src] = pct
                        total_share += pct
                        break
                    else:
                        print("Enter a value between 0 and 100.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        if abs(sum(shares.values()) - 1.0) > 0.01:
            print(f"Warning: The sum of shares is {sum(shares.values())*100:.1f}%. Adjusting to 100%. Shares will be normalized.")
            total = sum(shares.values())
            shares = {k: v/total for k, v in shares.items()}

    # --- Build industrial config ---
    from core.energy_profile_config import IndustrialConfig
    industrial_config = IndustrialConfig(total_consumption_kw=total_consumption_kwh, shares=shares)

    # --- Build simulation config ---
    from core.energy_profile_config import SimulationConfig
    sim_config = SimulationConfig(
        duration_days=duration_days,
        time_resolution_minutes=15,
        random_seed=42,
        country=country,
        experiment_name=f"{country.value}_{months}month",
        experiment_description=f"Simulation for {country.value} ({months} month(s))"
    )

    # --- Generate Data ---
    print(f"\n--- Generating data for experiment: {sim_config.experiment_name} ---")
    from core.synthetic_data_generator import SyntheticDataGenerator
    generator = SyntheticDataGenerator(sources, industrial_config, sim_config)
    data_df = generator.generate_complete_profile()

    # Save data
    output_dir = "data/synthetic"
    generator.save_data(data_df, output_dir)

    print("\n--- Generation Summary ---")
    print(data_df.head())
    print("\n" + "="*25)
    print(f"Total data points: {len(data_df)}")
    print("\nSimulation complete. Returning to main menu.\n")

def run_analysis():
    print("\n--- Analysis ---")
    # Always use the project root's data/synthetic folder
    project_root = Path(__file__).parent.parent.resolve()
    json_dir = project_root / "data" / "synthetic"
    if not json_dir.exists() or not json_dir.is_dir():
        print(f"Directory not found: {json_dir.resolve()}")
        return
    json_name = input("Enter the name of the JSON file (without .json): ").strip()
    json_path = json_dir / f"{json_name}.json"
    if not json_path.exists():
        print(f"File not found: {json_path}")
        json_files = sorted([f for f in json_dir.glob("*.json") if f.is_file() and not f.name.startswith('_') and f.stat().st_size > 0])
        if not json_files:
            print("No available JSON files in data/synthetic.")
        else:
            print("Available JSON files in data/synthetic:")
            for f in json_files:
                print("  -", f.stem)
        return
    with open(json_path, 'r') as f:
        result = json.load(f)
    df_plot = pd.DataFrame(result['data'])
    # Detect country from metadata
    country_val = result.get('metadata', {}).get('simulation_config', {}).get('country', 'brazil')
    from core.energy_profile_config import Country
    country = Country(country_val) if isinstance(country_val, str) else country_val
    save_pdf = input("Save plot as PDF in the same folder? (y/n): ").strip().lower()
    if save_pdf == 'y':
        pdf_path = json_path.with_suffix('.pdf')
        plot_dual_energy_figures(df_plot, country, save_path=pdf_path)
        print(f"Plot saved to {pdf_path}")
    else:
        plot_dual_energy_figures(df_plot, country)

def main():
    while True:
        print("\n" + "="*40)
        print("{:^40}".format("proj-BRA-GER CLI"))
        print("="*40)
        print("{:^40}".format("Main Menu"))
        print("-"*40)
        print("  1. Simulations")
        print("  2. Analyses")
        print("  0. Exit")
        print("-"*40)
        print("Type the number of the desired option and press Enter.")
        print("="*40)
        choice = input("Select an option: ").strip()
        if choice == '1':
            try:
                run_simulation()
            except Exception as e:
                print(f"Error during simulation: {e}")
        elif choice == '2':
            try:
                run_analysis()
            except Exception as e:
                print(f"Error during analysis: {e}")
        elif choice == '0':
            print("Exiting.")
            break
        else:
            print("Invalid option. Please enter 1, 2, or 0.")

if __name__ == "__main__":
    main()