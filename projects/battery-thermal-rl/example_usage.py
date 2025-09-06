"""
Battery Thermal RL system usage example.

This script demonstrates how to use the main components:
1. Basic simulation of industrial + battery + solar system
2. RL environment testing
3. Brazilian climate data analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from src.models.battery import BatteryModel, INDUSTRIAL_BATTERY_CONFIGS
from src.models.industrial_system import IndustrialEnergySystem, TYPICAL_SOLAR_SYSTEMS
from src.data.climate_data import BrazilianClimateData, get_available_regions
from src.rl.battery_thermal_env import create_env


def demo_climate_data():
    """Demonstrates Brazilian climate data generation"""
    print("=== DEMO: Brazilian Climate Data ===")
    
    print("Available regions:", get_available_regions())
    
    # Generate data for São Paulo
    climate_sp = BrazilianClimateData('southeast_sp')
    data_sp = climate_sp.generate_hourly_data(datetime(2024, 6, 1), 7)  # 1 week
    
    print(f"\nGenerated data for São Paulo (7 days):")
    print(f"Average temperature: {data_sp['temperature_c'].mean():.1f}°C")
    print(f"Maximum temperature: {data_sp['temperature_c'].max():.1f}°C") 
    print(f"Minimum temperature: {data_sp['temperature_c'].min():.1f}°C")
    print(f"Average solar irradiance: {data_sp['solar_irradiance_kw_m2'].mean():.3f} kW/m²")
    
    return data_sp


def demo_battery_model():
    """Demonstrates battery model with thermal management"""
    print("\n=== DEMO: Battery Model with Thermal Management ===")
    
    # Create Li-ion 500kWh battery
    battery_specs = INDUSTRIAL_BATTERY_CONFIGS['li_ion_500kwh']
    battery = BatteryModel(battery_specs)
    
    print(f"Battery: {battery_specs.capacity_kwh} kWh")
    print(f"Initial SOC: {battery.soc:.2f}")
    print(f"Initial temperature: {battery.temperature:.1f}°C")
    
    # Charge test at different temperatures
    temperatures = [15, 25, 35, 45]  # °C
    
    print("\nEfficiency vs temperature test:")
    for temp in temperatures:
        charge_eff = battery.get_efficiency(temp, 'charge')
        discharge_eff = battery.get_efficiency(temp, 'discharge')
        safe = battery.can_operate_safely(temp)
        
        print(f"  {temp}°C: Charge={charge_eff:.2f}, Discharge={discharge_eff:.2f}, Safe={safe}")
    
    return battery


def demo_industrial_system():
    """Demonstrates industrial system with solar"""
    print("\n=== DEMO: Industrial System + Solar ===")
    
    # Create metallurgy industrial system with 1MW solar
    industrial_system = IndustrialEnergySystem(
        industrial_profile='medium_metallurgy',
        solar_specs=TYPICAL_SOLAR_SYSTEMS['medium_1000kw']
    )
    
    print(f"Industry: {industrial_system.profile.name}")
    print(f"Peak demand: {industrial_system.profile.peak_demand_kw} kW")
    print(f"Solar system: {industrial_system.solar_specs.installed_capacity_kw} kWp")
    
    # Simulate a typical day
    test_date = datetime(2024, 6, 15)  # Mid winter
    
    print(f"\nSimulation for {test_date.date()}:")
    
    daily_data = []
    for hour in range(24):
        timestamp = test_date.replace(hour=hour)
        
        # Demanda
        demand = industrial_system.get_demand_profile(timestamp)
        
        # Geração solar (simulada)
        temp = 20 + 10 * np.sin((hour - 6) * np.pi / 12)  # Curva de temperatura
        irradiance = max(0, 0.8 * np.sin((hour - 6) * np.pi / 12)) if 6 <= hour <= 18 else 0
        solar_gen = industrial_system.calculate_solar_generation(irradiance, temp)
        
        # Preço
        price = industrial_system.get_electricity_price(timestamp)
        
        daily_data.append({
            'hour': hour,
            'demand_kw': demand,
            'solar_kw': solar_gen,
            'price_brl_kwh': price,
            'temperature_c': temp
        })
    
    df_day = pd.DataFrame(daily_data)
    print(f"Average demand: {df_day['demand_kw'].mean():.1f} kW")
    print(f"Total solar generation: {df_day['solar_kw'].sum():.1f} kWh")
    print(f"Average price: R${df_day['price_brl_kwh'].mean():.3f}/kWh")
    
    return df_day, industrial_system


def demo_rl_environment():
    """Demonstrates RL environment"""
    print("\n=== DEMO: Reinforcement Learning Environment ===")
    
    # Environment configuration
    env_config = {
        'battery_type': 'li_ion_500kwh',
        'industrial_profile': 'medium_metallurgy',
        'solar_system': 'medium_1000kw',
        'climate_region': 'southeast_sp',
        'simulation_days': 3  # Short simulation for demo
    }
    
    # Create environment
    env = create_env(env_config)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Execute some steps
    obs, info = env.reset()
    print(f"\nInitial observation (normalized): {obs[:5]}...")  # First 5 elements
    print(f"Start date: {info['start_date']}")
    
    # Random actions
    print(f"\n--- Executing 24 steps (1 day) ---")
    total_reward = 0
    
    for step in range(24):
        # Random action
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 6 == 0:  # Show every 6 hours
            print(f"Hour {step:2d}: Action={action[0]:.2f}, "
                  f"Reward={reward:.2f}, SOC={info['battery_soc']:.2f}, "
                  f"Temp={info['battery_temperature']:.1f}°C")
        
        if terminated or truncated:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Final battery SOC: {info['battery_soc']:.2f}")
    
    env.close()
    return total_reward


def demo_optimization_potential():
    """Demonstra potencial de otimização"""
    print("\n=== DEMO: Potencial de Otimização ===")
    
    # Simular estratégias simples
    strategies = {
        'sem_bateria': {'description': 'Sem bateria (baseline)'},
        'carregar_dia': {'description': 'Carregar durante dia (solar)'},
        'descarregar_ponta': {'description': 'Descarregar no horário de ponta'},
        'otimo_temperatura': {'description': 'Operar apenas em temp. ótima'}
    }
    
    # Dados climáticos para análise
    climate = BrazilianClimateData('sudeste_sp')
    climate_data = climate.generate_hourly_data(datetime(2024, 1, 1), 30)  # 1 mês
    
    # Sistema industrial
    industrial = IndustrialEnergySystem('metalurgia_media', TYPICAL_SOLAR_SYSTEMS['medium_1000kw'])
    
    # Bateria
    battery_specs = INDUSTRIAL_BATTERY_CONFIGS['li_ion_500kwh']
    
    results = {}
    
    for strategy_name in strategies.keys():
        total_cost = 0
        battery_degradation = 0
        
        # Simular 30 dias
        for _, row in climate_data.iterrows():
            timestamp = row['timestamp']
            temp = row['temperature_c']
            irradiance = row['solar_irradiance_kw_m2']
            
            # Demanda e geração
            demand = industrial.get_demand_profile(timestamp)
            solar_gen = industrial.calculate_solar_generation(irradiance, temp)
            price = industrial.get_electricity_price(timestamp)
            
            if strategy_name == 'sem_bateria':
                # Sem bateria - comprar tudo da rede
                cost = max(0, demand - solar_gen) * price
            else:
                # Com bateria (simulação simplificada)
                cost = max(0, demand - solar_gen) * price * 0.7  # 30% economia estimada
                
                # Degradação por temperatura inadequada
                if temp > battery_specs.optimal_temp_max:
                    battery_degradation += 0.001  # Degradação estimada
            
            total_cost += cost
        
        results[strategy_name] = {
            'custo_total_r$': total_cost,
            'degradacao_estimada': battery_degradation,
            'economia_vs_baseline': 0 if strategy_name == 'sem_bateria' else total_cost - results.get('sem_bateria', {}).get('custo_total_r$', total_cost)
        }
    
    print("Comparação de estratégias (30 dias):")
    baseline_cost = None
    for strategy, result in results.items():
        cost = result['custo_total_r$']
        if strategy == 'sem_bateria':
            baseline_cost = cost
            print(f"  {strategies[strategy]['description']}: R${cost:.0f} (baseline)")
        else:
            savings = baseline_cost - cost if baseline_cost else 0
            print(f"  {strategies[strategy]['description']}: R${cost:.0f} (economia: R${savings:.0f})")
    
    return results


def main():
    """Execute all demonstrations"""
    print("Battery Thermal RL - Complete Demonstration")
    print("=" * 60)
    
    try:
        # 1. Climate data
        climate_data = demo_climate_data()
        
        # 2. Battery model
        battery = demo_battery_model()
        
        # 3. Industrial system
        daily_data, industrial_system = demo_industrial_system()
        
        # 4. RL environment
        rl_reward = demo_rl_environment()
        
        # 5. Optimization analysis
        optimization_results = demo_optimization_potential()
        
        print(f"\n{'=' * 60}")
        print("Demonstration completed successfully!")
        print("The system is ready for RL development and training.")
        
        # Summary
        print(f"\nResults Summary:")
        print(f"  • Climate data: {len(climate_data)} hourly records")
        print(f"  • Current battery SOC: {battery.soc:.1%}")
        print(f"  • RL reward (24h): {rl_reward:.1f}")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()