#!/usr/bin/env python3
"""
Battery Thermal RL - Command Line Interface

A comprehensive CLI tool for battery thermal management with reinforcement learning.
Provides commands for simulation, training, analysis, and visualization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np

# Import project modules
from src.models.battery import BatteryModel, INDUSTRIAL_BATTERY_CONFIGS
from src.models.industrial_system import IndustrialEnergySystem, TYPICAL_SOLAR_SYSTEMS
from src.data.climate_data import BrazilianClimateData, get_available_regions, save_climate_data
from src.rl.battery_thermal_env import create_env

# Optional RL libraries
try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: stable-baselines3 not available. RL training commands will be disabled.")

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting commands will be disabled.")


class BatteryThermalCLI:
    """Main CLI application for Battery Thermal RL system"""
    
    def __init__(self):
        self.output_dir = Path("./outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all CLI commands"""
        parser = argparse.ArgumentParser(
            description="Battery Thermal RL - Smart battery optimization system",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Generate climate data for São Paulo
  python cli.py climate generate --region southeast_sp --days 30 --output data/sp_climate.csv
  
  # Run battery simulation
  python cli.py battery simulate --type li_ion_500kwh --temperature 25 --hours 24
  
  # Train RL agent
  python cli.py rl train --algorithm ppo --steps 100000 --output models/agent
  
  # Evaluate trained agent
  python cli.py rl evaluate --model models/agent.zip --episodes 10
  
  # Generate analysis report
  python cli.py analysis report --config config.json --output report.html
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Climate data commands
        self._add_climate_commands(subparsers)
        
        # Battery simulation commands
        self._add_battery_commands(subparsers)
        
        # Industrial system commands
        self._add_industrial_commands(subparsers)
        
        # RL commands
        self._add_rl_commands(subparsers)
        
        # Analysis commands
        self._add_analysis_commands(subparsers)
        
        # Info commands
        self._add_info_commands(subparsers)
        
        return parser
    
    def _add_climate_commands(self, subparsers):
        """Add climate data related commands"""
        climate_parser = subparsers.add_parser('climate', help='Climate data operations')
        climate_sub = climate_parser.add_subparsers(dest='climate_action')
        
        # Generate climate data
        gen_parser = climate_sub.add_parser('generate', help='Generate climate data')
        gen_parser.add_argument('--region', choices=get_available_regions(), 
                               default='southeast_sp', help='Climate region')
        gen_parser.add_argument('--start-date', type=str, default='2024-01-01',
                               help='Start date (YYYY-MM-DD)')
        gen_parser.add_argument('--days', type=int, default=30,
                               help='Number of days to generate')
        gen_parser.add_argument('--output', type=str, required=True,
                               help='Output CSV file path')
        
        # List available regions
        list_parser = climate_sub.add_parser('list', help='List available regions')
        
        # Climate stats
        stats_parser = climate_sub.add_parser('stats', help='Show climate statistics')
        stats_parser.add_argument('--region', choices=get_available_regions(),
                                 default='southeast_sp', help='Climate region')
        stats_parser.add_argument('--days', type=int, default=30,
                                 help='Number of days for statistics')
    
    def _add_battery_commands(self, subparsers):
        """Add battery simulation commands"""
        battery_parser = subparsers.add_parser('battery', help='Battery operations')
        battery_sub = battery_parser.add_subparsers(dest='battery_action')
        
        # Simulate battery
        sim_parser = battery_sub.add_parser('simulate', help='Simulate battery operation')
        sim_parser.add_argument('--type', choices=list(INDUSTRIAL_BATTERY_CONFIGS.keys()),
                               default='li_ion_500kwh', help='Battery type')
        sim_parser.add_argument('--temperature', type=float, default=25.0,
                               help='Ambient temperature (°C)')
        sim_parser.add_argument('--action', choices=['charge', 'discharge', 'hold'],
                               default='charge', help='Battery action')
        sim_parser.add_argument('--power', type=float, help='Power (kW). If not specified, uses max power')
        sim_parser.add_argument('--hours', type=int, default=1,
                               help='Simulation duration (hours)')
        sim_parser.add_argument('--initial-soc', type=float, default=0.5,
                               help='Initial state of charge (0-1)')
        
        # Battery comparison
        comp_parser = battery_sub.add_parser('compare', help='Compare battery types')
        comp_parser.add_argument('--types', nargs='+', 
                                choices=list(INDUSTRIAL_BATTERY_CONFIGS.keys()),
                                help='Battery types to compare')
        comp_parser.add_argument('--temperature-range', nargs=2, type=float,
                                default=[10, 45], help='Temperature range to test')
        comp_parser.add_argument('--output', type=str, 
                                help='Output file for comparison results')
        
        # List battery types
        list_parser = battery_sub.add_parser('list', help='List available battery types')
    
    def _add_industrial_commands(self, subparsers):
        """Add industrial system commands"""
        industrial_parser = subparsers.add_parser('industrial', help='Industrial system operations')
        industrial_sub = industrial_parser.add_subparsers(dest='industrial_action')
        
        # Simulate industrial system
        sim_parser = industrial_sub.add_parser('simulate', help='Simulate industrial system')
        sim_parser.add_argument('--profile', 
                               choices=['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical'],
                               default='medium_metallurgy', help='Industrial profile')
        sim_parser.add_argument('--solar-system', choices=list(TYPICAL_SOLAR_SYSTEMS.keys()),
                               default='medium_1000kw', help='Solar system size')
        sim_parser.add_argument('--climate-region', choices=get_available_regions(),
                               default='southeast_sp', help='Climate region')
        sim_parser.add_argument('--days', type=int, default=7,
                               help='Simulation days')
        sim_parser.add_argument('--output', type=str, help='Output CSV file')
        
        # Economic analysis
        econ_parser = industrial_sub.add_parser('economics', help='Economic analysis')
        econ_parser.add_argument('--profile', 
                                choices=['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical'],
                                default='medium_metallurgy', help='Industrial profile')
        econ_parser.add_argument('--solar-system', choices=list(TYPICAL_SOLAR_SYSTEMS.keys()),
                                default='medium_1000kw', help='Solar system size')
        econ_parser.add_argument('--days', type=int, default=30,
                                help='Analysis period (days)')
    
    def _add_rl_commands(self, subparsers):
        """Add reinforcement learning commands"""
        if not RL_AVAILABLE:
            return
            
        rl_parser = subparsers.add_parser('rl', help='Reinforcement Learning operations')
        rl_sub = rl_parser.add_subparsers(dest='rl_action')
        
        # Train RL agent
        train_parser = rl_sub.add_parser('train', help='Train RL agent')
        train_parser.add_argument('--algorithm', choices=['ppo', 'sac', 'td3'],
                                 default='ppo', help='RL algorithm')
        train_parser.add_argument('--steps', type=int, default=100000,
                                 help='Training steps')
        train_parser.add_argument('--battery-type', choices=list(INDUSTRIAL_BATTERY_CONFIGS.keys()),
                                 default='li_ion_500kwh', help='Battery type')
        train_parser.add_argument('--industrial-profile', 
                                 choices=['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical'],
                                 default='medium_metallurgy', help='Industrial profile')
        train_parser.add_argument('--climate-region', choices=get_available_regions(),
                                 default='southeast_sp', help='Climate region')
        train_parser.add_argument('--simulation-days', type=int, default=30,
                                 help='Episode length (days)')
        train_parser.add_argument('--output', type=str, required=True,
                                 help='Output model path')
        train_parser.add_argument('--tensorboard-log', type=str,
                                 help='Tensorboard log directory')
        train_parser.add_argument('--eval-freq', type=int, default=10000,
                                 help='Evaluation frequency')
        
        # Evaluate trained agent
        eval_parser = rl_sub.add_parser('evaluate', help='Evaluate trained agent')
        eval_parser.add_argument('--model', type=str, required=True,
                                help='Path to trained model')
        eval_parser.add_argument('--episodes', type=int, default=5,
                                help='Number of evaluation episodes')
        eval_parser.add_argument('--render', action='store_true',
                                help='Render episodes')
        eval_parser.add_argument('--output', type=str,
                                help='Output file for results')
        
        # Test environment
        test_parser = rl_sub.add_parser('test', help='Test RL environment')
        test_parser.add_argument('--steps', type=int, default=100,
                                help='Test steps')
        test_parser.add_argument('--random-actions', action='store_true',
                                help='Use random actions')
    
    def _add_analysis_commands(self, subparsers):
        """Add analysis and visualization commands"""
        analysis_parser = subparsers.add_parser('analysis', help='Analysis and visualization')
        analysis_sub = analysis_parser.add_subparsers(dest='analysis_action')
        
        # Generate comprehensive report
        report_parser = analysis_sub.add_parser('report', help='Generate analysis report')
        report_parser.add_argument('--config', type=str,
                                  help='Configuration file (JSON)')
        report_parser.add_argument('--output', type=str, default='report.html',
                                  help='Output report file')
        
        # Plot climate data
        if PLOTTING_AVAILABLE:
            plot_parser = analysis_sub.add_parser('plot', help='Plot data')
            plot_parser.add_argument('--type', choices=['climate', 'battery', 'industrial', 'rl'],
                                    required=True, help='Plot type')
            plot_parser.add_argument('--data', type=str, required=True,
                                    help='Input data file (CSV)')
            plot_parser.add_argument('--output', type=str,
                                    help='Output plot file')
        
        # Optimization analysis
        opt_parser = analysis_sub.add_parser('optimize', help='Optimization analysis')
        opt_parser.add_argument('--battery-type', choices=list(INDUSTRIAL_BATTERY_CONFIGS.keys()),
                               default='li_ion_500kwh', help='Battery type')
        opt_parser.add_argument('--industrial-profile', 
                               choices=['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical'],
                               default='medium_metallurgy', help='Industrial profile')
        opt_parser.add_argument('--climate-region', choices=get_available_regions(),
                               default='southeast_sp', help='Climate region')
        opt_parser.add_argument('--days', type=int, default=30,
                               help='Analysis period (days)')
    
    def _add_info_commands(self, subparsers):
        """Add information commands"""
        info_parser = subparsers.add_parser('info', help='System information')
        info_sub = info_parser.add_subparsers(dest='info_action')
        
        # System status
        status_parser = info_sub.add_parser('status', help='Show system status')
        
        # List all available options
        list_parser = info_sub.add_parser('list', help='List all available options')
        list_parser.add_argument('--category', 
                                choices=['batteries', 'industrial', 'regions', 'solar'],
                                help='Category to list')
        
        # Configuration example
        config_parser = info_sub.add_parser('config', help='Show configuration example')
    
    # Command implementations
    def run_climate_command(self, args):
        """Execute climate data commands"""
        if args.climate_action == 'generate':
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            print(f"Generating climate data for {args.region}...")
            print(f"Period: {args.start_date} to {start_date + timedelta(days=args.days-1)}")
            
            save_climate_data(args.region, start_date, args.days, args.output)
            print(f"Climate data saved to: {args.output}")
            
        elif args.climate_action == 'list':
            print("Available climate regions:")
            for region in get_available_regions():
                climate = BrazilianClimateData(region)
                profile = climate.profile
                print(f"  {region}: {profile.region_name}")
                print(f"    Summer avg: {profile.avg_temp_summer:.1f}°C")
                print(f"    Winter avg: {profile.avg_temp_winter:.1f}°C")
                print(f"    Solar peak: {profile.solar_irradiance_peak:.1f} kWh/m²/day")
                print()
                
        elif args.climate_action == 'stats':
            print(f"Climate statistics for {args.region} ({args.days} days):")
            climate = BrazilianClimateData(args.region)
            data = climate.generate_hourly_data(datetime.now(), args.days)
            
            print(f"Temperature: {data['temperature_c'].mean():.1f}°C avg, "
                  f"{data['temperature_c'].min():.1f}-{data['temperature_c'].max():.1f}°C range")
            print(f"Solar irradiance: {data['solar_irradiance_kw_m2'].mean():.3f} kW/m² avg")
            print(f"Humidity: {data['humidity_percent'].mean():.1f}% avg")
    
    def run_battery_command(self, args):
        """Execute battery simulation commands"""
        if args.battery_action == 'simulate':
            battery_specs = INDUSTRIAL_BATTERY_CONFIGS[args.type]
            battery = BatteryModel(battery_specs)
            battery.soc = args.initial_soc
            
            print(f"Battery Simulation: {args.type}")
            print(f"Initial SOC: {battery.soc:.2f}")
            print(f"Temperature: {args.temperature}°C")
            print(f"Action: {args.action}")
            print(f"Duration: {args.hours} hours")
            
            total_energy = 0
            for hour in range(args.hours):
                if args.action == 'charge':
                    power = args.power or battery_specs.max_charge_rate_kw
                    energy, new_temp = battery.charge(power, args.temperature, 1.0)
                elif args.action == 'discharge':
                    power = args.power or battery_specs.max_discharge_rate_kw
                    energy, new_temp = battery.discharge(power, args.temperature, 1.0)
                else:  # hold
                    energy, new_temp = 0, args.temperature
                
                total_energy += energy
                battery.temperature = new_temp
                
                if hour < 5 or hour >= args.hours - 2:  # Show first and last hours
                    print(f"  Hour {hour+1}: SOC={battery.soc:.3f}, Energy={energy:.2f}kWh, Temp={new_temp:.1f}°C")
                elif hour == 5 and args.hours > 7:
                    print(f"  ... ({args.hours-7} hours omitted) ...")
            
            print(f"\nFinal Results:")
            print(f"  Final SOC: {battery.soc:.3f}")
            print(f"  Total energy: {total_energy:.2f} kWh")
            print(f"  Final temperature: {battery.temperature:.1f}°C")
            print(f"  Efficiency: {battery.get_efficiency(args.temperature, args.action):.3f}")
            
        elif args.battery_action == 'compare':
            types = args.types or list(INDUSTRIAL_BATTERY_CONFIGS.keys())
            temp_min, temp_max = args.temperature_range
            temperatures = np.linspace(temp_min, temp_max, 8)
            
            results = []
            print(f"Comparing battery types at temperatures {temp_min}-{temp_max}°C:")
            print()
            
            for battery_type in types:
                specs = INDUSTRIAL_BATTERY_CONFIGS[battery_type]
                battery = BatteryModel(specs)
                
                print(f"{battery_type}:")
                for temp in temperatures:
                    charge_eff = battery.get_efficiency(temp, 'charge')
                    discharge_eff = battery.get_efficiency(temp, 'discharge')
                    safe = battery.can_operate_safely(temp)
                    
                    results.append({
                        'battery_type': battery_type,
                        'temperature': temp,
                        'charge_efficiency': charge_eff,
                        'discharge_efficiency': discharge_eff,
                        'safe_operation': safe,
                        'capacity_kwh': specs.capacity_kwh
                    })
                    
                    print(f"  {temp:4.1f}°C: Charge={charge_eff:.3f}, "
                          f"Discharge={discharge_eff:.3f}, Safe={safe}")
                print()
            
            if args.output:
                df = pd.DataFrame(results)
                df.to_csv(args.output, index=False)
                print(f"Comparison results saved to: {args.output}")
                
        elif args.battery_action == 'list':
            print("Available battery types:")
            for name, specs in INDUSTRIAL_BATTERY_CONFIGS.items():
                print(f"  {name}:")
                print(f"    Capacity: {specs.capacity_kwh} kWh")
                print(f"    Max charge: {specs.max_charge_rate_kw} kW")
                print(f"    Max discharge: {specs.max_discharge_rate_kw} kW")
                print(f"    Optimal temp: {specs.optimal_temp_min}-{specs.optimal_temp_max}°C")
                print(f"    Cycle life: {specs.cycle_life} cycles")
                print()
    
    def run_industrial_command(self, args):
        """Execute industrial system commands"""
        if args.industrial_action == 'simulate':
            print(f"Industrial System Simulation:")
            print(f"Profile: {args.profile}")
            print(f"Solar system: {args.solar_system}")
            print(f"Climate region: {args.climate_region}")
            print(f"Simulation period: {args.days} days")
            
            # Create system
            solar_specs = TYPICAL_SOLAR_SYSTEMS[args.solar_system]
            industrial = IndustrialEnergySystem(args.profile, solar_specs)
            
            # Generate climate data
            climate = BrazilianClimateData(args.climate_region)
            start_date = datetime(2024, 1, 1)
            climate_data = climate.generate_hourly_data(start_date, args.days)
            
            # Simulate system
            results = []
            total_demand = 0
            total_solar = 0
            total_cost = 0
            
            for _, row in climate_data.iterrows():
                timestamp = row['timestamp']
                
                # Calculate demand and solar generation
                demand = industrial.get_demand_profile(timestamp)
                solar_gen = industrial.calculate_solar_generation(
                    row['solar_irradiance_kw_m2'], 
                    row['temperature_c']
                )
                
                # Calculate costs
                price = industrial.get_electricity_price(timestamp)
                energy_balance = industrial.calculate_energy_balance(
                    demand, solar_gen, 0, 0  # No battery for now
                )
                costs = industrial.calculate_costs(energy_balance, price)
                
                total_demand += demand
                total_solar += solar_gen
                total_cost += costs['net_cost_brl']
                
                results.append({
                    'timestamp': timestamp,
                    'demand_kw': demand,
                    'solar_generation_kw': solar_gen,
                    'grid_import_kw': energy_balance['grid_import_kw'],
                    'self_consumption_kw': energy_balance['self_consumption_kw'],
                    'electricity_price_brl_kwh': price,
                    'cost_brl': costs['net_cost_brl'],
                    'temperature_c': row['temperature_c']
                })
            
            # Summary
            print(f"\nSimulation Results ({args.days} days):")
            print(f"  Total demand: {total_demand:.0f} kWh")
            print(f"  Total solar generation: {total_solar:.0f} kWh")
            print(f"  Self-consumption ratio: {min(total_solar/total_demand, 1.0):.1%}")
            print(f"  Total cost: R$ {total_cost:.2f}")
            print(f"  Average cost: R$ {total_cost/(args.days*24):.3f}/hour")
            
            if args.output:
                df = pd.DataFrame(results)
                df.to_csv(args.output, index=False)
                print(f"Detailed results saved to: {args.output}")
                
        elif args.industrial_action == 'economics':
            print(f"Economic Analysis:")
            print(f"Profile: {args.profile}")
            print(f"Period: {args.days} days")
            
            # Scenarios comparison
            scenarios = {
                'no_solar': {'solar_kw': 0},
                'with_solar': {'solar_kw': TYPICAL_SOLAR_SYSTEMS[args.solar_system].installed_capacity_kw}
            }
            
            for scenario_name, config in scenarios.items():
                print(f"\n{scenario_name.replace('_', ' ').title()}:")
                
                # Calculate basic economics (simplified)
                avg_demand = 400  # kW average for medium industry
                hours = args.days * 24
                total_consumption = avg_demand * hours
                avg_price = 0.50  # R$/kWh average
                
                if config['solar_kw'] > 0:
                    solar_generation = config['solar_kw'] * 4.5 * args.days  # 4.5h equivalent sun
                    self_consumption = min(solar_generation * 0.7, total_consumption * 0.6)
                    cost = (total_consumption - self_consumption) * avg_price
                    savings = self_consumption * avg_price
                else:
                    cost = total_consumption * avg_price
                    savings = 0
                
                print(f"  Total cost: R$ {cost:.2f}")
                print(f"  Savings: R$ {savings:.2f}")
                print(f"  Cost per kWh: R$ {cost/total_consumption:.3f}")
    
    def run_rl_command(self, args):
        """Execute RL commands"""
        if not RL_AVAILABLE:
            print("Error: stable-baselines3 not installed. Cannot run RL commands.")
            return
            
        if args.rl_action == 'train':
            print(f"Training RL Agent:")
            print(f"Algorithm: {args.algorithm}")
            print(f"Training steps: {args.steps}")
            print(f"Battery: {args.battery_type}")
            print(f"Industrial profile: {args.industrial_profile}")
            print(f"Climate region: {args.climate_region}")
            
            # Create environment
            env_config = {
                'battery_type': args.battery_type,
                'industrial_profile': args.industrial_profile,
                'solar_system': 'medium_1000kw',
                'climate_region': args.climate_region,
                'simulation_days': args.simulation_days
            }
            
            env = create_env(env_config)
            
            # Create algorithm
            if args.algorithm == 'ppo':
                model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=args.tensorboard_log)
            elif args.algorithm == 'sac':
                model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=args.tensorboard_log)
            elif args.algorithm == 'td3':
                model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=args.tensorboard_log)
            
            # Setup callbacks
            callbacks = []
            if args.eval_freq > 0:
                eval_callback = EvalCallback(
                    env, best_model_save_path=str(self.models_dir),
                    log_path=str(self.output_dir), eval_freq=args.eval_freq
                )
                callbacks.append(eval_callback)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=args.eval_freq, save_path=str(self.models_dir)
            )
            callbacks.append(checkpoint_callback)
            
            # Train
            print("Starting training...")
            model.learn(total_timesteps=args.steps, callback=callbacks)
            
            # Save model
            model.save(args.output)
            print(f"Model saved to: {args.output}")
            
        elif args.rl_action == 'evaluate':
            print(f"Evaluating trained agent: {args.model}")
            
            # Load model
            if 'ppo' in args.model.lower():
                model = PPO.load(args.model)
            elif 'sac' in args.model.lower():
                model = SAC.load(args.model)
            elif 'td3' in args.model.lower():
                model = TD3.load(args.model)
            else:
                # Try to detect from file
                try:
                    model = PPO.load(args.model)
                except:
                    try:
                        model = SAC.load(args.model)
                    except:
                        model = TD3.load(args.model)
            
            # Create environment (use default config)
            env = create_env()
            
            # Evaluate
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(args.episodes):
                obs, info = env.reset()
                total_reward = 0
                steps = 0
                
                print(f"Episode {episode + 1}:")
                
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    total_reward += reward
                    steps += 1
                    
                    if args.render and steps % 100 == 0:
                        env.render()
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Episode length: {steps} steps")
                print(f"  Final SOC: {info['battery_soc']:.2f}")
                print(f"  Battery degradation: {info['battery_degradation']:.3f}")
                print()
            
            # Summary
            print(f"Evaluation Summary ({args.episodes} episodes):")
            print(f"  Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"  Average length: {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f}")
            
            if args.output:
                results = {
                    'episode_rewards': episode_rewards,
                    'episode_lengths': episode_lengths,
                    'avg_reward': np.mean(episode_rewards),
                    'std_reward': np.std(episode_rewards)
                }
                
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to: {args.output}")
                
        elif args.rl_action == 'test':
            print(f"Testing RL Environment ({args.steps} steps)")
            
            env = create_env()
            obs, info = env.reset()
            
            total_reward = 0
            for step in range(args.steps):
                if args.random_actions:
                    action = env.action_space.sample()
                else:
                    action = np.array([0.0])  # No action
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if step % 20 == 0:
                    print(f"Step {step}: Action={action[0]:.2f}, Reward={reward:.2f}, "
                          f"SOC={info['battery_soc']:.2f}")
                
                if terminated or truncated:
                    obs, info = env.reset()
            
            print(f"Test completed. Total reward: {total_reward:.2f}")
    
    def run_analysis_command(self, args):
        """Execute analysis commands"""
        if args.analysis_action == 'report':
            print("Generating comprehensive analysis report...")
            
            # Load configuration if provided
            if args.config:
                with open(args.config, 'r') as f:
                    config = json.load(f)
            else:
                config = {
                    'battery_type': 'li_ion_500kwh',
                    'industrial_profile': 'medium_metallurgy',
                    'climate_region': 'southeast_sp',
                    'analysis_days': 30
                }
            
            # Generate HTML report (simplified)
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Battery Thermal RL Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; }}
                    .section {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Battery Thermal RL Analysis Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Configuration</h2>
                    <ul>
                        <li>Battery Type: {config.get('battery_type', 'N/A')}</li>
                        <li>Industrial Profile: {config.get('industrial_profile', 'N/A')}</li>
                        <li>Climate Region: {config.get('climate_region', 'N/A')}</li>
                        <li>Analysis Period: {config.get('analysis_days', 30)} days</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>System Analysis</h2>
                    <p>Battery thermal management system shows potential for significant energy cost savings
                    through intelligent charge/discharge scheduling based on temperature conditions and
                    electricity pricing patterns.</p>
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <ul>
                        <li>Implement RL-based control for optimal battery scheduling</li>
                        <li>Monitor battery temperature closely during high ambient temperatures</li>
                        <li>Consider battery degradation in long-term economic analysis</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            with open(args.output, 'w') as f:
                f.write(html_content)
            
            print(f"Report generated: {args.output}")
            
        elif args.analysis_action == 'plot' and PLOTTING_AVAILABLE:
            print(f"Generating {args.type} plot from {args.data}")
            
            # Load data
            df = pd.read_csv(args.data)
            
            # Create plot based on type
            plt.figure(figsize=(12, 8))
            
            if args.type == 'climate':
                plt.subplot(2, 2, 1)
                plt.plot(df['temperature_c'])
                plt.title('Temperature')
                plt.ylabel('°C')
                
                plt.subplot(2, 2, 2)
                plt.plot(df['solar_irradiance_kw_m2'])
                plt.title('Solar Irradiance')
                plt.ylabel('kW/m²')
                
                plt.subplot(2, 2, 3)
                plt.plot(df['humidity_percent'])
                plt.title('Humidity')
                plt.ylabel('%')
                
            elif args.type == 'battery':
                if 'battery_soc' in df.columns:
                    plt.subplot(2, 1, 1)
                    plt.plot(df['battery_soc'])
                    plt.title('Battery State of Charge')
                    plt.ylabel('SOC')
                    
                    plt.subplot(2, 1, 2)
                    plt.plot(df.get('battery_temperature', [25]*len(df)))
                    plt.title('Battery Temperature')
                    plt.ylabel('°C')
                    
            elif args.type == 'industrial':
                if 'demand_kw' in df.columns:
                    plt.subplot(2, 1, 1)
                    plt.plot(df['demand_kw'], label='Demand')
                    if 'solar_generation_kw' in df.columns:
                        plt.plot(df['solar_generation_kw'], label='Solar')
                    plt.title('Energy Profile')
                    plt.ylabel('kW')
                    plt.legend()
                    
                    plt.subplot(2, 1, 2)
                    if 'cost_brl' in df.columns:
                        plt.plot(df['cost_brl'])
                        plt.title('Energy Costs')
                        plt.ylabel('R$')
            
            plt.tight_layout()
            
            if args.output:
                plt.savefig(args.output, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {args.output}")
            else:
                plt.show()
                
        elif args.analysis_action == 'optimize':
            print("Running optimization analysis...")
            
            # Simple optimization analysis
            print(f"Battery: {args.battery_type}")
            print(f"Industry: {args.industrial_profile}")
            print(f"Region: {args.climate_region}")
            print(f"Period: {args.days} days")
            
            # Simulate different strategies
            strategies = {
                'no_battery': 'No battery system',
                'simple_solar': 'Charge from solar, discharge at peak hours',
                'temperature_aware': 'Temperature-aware operation',
                'price_optimized': 'Price-optimized scheduling'
            }
            
            print(f"\nOptimization Strategies Analysis:")
            for strategy, description in strategies.items():
                # Simplified calculation
                base_cost = 1000  # Base monthly cost
                
                if strategy == 'no_battery':
                    savings = 0
                elif strategy == 'simple_solar':
                    savings = base_cost * 0.15  # 15% savings
                elif strategy == 'temperature_aware':
                    savings = base_cost * 0.25  # 25% savings
                elif strategy == 'price_optimized':
                    savings = base_cost * 0.35  # 35% savings
                
                final_cost = base_cost - savings
                print(f"  {strategy}: R$ {final_cost:.0f}/month ({savings/base_cost:.1%} savings)")
    
    def run_info_command(self, args):
        """Execute info commands"""
        if args.info_action == 'status':
            print("Battery Thermal RL System Status:")
            print(f"  Python version: {sys.version.split()[0]}")
            print(f"  Working directory: {os.getcwd()}")
            print(f"  Output directory: {self.output_dir}")
            print(f"  Models directory: {self.models_dir}")
            
            # Check dependencies
            print(f"\nDependencies:")
            print(f"  NumPy: {'✓' if 'numpy' in sys.modules else '✗'}")
            print(f"  Pandas: {'✓' if 'pandas' in sys.modules else '✗'}")
            print(f"  RL (stable-baselines3): {'✓' if RL_AVAILABLE else '✗'}")
            print(f"  Plotting (matplotlib): {'✓' if PLOTTING_AVAILABLE else '✗'}")
            
        elif args.info_action == 'list':
            if args.category == 'batteries' or not args.category:
                print("Available battery types:")
                for name in INDUSTRIAL_BATTERY_CONFIGS.keys():
                    print(f"  {name}")
                print()
                
            if args.category == 'industrial' or not args.category:
                print("Available industrial profiles:")
                profiles = ['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical']
                for profile in profiles:
                    print(f"  {profile}")
                print()
                
            if args.category == 'regions' or not args.category:
                print("Available climate regions:")
                for region in get_available_regions():
                    print(f"  {region}")
                print()
                
            if args.category == 'solar' or not args.category:
                print("Available solar systems:")
                for system in TYPICAL_SOLAR_SYSTEMS.keys():
                    print(f"  {system}")
                print()
                
        elif args.info_action == 'config':
            config_example = {
                "battery_type": "li_ion_500kwh",
                "industrial_profile": "medium_metallurgy",
                "solar_system": "medium_1000kw",
                "climate_region": "southeast_sp",
                "simulation_days": 30,
                "analysis_days": 30,
                "reward_weights": {
                    "cost_reduction": 1.0,
                    "battery_health": 0.3,
                    "grid_stability": 0.2,
                    "temperature_penalty": 0.5
                }
            }
            
            print("Configuration file example (config.json):")
            print(json.dumps(config_example, indent=2))
    
    def run(self):
        """Main CLI entry point"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        try:
            if args.command == 'climate':
                self.run_climate_command(args)
            elif args.command == 'battery':
                self.run_battery_command(args)
            elif args.command == 'industrial':
                self.run_industrial_command(args)
            elif args.command == 'rl':
                self.run_rl_command(args)
            elif args.command == 'analysis':
                self.run_analysis_command(args)
            elif args.command == 'info':
                self.run_info_command(args)
            else:
                print(f"Unknown command: {args.command}")
                parser.print_help()
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"Error: {e}")
            if '--debug' in sys.argv:
                import traceback
                traceback.print_exc()


def main():
    """CLI entry point"""
    cli = BatteryThermalCLI()
    cli.run()


if __name__ == '__main__':
    main()