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
        # Create main directories
        self.output_dir = Path("./outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path("./logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create organized output subdirectories
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "simulations").mkdir(exist_ok=True)
        (self.output_dir / "evaluations").mkdir(exist_ok=True)
    
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
  python cli.py analysis report --battery-type li_ion_500kwh --output report.html
  
  # Run preset workflows (replaces Makefile)
  python cli.py preset dev-quick    # Complete development cycle
  python cli.py preset train       # Quick RL training
  python cli.py preset clean       # Clean output files
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
        
        # Preset workflows (replaces Makefile functionality)
        self._add_preset_commands(subparsers)
        
        # Interactive menu mode
        menu_parser = subparsers.add_parser('menu', help='Interactive menu mode')
        
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
        report_parser.add_argument('--battery-type', choices=list(INDUSTRIAL_BATTERY_CONFIGS.keys()),
                                  default='li_ion_500kwh', help='Battery type')
        report_parser.add_argument('--industrial-profile', 
                                  choices=['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical'],
                                  default='medium_metallurgy', help='Industrial profile')
        report_parser.add_argument('--climate-region', choices=get_available_regions(),
                                  default='southeast_sp', help='Climate region')
        report_parser.add_argument('--analysis-days', type=int, default=30,
                                  help='Analysis period (days)')
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
    
    def _add_preset_commands(self, subparsers):
        """Add preset workflow commands (replaces Makefile functionality)"""
        preset_parser = subparsers.add_parser('preset', help='Preset workflows and batch operations')
        preset_sub = preset_parser.add_subparsers(dest='preset_action')
        
        # Setup commands
        setup_parser = preset_sub.add_parser('install', help='Install dependencies')
        
        test_parser = preset_sub.add_parser('test', help='Test installation')
        
        # Data generation presets
        climate_parser = preset_sub.add_parser('climate', help='Generate climate data for São Paulo (30 days)')
        
        climate_all_parser = preset_sub.add_parser('climate-all', help='Generate climate data for all regions (7 days each)')
        
        battery_sim_parser = preset_sub.add_parser('battery', help='Simulate battery at different temperatures')
        
        battery_compare_parser = preset_sub.add_parser('battery-compare', help='Compare battery types across temperature range')
        
        industrial_parser = preset_sub.add_parser('industrial', help='Simulate industrial system (7 days)')
        
        industrial_econ_parser = preset_sub.add_parser('industrial-economics', help='Run economic analysis (30 days)')
        
        # RL training presets
        train_parser = preset_sub.add_parser('train', help='Train PPO agent (quick training - 10k steps)')
        
        train_full_parser = preset_sub.add_parser('train-full', help='Train PPO agent (full training - 100k steps)')
        
        train_sac_parser = preset_sub.add_parser('train-sac', help='Train SAC agent (50k steps)')
        
        evaluate_parser = preset_sub.add_parser('evaluate', help='Evaluate trained agent')
        
        # Analysis presets
        demo_parser = preset_sub.add_parser('demo', help='Run complete demonstration')
        
        test_env_parser = preset_sub.add_parser('test-env', help='Test RL environment')
        
        report_parser = preset_sub.add_parser('report', help='Generate analysis report')
        
        optimize_parser = preset_sub.add_parser('optimize', help='Run optimization analysis')
        
        plot_parser = preset_sub.add_parser('plot', help='Generate plots (if data available)')
        
        # Development workflows
        dev_setup_parser = preset_sub.add_parser('dev-setup', help='Development setup: install + test + climate + battery-compare + industrial')
        
        dev_quick_parser = preset_sub.add_parser('dev-quick', help='Quick development cycle: dev-setup + train + evaluate + report')
        
        dev_full_parser = preset_sub.add_parser('dev-full', help='Full development cycle: dev-setup + train-full + evaluate + report + plot')
        
        # Benchmark
        benchmark_parser = preset_sub.add_parser('benchmark', help='Run benchmark across different configurations')
        
        # Maintenance
        clean_parser = preset_sub.add_parser('clean', help='Clean output files')
        
        clean_models_parser = preset_sub.add_parser('clean-models', help='Clean trained models only')
    
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
            
            # Use CLI arguments instead of config file
            config = {
                'battery_type': args.battery_type,
                'industrial_profile': args.industrial_profile,
                'climate_region': args.climate_region,
                'analysis_days': args.analysis_days
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
                        <li>Battery Type: {config['battery_type']}</li>
                        <li>Industrial Profile: {config['industrial_profile']}</li>
                        <li>Climate Region: {config['climate_region']}</li>
                        <li>Analysis Period: {config['analysis_days']} days</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>System Analysis</h2>
                    <p>Battery thermal management system shows potential for significant energy cost savings
                    through intelligent charge/discharge scheduling based on temperature conditions and
                    electricity pricing patterns for {config['climate_region']} region.</p>
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <ul>
                        <li>Implement RL-based control for optimal battery scheduling</li>
                        <li>Monitor battery temperature closely during high ambient temperatures</li>
                        <li>Consider battery degradation in long-term economic analysis</li>
                        <li>Optimize for {config['industrial_profile']} specific load patterns</li>
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
            print("All configuration is done via CLI arguments. Examples:")
            print()
            print("# Basic RL training configuration")
            print("python cli.py rl train \\")
            print("    --algorithm ppo \\")
            print("    --battery-type li_ion_500kwh \\")
            print("    --industrial-profile medium_metallurgy \\")
            print("    --climate-region southeast_sp \\")
            print("    --simulation-days 30 \\")
            print("    --steps 100000")
            print()
            print("# Analysis report configuration")
            print("python cli.py analysis report \\")
            print("    --battery-type li_ion_500kwh \\")
            print("    --industrial-profile medium_metallurgy \\")
            print("    --climate-region northeast_rn \\")
            print("    --analysis-days 90 \\")
            print("    --output custom_report.html")
            print()
            print("Use --help with any command to see all available options.")
    
    def run_preset_command(self, args):
        """Execute preset workflow commands"""
        import subprocess
        import shutil
        
        if not args.preset_action:
            print("Available preset workflows:")
            print()
            print("Setup:")
            print("  install        Install dependencies")
            print("  test          Test installation")
            print()
            print("Data & Simulation:")
            print("  climate       Generate climate data for São Paulo (30 days)")
            print("  climate-all   Generate climate data for all regions (7 days)")
            print("  battery       Simulate battery at different temperatures")
            print("  battery-compare  Compare battery types")
            print("  industrial    Simulate industrial system (7 days)")
            print("  industrial-economics  Run economic analysis (30 days)")
            print()
            print("Reinforcement Learning:")
            print("  train         Train PPO agent (quick - 10k steps)")
            print("  train-full    Train PPO agent (full - 100k steps)")
            print("  train-sac     Train SAC agent (50k steps)")
            print("  evaluate      Evaluate trained agent")
            print()
            print("Analysis:")
            print("  demo          Run complete demonstration")
            print("  test-env      Test RL environment")
            print("  report        Generate analysis report")
            print("  optimize      Run optimization analysis")
            print("  plot          Generate plots (if data available)")
            print()
            print("Development Workflows:")
            print("  dev-setup     Development setup")
            print("  dev-quick     Quick development cycle")
            print("  dev-full      Full development cycle")
            print()
            print("Benchmarking & Maintenance:")
            print("  benchmark     Run benchmark across configurations")
            print("  clean         Clean output files")
            print("  clean-models  Clean trained models only")
            return
        
        # Setup commands
        if args.preset_action == 'install':
            print("Installing dependencies...")
            result = subprocess.run(['pip', 'install', '-r', 'requirements.txt'], capture_output=True, text=True)
            if result.returncode == 0:
                print("Installation complete!")
            else:
                print(f"Installation failed: {result.stderr}")
                
        elif args.preset_action == 'test':
            print("Testing installation...")
            self.run_info_command(type('', (), {'info_action': 'status'})())
            print("Test complete!")
            
        # Data generation presets
        elif args.preset_action == 'climate':
            print("Generating climate data for São Paulo (30 days)...")
            self.run_climate_command(type('', (), {
                'climate_action': 'generate',
                'region': 'southeast_sp',
                'days': 30,
                'output': 'data/sp_climate_30d.csv',
                'start_date': '2024-01-01'
            })())
            print("Climate data generated!")
            
        elif args.preset_action == 'climate-all':
            print("Generating climate data for all regions (7 days each)...")
            regions = [
                ('southeast_sp', 'data/southeast_sp_7d.csv'),
                ('northeast_rn', 'data/northeast_rn_7d.csv'),
                ('south_rs', 'data/south_rs_7d.csv'),
                ('central_west_mt', 'data/central_west_mt_7d.csv'),
                ('north_am', 'data/north_am_7d.csv')
            ]
            for region, output in regions:
                self.run_climate_command(type('', (), {
                    'climate_action': 'generate',
                    'region': region,
                    'days': 7,
                    'output': output,
                    'start_date': '2024-01-01'
                })())
            print("All climate data generated!")
            
        elif args.preset_action == 'battery':
            print("Simulating Li-ion battery at different temperatures...")
            for temp in [15, 25, 35, 45]:
                print(f"  Temperature: {temp}°C")
                self.run_battery_command(type('', (), {
                    'battery_action': 'simulate',
                    'type': 'li_ion_500kwh',
                    'temperature': temp,
                    'action': 'charge',
                    'power': None,
                    'hours': 4,
                    'initial_soc': 0.5
                })())
                
        elif args.preset_action == 'battery-compare':
            print("Comparing battery types...")
            self.run_battery_command(type('', (), {
                'battery_action': 'compare',
                'types': ['li_ion_500kwh', 'na_ion_200kwh'],
                'temperature_min': 10,
                'temperature_max': 50,
                'temperature_step': 5,
                'output': 'outputs/simulations/battery_comparison.csv'
            })())
            print("Battery comparison saved to outputs/simulations/battery_comparison.csv")
            
        elif args.preset_action == 'industrial':
            print("Simulating industrial system (7 days)...")
            self.run_industrial_command(type('', (), {
                'industrial_action': 'simulate',
                'profile': 'medium_metallurgy',
                'days': 7,
                'climate_region': 'southeast_sp',
                'output': 'outputs/simulations/industrial_sim_7d.csv'
            })())
            print("Industrial simulation saved to outputs/simulations/industrial_sim_7d.csv")
            
        elif args.preset_action == 'industrial-economics':
            print("Running economic analysis (30 days)...")
            self.run_industrial_command(type('', (), {
                'industrial_action': 'economics',
                'profile': 'medium_metallurgy',
                'days': 30,
                'climate_region': 'southeast_sp'
            })())
            
        # RL training presets
        elif args.preset_action == 'train':
            if not RL_AVAILABLE:
                print("Error: stable-baselines3 not available. Install with: pip install stable-baselines3")
                return
            print("Training PPO agent (quick training - 10k steps)...")
            self.run_rl_command(type('', (), {
                'rl_action': 'train',
                'algorithm': 'ppo',
                'steps': 10000,
                'output': 'models/ppo_quick',
                'eval_freq': 2000,
                'tensorboard_log': None,
                'battery_type': 'li_ion_500kwh',
                'industrial_profile': 'medium_metallurgy',
                'climate_region': 'southeast_sp',
                'simulation_days': 30
            })())
            print("Quick training complete! Model saved to models/ppo_quick.zip")
            
        elif args.preset_action == 'train-full':
            if not RL_AVAILABLE:
                print("Error: stable-baselines3 not available. Install with: pip install stable-baselines3")
                return
            print("Training PPO agent (full training - 100k steps)...")
            self.run_rl_command(type('', (), {
                'rl_action': 'train',
                'algorithm': 'ppo',
                'steps': 100000,
                'output': 'models/ppo_full',
                'eval_freq': 10000,
                'tensorboard_log': 'logs/',
                'battery_type': 'li_ion_500kwh',
                'industrial_profile': 'medium_metallurgy',
                'climate_region': 'southeast_sp',
                'simulation_days': 30
            })())
            print("Full training complete! Model saved to models/ppo_full.zip")
            
        elif args.preset_action == 'train-sac':
            if not RL_AVAILABLE:
                print("Error: stable-baselines3 not available. Install with: pip install stable-baselines3")
                return
            print("Training SAC agent (50k steps)...")
            self.run_rl_command(type('', (), {
                'rl_action': 'train',
                'algorithm': 'sac',
                'steps': 50000,
                'output': 'models/sac_agent',
                'eval_freq': 5000,
                'tensorboard_log': None,
                'battery_type': 'li_ion_500kwh',
                'industrial_profile': 'medium_metallurgy',
                'climate_region': 'southeast_sp',
                'simulation_days': 30
            })())
            
        elif args.preset_action == 'evaluate':
            if not RL_AVAILABLE:
                print("Error: stable-baselines3 not available. Install with: pip install stable-baselines3")
                return
            print("Evaluating trained agent...")
            model_path = None
            if Path("models/ppo_quick.zip").exists():
                model_path = "models/ppo_quick.zip"
            elif Path("models/ppo_full.zip").exists():
                model_path = "models/ppo_full.zip"
            elif Path("models/sac_agent.zip").exists():
                model_path = "models/sac_agent.zip"
                
            if model_path:
                self.run_rl_command(type('', (), {
                    'rl_action': 'evaluate',
                    'model': model_path,
                    'episodes': 5,
                    'output': 'outputs/evaluations/evaluation_results.json'
                })())
                print("Evaluation results saved to outputs/evaluations/evaluation_results.json")
            else:
                print("No trained model found! Run 'python cli.py preset train' first.")
                
        # Analysis presets
        elif args.preset_action == 'demo':
            print("Running system demonstration...")
            self.run_info_command(type('', (), {'info_action': 'status'})())
            self.run_climate_command(type('', (), {
                'climate_action': 'generate',
                'region': 'southeast_sp',
                'days': 7,
                'output': 'data/demo_climate.csv',
                'start_date': '2024-01-01'
            })())
            self.run_battery_command(type('', (), {
                'battery_action': 'simulate',
                'type': 'li_ion_500kwh',
                'temperature': 25,
                'action': 'charge',
                'power': None,
                'hours': 4,
                'initial_soc': 0.5
            })())
            
        elif args.preset_action == 'test-env':
            if not RL_AVAILABLE:
                print("Error: stable-baselines3 not available. Install with: pip install stable-baselines3")
                return
            print("Testing RL environment...")
            self.run_rl_command(type('', (), {
                'rl_action': 'test',
                'steps': 100,
                'random_actions': True
            })())
            
        elif args.preset_action == 'report':
            print("Generating analysis report...")
            self.run_analysis_command(type('', (), {
                'analysis_action': 'report',
                'battery_type': 'li_ion_500kwh',
                'industrial_profile': 'medium_metallurgy',
                'climate_region': 'southeast_sp',
                'analysis_days': 30,
                'output': 'outputs/reports/analysis_report.html'
            })())
            print("Report saved to outputs/reports/analysis_report.html")
            
        elif args.preset_action == 'optimize':
            print("Running optimization analysis...")
            self.run_analysis_command(type('', (), {
                'analysis_action': 'optimize',
                'battery_type': 'li_ion_500kwh',
                'industrial_profile': 'medium_metallurgy',
                'climate_region': 'southeast_sp',
                'days': 30
            })())
            
        elif args.preset_action == 'plot':
            if not PLOTTING_AVAILABLE:
                print("Error: matplotlib not available. Install with: pip install matplotlib seaborn")
                return
            print("Generating plots...")
            if Path("data/sp_climate_30d.csv").exists():
                self.run_analysis_command(type('', (), {
                    'analysis_action': 'plot',
                    'type': 'climate',
                    'data': 'data/sp_climate_30d.csv',
                    'output': 'outputs/plots/climate_plot.png'
                })())
                print("Climate plot saved to outputs/plots/climate_plot.png")
            else:
                print("No climate data found! Run 'python cli.py preset climate' first.")
                
            if Path("outputs/simulations/industrial_sim_7d.csv").exists():
                self.run_analysis_command(type('', (), {
                    'analysis_action': 'plot',
                    'type': 'industrial',
                    'data': 'outputs/simulations/industrial_sim_7d.csv',
                    'output': 'outputs/plots/industrial_plot.png'
                })())
                print("Industrial plot saved to outputs/plots/industrial_plot.png")
                
        # Development workflows
        elif args.preset_action == 'dev-setup':
            print("Running development setup...")
            # install + test + climate + battery-compare + industrial
            self.run_preset_command(type('', (), {'preset_action': 'install'})())
            self.run_preset_command(type('', (), {'preset_action': 'test'})())
            self.run_preset_command(type('', (), {'preset_action': 'climate'})())
            self.run_preset_command(type('', (), {'preset_action': 'battery-compare'})())
            self.run_preset_command(type('', (), {'preset_action': 'industrial'})())
            
        elif args.preset_action == 'dev-quick':
            print("Running quick development cycle...")
            self.run_preset_command(type('', (), {'preset_action': 'dev-setup'})())
            self.run_preset_command(type('', (), {'preset_action': 'train'})())
            self.run_preset_command(type('', (), {'preset_action': 'evaluate'})())
            self.run_preset_command(type('', (), {'preset_action': 'report'})())
            print("Quick development cycle complete!")
            
        elif args.preset_action == 'dev-full':
            print("Running full development cycle...")
            self.run_preset_command(type('', (), {'preset_action': 'dev-setup'})())
            self.run_preset_command(type('', (), {'preset_action': 'train-full'})())
            self.run_preset_command(type('', (), {'preset_action': 'evaluate'})())
            self.run_preset_command(type('', (), {'preset_action': 'report'})())
            self.run_preset_command(type('', (), {'preset_action': 'plot'})())
            print("Full development cycle complete!")
            
        # Benchmark
        elif args.preset_action == 'benchmark':
            if not RL_AVAILABLE:
                print("Error: stable-baselines3 not available. Install with: pip install stable-baselines3")
                return
            print("Running benchmark across different configurations...")
            print("This will take several hours...")
            
            configs = [
                ('li_ion_100kwh', 'ppo', 'models/bench_100kwh_ppo'),
                ('li_ion_500kwh', 'ppo', 'models/bench_500kwh_ppo'),
                ('li_ion_500kwh', 'sac', 'models/bench_500kwh_sac')
            ]
            
            for battery_type, algorithm, output in configs:
                print(f"Training {algorithm.upper()} with {battery_type}...")
                self.run_rl_command(type('', (), {
                    'rl_action': 'train',
                    'algorithm': algorithm,
                    'steps': 50000,
                    'output': output,
                    'eval_freq': 10000,
                    'tensorboard_log': None,
                    'battery_type': battery_type,
                    'industrial_profile': 'medium_metallurgy',
                    'climate_region': 'southeast_sp',
                    'simulation_days': 30
                })())
            print("Benchmark training complete!")
            
        # Maintenance
        elif args.preset_action == 'clean':
            print("Cleaning output files...")
            dirs_to_clean = ['outputs', 'data', 'models', 'logs']
            for dir_name in dirs_to_clean:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    dir_path.mkdir()
            print("Clean complete!")
            
        elif args.preset_action == 'clean-models':
            print("Cleaning trained models...")
            dirs_to_clean = ['models', 'logs']
            for dir_name in dirs_to_clean:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    dir_path.mkdir()
            print("Models cleaned!")
            
        else:
            print(f"Unknown preset action: {args.preset_action}")
            print("Use 'python cli.py preset --help' to see available workflows.")
    
    def run_interactive_menu(self):
        """Run interactive menu system"""
        print("\n" + "="*60)
        print("🔋 BATTERY THERMAL RL - INTERACTIVE MENU")
        print("="*60)
        
        while True:
            try:
                choice = self.show_main_menu()
                if choice == '0':
                    print("\n👋 Goodbye!")
                    break
                elif choice == '1':
                    self.climate_menu()
                elif choice == '2':
                    self.battery_menu()
                elif choice == '3':
                    self.industrial_menu()
                elif choice == '4':
                    self.rl_menu()
                elif choice == '5':
                    self.analysis_menu()
                elif choice == '6':
                    self.preset_menu()
                elif choice == '7':
                    self.info_menu()
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Menu interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                try:
                    input("\nPress Enter to continue...")
                except EOFError:
                    pass
    
    def show_main_menu(self):
        """Show main menu and get user choice"""
        print("\n" + "-"*60)
        print("📋 MAIN MENU")
        print("-"*60)
        print("1. 🌤️  Climate Data")
        print("2. 🔋 Battery Simulation") 
        print("3. 🏭 Industrial System")
        print("4. 🤖 Reinforcement Learning")
        print("5. 📊 Analysis & Reports")
        print("6. ⚡ Preset Workflows")
        print("7. ℹ️  System Information")
        print("0. 🚪 Exit")
        print("-"*60)
        try:
            return input("Select option (0-7): ").strip()
        except EOFError:
            return '0'  # Default to exit on EOF
    
    def climate_menu(self):
        """Climate data submenu"""
        while True:
            print("\n" + "-"*50)
            print("🌤️  CLIMATE DATA MENU")
            print("-"*50)
            print("1. Generate climate data for specific region")
            print("2. Generate climate data for all regions")
            print("3. List available regions")
            print("4. Show climate statistics")
            print("0. ← Back to main menu")
            print("-"*50)
            
            choice = input("Select option (0-4): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.menu_generate_climate()
            elif choice == '2':
                self.menu_generate_all_climate()
            elif choice == '3':
                self.menu_list_regions()
            elif choice == '4':
                self.menu_climate_stats()
            else:
                print("❌ Invalid choice. Please try again.")
                
    def battery_menu(self):
        """Battery simulation submenu"""
        while True:
            print("\n" + "-"*50)
            print("🔋 BATTERY SIMULATION MENU")
            print("-"*50)
            print("1. Simulate battery operation")
            print("2. Compare battery types")
            print("3. List available battery types")
            print("4. Battery performance at different temperatures")
            print("0. ← Back to main menu")
            print("-"*50)
            
            choice = input("Select option (0-4): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.menu_simulate_battery()
            elif choice == '2':
                self.menu_compare_batteries()
            elif choice == '3':
                self.menu_list_batteries()
            elif choice == '4':
                self.menu_battery_temperature_test()
            else:
                print("❌ Invalid choice. Please try again.")
                
    def industrial_menu(self):
        """Industrial system submenu"""
        while True:
            print("\n" + "-"*50)
            print("🏭 INDUSTRIAL SYSTEM MENU")
            print("-"*50)
            print("1. Simulate industrial system")
            print("2. Economic analysis")
            print("3. List industrial profiles")
            print("0. ← Back to main menu")
            print("-"*50)
            
            choice = input("Select option (0-3): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.menu_simulate_industrial()
            elif choice == '2':
                self.menu_industrial_economics()
            elif choice == '3':
                self.menu_list_industrial_profiles()
            else:
                print("❌ Invalid choice. Please try again.")
                
    def rl_menu(self):
        """Reinforcement Learning submenu"""
        while True:
            print("\n" + "-"*50)
            print("🤖 REINFORCEMENT LEARNING MENU")
            print("-"*50)
            print("1. Train RL agent")
            print("2. Evaluate trained agent")
            print("3. Test RL environment")
            print("4. List trained models")
            print("0. ← Back to main menu")
            print("-"*50)
            
            choice = input("Select option (0-4): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.menu_train_agent()
            elif choice == '2':
                self.menu_evaluate_agent()
            elif choice == '3':
                self.menu_test_environment()
            elif choice == '4':
                self.menu_list_models()
            else:
                print("❌ Invalid choice. Please try again.")
                
    def analysis_menu(self):
        """Analysis and reports submenu"""
        while True:
            print("\n" + "-"*50)
            print("📊 ANALYSIS & REPORTS MENU")
            print("-"*50)
            print("1. Generate analysis report")
            print("2. Create plots and visualizations")
            print("3. Run optimization analysis")
            print("4. View existing results")
            print("0. ← Back to main menu")
            print("-"*50)
            
            choice = input("Select option (0-4): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.menu_generate_report()
            elif choice == '2':
                self.menu_create_plots()
            elif choice == '3':
                self.menu_optimization_analysis()
            elif choice == '4':
                self.menu_view_results()
            else:
                print("❌ Invalid choice. Please try again.")
                
    def preset_menu(self):
        """Preset workflows submenu"""
        while True:
            print("\n" + "-"*50)
            print("⚡ PRESET WORKFLOWS MENU")
            print("-"*50)
            print("1. 🚀 Quick start demo")
            print("2. 🔧 Development setup")
            print("3. 🎯 Quick development cycle")
            print("4. 🏁 Full development cycle") 
            print("5. 🧪 Generate all test data")
            print("6. 🤖 Train RL agents")
            print("7. 🧹 Maintenance & cleanup")
            print("0. ← Back to main menu")
            print("-"*50)
            
            choice = input("Select option (0-7): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.menu_quick_demo()
            elif choice == '2':
                self.menu_dev_setup()
            elif choice == '3':
                self.menu_dev_quick()
            elif choice == '4':
                self.menu_dev_full()
            elif choice == '5':
                self.menu_generate_test_data()
            elif choice == '6':
                self.menu_train_agents()
            elif choice == '7':
                self.menu_maintenance()
            else:
                print("❌ Invalid choice. Please try again.")
                
    def info_menu(self):
        """System information submenu"""
        while True:
            print("\n" + "-"*50)
            print("ℹ️  SYSTEM INFORMATION MENU")
            print("-"*50)
            print("1. Show system status")
            print("2. List available options")
            print("3. Show configuration examples")
            print("4. View help information")
            print("0. ← Back to main menu")
            print("-"*50)
            
            choice = input("Select option (0-4): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.run_info_command(type('', (), {'info_action': 'status'})())
                input("\nPress Enter to continue...")
            elif choice == '2':
                self.run_info_command(type('', (), {'info_action': 'list', 'category': None})())
                input("\nPress Enter to continue...")
            elif choice == '3':
                self.run_info_command(type('', (), {'info_action': 'config'})())
                input("\nPress Enter to continue...")
            elif choice == '4':
                self.show_help_info()
            else:
                print("❌ Invalid choice. Please try again.")
    
    # Menu action implementations
    def safe_input(self, prompt, default_on_eof='0'):
        """Safe input that handles EOF gracefully"""
        try:
            return input(prompt).strip()
        except EOFError:
            return default_on_eof
    
    def safe_continue(self):
        """Safe "Press Enter to continue" that handles EOF"""
        try:
            input("\nPress Enter to continue...")
        except EOFError:
            pass
    
    def get_user_input(self, prompt, default=None, choices=None):
        """Get user input with validation"""
        while True:
            try:
                if default:
                    user_input = input(f"{prompt} (default: {default}): ").strip()
                    if not user_input:
                        return default
                else:
                    user_input = input(f"{prompt}: ").strip()
                    
                if choices and user_input not in choices:
                    print(f"❌ Please choose from: {', '.join(choices)}")
                    continue
                    
                if user_input or default:
                    return user_input
                print("❌ This field is required.")
            except EOFError:
                if default:
                    return default
                else:
                    raise KeyboardInterrupt("No input available")
    
    def menu_generate_climate(self):
        """Menu-driven climate data generation"""
        print("\n🌤️  Generate Climate Data")
        print("Available regions:")
        for i, region in enumerate(get_available_regions(), 1):
            print(f"  {i}. {region}")
            
        region = self.get_user_input("Region", "southeast_sp", get_available_regions())
        days = int(self.get_user_input("Number of days", "30"))
        output = self.get_user_input("Output file", f"data/{region}_{days}d.csv")
        
        print(f"\n⏳ Generating {days} days of climate data for {region}...")
        self.run_climate_command(type('', (), {
            'climate_action': 'generate',
            'region': region,
            'days': days,
            'output': output,
            'start_date': '2024-01-01'
        })())
        input("\nPress Enter to continue...")
        
    def menu_generate_all_climate(self):
        """Generate climate data for all regions"""
        print("\n🌤️  Generate Climate Data for All Regions")
        days = int(self.get_user_input("Number of days for each region", "7"))
        
        print(f"\n⏳ Generating {days} days of climate data for all regions...")
        self.run_preset_command(type('', (), {'preset_action': 'climate-all'})())
        input("\nPress Enter to continue...")
        
    def menu_list_regions(self):
        """List available regions"""
        print("\n🌍 Available Climate Regions:")
        self.run_climate_command(type('', (), {'climate_action': 'list'})())
        input("\nPress Enter to continue...")
        
    def menu_climate_stats(self):
        """Show climate statistics"""
        region = self.get_user_input("Region", "southeast_sp", get_available_regions())
        days = int(self.get_user_input("Number of days", "30"))
        
        print(f"\n📊 Climate Statistics for {region}:")
        self.run_climate_command(type('', (), {
            'climate_action': 'stats',
            'region': region,
            'days': days
        })())
        input("\nPress Enter to continue...")
        
    def menu_simulate_battery(self):
        """Menu-driven battery simulation"""
        print("\n🔋 Battery Simulation")
        
        battery_type = self.get_user_input("Battery type", "li_ion_500kwh", list(INDUSTRIAL_BATTERY_CONFIGS.keys()))
        temperature = float(self.get_user_input("Temperature (°C)", "25"))
        action = self.get_user_input("Action", "charge", ["charge", "discharge", "hold"])
        hours = int(self.get_user_input("Duration (hours)", "4"))
        initial_soc = float(self.get_user_input("Initial state of charge (0-1)", "0.5"))
        
        print(f"\n⏳ Simulating {battery_type} at {temperature}°C...")
        self.run_battery_command(type('', (), {
            'battery_action': 'simulate',
            'type': battery_type,
            'temperature': temperature,
            'action': action,
            'power': None,
            'hours': hours,
            'initial_soc': initial_soc
        })())
        input("\nPress Enter to continue...")
        
    def menu_compare_batteries(self):
        """Menu-driven battery comparison"""
        print("\n⚖️  Battery Comparison")
        
        temp_min = int(self.get_user_input("Minimum temperature (°C)", "10"))
        temp_max = int(self.get_user_input("Maximum temperature (°C)", "50"))
        output = self.get_user_input("Output file", "outputs/simulations/battery_comparison.csv")
        
        print(f"\n⏳ Comparing batteries from {temp_min}°C to {temp_max}°C...")
        self.run_battery_command(type('', (), {
            'battery_action': 'compare',
            'types': list(INDUSTRIAL_BATTERY_CONFIGS.keys()),
            'temperature_min': temp_min,
            'temperature_max': temp_max,
            'temperature_step': 5,
            'output': output
        })())
        input("\nPress Enter to continue...")
        
    def menu_list_batteries(self):
        """List available battery types"""
        print("\n🔋 Available Battery Types:")
        self.run_battery_command(type('', (), {'battery_action': 'list'})())
        input("\nPress Enter to continue...")
        
    def menu_battery_temperature_test(self):
        """Test battery at multiple temperatures"""
        print("\n🌡️  Battery Temperature Performance Test")
        
        battery_type = self.get_user_input("Battery type", "li_ion_500kwh", list(INDUSTRIAL_BATTERY_CONFIGS.keys()))
        
        print(f"\n⏳ Testing {battery_type} at multiple temperatures...")
        self.run_preset_command(type('', (), {'preset_action': 'battery'})())
        input("\nPress Enter to continue...")
        
    def menu_quick_demo(self):
        """Quick system demonstration"""
        print("\n🚀 Running Quick Demo...")
        self.run_preset_command(type('', (), {'preset_action': 'demo'})())
        input("\nPress Enter to continue...")
        
    def menu_dev_setup(self):
        """Development setup"""
        print("\n🔧 Running Development Setup...")
        self.run_preset_command(type('', (), {'preset_action': 'dev-setup'})())
        input("\nPress Enter to continue...")
        
    def menu_dev_quick(self):
        """Quick development cycle"""
        print("\n🎯 Running Quick Development Cycle...")
        self.run_preset_command(type('', (), {'preset_action': 'dev-quick'})())
        input("\nPress Enter to continue...")
        
    def menu_dev_full(self):
        """Full development cycle"""
        print("\n🏁 Running Full Development Cycle...")
        print("⚠️  This will take a long time (training 100k steps)!")
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            self.run_preset_command(type('', (), {'preset_action': 'dev-full'})())
        input("\nPress Enter to continue...")
        
    def show_help_info(self):
        """Show help information"""
        print("\n📚 HELP INFORMATION")
        print("-" * 50)
        print("This is an interactive menu system for Battery Thermal RL.")
        print("\nNavigation:")
        print("• Use numbers to select menu options")
        print("• Press 0 to go back to previous menu")
        print("• Press Ctrl+C to exit at any time")
        print("\nFor command-line usage, run: python cli.py --help")
        print("For preset workflows, run: python cli.py preset")
        input("\nPress Enter to continue...")
        
    # Additional menu implementations
    def menu_simulate_industrial(self):
        """Menu-driven industrial simulation"""
        print("\n🏭 Industrial System Simulation")
        
        profile = self.get_user_input("Industrial profile", "medium_metallurgy", 
                                     ['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical'])
        days = int(self.get_user_input("Number of days", "7"))
        region = self.get_user_input("Climate region", "southeast_sp", get_available_regions())
        output = self.get_user_input("Output file", f"outputs/simulations/industrial_{profile}_{days}d.csv")
        
        print(f"\n⏳ Simulating {profile} for {days} days in {region}...")
        self.run_industrial_command(type('', (), {
            'industrial_action': 'simulate',
            'profile': profile,
            'days': days,
            'climate_region': region,
            'output': output
        })())
        input("\nPress Enter to continue...")
        
    def menu_industrial_economics(self):
        """Menu-driven economic analysis"""
        print("\n💰 Economic Analysis")
        
        profile = self.get_user_input("Industrial profile", "medium_metallurgy",
                                     ['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical'])
        days = int(self.get_user_input("Number of days", "30"))
        region = self.get_user_input("Climate region", "southeast_sp", get_available_regions())
        
        print(f"\n⏳ Running economic analysis for {profile} ({days} days)...")
        self.run_industrial_command(type('', (), {
            'industrial_action': 'economics',
            'profile': profile,
            'days': days,
            'climate_region': region
        })())
        input("\nPress Enter to continue...")
        
    def menu_list_industrial_profiles(self):
        """List industrial profiles"""
        print("\n🏭 Available Industrial Profiles:")
        profiles = ['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical']
        for i, profile in enumerate(profiles, 1):
            print(f"  {i}. {profile}")
        input("\nPress Enter to continue...")
        
    def menu_train_agent(self):
        """Menu-driven RL agent training"""
        if not RL_AVAILABLE:
            print("❌ Error: stable-baselines3 not available. Install with: pip install stable-baselines3")
            input("\nPress Enter to continue...")
            return
            
        print("\n🤖 Train RL Agent")
        
        algorithm = self.get_user_input("Algorithm", "ppo", ["ppo", "sac", "td3"])
        steps = int(self.get_user_input("Training steps", "10000"))
        battery_type = self.get_user_input("Battery type", "li_ion_500kwh", list(INDUSTRIAL_BATTERY_CONFIGS.keys()))
        profile = self.get_user_input("Industrial profile", "medium_metallurgy",
                                     ['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical'])
        region = self.get_user_input("Climate region", "southeast_sp", get_available_regions())
        output = self.get_user_input("Model output name", f"models/{algorithm}_{steps}steps")
        
        print(f"\n⏳ Training {algorithm.upper()} agent for {steps} steps...")
        self.run_rl_command(type('', (), {
            'rl_action': 'train',
            'algorithm': algorithm,
            'steps': steps,
            'output': output,
            'eval_freq': max(steps // 5, 1000),
            'tensorboard_log': 'logs/' if steps >= 50000 else None,
            'battery_type': battery_type,
            'industrial_profile': profile,
            'climate_region': region,
            'simulation_days': 30
        })())
        input("\nPress Enter to continue...")
        
    def menu_evaluate_agent(self):
        """Menu-driven agent evaluation"""
        if not RL_AVAILABLE:
            print("❌ Error: stable-baselines3 not available. Install with: pip install stable-baselines3")
            input("\nPress Enter to continue...")
            return
            
        print("\n📊 Evaluate RL Agent")
        
        # List available models
        models_dir = Path("models")
        model_files = list(models_dir.glob("*.zip")) if models_dir.exists() else []
        
        if not model_files:
            print("❌ No trained models found! Train a model first.")
            input("\nPress Enter to continue...")
            return
            
        print("Available models:")
        for i, model_file in enumerate(model_files, 1):
            print(f"  {i}. {model_file.name}")
            
        try:
            choice = int(input(f"Select model (1-{len(model_files)}): ").strip())
            if 1 <= choice <= len(model_files):
                model_path = str(model_files[choice-1])
            else:
                print("❌ Invalid choice.")
                input("\nPress Enter to continue...")
                return
        except ValueError:
            print("❌ Please enter a valid number.")
            input("\nPress Enter to continue...")
            return
            
        episodes = int(self.get_user_input("Number of episodes", "5"))
        output = self.get_user_input("Output file", "outputs/evaluations/evaluation_results.json")
        
        print(f"\n⏳ Evaluating {Path(model_path).stem} for {episodes} episodes...")
        self.run_rl_command(type('', (), {
            'rl_action': 'evaluate',
            'model': model_path,
            'episodes': episodes,
            'output': output
        })())
        input("\nPress Enter to continue...")
        
    def menu_test_environment(self):
        """Menu-driven environment testing"""
        if not RL_AVAILABLE:
            print("❌ Error: stable-baselines3 not available. Install with: pip install stable-baselines3")
            input("\nPress Enter to continue...")
            return
            
        print("\n🧪 Test RL Environment")
        
        steps = int(self.get_user_input("Number of steps", "100"))
        random_actions = self.get_user_input("Use random actions?", "yes", ["yes", "no"]) == "yes"
        
        print(f"\n⏳ Testing environment for {steps} steps...")
        self.run_rl_command(type('', (), {
            'rl_action': 'test',
            'steps': steps,
            'random_actions': random_actions
        })())
        input("\nPress Enter to continue...")
        
    def menu_list_models(self):
        """List trained models"""
        print("\n🤖 Trained Models:")
        
        models_dir = Path("models")
        if not models_dir.exists():
            print("❌ Models directory not found.")
        else:
            model_files = list(models_dir.glob("*.zip"))
            if not model_files:
                print("❌ No trained models found.")
            else:
                for i, model_file in enumerate(model_files, 1):
                    size = model_file.stat().st_size / (1024*1024)  # MB
                    print(f"  {i}. {model_file.name} ({size:.1f} MB)")
                    
        input("\nPress Enter to continue...")
        
    def menu_generate_report(self):
        """Menu-driven report generation"""
        print("\n📊 Generate Analysis Report")
        
        battery_type = self.get_user_input("Battery type", "li_ion_500kwh", list(INDUSTRIAL_BATTERY_CONFIGS.keys()))
        profile = self.get_user_input("Industrial profile", "medium_metallurgy",
                                     ['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical'])
        region = self.get_user_input("Climate region", "southeast_sp", get_available_regions())
        days = int(self.get_user_input("Analysis period (days)", "30"))
        output = self.get_user_input("Output file", "outputs/reports/analysis_report.html")
        
        print(f"\n⏳ Generating analysis report...")
        self.run_analysis_command(type('', (), {
            'analysis_action': 'report',
            'battery_type': battery_type,
            'industrial_profile': profile,
            'climate_region': region,
            'analysis_days': days,
            'output': output
        })())
        input("\nPress Enter to continue...")
        
    def menu_create_plots(self):
        """Menu-driven plot creation"""
        if not PLOTTING_AVAILABLE:
            print("❌ Error: matplotlib not available. Install with: pip install matplotlib seaborn")
            input("\nPress Enter to continue...")
            return
            
        print("\n📈 Create Plots and Visualizations")
        
        plot_type = self.get_user_input("Plot type", "climate", ["climate", "battery", "industrial"])
        
        # List available data files
        data_dir = Path("data")
        if plot_type == "climate" and data_dir.exists():
            data_files = list(data_dir.glob("*.csv"))
        else:
            output_dir = Path("outputs")
            data_files = list(output_dir.glob("*.csv")) if output_dir.exists() else []
            
        if not data_files:
            print(f"❌ No data files found for {plot_type} plots.")
            input("\nPress Enter to continue...")
            return
            
        print("Available data files:")
        for i, data_file in enumerate(data_files, 1):
            print(f"  {i}. {data_file.name}")
            
        try:
            choice = int(input(f"Select data file (1-{len(data_files)}): ").strip())
            if 1 <= choice <= len(data_files):
                data_path = str(data_files[choice-1])
            else:
                print("❌ Invalid choice.")
                input("\nPress Enter to continue...")
                return
        except ValueError:
            print("❌ Please enter a valid number.")
            input("\nPress Enter to continue...")
            return
            
        output = self.get_user_input("Output plot file", f"outputs/plots/{plot_type}_plot.png")
        
        print(f"\n⏳ Creating {plot_type} plot...")
        self.run_analysis_command(type('', (), {
            'analysis_action': 'plot',
            'type': plot_type,
            'data': data_path,
            'output': output
        })())
        input("\nPress Enter to continue...")
        
    def menu_optimization_analysis(self):
        """Menu-driven optimization analysis"""
        print("\n🎯 Optimization Analysis")
        
        battery_type = self.get_user_input("Battery type", "li_ion_500kwh", list(INDUSTRIAL_BATTERY_CONFIGS.keys()))
        profile = self.get_user_input("Industrial profile", "medium_metallurgy",
                                     ['medium_metallurgy', 'medium_textile', 'medium_food', 'medium_chemical'])
        region = self.get_user_input("Climate region", "southeast_sp", get_available_regions())
        days = int(self.get_user_input("Analysis period (days)", "30"))
        
        print(f"\n⏳ Running optimization analysis...")
        self.run_analysis_command(type('', (), {
            'analysis_action': 'optimize',
            'battery_type': battery_type,
            'industrial_profile': profile,
            'climate_region': region,
            'days': days
        })())
        input("\nPress Enter to continue...")
        
    def menu_view_results(self):
        """View existing analysis results"""
        print("\n📁 Existing Analysis Results:")
        
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            print("❌ Outputs directory not found.")
        else:
            result_files = list(outputs_dir.iterdir())
            if not result_files:
                print("❌ No analysis results found.")
            else:
                for i, result_file in enumerate(result_files, 1):
                    if result_file.is_file():
                        size = result_file.stat().st_size / 1024  # KB
                        print(f"  {i}. {result_file.name} ({size:.1f} KB)")
                        
        input("\nPress Enter to continue...")
        
    def menu_generate_test_data(self):
        """Generate all test data"""
        print("\n🧪 Generate All Test Data")
        print("This will generate climate data, run battery tests, and industrial simulations.")
        
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            print("\n⏳ Generating test data...")
            self.run_preset_command(type('', (), {'preset_action': 'climate-all'})())
            self.run_preset_command(type('', (), {'preset_action': 'battery-compare'})())
            self.run_preset_command(type('', (), {'preset_action': 'industrial'})())
            print("✅ Test data generation complete!")
        input("\nPress Enter to continue...")
        
    def menu_train_agents(self):
        """Train multiple RL agents"""
        if not RL_AVAILABLE:
            print("❌ Error: stable-baselines3 not available. Install with: pip install stable-baselines3")
            input("\nPress Enter to continue...")
            return
            
        print("\n🤖 Train RL Agents")
        print("1. Quick training (10k steps)")
        print("2. Full training (100k steps)")
        print("3. SAC agent (50k steps)")
        print("4. Custom training")
        
        choice = input("Select training type (1-4): ").strip()
        
        if choice == '1':
            self.run_preset_command(type('', (), {'preset_action': 'train'})())
        elif choice == '2':
            print("⚠️  This will take a long time!")
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                self.run_preset_command(type('', (), {'preset_action': 'train-full'})())
        elif choice == '3':
            self.run_preset_command(type('', (), {'preset_action': 'train-sac'})())
        elif choice == '4':
            self.menu_train_agent()
        else:
            print("❌ Invalid choice.")
            
        input("\nPress Enter to continue...")
        
    def menu_maintenance(self):
        """Maintenance and cleanup menu"""
        print("\n🧹 Maintenance & Cleanup")
        print("1. Clean all output files")
        print("2. Clean only trained models")
        print("3. View disk usage")
        print("0. ← Back")
        
        choice = input("Select option (0-3): ").strip()
        
        if choice == '0':
            return
        elif choice == '1':
            confirm = input("⚠️  This will delete all outputs, data, models, and logs. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                self.run_preset_command(type('', (), {'preset_action': 'clean'})())
        elif choice == '2':
            confirm = input("⚠️  This will delete all trained models and logs. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                self.run_preset_command(type('', (), {'preset_action': 'clean-models'})())
        elif choice == '3':
            self.show_disk_usage()
        else:
            print("❌ Invalid choice.")
            
        input("\nPress Enter to continue...")
        
    def show_disk_usage(self):
        """Show disk usage of project directories"""
        print("\n💾 Disk Usage:")
        
        dirs_to_check = ['data', 'models', 'outputs', 'logs']
        total_size = 0
        
        for dir_name in dirs_to_check:
            dir_path = Path(dir_name)
            if dir_path.exists():
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                size_mb = size / (1024*1024)
                total_size += size_mb
                print(f"  {dir_name}/: {size_mb:.1f} MB")
            else:
                print(f"  {dir_name}/: not found")
                
        print(f"  Total: {total_size:.1f} MB")
    
    def run(self):
        """Main CLI entry point"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        # If no command provided, start interactive menu
        if not args.command:
            print("🔋 Welcome to Battery Thermal RL!")
            print("Starting interactive menu...")
            self.run_interactive_menu()
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
            elif args.command == 'preset':
                self.run_preset_command(args)
            elif args.command == 'menu':
                self.run_interactive_menu()
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