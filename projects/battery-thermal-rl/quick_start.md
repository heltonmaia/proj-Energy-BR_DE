# Quick Start Guide

Get started with Battery Thermal RL in minutes!

## Installation

```bash
# Navigate to project directory
cd projects/battery-thermal-rl

# Install dependencies
pip install -r requirements.txt
# OR using make
make install

# Test installation
python cli.py info status
# OR using make
make test
```

## Quick Demo

```bash
# Run system demonstration
make demo

# OR individual commands
python cli.py info status
python cli.py climate generate --region southeast_sp --days 7 --output data/demo_climate.csv
python cli.py battery simulate --type li_ion_500kwh --temperature 25 --hours 4
```

## CLI Usage Examples

### 1. Generate Climate Data
```bash
# Generate 30 days of climate data for São Paulo
python cli.py climate generate --region southeast_sp --days 30 --output data/sp_climate.csv

# Generate climate data for Rio Grande do Norte (excellent solar potential)
python cli.py climate generate --region northeast_rn --days 30 --output data/rn_climate.csv

# List available regions
python cli.py climate list

# Show climate statistics
python cli.py climate stats --region northeast_rn --days 30
```

### 2. Battery Simulation
```bash
# Simulate battery charging at 25°C for 4 hours
python cli.py battery simulate --type li_ion_500kwh --temperature 25 --action charge --hours 4

# Compare different battery types across temperature range
python cli.py battery compare --temperature-range 10 45 --output battery_comparison.csv

# List available battery types
python cli.py battery list
```

### 3. Industrial System
```bash
# Simulate industrial system for 7 days
python cli.py industrial simulate --profile medium_metallurgy --days 7 --output industrial_sim.csv

# Run economic analysis
python cli.py industrial economics --profile medium_metallurgy --days 30
```

### 4. Reinforcement Learning
```bash
# Train RL agent (quick training)
python cli.py rl train --algorithm ppo --steps 10000 --output models/ppo_agent

# Evaluate trained agent
python cli.py rl evaluate --model models/ppo_agent.zip --episodes 5

# Test environment with random actions
python cli.py rl test --steps 100 --random-actions
```

### 5. Analysis & Visualization
```bash
# Generate analysis report
python cli.py analysis report --config config.json --output report.html

# Create plots (if matplotlib installed)
python cli.py analysis plot --type climate --data data/sp_climate.csv --output climate_plot.png

# Run optimization analysis
python cli.py analysis optimize --battery-type li_ion_500kwh --days 30
```

## Using Make Commands

For convenience, use the provided Makefile:

```bash
# Show all available commands
make help

# Quick setup and demo
make dev-quick

# Generate all climate data
make climate-all

# Train and evaluate agent
make train
make evaluate

# Generate analysis report
make report

# Clean output files
make clean
```

## Configuration

Edit `config.json` to customize system parameters:

```json
{
  "battery_type": "li_ion_500kwh",
  "industrial_profile": "medium_metallurgy",
  "climate_region": "southeast_sp",
  "simulation_days": 30,
  "reward_weights": {
    "cost_reduction": 1.0,
    "battery_health": 0.3,
    "grid_stability": 0.2,
    "temperature_penalty": 0.5
  }
}
```

## Next Steps

1. **Explore different configurations**: Try different battery types, industrial profiles, and climate regions
2. **Train longer**: Use `make train-full` for better RL performance
3. **Analyze results**: Generate detailed reports and visualizations
4. **Customize**: Modify reward weights and system parameters to fit your use case

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
- **RL training issues**: Install PyTorch if using GPU: `pip install torch`
- **Plotting issues**: Make sure matplotlib is installed and working
- **Permission errors**: Check file permissions in output directories

## Getting Help

```bash
# General help
python cli.py --help

# Help for specific commands
python cli.py climate --help
python cli.py battery --help
python cli.py rl --help

# System information
python cli.py info status
python cli.py info list
```

