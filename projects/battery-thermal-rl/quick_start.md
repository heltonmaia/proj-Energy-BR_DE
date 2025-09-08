# Quick Start Guide

Get started with Battery Thermal RL in minutes!

## 🎯 Two Ways to Use the System

**📋 Interactive Menu Mode (Recommended for beginners)**
- Start with: `python cli.py` or `python cli.py menu`
- Navigate with numbers and get guided assistance
- Perfect for exploration and learning

**⚡ Command-Line Mode (Advanced users)**  
- Direct commands: `python cli.py climate generate --region southeast_sp`
- Batch operations: `python cli.py preset dev-quick`
- Perfect for automation and scripting

## Installation

```bash
# Navigate to project directory
cd projects/battery-thermal-rl

# Install dependencies
pip install -r requirements.txt
# OR using preset workflow
python cli.py preset install

# Test installation
python cli.py info status
# OR using preset workflow
python cli.py preset test
```

## Quick Demo

### 📋 Interactive Menu Way
```bash
# Start interactive menu
python cli.py

# Navigate to: 6. ⚡ Preset Workflows → 1. 🚀 Quick start demo
```

### ⚡ Command-Line Way
```bash
# Run system demonstration
python cli.py preset demo

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
python cli.py analysis report --output report.html

# Create plots (if matplotlib installed)
python cli.py analysis plot --type climate --data data/sp_climate.csv --output climate_plot.png

# Run optimization analysis
python cli.py analysis optimize --battery-type li_ion_500kwh --days 30
```

## Using Preset Commands

For convenience, use the built-in preset workflows (replaces Makefile):

```bash
# Show all available preset workflows
python cli.py preset

# Quick setup and demo
python cli.py preset dev-quick

# Generate all climate data
python cli.py preset climate-all

# Train and evaluate agent
python cli.py preset train
python cli.py preset evaluate

# Generate analysis report
python cli.py preset report

# Clean output files
python cli.py preset clean

# Complete development workflows
python cli.py preset dev-setup     # Setup: install + test + data
python cli.py preset dev-full      # Full development cycle
python cli.py preset benchmark     # Multi-configuration benchmark
```

## Output Organization

The system automatically organizes all results in the `outputs/` directory:

- `outputs/reports/` - HTML analysis reports
- `outputs/plots/` - Generated visualizations (PNG files)
- `outputs/simulations/` - Simulation results (CSV files)
- `outputs/evaluations/` - RL agent evaluation results (JSON files)

## Configuration

All parameters are configured via CLI arguments. For example:

```bash
# Customize battery type and region
python cli.py rl train --battery-type li_ion_500kwh --climate-region northeast_rn --steps 50000

# Customize industrial profile and analysis period
python cli.py industrial simulate --profile medium_food --climate-region south_rs --days 90

# All options available via --help
python cli.py rl train --help
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

