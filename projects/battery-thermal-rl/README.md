# Battery Thermal Management with Reinforcement Learning

**Intelligent battery optimization system for industrial applications** with solar panels, considering thermal management and Brazilian climate conditions.

## Objective

This project develops a **Reinforcement Learning** system to optimize battery usage in medium-sized industries with solar photovoltaic generation in Brazil, considering:

- **Intelligent thermal management** - Optimization based on ambient temperature
- **Real industrial profiles** - Modeling of different industry types
- **Solar integration** - Maximizing self-consumption and savings
- **Economic optimization** - Cost reduction with differentiated tariffs
- **Battery preservation** - Minimizing degradation and extending lifespan
- **Brazilian climate data** - Region-specific models

## System Architecture

```
Climate Data (Temperature, Solar) 
         ↓
Industrial System (Energy Demand)
         ↓
    RL Environment ← → RL Agent
         ↓
Battery with Thermal Model
         ↓
Optimization (Charge/Discharge)
```

## 📁 Project Structure

```
battery-thermal-rl/
├── src/                    # Source code
│   ├── models/             # System models
│   │   ├── battery.py      # Battery model with thermal management
│   │   └── industrial_system.py # Industrial system + solar
│   ├── data/               # Data generators
│   │   └── climate_data.py # Brazilian climate data
│   ├── rl/                 # Reinforcement Learning
│   │   └── battery_thermal_env.py # Gymnasium environment
│   ├── core/               # Core functionalities
│   └── utils/              # Utilities
├── data/                   # Generated climate data
├── models/                 # Trained RL models
├── outputs/                # All analysis results
│   ├── reports/            # HTML analysis reports
│   ├── plots/              # Generated visualizations
│   ├── simulations/        # Simulation results (CSV)
│   └── evaluations/        # RL evaluation results
├── logs/                   # Training logs (TensorBoard)
├── docs/                   # Mathematical documentation
│   ├── mathematical_modeling.md    # Complete mathematical tutorial
│   ├── mathematical_appendix.md    # Advanced theory and derivations
│   └── README.md           # Documentation guide
├── cli.py                  # Main CLI interface
├── quick_start.md          # Quick start guide  
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Implemented Features

### Battery Models
- **Li-ion and Na-ion** with real industrial specifications
- **Variable efficiency** with temperature
- **Degradation model** based on cycles and temperature
- **Thermal safety limits**

### Industrial System + Solar
- **Demand profiles** for different industry types (metallurgy, textile, food, chemical)
- **Photovoltaic system** with temperature correction
- **Brazilian tariff structure** with peak/off-peak hours
- **Complete economic calculations** (costs, savings, self-consumption)

### Brazilian Climate Data
- **5 regions** with distinct climate profiles optimized for battery thermal analysis:
  - **Southeast (SP)**: Moderate seasonal variation, balanced conditions for year-round operation
  - **Northeast (RN)**: High temperatures (29.5°C summer), exceptional solar potential (7.2 kWh/m²/day), ideal for testing thermal limits
  - **South (RS)**: Greatest thermal variation (13-25°C), challenging for battery thermal management
  - **Central-West (MT)**: Continental climate with high thermal amplitude (12°C daily variation)
  - **North (AM)**: Humid tropical, consistent temperatures, high humidity challenges
- **Hourly data** with temperature, solar irradiance and humidity patterns
- **Realistic seasonal** and stochastic variations based on historical data

### RL Environment (Gymnasium)
- **Observation space**: 15 features (battery state, climate, system, time)
- **Action space**: Continuous charge/discharge power [-1, +1]
- **Multi-objective reward function**:
  - Energy cost savings (weight: 1.0)
  - Battery health preservation (weight: 0.3)
  - Grid stability (weight: 0.2)
  - Temperature penalty (weight: 0.5)

## How to Use

### 1. Installation
```bash
cd projects/battery-thermal-rl
pip install -r requirements.txt

# Test installation
python cli.py info status
```

### 2. Interactive Menu Mode (Recommended for Beginners)
```bash
# Start interactive menu (default when no command provided)
python cli.py

# Or explicitly start menu mode
python cli.py menu
```

The interactive menu provides an intuitive interface with:
- 📋 **Main menu** with 7 categories
- 🔄 **Navigation** between submenus
- ✅ **Input validation** and defaults
- 📝 **Step-by-step guidance** for all operations
- 🎨 **Visual feedback** with emojis and progress indicators

### 3. Command-Line Mode (Advanced Users)
```bash
# Generate climate data for different regions
python cli.py climate generate --region southeast_sp --days 30 --output data/sp_climate.csv
python cli.py climate generate --region northeast_rn --days 30 --output data/rn_climate.csv

# Test battery performance under different conditions
python cli.py battery simulate --type li_ion_500kwh --temperature 25 --hours 4
python cli.py battery simulate --type li_ion_500kwh --temperature 35 --hours 4

# Train RL agent for specific regions and conditions
python cli.py rl train --algorithm ppo --climate-region northeast_rn --steps 10000 --output models/rn_agent
python cli.py rl train --algorithm ppo --climate-region southeast_sp --steps 10000 --output models/sp_agent
```

### 3. CLI Commands

The system includes a comprehensive CLI with the following commands:

- **Climate**: `python cli.py climate generate/list/stats`
- **Battery**: `python cli.py battery simulate/compare/list`  
- **Industrial**: `python cli.py industrial simulate/economics`
- **RL**: `python cli.py rl train/evaluate/test`
- **Analysis**: `python cli.py analysis report/plot/optimize`
- **Info**: `python cli.py info status/list/config`
- **Menu**: `python cli.py menu` (Interactive mode)

### 4. Using Preset Workflows (Replaces Makefile)
```bash
# Show all available preset workflows
python cli.py preset

# Quick development cycle (setup + train + evaluate + report)
python cli.py preset dev-quick

# Generate climate data for all Brazilian regions
python cli.py preset climate-all

# Compare battery types across temperature ranges
python cli.py preset battery-compare

# Train and evaluate RL agents
python cli.py preset train          # Quick training (10k steps)
python cli.py preset train-full     # Full training (100k steps) 
python cli.py preset evaluate       # Evaluate best model

# Generate comprehensive analysis
python cli.py preset report         # HTML report → outputs/reports/
python cli.py preset plot           # Generate visualizations → outputs/plots/

# Clean workspace
python cli.py preset clean          # Remove all outputs
python cli.py preset clean-models   # Remove only trained models

# Complete workflows
python cli.py preset dev-setup      # Setup: install + test + data generation
python cli.py preset dev-full       # Full cycle: setup + train-full + analysis
python cli.py preset benchmark      # Multi-configuration benchmark
```

### 5. Programmatic Usage

```python
# Create RL environment
from src.rl import create_env

env_config = {
    'battery_type': 'li_ion_500kwh',
    'industrial_profile': 'medium_metallurgy', 
    'solar_system': 'medium_1000kw',
    'climate_region': 'southeast_sp',
    'simulation_days': 30
}

env = create_env(env_config)

# Train RL agent
from stable_baselines3 import PPO

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate policy
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()
```

See [Quick Start Guide](quick_start.md) for detailed examples and [Mathematical Documentation](docs/README.md) for complete theoretical foundations.

## Expected Results

With RL optimization across different Brazilian regions:

### Cost Optimization
- **20-40% savings** in energy costs through intelligent scheduling
- **Peak hour consumption reduction** by 60% using battery discharge
- **Solar self-consumption increase** to 80-90% with optimal charge timing

### Battery Health Preservation
- **Battery lifespan extension** by 15-25% through thermal-aware operation
- **Reduced degradation** in high-temperature regions (RN: 29.5°C summer)
- **Temperature-optimized charging** during cooler periods

### Regional Performance Expectations
- **Northeast (RN)**: Best solar economics, thermal management critical
- **Southeast (SP)**: Balanced performance, optimal for baseline comparisons  
- **South (RS)**: Seasonal optimization crucial, best winter performance
- **Central-West (MT)**: Daily thermal cycle optimization opportunities
- **North (AM)**: Humidity considerations, consistent solar potential

## Use Cases & Applications

### Industrial Applications
```bash
# Metallurgy plant in Rio Grande do Norte (high solar, high temperatures)
python cli.py rl train --industrial-profile medium_metallurgy --climate-region northeast_rn --steps 50000

# Textile industry in São Paulo (moderate climate, predictable demand)
python cli.py industrial simulate --profile medium_textile --climate-region southeast_sp --days 30

# Food processing with continuous operation (24/7 cooling needs)
python cli.py battery compare --types li_ion_500kwh na_ion_200kwh --temperature-range 20 35
```

### Research & Analysis
```bash
# Compare performance across all Brazilian regions
python cli.py preset climate-all
python cli.py analysis optimize --climate-region northeast_rn --days 90
python cli.py analysis optimize --climate-region south_rs --days 90

# Battery degradation analysis under extreme conditions
python cli.py battery simulate --temperature 45 --action charge --hours 24
python cli.py rl evaluate --model models/rn_agent.zip --episodes 10
```

### Economic Studies
```bash
# ROI analysis for different regions and battery configurations
python cli.py industrial economics --profile medium_metallurgy --days 365
python cli.py preset report    # Quick analysis report generation
```

## Next Steps

1. **Regional RL training** - Optimize agents for specific Brazilian climate zones
2. **Multi-battery systems** - Extend to battery banks and hybrid storage
3. **Real-time integration** - Connect with weather APIs and energy market data
4. **Economic optimization** - Include battery replacement costs and degradation
5. **Industrial validation** - Partner with Brazilian industries for real-world testing
6. **Grid integration** - Model interaction with Brazilian electrical grid (SIN)

## Contributing

Contributions are welcome! Areas of interest:
- New battery models
- More precise climate data
- Advanced RL algorithms
- User interfaces
- Industrial use cases

---

*Project developed for intelligent energy systems research in Brazil*