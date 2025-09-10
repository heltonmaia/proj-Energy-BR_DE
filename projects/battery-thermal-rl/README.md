# Battery Thermal Management with Reinforcement Learning

**Advanced AI-powered battery optimization system for industrial applications** with solar panels, featuring thermal management and Brazilian climate conditions.

## 🎯 Overview

This project develops a **Reinforcement Learning system** to optimize battery usage in medium-sized industries with solar photovoltaic generation in Brazil, considering:

- **🌡️ Intelligent thermal management** - Temperature-aware optimization
- **🏭 Real industrial profiles** - Modeling of different industry types  
- **☀️ Solar integration** - Maximizing self-consumption and savings
- **💰 Economic optimization** - Cost reduction with time-of-use tariffs
- **🔋 Battery preservation** - Minimizing degradation and extending lifespan
- **🌍 Brazilian climate data** - Region-specific models and conditions

## 🏗️ System Architecture

```
Brazilian Climate Data (Temperature, Solar) 
         ↓
Industrial System (Energy Demand + Solar)
         ↓
    RL Environment ← → RL Agent (PPO/SAC/TD3)
         ↓
Battery with Thermal Model
         ↓
Intelligent Optimization (Charge/Discharge)
```

## 📁 Project Structure

```
battery-thermal-rl/
├── src/                           # Source code
│   ├── models/                   # System models
│   │   ├── battery.py           # Battery thermal model
│   │   └── industrial_system.py # Industrial system + solar
│   ├── data/                    # Data generators
│   │   └── climate_data.py      # Brazilian climate data
│   ├── rl/                      # Reinforcement Learning
│   │   └── battery_thermal_env.py # Gymnasium environment
│   └── utils/                   # Utilities and helpers
├── outputs/                      # All simulation results
│   ├── climate_data/            # Climate data + plots
│   ├── industrial_system/       # Industrial battery system + plots + analysis
│   └── reinforcement_learning/  # Trained RL models
├── docs/                        # Mathematical documentation
│   ├── mathematical_modeling.md # Complete mathematical tutorial
│   └── mathematical_appendix.md # Advanced theory and derivations
├── cli.py                       # Main CLI interface
├── quick_start.md              # Quick start guide  
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Key Features Implemented

### 🌤️ **Climate Data System**
- **5 Brazilian regions** with distinct climate profiles optimized for battery thermal analysis:
  - **Southeast (SP)**: Moderate conditions, balanced year-round operation
  - **Northeast (RN)**: High temperatures (29.5°C summer), exceptional solar potential (7.2 kWh/m²/day)
  - **South (RS)**: Greatest thermal variation (13-25°C), challenging for battery management
  - **Central-West (MT)**: Continental climate with high thermal amplitude (12°C daily variation)
  - **North (AM)**: Humid tropical, consistent temperatures
- **Hourly data generation** with temperature, solar irradiance, and humidity patterns
- **Realistic seasonal** and stochastic variations based on historical data
- **📊 Visualization**: Comparative plots showing daily average profiles across all regions

### 🔋 **Battery Models (Thermal-Aware)**
- **Li-ion and Na-ion** with real industrial specifications (100kWh, 500kWh, 200kWh)
- **Variable efficiency** with temperature (95% charge, 90% discharge at optimal temp)
- **Thermal modeling**: 
  - Optimal range: 15-25°C (Li-ion), 10-35°C (Na-ion)
  - Efficiency loss: 0.5% per degree deviation
  - Critical temperature: 45°C max
- **Degradation model**: Linear degradation based on cycles and temperature
- **Safety limits**: Automatic shutdown at critical temperatures
- **📊 Visualization**: 30-day evolution plots showing temperature vs SOC, efficiency, energy transfers

### 🏭 **Industrial System + Solar Integration**
- **Industrial profiles**: metallurgy, textile, food, chemical industries
- **Photovoltaic system** with temperature correction and realistic generation
- **Brazilian tariff structure**: Peak/off-peak hours with realistic pricing
- **Economic calculations**: Complete cost analysis, savings, self-consumption ratios
- **📊 Visualization**: Energy flows, daily patterns, cost analysis over full simulation period

### 🤖 **Reinforcement Learning System**
- **Environment**: Gymnasium-based with 15 observation features
- **Action Space**: Continuous charge/discharge power [-1, +1]
- **Algorithms**: PPO, SAC, TD3 from Stable-Baselines3
- **Multi-objective reward function**:
  - Energy cost savings (weight: 1.0)
  - Battery health preservation (weight: 0.3)  
  - Grid stability (weight: 0.2)
  - Temperature penalty (weight: 0.5)
- **Training**: Configurable episodes with TensorBoard logging
- **Evaluation**: Performance metrics and comparison with baseline strategies

## 📊 **Advanced Visualization System**

Each module includes comprehensive plotting capabilities:

### **Climate Data Plots** (`outputs/climate_data/`)
- Daily average temperature, solar, and humidity profiles (24h)
- Full 30-day evolution plots
- Multi-region comparison plots
- Battery performance implications analysis

### **Industrial Battery System Plots** (`outputs/industrial_system/`)
- **Integrated energy flows**: industrial demand, solar generation, battery operations, grid interactions
- **Battery performance**: SOC evolution, temperature management, thermal efficiency over time
- **Economic analysis**: daily energy costs, self-consumption ratios, cost savings visualization
- **System performance metrics**: comprehensive 4-panel analysis with key performance indicators
- **Multi-configuration comparisons**: different battery types, industrial profiles, and climate regions

## 🎮 How to Use

### **Installation**
```bash
cd projects/battery-thermal-rl
pip install -r requirements.txt

# Test installation
python cli.py info status
```

### **Interactive Menu Mode (Recommended)**
```bash
# Start interactive menu
python cli.py

# Navigate through:
# 1. 🌤️  Climate Data - Generate and visualize climate data
# 2. 🔋 Battery Simulation - Simulate battery performance  
# 3. 🏭 Industrial System - Complete industrial system simulation
# 4. 🤖 Reinforcement Learning - Train and evaluate RL agents
# 5. 📊 Analysis & Reports - Generate analysis reports
# 6. ⚡ Preset Workflows - Quick automation workflows
```

### **Command-Line Mode (Advanced)**
```bash
# Generate climate data
python cli.py climate generate --region northeast_rn --days 30 --output outputs/climate_data/rn_climate.csv

# Battery simulation with real climate data
python cli.py battery simulate --type li_ion_500kwh --climate-data outputs/climate_data/rn_climate.csv

# Industrial system simulation
python cli.py industrial simulate --profile medium_metallurgy --climate-region northeast_rn --days 30

# Train RL agent
python cli.py rl train --algorithm ppo --steps 100000 --climate-region northeast_rn --output outputs/reinforcement_learning/ppo_agent

# Evaluate trained agent
python cli.py rl evaluate --model outputs/reinforcement_learning/ppo_agent.zip --episodes 10
```

### **Preset Workflows**
```bash
# Show all available workflows
python cli.py preset

# Quick workflows
python cli.py preset dev-quick      # Setup + train + evaluate + report
python cli.py preset climate-all    # Generate all Brazilian regions data
python cli.py preset train         # Quick RL training (10k steps)
python cli.py preset train-full    # Full RL training (100k steps)  
python cli.py preset evaluate      # Evaluate best model
python cli.py preset benchmark     # Multi-configuration benchmark
```

## 🔬 **Technical Implementation**

### **Battery Physics Model**
- **Mathematical approach**: Direct mathematical formulas (no AI in physics)
- **Thermal efficiency**: `η = η_base - temp_deviation × 0.005`
- **Heat generation**: `Q = P × t × (1 - η) × 0.1`
- **Degradation**: `D = max(0.7, 1.0 - cycles/lifecycle)`

### **Reinforcement Learning**
- **Intelligent decision making**: AI-powered optimization
- **State space**: 15 features (battery, climate, system, temporal)
- **Action interpretation**: 
  - `> 0.05`: Charge mode
  - `< -0.05`: Discharge mode  
  - `[-0.05, 0.05]`: Hold mode
- **Reward calculation**: Multi-objective optimization with configurable weights

### **Climate Data Generation**
- **Statistical modeling**: Based on historical Brazilian weather data
- **Hourly patterns**: Realistic diurnal and seasonal cycles
- **Regional variations**: Temperature ranges, solar potential, humidity levels
- **Stochastic elements**: Natural weather variability

## 📈 **Expected Results**

### **Cost Optimization**
- **20-40% savings** in energy costs through intelligent scheduling
- **60% reduction** in peak hour consumption using battery discharge  
- **80-90% solar self-consumption** with optimal charge timing

### **Battery Health Preservation**
- **15-25% battery lifespan extension** through thermal-aware operation
- **Reduced degradation** in high-temperature regions (RN: 29.5°C summer)
- **Temperature-optimized charging** during cooler periods

### **Regional Performance**
- **Northeast (RN)**: Best solar economics, thermal management critical
- **Southeast (SP)**: Balanced performance, optimal for baseline comparisons
- **South (RS)**: Seasonal optimization crucial, best winter performance  
- **Central-West (MT)**: Daily thermal cycle optimization opportunities
- **North (AM)**: Humidity considerations, consistent solar potential

## 🎯 **Use Cases & Applications**

### **Industrial Applications**
```bash
# Metallurgy plant in hot climate (high solar, challenging temperatures)
python cli.py rl train --industrial-profile medium_metallurgy --climate-region northeast_rn --steps 50000

# Textile industry in moderate climate (predictable demand patterns)
python cli.py industrial simulate --profile medium_textile --climate-region southeast_sp --days 30

# Food processing with 24/7 operation (continuous cooling needs)
python cli.py battery compare --types li_ion_500kwh na_ion_200kwh --temperature-range 20 35
```

### **Research & Analysis**
```bash
# Compare performance across all Brazilian regions
python cli.py preset climate-all
python cli.py analysis optimize --climate-region northeast_rn --days 90

# Battery degradation analysis under extreme conditions
python cli.py battery simulate --temperature 45 --action charge --hours 24

# Multi-region RL training comparison
python cli.py rl train --climate-region northeast_rn --steps 100000
python cli.py rl train --climate-region south_rs --steps 100000
```

### **Economic Studies**
```bash
# ROI analysis for different regions and battery configurations
python cli.py industrial economics --profile medium_metallurgy --days 365
python cli.py preset report    # Generate comprehensive analysis report
```

## 📚 **Documentation Structure**

- **README.md** (this file): Complete system overview and usage
- **quick_start.md**: Step-by-step beginner guide
- **docs/mathematical_modeling.md**: Complete mathematical foundations
- **docs/mathematical_appendix.md**: Advanced theory and derivations

## 🔮 **Future Developments**

1. **🌐 Multi-battery systems** - Battery banks and hybrid storage
2. **📡 Real-time integration** - Weather APIs and energy market data  
3. **💹 Advanced economics** - Battery replacement costs and LCOE optimization
4. **🏛️ Grid integration** - Brazilian electrical system (SIN) modeling
5. **🤖 Advanced RL** - Multi-agent systems and hierarchical RL
6. **📱 Web interface** - Browser-based monitoring and control

## 🤝 **Contributing**

Areas of interest for contributions:
- New battery chemistry models (LFP, solid-state)
- Enhanced climate data sources and accuracy
- Advanced RL algorithms (A3C, Rainbow DQN)
- Real-world validation with industrial partners
- Extended economic modeling (CAPEX/OPEX optimization)

## 📄 **Dependencies**

### **Core Requirements**
```
pandas>=2.0.0          # Data manipulation
matplotlib>=3.7.0       # Plotting and visualization  
numpy>=1.24.0          # Numerical computations
```

### **Reinforcement Learning**
```
gymnasium>=0.29.0       # RL environment framework
stable-baselines3>=2.0.0 # RL algorithms (PPO, SAC, TD3)
```

### **Analysis & Visualization**
```
scipy>=1.10.0          # Scientific computations
scikit-learn>=1.3.0    # Machine learning utilities
seaborn>=0.12.0        # Statistical data visualization
```

---

*Project developed for intelligent energy systems research in Brazil - combining advanced AI with real-world industrial applications for sustainable energy optimization.*