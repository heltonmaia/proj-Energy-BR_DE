# Quick Start Guide

**Get started with Battery Thermal RL in just a few steps!**

This guide provides a step-by-step walkthrough to quickly demonstrate the system's capabilities. For comprehensive documentation, see [README.md](README.md).

## 🚀 Installation & Setup

```bash
# 1. Navigate to project directory
cd projects/battery-thermal-rl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test installation
python cli.py info status
```

## 🎮 Two Ways to Use the System

### **📋 Interactive Menu (Recommended for Beginners)**
```bash
# Start interactive menu
python cli.py

# Follow the numbered menu options:
# 1. 🌤️  Climate Data → Generate climate data
# 2. 🔋 Battery Simulation → Simulate battery performance  
# 3. 🏭 Industrial System → Complete system simulation
# 4. 🤖 Reinforcement Learning → Train AI agents
```

### **⚡ Command-Line (Advanced Users)**  
```bash
# Direct commands for automation
python cli.py climate generate --region northeast_rn --days 30
python cli.py battery simulate --type li_ion_500kwh
python cli.py rl train --algorithm ppo --steps 10000
```

## 🏃‍♂️ **5-Minute Demo Workflow**

### **Step 1: Generate Climate Data (30 seconds)**
```bash
# Interactive menu
python cli.py
# Select: 1 → 1 → northeast_rn → 7 days → Enter

# OR Command-line
python cli.py climate generate --region northeast_rn --days 7 --output outputs/climate_data/demo_climate.csv
```

### **Step 2: Run Battery Simulation (1 minute)**
```bash
# Interactive menu  
python cli.py
# Select: 2 → 1 → li_ion_500kwh → 1 (use climate data) → charge → Enter

# OR Command-line
python cli.py battery simulate --type li_ion_500kwh --action charge --temperature 25 --hours 4
```

### **Step 3: Industrial System Simulation (2 minutes)**
```bash
# Interactive menu
python cli.py  
# Select: 3 → 1 → medium_textile → 7 days → northeast_rn → medium_1000kw → Enter

# OR Command-line
python cli.py industrial simulate --profile medium_textile --climate-region northeast_rn --days 7
```

### **Step 4: Generate Visualizations (1 minute)**
```bash
# Interactive menu - go to each module and select "📊 Plot [module] results"
# 1. 🌤️  Climate Data → 6 → Plot all regions comparison
# 2. 🔋 Battery Simulation → 5 → Plot individual file
# 3. 🏭 Industrial System → 4 → Plot individual file
```

### **Step 5: Train RL Agent (Optional - 30 seconds for quick demo)**
```bash
# Interactive menu
python cli.py
# Select: 4 → 1 → ppo → 1000 steps → Enter

# OR Command-line (quick training)
python cli.py rl train --algorithm ppo --steps 1000 --output outputs/reinforcement_learning/demo_agent
```

## 📊 **What You'll See**

After completing the demo:

### **📁 Generated Files:**
```
outputs/
├── climate_data/
│   ├── northeast_rn_7d.csv                    # Climate data
│   └── climate_all_regions_comparison.png     # Climate comparison plot
├── battery_simulation/  
│   ├── battery_sim_li_ion_500kwh_charge.csv   # Battery simulation data
│   └── battery_plot_*.png                     # Battery analysis plots
├── industrial_system/
│   ├── industrial_medium_textile_7d.csv       # Industrial simulation data  
│   └── industrial_plot_*.png                  # Industrial analysis plots
└── reinforcement_learning/
    └── demo_agent.zip                          # Trained RL model
```

### **📈 Visualizations Created:**
- **Climate plots**: Daily temperature and solar patterns across Brazilian regions
- **Battery plots**: 7-day SOC evolution, efficiency trends, energy transfers  
- **Industrial plots**: Energy flows, cost analysis, daily consumption patterns
- **Comparative plots**: Multi-region and multi-configuration comparisons

## 🎯 **Next Steps - Explore Advanced Features**

### **🌍 Multi-Region Analysis**
```bash
# Generate data for all Brazilian regions
python cli.py preset climate-all

# Compare battery performance across regions  
python cli.py battery simulate --type li_ion_500kwh --climate-region northeast_rn --days 30
python cli.py battery simulate --type li_ion_500kwh --climate-region south_rs --days 30
```

### **🤖 Advanced RL Training**
```bash
# Full training session (10-30 minutes)
python cli.py rl train --algorithm ppo --steps 100000 --climate-region northeast_rn

# Evaluate trained agent
python cli.py rl evaluate --model outputs/reinforcement_learning/ppo_agent.zip --episodes 5
```

### **🏭 Industrial Optimization**
```bash
# Compare different industrial profiles
python cli.py industrial simulate --profile medium_metallurgy --days 30
python cli.py industrial simulate --profile medium_food --days 30

# Economic analysis
python cli.py industrial economics --profile medium_metallurgy --days 365
```

### **⚡ Automation Workflows**
```bash
# Complete development cycle
python cli.py preset dev-full     # Full setup + training + analysis  

# Comprehensive benchmarking
python cli.py preset benchmark    # Multi-configuration analysis

# Quick training and evaluation cycle
python cli.py preset train && python cli.py preset evaluate
```

## 🔧 **Troubleshooting**

### **Common Issues:**
```bash
# Missing dependencies
pip install -r requirements.txt

# Plotting issues (matplotlib)
pip install matplotlib seaborn

# RL training issues (optional advanced features)  
pip install stable-baselines3[extra] torch

# Check system status
python cli.py info status
```

### **Getting Help:**
```bash
# General help
python cli.py --help

# Module-specific help
python cli.py climate --help
python cli.py battery --help
python cli.py rl --help

# List all available options
python cli.py info list
```

## 🎓 **Understanding the Results**

### **Climate Data Analysis:**
- **Temperature ranges**: Optimal (15-25°C), challenging (>30°C), extreme (>40°C)  
- **Solar potential**: Northeast Brazil offers exceptional solar resources (7+ kWh/m²/day)
- **Regional variations**: Significant differences impact battery performance

### **Battery Performance:**
- **SOC evolution**: Shows charge/discharge patterns over time
- **Efficiency curves**: Temperature dependency clearly visible  
- **Energy transfers**: Cumulative energy flow analysis
- **Degradation tracking**: Long-term battery health monitoring

### **Industrial System Economics:**
- **Self-consumption ratios**: Typically 60-85% with optimal sizing
- **Cost savings**: 20-40% reduction with intelligent battery management
- **Peak shaving**: Significant reduction in demand charges
- **Solar integration**: Maximizing renewable energy utilization

### **RL Agent Performance:**
- **Learning curves**: Reward improvement over training episodes
- **Policy evaluation**: Comparison with baseline strategies
- **Multi-objective optimization**: Balancing cost, battery health, and grid stability

---

**🎯 Ready to explore?** Start with the interactive menu (`python cli.py`) and follow the guided workflows!

For detailed technical information, mathematical foundations, and advanced usage, see the complete [README.md](README.md) and documentation in the `docs/` folder.