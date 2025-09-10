# Energy Systems Research - RL & Optimization

**Advanced AI-powered energy management systems** for industrial applications across Brazil and Germany markets, featuring two complementary project lines that together provide comprehensive energy optimization solutions.

## Research Vision

This multi-project repository develops cutting-edge **Reinforcement Learning and optimization techniques** for energy systems, addressing two critical challenges in modern industrial energy management:

1. **System-level energy optimization** - DESS management and renewable integration
2. **Component-level thermal optimization** - Battery health preservation and performance

## Projects Overview

### 🔋 [Energy Profile RL](projects/energy-profile-rl/) - **DESS Management**
**Production-ready** synthetic energy profile simulator with advanced RL-based DESS (Decentralized Energy Supply System) management for industrial applications.

- **Focus**: Industrial energy optimization, multi-component DESS control
- **Status**: ✅ **Fully implemented and operational**
- **RL Algorithm**: PPO for coordinated battery, electrolyzer, and fuel cell control
- **Geographic Scope**: Brazil & Germany energy systems comparative analysis
- **Key Features**: 
  - Real-time synthetic data generation with renewable profiles
  - Cost optimization with operational constraints
  - Grid reliability analysis (SAIDI/SAIFI metrics)
  - TensorBoard training visualization
  - Industrial consumption pattern modeling

### 🌡️ [Battery Thermal RL](projects/battery-thermal-rl/) - **Thermal Management**
**Advanced AI-powered battery optimization system** for industrial applications with solar integration, featuring intelligent thermal management optimized for Brazilian climate conditions.

- **Focus**: Temperature-aware battery optimization, degradation prevention
- **Status**: ✅ **Fully implemented with comprehensive features**
- **RL Algorithm**: PPO, SAC, TD3 for thermal-aware charge/discharge control
- **Geographic Scope**: 5 Brazilian regions with distinct climate profiles
- **Key Features**:
  - Intelligent thermal management (15-45°C operational range)
  - Real industrial profiles (metallurgy, textile, food, chemical)
  - Solar photovoltaic integration with Brazilian tariff structures
  - Multi-objective optimization (cost, health, grid stability, temperature)
  - Advanced visualization system with economic analysis
  - Comprehensive climate data modeling

## Complementary Architecture

These projects are designed to work together, addressing different scales of energy optimization:

```
System Level (Energy Profile RL)
├── DESS Management (Battery + Electrolyzer + Fuel Cell)
├── Renewable Integration (Solar + Wind + Hydro)  
├── Grid Reliability Analysis
└── Industrial Demand Modeling
                    ↓
        **Data Exchange & Integration**
                    ↓
Component Level (Battery Thermal RL)
├── Temperature-Aware Battery Control
├── Thermal Degradation Prevention
├── Climate-Specific Optimization
└── Real-Time Performance Monitoring
```

## 📁 Project Structure

```
proj-Energy-BR_DE/
├── projects/
│   ├── energy-profile-rl/          # System-level DESS management
│   │   ├── src/core/dess_system.py # Multi-component energy system
│   │   ├── data/synthetic/         # Energy profiles & contracts
│   │   ├── models/                 # Trained RL agents
│   │   └── results/                # Performance evaluations
│   └── battery-thermal-rl/         # Component-level thermal optimization
│       ├── src/models/battery.py   # Thermal-aware battery models
│       ├── outputs/climate_data/   # Brazilian climate analysis
│       ├── outputs/industrial_system/ # Industrial integration results
│       └── docs/rl.md              # Comprehensive RL documentation
├── requirements-common.txt         # Shared dependencies
└── README.md                       # This overview
```

## 🚀 Getting Started

### **Quick Setup**
```bash
# Clone and setup
git clone <repository>
cd proj-Energy-BR_DE
pip install -r requirements-common.txt
```

### **Option 1: Energy Profile RL (System Management)**
```bash
cd projects/energy-profile-rl/
pip install -r requirements.txt
python src/app_cli.py  # Interactive menu

# Or direct training
python src/core/train.py
tensorboard --logdir logs/  # Monitor training
```

### **Option 2: Battery Thermal RL (Thermal Optimization)**
```bash
cd projects/battery-thermal-rl/
pip install -r requirements.txt
python cli.py  # Interactive menu

# Quick workflows
python cli.py preset dev-quick      # Complete workflow
python cli.py preset climate-all    # Generate Brazilian climate data
python cli.py rl train --algorithm ppo --steps 100000
```

### **Advanced: Combined Workflow**
```bash
# 1. Train system-level DESS management
cd projects/energy-profile-rl/
python src/core/train.py

# 2. Use results for thermal-aware optimization
cd ../battery-thermal-rl/
python cli.py rl train --climate-region northeast_rn --steps 100000

# 3. Integrated analysis
python cli.py preset benchmark  # Multi-configuration comparison
```

## 🔬 Research Focus

### **System-Level Energy Modeling**
- **Synthetic energy profile generation** with realistic renewable patterns (solar, wind, hydro)
- **Brazil vs Germany** comprehensive energy system comparison and market analysis
- **Industrial consumption patterns** with automatic visualization and grid reliability studies (SAIDI/SAIFI)
- **Renewable curtailment modeling** and transmission bottleneck analysis
- **Multi-component DESS systems** with coordinated battery, electrolyzer, and fuel cell management

### **Component-Level Thermal Optimization**
- **Climate-aware battery management** across 5 Brazilian regions with distinct thermal profiles
- **Industrial integration** with real consumption patterns (metallurgy, textile, food, chemical)
- **Solar photovoltaic systems** with temperature correction and Brazilian tariff structures
- **Degradation prevention** through intelligent thermal management (15-45°C operational range)
- **Multi-objective optimization** balancing cost, battery health, grid stability, and temperature

### **Advanced RL Applications**
- **System Management**: PPO for coordinated DESS optimization with energy supply guarantee
- **Thermal Control**: PPO, SAC, TD3 algorithms for temperature-aware charge/discharge decisions  
- **Cost Minimization**: Operational cost reduction while ensuring energy supply and battery longevity
- **Safety Optimization**: Thermal safety constraints with performance trade-offs and critical temperature management
- **Real-time Decision Making**: Continuous control with hourly optimization cycles

### **Technical Innovations**
- **Multi-scale optimization**: System-level and component-level integration
- **Physics-informed models**: Realistic battery thermal dynamics with efficiency curves
- **Climate integration**: Regional Brazilian weather data with industrial applications
- **Advanced visualization**: Comprehensive plotting systems with economic analysis
- **Cross-market analysis**: Brazil-Germany comparative studies with different energy portfolios

## 📊 Monitoring & Visualization

### **Training Monitoring**
Both projects provide comprehensive monitoring capabilities:
```bash
# Energy Profile RL - DESS training monitoring
tensorboard --logdir projects/energy-profile-rl/logs/

# Battery Thermal RL - Thermal optimization monitoring  
tensorboard --logdir projects/battery-thermal-rl/logs/
```

### **Advanced Visualization Systems**
- **Energy Profile RL**: Industrial consumption analysis, renewable generation patterns, DESS performance metrics
- **Battery Thermal RL**: Climate data analysis, battery thermal evolution, economic performance, multi-region comparisons

## **Expected Results & Performance**

### **System-Level Optimization (Energy Profile RL)**
- **20-40% operational cost reduction** through intelligent DESS management
- **Grid reliability improvement** with renewable integration strategies
- **Energy supply guarantee** while minimizing operational costs
- **Cross-market insights** comparing Brazil's hydro-dominated vs Germany's renewable-diverse systems

### **Component-Level Optimization (Battery Thermal RL)**
- **20-40% energy cost savings** through intelligent scheduling with Brazilian tariff structures
- **15-25% battery lifespan extension** via thermal-aware operation
- **60% reduction in peak hour consumption** using optimized battery discharge
- **80-90% solar self-consumption** with optimal charge timing
- **Regional performance optimization** across 5 Brazilian climate zones

### **Combined System Benefits**
- **Hierarchical optimization**: System-level planning + component-level execution
- **Thermal-aware system management**: DESS decisions considering battery temperature constraints
- **Enhanced industrial applications**: Complete energy management from grid to component level
- **Comprehensive economic analysis**: CAPEX/OPEX optimization across multiple scales

## **Applications & Use Cases**

### **Industrial Applications**
- **Medium-large industries** with solar integration in Brazil
- **Energy-intensive operations** (metallurgy, textile, food processing, chemical)
- **Multi-component energy systems** with battery storage, hydrogen production, and renewable generation
- **Climate-specific optimizations** for different Brazilian regions

### **Research Applications**
- **Energy system modeling** and renewable integration studies
- **Battery thermal management** research with real climate conditions
- **RL algorithm development** for energy applications
- **Economic analysis** of industrial energy systems
- **Cross-country energy market** comparative studies (Brazil-Germany)

## **Contributing**

Each project maintains independent development cycles while sharing common utilities and methodologies:

### **Energy Profile RL Contributions**
- DESS component modeling and optimization
- New renewable energy profiles and industrial patterns
- Grid reliability analysis enhancements
- Cross-country energy market extensions

### **Battery Thermal RL Contributions**
- New battery chemistry models (LFP, solid-state, Na-ion variations)
- Enhanced climate data sources and regional coverage
- Advanced RL algorithms for thermal management
- Industrial integration patterns and economic modeling

### **Shared Infrastructure**
- Common data formats and exchange protocols
- Shared visualization utilities and analysis tools
- Cross-project integration capabilities
- Documentation and testing frameworks

See individual project READMEs for specific contribution guidelines and development setup instructions.

---

**🔬 Advanced AI-Powered Energy Research** - *Combining system-level optimization with component-level thermal intelligence for next-generation industrial energy management*

