# Energy Profile RL - DESS Management

Synthetic energy profile simulator for Brazil and Germany, with industrial consumption analysis and reinforcement learning-based DESS (Decentralized Energy Supply System) management.

## Overview

This project focuses on:
- **Synthetic energy profile generation** (solar, wind, hydropower)
- **Industrial consumption analysis** with automatic visualization  
- **RL-based DESS optimization** for cost minimization and energy supply guarantee
- **Comparative analysis** between Brazil and Germany energy systems

## Project Structure

```
energy-profile-rl/
├── src/
│   ├── app_cli.py              # Main CLI entry point
│   ├── core/
│   │   ├── dess_system.py      # DESS physics simulation
│   │   ├── rl_dess_env.py      # RL environment for DESS
│   │   ├── train.py            # RL training script
│   │   ├── evaluate.py         # Model evaluation & visualization
│   │   ├── energy_profile_*.py # Energy profile generation
│   │   └── synthetic_data_*.py # Synthetic data utilities
│   └── utils/
│       └── plot.py             # Visualization utilities
├── data/
│   ├── real/                   # Real-world energy data
│   └── synthetic/              # Generated profiles & contracts
├── logs/                       # TensorBoard training logs
├── models/                     # Trained RL models
├── results/                    # Evaluation results
└── docs/                       # Project documentation
```

## CLI Menu Options

Run `python src/app_cli.py` for:

1. **Generate Contract Prices & Validation Plot**
2. **Generate Full Industry Profile (Energy-based)**  
3. **Train DESS Management Agent (RL)**
4. **Evaluate Trained Agent**
5. **Clean all __pycache__ folders**

## Key Features

### Energy Systems Modeling
- Brazil: 67% hydro, 15% wind, 7% solar
- Germany: 31.5% wind, 13.8% solar
- Grid reliability analysis (SAIDI/SAIFI metrics)
- Renewable curtailment modeling

### DESS Components
- **Battery storage** with charge/discharge efficiency
- **Electrolyzer** for hydrogen production
- **Fuel cell** for hydrogen-to-electricity conversion
- **Industrial demand** profile simulation

### RL Environment
- **State**: Hour/day, industrial demand, generation, battery SOC, H₂ storage
- **Actions**: Battery power, electrolyzer power, fuel cell power
- **Reward**: Operational cost minimization + energy supply guarantee

## Usage

### Quick Start
```bash
pip install -r requirements.txt
python src/app_cli.py
```

### Direct Training
```bash
python src/core/train.py
```

### TensorBoard Monitoring
```bash
tensorboard --logdir logs/
```

## Technical Details

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: Stable-Baselines3 + Gymnasium
- **Time Resolution**: Configurable (15min - 1h)
- **Simulation Period**: 30+ days for training

---

*Part of the Energy Systems RL Research Project*