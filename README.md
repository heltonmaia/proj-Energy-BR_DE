# Energy Systems Research - RL & Optimization

Multi-project repository for energy systems research using reinforcement learning and optimization techniques, focusing on Brazil and Germany energy markets.

## Projects Overview

### 🔋 [Energy Profile RL](projects/energy-profile-rl/)
Synthetic energy profile simulator with RL-based DESS (Decentralized Energy Supply System) management.

- **Focus**: Industrial energy optimization, DESS management
- **RL Algorithm**: PPO for battery, electrolyzer, and fuel cell control
- **Regions**: Brazil & Germany energy systems comparison
- **Key Features**: Synthetic data generation, cost optimization, grid analysis

### 🌡️ [Battery Thermal RL](projects/battery-thermal-rl/) 🚧
Projeto futuro para modelagem de temperatura de baterias e otimização com RL.

- **Focus**: Controle de temperatura, prevenção de degradação
- **Status**: Em desenvolvimento - estrutura básica criada
- **Objetivo**: Estratégias de gerenciamento térmico baseadas em RL

## Project Structure

```
proj-Energy-BR_DE/
├── projects/
│   ├── energy-profile-rl/          # DESS management & energy profiles
│   └── battery-thermal-rl/         # Battery thermal optimization (em desenvolvimento)
├── requirements-common.txt         # Dependências compartilhadas
└── README.md                       # Este arquivo de overview
```

## Getting Started

### 1. Install Common Dependencies
```bash
pip install -r requirements-common.txt
```

### 2. Choose Your Project

#### Energy Profile RL (DESS Management)
```bash
cd projects/energy-profile-rl/
pip install -r requirements.txt
python src/app_cli.py
```

#### Battery Thermal RL (Em Desenvolvimento)
```bash
cd projects/battery-thermal-rl/
pip install -r requirements.txt
# Aguardando implementação
```

### 3. Alternative: Use uv (recommended)
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
uv pip install -r requirements-common.txt
```

## Research Focus

### Energy Systems Modeling
- **Synthetic energy profile generation** for solar, wind, hydropower
- **Brazil vs Germany** energy system comparison and analysis
- **Industrial consumption patterns** and grid reliability studies
- **Renewable curtailment** and transmission bottleneck modeling

### Reinforcement Learning Applications
- **DESS optimization**: Battery, electrolyzer, fuel cell management
- **Cost minimization**: Operational cost reduction while ensuring energy supply
- **Thermal management**: (Planejado) Battery temperature control and degradation prevention
- **Safety optimization**: (Planejado) Thermal safety constraints and performance trade-offs

### Technical Innovations
- **Multi-agent RL** for complex energy systems
- **Physics-informed models** for realistic battery and thermal dynamics
- **Real-time optimization** for industrial energy management
- **Comparative analysis** across different energy markets and technologies

## Monitoring & Visualization

All projects support TensorBoard monitoring:
```bash
tensorboard --logdir projects/[project-name]/logs/
# Open browser: http://localhost:6006
```

## Contributing

Each project has independent development cycles but shares common utilities. See individual project READMEs for specific contribution guidelines.

---

*Multi-project research repository for energy systems optimization using reinforcement learning*
