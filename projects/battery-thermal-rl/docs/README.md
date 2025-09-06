# Documentation

**Battery Thermal RL - Comprehensive Documentation**

This directory contains detailed mathematical and technical documentation for the Battery Thermal RL system.

## Documents

### 📊 [Mathematical Modeling Tutorial](mathematical_modeling.md)
**Complete mathematical framework for all system components**

A comprehensive guide covering:
- Battery thermal dynamics and electrochemistry
- Industrial energy system modeling
- Solar photovoltaic power generation
- Brazilian climate data generation
- Economic optimization formulations
- Reinforcement learning mathematical framework
- Implementation details and numerical methods

**Target Audience:** Researchers, engineers, and developers who need to understand the mathematical foundations of the system.

### 📐 [Mathematical Appendix](mathematical_appendix.md)
**Advanced derivations and theoretical background**

Supporting mathematical content including:
- Electrochemical fundamentals and battery physics
- Advanced climate modeling derivations
- Dynamic programming and optimization theory
- Deep reinforcement learning theory
- Numerical methods and stability analysis
- Model validation and sensitivity analysis

**Target Audience:** Academic researchers and advanced practitioners requiring deep theoretical understanding.

## Quick Navigation

### By Topic

**Battery Modeling:**
- [Thermal dynamics](mathematical_modeling.md#14-battery-thermal-dynamics)
- [Degradation models](mathematical_modeling.md#15-battery-degradation-model)
- [Electrochemical fundamentals](mathematical_appendix.md#a-battery-electrochemical-fundamentals)

**Industrial Systems:**
- [Demand profiles](mathematical_modeling.md#21-industrial-demand-profile)
- [Load modeling](mathematical_appendix.md#c-industrial-load-modeling)
- [Economic analysis](mathematical_modeling.md#5-economic-modeling)

**Climate & Solar:**
- [Climate data generation](mathematical_modeling.md#4-climate-data-generation)
- [Solar PV modeling](mathematical_modeling.md#3-solar-photovoltaic-model)
- [Advanced climate correlations](mathematical_appendix.md#b-climate-modeling-derivations)

**Reinforcement Learning:**
- [MDP formulation](mathematical_modeling.md#61-markov-decision-process-formulation)
- [Reward function design](mathematical_modeling.md#62-reward-function-design)
- [Advanced RL theory](mathematical_appendix.md#e-reinforcement-learning-theory)

**Implementation:**
- [Numerical integration](mathematical_modeling.md#81-numerical-integration)
- [Validation methods](mathematical_modeling.md#83-validation-methods)
- [Numerical methods](mathematical_appendix.md#g-numerical-methods)

## Mathematical Notation

### Common Symbols

| Symbol | Description | Units |
|--------|-------------|-------|
| $t$ | Time | [h] or [s] |
| $T$ | Temperature | [°C] or [K] |
| $P$ | Power | [kW] |
| $E$ | Energy | [kWh] |
| $SOC$ | State of Charge | [0-1] |
| $\eta$ | Efficiency | [0-1] |
| $G$ | Solar Irradiance | [kW/m²] |
| $C$ | Cost | [R$] |
| $\pi$ | Policy (RL) | - |
| $\theta$ | Parameters | - |

### Subscripts and Superscripts

| Notation | Meaning |
|----------|---------|
| $x_t$ | Value at time $t$ |
| $x_{max}$ | Maximum value |
| $x_{opt}$ | Optimal value |
| $\hat{x}$ | Estimated value |
| $x^*$ | Optimal solution |
| $\bar{x}$ | Average value |

## Validation and Testing

All mathematical models have been validated through:

- ✅ **Unit Tests** - Each equation implemented with test cases
- ✅ **Integration Tests** - System-level behavior verification  
- ✅ **Physical Constraints** - Energy conservation, power limits
- ✅ **Benchmark Comparison** - Against analytical solutions where available
- ✅ **Empirical Validation** - Using realistic Brazilian energy data


## References

### Primary Sources
- Plett, G. L. (2015). "Battery Management Systems, Volume I: Battery Modeling"
- Duffie, J. A., & Beckman, W. A. (2013). "Solar Engineering of Thermal Processes" 
- Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"

### Brazilian Energy Data
- EPE - Empresa de Pesquisa Energética (2024)
- ANEEL - Agência Nacional de Energia Elétrica (2024)
- INMET - Instituto Nacional de Meteorologia (2024)

### Implementation References
- See [Mathematical Modeling Tutorial](mathematical_modeling.md#references) for complete bibliography

---

**Note:** These documents serve as the theoretical foundation for understanding and extending the Battery Thermal RL system. For practical usage, refer to the main [README.md](../README.md) and [Quick Start Guide](../quick_start.md).