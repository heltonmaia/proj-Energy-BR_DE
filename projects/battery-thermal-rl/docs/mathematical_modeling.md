# Mathematical Modeling Tutorial

**Battery Thermal Management with Reinforcement Learning - Complete Mathematical Framework**

This tutorial provides a comprehensive mathematical foundation for all models implemented in the Battery Thermal RL system - an advanced AI-powered battery optimization system for medium-sized industries with solar photovoltaic generation in Brazil.

The system combines rigorous mathematical modeling (for battery physics and industrial systems) with intelligent decision-making (via reinforcement learning) to optimize battery usage considering thermal management, economic factors, and Brazilian climate conditions.

**What This Document Contains:**
- Complete mathematical formulations for battery thermal dynamics, industrial energy systems, solar PV modeling, Brazilian climate data generation, economic optimization, and reinforcement learning frameworks
- Implementation details with discrete-time equations used in the actual code
- Validation methods and numerical considerations for practical deployment

**Prerequisites:** Basic knowledge of thermodynamics, electrochemistry, economics, and machine learning. For practical usage without mathematical details, see the main [README.md](../README.md) and [Quick Start Guide](../quick_start.md).

## Table of Contents

1. [Battery Thermal Model](#1-battery-thermal-model)
2. [Industrial Energy System](#2-industrial-energy-system)
3. [Solar Photovoltaic Model](#3-solar-photovoltaic-model)
4. [Climate Data Generation](#4-climate-data-generation)
5. [Economic Modeling](#5-economic-modeling)
6. [Reinforcement Learning Framework](#6-reinforcement-learning-framework)
7. [Optimization Objectives](#7-optimization-objectives)
8. [Implementation Details](#8-implementation-details)

---

## 1. Battery Thermal Model

### 1.1 Fundamental Battery Parameters

The battery model is characterized by the following state variables:

- **State of Charge (SOC)**: $\text{SOC}(t) \in [SOC_{min}, SOC_{max}]$
- **Battery Temperature**: $T_b(t)$ [°C]
- **Degradation Factor**: $D(t) \in [0, 1]$
- **Cycle Count**: $N_{cycles}(t)$

### 1.2 State of Charge Dynamics

The SOC evolution follows the fundamental energy conservation principle:

$$\frac{d\text{SOC}}{dt} = \frac{1}{C_{nom}} \cdot \frac{P_{charge}(t) \cdot \eta_{charge}(T_b) - P_{discharge}(t) / \eta_{discharge}(T_b)}{3600}$$

Where:
- $C_{nom}$: Nominal capacity [kWh]
- $P_{charge}(t)$, $P_{discharge}(t)$: Charge/discharge power [kW]
- $\eta_{charge}(T_b)$, $\eta_{discharge}(T_b)$: Temperature-dependent efficiencies

**Discrete Implementation:**
```
SOC(t+Δt) = SOC(t) + (E_charge × η_charge - E_discharge / η_discharge) / C_nom
```

### 1.3 Temperature-Dependent Efficiency

The efficiency functions are modeled based on empirical battery characteristics:

$$\eta_{charge}(T_b) = \eta_{base,charge} \cdot f_{temp}(T_b) \cdot D(t)$$

$$\eta_{discharge}(T_b) = \eta_{base,discharge} \cdot f_{temp}(T_b) \cdot D(t)$$

Where the temperature correction factor is:

$$f_{temp}(T_b) = \begin{cases}
1.0 & \text{if } T_{opt,min} \leq T_b \leq T_{opt,max} \\
1.0 - \alpha \cdot |T_b - T_{opt,nearest}| & \text{otherwise}
\end{cases}$$

**Parameters:**
- $\eta_{base,charge} = 0.95$, $\eta_{base,discharge} = 0.90$ (Li-ion)
- $T_{opt,min} = 15°C$, $T_{opt,max} = 25°C$ 
- $\alpha = 0.005$ [efficiency loss per °C deviation]

### 1.4 Battery Thermal Dynamics

The battery temperature evolution considers heat generation from losses and ambient cooling:

$$C_{th} \frac{dT_b}{dt} = Q_{losses}(t) - h \cdot A \cdot (T_b(t) - T_{ambient}(t))$$

**Heat Generation from Losses:**
$$Q_{losses}(t) = P_{charge}(t) \cdot (1 - \eta_{charge}) + P_{discharge}(t) \cdot (1 - \eta_{discharge})$$

**Simplified Discrete Model (implemented):**
$$T_b(t+\Delta t) = T_{ambient}(t) + \frac{Q_{losses} \cdot \Delta t}{k_{cooling}}$$

Where $k_{cooling} = 10.0$ is an empirical cooling factor.

### 1.5 Battery Degradation Model

Battery degradation follows a cycle-life model with temperature acceleration:

$$D(t) = \max(D_{min}, 1.0 - \frac{N_{cycles}(t)}{N_{life}} \cdot f_{temp,degradation}(T_{avg}))$$

**Temperature Acceleration Factor:**
$$f_{temp,degradation}(T_{avg}) = \exp\left(\frac{E_a}{k_B} \left(\frac{1}{T_{ref}} - \frac{1}{T_{avg} + 273.15}\right)\right)$$

**Simplified Implementation:**
```
degradation_factor = max(0.7, 1.0 - (cycles / cycle_life))
```

Where:
- $N_{life} = 6000$ cycles (Li-ion), $8000$ cycles (Na-ion)
- $D_{min} = 0.7$ (70% end-of-life capacity)

---

## 2. Industrial Energy System

### 2.1 Industrial Demand Profile

Industrial energy demand is modeled as a multi-component time series:

$$P_{demand}(t) = P_{base} + P_{operational}(t) \cdot f_{seasonal}(t) \cdot f_{weekend}(t) \cdot f_{random}(t)$$

**Base Load:**
$$P_{base} = \text{constant base consumption [kW]}$$

**Operational Component:**
$$P_{operational}(t) = \begin{cases}
(P_{peak} - P_{base}) \cdot g(h) & \text{if } h_{start} \leq h \leq h_{end} \\
0 & \text{otherwise}
\end{cases}$$

**Operational Profile Function:**
$$g(h) = \begin{cases}
0.9 & \text{if } h \in [6,8] \cup [17,19] \text{ (peak periods)} \\
0.8 & \text{if } h \in [9,16] \text{ (normal operation)} \\
0.6 & \text{otherwise}
\end{cases}$$

**Seasonal Variation:**
$$f_{seasonal}(t) = 1.0 + A_{seasonal} \cdot \sin\left(\frac{2\pi(month - 1)}{12}\right)$$

**Weekend Factor:**
$$f_{weekend}(t) = \begin{cases}
f_{weekend,factor} & \text{if weekend} \\
1.0 & \text{if weekday}
\end{cases}$$

**Random Variation:**
$$f_{random}(t) \sim \mathcal{N}(1.0, 0.05^2)$$

(5% standard deviation)

### 2.2 Industrial Profile Parameters

| Profile | $P_{base}$ [kW] | $P_{peak}$ [kW] | $h_{start}$ | $h_{end}$ | $f_{weekend}$ | $A_{seasonal}$ |
|---------|----------------|----------------|-------------|-----------|---------------|----------------|
| Medium Metallurgy | 200 | 800 | 6 | 22 | 0.3 | 0.15 |
| Medium Textile | 150 | 600 | 7 | 19 | 0.2 | 0.25 |
| Medium Food | 180 | 500 | 5 | 21 | 0.6 | 0.20 |
| Medium Chemical | 400 | 1200 | 0 | 24 | 0.9 | 0.10 |

---

## 3. Solar Photovoltaic Model

### 3.1 Solar Irradiance to Power Conversion

The solar power generation model incorporates multiple efficiency factors:

$$P_{solar}(t) = P_{rated} \cdot \frac{G(t)}{G_{STC}} \cdot \eta_{panel}(T_{ambient}) \cdot \eta_{system} \cdot \eta_{degradation}(t)$$

Where:
- $P_{rated}$: Rated capacity [kWp]
- $G(t)$: Solar irradiance [kW/m²]
- $G_{STC} = 1.0$ kW/m² (Standard Test Conditions)

### 3.2 Temperature Coefficient for Panels

Solar panel efficiency decreases with temperature:

$$\eta_{panel}(T_{ambient}) = \eta_{nom} \cdot [1 + \gamma \cdot (T_{ambient} - T_{STC})]$$

Where:
- $\eta_{nom} = 0.20$ (20% nominal efficiency)
- $\gamma = -0.004$ /°C (temperature coefficient)
- $T_{STC} = 25°C$

**Implementation:**
```
temp_correction = max(0.5, 1 + (-0.004) * (T_ambient - 25))
```

### 3.3 System and Degradation Factors

**System Efficiency:** $\eta_{system} = 0.85$ (inverters, cables, dust, etc.)

**Annual Degradation:** $\eta_{degradation}(t) = (1 - r_{deg})^{t_{years}}$
- $r_{deg} = 0.005$ (0.5% per year)

---

## 4. Climate Data Generation

### 4.1 Temperature Modeling

**Seasonal Temperature Profile:**
$$T_{seasonal}(month) = \begin{cases}
T_{summer} & \text{if } month \in \{12, 1, 2, 3\} \\
T_{winter} & \text{if } month \in \{6, 7, 8, 9\} \\
T_{transition}(month) & \text{otherwise}
\end{cases}$$

**Transition Periods:**
$$T_{transition}(month) = \begin{cases}
T_{summer} \cdot (1-w) + T_{winter} \cdot w & \text{if } month \in \{4, 5\} \text{ (autumn)} \\
T_{winter} \cdot (1-w) + T_{summer} \cdot w & \text{if } month \in \{10, 11\} \text{ (spring)}
\end{cases}$$

Where $w = \frac{month - start_{month}}{3}$ (linear interpolation weight)

### 4.2 Daily Temperature Cycle

**Diurnal Variation:**
$$T(hour) = T_{seasonal} + \frac{\Delta T_{daily}}{2} \cdot \sin\left(\frac{2\pi(hour - 6)}{24}\right) + \epsilon$$

Where:
- $\Delta T_{daily}$: Daily temperature variation [°C]
- $\epsilon \sim \mathcal{N}(0, 1.0^2)$: Random variation

### 4.3 Solar Irradiance Generation

**Daily Solar Profile (6h to 18h):**
$$G(hour) = \begin{cases}
G_{peak} \cdot f_{solar}(hour) \cdot f_{seasonal} \cdot f_{clouds} & \text{if } 6 \leq hour \leq 18 \\
0 & \text{otherwise}
\end{cases}$$

**Solar Profile Function:**
$$f_{solar}(hour) = 4 \cdot \frac{hour - 6}{12} \cdot \left(1 - \frac{hour - 6}{12}\right)$$

This creates a parabolic curve with peak at noon (hour = 12).

**Seasonal Factor:**
$$f_{seasonal} = 0.8 + 0.4 \cdot \sin\left(\frac{2\pi(month - 6)}{12}\right)$$

**Cloud Factor:**
$$f_{clouds} \sim \max(0.1, \mathcal{N}(0.7, 0.3^2))$$

### 4.4 Regional Climate Parameters

| Region | $T_{summer}$ [°C] | $T_{winter}$ [°C] | $\Delta T_{daily}$ [°C] | $G_{peak}$ [kWh/m²/day] |
|--------|------------------|------------------|------------------------|-------------------------|
| Southeast SP | 24.5 | 16.8 | 8.0 | 5.5 |
| Northeast RN | 29.5 | 26.2 | 7.0 | 7.2 |
| South RS | 25.1 | 13.2 | 9.2 | 4.8 |
| Central-West MT | 27.8 | 22.1 | 12.0 | 6.2 |
| North AM | 27.5 | 26.8 | 4.5 | 5.0 |

---

## 5. Economic Modeling

### 5.1 Brazilian Electricity Tariff Structure

The time-of-use tariff structure follows Brazilian regulations:

$$\text{Price}(t) = \begin{cases}
P_{peak} & \text{if } 18 \leq h < 21 \text{ and weekday} \\
P_{intermediate} & \text{if } h \in \{17, 21\} \text{ and weekday} \\
P_{weekend} & \text{if weekend} \\
P_{off,peak} & \text{otherwise}
\end{cases}$$

**Typical Tariffs (R$/kWh, 2024):**
- $P_{peak} = 0.85$ R$/kWh
- $P_{intermediate} = 0.65$ R$/kWh  
- $P_{off,peak} = 0.45$ R$/kWh
- $P_{weekend} = 0.35$ R$/kWh

### 5.2 Energy Balance Equations

**Grid Import:**
$$P_{import}(t) = \max(0, P_{demand}(t) + P_{charge}(t) - P_{solar}(t) - P_{discharge}(t))$$

**Grid Export:**
$$P_{export}(t) = \max(0, P_{solar}(t) + P_{discharge}(t) - P_{demand}(t) - P_{charge}(t))$$

**Self-Consumption:**
$$P_{self}(t) = \min(P_{demand}(t), P_{solar}(t) + P_{discharge}(t))$$

### 5.3 Cost Calculations

**Hourly Costs:**
$$C_{import}(t) = P_{import}(t) \cdot \text{Price}(t)$$
$$R_{export}(t) = P_{export}(t) \cdot \text{Price}(t) \cdot f_{export}$$
$$S_{self}(t) = P_{self}(t) \cdot \text{Price}(t)$$

Where $f_{export} = 0.7$ (70% compensation for exported energy).

**Net Cost:**
$$C_{net}(t) = C_{import}(t) - R_{export}(t)$$

---

## 6. Reinforcement Learning Framework

### 6.1 Markov Decision Process Formulation

The battery optimization problem is formulated as an MDP: $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$

**State Space $\mathcal{S}$:**
The state vector contains 15 normalized features:
$$s_t = [s_1, s_2, ..., s_{15}]^T \in [0,1]^{15}$$

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | Battery SOC | $\in [0,1]$ |
| 1 | Battery Temperature | $T_b / 50$ |
| 2 | Battery Degradation | $D \in [0,1]$ |
| 3 | Ambient Temperature | $(T_{ambient} + 5) / 55$ |
| 4 | Solar Irradiance | $G \times 12$ |
| 5 | Industrial Demand | $P_{demand} / P_{peak}$ |
| 6 | Solar Generation | $P_{solar} / P_{rated}$ |
| 7 | Electricity Price | $(Price - 0.3) / 0.7$ |
| 8 | Hour of Day | $hour / 23$ |
| 9 | Day of Week | $weekday / 6$ |
| 10 | Month | $(month - 1) / 11$ |
| 11 | Can Charge | $\{0, 1\}$ |
| 12 | Can Discharge | $\{0, 1\}$ |
| 13 | Excess Solar | $\min(1, excess / 500)$ |
| 14 | Energy Deficit | $\min(1, deficit / P_{peak})$ |

**Action Space $\mathcal{A}$:**
Continuous action space: $a_t \in [-1, +1]$

Action interpretation:
$$\text{Action} = \begin{cases}
\text{Charge at } |a_t| \times P_{max,charge} & \text{if } a_t > 0.05 \\
\text{Discharge at } |a_t| \times P_{max,discharge} & \text{if } a_t < -0.05 \\
\text{Hold} & \text{otherwise}
\end{cases}$$

### 6.2 Reward Function Design

The multi-objective reward function balances economic optimization with battery health:

$$R(s_t, a_t) = w_1 R_{cost}(t) + w_2 R_{health}(t) + w_3 R_{stability}(t) + w_4 R_{temp}(t)$$

**Cost Reduction Reward:**
$$R_{cost}(t) = (C_{baseline}(t) - C_{actual}(t)) \times 10$$

Where $C_{baseline}$ is the cost without battery (direct grid consumption).

**Battery Health Reward:**
$$R_{health}(t) = -\begin{cases}
|\Delta T_{optimal}| \times 0.05 & \text{if } T_{ambient} \notin [T_{opt,min}, T_{opt,max}] \\
(1 - D(t)) \times 10 & \text{degradation penalty} \\
0 & \text{otherwise}
\end{cases}$$

**Grid Stability Reward:**
$$R_{stability}(t) = -\begin{cases}
P_{import}(t) \times 0.01 & \text{if peak hours} \\
0 & \text{otherwise}
\end{cases}$$

**Temperature Penalty:**
$$R_{temp}(t) = -\begin{cases}
50 & \text{if } T_{ambient} > T_{critical} \\
0 & \text{otherwise}
\end{cases}$$

**Default Weights:**
$w_1 = 1.0$, $w_2 = 0.3$, $w_3 = 0.2$, $w_4 = 0.5$

---

## 7. Optimization Objectives

### 7.1 Primary Objective: Cost Minimization

**Daily Cost Function:**
$$J_{cost} = \sum_{t=0}^{T-1} C_{net}(t) \cdot \Delta t$$

**Constraints:**
- $SOC_{min} \leq SOC(t) \leq SOC_{max}$
- $|P_{charge}(t)| \leq P_{max,charge}$
- $|P_{discharge}(t)| \leq P_{max,discharge}$
- $T_b(t) \leq T_{critical}$

### 7.2 Secondary Objective: Battery Preservation

**Degradation Minimization:**
$$J_{degradation} = \int_0^T f_{degradation}(T_b(t), SOC(t), P(t)) dt$$

**Thermal Stress Function:**
$$f_{degradation}(T_b, SOC, P) = \begin{cases}
\alpha_{temp} \cdot (T_b - T_{opt})^2 & \text{thermal stress} \\
\alpha_{cyc} \cdot |P| & \text{cycling stress} \\
\alpha_{soc} \cdot |SOC - 0.5| & \text{SOC stress}
\end{cases}$$

### 7.3 Multi-Objective Optimization

**Pareto Optimization:**
$$\min_{policy} (J_{cost}, J_{degradation})$$

The RL agent learns to navigate the Pareto frontier based on the weighted reward function.

---

## 8. Implementation Details

### 8.1 Numerical Integration

**Discrete Time Steps:** $\Delta t = 1$ hour

**State Update Equations:**
```python
# SOC update
energy_change = (P_charge * eta_charge - P_discharge / eta_discharge) * dt
SOC_new = SOC + energy_change / C_nominal

# Temperature update  
heat_loss = P_charge * (1 - eta_charge) + P_discharge * (1 - eta_discharge)
T_battery_new = T_ambient + heat_loss * dt / k_cooling

# Degradation update (daily)
if step % 24 == 0:
    cycles += 0.5  # Half cycle per day estimate
    degradation = max(0.7, 1.0 - cycles / cycle_life)
```

### 8.2 Stability and Convergence

**Numerical Stability:**
- All state variables are normalized to [0,1] range
- Actions are clipped to valid ranges
- Reward scaling prevents explosive gradients

**RL Training Stability:**
- Experience replay buffer
- Target network updates
- Gradient clipping: $\|\nabla\| \leq 0.5$
- Learning rate scheduling: $\alpha(t) = \alpha_0 \cdot 0.995^{epoch}$

### 8.3 Validation Methods

**Physical Constraints Validation:**
- Energy conservation: $\sum E_{in} = \sum E_{out} + \Delta E_{stored}$
- Power limits: $|P(t)| \leq P_{max}$
- SOC bounds: $SOC \in [SOC_{min}, SOC_{max}]$

**Model Verification:**
- Unit tests for each mathematical function
- Integration tests for system behavior
- Benchmarking against analytical solutions

---

## References

1. **Battery Modeling:**
   - Plett, G. L. (2015). "Battery Management Systems, Volume I: Battery Modeling"
   - Newman, J., & Thomas-Alyea, K. E. (2012). "Electrochemical Systems"

2. **Solar Modeling:**
   - Duffie, J. A., & Beckman, W. A. (2013). "Solar Engineering of Thermal Processes"
   - Green, M. A. (1982). "Solar Cells: Operating Principles, Technology and System Applications"

3. **Reinforcement Learning:**
   - Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"
   - Lillicrap, T. P., et al. (2015). "Continuous control with deep reinforcement learning"

4. **Brazilian Energy System:**
   - EPE - Empresa de Pesquisa Energética (2024). "Anuário Estatístico de Energia Elétrica"
   - ANEEL - Agência Nacional de Energia Elétrica (2024). "Tarifas de Energia Elétrica"

---

**Note:** This mathematical framework serves as the theoretical foundation for all implementations in the Battery Thermal RL system. Each equation has been validated through unit tests and empirical observations to ensure physical consistency and numerical stability.