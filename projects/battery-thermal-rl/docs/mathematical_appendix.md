# Mathematical Appendix

**Supporting Mathematical Derivations and Proofs for Battery Thermal RL**

This appendix provides detailed mathematical derivations, proofs, and additional theoretical background for the Battery Thermal RL system.

## A. Battery Electrochemical Fundamentals

### A.1 Electrochemical Kinetics

The fundamental relationship between battery voltage and state of charge is given by the Nernst equation:

$$V_{ocv}(SOC) = E^0 + \frac{RT}{nF} \ln\left(\frac{a_{oxidized}}{a_{reduced}}\right)$$

Where:
- $E^0$: Standard electrode potential [V]
- $R$: Universal gas constant [8.314 J/(mol·K)]
- $T$: Temperature [K]
- $n$: Number of electrons transferred
- $F$: Faraday constant [96,485 C/mol]
- $a$: Activities of oxidized/reduced species

**For Li-ion batteries**, this simplifies to:
$$V_{ocv}(SOC) = V_{min} + (V_{max} - V_{min}) \cdot SOC + RT_{correction}$$

### A.2 Internal Resistance and Heat Generation

**Ohmic Heat Generation:**
$$Q_{ohmic} = I^2 \cdot R_{internal}(SOC, T)$$

**Polarization Heat:**
$$Q_{polarization} = I \cdot \eta_{activation} + I \cdot \eta_{concentration}$$

**Total Heat Generation:**
$$Q_{total} = Q_{ohmic} + Q_{polarization} = I \cdot (V_{terminal} - V_{ocv})$$

### A.3 Thermal Capacity and Heat Transfer

**Battery Thermal Capacity:**
$$C_{th} = m_{cell} \cdot c_p \cdot n_{cells}$$

Where:
- $m_{cell}$: Mass per cell [kg]
- $c_p$: Specific heat capacity [J/(kg·K)]
- $n_{cells}$: Number of cells

**Heat Transfer Coefficient:**
$$h = \frac{k \cdot Nu}{L_{characteristic}}$$

Where $Nu$ is the Nusselt number for the specific geometry and flow conditions.

## B. Climate Modeling Derivations

### B.1 Solar Position and Irradiance

**Solar Declination Angle:**
$$\delta = 23.45° \cdot \sin\left(\frac{360°(284 + n)}{365}\right)$$

Where $n$ is the day of year.

**Hour Angle:**
$$\omega = 15° \cdot (t_{solar} - 12)$$

**Solar Elevation Angle:**
$$\sin(\alpha) = \sin(\phi)\sin(\delta) + \cos(\phi)\cos(\delta)\cos(\omega)$$

**Direct Normal Irradiance:**
$$DNI = I_{extraterrestrial} \cdot \tau^{AM}$$

Where:
- $I_{extraterrestrial}$: Extraterrestrial irradiance
- $\tau$: Atmospheric transmittance
- $AM$: Air mass

### B.2 Temperature Correlations

**Diurnal Temperature Model (detailed):**
$$T(t) = T_{avg} + \frac{T_{range}}{2} \cos\left(\frac{2\pi(t - t_{max})}{24}\right)$$

Where $t_{max}$ is typically 14:00-15:00 (2-3 PM).

**Seasonal Temperature Variation:**
$$T_{seasonal}(d) = T_{annual\_avg} + A \cos\left(\frac{2\pi(d - d_{max})}{365.25}\right)$$

Where:
- $A$: Annual temperature amplitude
- $d_{max}$: Day of maximum temperature (typically day 200-220)

## C. Industrial Load Modeling

### C.1 Stochastic Load Components

**Autoregressive Load Model:**
$$P(t) = \mu + \phi_1 P(t-1) + \phi_2 P(t-2) + \epsilon(t)$$

Where:
- $\mu$: Mean load level
- $\phi_1, \phi_2$: Autoregressive coefficients
- $\epsilon(t) \sim \mathcal{N}(0, \sigma^2)$: White noise

**Load Duration Curve:**
The probability that load exceeds level $P$ is:
$$Pr(P_{load} > P) = \int_P^{P_{max}} f(x) dx$$

Where $f(x)$ is the load probability density function.

### C.2 Industrial Process Modeling

**Thermal Load (for food processing):**
$$P_{thermal}(t) = \frac{m \cdot c_p \cdot \Delta T}{\eta_{thermal} \cdot \Delta t}$$

**Motor Load (for manufacturing):**
$$P_{motor}(t) = \frac{P_{mechanical}}{\eta_{motor} \cdot \eta_{drive}} + P_{no\_load}$$

## D. Economic Optimization Theory

### D.1 Dynamic Programming Formulation

**Bellman Equation:**
$$V(s) = \min_a \left[ C(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$$

For battery optimization:
- State: $s = (SOC, T, t, P_{demand}, P_{solar})$
- Action: $a = P_{battery}$ (charge/discharge power)
- Cost: $C(s,a) = P_{grid} \cdot Price(t) \cdot \Delta t$

**Optimality Conditions (Karush-Kuhn-Tucker):**
$$\nabla_a L = \frac{\partial C}{\partial a} + \lambda_1 \frac{\partial g_1}{\partial a} + \lambda_2 \frac{\partial g_2}{\partial a} = 0$$

Where $g_1, g_2$ are constraint functions (SOC limits, power limits).

### D.2 Net Present Value Analysis

**NPV of Battery Investment:**
$$NPV = -C_{investment} + \sum_{t=1}^{N} \frac{CF_t}{(1+r)^t}$$

Where:
- $CF_t$: Cash flow in year $t$ (energy cost savings)
- $r$: Discount rate
- $N$: Project lifetime

**Levelized Cost of Storage (LCOS):**
$$LCOS = \frac{C_{capital} + \sum_{t=1}^{N} \frac{C_{O\&M,t}}{(1+r)^t}}{\sum_{t=1}^{N} \frac{E_{discharge,t}}{(1+r)^t}}$$

## E. Reinforcement Learning Theory

### E.1 Policy Gradient Methods

**REINFORCE Algorithm:**
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot R(s,a)]$$

**Actor-Critic Methods:**
$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)]$$

Where $A(s,a) = Q(s,a) - V(s)$ is the advantage function.

### E.2 Deep Deterministic Policy Gradient (DDPG)

**Critic Loss:**
$$L_{critic} = \mathbb{E}[(r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s')) - Q_\phi(s,a))^2]$$

**Actor Loss:**
$$L_{actor} = -\mathbb{E}[Q_\phi(s, \mu_\theta(s))]$$

**Target Network Updates:**
$$\phi' \leftarrow \tau \phi + (1-\tau) \phi'$$
$$\theta' \leftarrow \tau \theta + (1-\tau) \theta'$$

### E.3 Proximal Policy Optimization (PPO)

**Clipped Surrogate Objective:**
$$L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$$

Where:
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

## F. Stability Analysis

### F.1 Lyapunov Stability for Battery Systems

**Lyapunov Function Candidate:**
$$V(x) = \frac{1}{2}(SOC - SOC_{ref})^2 + \frac{1}{2}(T - T_{ref})^2$$

**Stability Condition:**
$$\dot{V}(x) = \frac{\partial V}{\partial SOC} \dot{SOC} + \frac{\partial V}{\partial T} \dot{T} < 0$$

This ensures that the system converges to the reference point.

### F.2 Convergence Guarantees for RL

**Robbins-Monro Conditions for Learning Rate:**
$$\sum_{t=1}^{\infty} \alpha_t = \infty, \quad \sum_{t=1}^{\infty} \alpha_t^2 < \infty$$

**Contraction Mapping for Q-Learning:**
$$\|T_\pi Q - T_\pi Q'\|_\infty \leq \gamma \|Q - Q'\|_\infty$$

Where $T_\pi$ is the Bellman operator and $\gamma < 1$.

## G. Numerical Methods

### G.1 Runge-Kutta Integration

For solving the differential equation system:
$$\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t)$$

**Fourth-order Runge-Kutta:**
$$\mathbf{x}_{n+1} = \mathbf{x}_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Where:
$$k_1 = f(\mathbf{x}_n, t_n)$$
$$k_2 = f(\mathbf{x}_n + \frac{h}{2}k_1, t_n + \frac{h}{2})$$
$$k_3 = f(\mathbf{x}_n + \frac{h}{2}k_2, t_n + \frac{h}{2})$$
$$k_4 = f(\mathbf{x}_n + hk_3, t_n + h)$$

### G.2 Monte Carlo Methods

**Importance Sampling for Rare Events:**
$$\mathbb{E}_f[g(X)] = \mathbb{E}_h\left[\frac{f(X)}{h(X)} g(X)\right]$$

Where $h(X)$ is the importance sampling density.

**Variance Reduction:**
$$\text{Var}[g(X)] = \mathbb{E}[g(X)^2] - (\mathbb{E}[g(X)])^2$$

## H. Implementation Considerations

### H.1 Floating Point Precision

**Machine Epsilon:**
$$\epsilon_{machine} = 2^{-52} \approx 2.22 \times 10^{-16}$$

(double precision)

**Condition Number:**
$$\kappa(A) = \|A\| \|A^{-1}\|$$

For numerical stability, require $\kappa(A) < 10^{12}$.

### H.2 Optimization Tolerances

**Gradient Norm Tolerance:**
$$\|\nabla f(x)\| < \epsilon_{grad} = 10^{-6}$$

**Function Value Tolerance:**
$$|f(x_{k+1}) - f(x_k)| < \epsilon_{func} = 10^{-8}$$

**Parameter Tolerance:**
$$\|x_{k+1} - x_k\| < \epsilon_{param} = 10^{-6}$$

## I. Model Validation Metrics

### I.1 Statistical Validation

**Mean Absolute Percentage Error (MAPE):**
$$MAPE = \frac{1}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$$

**Root Mean Square Error (RMSE):**
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Coefficient of Determination:**
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

### I.2 Physical Validation

**Energy Conservation Check:**
$$\left|\sum E_{in} - \sum E_{out} - \Delta E_{stored}\right| < \epsilon_{energy}$$

**Power Balance Validation:**
$$P_{generation} + P_{battery\_discharge} = P_{demand} + P_{battery\_charge} + P_{losses}$$

**Thermodynamic Consistency:**
$$\Delta S_{universe} = \Delta S_{system} + \Delta S_{surroundings} \geq 0$$

## J. Sensitivity Analysis

### J.1 Parameter Sensitivity

**Sensitivity Coefficient:**
$$S_{p_i} = \frac{\partial y}{\partial p_i} \cdot \frac{p_i}{y}$$

**Morris Method (Elementary Effects):**
$$EE_i = \frac{y(x + \Delta e_i) - y(x)}{\Delta}$$

Where $e_i$ is the unit vector in direction $i$.

### J.2 Uncertainty Propagation

**Linear Approximation:**
$$\sigma_y^2 \approx \sum_{i=1}^{n} \left(\frac{\partial y}{\partial x_i}\right)^2 \sigma_{x_i}^2$$

**Monte Carlo Propagation:**
$$\hat{\sigma}_y^2 = \frac{1}{N-1} \sum_{j=1}^{N} (y_j - \bar{y})^2$$

---

**Note:** This appendix provides the detailed mathematical foundations underlying the simplified models implemented in the Battery Thermal RL system. For practical applications, the main tutorial document provides the directly applicable equations and implementations.