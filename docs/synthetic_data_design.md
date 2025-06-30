# Synthetic Data Generation Design Document

**Version:** 1.0
**Author:** Project BRA-GER Team
**Date:** [Current Date]

## 1. Introduction

This document outlines the design and methodology for the synthetic data generator, a core component of the `dess_simulation` project. The primary goal of this module is to produce realistic, high-resolution time-series data for various energy system components (solar, wind, grid load, and pricing) across different geographical and economic contexts, specifically focusing on regional profiles in Brazil and Germany.

This synthetic data is crucial for:
-   Prototyping and testing the simulation engine.
-   Developing and evaluating control strategies (both heuristic and Reinforcement Learning-based).
-   Creating a diverse set of scenarios to ensure the robustness of our models.
-   Providing a reproducible baseline for experiments without relying on often-inaccessible or incomplete real-world data.

The design emphasizes modularity, configurability, and physical plausibility.

## 2. System Architecture

The data generation process is managed by two primary modules within the `src/core` directory:

1.  **`energy_profiles.py`**: This module acts as a configuration database. It defines data structures (`SolarConfig`, `WindConfig`, `GridConfig`) and contains pre-configured, realistic parameter sets for various regions in Brazil and Germany. This separation of configuration from logic allows for easy expansion to new regions.

2.  **`synthetic_data_generator.py`**: This module contains the `SyntheticDataGenerator` class, which ingests configuration objects from `energy_profiles.py` and applies mathematical and stochastic models to generate the final time-series data.

The generation workflow is as follows:
1.  A user or script selects a target profile (e.g., `GERMANY_NORTH` or a predefined scenario like `brazil_northeast_1week`) via the Command-Line Interface (`src/app_cli.py`).
2.  The `CountryProfileManager` from `energy_profiles.py` loads the corresponding `SolarConfig`, `WindConfig`, and `GridConfig` objects.
3.  These configuration objects are passed to an instance of `SyntheticDataGenerator`.
4.  The generator executes its methods (`generate_solar_profile`, etc.) using the specific parameters from the configuration.
5.  The final data is compiled into a pandas DataFrame and saved to a CSV file in the `data/synthetic/` directory.

## 3. Data Models and Methodologies

### 3.1. Solar Power Generation Model

The solar generation model calculates power output (kW) based on the following factors:

-   **Solar Position:** Sun elevation is calculated using a standard solar position algorithm based on latitude, day of year, and time of day. This forms the basis for clear-sky irradiance.
-   **Seasonal Variation:** A cosine function adjusts the peak irradiance throughout the year to account for seasonal changes in solar intensity, with parameters adjusted for Northern and Southern hemispheres.
-   **Cloud Cover:** This is modeled stochastically. Each day has a defined probability (`cloud_probability`) of experiencing a significant cloud event. When an event is triggered, its duration and the percentage of irradiance reduction are randomized within configured bounds, simulating the sharp and sustained drops seen in real-world data.
-   **Physical and System Losses:** The model accounts for:
    -   `panel_efficiency`: The conversion efficiency of the panels.
    -   `system_losses`: Inverter, cabling, and transformer losses.
    -   `dust_factor`/`soiling_rate`: Power loss due to accumulation of dust, with a periodic "reset" to simulate cleaning (`cleaning_frequency_days`).
    -   `annual_degradation`: A small, linear reduction in output over the simulation's duration.

### 3.2. Wind Power Generation Model

The wind power model simulates turbine output (kW) based on a synthetic wind speed time-series:

-   **Wind Speed Distribution:** The underlying wind speed is modeled using a **Weibull distribution**, which is the industry standard for characterizing wind regimes. The `weibull_k` (shape) and `weibull_c` (scale) parameters are defined for each region.
-   **Power Curve:** The conversion from wind speed to power follows a standard turbine power curve:
    1.  Zero power below `cut_in_speed`.
    2.  A cubic power increase between `cut_in_speed` and `rated_speed`.
    3.  Constant `turbine_power_rating` between `rated_speed` and `cut_out_speed`.
    4.  Zero power above `cut_out_speed` for safety.
-   **Availability and Losses:**
    -   `availability_factor`: Accounts for general uptime.
    -   `wake_losses`: A fixed percentage reduction to simulate aerodynamic interference in wind farms.
    -   `maintenance_probability`: A stochastic model for scheduling maintenance downtime.

### 3.3. Grid Model (Load and Pricing)

The grid model generates two primary time-series: the manufacturing/consumer load and the electricity tariffs.

-   **Load Profile:** The load is modeled as a superposition of several patterns:
    -   **Base Load:** A constant minimum consumption (`base_consumption`).
    -   **Daily Pattern:** A sinusoidal wave to simulate lower consumption at night and higher during the day.
    -   **Peak Hours:** A multiplicative factor (`peak_consumption_factor`) is applied during evening peak hours.
    -   **Weekly Pattern:** A `weekend_factor` reduces consumption on Saturdays and Sundays.
-   **Tariff Profile:** Prices are modeled to simulate Time-of-Use (TOU) tariffs:
    -   `base_tariff`: The standard price.
    -   `peak_multiplier`: Increases the price during peak hours.
    -   `off_peak_multiplier`: Decreases the price during late-night/early-morning hours.
    -   `feed_in_tariff`: A separate, often lower, price for exporting energy to the grid.
-   **Grid Reliability:**
    -   `outage_probability`: A stochastic model introduces grid outages, setting the load to zero for a randomized duration. This is critical for testing the resilience of the local energy system.

## 4. Data Sources and Parameter Justification

The parameters within `energy_profiles.py` were established by consulting public databases, national energy reports, and technical literature to ensure they reflect realistic conditions for each specified region.

-   **Solar Data (Irradiance, Latitude):**
    -   **Source:** Global Solar Atlas, a free, web-based application developed by the World Bank.
    -   **Justification:** Provides long-term average data on solar resource potential (GHI, DNI) and optimal panel tilt angles for locations worldwide, which informed latitude-specific parameters and seasonal variations.
    -   **Link:** [https://globalsolaratlas.info/](https://globalsolaratlas.info/)

-   **Wind Data (Wind Speed, Weibull Parameters):**
    -   **Source:** Global Wind Atlas, a companion tool to the Global Solar Atlas, also by the World Bank in partnership with DTU.
    -   **Justification:** Used to determine average wind speeds at different hub heights and typical Weibull distribution parameters for the selected regions in Brazil and Germany.
    -   **Link:** [https://globalwindatlas.info/](https://globalwindatlas.info/)

-   **Grid Tariffs and Consumption Patterns (Brazil):**
    -   **Source:** ANEEL (Agência Nacional de Energia Elétrica) - Brazilian Electricity Regulatory Agency. Reports and public data on tariff structures (Bandeiras Tarifárias, Tarifa Branca) and national consumption data from EPE (Empresa de Pesquisa Energética).
    -   **Justification:** ANEEL's data provided realistic base tariffs, peak hour definitions, and multipliers. EPE's "Balanço Energético Nacional (BEN)" reports helped characterize industrial vs. residential load patterns.
    -   **Link (ANEEL):** [https://www.gov.br/aneel/](https://www.gov.br/aneel/)
    -   **Link (EPE):** [https://www.epe.gov.br/](https://www.epe.gov.br/)

-   **Grid Tariffs and Consumption Patterns (Germany):**
    -   **Source:** BDEW (Bundesverband der Energie- und Wasserwirtschaft) and Fraunhofer ISE (Institute for Solar Energy Systems).
    -   **Justification:** BDEW publishes yearly analyses of electricity prices for households and industry. Fraunhofer's "Energy Charts" provide extensive real-time and historical data on generation, consumption, and prices in Germany, which informed parameters for `GERMANY_*` profiles.
    -   **Link (BDEW):** [https://www.bdew.de/](https://www.bdew.de/)
    -   **Link (Fraunhofer ISE):** [https://www.energy-charts.info/](https://www.energy-charts.info/)

-   **Technical Parameters (Turbine/Panel Specs, Losses):**
    -   **Source:** Technical datasheets from major manufacturers (e.g., Vestas, Siemens Gamesa for wind; Jinko Solar, Trina Solar for solar) and NREL (National Renewable Energy Laboratory) publications.
    -   **Justification:** Parameters like `cut_in_speed`, `panel_efficiency`, `temperature_coefficient`, and `system_losses` were derived from typical values found in modern, commercially available technology.

By grounding our synthetic data in these sources, we aim to create profiles that are not just randomly generated but are representative of the operational realities and economic conditions in each target region.