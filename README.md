proj-BRA-GER/
├── docs/                           # Design and architecture documentation
│   ├── architecture.md             # High-level system architecture
│   └── requirements.md             # Functional and non-functional requirements
├── data/                           # Raw input data
│   ├── synthetic/                  # Synthetically generated profiles and trajectories
│   │   ├── energy_profiles.csv     # Synthetic generation profiles (solar, wind, grid)
│   │   └── control_trajectories.json  # Synthetic control trajectories (actions)
│   └── real/                       # Collected real-world data
│       ├── solar_profile.csv       # Actual solar generation measurements
│       ├── wind_profile.csv        # Actual wind generation measurements
│       └── grid_prices.csv         # Public grid tariffs and consumption data
├── tmp/                            # Temporary files and scratch data (ignored)
├── notebooks/                      # Notebooks for exploratory analysis and prototyping
│   └── 01_initial_analysis.ipynb   # Profile analysis and initial simulations
├── .gitignore                      # Specifies untracked files and folders
├── src/                            # Main source code
│   ├── __init__.py                 # Marks Python package
│   ├── app_ui.py                   # Streamlit interface (future)
│   ├── app_cli.py                  # CLI for command-line simulations
│   ├── core/                       # Core simulation logic and models
│   │   ├── __init__.py
│   │   ├── simulation_engine.py    # Time-step based simulation engine
│   │   ├── energy_source_models.py # Energy source models: Solar, Wind, Grid
│   │   ├── battery_model.py        # Battery model (charging, discharging dynamics)
│   │   ├── hydrogen_models.py      # Models for electrolyzer, H2 storage, and fuel cell
│   │   ├── synthetic_data_generator.py  # Synthetic data generator for profiles and trajectories
│   │   └── rl_environment.py       # RL/IRL environment definition (Gymnasium)
│   └── utils/                      # Utility functions and visualization
│       ├── __init__.py
│       └── plot.py           # Plotting helper functions
├── tests/                          # Unit and integration tests
│   ├── __init__.py
│   ├── test_simulation_engine.py   # Tests for the simulation engine
│   ├── test_battery_model.py       # Tests for the battery model
│   ├── test_energy_sources.py      # Tests for the energy source models
│   ├── test_hydrogen_models.py     # Tests for the hydrogen-based models
│   └── test_synthetic_data.py      # Tests for the synthetic data generator
├── README.md                       # Main project documentation (this file)
├── requirements.txt                # Python package dependencies
└── LICENSE                         # Project license