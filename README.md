# proj-BRA-GER

Synthetic energy profile simulator for Brazil and Germany, with industrial consumption analysis and renewable generation.

## Project Structure (Short)

```
proj-BRA-GER/
├── src/        # Main source code
├── data/       # Real and synthetic data
├── results/    # Results and reports
├── models/     # Trained models
├── logs/       # Execution and training logs
├── tests/      # Automated tests
├── tmp/        # Temporary files
├── docs/       # Documentation
├── .venv/      # Python virtual environment
├── README.md   # Main documentation
├── requirements.txt # Project dependencies
└── LICENSE     # Project license
```

## Project Structure

```
proj-BRA-GER/
├── src/                            # Main source code
│   ├── core/                       # Core simulation logic and models
│   │   ├── dess_system.py
│   │   ├── energy_profile_config.py
│   │   ├── energy_profile_generator.py
│   │   ├── evaluate.py
│   │   ├── rl_dess_env.py
│   │   ├── rl_env.py
│   │   ├── synthetic_data_generator.py
│   │   └── train.py
│   └── utils/                      # Utility functions and visualization
│       └── plot.py
├── data/                           # Input and output data
│   ├── real/                       # Collected real-world data
│   │   ├── Historico_do_Preco_Horario_*.xlsx
│   │   ├── Historico_do_Preco_Medio_Mensal_*.xls
│   │   ├── Historico_do_Preco_Medio_Semanal_*.json/xls
│   │   └── price_analysis.pdf
│   └── synthetic/                  # Synthetically generated profiles and plots
│       ├── *.csv, *.json           # Synthetic data files
│       └── *.png                   # Plots and visualizations
├── results/                        # Evaluation results and reports
│   ├── *.csv
│   └── *.png
├── models/                         # Trained models and checkpoints
│   └── *.zip
├── logs/                           # Training and evaluation logs
│   └── PPO_*/
│       └── events.out.tfevents.*
├── tests/                          # Unit and integration tests
├── tmp/                            # Temporary files and scratch data (ignored)
├── docs/                           # Documentation and project design
│   ├── Proposal PIPC CAPES DFG Call 33 2023 Final version (3).pdf
│   └── synthetic_data_design.md
├── .venv/                          # Python virtual environment (optional, ignored)
├── README.md                       # Main project documentation (this file)
├── requirements.txt                # Python package dependencies
└── LICENSE                         # Project license
```

## How to Run

### 1. Install dependencies with [uv](https://github.com/astral-sh/uv) (recommended)

If you don't have uv installed:
```
curl -Ls https://astral.sh/uv/install.sh | sh
```

Then, install the project dependencies:
```
uv pip install -r requirements.txt
```

### 2. Or install dependencies with pip
```
pip install -r requirements.txt
```

### 3. Run the simulator via CLI
```
python src/app_cli.py
```

## About

- Generation of synthetic energy profiles (solar, wind, hydropower, etc).
- Industrial consumption analysis and automatic visualization.
- Data and plots are saved in `data/synthetic/`.

## Visualizing Training with TensorBoard

To monitor and visualize the training process, you can use TensorBoard. If you trained your model with Stable Baselines3 and set the `tensorboard_log` parameter, logs will be saved (e.g., in `ppo_dess_tensorboard/`).

1. Install TensorBoard (if not already installed):
   ```
   pip install tensorboard
   ```

2. Run TensorBoard pointing to your log directory:
   ```
   tensorboard --logdir log/
   ```

3. Open your browser and go to:
   ```
   http://localhost:6006
   ```

You will be able to visualize training metrics, rewards, losses, and more in real time.

---
