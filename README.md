# proj-BRA-GER

Synthetic energy profile simulator for Brazil and Germany, with industrial consumption analysis and renewable generation.


## Project Structure

```

proj-Energy-BR_DE/
├── data/                           # Input and output data
│   ├── real/                       # Collected real-world data
│   └── synthetic/                  # Synthetic profiles, plots, and contract files
├── docs/                           # Documentation and project design
├── logs/                           # Training and evaluation logs (TensorBoard, etc.)
│   └── PPO_*/                      # RL training runs (subfolders)
├── models/                         # Trained models and checkpoints
├── results/                        # Evaluation results and reports
│   └── price_350/                  # Example of grouped evaluation results
├── src/                            # Main source code
│   ├── app_cli.py                  # Main CLI entry point
│   ├── _app_ui.py                  # (UI helper, if used)
│   ├── core/                       # Core simulation logic and models
│   │   ├── dess_system.py
│   │   ├── energy_profile_config.py
│   │   ├── energy_profile_generator.py
│   │   ├── evaluate.py
│   │   ├── rl_dess_env.py
│   │   ├── synthetic_data_generator.py
│   │   ├── train.py
│   │   └── __init__.py
│   ├── utils/                      # Utility functions and visualization
│   │   ├── plot.py
│   │   └── __init__.py
│   └── __init__.py
├── tmp/                            # Temporary files and scratch data (ignored)
├── README.md                       # Main project documentation (this file)
├── requirements.txt                # Python package dependencies
└── LICENSE                         # Project license
```

## CLI Main Menu Options

When running `python src/app_cli.py`, the following options are available:

1. **Generate Contract Prices & Validation Plot**  
   Create contract price JSONs and validation plots for a selected year and region.
2. **Generate Full Industry Profile (Energy-based)**  
   Generate a full synthetic energy profile using the contract JSON from option 1.
3. **Train DESS Management Agent (RL)**  
   Train a reinforcement learning agent to manage the decentralized energy system.
4. **Evaluate Trained Agent**  
   Run evaluation and generate plots for a trained RL agent.
5. **Clean all __pycache__ folders**  
   Recursively remove all Python `__pycache__` folders from the project for maintenance.
0. **Exit**

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
