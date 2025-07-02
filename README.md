# proj-BRA-GER

Synthetic energy profile simulator for Brazil and Germany, with industrial consumption analysis and renewable generation.

## Project Structure

```
proj-BRA-GER/
├── docs/                           # Design and architecture documentation
│   ├── architecture.md             # High-level system architecture
│   └── requirements.md             # Functional and non-functional requirements
├── data/                           # Raw input data
│   ├── synthetic/                  # Synthetically generated profiles and trajectories
│   └── real/                       # Collected real-world data
├── tmp/                            # Temporary files and scratch data (ignored)
├── notebooks/                      # Notebooks for exploratory analysis and prototyping
├── src/                            # Main source code
│   ├── core/                       # Core simulation logic and models
│   └── utils/                      # Utility functions and visualization
├── tests/                          # Unit and integration tests
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

---

Feel free to adapt or expand according to your project's focus!