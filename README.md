# proj-BRA-GER

Simulador de perfis energéticos sintéticos para Brasil e Alemanha, com análise de consumo industrial e geração renovável.

## Estrutura do Projeto

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

## Como rodar

1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

2. Execute o simulador via CLI:
   ```
   python src/app_cli.py
   ```

## Sobre

- Geração de perfis sintéticos de energia (solar, eólica, hidrelétrica, etc).
- Análise de consumo industrial e visualização automática.
- Dados e gráficos salvos em `data/synthetic/`.

---

Sinta-se à vontade para adaptar ou expandir conforme o foco do seu projeto!