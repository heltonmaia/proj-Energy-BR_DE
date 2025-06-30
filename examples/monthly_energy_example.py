#!/usr/bin/env python3
"""
Exemplo de gráficos mensais de geração e consumo de energia
Demonstra como os dados são agrupados por mês e exibidos em gráficos de barras empilhadas.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils.plot import plot_dual_energy_figures
from core.energy_profile_config import Country

def create_sample_monthly_data():
    """Cria dados de exemplo para 12 meses"""
    
    # Criar timestamps para 12 meses (15 minutos de intervalo)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31, 23, 45)
    
    timestamps = []
    current = start_date
    while current <= end_date:
        timestamps.append(current)
        current += timedelta(minutes=15)
    
    # Criar DataFrame
    df = pd.DataFrame({'timestamp': timestamps})
    
    # Adicionar dados de geração (variação sazonal)
    months = df['timestamp'].dt.month
    
    # Solar: variação sazonal (mais no verão)
    solar_factor = 1 + 0.3 * np.sin(2 * np.pi * (months - 6) / 12)
    df['solar_power_kw'] = 100 * solar_factor + np.random.normal(0, 10, len(df))
    df['solar_power_kw'] = df['solar_power_kw'].clip(lower=0)
    
    # Wind: variação sazonal (mais no inverno)
    wind_factor = 1 + 0.2 * np.sin(2 * np.pi * (months - 12) / 12)
    df['wind_power_kw'] = 80 * wind_factor + np.random.normal(0, 15, len(df))
    df['wind_power_kw'] = df['wind_power_kw'].clip(lower=0)
    
    # Grid: complemento para manter consumo constante
    df['grid_power_kw'] = 50 + np.random.normal(0, 5, len(df))
    df['grid_power_kw'] = df['grid_power_kw'].clip(lower=0)
    
    # Consumo industrial (constante com pequena variação)
    base_consumption = 200
    df['industrial_consumption_kw'] = base_consumption + np.random.normal(0, 10, len(df))
    
    # Distribuir consumo pelas fontes (baseado em shares configurados)
    solar_share = 0.4
    wind_share = 0.3
    grid_share = 0.3
    
    df['solar_used_kw'] = df['industrial_consumption_kw'] * solar_share
    df['wind_used_kw'] = df['industrial_consumption_kw'] * wind_share
    df['grid_used_kw'] = df['industrial_consumption_kw'] * grid_share
    
    return df

def main():
    print("Criando exemplo de gráficos mensais de energia...")
    
    # Criar dados de exemplo
    df = create_sample_monthly_data()
    
    print(f"Dados criados: {len(df)} pontos de dados")
    print(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print(f"Total de meses: {df['timestamp'].dt.to_period('M').nunique()}")
    
    # Criar gráficos
    plot_dual_energy_figures(
        df=df,
        country=Country.BRAZIL,
        save_path="examples/monthly_energy_analysis.png"
    )
    
    print("\nGráfico salvo como: examples/monthly_energy_analysis.png")
    print("\nO gráfico mostra:")
    print("- Painel superior esquerdo: Geração temporal por fonte")
    print("- Painel superior direito: Geração mensal empilhada por fonte")
    print("- Painel inferior esquerdo: Consumo temporal por fonte")
    print("- Painel inferior direito: Consumo mensal empilhado por fonte")

if __name__ == "__main__":
    main() 