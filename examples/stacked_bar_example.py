#!/usr/bin/env python3
"""
Exemplo de uso da função plot_stacked_bar_chart
Demonstra como criar gráficos de barras empilhadas no estilo solicitado.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from utils.plot import plot_stacked_bar_chart

def example_monthly_generation():
    """Exemplo de geração mensal de energia renovável no Brasil"""
    
    # Definir meses
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Dados mensais fixos (GWh) - exemplo baseado em dados reais do Brasil
    monthly_data = {
        "Hydropower": np.full(12, 33200),
        "Wind": np.full(12, 9000),
        "Solar PV": np.full(12, 5900),
        "Biomass": np.full(12, 5100),
        "Biogas": np.full(12, 300)
    }
    
    # Criar gráfico
    plot_stacked_bar_chart(
        data_dict=monthly_data,
        title="Monthly Renewable Energy Generation in Brazil (Estimated 2024)",
        ylabel="Generation (GWh)",
        xlabel="Month",
        save_path="monthly_renewable_generation_brazil.png"
    )

def example_daily_consumption():
    """Exemplo de consumo diário por fonte"""
    
    # Definir dias da semana
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # Dados de consumo diário (kWh) - exemplo industrial
    daily_data = {
        "Solar": [1200, 1150, 1180, 1220, 1190, 1100, 1050],
        "Wind": [800, 850, 900, 750, 820, 780, 720],
        "Grid": [600, 650, 580, 700, 620, 680, 750],
        "Battery": [200, 180, 220, 190, 210, 240, 280]
    }
    
    # Criar gráfico
    plot_stacked_bar_chart(
        data_dict=daily_data,
        title="Daily Energy Consumption by Source - Industrial Facility",
        ylabel="Consumption (kWh)",
        xlabel="Day of Week",
        save_path="daily_consumption_by_source.png"
    )

def example_hourly_generation():
    """Exemplo de geração horária"""
    
    # Definir horas do dia
    hours = [f"{h:02d}:00" for h in range(24)]
    
    # Dados de geração horária (kW) - exemplo solar + wind
    hourly_data = {
        "Solar": [0, 0, 0, 0, 0, 0, 50, 200, 400, 600, 800, 1000, 
                 1000, 800, 600, 400, 200, 50, 0, 0, 0, 0, 0, 0],
        "Wind": [300, 280, 320, 350, 380, 400, 420, 450, 480, 500, 
                520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 
                720, 740, 760, 780]
    }
    
    # Criar gráfico
    plot_stacked_bar_chart(
        data_dict=hourly_data,
        title="Hourly Renewable Energy Generation",
        ylabel="Generation (kW)",
        xlabel="Hour of Day",
        save_path="hourly_generation.png"
    )

if __name__ == "__main__":
    print("Criando exemplos de gráficos de barras empilhadas...")
    
    # Criar diretório de exemplos se não existir
    os.makedirs("examples", exist_ok=True)
    
    # Executar exemplos
    example_monthly_generation()
    example_daily_consumption()
    example_hourly_generation()
    
    print("Exemplos criados com sucesso!")
    print("Arquivos gerados:")
    print("- monthly_renewable_generation_brazil.png")
    print("- daily_consumption_by_source.png")
    print("- hourly_generation.png") 