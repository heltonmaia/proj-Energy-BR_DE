# utils/plot.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import matplotlib.dates as mdates
import locale
import matplotlib as mpl

# Configura o locale para formatação de moeda brasileira (BRL)
try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except locale.Error:
    print("Warning: Brazilian locale 'pt_BR.UTF-8' not found. Using default for currency formatting.")

def plot_real_pld(pattern_loader, region, year, save_path=None):
    df_real = pattern_loader.df[pattern_loader.df['year'] == year]
    if df_real.empty:
        print(f"Warning: No historical data found for year {year}. Skipping real PLD plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(df_real.index, df_real[region], label=f'Real PLD (BRL/MWh)', color='darkblue')

    ax.set_ylabel('PLD (BRL/MWh)')
    ax.set_xlabel('Date')
    ax.set_title(f'Real Historical PLD for {region} - {year}', fontsize=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_energy_profiles(df, title, save_path=None):
    df_plot = df.copy()
    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
    df_plot['total_generation'] = df_plot['solar_generation_kw'] + df_plot['wind_generation_kw']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))

    unique_days = df_plot['timestamp'].dt.normalize().unique()

    for day in unique_days:
        ax.axvspan(day + pd.Timedelta(hours=18), day + pd.Timedelta(hours=30),
                   facecolor='#EAF4FF', zorder=0)

    # Encontrar todos os dias únicos
    all_days = df_plot['timestamp'].dt.normalize().unique()
    for day in all_days:
        weekday = pd.Timestamp(day).weekday()
        if weekday == 5:  # Sábado
            ax.axvspan(day, day + pd.Timedelta(days=1), facecolor='#FFF9E5', zorder=1)
        elif weekday == 6:  # Domingo
            ax.axvspan(day, day + pd.Timedelta(days=1), facecolor='#FFF9E5', zorder=1)

    ax.fill([], [], color='#EAF4FF', label='Night Time')
    ax.fill([], [], color='#FFF9E5', label='Weekend')

    ax.fill_between(df_plot['timestamp'], df_plot['industrial_consumption_kw'], df_plot['total_generation'],
                    where=(df_plot['total_generation'] > df_plot['industrial_consumption_kw']),
                    color='lightgreen', alpha=0.7, interpolate=True, label='Energy Surplus (Self-sufficient)',
                    zorder=2)

    ax.plot(df_plot['timestamp'], df_plot['solar_generation_kw'], label='Solar Generation (kW)',
            color='orange', linewidth=2, zorder=3)
    ax.plot(df_plot['timestamp'], df_plot['wind_generation_kw'], label='Wind Generation (kW)',
            color='skyblue', linewidth=1.5, zorder=3)

    ax.plot(df_plot['timestamp'], df_plot['grid_used_kw'], label='Grid Power Used (kW)',
            color='red', linestyle='--', alpha=0.9, linewidth=2.5, zorder=4)

    ax.plot(df_plot['timestamp'], df_plot['industrial_consumption_kw'], label='Industrial Consumption (kW)',
            color='black', linewidth=2.5, zorder=5)

    ax.set_ylabel('Power (kW)', fontsize=14)
    ax.set_xlabel('Date and Time', fontsize=14)
    ax.set_title(title, fontsize=18, pad=20)

    handles, labels = ax.get_legend_handles_labels()
    label_order = [
        'Industrial Consumption (kW)',
        'Solar Generation (kW)',
        'Wind Generation (kW)',
        'Grid Power Used (kW)',
        'Energy Surplus (Self-sufficient)',
        'Night Time',
        'Weekend'
    ]
    new_handles, new_labels = [], []
    for label in label_order:
        if label in labels:
            index = labels.index(label)
            new_handles.append(handles[index])
            new_labels.append(labels[index])
    ax.legend(new_handles, new_labels, loc='upper right', fontsize=11,
              frameon=True, facecolor='white', framealpha=0.9)

    # Eixo X: dias da semana em inglês
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%a, %d', locale='C'))
    except TypeError:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%a, %d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%Hh'))

    ax.tick_params(axis='x', which='major', pad=15, labelsize=12, labelrotation=0)
    ax.tick_params(axis='x', which='minor', labelsize=10)

    ax.set_ylim(bottom=0)
    ax.set_xlim(df_plot['timestamp'].iloc[0], df_plot['timestamp'].iloc[-1])

    ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray', alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_monthly_consumption_summary(df, sim_config, save_path=None):
    """
    Creates a stacked bar chart showing total monthly energy consumption (kWh) by source.
    """
    print("Generating monthly consumption summary plot...")
    
    time_step_hours = sim_config.time_resolution_minutes / 60.0
    
    df_plot = df.copy()
    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
    
    df_plot['solar_used_kwh'] = df_plot['solar_used_kw'] * time_step_hours
    df_plot['wind_used_kwh'] = df_plot['wind_used_kw'] * time_step_hours
    df_plot['grid_used_kwh'] = df_plot['grid_used_kw'] * time_step_hours
    
    df_plot['month'] = df_plot['timestamp'].dt.to_period('M').astype(str)
    monthly_summary = df_plot.groupby('month')[['solar_used_kwh', 'wind_used_kwh', 'grid_used_kwh']].sum()

    monthly_summary.rename(columns={
        'solar_used_kwh': 'Solar (On-site)',
        'wind_used_kwh': 'Wind (On-site)',
        'grid_used_kwh': 'Grid (Purchased)'
    }, inplace=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    monthly_summary.plot(
        kind='bar', 
        stacked=True, 
        ax=ax, 
        color=['orange', 'skyblue', 'gray'],
        edgecolor='black'
    )
    
    ax.set_title(f'Monthly Energy Consumption by Source ({sim_config.duration_days // 30} Months)', fontsize=16)
    ax.set_ylabel('Total Energy Consumed (kWh)')
    ax.set_xlabel('Month')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(False)
    try:
        ax.legend(title='Energy Source', loc='upper right')
    except Exception:
        ax.legend(title='Energy Source', loc='upper right', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    y_max = monthly_summary.sum(axis=1).max()
    ax.set_ylim(0, y_max * 1.20)

    for i, total in enumerate(monthly_summary.sum(axis=1)):
        ax.text(i, total * 1.01, f'{total:,.0f} kWh', ha='center', va='bottom', fontsize=9, weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_monthly_cost_summary(df, sim_config, save_path=None):
    """
    Creates a stacked bar chart showing total monthly costs (BRL) by component.
    """
    print("Generating monthly cost summary plot...")
    
    df_plot = df.copy()
    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
    
    df_plot['month'] = df_plot['timestamp'].dt.to_period('M').astype(str)
    cost_columns = ['cost_solar', 'cost_wind', 'cost_grid_contract', 'cost_grid_spot']
    monthly_summary = df_plot.groupby('month')[cost_columns].sum()

    monthly_summary.rename(columns={
        'cost_solar': 'Solar (LCOE)',
        'cost_wind': 'Wind (LCOE)',
        'cost_grid_contract': 'Grid (Contract Price)',
        'cost_grid_spot': 'Grid (Spot Price/PLD)'
    }, inplace=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    monthly_summary.plot(
        kind='bar', 
        stacked=True, 
        ax=ax, 
        color=['orange', 'skyblue', 'dimgray', 'crimson'],
        edgecolor='black'
    )
    
    ax.set_title(f'Monthly Energy Cost Analysis ({sim_config.duration_days // 30} Months)', fontsize=16)
    ax.set_ylabel('Total Cost (BRL)')
    ax.set_xlabel('Month')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(False)
    try:
        ax.legend(title='Cost Component', loc='upper right')
    except Exception:
        ax.legend(title='Cost Component', loc='upper right', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    y_max = monthly_summary.sum(axis=1).max()
    ax.set_ylim(0, y_max * 1.10)

    for i, total in enumerate(monthly_summary.sum(axis=1)):
        ax.text(i, total * 1.01, locale.currency(total, grouping=True), ha='center', va='bottom', fontsize=9, weight='bold')

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: locale.currency(x, symbol=True, grouping=True)))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_rl_results(rl_result_path, save_path=None):
    """Plots the results of an RL simulation run."""
    with open(rl_result_path, 'r') as f:
        history = json.load(f)
    
    steps = [h['step'] for h in history]
    allocations = np.array([h['allocation_kw'] for h in history])
    costs = np.array([h['cost'] for h in history])
    sust_scores = np.array([h['sust_score'] for h in history])
    rewards = np.array([h['reward'] for h in history])
    
    contract_file_path = rl_result_path.replace('.rl_result.json', '.json')
    with open(contract_file_path, 'r') as f:
        data = json.load(f)
    sources = list(data['metadata']['contracts'].keys())
    
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    for i, src in enumerate(sources):
        axs[0].plot(steps, allocations[:, i], label=src.capitalize())
    axs[0].set_ylabel('Allocated Power (kW)')
    axs[0].set_title('RL Agent: Power Allocation per Source')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(steps, np.cumsum(costs), label='Cumulative Cost', color='red')
    axs[1].set_ylabel('Cumulative Cost', color='red')
    axs[1].tick_params(axis='y', labelcolor='red')
    ax2 = axs[1].twinx()
    ax2.plot(steps, np.cumsum(sust_scores), label='Cumulative Sustainability', color='green')
    ax2.set_ylabel('Cumulative Sustainability Score', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    axs[1].set_title('Cumulative Cost and Sustainability Score')
    axs[1].grid(True, alpha=0.3)
    
    axs[2].plot(steps, rewards, label='Reward per Step', color='blue')
    axs[2].set_ylabel('Reward')
    axs[2].set_xlabel('Simulation Step')
    axs[2].set_title('Reward per Step')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"RL results plot saved to: {save_path}")
    else:
        plt.show()