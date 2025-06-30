import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_stacked_bar_chart(data_dict, title, ylabel, xlabel="Source", save_path=None, figsize=(12, 6)):
    """
    Cria gráfico de barras empilhadas no estilo solicitado.
    
    Args:
        data_dict: Dicionário com dados {source: values}
        title: Título do gráfico
        ylabel: Label do eixo Y
        xlabel: Label do eixo X
        save_path: Caminho para salvar o gráfico
        figsize: Tamanho da figura
    """
    # Criar DataFrame
    df = pd.DataFrame(data_dict)
    
    # Plotagem
    plt.figure(figsize=figsize)
    df.plot(kind='bar', stacked=True, colormap="Set3", edgecolor='black')
    
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Source", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Stacked bar chart saved to: {save_path}")
    else:
        plt.show()

def plot_energy_profiles(df, region_name, save_path=None, max_points=1000):
    # Garantir que timestamp é datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Amostrar se muitos pontos
    if len(df) > max_points:
        df_plot = df.iloc[::len(df)//max_points]
    else:
        df_plot = df
    plt.figure(figsize=(14, 7))
    # Geração
    plt.plot(df_plot['timestamp'], df_plot['solar_power_kw'], label='Solar Generation (kW)', color='orange', linewidth=1)
    plt.plot(df_plot['timestamp'], df_plot['wind_power_kw'], label='Wind Generation (kW)', color='skyblue', linewidth=1)
    plt.plot(df_plot['timestamp'], df_plot['grid_power_kw'], label='Grid Generation (kW)', color='gray', linewidth=1)
    # Consumo industrial
    plt.plot(df_plot['timestamp'], df_plot['industrial_consumption_kw'], label='Industrial Consumption (kW)', color='green', linewidth=1.5, linestyle='--')
    plt.plot(df_plot['timestamp'], df_plot['solar_used_kw'], label='Solar Used (kW)', color='orange', linewidth=1, linestyle=':')
    plt.plot(df_plot['timestamp'], df_plot['wind_used_kw'], label='Wind Used (kW)', color='skyblue', linewidth=1, linestyle=':')
    plt.plot(df_plot['timestamp'], df_plot['grid_used_kw'], label='Grid Used (kW)', color='gray', linewidth=1, linestyle=':')
    plt.xlabel('Timestamp')
    plt.ylabel('Power (kW)')
    plt.title(f"Synthetic Energy Profiles - {region_name}")
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=30)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_dual_energy_figures(df, country, save_path=None, max_points=1000):
    """
    Plots dual figures with energy generation and consumption.
    Updated to handle data saved as JSON.
    """
    import matplotlib.ticker as mtick
    
    # Convert timestamp to datetime if necessary
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert timestamps to days since start
    start_time = df['timestamp'].iloc[0]
    df['days'] = (df['timestamp'] - start_time).dt.total_seconds() / (24 * 3600)
    
    # Sample if too many points
    if len(df) > max_points:
        df_plot = df.iloc[::max(1, len(df)//max_points)]
    else:
        df_plot = df
    
    # Identify available sources
    power_cols = [col for col in df_plot.columns if col.endswith('_power_kw')]
    used_cols = [col for col in df_plot.columns if col.endswith('_used_kw')]
    
    all_sources = [col.replace('_power_kw','') for col in power_cols]
    used_sources = [col.replace('_used_kw','') for col in used_cols]
    
    # Calculate temporal resolution
    if len(df) > 1:
        try:
            t0 = pd.to_datetime(df['timestamp'].iloc[0])
            t1 = pd.to_datetime(df['timestamp'].iloc[1])
            time_res_h = float((t1 - t0).total_seconds()) / 3600.0
        except Exception:
            time_res_h = 0.25  # 15 minutes by default
    else:
        time_res_h = 0.25
    
    # Group data by month
    df['month'] = df['timestamp'].dt.to_period('M')
    df['month_number'] = df['timestamp'].dt.month + (df['timestamp'].dt.year - df['timestamp'].dt.year.iloc[0]) * 12
    
    # Calculate monthly totals for generation and consumption (kWh)
    monthly_generation = {}
    monthly_consumption = {}
    
    for src in all_sources:
        col_name = f'{src}_power_kw'
        if col_name in df.columns:
            monthly_data = df.groupby('month')[col_name].sum() * time_res_h
            monthly_generation[src] = monthly_data
    
    for src in used_sources:
        col_name = f'{src}_used_kw'
        if col_name in df.columns:
            monthly_data = df.groupby('month')[col_name].sum() * time_res_h
            monthly_consumption[src] = monthly_data
    
    # Calculate overall totals for percentages
    total_generated = {}
    for src in all_sources:
        if src in monthly_generation:
            total_generated[src] = float(monthly_generation[src].sum())
    
    total_consumed = {}
    for src in used_sources:
        if src in monthly_consumption:
            total_consumed[src] = float(monthly_consumption[src].sum())
    
    # Calculate consumption percentages
    total_consumption_kwh = sum(total_consumed.values())
    consumption_percentages = {}
    for src in used_sources:
        if src in total_consumed and total_consumption_kwh > 0:
            consumption_percentages[src] = (total_consumed[src] / total_consumption_kwh) * 100
    
    # Get price and shares configuration
    try:
        from core.energy_profile_config import CountryProfileManager, Country
        sources_cfg, industrial_cfg = CountryProfileManager.get_profile(country)
        price_unit = 'BRL' if country == Country.BRAZIL else 'EUR'
        
        # Calculate prices
        price_generated = {}
        for src in all_sources:
            if src in sources_cfg and src in total_generated:
                price_generated[src] = float(total_generated[src]) * float(sources_cfg[src].avg_cost) / 1000.0
        
        price_consumed = {}
        for src in used_sources:
            if src in sources_cfg and src in total_consumed:
                price_consumed[src] = float(total_consumed[src]) * float(sources_cfg[src].avg_cost) / 1000.0
        
        # Get configured shares
        configured_shares = industrial_cfg.shares
    except Exception as e:
        print(f"Warning: Could not load price configuration: {e}")
        price_unit = 'USD'
        price_generated = {src: 0.0 for src in all_sources}
        price_consumed = {src: 0.0 for src in used_sources}
        configured_shares = {}
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [3, 1]})
    
    # 1. Generation by source (without consumption)
    ax1 = axs[0,0]
    colors = ['orange', 'skyblue', 'green', 'red', 'purple', 'brown']
    for i, src in enumerate(all_sources):
        col_name = f'{src}_power_kw'
        if col_name in df_plot.columns:
            color = colors[i % len(colors)]
            ax1.plot(df_plot['days'], df_plot[col_name], 
                    label=f'{src.capitalize()} Generation (kW)', color=color, linewidth=1)
    ax1.set_ylabel('Generation Power (kW)', color='blue')
    ax1.set_xlabel('Days')
    ax1.set_title(f'Energy Generation by Source - {country.value.capitalize()}')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Helper function to format currency
    def format_currency(value, unit):
        if unit == 'BRL':
            return f'R$ {int(value):,}'.replace(',', '.')
        elif unit == 'EUR':
            return f'€ {int(value):,}'.replace(',', '.')
        else:
            return f'{int(value):,}'

    # 1b. Monthly stacked generation bars (upper right panel) - GWh scale
    if monthly_generation:
        # Create DataFrame for monthly stacked chart
        generation_data = {}
        for src in all_sources:
            if src in monthly_generation:
                # Convert to GWh
                generation_data[src.capitalize()] = monthly_generation[src].values / 1000000
        
        # Get month numbers (1, 2, 3, ...)
        month_numbers = list(range(1, len(monthly_generation[list(monthly_generation.keys())[0]]) + 1))
        
        # Plot stacked chart
        df_gen = pd.DataFrame(generation_data, index=month_numbers)
        df_gen.plot(kind='bar', stacked=True, ax=axs[0,1], colormap="Set3", edgecolor='black')
        
        # Add total values on top of bars (in GWh)
        total_values = df_gen.sum(axis=1)
        for i, total in enumerate(total_values):
            axs[0,1].text(i, total, f'{total:.2f} GWh', ha='center', va='bottom', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        axs[0,1].set_title('Monthly Generation (GWh)')
        axs[0,1].set_ylabel('GWh')
        axs[0,1].set_xlabel('Month')
        axs[0,1].tick_params(axis='x', rotation=0)
        axs[0,1].grid(axis='y', linestyle='--', alpha=0.7)
        axs[0,1].legend(title="Source", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Detailed temporal consumption
    if 'industrial_consumption_kw' in df_plot.columns:
        axs[1,0].plot(df_plot['days'], df_plot['industrial_consumption_kw'], 
                     label='Total Industrial Consumption (kW)', color='black', linewidth=2)
    for i, src in enumerate(used_sources):
        col_name = f'{src}_used_kw'
        if col_name in df_plot.columns:
            color = colors[i % len(colors)]
            axs[1,0].plot(df_plot['days'], df_plot[col_name], 
                         label=f'{src.capitalize()} Used (kW)', color=color, linewidth=1.5)
    axs[1,0].set_ylabel('Consumption Power (kW)')
    axs[1,0].set_xlabel('Days')
    axs[1,0].set_title('Simulated Industrial Consumption by Source')
    axs[1,0].legend()
    axs[1,0].grid(True, alpha=0.3)
    
    # 2b. Monthly stacked consumption bars (lower right panel) - kWh with currency
    if monthly_consumption:
        # Create DataFrame for monthly stacked chart
        consumption_data = {}
        for src in used_sources:
            if src in monthly_consumption:
                consumption_data[src.capitalize()] = monthly_consumption[src].values
        
        # Get month numbers (1, 2, 3, ...)
        month_numbers = list(range(1, len(monthly_consumption[list(monthly_consumption.keys())[0]]) + 1))
        
        # Plot stacked chart
        df_cons = pd.DataFrame(consumption_data, index=month_numbers)
        df_cons.plot(kind='bar', stacked=True, ax=axs[1,1], colormap="Set3", edgecolor='black')
        
        # Add total values on top of bars (kWh + currency)
        total_values = df_cons.sum(axis=1)
        for i, total in enumerate(total_values):
            # Calculate monthly cost
            monthly_cost = 0
            for src in used_sources:
                if src in monthly_consumption and src in price_consumed:
                    src_monthly_kwh = monthly_consumption[src].iloc[i]
                    src_cost_per_kwh = price_consumed[src] / total_consumed[src] if total_consumed[src] > 0 else 0
                    monthly_cost += src_monthly_kwh * src_cost_per_kwh
            
            # Format text with kWh and currency
            cost_text = format_currency(monthly_cost, price_unit)
            text = f'{total:.0f} kWh\n{cost_text}'
            axs[1,1].text(i, total, text, ha='center', va='bottom', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        total_consumption_kwh = sum(total_consumed.values())
        total_cost = sum(price_consumed.values())
        cost_text = format_currency(total_cost, price_unit)
        title = f'Monthly Consumption (kWh)\nTotal: {total_consumption_kwh:.0f} kWh - {cost_text}'
        axs[1,1].set_title(title)
        axs[1,1].set_ylabel('kWh')
        axs[1,1].set_xlabel('Month')
        axs[1,1].tick_params(axis='x', rotation=0)
        axs[1,1].grid(axis='y', linestyle='--', alpha=0.7)
        axs[1,1].legend(title="Source", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Final formatting
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {save_path}")
    else:
        plt.show() 