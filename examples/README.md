# Stacked Bar Chart Examples

This directory contains examples of how to use the `plot_stacked_bar_chart` function implemented in the `utils.plot` module.

## Function `plot_stacked_bar_chart`

The function creates stacked bar charts in the requested style, with the following characteristics:

- **Colors**: Uses "Set3" colormap for attractive colors
- **Borders**: Bars with black edges for better definition
- **Grid**: Horizontal grid with dashed lines
- **Legend**: Positioned to the right of the chart
- **Formatting**: Title, axis labels and optimized layout

### Parameters:

- `data_dict`: Dictionary with data {source: values}
- `title`: Chart title
- `ylabel`: Y-axis label
- `xlabel`: X-axis label (default: "Source")
- `save_path`: Path to save the chart (optional)
- `figsize`: Figure size (default: (12, 6))

## Included Examples

### 1. Monthly Renewable Energy Generation (Brazil)
**File**: `monthly_renewable_generation_brazil.png`

Demonstrates monthly renewable energy generation in Brazil, including:
- Hydropower
- Wind
- Solar PV
- Biomass
- Biogas

### 2. Daily Consumption by Source (Industrial Facility)
**File**: `daily_consumption_by_source.png`

Shows daily energy consumption of an industrial facility by source:
- Solar
- Wind
- Grid
- Battery

### 3. Hourly Renewable Energy Generation
**File**: `hourly_generation.png`

Illustrates hourly variation of renewable energy generation over 24 hours:
- Solar (diurnal variation)
- Wind (continuous variation)

### 4. Complete Monthly Energy Analysis
**File**: `monthly_energy_analysis.png`

**NEW**: Demonstrates complete monthly generation and consumption analysis with:
- **Upper left panel**: Temporal generation by source over time
- **Upper right panel**: Monthly stacked generation by source (stacked bars) - **GWh scale**
- **Lower left panel**: Temporal consumption by source over time
- **Lower right panel**: Monthly stacked consumption by source (stacked bars) - **kWh + Currency**

This example shows how data is automatically grouped by month and displayed in stacked bar charts, allowing visualization of:
- Seasonal variation of solar and wind generation
- Monthly distribution of consumption by source
- Monthly totals on top of each bar
- Comparison between generation and consumption throughout the year

**Key Features**:
- **Month numbering**: X-axis shows months as 1, 2, 3, 4... instead of month names
- **English labels**: All text and labels are in English
- **Clean layout**: No rotation needed for month labels
- **GWh scale for generation**: More appropriate scale for large generation volumes
- **Currency values for consumption**: Shows monthly costs in local currency (BRL/EUR)

## How to Run Examples

```bash
# Navigate to project directory
cd proj-BRA_GER

# Run basic stacked bar examples
python examples/stacked_bar_example.py

# Run complete monthly analysis example
python examples/monthly_energy_example.py
```

## Usage in Your Code

```python
from utils.plot import plot_stacked_bar_chart, plot_dual_energy_figures

# For simple stacked bar charts
data = {
    "Solar": [100, 150, 200, 180],
    "Wind": [80, 120, 160, 140],
    "Grid": [50, 60, 70, 65]
}

plot_stacked_bar_chart(
    data_dict=data,
    title="Energy Generation by Source",
    ylabel="Power (kW)",
    xlabel="Time Period",
    save_path="my_chart.png"
)

# For complete analysis with temporal data (automatic monthly grouping)
plot_dual_energy_figures(
    df=your_dataframe,  # DataFrame with columns timestamp, *_power_kw, *_used_kw
    country=Country.BRAZIL,
    save_path="monthly_analysis.png"
)
```

## Modifications to Existing Functions

The `plot_dual_energy_figures` and other plotting functions have been updated to use the new monthly stacked bar chart style with improved scales and currency information.

### Main Changes:

1. **Monthly grouping**: Data is automatically grouped by month using `df.groupby('month')`
2. **Monthly stacked bars**: Replacement of horizontal bars with vertical monthly stacked bars
3. **Consistent colors**: Use of "Set3" colormap in all charts
4. **Total values**: Display of monthly totals on top of each bar
5. **Grid**: Addition of horizontal grid to facilitate value reading
6. **Layout**: Legends positioned to the right to avoid overlapping the chart
7. **Month numbering**: X-axis shows months as 1, 2, 3, 4... for clean presentation
8. **English interface**: All comments, labels and text in English
9. **GWh scale for generation**: Generation values displayed in GWh (more appropriate for large volumes)
10. **Currency values for consumption**: Monthly consumption costs shown in local currency

### Monthly Data Structure:

The `plot_dual_energy_figures` function now:
- Automatically groups data by month using `timestamp.dt.to_period('M')`
- Calculates monthly totals for each energy source
- Creates stacked bar charts showing monthly evolution
- Displays total values on top of each monthly bar
- Maintains original temporal charts for comparison
- Uses simple month numbering (1, 2, 3...) on X-axis
- All text and labels in English
- **Generation in GWh**: Converts kWh to GWh for better readability
- **Consumption with currency**: Shows both kWh and monetary cost for each month

### Value Display Format:

- **Generation (Upper Right)**: `X.XX GWh` (e.g., "1.25 GWh")
- **Consumption (Lower Right)**: `X kWh` + `Currency` (e.g., "1000 kWh" + "R$ 500")
- **Currency Support**: BRL for Brazil, EUR for Germany, USD as fallback 