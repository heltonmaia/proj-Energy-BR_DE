# Changelog

## [2024-12-19] - Monthly Stacked Bar Charts with GWh Scale and Currency Values

### Added
- New `plot_stacked_bar_chart()` function in `src/utils/plot.py` to create stacked bar charts in the requested style
- Automatic monthly data grouping in `plot_dual_energy_figures()` function
- Monthly stacked bar charts for energy generation and consumption
- Usage examples in `examples/stacked_bar_example.py` and `examples/monthly_energy_example.py`
- Complete documentation in `examples/README.md`
- **GWh scale for generation**: More appropriate scale for large generation volumes
- **Currency values for consumption**: Monthly costs displayed in local currency (BRL/EUR/USD)

### Modified
- `plot_dual_energy_figures()` function now groups data by month using `df.groupby('month')`
- Replacement of horizontal bar charts with vertical monthly stacked bar charts
- Addition of total values on top of each monthly bar
- **Month numbering**: X-axis now shows months as 1, 2, 3, 4... instead of month names
- **English interface**: All comments, labels, and text converted to English
- Consistent use of "Set3" colormap in all charts
- No rotation needed for month labels (clean horizontal display)
- **Generation scale**: Values now displayed in GWh instead of kWh for better readability
- **Consumption information**: Enhanced with both kWh values and monetary costs

### Chart Features
- **Colors**: "Set3" colormap for attractive and consistent colors
- **Borders**: Bars with black edges for better definition
- **Grid**: Horizontal grid with dashed lines
- **Legend**: Positioned to the right of the chart
- **Values**: Monthly totals displayed on top of each bar
- **Layout**: Optimized for monthly data visualization
- **Month numbering**: Simple 1, 2, 3... numbering on X-axis
- **English labels**: All text and interface elements in English
- **GWh generation**: Generation values in GWh scale (e.g., "1.25 GWh")
- **Currency consumption**: Consumption values with monetary costs (e.g., "1000 kWh + R$ 500")

### Monthly Chart Structure
- **Upper left panel**: Temporal generation by source (maintained)
- **Upper right panel**: Monthly stacked generation by source (NEW) - **GWh scale**
- **Lower left panel**: Temporal consumption by source (maintained)
- **Lower right panel**: Monthly stacked consumption by source (NEW) - **kWh + Currency**

### Created Examples
1. `monthly_renewable_generation_brazil.png` - Monthly renewable energy generation
2. `daily_consumption_by_source.png` - Daily consumption by source
3. `hourly_generation.png` - Hourly generation
4. `monthly_energy_analysis.png` - Complete monthly analysis (12 months)

### Compatibility
- Maintains compatibility with existing data
- Works automatically with data of any duration (1 month to multiple years)
- Automatic monthly grouping regardless of temporal data resolution
- Clean month numbering (1, 2, 3...) for better readability
- Full English interface for international use
- **Multi-currency support**: BRL (Brazil), EUR (Germany), USD (fallback)
- **Automatic scale conversion**: kWh to GWh for generation, kWh + currency for consumption 