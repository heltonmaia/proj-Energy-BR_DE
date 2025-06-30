"""
Energy Profiles Configuration for Multi-Country Energy Systems
==============================================================

Configuration classes and country-specific profiles for synthetic energy
data generation. Contains realistic parameters for Brazil and Germany
energy systems with multiple regional profiles.

"""

from dataclasses import dataclass
from typing import Tuple, Dict
from enum import Enum


class Country(Enum):
    """Supported countries for energy profile generation"""
    BRAZIL = "brazil"
    GERMANY = "germany"


class Region(Enum):
    """Regional profiles for different countries"""
    # Brazil regions
    BRAZIL_NORTHEAST = "brazil_northeast"  # High solar, growing wind
    BRAZIL_SOUTHEAST = "brazil_southeast"  # High consumption, mixed sources
    BRAZIL_SOUTH = "brazil_south"         # Good wind, moderate solar
    BRAZIL_AMAZON = "brazil_amazon"       # Grid challenges, hydro dominant
    BRAZIL_RURAL = "brazil_rural"         # Distributed generation, reliability issues
    
    # Germany regions
    GERMANY_NORTH = "germany_north"       # Offshore wind, good onshore wind
    GERMANY_SOUTH = "germany_south"       # High solar, industrial consumption
    GERMANY_EAST = "germany_east"         # Wind expansion, grid modernization
    GERMANY_WEST = "germany_west"         # Industrial, mixed renewable
    GERMANY_OFFSHORE = "germany_offshore" # Pure offshore wind focus


@dataclass
class SolarConfig:
    """Configuration parameters for solar energy generation"""
    # Physical parameters
    panel_area: float = 100.0  # m² - Total panel area
    panel_efficiency: float = 0.20  # 20% - Modern solar panel efficiency
    peak_irradiance: float = 1000.0  # W/m² - Peak solar irradiance (STC)
    
    # Location parameters
    latitude: float = -5.8  # Default: Natal, Brazil
    longitude: float = -35.2  # Default: Natal, Brazil
    
    # Environmental factors
    temperature_coefficient: float = -0.004  # %/°C - Power loss per degree
    dust_factor: float = 0.95  # Power loss due to dust/soiling
    system_losses: float = 0.85  # System losses (inverter, cables, etc.)
    
    # Weather variability
    cloud_probability: float = 0.3  # Daily probability of cloudy conditions
    cloud_duration_hours: Tuple[float, float] = (1.0, 6.0)  # Min, max cloud duration
    cloud_reduction_factor: Tuple[float, float] = (0.1, 0.8)  # Irradiance during clouds
    
    # Seasonal variation
    seasonal_variation: float = 0.2  # ±20% seasonal variation
    
    # Maintenance and soiling
    soiling_rate: float = 0.001  # Daily soiling accumulation rate
    cleaning_frequency_days: int = 30  # Cleaning every 30 days
    maintenance_downtime_hours: float = 4.0  # Hours of maintenance downtime
    
    # Performance degradation
    annual_degradation: float = 0.005  # 0.5% annual degradation


@dataclass
class WindConfig:
    """Configuration parameters for wind energy generation"""
    # Turbine specifications
    turbine_power_rating: float = 2000.0  # kW - Rated power
    rotor_diameter: float = 80.0  # m - Rotor diameter
    hub_height: float = 80.0  # m - Hub height
    cut_in_speed: float = 3.0  # m/s - Cut-in wind speed
    rated_speed: float = 12.0  # m/s - Rated wind speed
    cut_out_speed: float = 25.0  # m/s - Cut-out wind speed
    
    # Site characteristics
    average_wind_speed: float = 7.5  # m/s - Annual average wind speed
    weibull_k: float = 2.0  # Weibull shape parameter
    weibull_c: float = 8.5  # Weibull scale parameter
    turbulence_intensity: float = 0.15  # 15% turbulence intensity
    
    # Environmental factors
    air_density: float = 1.225  # kg/m³ - Air density at sea level
    power_coefficient: float = 0.45  # Cp - Power coefficient
    
    # Maintenance and availability
    availability_factor: float = 0.95  # 95% turbine availability
    maintenance_probability: float = 0.001  # Daily probability of maintenance
    maintenance_duration_hours: Tuple[float, float] = (4.0, 24.0)
    
    # Seasonal and daily patterns
    seasonal_wind_variation: float = 0.15  # ±15% seasonal variation
    daily_wind_pattern: bool = True  # Enable daily wind patterns
    offshore: bool = False  # Offshore wind characteristics
    
    # Performance characteristics
    power_curve_smoothing: float = 0.1  # Power curve smoothing factor
    wake_losses: float = 0.05  # Wake losses in wind farms


@dataclass
class GridConfig:
    """Configuration parameters for grid energy and reliability"""
    # Tariff structure (in USD/kWh for standardization)
    base_tariff: float = 0.15  # $/kWh - Base electricity tariff
    peak_multiplier: float = 1.8  # Peak hour tariff multiplier
    off_peak_multiplier: float = 0.7  # Off-peak tariff multiplier
    
    # Time periods
    peak_hours: Tuple[int, int] = (18, 21)  # Peak consumption hours
    off_peak_hours: Tuple[int, int] = (23, 6)  # Off-peak hours
    
    # Demand patterns
    base_consumption: float = 50.0  # kW - Base consumption
    peak_consumption_factor: float = 1.5  # Peak consumption multiplier
    weekend_factor: float = 0.8  # Weekend consumption factor
    seasonal_consumption_variation: float = 0.1  # ±10% seasonal variation
    
    # Grid reliability
    outage_probability: float = 0.002  # Daily probability of outages
    outage_duration_hours: Tuple[float, float] = (0.5, 4.0)
    voltage_stability: float = 0.95  # Voltage stability factor (0-1)
    frequency_stability: float = 0.98  # Frequency stability factor
    
    # Grid services and curtailment
    curtailment_probability: float = 0.001  # Renewable curtailment probability
    curtailment_factor: Tuple[float, float] = (0.5, 0.9)  # Curtailment reduction
    
    # Infrastructure characteristics
    grid_modernization_level: float = 0.8  # Grid modernization level (0-1)
    distributed_generation_penetration: float = 0.1  # DG penetration level
    energy_storage_capacity: float = 0.0  # Grid-scale storage capacity (kWh)
    
    # Economic factors
    feed_in_tariff: float = 0.08  # Feed-in tariff for excess generation
    net_metering: bool = True  # Net metering availability


@dataclass
class SourceConfig:
    share: float  # Participação na matriz (0-1)
    monthly_generation_gwh: float  # Geração mensal média (GWh)
    avg_cost: float  # Custo médio industrial (BRL/MWh ou EUR/MWh)


@dataclass
class IndustrialConfig:
    total_consumption_kw: float = 500.0
    # Shares por fonte (devem somar 1.0)
    shares: Dict[str, float] = None


@dataclass
class SimulationConfig:
    """Overall simulation configuration"""
    duration_days: int = 7
    time_resolution_minutes: int = 15
    country: Country = Country.BRAZIL
    random_seed: int = 42
    experiment_name: str = "default_experiment"
    experiment_description: str = "Synthetic energy generation profile"
    
    # Output configuration
    output_format: str = "json"  # "json" or "csv"
    include_metadata: bool = True
    include_statistics: bool = True
    include_weather_data: bool = False
    
    # Experiment metadata
    location_description: str = "Default location"
    
    # Advanced options
    enable_forecasting_errors: bool = True
    forecasting_error_std: float = 0.1  # 10% standard deviation
    enable_grid_interactions: bool = True


class CountryProfileManager:
    """Manages country-specific and regional energy profiles"""
    
    @staticmethod
    def get_profile(country: Country):
        if country == Country.BRAZIL:
            sources = {
                'hydropower': SourceConfig(share=0.55, monthly_generation_gwh=33200, avg_cost=275),
                'wind': SourceConfig(share=0.13, monthly_generation_gwh=9000, avg_cost=215),
                'solar': SourceConfig(share=0.10, monthly_generation_gwh=5900, avg_cost=215),
                'biomass': SourceConfig(share=0.08, monthly_generation_gwh=5100, avg_cost=350),
                'biogas': SourceConfig(share=0.01, monthly_generation_gwh=300, avg_cost=450),
            }
            industrial = IndustrialConfig(
                total_consumption_kw=500.0,
                shares={'hydropower': 0.55, 'wind': 0.13, 'solar': 0.10, 'biomass': 0.08, 'biogas': 0.01}
            )
        elif country == Country.GERMANY:
            sources = {
                'wind': SourceConfig(share=0.315, monthly_generation_gwh=11367, avg_cost=90),
                'solar': SourceConfig(share=0.138, monthly_generation_gwh=6017, avg_cost=110),
                'biomass': SourceConfig(share=0.065, monthly_generation_gwh=3000, avg_cost=135),
                'hydropower': SourceConfig(share=0.047, monthly_generation_gwh=1725, avg_cost=95),
            }
            industrial = IndustrialConfig(
                total_consumption_kw=500.0,
                shares={'wind': 0.315, 'solar': 0.138, 'biomass': 0.065, 'hydropower': 0.047}
            )
        else:
            raise ValueError(f"Unsupported country: {country}")
        
        return sources, industrial
    
    @staticmethod
    def get_region_description(region: Region) -> str:
        """Get description for each region"""
        descriptions = {
            Region.BRAZIL_NORTHEAST: "Northeast Brazil - High solar irradiance, growing wind sector, semi-arid climate",
            Region.BRAZIL_SOUTHEAST: "Southeast Brazil - Industrial hub, high consumption, mixed renewable sources",
            Region.BRAZIL_SOUTH: "South Brazil - Good wind resources, moderate solar, better grid infrastructure",
            Region.BRAZIL_AMAZON: "Amazon Brazil - Grid challenges, high solar potential, remote locations",
            Region.BRAZIL_RURAL: "Rural Brazil - Distributed generation focus, reliability challenges",
            Region.GERMANY_NORTH: "North Germany - Strong wind resources, offshore wind, reliable grid",
            Region.GERMANY_SOUTH: "South Germany - High solar potential, industrial consumption, advanced grid",
            Region.GERMANY_EAST: "East Germany - Wind expansion area, grid modernization, renewable focus",
            Region.GERMANY_WEST: "West Germany - Industrial region, mixed renewable sources, high consumption",
            Region.GERMANY_OFFSHORE: "German Offshore - Pure offshore wind generation, large-scale turbines"
        }
        return descriptions.get(region, "Unknown region")
    
    @staticmethod
    def list_available_profiles():
        """List all available regional profiles"""
        print("\nAvailable Regional Energy Profiles:")
        print("="*50)
        
        print("\n🇧🇷 BRAZIL PROFILES:")
        for region in [Region.BRAZIL_NORTHEAST, Region.BRAZIL_SOUTHEAST, 
                      Region.BRAZIL_SOUTH, Region.BRAZIL_AMAZON, Region.BRAZIL_RURAL]:
            print(f"  • {region.value}: {CountryProfileManager.get_region_description(region)}")
        
        print("\n🇩🇪 GERMANY PROFILES:")
        for region in [Region.GERMANY_NORTH, Region.GERMANY_SOUTH, 
                      Region.GERMANY_EAST, Region.GERMANY_WEST, Region.GERMANY_OFFSHORE]:
            print(f"  • {region.value}: {CountryProfileManager.get_region_description(region)}")


# Pre-defined simulation scenarios for quick access
PREDEFINED_SCENARIOS = {
    "brazil_northeast_1week": SimulationConfig(
        duration_days=7,
        country=Country.BRAZIL,
        experiment_name="brazil_northeast_1week",
        experiment_description="1-week simulation of Northeast Brazil energy profile"
    ),
    
    "germany_north_1month": SimulationConfig(
        duration_days=30,
        country=Country.GERMANY,
        experiment_name="germany_north_1month",
        experiment_description="1-month simulation of North Germany energy profile"
    ),
    
    "brazil_rural_1year": SimulationConfig(
        duration_days=365,
        country=Country.BRAZIL,
        experiment_name="brazil_rural_1year",
        experiment_description="1-year simulation of Rural Brazil energy profile"
    ),
    
    "germany_offshore_6months": SimulationConfig(
        duration_days=180,
        country=Country.GERMANY,
        experiment_name="germany_offshore_6months",
        experiment_description="6-month simulation of German offshore wind profile"
    ),
    
    "comparison_study": SimulationConfig(
        duration_days=30,
        country=Country.BRAZIL,
        experiment_name="comparison_study",
        experiment_description="Multi-country comparison study"
    )
}