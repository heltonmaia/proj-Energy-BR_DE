# src/core/energy_profile_config.py

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

# Configurações para o gerador de perfis de energia
@dataclass
class IndustrialConfig:
    """Configuration for the industrial energy consumption profile."""
    base_load_kw: float = 100.0
    work_shift_load_kw: float = 400.0
    work_start_hour: int = 8
    work_end_hour: int = 18
    work_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4]) # Seg-Sex
    # Parâmetros de contrato com a rede
    grid_contract_price_brl_per_mwh: float = 250.0  # Preço fixo negociado
    grid_contract_volume_kw: float = 200.0 # Volume de potência que pode ser comprado a preço de contrato

@dataclass
class OnSiteGenerationConfig:
    """Configuration for on-site electricity generation."""
    solar_installed_kw: float = 500.0
    wind_installed_kw: float = 300.0
    # Custo Nivelado de Energia (LCOE) para geração própria
    solar_lcoe_brl_per_mwh: float = 180.0
    wind_lcoe_brl_per_mwh: float = 220.0

# Configurações originais
class Country(Enum):
    BRAZIL = "brazil"
    USA = "usa"
    GERMANY = "germany"
    
@dataclass
class ContractConfig:
    """Configuration for a single energy contract."""
    source_name: str
    quantity_kw: float
    base_price: float

@dataclass
class SourceConfig:
    name: str
    base_price: float
    quantity_kw: float
    historical_base_region: Optional[str] = None 
    
@dataclass
class SimulationConfig:
    duration_days: int = 365
    time_resolution_minutes: int = 15
    country: Country = Country.BRAZIL
    random_seed: Optional[int] = 42
    experiment_name: str = "default_experiment"
    experiment_description: str = "Default simulation"
    output_format: str = "json"
    include_metadata: bool = True
    include_statistics: bool = True
    include_weather_data: bool = False
    location_description: str = "Default location"
    enable_forecasting_errors: bool = True
    forecasting_error_std: float = 0.1
    enable_grid_interactions: bool = True

# Mapeamento para o Brasil com base na relevância regional de cada fonte
brazil_sources: List[SourceConfig] = [
    SourceConfig(name="hydropower", base_price=275, quantity_kw=1000.0, historical_base_region="SOUTHEAST"),
    SourceConfig(name="wind", base_price=215, quantity_kw=1000.0, historical_base_region="NORTHEAST"),
    SourceConfig(name="solar", base_price=215, quantity_kw=1000.0, historical_base_region="NORTHEAST"),
    SourceConfig(name="biomass", base_price=350, quantity_kw=1000.0, historical_base_region="SOUTHEAST"),
    SourceConfig(name="biogas", base_price=450, quantity_kw=1000.0, historical_base_region="SOUTHEAST"),
    SourceConfig(name="grid", base_price=300, quantity_kw=5000.0, historical_base_region="SOUTHEAST"), 
]

# Exemplo de configuração para outros países
usa_sources: List[SourceConfig] = [] 
germany_sources: List[SourceConfig] = []

def get_configs_for_country(country: Country) -> List[SourceConfig]:
    if country == Country.BRAZIL:
        return brazil_sources
    elif country == Country.USA:
        return usa_sources
    elif country == Country.GERMANY:
        return germany_sources
    else:
        raise ValueError(f"No configuration available for country: {country}")