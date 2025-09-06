# src/core/energy_profile_config.py

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class Country(Enum):
    BRAZIL = "brazil"
    USA = "usa"
    GERMANY = "germany"

@dataclass
class IndustrialConfig:
    """Configuration for the industrial energy consumption profile."""
    base_load_kw: float = 100.0
    work_shift_load_kw: float = 400.0
    work_start_hour: int = 8
    work_end_hour: int = 18
    work_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4]) # Seg-Sex
    grid_contract_price_brl_per_mwh: float = 250.0
    grid_contract_volume_kw: float = 200.0

@dataclass
class OnSiteGenerationConfig:
    """Configuration for on-site electricity generation."""
    solar_installed_kw: float = 500.0
    wind_installed_kw: float = 300.0
    solar_lcoe_brl_per_mwh: float = 180.0
    wind_lcoe_brl_per_mwh: float = 220.0

@dataclass
class SourceConfig:
    """Configuration for a single ENERGY SOURCE CONTRACT (for price generation)."""
    name: str
    base_price: float
    quantity_kw: float
    historical_base_region: Optional[str] = None

@dataclass
class DESSConfig:
    """Configuration for the Hydrogen-based Decentralized Energy Supply System."""
    battery_capacity_kwh: float = 200.0
    battery_max_charge_kw: float = 50.0
    battery_max_discharge_kw: float = 50.0
    battery_charge_efficiency: float = 0.95
    battery_discharge_efficiency: float = 0.95
    electrolyzer_capacity_kw: float = 100.0
    electrolyzer_efficiency_kwh_per_kg: float = 55.0
    h2_storage_capacity_kg: float = 50.0
    fuel_cell_capacity_kw: float = 80.0
    fuel_cell_efficiency_kg_per_kwh: float = 0.025

# --- VERSÃO ÚNICA E CORRETA DA SimulationConfig ---
@dataclass
class SimulationConfig:
    """Unified configuration for any simulation run."""
    # Parâmetros básicos da simulação
    duration_days: int = 365
    time_resolution_minutes: int = 15
    country: Country = Country.BRAZIL
    random_seed: Optional[int] = 42

    # Metadados do experimento
    experiment_name: str = "default_experiment"
    experiment_description: str = "Default simulation"

    # Configuração do DESS (opcional, mas disponível)
    dess_config: DESSConfig = field(default_factory=DESSConfig)

# Mapeamento para contratos de preço no Brasil
brazil_sources: List[SourceConfig] = [
    SourceConfig(name="hydropower", base_price=275, quantity_kw=1000.0, historical_base_region="SOUTHEAST"),
    SourceConfig(name="wind", base_price=215, quantity_kw=1000.0, historical_base_region="NORTHEAST"),
    SourceConfig(name="solar", base_price=215, quantity_kw=1000.0, historical_base_region="NORTHEAST"),
    SourceConfig(name="biomass", base_price=350, quantity_kw=1000.0, historical_base_region="SOUTHEAST"),
    SourceConfig(name="biogas", base_price=450, quantity_kw=1000.0, historical_base_region="SOUTHEAST"),
    SourceConfig(name="grid", base_price=300, quantity_kw=5000.0, historical_base_region="SOUTHEAST"),
]

def get_configs_for_country(country: Country) -> List[SourceConfig]:
    if country == Country.BRAZIL:
        return brazil_sources
    else:
        raise ValueError(f"No configuration available for country: {country}")