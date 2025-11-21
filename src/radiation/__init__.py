"""
城市地表能量平衡计算模块

本模块基于 SEBAL (Surface Energy Balance Algorithm for Land) 模型实现，
提供完整的地表能量平衡计算功能。

主要功能:
- 太阳辐射计算（考虑地形效应）
- 净辐射计算（短波+长波）
- 土壤热通量计算
- 感热通量计算
- 潜热通量计算
- 完整能量平衡计算

模块结构:
- constants: 物理常量定义
- solar_radiation: 太阳辐射计算
- calc_net_radiation: 净辐射计算器
- calc_soil_heat: 土壤热通量计算器
- calc_sensible_heat: 感热通量计算器
- calc_latent_heat: 潜热通量计算器
- balance_equation: 能量平衡协调函数
- raster_io: GeoTIFF 文件 I/O

参考文献:
    Bastiaanssen et al., 1998. SEBAL: 1. Formulation. Journal of Hydrology.
    Laipelt et al., 2021. Long-term monitoring of evapotranspiration using SEBAL.
"""

__version__ = '0.2.0'

# 物理常量
from .constants import (
    STEFAN_BOLTZMANN,
    SOLAR_CONSTANT,
    AIR_DENSITY,
    SPECIFIC_HEAT_AIR,
    LATENT_HEAT_VAPORIZATION,
    WATER_VAPOR_RATIO,
    PSYCHROMETRIC_CONSTANT
)

# 太阳辐射模块
from .solar_radiation import (
    SolarConstantsCalculator,
    SolarRadiationCalculator,
    calculate_dem_solar_radiation,
    calculate_slope_aspect,
    parse_datetime
)

# 能量通量计算器
from .calc_net_radiation import NetRadiationCalculator
from .calc_soil_heat import SoilHeatFluxCalculator
from .calc_sensible_heat import SensibleHeatFluxCalculator
from .calc_latent_heat import LatentHeatFluxCalculator

# 能量平衡函数
from .balance_equation import (
    calculate_energy_balance,
    calculate_evaporative_fraction
)

# I/O 功能
from .raster_io import load_dem, save_radiation

__all__ = [
    # 版本
    '__version__',

    # 物理常量
    'STEFAN_BOLTZMANN',
    'SOLAR_CONSTANT',
    'AIR_DENSITY',
    'SPECIFIC_HEAT_AIR',
    'LATENT_HEAT_VAPORIZATION',
    'WATER_VAPOR_RATIO',
    'PSYCHROMETRIC_CONSTANT',

    # 太阳辐射
    'SolarConstantsCalculator',
    'SolarRadiationCalculator',
    'calculate_dem_solar_radiation',
    'calculate_slope_aspect',
    'parse_datetime',

    # 能量通量计算器
    'NetRadiationCalculator',
    'SoilHeatFluxCalculator',
    'SensibleHeatFluxCalculator',
    'LatentHeatFluxCalculator',

    # 能量平衡函数
    'calculate_energy_balance',
    'calculate_evaporative_fraction',

    # I/O 功能
    'load_dem',
    'save_radiation',
]
