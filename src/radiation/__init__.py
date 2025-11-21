"""
城市地表能量平衡计算模块（系数法）

本模块实现基于系数分解的能量平衡计算，用于街区级回归分析。

核心思想:
    不需要Ta（近地表气温）的具体值，而是计算Ta的系数：
    f(Ta) = coeff_Ta × Ta + residual = 0
    
    在街区聚合后，通过ALS回归求解每个街区的Ta。

主要功能:
- 太阳辐射计算（考虑地形效应）
- 净辐射系数计算（∂Q*/∂Ta）
- 土壤热通量计算（不依赖Ta）
- 感热通量系数计算（∂QH/∂Ta）
- 潜热通量计算（不依赖Ta）
- 能量平衡系数计算（用于街区回归）

模块结构:
- constants: 物理常量定义
- solar_radiation: 太阳辐射计算
- calc_net_radiation: 净辐射系数计算器
- calc_soil_heat: 土壤热通量计算器
- calc_sensible_heat: 感热通量系数计算器
- calc_latent_heat: 潜热通量计算器
- balance_equation: 能量平衡系数计算函数
- raster_io: GeoTIFF 文件 I/O

参考文献:
    Bastiaanssen et al., 1998. SEBAL: 1. Formulation. Journal of Hydrology.
    doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
"""

__version__ = '0.2.0'

# 物理常量
from .constants import (
    STEFAN_BOLTZMANN,
    SOLAR_CONSTANT,
    SPECIFIC_HEAT_AIR,
    GAS_CONSTANT_DRY_AIR,
    STANDARD_PRESSURE,
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
    calculate_energy_balance_coefficients,
    validate_energy_balance
)

# I/O 功能
from .raster_io import load_dem, save_radiation

__all__ = [
    # 版本
    '__version__',

    # 物理常量
    'STEFAN_BOLTZMANN',
    'SOLAR_CONSTANT',
    'SPECIFIC_HEAT_AIR',
    'GAS_CONSTANT_DRY_AIR',
    'STANDARD_PRESSURE',
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

    # 能量平衡系数计算（用于街区回归）
    'calculate_energy_balance_coefficients',
    'validate_energy_balance',

    # I/O 功能
    'load_dem',
    'save_radiation',
]
