"""
空气动力学参数计算模块

本模块提供城市地表能量平衡计算中所需的空气动力学参数计算功能。

主要功能:
- 饱和水汽压计算
- 实际水汽压计算（从相对湿度或露点温度）
- 露点温度空间降尺度（基于土地覆盖和FVC）
- DEM空间降尺度（基于ERA5气压约束）
- 空气密度计算（作为气温的函数）
- 风速计算（从ERA5-Land风速分量）
- 大气阻抗计算（rah - 大气湍流热交换阻抗）
- 表面阻抗计算（rs - 下垫面植被阻抗）

注意:
    粗糙度参数计算（基于建筑数据）现位于 landscape 模块

数据来源:
- ERA5-Land 再分析数据（风速、气压、露点温度等）
- 建筑数据（高度、平面面积、正面面积）
- LCZ 分类（参数化方案）

参考文献:
    - 晴朗无风条件下城市生态空间对城市降温作用量化模型
    - Bottema & Mestayer (1998): Urban roughness mapping
"""

__version__ = '0.1.0'

from .vapor_pressure import (
    calculate_saturation_vapor_pressure,
    calculate_actual_vapor_pressure,
    calculate_actual_vapor_pressure_from_dewpoint,
    calculate_vapor_pressure_deficit,
    calculate_vapor_pressure_deficit_from_dewpoint
)

from .dewpoint_downscaling import (
    downscale_dewpoint_temperature,
    validate_downscaling,
    print_validation_report,
    sensitivity_analysis
)

from .dem_downscaling import (
    adjust_pressure_for_elevation
)

from .atmospheric import (
    calculate_air_density,
    calculate_wind_speed,
    adjust_wind_speed_height
)

from .resistance import (
    calculate_aerodynamic_resistance,
    calculate_surface_resistance
)

from .constants import (
    LCZ_OBUKHOV_LENGTH,
    DEFAULT_OBUKHOV_LENGTH,
    get_obukhov_length_from_lcz
)

from .workflow import (
    calculate_aerodynamic_parameters,
    calculate_roughness_from_lcz
)

# Note: roughness calculation has been moved to landscape module

__all__ = [
    '__version__',
    # 水汽压计算
    'calculate_saturation_vapor_pressure',
    'calculate_actual_vapor_pressure',
    'calculate_actual_vapor_pressure_from_dewpoint',
    'calculate_vapor_pressure_deficit',
    'calculate_vapor_pressure_deficit_from_dewpoint',
    # 露点温度降尺度
    'downscale_dewpoint_temperature',
    'validate_downscaling',
    'print_validation_report',
    'sensitivity_analysis',
    # DEM降尺度
    'adjust_pressure_for_elevation',
    # 大气参数计算
    'calculate_air_density',
    'calculate_wind_speed',
    'adjust_wind_speed_height',
    # 阻抗计算
    'calculate_aerodynamic_resistance',
    'calculate_surface_resistance',
    # LCZ常量
    'LCZ_OBUKHOV_LENGTH',
    'DEFAULT_OBUKHOV_LENGTH',
    'get_obukhov_length_from_lcz',
    # 工作流
    'calculate_aerodynamic_parameters',
    'calculate_roughness_from_lcz',
]

