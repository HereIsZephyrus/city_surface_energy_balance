"""
景观参数模块

提供景观层面的参数计算功能。

主要功能:
- 城市粗糙度参数计算（基于 Bottema & Mestayer, 1998）
  * 零平面位移高度 (z_d)
  * 粗糙度长度 (z_0)
  * 城市冠层孔隙度 (P)

数据需求:
- 建筑数据（高度、平面面积、正面面积）

注意:
    - 阻抗计算（rah, rs）位于 aerodynamics 模块
    - LCZ相关常量位于 aerodynamics.constants 模块

参考文献:
    - Bottema & Mestayer (1998): Urban roughness mapping
    - doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
"""

__version__ = '0.1.0'

from .roughness import (
    BuildingData,
    RoughnessParameters,
    calculate_volume_weighted_height,
    calculate_plan_area_density,
    calculate_frontal_area_density,
    estimate_frontal_area_density_from_plan,
    calculate_zero_plane_displacement,
    calculate_roughness_length,
    calculate_canopy_porosity_fixed,
    calculate_canopy_porosity_variable,
    calculate_roughness_parameters,
    calculate_roughness_for_raster
)

__all__ = [
    '__version__',
    # 粗糙度参数计算
    'BuildingData',
    'RoughnessParameters',
    'calculate_volume_weighted_height',
    'calculate_plan_area_density',
    'calculate_frontal_area_density',
    'estimate_frontal_area_density_from_plan',
    'calculate_zero_plane_displacement',
    'calculate_roughness_length',
    'calculate_canopy_porosity_fixed',
    'calculate_canopy_porosity_variable',
    'calculate_roughness_parameters',
    'calculate_roughness_for_raster',
]

