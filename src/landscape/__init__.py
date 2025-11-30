"""
景观参数模块

提供景观层面的参数计算功能。

主要功能:
- 城市粗糙度参数计算（基于 Bottema & Mestayer, 1998）
  * 零平面位移高度 (z_d)
  * 粗糙度长度 (z_0)
  * 城市冠层孔隙度 (P)
- 迎风面积计算（扫描线法，考虑遮挡效应）
  * 单个建筑群的有效迎风面积
  * 按街区分组批量计算
  * 可视化工具

数据需求:
- 建筑数据（高度、平面面积、正面面积）
- 迎风面积计算需要：minp, maxp, height

注意:
    - 阻抗计算（rah, rs）位于 aerodynamics 模块
    - LCZ相关常量位于 aerodynamics.constants 模块

参考文献:
    - Bottema & Mestayer (1998): Urban roughness mapping
    - doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
"""

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
    calculate_roughness_for_raster,
)

from .frontal_area import (
    calculate_frontal_area_for_cluster,
    calculate_frontal_area_by_district,
    get_skyline_profile,
    plot_frontal_area_profile,
    plot_district_comparison,
    # 建筑数据处理（使用扫描线法）
    BUILDING_FIELDS,
    ClusterRoughnessResult,
    calculate_cluster_roughness,
    calculate_roughness_from_buildings,
)

__all__ = [
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
    # 建筑数据处理
    'BUILDING_FIELDS',
    'ClusterRoughnessResult',
    'calculate_cluster_roughness',
    'calculate_roughness_from_buildings',
    # 迎风面积计算
    'calculate_frontal_area_for_cluster',
    'calculate_frontal_area_by_district',
    'get_skyline_profile',
    'plot_frontal_area_profile',
    'plot_district_comparison',
]

