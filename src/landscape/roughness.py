"""
城市粗糙度参数计算模块

基于《Mapping the Roughness Parameters in a Large Urban Area for Urban Climate Applications》
(Bottema & Mestayer, 1998) 的方法计算城市粗糙度参数。

核心参数:
    - 零平面位移高度 (z_d)
    - 粗糙度长度 (z_0)
    - 城市冠层孔隙度 (P)

输入数据要求:
    - 建筑高度 (h_i)
    - 建筑平面面积/足迹面积 (footprint area)
    - 建筑正面面积 (frontal area)
    - 地块总面积 (plot area)

参考文献:
    Bottema, M., & Mestayer, P. G. (1998). Urban roughness mapping-validation 
    techniques and some first results. Journal of Wind Engineering and Industrial 
    Aerodynamics, 74, 163-173.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class BuildingData:
    """
    建筑数据结构

    属性:
        heights: 建筑高度数组 (m)
        footprint_areas: 建筑平面面积/足迹面积数组 (m²)
        frontal_areas: 建筑正面面积数组 (m²) - 可选，用于更精确计算
        plot_area: 地块总面积 (m²)
    """
    heights: np.ndarray
    footprint_areas: np.ndarray
    plot_area: float
    frontal_areas: Optional[np.ndarray] = None

    def __post_init__(self):
        """验证输入数据"""
        if len(self.heights) != len(self.footprint_areas):
            raise ValueError("建筑高度和平面面积数组长度必须相同")

        if self.frontal_areas is not None and len(self.frontal_areas) != len(self.heights):
            raise ValueError("正面面积数组长度必须与建筑数量相同")

        if self.plot_area <= 0:
            raise ValueError("地块面积必须大于0")


def calculate_volume_weighted_height(
        building_heights: np.ndarray,
        building_footprint_areas: np.ndarray
) -> float:
    """
    计算建筑体积平均高度 h

    公式: h = Σ(V_i × h_i) / Σ(V_i)
    其中: V_i = footprint_area_i × h_i

    说明:
        以建筑体积为权重，避免传统"算术平均高度"忽略大体积建筑影响的误差，
        更真实反映建筑高度特征。

    参数:
        building_heights: 单栋建筑高度数组 h_i (m)
        building_footprint_areas: 单栋建筑足迹面积数组 (m²)

    返回:
        建筑体积平均高度 h (m)

    示例:
        >>> heights = np.array([10.0, 20.0, 15.0])
        >>> areas = np.array([100.0, 200.0, 150.0])
        >>> h = calculate_volume_weighted_height(heights, areas)
    """
    if len(building_heights) == 0:
        return 0.0

    # 计算各建筑体积 V_i = footprint_area × height
    volumes = building_footprint_areas * building_heights

    # 体积加权平均高度
    h = np.sum(volumes * building_heights) / np.sum(volumes)

    return float(h)


def calculate_plan_area_density(
        building_footprint_areas: np.ndarray,
        plot_area: float
) -> float:
    """
    计算平面面积密度 λ_p (lambda_p)

    公式: λ_p = A_P / A_T
    其中:
        A_P: 建筑总平面面积（单栋建筑足迹面积之和）
        A_T: 地块总面积

    说明:
        简单比值反映建筑在水平方向的密集程度，取值0-1，值越大表示建筑越密集。

    参数:
        building_footprint_areas: 单栋建筑足迹面积数组 (m²)
        plot_area: 地块总面积 A_T (m²)

    返回:
        平面面积密度 λ_p，范围 [0, 1]

    示例:
        >>> areas = np.array([100.0, 200.0, 150.0])
        >>> plot_area = 1000.0
        >>> lambda_p = calculate_plan_area_density(areas, plot_area)
    """
    A_P = np.sum(building_footprint_areas)
    lambda_p = A_P / plot_area

    # 确保在合理范围内（理论上不应超过1，但实际数据可能有误差）
    lambda_p = np.clip(lambda_p, 0.0, 1.0)

    return float(lambda_p)


def calculate_frontal_area_density(
        building_frontal_areas: np.ndarray,
        plot_area: float
) -> float:
    """
    计算正面面积密度 λ_F (lambda_F)

    公式: λ_F = A_F / A_T
    其中:
        A_F: 建筑总正面面积
        A_T: 地块总面积

    说明:
        反映建筑在气流方向的垂直阻挡强度。A_F随气流方向变化，
        需沿径向（覆盖主要风向）计算多个方向后取平均值。

    参数:
        building_frontal_areas: 单栋建筑正面面积数组 (m²)
        plot_area: 地块总面积 A_T (m²)

    返回:
        正面面积密度 λ_F，范围 [0, 1]

    注意:
        如果没有正面面积数据，可以使用简化估算: λ_F ≈ 0.5 × λ_p

    示例:
        >>> frontal_areas = np.array([80.0, 160.0, 120.0])
        >>> plot_area = 1000.0
        >>> lambda_F = calculate_frontal_area_density(frontal_areas, plot_area)
    """
    A_F = np.sum(building_frontal_areas)
    lambda_F = A_F / plot_area

    # 确保在合理范围内
    lambda_F = np.clip(lambda_F, 0.0, 1.0)

    return float(lambda_F)


def estimate_frontal_area_density_from_plan(
        lambda_p: float,
        height_to_width_ratio: float = 0.8
) -> float:
    """
    从平面面积密度估算正面面积密度（当缺少正面面积数据时）

    简化估算: λ_F ≈ α × λ_p
    其中 α 是建筑高宽比系数，通常取0.5-0.8

    参数:
        lambda_p: 平面面积密度
        height_to_width_ratio: 高宽比系数，默认0.8

    返回:
        估算的正面面积密度 λ_F

    注意:
        这是一个粗略估算，实际应用中建议使用真实的正面面积数据
    """
    lambda_F = height_to_width_ratio * lambda_p
    return float(np.clip(lambda_F, 0.0, 1.0))


def calculate_zero_plane_displacement(
        mean_building_height: float,
        lambda_p: float
) -> float:
    """
    计算零平面位移高度 z_d

    公式: z_d = h × (λ_p)^0.6

    基于 Bottema 和 Mestayer (1998) 简化模型，适配不规则建筑群。
    无需考虑建筑体积分布与回流区，仅通过 h（反映建筑高度特征）
    与 λ_p（反映建筑水平密集度）快速估算。

    参数:
        mean_building_height: 建筑体积平均高度 h (m)
        lambda_p: 平面面积密度 λ_p，范围 [0, 1]

    返回:
        零平面位移高度 z_d (m)

    说明:
        z_d 反映气流被建筑整体抬升的高度，为后续 z_0 计算提供基础。

    示例:
        >>> h = 15.0  # 平均建筑高度15米
        >>> lambda_p = 0.4  # 建筑覆盖率40%
        >>> z_d = calculate_zero_plane_displacement(h, lambda_p)
    """
    if lambda_p <= 0:
        return 0.0

    z_d = mean_building_height * (lambda_p ** 0.6)

    # z_d 不应超过平均建筑高度
    z_d = min(z_d, mean_building_height * 0.9)

    return float(z_d)


def calculate_roughness_length(
        mean_building_height: float,
        zero_plane_displacement: float,
        lambda_F: float,
        von_karman_constant: float = 0.4,
        drag_coefficient: float = 0.8
) -> float:
    """
    计算粗糙度长度 z_0

    公式: z_0 = (h - z_d) × exp(-κ / √(0.5 × C_Dh × λ_F))

    其中:
        h: 建筑体积平均高度 (m)
        z_d: 零平面位移高度 (m)
        κ: 冯·卡门常数 (通常 = 0.4)
        C_Dh: 孤立障碍物拖曳系数 (通常 = 0.8)
        λ_F: 正面面积密度

    基于 Bottema 和 Mestayer (1998) 气动模型改进，通过 λ_F 量化建筑
    对气流的阻挡效应，突破传统模型仅适用于规则建筑群的限制。

    参数:
        mean_building_height: 建筑体积平均高度 h (m)
        zero_plane_displacement: 零平面位移高度 z_d (m)
        lambda_F: 正面面积密度 λ_F，范围 [0, 1]
        von_karman_constant: 冯·卡门常数 κ，默认 0.4
        drag_coefficient: 拖曳系数 C_Dh，默认 0.8

    返回:
        粗糙度长度 z_0 (m)

    说明:
        z_0 反映地表对气流的摩擦作用强度，核心用于筛选低粗糙度通风路径
        （通常 z_0 < 0.5m 为候选）。

    示例:
        >>> h = 15.0
        >>> z_d = 9.0
        >>> lambda_F = 0.3
        >>> z_0 = calculate_roughness_length(h, z_d, lambda_F)
    """
    if lambda_F <= 0:
        # 当没有建筑时，使用最小粗糙度
        return 0.01

    # 避免数值计算问题
    lambda_F = max(lambda_F, 0.001)

    # 计算指数项
    exponent = -von_karman_constant / np.sqrt(0.5 * drag_coefficient * lambda_F)

    # 计算粗糙度
    z_0 = (mean_building_height - zero_plane_displacement) * np.exp(exponent)

    # 限制在合理范围 (0.01m - 5m)
    z_0 = np.clip(z_0, 0.01, 5.0)

    return float(z_0)


def calculate_canopy_porosity_fixed(
        building_volumes: np.ndarray,
        plot_area: float,
        canopy_height: float = 40.0
) -> float:
    """
    计算固定冠层高度孔隙度 P_h-const

    公式: P_h-const = (A_T × h_const - V) / (A_T × h_const)

    其中:
        A_T: 地块总面积
        h_const: 固定冠层高度（建议取超过95%建筑的高度）
        V: 地块内建筑总体积

    说明:
        假设冠层高度统一，快速估算开放空气体积比例，计算简便但精度略低。

    参数:
        building_volumes: 建筑体积数组 V_i (m³)
        plot_area: 地块总面积 A_T (m²)
        canopy_height: 固定冠层高度 h_const (m)，默认40m

    返回:
        固定冠层高度孔隙度 P_h-const，范围 [0, 1]

    示例:
        >>> volumes = np.array([1000.0, 4000.0, 2250.0])
        >>> plot_area = 1000.0
        >>> P = calculate_canopy_porosity_fixed(volumes, plot_area, 40.0)
    """
    total_volume = np.sum(building_volumes)
    canopy_volume = plot_area * canopy_height

    if canopy_volume <= 0:
        return 0.0

    P = (canopy_volume - total_volume) / canopy_volume

    # 确保在合理范围
    P = np.clip(P, 0.0, 1.0)

    return float(P)


def calculate_canopy_porosity_variable(
        building_volumes: np.ndarray,
        building_heights: np.ndarray,
        plot_area: float
) -> float:
    """
    计算可变冠层高度孔隙度 P_h-var

    公式: P_h-var = (A_T × h_UCL - V) / (A_T × h_UCL)

    其中:
        A_T: 地块总面积
        h_UCL: 地块内最高建筑高度
        V: 地块内建筑总体积

    说明:
        以地块内实际最高建筑为冠层高度，更贴合不同区域的冠层结构差异，
        精度高于 P_h-const。可识别"高 z_0 但高 P"的特殊区域
        （如高层公寓间绿地）。

    参数:
        building_volumes: 建筑体积数组 V_i (m³)
        building_heights: 建筑高度数组 h_i (m)
        plot_area: 地块总面积 A_T (m²)

    返回:
        可变冠层高度孔隙度 P_h-var，范围 [0, 1]

    示例:
        >>> volumes = np.array([1000.0, 4000.0, 2250.0])
        >>> heights = np.array([10.0, 20.0, 15.0])
        >>> plot_area = 1000.0
        >>> P = calculate_canopy_porosity_variable(volumes, heights, plot_area)
    """
    if len(building_heights) == 0:
        return 1.0

    total_volume = np.sum(building_volumes)
    h_UCL = np.max(building_heights)
    canopy_volume = plot_area * h_UCL

    if canopy_volume <= 0:
        return 0.0

    P = (canopy_volume - total_volume) / canopy_volume

    # 确保在合理范围
    P = np.clip(P, 0.0, 1.0)

    return float(P)


@dataclass
class RoughnessParameters:
    """
    粗糙度参数结果

    属性:
        mean_height: 建筑体积平均高度 h (m)
        lambda_p: 平面面积密度
        lambda_F: 正面面积密度
        zero_plane_displacement: 零平面位移高度 z_d (m)
        roughness_length: 粗糙度长度 z_0 (m)
        canopy_porosity_fixed: 固定冠层高度孔隙度 P_h-const（可选）
        canopy_porosity_variable: 可变冠层高度孔隙度 P_h-var（可选）
    """
    mean_height: float
    lambda_p: float
    lambda_F: float
    zero_plane_displacement: float
    roughness_length: float
    canopy_porosity_fixed: Optional[float] = None
    canopy_porosity_variable: Optional[float] = None

    def __repr__(self) -> str:
        """格式化输出"""
        lines = [
            "粗糙度参数计算结果:",
            f"  建筑体积平均高度 (h):        {self.mean_height:.2f} m",
            f"  平面面积密度 (λ_p):          {self.lambda_p:.4f}",
            f"  正面面积密度 (λ_F):          {self.lambda_F:.4f}",
            f"  零平面位移高度 (z_d):        {self.zero_plane_displacement:.2f} m",
            f"  粗糙度长度 (z_0):            {self.roughness_length:.4f} m",
        ]

        if self.canopy_porosity_fixed is not None:
            lines.append(f"  固定冠层孔隙度 (P_h-const): {self.canopy_porosity_fixed:.4f}")

        if self.canopy_porosity_variable is not None:
            lines.append(f"  可变冠层孔隙度 (P_h-var):   {self.canopy_porosity_variable:.4f}")

        return "\n".join(lines)


def calculate_roughness_parameters(
        building_data: BuildingData,
        calculate_porosity: bool = True,
        fixed_canopy_height: float = 40.0,
        use_frontal_area: bool = True
) -> RoughnessParameters:
    """
    计算完整的城市粗糙度参数集

    这是主要的接口函数，整合所有计算步骤。

    参数:
        building_data: 建筑数据对象
        calculate_porosity: 是否计算冠层孔隙度，默认 True
        fixed_canopy_height: 固定冠层高度 (m)，默认 40m
        use_frontal_area: 是否使用正面面积数据（如果有），默认 True

    返回:
        RoughnessParameters 对象，包含所有计算结果

    计算流程:
        1. 计算建筑体积平均高度 h
        2. 计算平面面积密度 λ_p
        3. 计算或估算正面面积密度 λ_F
        4. 计算零平面位移高度 z_d
        5. 计算粗糙度长度 z_0
        6. （可选）计算冠层孔隙度 P

    示例:
        >>> # 准备建筑数据
        >>> heights = np.array([10.0, 20.0, 15.0, 12.0])
        >>> footprints = np.array([100.0, 200.0, 150.0, 120.0])
        >>> frontals = np.array([80.0, 160.0, 120.0, 96.0])
        >>> plot_area = 1000.0
        >>> 
        >>> building_data = BuildingData(
        ...     heights=heights,
        ...     footprint_areas=footprints,
        ...     frontal_areas=frontals,
        ...     plot_area=plot_area
        ... )
        >>> 
        >>> # 计算粗糙度参数
        >>> params = calculate_roughness_parameters(building_data)
        >>> print(params)
    """
    # 1. 计算建筑体积平均高度
    mean_height = calculate_volume_weighted_height(
        building_data.heights,
        building_data.footprint_areas
    )

    # 2. 计算平面面积密度
    lambda_p = calculate_plan_area_density(
        building_data.footprint_areas,
        building_data.plot_area
    )

    # 3. 计算正面面积密度
    if use_frontal_area and building_data.frontal_areas is not None:
        lambda_F = calculate_frontal_area_density(
            building_data.frontal_areas,
            building_data.plot_area
        )
    else:
        # 使用简化估算
        lambda_F = estimate_frontal_area_density_from_plan(lambda_p)

    # 4. 计算零平面位移高度
    z_d = calculate_zero_plane_displacement(mean_height, lambda_p)

    # 5. 计算粗糙度长度
    z_0 = calculate_roughness_length(mean_height, z_d, lambda_F)

    # 6. （可选）计算冠层孔隙度
    P_fixed = None
    P_variable = None

    if calculate_porosity:
        volumes = building_data.footprint_areas * building_data.heights

        P_fixed = calculate_canopy_porosity_fixed(
            volumes,
            building_data.plot_area,
            fixed_canopy_height
        )

        P_variable = calculate_canopy_porosity_variable(
            volumes,
            building_data.heights,
            building_data.plot_area
        )

    return RoughnessParameters(
        mean_height=mean_height,
        lambda_p=lambda_p,
        lambda_F=lambda_F,
        zero_plane_displacement=z_d,
        roughness_length=z_0,
        canopy_porosity_fixed=P_fixed,
        canopy_porosity_variable=P_variable
    )


def calculate_roughness_for_raster(
        building_height_raster: np.ndarray,
        building_density_raster: np.ndarray,
        frontal_density_raster: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为栅格数据批量计算粗糙度参数

    适用于从建筑数据栅格化后的场景，返回栅格化的 z_d 和 z_0。

    参数:
        building_height_raster: 建筑高度栅格 (m)
        building_density_raster: 建筑密度栅格（平面面积比）
        frontal_density_raster: 正面面积密度栅格（可选）

    返回:
        (z_d_raster, z_0_raster): 零平面位移高度和粗糙度长度栅格

    示例:
        >>> height_raster = np.random.rand(100, 100) * 20
        >>> density_raster = np.random.rand(100, 100) * 0.5
        >>> z_d, z_0 = calculate_roughness_for_raster(height_raster, density_raster)
    """
    # 计算零平面位移高度
    z_d_raster = building_height_raster * (building_density_raster ** 0.6)

    # 处理正面面积密度
    if frontal_density_raster is None:
        frontal_density_raster = estimate_frontal_area_density_from_plan(
            building_density_raster
        )

    # 避免除以零
    frontal_density_safe = np.maximum(frontal_density_raster, 0.001)

    # 计算粗糙度长度
    von_karman = 0.4
    drag_coef = 0.8
    exponent = -von_karman / np.sqrt(0.5 * drag_coef * frontal_density_safe)
    z_0_raster = (building_height_raster - z_d_raster) * np.exp(exponent)

    # 限制在合理范围
    z_0_raster = np.clip(z_0_raster, 0.01, 5.0)
    z_d_raster = np.clip(z_d_raster, 0.0, building_height_raster * 0.9)

    return z_d_raster.astype(np.float32), z_0_raster.astype(np.float32)