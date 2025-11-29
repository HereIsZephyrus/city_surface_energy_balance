"""
阻抗参数计算模块

提供大气湍流热交换阻抗(rah)和表面阻抗(rs)的计算功能。

阻抗计算是SEBAL模型中不确定性最大的部分，需要根据：
- LCZ分类（用于Obukhov长度估计和地表类型判断）
- 地表粗糙度
- 风速
- 植被类型和覆盖度

假设条件:
- 晴朗白天：大气稳定度固定为不稳定（unstable）
- Obukhov长度根据LCZ类型估计

LCZ编码方案（简化版，与 utils.mapping 一致）:
    1-9:  城市建筑类型
    10:   裸岩/铺装 (原LCZ E) - 不透水面
    11:   密集树木 (原LCZ A) - 植被
    12:   灌木/低矮植被 (原LCZ C/D) - 植被
    13:   裸土/沙地 (原LCZ F) - 裸土
    14:   水体 (原LCZ G) - 水体

数据来源:
- ERA5-Land: 风速数据
- 建筑数据: 粗糙度计算
- LCZ分类: 参数化方案
- 遥感数据: NDVI、LAI等

公式参考:
    文档《晴朗无风条件下城市生态空间对城市降温作用量化模型》
"""

import numpy as np
from typing import Optional
from .constants import get_obukhov_length_from_lcz


# LCZ 类型常量（与 utils.mapping.LCZ 一致）
_LCZ_BARE_ROCK = 10     # 裸岩/铺装
_LCZ_DENSE_TREES = 11   # 密集树木
_LCZ_BUSH_GRASS = 12    # 灌木/低矮植被
_LCZ_BARE_SOIL = 13     # 裸土/沙地
_LCZ_WATER = 14         # 水体


def calculate_aerodynamic_resistance(
        wind_speed: np.ndarray,
        roughness_length: np.ndarray,
        lcz: np.ndarray,
        displacement_height: Optional[np.ndarray] = None,
        measurement_height: float = 10.0
) -> np.ndarray:
    """
    计算大气湍流热交换阻抗 rah (s/m)

    假设条件:
    - 晴朗白天，大气稳定度固定为不稳定（unstable）
    - Obukhov长度根据LCZ类型估计

    根据地表类型采用不同的参数化方案:

    城市/不透水面 (unsaturated, LCZ 1-10):
        rah = (Φv/Φm) × (uz/u*²) + a × u*^(-2/3)
        其中 a = 6.266

    植被/饱和面 (saturated, LCZ 11-12):
        rah = ln((z - d) / z0) / (k² × Uw)

    裸土 (LCZ 13):
        使用简化的 saturated 公式

    水体 (LCZ 14):
        特殊处理，使用极小粗糙度

    参数:
        wind_speed: 近地表风速 uz (m/s) - ndarray, 来自ERA5-Land
        roughness_length: 地表粗糙度 z0 (m) - ndarray
        lcz: LCZ分类 (1-14) - ndarray
        displacement_height: 零平面位移高度 d (m) - ndarray, 可选
        measurement_height: 风速测量高度 (m) - scalar, 默认10m (ERA5标准)

    返回:
        大气湍流热交换阻抗 rah (s/m) - ndarray

    注意:
        - 晴朗白天大气不稳定，L < 0（由LCZ确定）
        - 城市地表粗糙度可从建筑数据计算
        - LCZ类型应为整数1-14
    """
    # 常数
    kappa = 0.41  # Von Karman常数
    a = 6.266     # 边界层阻抗系数（城市不透水面）

    # 根据LCZ获取Obukhov长度
    obukhov_length = get_obukhov_length_from_lcz(lcz)

    # 根据地表类型设置位移高度
    # 基于 LCZ 的默认估算:
    #   城市建筑 (LCZ 1-9): d ≈ 6.5 × z0 (Grimmond & Oke 1999)
    #   铺装/裸岩 (LCZ 10): d ≈ 0 (几乎无位移)
    #   植被 (LCZ 11-12): d ≈ 0.67 × z0
    #   裸土 (LCZ 13): d ≈ 0
    #   水体 (LCZ 14): d = 0
    lcz_displacement = np.where(
        lcz <= 9,
        6.5 * roughness_length,   # 城市建筑
        np.where(
            (lcz == _LCZ_DENSE_TREES) | (lcz == _LCZ_BUSH_GRASS),
            0.67 * roughness_length,  # 植被
            0.0  # 铺装、裸土、水体
        )
    )
    
    if displacement_height is None:
        displacement_height = lcz_displacement
    else:
        # 合并建筑数据和 LCZ 估算
        # 建筑数据 > 0 的区域使用建筑值，其他使用 LCZ 估算
        valid_building_mask = (displacement_height > 0) & np.isfinite(displacement_height)
        displacement_height = np.where(valid_building_mask, displacement_height, lcz_displacement)

    # 安全处理
    wind_speed_safe = np.maximum(wind_speed, 0.1)  # 最小风速 0.1 m/s
    z0_safe = np.maximum(roughness_length, 0.0001)  # 最小粗糙度 0.1mm（水体需要）

    # 计算有效高度 z - d
    z_effective = measurement_height - displacement_height
    z_effective_safe = np.maximum(z_effective, z0_safe * 2)  # 确保 z - d > z0

    # 稳定度参数 ζ = (z - d) / L
    # 晴朗白天 L < 0（不稳定），所以 ζ < 0
    zeta = z_effective_safe / obukhov_length

    # 不稳定条件下的稳定度修正函数（Paulson, 1970; Dyer, 1974）
    # x = (1 - 16ζ)^0.25
    x = (1 - 16 * zeta) ** 0.25

    # 动量稳定度修正 ψm
    psi_m = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi / 2

    # === 计算摩擦速度 u* ===
    # u* = k × uz / [ln((z-d)/z0) - ψm]
    u_star = kappa * wind_speed_safe / (np.log(z_effective_safe / z0_safe) - psi_m)
    u_star_safe = np.maximum(u_star, 0.01)  # 防止除零

    # 稳定度修正因子 Φm 和 Φv（不稳定条件）
    # Φm = (1 - 16ζ)^(-1/4)
    # Φv = Φm² = (1 - 16ζ)^(-1/2)
    phi_m = (1 - 16 * zeta) ** (-0.25)
    phi_v = phi_m ** 2

    # 初始化结果数组
    rah = np.zeros_like(wind_speed, dtype=np.float32)

    # === 情况1: 城市/不透水面 (LCZ 1-10) - unsaturated 公式 ===
    # rah = (Φv/Φm) × (uz/u*²) + a × u*^(-2/3)
    urban_mask = (lcz >= 1) & (lcz <= _LCZ_BARE_ROCK)
    r_am = wind_speed_safe / (u_star_safe ** 2)   # 动力阻抗 r_am = uz / u*²
    r_b = a * (u_star_safe ** (-2/3))              # 边界层阻抗 r_b = a × u*^(-2/3)
    rah_urban = (phi_v / phi_m) * r_am + r_b       # Φv/Φm = Φm
    rah[urban_mask] = rah_urban[urban_mask]

    # === 情况2: 植被/饱和面 (LCZ 11-12) - saturated 公式 ===
    # rah = ln((z - d) / z0) / (k² × Uw)
    # 注：文档原式为 ln((Z_tree - d) × Z_ρ)，此处简化为 ln((z-d)/z0)
    veg_mask = (lcz == _LCZ_DENSE_TREES) | (lcz == _LCZ_BUSH_GRASS)
    rah_veg = np.log(z_effective_safe / z0_safe) / (kappa ** 2 * wind_speed_safe)
    rah[veg_mask] = rah_veg[veg_mask]

    # === 情况3: 裸土/沙地 (LCZ 13) ===
    # 使用与植被相似的 saturated 公式，但无位移高度
    bare_soil_mask = (lcz == _LCZ_BARE_SOIL)
    rah_soil = np.log(measurement_height / z0_safe) / (kappa ** 2 * wind_speed_safe)
    rah[bare_soil_mask] = rah_soil[bare_soil_mask]

    # === 情况4: 水体 (LCZ 14) - 特殊处理 ===
    # 水面粗糙度极小（~0.0001m），使用固定粗糙度
    water_mask = (lcz == _LCZ_WATER)
    z0_water = 0.0001  # 静水面粗糙度
    rah_water = np.log(measurement_height / z0_water) / (kappa ** 2 * wind_speed_safe)
    rah[water_mask] = rah_water[water_mask]

    # 限制在合理范围 (通常 10-500 s/m)
    rah = np.clip(rah, 10.0, 500.0)

    return rah.astype(np.float32)


def calculate_surface_resistance(
        ndvi: np.ndarray,
        lai: Optional[np.ndarray] = None,
        soil_moisture: Optional[np.ndarray] = None,
        land_cover_type: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    计算表面阻抗 rs (s/m)

    表面阻抗反映植被气孔阻力和土壤蒸发阻力。

    基于Penman-Monteith方程的参数化:
        rs = rs_min / LAI × f1(Ta) / [f2(VPD) × f3(ψ)]

    简化方案（基于NDVI）:
        - 高植被覆盖: rs较小（易蒸腾）
        - 裸地/不透水面: rs较大（难蒸发）

    参数:
        ndvi: 归一化植被指数 - ndarray, 范围-1到1
        lai: 叶面积指数 - ndarray, 可选
        soil_moisture: 土壤湿度 - ndarray, 可选
        land_cover_type: 土地覆盖类型 - ndarray, 可选

    返回:
        表面阻抗 rs (s/m) - ndarray

    注意:
        - 不透水面 rs → 无穷大（实际取很大值）
        - 植被茂盛 rs → 最小气孔阻力
        - 需要考虑土壤水分影响

    TODO:
        完整实现需要:
        1. LAI数据（遥感反演或查找表）
        2. 土壤水分数据（ERA5-Land或遥感估算）
        3. 详细的土地覆盖分类
        4. 不同植被类型的参数化方案
    """
    # 这是一个基于NDVI的简化实现

    # NDVI阈值
    ndvi_min = 0.0  # 裸地阈值
    ndvi_max = 0.8  # 高植被阈值

    # 归一化NDVI (0-1)
    ndvi_norm = np.clip((ndvi - ndvi_min) / (ndvi_max - ndvi_min), 0.0, 1.0)

    # 基于NDVI的线性插值
    # 高植被: rs ≈ 50 s/m
    # 裸地/不透水面: rs ≈ 500 s/m
    rs_min = 50.0
    rs_max = 500.0

    rs = rs_max - ndvi_norm * (rs_max - rs_min)

    # 如果提供了土壤湿度，进行修正
    if soil_moisture is not None:
        # 干燥土壤增大阻抗
        moisture_factor = np.clip(soil_moisture, 0.1, 1.0)
        rs = rs / moisture_factor

    # 限制在合理范围
    rs = np.clip(rs, 30.0, 1000.0)

    return rs.astype(np.float32)


def estimate_roughness_length_from_buildings(
        building_height: np.ndarray,
        building_density: np.ndarray
) -> np.ndarray:
    """
    根据建筑高度和密度估算地表粗糙度

    城市粗糙度参数化方案：
        z0 ≈ 0.1 × H × λp

    其中:
        H: 平均建筑高度
        λp: 建筑平面面积比

    参数:
        building_height: 平均建筑高度 (m) - ndarray
        building_density: 建筑密度/面积比 (0-1) - ndarray

    返回:
        粗糙度长度 z0 (m) - ndarray

    参考:
        Grimmond & Oke, 1999
    """
    # 简化的参数化方案
    z0 = 0.1 * building_height * building_density

    # 限制在合理范围 (0.01m - 5m)
    z0 = np.clip(z0, 0.01, 5.0)

    return z0.astype(np.float32)

