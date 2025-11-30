"""
能量平衡方程模块

计算能量平衡方程的系数和残差，用于街区级回归分析。

核心思想:
    不需要Ta的具体值，将能量平衡分解为二次方程：
    f(Ta) = coeff2 × Ta² + coeff1 × Ta + residual = 0

    然后在街区聚合后，通过求根公式或ALS回归求解每个街区的Ta。

能量平衡方程:
    Q* - ΔQ_storage + QH - QE = 0

    符号约定:
    - QH = ρCp(Ta - Ts)/rah  (当 Ta < Ts 时为负)
    - 标准 H = ρCp(Ts - Ta)/rah (当 Ts > Ta 时为正)
    - 因此 QH = -H，能量方程用 +QH 而非 -QH

    Q* 使用二阶泰勒展开近似 L↓(Ta):
    L↓ = εₐσTa⁴ ≈ 6εₐσTa₀²×Ta² - 8εₐσTa₀³×Ta + 3εₐσTa₀⁴

储热通量 ΔQ_storage:
    - 自然表面 (LCZ 11-14): SEBAL 公式计算
    - 不透水面 (LCZ 1-10): β_impervious × Q* (β ≈ 0.35)
    - 建筑额外贡献: 通过 ALS 回归估计

分解为:
    Q*_coeff2×Ta² + [Q*_coeff1 + QH_coeff]×Ta + [Q*_const - ΔQ_storage + QH_const - QE] = 0
    即: f_Ta_coeff2×Ta² + f_Ta_coeff1×Ta + residual = 0

参考:
    doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
"""

import numpy as np
from typing import Dict
from .calc_net_radiation import NetRadiationCalculator
from .calc_soil_heat import StorageHeatFluxCalculator
from .calc_sensible_heat import SensibleHeatFluxCalculator
from .calc_latent_heat import LatentHeatFluxCalculator
from .constants import STEFAN_BOLTZMANN, GAS_CONSTANT_DRY_AIR


def calculate_air_density(
    surface_pressure: np.ndarray,
    era5_air_temperature: np.ndarray
) -> float:
    """
    计算空气密度的参考值（使用ERA5-Land气温）

    ρ = P / (R_d × Ta)

    参数:
        surface_pressure: 地表气压 P (Pa) - ndarray
        era5_air_temperature: ERA5-Land气温 (K) - ndarray或scalar

    返回:
        平均空气密度 (kg/m³) - scalar
    """
    P_mean = np.nanmean(surface_pressure)
    Ta_mean = np.nanmean(era5_air_temperature)
    rho = P_mean / (GAS_CONSTANT_DRY_AIR * Ta_mean)
    return float(rho)


def calculate_energy_balance_coefficients(
    shortwave_down: np.ndarray,
    surface_temperature: np.ndarray,
    elevation: np.ndarray,
    albedo: np.ndarray,
    ndvi: np.ndarray,
    saturation_vapor_pressure: np.ndarray,
    actual_vapor_pressure: np.ndarray,
    aerodynamic_resistance: np.ndarray,
    surface_resistance: np.ndarray,
    surface_emissivity: np.ndarray,
    surface_pressure: np.ndarray,
    era5_air_temperature: np.ndarray,
    lcz: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    计算能量平衡方程关于Ta的二次、一次系数和残差

    将能量平衡分解为二次方程：
    f(Ta) = coeff2 × Ta² + coeff1 × Ta + residual = 0

    使用二阶泰勒展开近似 L↓(Ta)，提高温度估算精度。

    储热通量根据 LCZ 分类:
    - 自然表面: SEBAL 公式直接计算，计入 residual
    - 不透水面: 返回 storage_feature，系数 β 在 ALS 回归中估计

    参数:
        shortwave_down: 短波下行辐射 (W/m²)
        surface_temperature: 地表温度 Ts (K)
        elevation: 高程 (m)
        albedo: 反照率
        ndvi: NDVI
        saturation_vapor_pressure: es (kPa)
        actual_vapor_pressure: ea (kPa)
        aerodynamic_resistance: rah (s/m)
        surface_resistance: rs (s/m)
        surface_emissivity: ε₀
        surface_pressure: P (Pa)
        era5_air_temperature: ERA5-Land气温 (K)
        lcz: LCZ分类栅格 (1-14)

    返回:
        {
            'f_Ta_coeff2': Ta²系数 (W/m²/K²),
            'f_Ta_coeff1': Ta系数 (W/m²/K),
            'residual': 残差项 (W/m²),
            'era5_air_temperature': ERA5气温 (K),
            'storage_feature': 储热回归特征 (W/m²)
                              不透水面 = Q*, 自然表面 = 0

            # 分项（调试用）
            'Q_star_coeff2': Q*的Ta²系数,
            'Q_star_coeff1': Q*的Ta系数,
            'QH_coeff': ∂QH/∂Ta,
            'Q_star_const': Q*的常数部分,
            'storage_heat_flux': 自然表面 SEBAL 计算的 ΔQ_Sg,
            'latent_heat_flux': QE
        }
    """
    # 步骤1: 计算空气密度参考值（使用ERA5气温）
    rho_ref = calculate_air_density(surface_pressure, era5_air_temperature)

    # 获取参考气温（用于系数计算）
    ta_reference = float(np.nanmean(era5_air_temperature))

    # 步骤2: 计算净辐射系数
    net_rad_calc = NetRadiationCalculator(
        shortwave_down=shortwave_down,
        surface_temperature=surface_temperature,
        elevation=elevation,
        albedo=albedo,
        surface_emissivity=surface_emissivity,
        ta_reference=ta_reference
    )
    Q_star_coeffs = net_rad_calc.net_radiation_coefficient

    # 步骤3: 计算感热通量系数
    sensible_calc = SensibleHeatFluxCalculator(
        surface_temperature=surface_temperature,
        aerodynamic_resistance=aerodynamic_resistance,
        air_density=rho_ref
    )
    QH_coeffs = sensible_calc.sensible_heat_coefficient

    # 步骤4: 计算储热通量（不依赖Ta）
    # 使用参考气温计算Q*₀来估算
    L_up = surface_emissivity * STEFAN_BOLTZMANN * (surface_temperature ** 4)
    epsilon_a = net_rad_calc.atmospheric_emissivity
    L_down_ref = epsilon_a * STEFAN_BOLTZMANN * (ta_reference ** 4)

    Q_star_ref = (
        (1 - albedo) * shortwave_down +
        surface_emissivity * L_down_ref -
        L_up
    )

    # 使用 LCZ 分类计算储热
    # - 自然表面 (LCZ 11-14): SEBAL 公式直接计算 ΔQ_Sg
    # - 不透水面 (LCZ 1-10): 使用基础储热系数 β_impervious × Q* (默认 0.35)
    # - 建筑区 (LCZ 1-9): 额外返回建筑特征用于 ALS 回归估计建筑额外贡献
    storage_calc = StorageHeatFluxCalculator(
        net_radiation=Q_star_ref,
        surface_temperature=surface_temperature,
        albedo=albedo,
        ndvi=ndvi,
        lcz=lcz
    )
    # 获取储热通量和建筑回归特征
    Delta_Q_storage = storage_calc.storage_heat_flux  # 自然表面SEBAL，不透水面=β×Q*
    storage_feature = storage_calc.storage_feature    # 建筑区Q*，其他=0（用于估计建筑额外贡献）

    # 步骤5: 计算潜热通量 QE（不依赖Ta）
    latent_flux_calc = LatentHeatFluxCalculator(
        saturation_vapor_pressure=saturation_vapor_pressure,
        actual_vapor_pressure=actual_vapor_pressure,
        aerodynamic_resistance=aerodynamic_resistance,
        surface_resistance=surface_resistance,
        air_density=rho_ref
    )
    QE = latent_flux_calc.latent_heat_flux

    # 步骤6: 组合系数和残差
    # 能量平衡方程: Q* - ΔQ_storage + QH - QE = 0
    # 
    # 注意符号约定:
    #   QH = ρCp(Ta - Ts)/rah  (代码定义，当 Ta < Ts 时为负)
    #   H  = ρCp(Ts - Ta)/rah  (标准定义，当 Ts > Ta 时为正，表示热量从地表到大气)
    #   因此 QH = -H
    #
    # 标准能量平衡: Q* - ΔQ_storage - H - QE = 0
    # 代入 H = -QH:  Q* - ΔQ_storage - (-QH) - QE = Q* - ΔQ_storage + QH - QE = 0
    #
    # Q* 使用二阶泰勒展开:
    #   Q* = Q*_coeff2×Ta² + Q*_coeff1×Ta + Q*_const
    #
    # QH 是线性的:
    #   QH = QH_coeff×Ta + QH_const
    #
    # 展开: [Q*_coeff2×Ta² + Q*_coeff1×Ta + Q*_const] - ΔQ + [QH_coeff×Ta + QH_const] - QE = 0
    # 整理: Q*_coeff2×Ta² + [Q*_coeff1 + QH_coeff]×Ta + [Q*_const - ΔQ + QH_const - QE] = 0

    f_Ta_coeff2 = Q_star_coeffs['coeff2']  # 二次项系数来自Q*
    f_Ta_coeff1 = Q_star_coeffs['coeff1'] + QH_coeffs['coeff']  # 一次项系数
    residual = Q_star_coeffs['const'] - Delta_Q_storage + QH_coeffs['const'] - QE

    # 返回结果
    return {
        # 主要输出（用于街区回归）
        'f_Ta_coeff2': f_Ta_coeff2.astype(np.float32),  # 二次项系数
        'f_Ta_coeff1': f_Ta_coeff1.astype(np.float32),  # 一次项系数
        'residual': residual.astype(np.float32),
        'era5_air_temperature': np.asarray(era5_air_temperature).astype(np.float32),

        # 建筑储热回归特征（用于 ALS 回归估计建筑额外贡献）
        # 在 ALS 中: f(Ta) + β_building·storage_feature = 0
        # storage_feature = Q* (建筑区 LCZ 1-9), 0 (其他区域)
        # 注意：不透水面基础储热已在 storage_heat_flux 中计算
        'storage_feature': storage_feature.astype(np.float32),

        # 分项输出（用于调试和分析）
        'Q_star_coeff2': Q_star_coeffs['coeff2'],
        'Q_star_coeff1': Q_star_coeffs['coeff1'],
        'QH_coeff': QH_coeffs['coeff'],
        'Q_star_const': Q_star_coeffs['const'],
        'storage_heat_flux': Delta_Q_storage.astype(np.float32),  # 完整储热（自然SEBAL + 不透水面β×Q*）
        'soil_heat_flux': Delta_Q_storage.astype(np.float32),     # 保留旧名以兼容
        'latent_heat_flux': QE.astype(np.float32)
    }


def validate_energy_balance(
    f_Ta_coeff2: np.ndarray,
    f_Ta_coeff1: np.ndarray,
    residual: np.ndarray,
    ta_test: float = 298.15
) -> None:
    """
    验证系数计算的合理性

    对于二次方程 f(Ta) = a×Ta² + b×Ta + c = 0
    使用求根公式: Ta = (-b ± sqrt(b² - 4ac)) / (2a)

    参数:
        f_Ta_coeff2: Ta²系数栅格 a (W/m²/K²)
        f_Ta_coeff1: Ta系数栅格 b (W/m²/K)
        residual: 残差栅格 c (W/m²)
        ta_test: 测试用的Ta值 (K)
    """
    print("\n" + "=" * 70)
    print("能量平衡系数验证 (二阶泰勒展开)")
    print("=" * 70)

    print("\n【Ta²系数统计】 (a)")
    print(f"  平均值: {np.nanmean(f_Ta_coeff2):>10.6f} W/m²/K²")
    print(f"  标准差: {np.nanstd(f_Ta_coeff2):>10.6f} W/m²/K²")

    print("\n【Ta系数统计】 (b)")
    print(f"  平均值: {np.nanmean(f_Ta_coeff1):>8.3f} W/m²/K")
    print(f"  标准差: {np.nanstd(f_Ta_coeff1):>8.3f} W/m²/K")
    print(f"  最小值: {np.nanmin(f_Ta_coeff1):>8.3f} W/m²/K")
    print(f"  最大值: {np.nanmax(f_Ta_coeff1):>8.3f} W/m²/K")

    print("\n【残差统计】 (c)")
    print(f"  平均值: {np.nanmean(residual):>8.2f} W/m²")
    print(f"  标准差: {np.nanstd(residual):>8.2f} W/m²")
    print(f"  最小值: {np.nanmin(residual):>8.2f} W/m²")
    print(f"  最大值: {np.nanmax(residual):>8.2f} W/m²")

    # 测试：用Ta_test计算能量平衡
    balance = f_Ta_coeff2 * (ta_test ** 2) + f_Ta_coeff1 * ta_test + residual
    print(f"\n【使用Ta={ta_test:.2f}K时的能量平衡】")
    print(f"  f(Ta) 平均: {np.nanmean(balance):>8.2f} W/m²")
    print(f"  f(Ta) 标准差: {np.nanstd(balance):>8.2f} W/m²")
    print("  理想情况: f(Ta) 应接近 0 W/m²")

    # 使用求根公式估算Ta使能量平衡为0
    # Ta = (-b ± sqrt(b² - 4ac)) / (2a)
    a = f_Ta_coeff2
    b = f_Ta_coeff1
    c = residual

    discriminant = b**2 - 4*a*c

    # 只处理判别式 >= 0 的情况（有实数解）
    valid_mask = (discriminant >= 0) & (np.abs(a) > 1e-10)

    # 计算两个根
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
    Ta_plus = np.where(valid_mask, (-b + sqrt_disc) / (2*a), np.nan)
    Ta_minus = np.where(valid_mask, (-b - sqrt_disc) / (2*a), np.nan)

    # 选择物理上合理的解（在合理温度范围内）
    # 通常选择较接近参考温度的解
    Ta_estimate = np.where(
        (Ta_plus > 273.15) & (Ta_plus < 323.15),
        Ta_plus,
        np.where((Ta_minus > 273.15) & (Ta_minus < 323.15), Ta_minus, np.nan)
    )

    Ta_valid = Ta_estimate[np.isfinite(Ta_estimate) & 
                           (Ta_estimate > 273.15) & (Ta_estimate < 323.15)]

    if len(Ta_valid) > 0:
        print("\n【估算Ta使f(Ta)=0 (二次方程求根)】")
        print(f"  平均Ta: {np.mean(Ta_valid):>8.2f} K ({np.mean(Ta_valid)-273.15:>5.2f}°C)")
        print(f"  标准差: {np.std(Ta_valid):>8.2f} K")
        print(f"  范围: [{np.min(Ta_valid)-273.15:>5.2f}, {np.max(Ta_valid)-273.15:>5.2f}]°C")
        print(f"  有效像素: {len(Ta_valid):,d} ({len(Ta_valid)/Ta_estimate.size*100:.1f}%)")
    else:
        print("\n【警告】没有找到物理上合理的 Ta 解")

    # 判别式统计
    disc_valid = discriminant[valid_mask]
    if len(disc_valid) > 0:
        negative_disc = np.sum(discriminant < 0)
        print(f"\n【判别式统计】")
        print(f"  判别式<0 (无实根): {negative_disc:,d} ({negative_disc/discriminant.size*100:.1f}%)")

    print("=" * 70 + "\n")
