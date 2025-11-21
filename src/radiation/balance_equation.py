"""
能量平衡方程模块

计算能量平衡方程的系数和残差，用于街区级回归分析。

核心思想:
    不需要Ta的具体值，将能量平衡分解为：
    f(Ta) = coeff_Ta × Ta + residual = 0
    
    然后在街区聚合后，通过ALS回归求解每个街区的Ta。

能量平衡方程:
    Q* - ΔQSg - QH - QE = 0

分解为:
    [Q*_coeff - QH_coeff]×Ta + [Q*_const - ΔQSg - QH_const - QE] = 0
    即: f_Ta_coeff×Ta + residual = 0

参考:
    doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
"""

import numpy as np
from typing import Dict, Optional
from .calc_net_radiation import NetRadiationCalculator
from .calc_soil_heat import SoilHeatFluxCalculator
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
    P_mean = np.mean(surface_pressure)
    Ta_mean = np.mean(era5_air_temperature)
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
    impervious_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    计算能量平衡方程关于Ta的总系数和残差
    
    将能量平衡分解为：
    f(Ta) = coeff_Ta × Ta + residual = 0
    
    参数:
        shortwave_down: 短波下行辐射 (W/m²) - ndarray
        surface_temperature: 地表温度 Ts (K) - ndarray
        elevation: 高程 (m) - ndarray
        albedo: 反照率 - ndarray
        ndvi: NDVI - ndarray
        saturation_vapor_pressure: es (kPa) - ndarray
        actual_vapor_pressure: ea (kPa) - ndarray
        aerodynamic_resistance: rah (s/m) - ndarray
        surface_resistance: rs (s/m) - ndarray
        surface_emissivity: ε₀ - ndarray
        surface_pressure: P (Pa) - ndarray
        era5_air_temperature: ERA5-Land气温 (K) - ndarray或scalar
                             用于空气密度计算和系数线性化
        impervious_mask: 不透水面mask - ndarray, optional
    
    返回:
        {
            'f_Ta_coeff': 总系数 ∂f/∂Ta (W/m²/K),
            'residual': 残差项 (W/m²),
            'era5_air_temperature': ERA5气温 (K) - 用于Ta初始化和参考
            
            # 分项（调试用）
            'Q_star_coeff': ∂Q*/∂Ta,
            'QH_coeff': ∂QH/∂Ta,
            'Q_star_const': Q*的常数部分,
            'soil_heat_flux': ΔQSg,
            'latent_heat_flux': QE
        }
    
    注意:
        能量平衡方程: f(Ta) = 0
        即: f_Ta_coeff×Ta + residual = 0
    """
    # 步骤1: 计算空气密度参考值（使用ERA5气温）
    rho_ref = calculate_air_density(surface_pressure, era5_air_temperature)
    
    # 获取参考气温（用于系数计算）
    ta_reference = float(np.mean(era5_air_temperature))
    
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
    
    # 步骤4: 计算土壤热通量 ΔQSg（不依赖Ta）
    # 使用参考气温计算Q*₀来估算
    L_up = surface_emissivity * STEFAN_BOLTZMANN * (surface_temperature ** 4)
    epsilon_a = net_rad_calc.atmospheric_emissivity
    L_down_ref = epsilon_a * STEFAN_BOLTZMANN * (ta_reference ** 4)
    
    Q_star_ref = (
        (1 - albedo) * shortwave_down +
        surface_emissivity * L_down_ref -
        L_up
    )
    
    soil_flux_calc = SoilHeatFluxCalculator(
        net_radiation=Q_star_ref,
        surface_temperature=surface_temperature,
        albedo=albedo,
        ndvi=ndvi
    )
    Delta_QSg = soil_flux_calc.soil_heat_flux
    
    # 不透水面的土壤热通量为0
    if impervious_mask is not None:
        Delta_QSg = np.where(impervious_mask, 0.0, Delta_QSg)
    
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
    # 能量平衡方程: Q* - ΔQSg - QH - QE = 0
    # 展开: [Q_star_coeff×Ta + Q_star_const] - ΔQSg - [QH_coeff×Ta + QH_const] - QE = 0
    # 整理: [Q_star_coeff - QH_coeff]×Ta + [Q_star_const - ΔQSg - QH_const - QE] = 0
    
    f_Ta_coeff = Q_star_coeffs['coeff'] - QH_coeffs['coeff']
    residual = Q_star_coeffs['const'] - Delta_QSg - QH_coeffs['const'] - QE
    
    # 返回结果
    return {
        # 主要输出（用于街区回归）
        'f_Ta_coeff': f_Ta_coeff.astype(np.float32),
        'residual': residual.astype(np.float32),
        'era5_air_temperature': np.asarray(era5_air_temperature).astype(np.float32),
        
        # 分项输出（用于调试和分析）
        'Q_star_coeff': Q_star_coeffs['coeff'],
        'Q_star_const': Q_star_coeffs['const'],
        'QH_coeff': QH_coeffs['coeff'],
        'QH_const': QH_coeffs['const'],
        'soil_heat_flux': Delta_QSg.astype(np.float32),
        'latent_heat_flux': QE.astype(np.float32)
    }


def validate_energy_balance(
    f_Ta_coeff: np.ndarray,
    residual: np.ndarray,
    ta_test: float = 298.15
) -> None:
    """
    验证系数计算的合理性
    
    参数:
        f_Ta_coeff: Ta系数栅格 (W/m²/K)
        residual: 残差栅格 (W/m²)
        ta_test: 测试用的Ta值 (K)
    """
    print("\n" + "=" * 70)
    print("能量平衡系数验证")
    print("=" * 70)
    
    print("\n【Ta系数统计】 (∂f/∂Ta)")
    print(f"  平均值: {np.mean(f_Ta_coeff):>8.3f} W/m²/K")
    print(f"  标准差: {np.std(f_Ta_coeff):>8.3f} W/m²/K")
    print(f"  最小值: {np.min(f_Ta_coeff):>8.3f} W/m²/K")
    print(f"  最大值: {np.max(f_Ta_coeff):>8.3f} W/m²/K")
    
    print("\n【残差统计】 (不依赖Ta的部分)")
    print(f"  平均值: {np.mean(residual):>8.2f} W/m²")
    print(f"  标准差: {np.std(residual):>8.2f} W/m²")
    print(f"  最小值: {np.min(residual):>8.2f} W/m²")
    print(f"  最大值: {np.max(residual):>8.2f} W/m²")
    
    # 测试：用Ta_test计算能量平衡
    balance = f_Ta_coeff * ta_test + residual
    print(f"\n【使用Ta={ta_test:.2f}K时的能量平衡】")
    print(f"  f(Ta) 平均: {np.mean(balance):>8.2f} W/m²")
    print(f"  f(Ta) 标准差: {np.std(balance):>8.2f} W/m²")
    print("  理想情况: f(Ta) 应接近 0 W/m²")
    
    # 估算Ta使能量平衡为0
    Ta_estimate = -residual / f_Ta_coeff
    Ta_valid = Ta_estimate[(Ta_estimate > 273.15) & (Ta_estimate < 323.15)]
    
    if len(Ta_valid) > 0:
        print("\n【估算Ta使f(Ta)=0】")
        print(f"  平均Ta: {np.mean(Ta_valid):>8.2f} K ({np.mean(Ta_valid)-273.15:>5.2f}°C)")
        print(f"  标准差: {np.std(Ta_valid):>8.2f} K")
        print(f"  范围: [{np.min(Ta_valid)-273.15:>5.2f}, {np.max(Ta_valid)-273.15:>5.2f}]°C")
    
    print("=" * 70 + "\n")
