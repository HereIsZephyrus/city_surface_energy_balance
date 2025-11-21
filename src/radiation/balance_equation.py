"""
能量平衡方程模块

协调各个能量通量计算器，完成完整的地表能量平衡计算。

能量平衡方程:
    Q* = ΔQSg + QH + QE

其中:
    - Q*: 净辐射 (W/m²)
    - ΔQSg: 土壤热通量 (W/m²)
    - QH: 感热通量 (W/m²)
    - QE: 潜热通量 (W/m²)

基于 SEBAL 模型实现。
"""

from typing import Optional
import numpy as np
from .calc_net_radiation import NetRadiationCalculator
from .calc_soil_heat import SoilHeatFluxCalculator
from .calc_sensible_heat import SensibleHeatFluxCalculator
from .calc_latent_heat import LatentHeatFluxCalculator

def calculate_energy_balance(shortwave_down: np.ndarray,
                            surface_temperature: np.ndarray,
                            air_temperature: float,
                            elevation: np.ndarray,
                            albedo: np.ndarray,
                            ndvi: np.ndarray,
                            aerodynamic_resistance: np.ndarray,
                            surface_resistance: np.ndarray,
                            actual_vapor_pressure: np.ndarray,
                            surface_emissivity: Optional[np.ndarray] = None,
                            impervious_mask: Optional[np.ndarray] = None) -> dict:
    """
    计算完整的能量平衡

    Workflow:
    1. 计算净辐射 Q*
    2. 计算土壤热通量 ΔQSg
    3. 计算感热通量 QH
    4. 计算潜热通量 QE (从能量平衡残差)

    Energy Balance: Q* = ΔQSg + QH + QE

    Parameters:
        shortwave_down: 短波下行辐射 (W/m²) - ndarray
        surface_temperature: 地表温度 (K) - ndarray
        air_temperature: 近地表气温 (K) - scalar
        elevation: 高程 (m) - ndarray
        albedo: 地表反照率 (0-1) - ndarray
        ndvi: 归一化植被指数 - ndarray
        aerodynamic_resistance: 大气湍流热交换阻抗 (s/m) - ndarray
        surface_resistance: 表面阻抗 (s/m) - ndarray
        actual_vapor_pressure: 近地表实际水汽压 (kPa) - ndarray
        surface_emissivity: 地表发射率 (0-1) - ndarray, 可选
        impervious_mask: 不透水面掩膜 - ndarray, 可选

    Returns:
        dict containing:
        - net_radiation: 净辐射 Q* (W/m²)
        - soil_heat_flux: 土壤热通量 ΔQSg (W/m²)
        - sensible_heat_flux: 感热通量 QH (W/m²)
        - latent_heat_flux: 潜热通量 QE (W/m²)
        - energy_balance_residual: 能量平衡残差 (W/m²)
        - longwave_down: 长波下行辐射 (W/m²)
        - longwave_up: 长波上行辐射 (W/m²)
    """
    # Step 1: 计算净辐射
    net_rad_calc = NetRadiationCalculator(
        shortwave_down=shortwave_down,
        surface_temperature=surface_temperature,
        air_temperature=air_temperature,
        elevation=elevation,
        albedo=albedo,
        surface_emissivity=surface_emissivity
    )

    Q_star = net_rad_calc.net_radiation
    L_down = net_rad_calc.longwave_down
    L_up = net_rad_calc.longwave_up

    # Step 2: 计算土壤热通量
    soil_flux_calc = SoilHeatFluxCalculator(
        net_radiation=Q_star,
        surface_temperature=surface_temperature,
        albedo=albedo,
        ndvi=ndvi,
        impervious_mask=impervious_mask
    )

    delta_QSg = soil_flux_calc.soil_heat_flux

    # Step 3: 计算感热通量
    sensible_calc = SensibleHeatFluxCalculator(
        air_temperature=air_temperature,
        surface_temperature=surface_temperature,
        aerodynamic_resistance=aerodynamic_resistance
    )

    QH = sensible_calc.sensible_heat_flux

    # Step 4: 计算潜热通量
    latent_calc = LatentHeatFluxCalculator(
        surface_temperature=surface_temperature,
        actual_vapor_pressure=actual_vapor_pressure,
        aerodynamic_resistance=aerodynamic_resistance,
        surface_resistance=surface_resistance
    )

    QE = latent_calc.latent_heat_flux

    # 能量平衡检验: Q* = ΔQSg + QH + QE
    energy_balance_residual = Q_star - (delta_QSg + QH + QE)

    return {
        'net_radiation': Q_star,
        'soil_heat_flux': delta_QSg,
        'sensible_heat_flux': QH,
        'latent_heat_flux': QE,
        'energy_balance_residual': energy_balance_residual,
        'longwave_down': L_down,
        'longwave_up': L_up,
        'available_energy': Q_star - delta_QSg  # 可用能量
    }


def calculate_evaporative_fraction(latent_heat_flux: np.ndarray,
                                   available_energy: np.ndarray) -> np.ndarray:
    """
    计算蒸散发比 (Evaporative Fraction)

    Formula: EF = λE / (Q* - G)

    Parameters:
        latent_heat_flux: 潜热通量 QE (W/m²)
        available_energy: 可用能量 Q* - G (W/m²)

    Returns:
        蒸散发比 EF (0-1) - ndarray
    """
    # 避免除以零
    available_energy_safe = np.maximum(available_energy, 1.0)

    EF = latent_heat_flux / available_energy_safe

    # 限制在0-1之间
    EF = np.clip(EF, 0.0, 1.0)

    return EF.astype(np.float32)

