"""
感热通量系数计算模块

计算感热通量关于Ta的系数和常数项。

核心思想:
    QH = ρCp(Ta - Ts)/rah
       = (ρCp/rah) × Ta - (ρCp×Ts/rah)
       = coeff × Ta + const

参考文献:
    Bastiaanssen et al., 1998. SEBAL model formulation.
"""

import numpy as np
from typing import Dict
from .constants import SPECIFIC_HEAT_AIR


class SensibleHeatFluxCalculator:
    """
    感热通量系数计算器
    
    计算感热通量关于Ta的系数和常数项，用于街区回归。
    不需要Ta的具体值。
    
    公式: QH = coeff × Ta + const
    """

    def __init__(
        self,
        surface_temperature: np.ndarray,
        aerodynamic_resistance: np.ndarray,
        air_density: float,
        specific_heat: float = SPECIFIC_HEAT_AIR
    ):
        """
        初始化感热通量系数计算器
        
        参数:
            surface_temperature: 地表温度 Ts (K) - ndarray
            aerodynamic_resistance: rah (s/m) - ndarray
            air_density: 空气密度 ρ (kg/m³) - scalar
                        使用ERA5平均值或参考值
            specific_heat: Cp (J/kg/K) - scalar
        """
        self.Ts = surface_temperature
        self.rah = aerodynamic_resistance
        self.rho = air_density
        self.Cp = specific_heat

    @property
    def sensible_heat_coefficient(self) -> Dict[str, np.ndarray]:
        """
        计算感热通量的系数和常数项
        
        QH = ρCp(Ta - Ts)/rah
           = (ρCp/rah) × Ta - (ρCp×Ts/rah)
        
        返回:
            {
                'coeff': ∂QH/∂Ta = ρCp/rah (W/m²/K),
                'const': -ρCp×Ts/rah (W/m²)
            }
        """
        # 避免除以零
        rah_safe = np.maximum(self.rah, 1.0)
        
        # Ta的系数
        coeff = self.rho * self.Cp / rah_safe
        
        # 常数项
        const = -self.rho * self.Cp * self.Ts / rah_safe
        
        return {
            'coeff': coeff.astype(np.float32),
            'const': const.astype(np.float32)
        }
