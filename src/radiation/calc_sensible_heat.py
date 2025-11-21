"""
感热通量计算模块

基于 SEBAL 模型计算感热通量。

公式:
    QH = ρCp(Ta - Ts)/rah

其中:
    - ρ: 空气密度 (kg/m³)
    - Cp: 空气定压比热容 (J/(kg·K))
    - rah: 大气湍流热交换阻抗 (s/m)
"""

import numpy as np
from typing import Optional
from .constants import AIR_DENSITY, SPECIFIC_HEAT_AIR

class SensibleHeatFluxCalculator:
    """
    感热通量计算器

    Formula: QH = ρCp(Ta - Ts)/rah

    where:
    - ρ: 空气密度 (kg/m³)
    - Cp: 空气定压比热容 (J/kg/K)
    - rah: 大气湍流热交换阻抗 (s/m)
    """

    def __init__(self,
                 air_temperature: float,
                 surface_temperature: np.ndarray,
                 aerodynamic_resistance: np.ndarray,
                 air_density: float = AIR_DENSITY,
                 specific_heat: float = SPECIFIC_HEAT_AIR):
        """
        初始化感热通量计算器

        Parameters:
            air_temperature: 近地表气温 Ta (K) - scalar
            surface_temperature: 地表温度 Ts (K) - ndarray
            aerodynamic_resistance: 大气湍流热交换阻抗 rah (s/m) - ndarray
            air_density: 空气密度 (kg/m³) - scalar
            specific_heat: 空气定压比热容 (J/kg/K) - scalar
        """
        self.Ta = air_temperature
        self.Ts = surface_temperature
        self.rah = aerodynamic_resistance
        self.rho = air_density
        self.Cp = specific_heat

    @property
    def sensible_heat_flux(self) -> np.ndarray:
        """
        计算感热通量

        Formula: QH = ρCp(Ta - Ts)/rah

        Returns:
            感热通量 QH (W/m²) - ndarray
        """
        # 避免除以零
        rah_safe = np.maximum(self.rah, 1.0)

        QH = self.rho * self.Cp * (self.Ta - self.Ts) / rah_safe

        return QH.astype(np.float32)
