"""
潜热通量计算模块

基于 SEBAL 模型计算潜热通量。

公式:
    QE = ρCp(es - ea) / [γ(rah + rs)]

其中:
    - es: 地表温度对应的饱和水汽压 (kPa)
    - ea: 近地表实际水汽压 (kPa)
    - γ: 干湿计常数 ≈ 0.067 kPa/K
    - rah: 大气湍流热交换阻抗 (s/m)
    - rs: 下垫面植被阻抗/表面阻抗 (s/m)
"""

import numpy as np
from .constants import (
    AIR_DENSITY,
    SPECIFIC_HEAT_AIR,
    LATENT_HEAT_VAPORIZATION,
    PSYCHROMETRIC_CONSTANT
)

class LatentHeatFluxCalculator:
    """
    潜热通量计算器

    Formula: QE = ρCp(es - ea) / [γ(rah + rs)]

    where:
    - es: 地表温度对应的饱和水汽压 (kPa)
    - ea: 近地表实际水汽压 (kPa)
    - γ: 干湿计常数 ≈ Cp/(λ×ε) ≈ 0.067 kPa/K
    - rah: 大气湍流热交换阻抗 (s/m)
    - rs: 下垫面植被阻抗/表面阻抗 (s/m)
    """

    def __init__(self,
                 surface_temperature: np.ndarray,
                 actual_vapor_pressure: np.ndarray,
                 aerodynamic_resistance: np.ndarray,
                 surface_resistance: np.ndarray,
                 air_density: float = AIR_DENSITY,
                 specific_heat: float = SPECIFIC_HEAT_AIR,
                 latent_heat: float = LATENT_HEAT_VAPORIZATION):
        """
        初始化潜热通量计算器

        Parameters:
            surface_temperature: 地表温度 Ts (K) - ndarray
            actual_vapor_pressure: 近地表实际水汽压 ea (kPa) - ndarray
            aerodynamic_resistance: 大气湍流热交换阻抗 rah (s/m) - ndarray
            surface_resistance: 表面阻抗 rs (s/m) - ndarray
            air_density: 空气密度 (kg/m³) - scalar
            specific_heat: 空气定压比热容 (J/kg/K) - scalar
            latent_heat: 水的汽化潜热 (J/kg) - scalar
        """
        self.Ts = surface_temperature
        self.ea = actual_vapor_pressure
        self.rah = aerodynamic_resistance
        self.rs = surface_resistance
        self.rho = air_density
        self.Cp = specific_heat
        self.lambda_v = latent_heat

    @property
    def saturation_vapor_pressure(self) -> np.ndarray:
        """
        计算饱和水汽压（Magnus公式）

        Formula: es = 0.6108 × exp[17.27(Ts - 273.15)/(Ts - 35.85)]

        Returns:
            饱和水汽压 es (kPa) - ndarray
        """
        Ts_celsius = self.Ts - 273.15
        es = 0.6108 * np.exp(17.27 * Ts_celsius / (Ts_celsius + 237.3))
        return es

    @property
    def psychrometric_constant(self) -> float:
        """
        计算干湿计常数

        Formula: γ = Cp / (λ × ε) ≈ 0.067 kPa/K
        where ε = 0.622 (水汽分子量/干空气分子量)

        Returns:
            干湿计常数 γ (kPa/K) - scalar
        """
        # 使用标准值（标准大气压条件下）
        return PSYCHROMETRIC_CONSTANT

    @property
    def latent_heat_flux(self) -> np.ndarray:
        """
        计算潜热通量

        Formula: QE = ρCp(es - ea) / [γ(rah + rs)]

        Returns:
            潜热通量 QE (W/m²) - ndarray
        """
        es = self.saturation_vapor_pressure
        gamma = self.psychrometric_constant

        # 避免除以零
        total_resistance = self.rah + self.rs
        total_resistance_safe = np.maximum(total_resistance, 1.0)

        # 计算潜热通量
        # 注意：需要将kPa转换为Pa，并考虑干湿计常数
        # QE = ρCp(es - ea) / [γ(rah + rs)]
        # 这里gamma已经包含了单位转换
        QE = (self.rho * self.Cp * (es - self.ea) * 1000 / 
              (gamma * total_resistance_safe))

        # 限制潜热通量为非负值（蒸发）
        QE = np.maximum(QE, 0.0)

        return QE.astype(np.float32)
