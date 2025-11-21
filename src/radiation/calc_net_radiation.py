"""
净辐射计算模块

基于 SEBAL 模型计算地表净辐射通量。

公式:
    Q* = (1 - α)S↓ + ε₀L↓ - L↑

其中:
    - Q*: 净辐射 (W/m²)
    - α: 地表反照率
    - S↓: 短波下行辐射 (W/m²)
    - L↓: 长波下行辐射 = εₐσTₐ⁴
    - L↑: 长波上行辐射 = ε₀σTₛ⁴
    - εₐ: 大气发射率 = 0.85 × (-ln τsw)^0.09
    - ε₀: 地表发射率
    - τsw: 大气透射率 = 0.75 + 2×10⁻⁵×Ele

参考文献:
    Laipelt et al., 2021. Long-term monitoring of evapotranspiration using the SEBAL algorithm.
"""

import numpy as np
from .constants import STEFAN_BOLTZMANN


class NetRadiationCalculator:
    """
    净辐射计算器

    公式: Q* = (1 - α)S↓ + ε₀L↓ - L↑

    组成部分:
    - S↓: 短波下行辐射 (已由 solar_radiation 模块计算)
    - L↓: 长波下行辐射 = εₐσTₐ⁴
    - L↑: 长波上行辐射 = ε₀σTₛ⁴
    - εₐ: 大气发射率 = 0.85 × (-ln(τsw))^0.09
    - τsw: 大气透射率 = 0.75 + 2×10⁻⁵×Elv
    """

    def __init__(self, 
                 shortwave_down: np.ndarray,
                 surface_temperature: np.ndarray,
                 air_temperature: float,
                 elevation: np.ndarray,
                 albedo: np.ndarray,
                 surface_emissivity: np.ndarray):
        """
        初始化净辐射计算器

        参数:
            shortwave_down: 短波下行辐射 S↓ (W/m²) - ndarray
            surface_temperature: 地表温度 Ts (K) - ndarray
            air_temperature: 近地表气温 Ta (K) - scalar
            elevation: 高程 (m) - ndarray
            albedo: 地表反照率 α (0-1) - ndarray
            surface_emissivity: 地表发射率 ε₀ (0-1) - ndarray
        """
        self.S_down = shortwave_down
        self.Ts = surface_temperature
        self.Ta = air_temperature
        self.elevation = elevation
        self.albedo = albedo
        self.epsilon_0 = surface_emissivity

    @property
    def atmospheric_transmissivity(self) -> np.ndarray:
        """
        计算大气透射率（高程修正）

        公式: τ_sw = 0.75 + 2×10⁻⁵ × Ele

        参考: Laipelt et al., 2021

        返回:
            大气透射率 τ_sw (0.1-1.0) - ndarray
        """
        tau_sw = 0.75 + 2e-5 * self.elevation
        return np.clip(tau_sw, 0.1, 1.0)

    @property
    def atmospheric_emissivity(self) -> np.ndarray:
        """
        计算大气发射率

        公式: εₐ = 0.85 × (-ln(τsw))^0.09

        参考: SEBAL 模型

        返回:
            大气发射率 εₐ (0.7-1.0) - ndarray
        """
        tau_sw = self.atmospheric_transmissivity
        # 避免 log(0)
        tau_sw_safe = np.maximum(tau_sw, 1e-6)
        epsilon_a = 0.85 * np.power(-np.log(tau_sw_safe), 0.09)
        return np.clip(epsilon_a, 0.7, 1.0)

    @property
    def longwave_down(self) -> np.ndarray:
        """
        计算长波下行辐射

        公式: L↓ = εₐσTₐ⁴

        返回:
            长波下行辐射 (W/m²) - ndarray
        """
        epsilon_a = self.atmospheric_emissivity
        L_down = epsilon_a * STEFAN_BOLTZMANN * (self.Ta ** 4)
        return L_down

    @property
    def longwave_up(self) -> np.ndarray:
        """
        计算长波上行辐射

        公式: L↑ = ε₀σTₛ⁴

        返回:
            长波上行辐射 (W/m²) - ndarray
        """
        L_up = self.epsilon_0 * STEFAN_BOLTZMANN * (self.Ts ** 4)
        return L_up

    @property
    def net_radiation(self) -> np.ndarray:
        """
        计算净辐射

        公式: Q* = (1 - α)S↓ + ε₀L↓ - L↑

        返回:
            净辐射 Q* (W/m²) - ndarray
        """
        Q_star = ((1 - self.albedo) * self.S_down + 
                  self.epsilon_0 * self.longwave_down - 
                  self.longwave_up)

        return Q_star.astype(np.float32)
