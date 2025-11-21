"""
净辐射系数计算模块

计算能量平衡中净辐射部分的系数和常数项。

核心思想:
    不需要Ta的具体值，只计算Ta的系数：
    Q* = Q*_coeff × Ta + Q*_const
    
    其中:
    - Q*_coeff = ε₀ × ∂L↓/∂Ta = ε₀ × 4εₐσTa₀³
    - Q*_const = (1-α)S↓ + ε₀×L↓_const - L↑

公式:
    Q* = (1 - α)S↓ + ε₀L↓ - L↑
    L↓ = εₐσTa⁴ ≈ 4εₐσTa₀³ × Ta + const (线性化)

参考文献:
    Laipelt et al., 2021. Long-term monitoring of evapotranspiration using the SEBAL algorithm.
"""

import numpy as np
from typing import Dict
from .constants import STEFAN_BOLTZMANN


class NetRadiationCalculator:
    """
    净辐射系数计算器
    
    计算净辐射关于Ta的系数和常数项，用于街区回归。
    不需要Ta的具体值。
    
    公式: Q* = Q*_coeff × Ta + Q*_const
    """

    def __init__(
        self,
        shortwave_down: np.ndarray,
        surface_temperature: np.ndarray,
        elevation: np.ndarray,
        albedo: np.ndarray,
        surface_emissivity: np.ndarray,
        ta_reference: float = 298.15
    ):
        """
        初始化净辐射系数计算器
        
        参数:
            shortwave_down: 短波下行辐射 S↓ (W/m²) - ndarray
            surface_temperature: 地表温度 Ts (K) - ndarray
            elevation: 高程 (m) - ndarray
            albedo: 地表反照率 α (0-1) - ndarray
            surface_emissivity: 地表发射率 ε₀ (0-1) - ndarray
            ta_reference: 参考气温 Ta₀ (K) - 用于系数线性化
        """
        self.S_down = shortwave_down
        self.Ts = surface_temperature
        self.elevation = elevation
        self.albedo = albedo
        self.epsilon_0 = surface_emissivity
        self.Ta0 = ta_reference

    @property
    def atmospheric_transmissivity(self) -> np.ndarray:
        """
        计算大气透射率（高程修正）
        
        公式: τ_sw = 0.75 + 2×10⁻⁵ × Ele
        
        返回:
            大气透射率 τ_sw (0.1-1.0) - ndarray
        """
        tau_sw = 0.75 + 2e-5 * self.elevation
        return np.clip(tau_sw, 0.1, 1.0)

    @property
    def atmospheric_emissivity(self) -> np.ndarray:
        """
        计算大气发射率
        
        公式: εₐ = 0.85 × (-ln τ_sw)^0.09
        
        返回:
            大气发射率 εₐ (0-1) - ndarray
        """
        tau = self.atmospheric_transmissivity
        # 避免 log(0)
        tau_safe = np.maximum(tau, 0.01)
        epsilon_a = 0.85 * (-np.log(tau_safe)) ** 0.09
        return np.clip(epsilon_a, 0.7, 1.0)

    @property
    def longwave_up(self) -> np.ndarray:
        """
        计算长波上行辐射（不依赖Ta）
        
        公式: L↑ = ε₀σTs⁴
        
        返回:
            长波上行辐射 (W/m²) - ndarray
        """
        return self.epsilon_0 * STEFAN_BOLTZMANN * (self.Ts ** 4)

    @property
    def longwave_down_coefficient(self) -> Dict[str, np.ndarray]:
        """
        计算长波下行辐射的系数和常数项
        
        L↓ = εₐσTa⁴
        线性化: L↓ ≈ 4εₐσTa₀³ × Ta + const
        
        返回:
            {
                'coeff': ∂L↓/∂Ta 在Ta₀处 (W/m²/K),
                'const': 常数项 (W/m²)
            }
        """
        epsilon_a = self.atmospheric_emissivity
        
        # ∂L↓/∂Ta = 4εₐσTa³
        coeff = 4 * epsilon_a * STEFAN_BOLTZMANN * (self.Ta0 ** 3)
        
        # 常数项 = L↓(Ta₀) - coeff × Ta₀ = -3εₐσTa₀⁴
        const = -3 * epsilon_a * STEFAN_BOLTZMANN * (self.Ta0 ** 4)
        
        return {
            'coeff': coeff.astype(np.float32),
            'const': const.astype(np.float32)
        }

    @property
    def net_radiation_coefficient(self) -> Dict[str, np.ndarray]:
        """
        计算净辐射的系数和常数项
        
        Q* = (1-α)S↓ + ε₀L↓ - L↑
           = (1-α)S↓ + ε₀[coeff_L×Ta + const_L] - L↑
           = [ε₀×coeff_L]×Ta + [(1-α)S↓ + ε₀×const_L - L↑]
        
        返回:
            {
                'coeff': ∂Q*/∂Ta (W/m²/K),
                'const': Q*的常数部分 (W/m²)
            }
        """
        L_down = self.longwave_down_coefficient
        L_up = self.longwave_up
        
        # Q*的Ta系数 = ε₀ × ∂L↓/∂Ta
        Q_star_coeff = self.epsilon_0 * L_down['coeff']
        
        # Q*的常数项 = (1-α)S↓ + ε₀×L↓_const - L↑
        Q_star_const = (
            (1 - self.albedo) * self.S_down +
            self.epsilon_0 * L_down['const'] -
            L_up
        )
        
        return {
            'coeff': Q_star_coeff.astype(np.float32),
            'const': Q_star_const.astype(np.float32)
        }
