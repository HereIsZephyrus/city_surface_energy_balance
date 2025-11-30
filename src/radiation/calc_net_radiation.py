"""
净辐射系数计算模块

计算能量平衡中净辐射部分的系数和常数项。

核心思想:
    不需要Ta的具体值，计算Ta的二次和一次系数：
    Q* = Q*_coeff2 × Ta² + Q*_coeff1 × Ta + Q*_const

    其中:
    - Q*_coeff2 = ε₀ × 6εₐσTa₀²  (二次项系数)
    - Q*_coeff1 = ε₀ × (-8εₐσTa₀³)  (一次项系数)
    - Q*_const = (1-α)S↓ + ε₀×3εₐσTa₀⁴ - L↑

公式:
    Q* = (1 - α)S↓ + ε₀L↓ - L↑
    L↓ = εₐσTa⁴ 
    
    二阶泰勒展开: L↓ ≈ 6εₐσTa₀²×Ta² - 8εₐσTa₀³×Ta + 3εₐσTa₀⁴
    
    推导:
    L↓(Ta) ≈ L↓(Ta₀) + L↓'(Ta₀)(Ta-Ta₀) + L↓''(Ta₀)/2×(Ta-Ta₀)²
    其中 L↓'(Ta₀) = 4εₐσTa₀³, L↓''(Ta₀) = 12εₐσTa₀²
    展开整理得上式

参考文献:
    Laipelt et al., 2021. Long-term monitoring of evapotranspiration using the SEBAL algorithm.
"""

import numpy as np
from typing import Dict
from .constants import STEFAN_BOLTZMANN


class NetRadiationCalculator:
    """
    净辐射系数计算器

    计算净辐射关于Ta的二次和一次系数，用于街区回归。
    不需要Ta的具体值。

    公式: Q* = Q*_coeff2 × Ta² + Q*_coeff1 × Ta + Q*_const
    
    使用二阶泰勒展开提高L↓的近似精度。
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
        计算长波下行辐射的二阶泰勒展开系数

        L↓ = εₐσTa⁴
        
        二阶泰勒展开:
        L↓(Ta) ≈ L↓(Ta₀) + L↓'(Ta₀)(Ta-Ta₀) + L↓''(Ta₀)/2×(Ta-Ta₀)²
        
        其中:
        - L↓(Ta₀) = εₐσTa₀⁴
        - L↓'(Ta₀) = 4εₐσTa₀³
        - L↓''(Ta₀) = 12εₐσTa₀²
        
        展开整理为标准形式 L↓ = a×Ta² + b×Ta + c:
        - a = 6εₐσTa₀² (L↓''(Ta₀)/2)
        - b = -8εₐσTa₀³ (L↓'(Ta₀) - L↓''(Ta₀)×Ta₀)
        - c = 3εₐσTa₀⁴ (L↓(Ta₀) - L↓'(Ta₀)×Ta₀ + L↓''(Ta₀)/2×Ta₀²)

        返回:
            {
                'coeff2': Ta²系数 (W/m²/K²),
                'coeff1': Ta系数 (W/m²/K),
                'const': 常数项 (W/m²)
            }
        """
        epsilon_a = self.atmospheric_emissivity
        sigma = STEFAN_BOLTZMANN
        Ta0 = self.Ta0

        # 二次项系数: 6εₐσTa₀²
        coeff2 = 6 * epsilon_a * sigma * (Ta0 ** 2)

        # 一次项系数: -8εₐσTa₀³
        coeff1 = -8 * epsilon_a * sigma * (Ta0 ** 3)

        # 常数项: 3εₐσTa₀⁴
        const = 3 * epsilon_a * sigma * (Ta0 ** 4)

        return {
            'coeff2': coeff2.astype(np.float32),
            'coeff1': coeff1.astype(np.float32),
            'const': const.astype(np.float32)
        }

    @property
    def net_radiation_coefficient(self) -> Dict[str, np.ndarray]:
        """
        计算净辐射的二次、一次系数和常数项

        Q* = (1-α)S↓ + ε₀L↓ - L↑
           = (1-α)S↓ + ε₀[a×Ta² + b×Ta + c] - L↑
           = [ε₀×a]×Ta² + [ε₀×b]×Ta + [(1-α)S↓ + ε₀×c - L↑]

        返回:
            {
                'coeff2': ∂²Q*/∂Ta² × 1/2 (W/m²/K²),
                'coeff1': ∂Q*/∂Ta (W/m²/K),
                'const': Q*的常数部分 (W/m²)
            }
        """
        L_down = self.longwave_down_coefficient
        L_up = self.longwave_up

        # Q*的Ta²系数 = ε₀ × L↓的Ta²系数
        Q_star_coeff2 = self.epsilon_0 * L_down['coeff2']

        # Q*的Ta系数 = ε₀ × L↓的Ta系数
        Q_star_coeff1 = self.epsilon_0 * L_down['coeff1']

        # Q*的常数项 = (1-α)S↓ + ε₀×L↓_const - L↑
        Q_star_const = (
            (1 - self.albedo) * self.S_down +
            self.epsilon_0 * L_down['const'] -
            L_up
        )

        return {
            'coeff2': Q_star_coeff2.astype(np.float32),
            'coeff1': Q_star_coeff1.astype(np.float32),
            'const': Q_star_const.astype(np.float32)
        }
