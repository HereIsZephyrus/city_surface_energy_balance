"""
土壤热通量计算模块

基于 SEBAL 模型计算土壤热通量。

公式:
    ΔQSg/Q* = (Ts - 273.15)/α × (0.0038α + 0.0074α²) × (1 - 0.98NDVI⁴)

注意: 对于不透水面，ΔQSg ≈ 0
"""

import numpy as np
from typing import Optional


class SoilHeatFluxCalculator:
    """
    土壤热通量计算器

    Formula: ΔQSg/Q* = (Ts - 273.15)/α × (0.0038α + 0.0074α²) × (1 - 0.98NDVI⁴)

    Note: 对于不透水面，ΔQSg ≈ 0
    """

    def __init__(self,
                 net_radiation: np.ndarray,
                 surface_temperature: np.ndarray,
                 albedo: np.ndarray,
                 ndvi: np.ndarray,
                 impervious_mask: Optional[np.ndarray] = None):
        """
        初始化土壤热通量计算器

        Parameters:
            net_radiation: 净辐射 Q* (W/m²) - ndarray
            surface_temperature: 地表温度 Ts (K) - ndarray
            albedo: 地表反照率 α (0-1) - ndarray
            ndvi: 归一化植被指数 (-1 to 1) - ndarray
            impervious_mask: 不透水面掩膜 (True=不透水) - ndarray, 可选
        """
        self.Q_star = net_radiation
        self.Ts = surface_temperature
        self.albedo = albedo
        self.ndvi = ndvi
        self.impervious_mask = impervious_mask

    @property
    def soil_heat_flux(self) -> np.ndarray:
        """
        计算土壤热通量

        Formula: 
        - 自然表面: ΔQSg = Q* × (Ts-273.15)/α × (0.0038α + 0.0074α²) × (1-0.98NDVI⁴)
        - 不透水面: ΔQSg ≈ 0

        Returns:
            土壤热通量 ΔQSg (W/m²) - ndarray
        """
        # 温度项（转换为摄氏度）
        Ts_celsius = self.Ts - 273.15

        # 避免除以零
        albedo_safe = np.maximum(self.albedo, 0.01)

        # 计算比例系数
        ratio = (Ts_celsius / albedo_safe * 
                (0.0038 * albedo_safe + 0.0074 * albedo_safe**2) * 
                (1 - 0.98 * self.ndvi**4))

        # 计算土壤热通量
        delta_QSg = self.Q_star * ratio

        # 对于不透水面，设置为0
        if self.impervious_mask is not None:
            delta_QSg = np.where(self.impervious_mask, 0.0, delta_QSg)

        # 限制在合理范围内 (通常为 0.1-0.5 × Q*)
        delta_QSg = np.clip(delta_QSg, 0, 0.5 * self.Q_star)

        return delta_QSg.astype(np.float32)
