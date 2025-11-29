"""
地面/建筑储热通量计算模块

根据 LCZ 类型分类计算储热:
- 自然表面: 使用 SEBAL 公式直接计算土壤热通量 (ΔQ_Sg)
- 不透水面: 返回储热特征 X_storage = Q*，系数 β 在 ALS 回归中估计

在 ALS 回归框架中:
    f(Ta) + β·X_storage = 0

    其中:
    - X_storage = Q* (对于不透水面)
    - X_storage = 0 (对于自然表面，已用 SEBAL 计算)
    - β 是待估计的储热系数

公式:
    自然表面 SEBAL: ΔQ_Sg/Q* = (Ts - 273.15)/α × (0.0038α + 0.0074α²) × (1 - 0.98NDVI⁴)

参考:
    - SEBAL (Bastiaanssen et al., 1998)
    - 晴朗无风条件下城市生态空间对城市降温作用量化模型
"""

import numpy as np
from typing import Tuple

from ..utils import LCZ_NEEDS_STORAGE_REGRESSION


class StorageHeatFluxCalculator:
    """
    地面/建筑储热通量计算器

    根据 LCZ 类型自动选择计算方法:
    - 自然表面: SEBAL 公式直接计算 ΔQ_Sg
    - 不透水面: 返回储热特征 X_storage，系数由 ALS 回归确定
    """

    def __init__(
        self,
        net_radiation: np.ndarray,
        surface_temperature: np.ndarray,
        albedo: np.ndarray,
        ndvi: np.ndarray,
        lcz: np.ndarray
    ):
        """
        初始化储热通量计算器

        参数:
            net_radiation: 净辐射 Q* (W/m²)
            surface_temperature: 地表温度 Ts (K)
            albedo: 地表反照率 α (0-1)
            ndvi: 归一化植被指数 (-1 to 1)
            lcz: LCZ分类栅格 (1-14)
        """
        self.Q_star = net_radiation
        self.Ts = surface_temperature
        self.albedo = albedo
        self.ndvi = ndvi
        self.lcz = lcz.astype(int)

    def _calculate_sebal_flux(self) -> np.ndarray:
        """
        使用 SEBAL 公式计算自然表面的土壤热通量

        公式: ΔQ_Sg = Q* × (Ts-273.15)/α × (0.0038α + 0.0074α²) × (1-0.98NDVI⁴)
        """
        Ts_celsius = self.Ts - 273.15
        albedo_safe = np.maximum(self.albedo, 0.01)

        ratio = (Ts_celsius / albedo_safe *
                (0.0038 * albedo_safe + 0.0074 * albedo_safe**2) *
                (1 - 0.98 * self.ndvi**4))

        delta_Q = self.Q_star * ratio
        return np.clip(delta_Q, 0, 0.5 * np.abs(self.Q_star))

    def _get_regression_mask(self) -> np.ndarray:
        """
        获取需要储热回归的像素掩码

        返回:
            bool数组: True = 不透水面（需要回归），False = 自然表面（用SEBAL）
        """
        regression_mask = np.zeros_like(self.lcz, dtype=bool)
        for lcz_val, needs_regression in LCZ_NEEDS_STORAGE_REGRESSION.items():
            if needs_regression:
                regression_mask |= (self.lcz == lcz_val)
        return regression_mask

    @property
    def storage_heat_flux(self) -> np.ndarray:
        """
        计算自然表面的储热通量（SEBAL）

        注意: 不透水面的储热在此设为 0，其储热系数将在 ALS 回归中估计

        返回:
            储热通量 (W/m²) - 自然表面为 ΔQ_Sg，不透水面为 0
        """
        sebal_flux = self._calculate_sebal_flux()
        regression_mask = self._get_regression_mask()
        result = np.where(regression_mask, 0.0, sebal_flux)
        return result.astype(np.float32)

    @property
    def storage_feature(self) -> np.ndarray:
        """
        获取储热回归特征 X_storage

        用于 ALS 回归: f(Ta) + β·X_storage = 0

        返回:
            储热特征 (W/m²):
            - 不透水面: Q* (作为回归特征)
            - 自然表面: 0 (不参与储热回归)
        """
        regression_mask = self._get_regression_mask()
        feature = np.where(regression_mask, self.Q_star, 0.0)
        return feature.astype(np.float32)

    def get_flux_and_feature(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时返回储热通量和回归特征

        返回:
            (storage_flux, storage_feature):
            - storage_flux: 自然表面的 SEBAL 计算结果，不透水面为 0
            - storage_feature: 不透水面的 Q*，自然表面为 0
        """
        return self.storage_heat_flux, self.storage_feature
