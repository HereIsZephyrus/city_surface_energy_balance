"""
地面/建筑储热通量计算模块

根据 LCZ 类型分类计算储热:
- 自然表面 (LCZ 11-13): 使用 SEBAL 公式直接计算土壤热通量 (ΔQ_Sg)
- 水体 (LCZ 14): 使用固定储热系数 (β_water ≈ 0.40，适用于城市浅湖)
- 不透水面 (LCZ 1-10): 使用基础储热系数 + 建筑额外贡献

储热计算分解:
    ΔQ_storage = ΔQ_impervious + ΔQ_building

    其中:
    - ΔQ_impervious = β_impervious × Q*  (不透水面基础储热，β ≈ 0.35)
    - ΔQ_building: 建筑额外储热贡献，通过 ALS 回归估计

在 ALS 回归框架中:
    f(Ta) + β_building·X_building = 0

    其中:
    - X_building = Q* (对于建筑区 LCZ 1-9)
    - X_building = 0 (对于其他区域)
    - β_building 是待估计的建筑储热系数

公式:
    自然表面 SEBAL: ΔQ_Sg/Q* = (Ts - 273.15)/α × (0.0038α + 0.0074α²) × (1 - 0.98NDVI⁴)
    不透水面基础: ΔQ_impervious = 0.35 × Q*
    水体: ΔQ_water = 0.40 × Q* (城市浅湖，深度<3m)

水体储热系数参考:
    - 浅水体 (< 2m): 0.40-0.50
    - 中等水体 (2-10m): 0.25-0.35
    - 深水体 (> 10m): 0.10-0.20
    武汉城市湖泊（东湖、南湖等）平均深度约2-3m，属于浅-中等水体

参考:
    - SEBAL (Bastiaanssen et al., 1998)
    - Grimmond & Oke (1999): Heat storage in urban areas
    - 晴朗无风条件下城市生态空间对城市降温作用量化模型
"""

import numpy as np
from typing import Tuple

from ..utils import LCZ_NEEDS_STORAGE_REGRESSION, URBAN_LCZ_TYPES

# 不透水面基础储热系数（典型值 0.30-0.40）
IMPERVIOUS_STORAGE_COEFFICIENT = 0.30

# 水体储热系数（城市浅湖，深度约2-3m）
# 浅水体 (< 2m): 0.40-0.50
# 中等水体 (2-10m): 0.25-0.35
# 深水体 (> 10m): 0.10-0.20
WATER_STORAGE_COEFFICIENT = 0.45

# 水体 LCZ 编码
_LCZ_WATER = 14


class StorageHeatFluxCalculator:
    """
    地面/建筑储热通量计算器

    根据 LCZ 类型自动选择计算方法:
    - 自然表面 (LCZ 11-13): SEBAL 公式直接计算 ΔQ_Sg
    - 水体 (LCZ 14): 固定储热系数 × Q* (适用于城市浅湖)
    - 不透水面 (LCZ 1-10): 基础储热系数 × Q*
    - 建筑区 (LCZ 1-9): 额外返回建筑特征用于 ALS 回归
    """

    def __init__(
        self,
        net_radiation: np.ndarray,
        surface_temperature: np.ndarray,
        albedo: np.ndarray,
        ndvi: np.ndarray,
        lcz: np.ndarray,
        impervious_coefficient: float = IMPERVIOUS_STORAGE_COEFFICIENT,
        water_coefficient: float = WATER_STORAGE_COEFFICIENT
    ):
        """
        初始化储热通量计算器

        参数:
            net_radiation: 净辐射 Q* (W/m²)
            surface_temperature: 地表温度 Ts (K)
            albedo: 地表反照率 α (0-1)
            ndvi: 归一化植被指数 (-1 to 1)
            lcz: LCZ分类栅格 (1-14)
            impervious_coefficient: 不透水面基础储热系数 (默认 0.30)
            water_coefficient: 水体储热系数 (默认 0.40，适用于城市浅湖)
                              浅水体 (< 2m): 0.40-0.50
                              中等水体 (2-10m): 0.25-0.35
                              深水体 (> 10m): 0.10-0.20
        """
        self.Q_star = net_radiation
        self.Ts = surface_temperature
        self.albedo = albedo
        self.ndvi = ndvi
        self.lcz = lcz.astype(int)
        self.beta_impervious = impervious_coefficient
        self.beta_water = water_coefficient

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

    def _get_impervious_mask(self) -> np.ndarray:
        """
        获取不透水面的像素掩码 (LCZ 1-10)

        返回:
            bool数组: True = 不透水面，False = 自然表面
        """
        impervious_mask = np.zeros_like(self.lcz, dtype=bool)
        for lcz_val, needs_regression in LCZ_NEEDS_STORAGE_REGRESSION.items():
            if needs_regression:
                impervious_mask |= (self.lcz == lcz_val)
        return impervious_mask

    def _get_building_mask(self) -> np.ndarray:
        """
        获取建筑区的像素掩码 (LCZ 1-9)

        返回:
            bool数组: True = 建筑区，False = 非建筑区
        """
        building_mask = np.zeros_like(self.lcz, dtype=bool)
        for lcz_val in URBAN_LCZ_TYPES:
            building_mask |= (self.lcz == lcz_val)
        return building_mask

    def _get_water_mask(self) -> np.ndarray:
        """
        获取水体的像素掩码 (LCZ 14)

        返回:
            bool数组: True = 水体，False = 非水体
        """
        return self.lcz == _LCZ_WATER

    @property
    def storage_heat_flux(self) -> np.ndarray:
        """
        计算储热通量

        分类处理:
        - 自然表面 (LCZ 11-13): SEBAL 公式
        - 水体 (LCZ 14): β_water × Q* (固定储热系数)
        - 不透水面 (LCZ 1-10): β_impervious × Q*

        返回:
            储热通量 (W/m²)
        """
        # 计算 SEBAL 通量（用于自然表面，不含水体）
        sebal_flux = self._calculate_sebal_flux()

        # 计算不透水面基础储热
        impervious_flux = self.beta_impervious * self.Q_star

        # 计算水体储热（使用固定系数，不用 SEBAL 公式）
        water_flux = self.beta_water * self.Q_star

        # 获取掩码
        impervious_mask = self._get_impervious_mask()
        water_mask = self._get_water_mask()

        # 合并：按优先级处理
        # 1. 不透水面使用固定系数
        # 2. 水体使用水体储热系数
        # 3. 其他自然表面使用 SEBAL
        result = np.where(impervious_mask, impervious_flux, sebal_flux)
        result = np.where(water_mask, water_flux, result)

        # 限制在合理范围
        result = np.clip(result, 0, 0.5 * np.abs(self.Q_star))

        return result.astype(np.float32)

    @property
    def storage_feature(self) -> np.ndarray:
        """
        获取建筑储热回归特征 X_building

        用于 ALS 回归估计建筑额外储热贡献:
            f(Ta) + β_building·X_building = 0

        返回:
            建筑储热特征 (W/m²):
            - 建筑区 (LCZ 1-9): Q* (作为回归特征)
            - 其他区域: 0 (不参与建筑储热回归)

        注意:
            不透水面的基础储热已在 storage_heat_flux 中计算，
            此特征仅用于估计建筑的额外贡献。
        """
        building_mask = self._get_building_mask()
        feature = np.where(building_mask, self.Q_star, 0.0)
        return feature.astype(np.float32)

    def get_flux_and_feature(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时返回储热通量和建筑回归特征

        返回:
            (storage_flux, building_feature):
            - storage_flux: 完整的储热通量（自然表面 SEBAL，不透水面 β×Q*）
            - building_feature: 建筑区的 Q*，用于 ALS 回归估计额外贡献
        """
        return self.storage_heat_flux, self.storage_feature

    @property
    def impervious_storage_flux(self) -> np.ndarray:
        """
        仅获取不透水面的基础储热通量（调试用）

        返回:
            不透水面基础储热 (W/m²): β_impervious × Q* (仅不透水区域)
        """
        impervious_mask = self._get_impervious_mask()
        result = np.where(impervious_mask, self.beta_impervious * self.Q_star, 0.0)
        return result.astype(np.float32)

    @property
    def water_storage_flux(self) -> np.ndarray:
        """
        仅获取水体的储热通量（调试用）

        返回:
            水体储热 (W/m²): β_water × Q* (仅水体区域)
        """
        water_mask = self._get_water_mask()
        result = np.where(water_mask, self.beta_water * self.Q_star, 0.0)
        return result.astype(np.float32)
