"""
波段映射与LCZ配置

集中管理多源数据的波段索引和LCZ参数映射。

LCZ编码方案（简化版）:
    1-9:  城市建筑类型（Stewart & Oke 2012）
    10:   裸岩/铺装 (原LCZ E)
    11:   密集树木 (原LCZ A)
    12:   灌木/低矮植被 (原LCZ C/D)
    13:   裸土/沙地 (原LCZ F)
    14:   水体 (原LCZ G)
"""

from typing import Dict


# ============================================================================
# ERA5-Land 波段映射
# ============================================================================

ERA5_BANDS: Dict[str, int] = {
    'surface_pressure': 3,         # Pa - 地表气压
    'dewpoint_temperature_2m': 6,  # K - 2米露点温度
    'u_component_of_wind_10m': 5,  # m/s - 10米U风分量
    'v_component_of_wind_10m': 2,  # m/s - 10米V风分量
    'temperature_2m': 4            # K - 2米气温
}


# ============================================================================
# Landsat 波段映射
# ============================================================================

LANDSAT_BANDS: Dict[str, int] = {
    'ndvi': 9,        # 归一化植被指数
    'fvc': 10,        # 植被覆盖度
    'lst': 19,        # K - 地表温度
    'emissivity': 14  # 地表发射率
}


# ============================================================================
# LCZ 类型定义
# ============================================================================

class LCZ:
    """
    Local Climate Zone 类型编码

    城市建筑类型 (1-9):
        1: 密集高层    2: 密集中层    3: 密集低层
        4: 开阔高层    5: 开阔中层    6: 开阔低层
        7: 轻质低层    8: 大型低层    9: 稀疏建筑

    自然/地表类型 (10-14):
        10: 裸岩/铺装 (E)
        11: 密集树木 (A)
        12: 灌木/低矮植被 (C/D)
        13: 裸土/沙地 (F)
        14: 水体 (G)
    """
    # 城市建筑类型
    COMPACT_HIGH = 1      # 密集高层
    COMPACT_MID = 2       # 密集中层
    COMPACT_LOW = 3       # 密集低层
    OPEN_HIGH = 4         # 开阔高层
    OPEN_MID = 5          # 开阔中层
    OPEN_LOW = 6          # 开阔低层
    LIGHTWEIGHT_LOW = 7   # 轻质低层
    LARGE_LOW = 8         # 大型低层
    SPARSELY_BUILT = 9    # 稀疏建筑

    # 自然/地表类型
    BARE_ROCK = 10        # 裸岩/铺装 (原LCZ E)
    DENSE_TREES = 11      # 密集树木 (原LCZ A)
    BUSH_GRASS = 12       # 灌木/低矮植被 (原LCZ C/D)
    BARE_SOIL = 13        # 裸土/沙地 (原LCZ F)
    WATER = 14            # 水体 (原LCZ G)


# ============================================================================
# LCZ 参数映射
# ============================================================================

# LCZ类型 -> 粗糙度长度 z0 (m)
LCZ_ROUGHNESS: Dict[int, float] = {
    # 城市建筑类型
    LCZ.COMPACT_HIGH: 2.0,
    LCZ.COMPACT_MID: 1.5,
    LCZ.COMPACT_LOW: 1.0,
    LCZ.OPEN_HIGH: 1.5,
    LCZ.OPEN_MID: 0.8,
    LCZ.OPEN_LOW: 0.5,
    LCZ.LIGHTWEIGHT_LOW: 0.3,
    LCZ.LARGE_LOW: 0.5,
    LCZ.SPARSELY_BUILT: 0.3,
    # 自然/地表类型
    LCZ.BARE_ROCK: 0.01,
    LCZ.DENSE_TREES: 1.0,
    LCZ.BUSH_GRASS: 0.1,
    LCZ.BARE_SOIL: 0.005,
    LCZ.WATER: 0.001,
}

# LCZ类型 -> 是否为不透水面
LCZ_IMPERVIOUS: Dict[int, bool] = {
    LCZ.COMPACT_HIGH: True,
    LCZ.COMPACT_MID: True,
    LCZ.COMPACT_LOW: True,
    LCZ.OPEN_HIGH: True,
    LCZ.OPEN_MID: True,
    LCZ.OPEN_LOW: True,
    LCZ.LIGHTWEIGHT_LOW: True,
    LCZ.LARGE_LOW: True,
    LCZ.SPARSELY_BUILT: False,  # 稀疏建筑区有较多自然地表
    LCZ.BARE_ROCK: True,        # 铺装地表
    LCZ.DENSE_TREES: False,
    LCZ.BUSH_GRASS: False,
    LCZ.BARE_SOIL: False,
    LCZ.WATER: False,
}

# 城市建筑类型集合 (1-9)
URBAN_LCZ_TYPES = {1, 2, 3, 4, 5, 6, 7, 8, 9}

# 自然地表类型集合 (10-14)
NATURAL_LCZ_TYPES = {10, 11, 12, 13, 14}

# LCZ类型 -> 是否需要建筑储热特征 (用于 ALS 回归)
# True: 不透水面，需要在 ALS 中估计储热系数 β
# False: 自然表面，使用 SEBAL 公式直接计算 ΔQ_Sg
#
# 在 ALS 回归中:
#   f(Ta) + β·X_storage = 0
# 其中 X_storage = Q* (对于不透水面), 0 (对于自然表面)
# β 是待估计的储热系数
LCZ_NEEDS_STORAGE_REGRESSION: Dict[int, bool] = {
    # 城市建筑类型：需要在 ALS 中估计储热系数
    LCZ.COMPACT_HIGH: True,
    LCZ.COMPACT_MID: True,
    LCZ.COMPACT_LOW: True,
    LCZ.OPEN_HIGH: True,
    LCZ.OPEN_MID: True,
    LCZ.OPEN_LOW: True,
    LCZ.LIGHTWEIGHT_LOW: True,
    LCZ.LARGE_LOW: True,
    LCZ.SPARSELY_BUILT: True,   # 稀疏建筑也参与回归
    # 铺装地表：需要在 ALS 中估计储热系数
    LCZ.BARE_ROCK: True,
    # 自然表面：使用 SEBAL 公式，不参与储热回归
    LCZ.DENSE_TREES: False,
    LCZ.BUSH_GRASS: False,
    LCZ.BARE_SOIL: False,
    LCZ.WATER: False,
}


# ============================================================================
# 便捷函数
# ============================================================================

def get_roughness_from_lcz(lcz_value: int) -> float:
    """根据LCZ类型获取粗糙度长度 (m)"""
    return LCZ_ROUGHNESS.get(lcz_value, 0.5)


def is_impervious(lcz_value: int) -> bool:
    """判断LCZ类型是否为不透水面"""
    return LCZ_IMPERVIOUS.get(lcz_value, False)


def is_urban(lcz_value: int) -> bool:
    """判断是否为城市建筑类型"""
    return lcz_value in URBAN_LCZ_TYPES


def is_natural(lcz_value: int) -> bool:
    """判断是否为自然地表类型"""
    return lcz_value in NATURAL_LCZ_TYPES


def needs_storage_regression(lcz_value: int) -> bool:
    """
    判断LCZ类型是否需要在 ALS 回归中估计储热系数

    不透水面: True (储热系数 β 由回归确定)
    自然表面: False (使用 SEBAL 公式直接计算)
    """
    return LCZ_NEEDS_STORAGE_REGRESSION.get(lcz_value, False)


def use_sebal_formula(lcz_value: int) -> bool:
    """
    判断是否应使用SEBAL公式计算土壤热通量

    自然表面使用SEBAL公式，不透水面的储热由回归确定。
    """
    return not needs_storage_regression(lcz_value)


# ============================================================================
# ALS 回归线性特征定义
# ============================================================================
#
# 能量平衡方程:
#   Q* + Q_F = Q_H + Q_E + ΔQ_Sb + ΔQ_Sg + ΔQ_A
#
# 整理为 ALS 回归形式:
#   f(Ta) + Σαi·X_Fi + Σβi·X_Si + γ·X_A = 0
#
# 其中:
#   f(Ta) = coeff_Ta × Ta + residual  (能量平衡的 Ta 相关项)
#   X_Fi: 人为热 Q_F 相关特征
#   X_Si: 建筑储热 ΔQ_Sb 相关特征
#   X_A:  水平交换 ΔQ_A 相关特征
#
# ============================================================================

# 人为热 Q_F 相关特征 (需要从街区属性获取)
# Q_F 受到 LCZ类型、不透水面面积、人口 的影响
ANTHROPOGENIC_HEAT_FEATURES = [
    'population',           # 人口数量
    'building_volume',      # 建筑总体积 (m³)
    'lcz_type',            # LCZ 类型 (分类变量，需 one-hot 编码)
]

# 建筑储热 ΔQ_Sb 相关特征
# ΔQ_Sb 受到 建筑体积、LCZ类型(遮蔽)、植被覆盖度 的影响
# 注意: storage_feature (= Q* for 不透水面) 由能量平衡模块自动计算
STORAGE_HEAT_FEATURES = [
    'storage_feature',      # Q* (不透水面), 0 (自然表面) - 由模块计算
    'building_volume',      # 建筑总体积 (m³)
    'fvc',                  # 植被覆盖度
]

# 水平交换 ΔQ_A 相关特征
# ΔQ_A 代表与相邻 LCZ 的热量交换，在 ALS 中作为待估计的线性项
# 特征值 = 1 (常数项)，系数 γ 由回归确定
# 或者使用邻域温度差等更复杂的特征
HORIZONTAL_EXCHANGE_FEATURES = [
    'neighbor_temp_diff',   # 与相邻街区的温度差 (需要空间计算)
]

# 完整的 ALS 回归特征列表
ALS_FEATURE_GROUPS = {
    'anthropogenic': ANTHROPOGENIC_HEAT_FEATURES,  # α 系数
    'storage': STORAGE_HEAT_FEATURES,              # β 系数
    'horizontal': HORIZONTAL_EXCHANGE_FEATURES,    # γ 系数
}
