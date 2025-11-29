"""
工具模块

提供日志记录、栅格数据管理、波段映射等通用功能
"""

from .raster_manager import (
    RasterBand,
    RasterCollection,
    RasterData,
)
from .cached_collection import CachedRasterCollection

from .mapping import (
    ERA5_BANDS,
    LANDSAT_BANDS,
    LCZ,
    LCZ_ROUGHNESS,
    LCZ_IMPERVIOUS,
    LCZ_NEEDS_STORAGE_REGRESSION,
    URBAN_LCZ_TYPES,
    NATURAL_LCZ_TYPES,
    get_roughness_from_lcz,
    needs_storage_regression,
    is_impervious,
    is_urban,
    is_natural,
    use_sebal_formula,
    # ALS 回归特征定义
    ANTHROPOGENIC_HEAT_FEATURES,
    STORAGE_HEAT_FEATURES,
    HORIZONTAL_EXCHANGE_FEATURES,
    ALS_FEATURE_GROUPS,
)

__all__ = [
    # 栅格管理
    'RasterBand',
    'RasterCollection',
    'RasterData',
    'CachedRasterCollection',

    # 波段映射
    'ERA5_BANDS',
    'LANDSAT_BANDS',

    # LCZ配置
    'LCZ',
    'LCZ_ROUGHNESS',
    'LCZ_IMPERVIOUS',
    'LCZ_NEEDS_STORAGE_REGRESSION',
    'URBAN_LCZ_TYPES',
    'NATURAL_LCZ_TYPES',
    'get_roughness_from_lcz',
    'needs_storage_regression',
    'is_impervious',
    'is_urban',
    'is_natural',
    'use_sebal_formula',

    # ALS 回归特征定义
    'ANTHROPOGENIC_HEAT_FEATURES',
    'STORAGE_HEAT_FEATURES',
    'HORIZONTAL_EXCHANGE_FEATURES',
    'ALS_FEATURE_GROUPS',
]

