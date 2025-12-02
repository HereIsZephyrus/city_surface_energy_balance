"""
街区级回归分析模块

提供从栅格能量平衡系数到街区级气温反演的工具链。

核心组件:
    - aggregator: 栅格到街区的空间聚合
    - als_regression: 交替最小二乘法（ALS）求解

物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si = 0

参考文献:
    doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
"""

__version__ = '0.3.0'

from .aggregator import DistrictAggregator
from .als_regression import ALSRegression

__all__ = [
    '__version__',
    # 聚合
    'DistrictAggregator',
    # ALS 回归
    'ALSRegression',
]
