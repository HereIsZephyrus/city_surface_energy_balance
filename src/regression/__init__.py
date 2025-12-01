"""
街区级回归分析模块

提供从栅格能量平衡系数到街区级气温反演的完整工具链。

核心组件:
    - aggregator: 栅格到街区的空间聚合
    - als_regression: 交替最小二乘法（ALS）求解
    - spatial_analysis: 空间自相关分析（水平交换项 ΔQ_A）
    - counterfactual_analysis: 反事实分析（自然景观降温贡献）

物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si = 0

    水平交换项 ΔQ_A（空间滞后模型）:
    Ta = μ + ρ·W·Ta + ε

    反事实分析（降温贡献）:
    ΔT_cooling = Ta_counterfactual - Ta_original

参考文献:
    doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
"""

__version__ = '0.3.0'

from .aggregator import DistrictAggregator
from .als_regression import ALSRegression
from .spatial_analysis import SpatialWeightMatrix, analyze_spatial_autocorrelation
from .counterfactual_analysis import (
    CounterfactualAnalyzer,
    CounterfactualResult,
    estimate_cooling_contribution,
    spatial_spillover_analysis,
    LCZ_NATURAL,
    LCZ_BUILT,
)

__all__ = [
    '__version__',
    # 聚合
    'DistrictAggregator',
    # ALS 回归
    'ALSRegression',
    # 空间分析
    'SpatialWeightMatrix',
    'analyze_spatial_autocorrelation',
    # 反事实分析
    'CounterfactualAnalyzer',
    'CounterfactualResult',
    'estimate_cooling_contribution',
    'spatial_spillover_analysis',
    'LCZ_NATURAL',
    'LCZ_BUILT',
]
