"""
空间分析模块

提供空间自相关分析、溢出效应分析和反事实分析工具。

核心组件:
    - spatial_analysis: 空间权重矩阵和空间滞后模型（水平交换项 ΔQ_A）
    - overflow_analysis: 自然景观对建成区的降温溢出效应分析
    - counterfactual_analysis: 反事实分析（自然景观降温贡献估算）

空间滞后模型:
    Ta = μ + ρ·W·Ta + ε
    
    其中:
    - Ta: 各街区气温
    - W: 空间权重矩阵（行标准化）
    - ρ: 空间自相关系数
    - μ: 截距
    - ε: 残差

溢出效应分析:
    量化自然景观（绿地、水体）对相邻建成区的降温溢出效应。

反事实分析:
    通过LCZ替换情景模拟，估算自然景观的降温贡献。
    ΔT_cooling = Ta_counterfactual - Ta_original
    
参考文献:
    Anselin, L. (1995). Local Indicators of Spatial Association—LISA.
    Geographical Analysis, 27(2), 93-115.
"""

# 常量
from .constants import (
    LCZ_NATURAL_DEFAULT,
    LCZ_BUILT_DEFAULT,
    LCZ_NATURAL_NUMERIC,
    LCZ_BUILT_NUMERIC,
    LCZ_NAMES,
    LCZ_NATURAL_NAMES,
    LCZ_BUILT_NAMES,
    LCZ_NATURAL_STRING,
)

# 空间分析
from .spatial_analysis import SpatialWeightMatrix, analyze_spatial_autocorrelation

# 溢出效应分析
from .overflow_analysis import (
    OverflowResult,
    analyze_overflow,
)

# 反事实分析
from .counterfactual_analysis import (
    CounterfactualAnalyzer,
    CounterfactualResult,
    estimate_cooling_contribution,
)

__all__ = [
    '__version__',
    # 常量
    'LCZ_NATURAL_DEFAULT',
    'LCZ_BUILT_DEFAULT',
    'LCZ_NATURAL_NUMERIC',
    'LCZ_BUILT_NUMERIC',
    'LCZ_NAMES',
    'LCZ_NATURAL_NAMES',
    'LCZ_BUILT_NAMES',
    'LCZ_NATURAL_STRING',
    # 空间分析
    'SpatialWeightMatrix',
    'analyze_spatial_autocorrelation',
    # 溢出效应分析
    'OverflowResult',
    'analyze_overflow',
    # 反事实分析
    'CounterfactualAnalyzer',
    'CounterfactualResult',
    'estimate_cooling_contribution',
]
