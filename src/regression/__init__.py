"""
街区级回归分析模块

提供从栅格能量平衡系数到街区级气温反演的完整工具链。

核心功能:
    1. 栅格到街区的空间聚合
    2. 回归数据准备（特征矩阵构建）
    3. 交替最小二乘法（ALS）求解
    4. 结果导出和可视化

物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si = 0
    
    其中:
    - f(Ta): 能量平衡系数（关于Ta的线性项）
    - X_Fi: 人为热相关特征（LCZ类型、不透水面、人口等）
    - X_Si: 建筑储热相关特征（建筑体积、植被覆盖等）
    - αi, βi: 待求回归系数
    - Ta: 每个街区的气温（待优化）

工作流:
    1. 使用 radiation.calculate_energy_balance_coefficients() 计算栅格系数
    2. 使用 DistrictAggregator 聚合到街区
    3. 使用 DistrictRegressionModel 求解街区Ta

参考文献:
    doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
    doc/DISTRICT_REGRESSION_WORKFLOW.md
"""

__version__ = '0.1.0'

from .district_regression import (
    DistrictAggregator,
    DistrictRegressionModel
)

__all__ = [
    '__version__',
    'DistrictAggregator',
    'DistrictRegressionModel',
]

