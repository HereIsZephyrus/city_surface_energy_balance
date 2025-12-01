"""
City Surface Energy Balance

城市地表能量平衡计算框架

本项目实现基于系数分解的能量平衡计算，用于街区级近地表气温反演。

核心模块:
    - radiation: 辐射和能量平衡系数计算
    - aerodynamics: 空气动力学参数（粗糙度、阻抗、水汽压等）
    - regression: 街区级回归分析
    - utils: 栅格数据管理等工具
    - landscape: 景观参数计算

工作流:
    1. 数据准备: 使用 utils.RasterCollection 加载多源数据
    2. 参数计算: 使用 aerodynamics 模块计算空气动力学参数
    3. 系数计算: 使用 radiation.calculate_energy_balance_coefficients
    4. 空间聚合: 使用 regression.DistrictAggregator
    5. 回归求解: 使用 regression.ALSRegression

快速开始:
    >>> from src.radiation import calculate_energy_balance_coefficients
    >>> from src.regression import DistrictAggregator, ALSRegression
    >>> from src.utils import RasterCollection

    >>> # Step 1: 加载数据
    >>> collection = RasterCollection()
    >>> collection.add_raster('LST', 'landsat_lst.tif')
    >>> # ... 加载其他数据

    >>> # Step 2: 计算能量平衡系数（栅格）
    >>> coeffs = calculate_energy_balance_coefficients(...)

    >>> # Step 3: 聚合到街区
    >>> aggregator = DistrictAggregator()
    >>> aggregated = aggregator.aggregate_rasters_to_districts(...)

    >>> # Step 4: 回归求解（每个街区一个Ta）
    >>> model = ALSRegression()
    >>> results = model.fit(...)

参考文献:
    doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
    doc/DISTRICT_REGRESSION_WORKFLOW.md
"""

__version__ = '0.3.0'
__author__ = 'City Surface Energy Balance Team'

# 主要模块导入
from . import radiation
from . import aerodynamics
from . import regression
from . import utils
from . import landscape

# 便捷导入：核心函数
from .radiation import (
    calculate_energy_balance_coefficients,
    validate_energy_balance
)

from .regression import (
    DistrictAggregator,
    ALSRegression
)

from .utils import (
    RasterBand,
    RasterCollection,
    RasterData,
)

__all__ = [
    # 版本信息
    '__version__',
    '__author__',

    # 模块
    'radiation',
    'aerodynamics',
    'regression',
    'utils',
    'landscape',

    # 核心函数（便捷访问）
    'calculate_energy_balance_coefficients',
    'validate_energy_balance',
    'DistrictAggregator',
    'ALSRegression',
    'RasterCollection',
    'RasterData',
]

