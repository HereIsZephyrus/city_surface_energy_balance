"""
LCZ (Local Climate Zone) 常量定义

统一管理 LCZ 分类常量，避免代码重复。

参考文献:
    Stewart, I. D., & Oke, T. R. (2012). Local Climate Zones for urban
    temperature studies. Bulletin of the American Meteorological Society.
"""

from typing import Dict, List

# ============================================================================
# LCZ 数值编码（数据集常用格式）
# ============================================================================

# 自然景观 LCZ（数值型）
LCZ_NATURAL_NUMERIC: List[int] = [11, 12, 13, 14, 17]
"""自然景观 LCZ 数值编码: 11=密集树木, 12=稀疏树木, 13=灌木, 14=低矮植被, 17=水体"""

# 建成区 LCZ（数值型）
LCZ_BUILT_NUMERIC: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
"""建成区 LCZ 数值编码: 1-10"""

# 默认分析用的自然景观 LCZ（不含裸岩/裸土）
LCZ_NATURAL_DEFAULT: List[int] = [11, 12, 13, 14]
"""默认自然景观 LCZ（用于溢出分析）: 树木、灌木、低矮植被"""

LCZ_BUILT_DEFAULT: List[int] = list(range(1, 11))
"""默认建成区 LCZ: 1-10"""

# ============================================================================
# LCZ 名称映射
# ============================================================================

LCZ_NAMES: Dict[int, str] = {
    # 建成区
    1: '紧凑高层',
    2: '紧凑中层',
    3: '紧凑低层',
    4: '开敞高层',
    5: '开敞中层',
    6: '开敞低层',
    7: '轻质低层',
    8: '大型低层',
    9: '稀疏建筑',
    10: '重工业',
    # 自然景观
    11: '密集树木',
    12: '稀疏树木/灌木',
    13: '裸土/沙地',
    14: '低矮植被',
    15: '裸岩/铺面',
    16: '裸土/沙',
    17: '水体',
}

# 字符串格式的 LCZ 名称（部分数据集使用）
LCZ_NATURAL_NAMES: Dict[str, str] = {
    'A': '密集树木 (Dense Trees)',
    'B': '稀疏树木 (Scattered Trees)',
    'C': '灌木 (Bush/Scrub)',
    'D': '低矮植被 (Low Plants)',
    'E': '裸岩/铺面 (Bare Rock/Paved)',
    'F': '裸土/沙 (Bare Soil/Sand)',
    'G': '水体 (Water)',
}

LCZ_BUILT_NAMES: Dict[str, str] = {
    '1': '紧凑高层 (Compact High-rise)',
    '2': '紧凑中层 (Compact Mid-rise)',
    '3': '紧凑低层 (Compact Low-rise)',
    '4': '开敞高层 (Open High-rise)',
    '5': '开敞中层 (Open Mid-rise)',
    '6': '开敞低层 (Open Low-rise)',
    '7': '轻质低层 (Lightweight Low-rise)',
    '8': '大型低层 (Large Low-rise)',
    '9': '稀疏建筑 (Sparsely Built)',
    '10': '重工业 (Heavy Industry)',
}

# 字符串格式的自然景观 LCZ 值
LCZ_NATURAL_STRING: List[str] = ['A', 'B', 'C', 'D', 'G']
"""字符串格式的自然景观 LCZ（不含 E=裸岩, F=裸土）"""

