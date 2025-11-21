"""
空气动力学相关常量

LCZ分类的Obukhov长度估计等参数
"""

import numpy as np
from typing import Dict

# LCZ类型的Obukhov长度估计 (m)
# 晴朗无风的白天中午时段，地表加热导致大气不稳定，L < 0
LCZ_OBUKHOV_LENGTH: Dict[int, float] = {
    # 城市建筑类型 (LCZ 1-10)
    1: -15.0,   # 密集高层建筑
    2: -20.0,   # 密集中层建筑
    3: -25.0,   # 密集低层建筑
    4: -18.0,   # 开阔高层建筑
    5: -22.0,   # 开阔中层建筑
    6: -28.0,   # 开阔低层建筑
    7: -12.0,   # 轻型低层建筑
    8: -16.0,   # 大型低层建筑
    9: -20.0,   # 稀疏建筑
    10: -14.0,  # 工业区
    
    # 自然地表类型 (LCZ A-G, 编号 11-17)
    11: -35.0,  # LCZ A: 密集树木
    12: -30.0,  # LCZ B: 稀疏树木
    13: -25.0,  # LCZ C: 灌木
    14: -28.0,  # LCZ D: 低矮植被
    15: -22.0,  # LCZ E: 裸地/土壤
    16: -18.0,  # LCZ F: 裸土/沙地
    17: -40.0,  # LCZ G: 水体
}

DEFAULT_OBUKHOV_LENGTH: float = -25.0


def get_obukhov_length_from_lcz(
    lcz: np.ndarray,
    default: float = DEFAULT_OBUKHOV_LENGTH
) -> np.ndarray:
    """
    根据LCZ分类获取Obukhov长度
    
    参数:
        lcz: LCZ类型栅格 (1-17) - ndarray
        default: 默认值 - scalar
    
    返回:
        Obukhov长度栅格 (m) - ndarray
    """
    L = np.full_like(lcz, default, dtype=np.float32)
    
    for lcz_type, obukhov_length in LCZ_OBUKHOV_LENGTH.items():
        mask = (lcz == lcz_type)
        L[mask] = obukhov_length
    
    return L
