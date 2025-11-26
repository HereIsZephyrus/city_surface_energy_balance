"""
能量平衡计算工作流

整合能量平衡系数计算流程，提供统一接口。

计算内容:
    - 发射率估算（如未提供）
    - 反照率估算（如未提供）
    - 储热通量（基于LCZ分类计算：自然表面SEBAL / 不透水面储热系数）
    - 能量平衡系数 (f_Ta_coeff, residual)

使用示例:
    >>> from src.radiation import calculate_energy_balance
    >>> coeffs = calculate_energy_balance(data, aero_params)
"""

from typing import Dict
import numpy as np

from .balance_equation import calculate_energy_balance_coefficients, validate_energy_balance
from ..utils import LCZ_IMPERVIOUS, URBAN_LCZ_TYPES, NATURAL_LCZ_TYPES


def calculate_energy_balance(
    data: Dict,
    aero_params: Dict,
    shortwave_down: float = 800.0,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    计算能量平衡系数
    
    整合能量平衡系数计算流程。
    储热通量根据 LCZ 类型自动选择计算方法:
    - 自然表面: SEBAL 公式 (ΔQ_Sg)
    - 不透水面: 储热系数 (ΔQ_Sb)
    
    参数:
        data: 数据字典，需包含:
            - era5: ERA5数据 (temperature_2m)
            - landsat: Landsat数据 (lst, ndvi, [albedo], [emissivity])
            - dem: DEM高程数据
            - lcz: LCZ分类数据 (1-14)
            - shape: 数据形状
        aero_params: 空气动力学参数字典
        shortwave_down: 短波下行辐射 (W/m²)，默认800
        verbose: 是否打印统计信息
    
    返回:
        dict: 能量平衡系数
            - f_Ta_coeff: Ta系数 (W/m²/K)
            - residual: 残差项 (W/m²)
            - storage_heat_flux: 储热通量 (W/m²)
            - 其他分项（用于调试）
    """
    landsat = data['landsat']
    era5 = data['era5']
    lcz = data['lcz'].astype(int)
    
    if verbose:
        print("\n" + "=" * 60)
        print("计算能量平衡系数")
        print("=" * 60)
    
    # 准备短波辐射数组
    sw_down = np.full(data['shape'], shortwave_down)
    
    # 发射率（如果没有则估算）
    if 'emissivity' in landsat and landsat['emissivity'] is not None:
        emissivity = landsat['emissivity']
        if verbose:
            print("使用 Landsat 发射率数据")
    else:
        emissivity = 0.95 + 0.03 * landsat['ndvi']
        emissivity = np.clip(emissivity, 0.9, 0.99)
        if verbose:
            print("从 NDVI 估算发射率")
    
    # 反照率（如果没有则估算）
    if 'albedo' in landsat and landsat['albedo'] is not None:
        albedo = landsat['albedo']
        if verbose:
            print("使用 Landsat 反照率数据")
    else:
        albedo = 0.2 - 0.1 * landsat['ndvi']
        albedo = np.clip(albedo, 0.05, 0.4)
        if verbose:
            print("从 NDVI 估算反照率")
    
    # 统计 LCZ 分布
    if verbose:
        # 不透水面统计
        impervious_mask = np.zeros_like(lcz, dtype=bool)
        for lcz_val, is_imperv in LCZ_IMPERVIOUS.items():
            if is_imperv:
                impervious_mask |= (lcz == lcz_val)
        imperv_ratio = np.nansum(impervious_mask) / np.sum(~np.isnan(lcz.astype(float))) * 100
        
        # 城市/自然分布
        urban_mask = np.isin(lcz, list(URBAN_LCZ_TYPES))
        natural_mask = np.isin(lcz, list(NATURAL_LCZ_TYPES))
        urban_ratio = np.nansum(urban_mask) / np.sum(~np.isnan(lcz.astype(float))) * 100
        natural_ratio = np.nansum(natural_mask) / np.sum(~np.isnan(lcz.astype(float))) * 100
        
        print("LCZ 分布:")
        print(f"  城市建筑 (1-9): {urban_ratio:.1f}%")
        print(f"  自然地表 (10-14): {natural_ratio:.1f}%")
        print(f"  不透水面: {imperv_ratio:.1f}%")
        print("储热计算:")
        print("  自然表面: SEBAL 公式直接计算")
        print("  不透水面: 储热系数在 ALS 回归中估计")
    
    # 计算能量平衡系数（使用 LCZ 分类计算储热通量）
    if verbose:
        print("计算能量平衡系数...")
    
    coeffs = calculate_energy_balance_coefficients(
        shortwave_down=sw_down,
        surface_temperature=landsat['lst'],
        elevation=data['dem'],
        albedo=albedo,
        ndvi=landsat['ndvi'],
        saturation_vapor_pressure=aero_params['saturation_vapor_pressure'],
        actual_vapor_pressure=aero_params['actual_vapor_pressure'],
        aerodynamic_resistance=aero_params['aerodynamic_resistance'],
        surface_resistance=aero_params['surface_resistance'],
        surface_emissivity=emissivity,
        surface_pressure=aero_params['pressure'],
        era5_air_temperature=era5['temperature_2m'],
        lcz=lcz  # 使用 LCZ 进行分类计算
    )
    
    # 验证
    if verbose:
        validate_energy_balance(coeffs['f_Ta_coeff'], coeffs['residual'])
    
    return coeffs
