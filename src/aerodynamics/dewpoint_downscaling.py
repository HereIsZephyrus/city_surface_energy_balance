"""
露点温度空间降尺度模块

提供基于土地覆盖类型和植被覆盖度的露点温度空间降尺度功能，
将ERA5-Land的粗分辨率露点温度(~11km)降尺度到高分辨率遥感数据(30-100m)。

核心思想:
    - 不同土地覆盖类型有不同的水汽释放特性
    - 植被覆盖度影响蒸腾和局地湿度
    - 保持区域平均值与ERA5-Land一致（质量守恒）
    - 结合地表特征（反照率、温度）进行精细化调整

土地覆盖类型编码:
    0: 不透水面 (Impervious) - 城市建成区、道路、广场等
    1: 水体 (Water) - 河流、湖泊、水库等
    2: 植被 (Vegetation) - 森林、草地、农田等
    3: 混合 (Mixed) - 过渡区域、低植被覆盖区等

数据来源:
    - ERA5-Land: dewpoint_temperature_2m (K)
    - 遥感数据: FVC, 土地覆盖分类, 反照率, 地表温度

参考文献:
    - doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
    - 基于能量-水分平衡的降尺度理论
"""

import numpy as np
from typing import Optional, Dict, Tuple


def downscale_dewpoint_temperature(
        era5_dewpoint_temperature: float,
        land_cover_classification: np.ndarray,
        fractional_vegetation_cover: np.ndarray,
        albedo: Optional[np.ndarray] = None,
        surface_temperature: Optional[np.ndarray] = None,
        adjustment_intensity: float = 0.5
) -> np.ndarray:
    """
    露点温度空间降尺度（基于土地覆盖和FVC）
    
    将ERA5-Land的粗分辨率露点温度降尺度到高分辨率遥感栅格，
    考虑不同土地类型的水汽释放特性和植被覆盖度的精细化影响。
    
    物理依据:
        - 水体: 蒸发最强，局地水汽含量高，Td偏高
        - 植被: 蒸腾作用，水汽释放取决于FVC，Td略高
        - 不透水面: 干燥，水汽贫乏，Td偏低
        - FVC越高 → 蒸腾越强 → 局地ea和Td越高
    
    参数:
        era5_dewpoint_temperature: ERA5-Land的露点温度 Td (K) - scalar
                                  来自ERA5-Land: 'dewpoint_temperature_2m'
                                  空间分辨率: ~11km
        land_cover_classification: 土地覆盖分类 - ndarray
                                  0=不透水面, 1=水体, 2=植被, 3=混合
                                  高分辨率 (30-100m)
        fractional_vegetation_cover: 植被覆盖度 FVC (0-1) - ndarray
                                    从NDVI计算得到
                                    范围: 0 (无植被) 到 1 (完全覆盖)
        albedo: 地表反照率 (0-1) - ndarray, 可选
               用于不透水面的精细化修正
               高反照率材料（混凝土）vs 低反照率（沥青）
        surface_temperature: 地表温度 Ts (K) - ndarray, 可选
                           用于水体的精细化修正
                           温水体蒸发更强
        adjustment_intensity: 调整强度系数 (0-1) - scalar
                            0 = 不调整（全部使用ERA5值）
                            1 = 最大调整
                            推荐: 0.3-0.5 (保守), 0.5-0.7 (标准)
    
    返回:
        降尺度后的露点温度 Td (K) - ndarray
        与输入的土地覆盖分类具有相同的形状和分辨率
    
    约束:
        1. 质量守恒: 区域平均值 = ERA5-Land原始值
        2. 物理约束: 233.15 K ≤ Td ≤ 313.15 K
        3. 热力学约束: Td ≤ Ts - 0.5 K
    
    注意:
        - 调整强度建议从保守值开始（0.3-0.5）
        - 需要通过敏感性分析和地面观测验证
        - 降尺度引入的不确定性需要量化评估
    
    示例:
        >>> era5_td = 293.15  # 20°C
        >>> land_cover = np.array([[0, 1], [2, 3]])  # 2x2示例
        >>> fvc = np.array([[0.1, 0.0], [0.7, 0.4]])
        >>> td_fine = downscale_dewpoint_temperature(era5_td, land_cover, fvc)
        >>> print(td_fine.mean() - era5_td)  # 应该 ≈ 0（质量守恒）
        0.0
    """
    # 验证输入
    if not (0.0 <= adjustment_intensity <= 1.0):
        raise ValueError(f"adjustment_intensity必须在[0, 1]范围内，当前值: {adjustment_intensity}")
    
    if land_cover_classification.shape != fractional_vegetation_cover.shape:
        raise ValueError(
            f"土地覆盖分类和FVC的形状必须一致: "
            f"{land_cover_classification.shape} vs {fractional_vegetation_cover.shape}"
        )
    
    # 初始化为ERA5值
    td_downscaled = np.full_like(
        fractional_vegetation_cover,
        era5_dewpoint_temperature,
        dtype=np.float32
    )
    
    # ========== 第1步: 基于土地覆盖类型的基准调整 ==========
    
    # 各类型的基准调整量 (K)
    # 基于物理过程的经验值，可根据实际情况调整
    TYPE_ADJUSTMENTS = {
        0: -1.5,  # 不透水面: 干燥，水汽贫乏
        1: +2.0,  # 水体: 高蒸发，水汽充足
        2: +0.5,  # 植被: 蒸腾作用，局地增湿
        3: 0.0    # 混合: 接近区域平均
    }
    
    # 应用类型基准调整
    for land_type, base_adjustment in TYPE_ADJUSTMENTS.items():
        mask = (land_cover_classification == land_type)
        if mask.any():
            td_downscaled[mask] += adjustment_intensity * base_adjustment
    
    # ========== 第2步: 基于FVC的精细化调整 ==========
    
    # 植被和混合区域：FVC越高，蒸腾越强
    vegetation_mask = (land_cover_classification == 2) | (land_cover_classification == 3)
    if vegetation_mask.any():
        # FVC从0.5偏离的程度决定调整量
        # FVC=1.0 → +1.0 K, FVC=0.0 → -1.0 K, FVC=0.5 → 0 K
        fvc_deviation = fractional_vegetation_cover[vegetation_mask] - 0.5
        fvc_adjustment = adjustment_intensity * 1.0 * fvc_deviation
        td_downscaled[vegetation_mask] += fvc_adjustment
    
    # ========== 第3步: 不透水面的反照率修正（可选）==========
    
    if albedo is not None:
        impervious_mask = (land_cover_classification == 0)
        if impervious_mask.any():
            # 高反照率材料（如混凝土）比低反照率（如沥青）略湿
            # 归一化反照率: 中心=0.15, 范围=±0.15
            albedo_normalized = np.clip(
                (albedo[impervious_mask] - 0.15) / 0.15,
                -1.0,
                1.0
            )
            albedo_adjustment = adjustment_intensity * 0.5 * albedo_normalized
            td_downscaled[impervious_mask] += albedo_adjustment
    
    # ========== 第4步: 水体的温度修正（可选）==========
    
    if surface_temperature is not None:
        water_mask = (land_cover_classification == 1)
        if water_mask.any():
            # 温水体蒸发更强: LST每偏离25°C高5K，Td增加约0.5K
            lst_celsius = surface_temperature[water_mask] - 273.15
            lst_adjustment = adjustment_intensity * 0.1 * (lst_celsius - 25.0)
            # 限制调整范围在±2K
            lst_adjustment = np.clip(lst_adjustment, -2.0, 2.0)
            td_downscaled[water_mask] += lst_adjustment
    
    # ========== 第5步: 质量守恒约束 ==========
    
    # 确保区域平均值与ERA5-Land完全一致
    td_mean_before = td_downscaled.mean()
    mass_conservation_correction = td_mean_before - era5_dewpoint_temperature
    td_downscaled -= mass_conservation_correction
    
    # ========== 第6步: 物理约束 ==========
    
    # 温度范围约束
    td_downscaled = np.clip(td_downscaled, 233.15, 313.15)  # -40°C 到 40°C
    
    # 热力学约束: Td不能超过地表温度
    if surface_temperature is not None:
        td_downscaled = np.minimum(td_downscaled, surface_temperature - 0.5)
    
    return td_downscaled.astype(np.float32)


def validate_downscaling(
        td_original: float,
        td_downscaled: np.ndarray,
        land_cover_classification: np.ndarray,
        surface_temperature: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    验证降尺度结果的合理性
    
    检查项目:
        1. 质量守恒: 区域平均值误差
        2. 物理约束: 温度范围、Td < Ts
        3. 空间模式: 各土地类型的调整量
        4. 变异统计: 标准差、范围
    
    参数:
        td_original: ERA5-Land原始露点温度 (K)
        td_downscaled: 降尺度后的露点温度 (K) - ndarray
        land_cover_classification: 土地覆盖分类 - ndarray
        surface_temperature: 地表温度 (K) - ndarray, 可选
    
    返回:
        验证结果字典，包含:
            - mass_conservation_error: 质量守恒误差 (K)
            - valid_range: 温度范围是否合理 (bool)
            - valid_vs_lst: Td < Ts 检验结果 (bool)
            - type_adjustments: 各类型的平均调整量 (dict)
            - spatial_std: 空间标准差 (K)
            - spatial_range: 空间范围 (K)
    """
    results = {}
    
    # 1. 质量守恒
    td_mean = td_downscaled.mean()
    results['mass_conservation_error'] = abs(td_mean - td_original)
    results['mass_conserved'] = results['mass_conservation_error'] < 0.1  # 容差0.1K
    
    # 2. 物理约束
    results['valid_range'] = ((td_downscaled >= 233.15) & 
                              (td_downscaled <= 313.15)).all()
    
    if surface_temperature is not None:
        results['valid_vs_lst'] = (td_downscaled < surface_temperature).all()
    else:
        results['valid_vs_lst'] = None
    
    # 3. 空间模式分析
    land_type_names = {
        0: 'impervious',
        1: 'water',
        2: 'vegetation',
        3: 'mixed'
    }
    
    type_adjustments = {}
    for land_type, name in land_type_names.items():
        mask = (land_cover_classification == land_type)
        if mask.any():
            td_adj = td_downscaled[mask].mean() - td_original
            count = mask.sum()
            type_adjustments[name] = {
                'adjustment': float(td_adj),
                'count': int(count)
            }
    results['type_adjustments'] = type_adjustments
    
    # 4. 空间变异统计
    results['spatial_std'] = float(td_downscaled.std())
    results['spatial_range'] = float(td_downscaled.max() - td_downscaled.min())
    results['spatial_min'] = float(td_downscaled.min())
    results['spatial_max'] = float(td_downscaled.max())
    
    return results


def print_validation_report(validation_results: Dict[str, any]) -> None:
    """
    打印降尺度验证报告
    
    参数:
        validation_results: validate_downscaling()的返回结果
    """
    print("\n" + "=" * 60)
    print("露点温度降尺度验证报告")
    print("=" * 60)
    
    # 质量守恒
    print("\n【1. 质量守恒检验】")
    error = validation_results['mass_conservation_error']
    status = "✓ 通过" if validation_results['mass_conserved'] else "✗ 失败"
    print(f"  误差: {error:.6f} K {status} (容差 < 0.1 K)")
    
    # 物理约束
    print("\n【2. 物理约束检验】")
    range_status = "✓ 通过" if validation_results['valid_range'] else "✗ 失败"
    print(f"  温度范围 [-40°C, 40°C]: {range_status}")
    
    if validation_results['valid_vs_lst'] is not None:
        lst_status = "✓ 通过" if validation_results['valid_vs_lst'] else "✗ 失败"
        print(f"  Td < Ts 约束: {lst_status}")
    
    # 空间模式
    print("\n【3. 各土地类型调整量】")
    type_names_cn = {
        'impervious': '不透水面',
        'water': '水体',
        'vegetation': '植被',
        'mixed': '混合'
    }
    
    for name_en, name_cn in type_names_cn.items():
        if name_en in validation_results['type_adjustments']:
            adj_info = validation_results['type_adjustments'][name_en]
            adj = adj_info['adjustment']
            count = adj_info['count']
            print(f"  {name_cn:8s}: {adj:+.3f} K (n={count:,})")
    
    # 空间变异
    print("\n【4. 空间变异统计】")
    print(f"  标准差: {validation_results['spatial_std']:.3f} K")
    print(f"  范围: {validation_results['spatial_range']:.3f} K")
    print(f"  最小值: {validation_results['spatial_min']:.2f} K "
          f"({validation_results['spatial_min']-273.15:.2f}°C)")
    print(f"  最大值: {validation_results['spatial_max']:.2f} K "
          f"({validation_results['spatial_max']-273.15:.2f}°C)")
    
    print("\n" + "=" * 60 + "\n")


def sensitivity_analysis(
        era5_dewpoint_temperature: float,
        land_cover_classification: np.ndarray,
        fractional_vegetation_cover: np.ndarray,
        adjustment_intensities: Optional[list] = None,
        **kwargs
) -> Dict[float, Tuple[np.ndarray, Dict]]:
    """
    露点温度降尺度的敏感性分析
    
    测试不同adjustment_intensity参数对降尺度结果的影响。
    
    参数:
        era5_dewpoint_temperature: ERA5-Land的露点温度 (K)
        land_cover_classification: 土地覆盖分类
        fractional_vegetation_cover: 植被覆盖度
        adjustment_intensities: 待测试的调整强度列表
                               默认: [0.0, 0.3, 0.5, 0.7, 1.0]
        **kwargs: 其他传递给downscale_dewpoint_temperature的参数
    
    返回:
        字典，键为adjustment_intensity，值为(降尺度结果, 验证结果)元组
    """
    if adjustment_intensities is None:
        adjustment_intensities = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    results = {}
    
    print("\n" + "=" * 60)
    print("露点温度降尺度敏感性分析")
    print("=" * 60)
    
    for intensity in adjustment_intensities:
        print(f"\n测试 adjustment_intensity = {intensity}")
        
        td_downscaled = downscale_dewpoint_temperature(
            era5_dewpoint_temperature=era5_dewpoint_temperature,
            land_cover_classification=land_cover_classification,
            fractional_vegetation_cover=fractional_vegetation_cover,
            adjustment_intensity=intensity,
            **kwargs
        )
        
        validation = validate_downscaling(
            td_original=era5_dewpoint_temperature,
            td_downscaled=td_downscaled,
            land_cover_classification=land_cover_classification,
            surface_temperature=kwargs.get('surface_temperature')
        )
        
        results[intensity] = (td_downscaled, validation)
        
        # 简要输出
        print(f"  空间标准差: {validation['spatial_std']:.3f} K")
        print(f"  空间范围: {validation['spatial_range']:.3f} K")
    
    print("\n" + "=" * 60 + "\n")
    
    return results

