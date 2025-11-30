"""
DEM空间降尺度模块

提供基于ERA5气压约束的DEM空间降尺度功能，
将粗分辨率DEM数据降尺度到高分辨率遥感数据(30-100m)。

核心思想:
    - 使用ERA5每个栅格的气压作为约束
    - 基于等温大气模型进行空间插值
    - 保持与ERA5气压场的物理一致性
    - 结合地形特征进行精细化调整

物理依据:
    - 气压随高度呈指数衰减（等温大气模型）
    - 每个ERA5栅格提供独立的约束条件
    - 局部地形影响气压的空间分布

数据来源:
    - ERA5-Land: surface_pressure (Pa) - 低分辨率约束
    - DEM: 高分辨率地形数据
    - 温度: 用于大气模型参数化

参考文献:
    - doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
    - 基于能量-水分平衡的降尺度理论
    - 等温大气模型在气象学中的应用
"""

import numpy as np
from typing import Union, Optional
from scipy import ndimage


def adjust_pressure_for_elevation(
    era5_pressure: np.ndarray,
    dem_elevation: np.ndarray,
    temperature: Union[np.ndarray, float],
    era5_reference_elevation: Optional[np.ndarray] = None,
    era5_grid_size_meters: float = 11000.0,
    pixel_size_meters: float = 30.0
) -> np.ndarray:
    """
    根据DEM高程订正ERA5气压（使用每个ERA5栅格的约束）

    将高分辨率DEM数据与ERA5气压数据结合，使用等温大气模型订正气压。
    每个ERA5栅格提供独立的参考高程约束，确保空间物理一致性。

    方法说明：
        1. 为每个ERA5栅格计算参考高程（基于DEM在该栅格区域的平均值）
        2. 计算高分辨率DEM与ERA5参考高程的差值
        3. 使用气压高度公式对每个像素进行订正
        4. 保持ERA5栅格尺度的平均气压不变

    参数:
        era5_pressure: ERA5地表气压 (Pa) - ndarray，已对齐到目标分辨率
        dem_elevation: DEM高程 (m) - ndarray，与era5_pressure相同形状
        temperature: 参考温度 (K) - ndarray或scalar，用于气压高度公式
        era5_reference_elevation: ERA5参考高程 (m) - ndarray，可选
                                 如果提供，直接使用（应与dem_elevation相同形状）
                                 如果为None，从DEM计算每个ERA5栅格的平均高程
        era5_grid_size_meters: ERA5网格大小（米），默认11000m（约11km）
        pixel_size_meters: 当前像素大小（米），默认30m（Landsat分辨率）

    返回:
        订正后的气压 (Pa) - ndarray，与输入形状相同

    公式:
        气压高度订正（等温大气模型）:
        P(z) = P0 × exp(-g × Δz / (R_d × T))

        其中:
        - P0: ERA5原始气压（在ERA5栅格尺度上保持不变）
        - Δz: 高程差 = DEM - ERA5参考高程
        - g: 重力加速度 = 9.80665 m/s²
        - R_d: 干空气气体常数 = 287.0 J/(kg·K)
        - T: 参考温度 (K)

    注意:
        - 每个ERA5栅格独立计算参考高程，确保局部物理一致性
        - 高程每升高100m，气压约降低12 hPa（在标准大气条件下）
        - 温度应使用实际大气温度，而不是地表温度

    参考:
        Wallace & Hobbs, 2006. Atmospheric Science: An Introductory Survey.
    """
    # 确保输入是numpy数组
    era5_pressure = np.asarray(era5_pressure, dtype=np.float64)
    dem_elevation = np.asarray(dem_elevation, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64) if isinstance(temperature, np.ndarray) else temperature

    # 检查形状匹配
    if era5_pressure.shape != dem_elevation.shape:
        raise ValueError(
            f"ERA5气压形状 {era5_pressure.shape} 与DEM形状 {dem_elevation.shape} 不匹配"
        )

    # 计算ERA5参考高程
    if era5_reference_elevation is None:
        # 方法：将DEM聚合到ERA5栅格尺度，计算每个ERA5栅格的平均高程
        era5_grid_pixels = int(np.round(era5_grid_size_meters / pixel_size_meters))
        block_size = max(1, era5_grid_pixels)

        # 使用uniform_filter计算每个ERA5栅格区域的平均高程
        try:
            era5_ref_elevation = ndimage.uniform_filter(
                dem_elevation,
                size=block_size,
                mode='constant',
                cval=np.nan
            )
        except ImportError:
            # 如果没有scipy，使用简单的numpy块平均方法
            h, w = dem_elevation.shape
            n_blocks_h = max(1, h // block_size)
            n_blocks_w = max(1, w // block_size)

            if n_blocks_h < 2 or n_blocks_w < 2:
                # 如果块太小，使用全局平均
                era5_ref_elevation = np.full_like(
                    dem_elevation,
                    np.nanmean(dem_elevation),
                    dtype=np.float64
                )
            else:
                # 计算每个块的平均值
                block_h = h // n_blocks_h
                block_w = w // n_blocks_w

                era5_ref_elevation = np.full_like(dem_elevation, np.nan, dtype=np.float64)

                for i in range(n_blocks_h):
                    for j in range(n_blocks_w):
                        i_start = i * block_h
                        i_end = min((i + 1) * block_h, h)
                        j_start = j * block_w
                        j_end = min((j + 1) * block_w, w)

                        block_data = dem_elevation[i_start:i_end, j_start:j_end]
                        block_mean = np.nanmean(block_data)

                        era5_ref_elevation[i_start:i_end, j_start:j_end] = block_mean
        # 方法：将DEM聚合到ERA5栅格尺度，计算每个ERA5栅格的平均高程
        era5_grid_pixels = int(np.round(era5_grid_size_meters / pixel_size_meters))
        block_size = max(1, era5_grid_pixels)

        # 使用uniform_filter计算每个ERA5栅格区域的平均高程
        try:
            era5_ref_elevation = ndimage.uniform_filter(
                dem_elevation,
                size=block_size,
                mode='constant',
                cval=np.nan
            )
        except ImportError:
            # 如果没有scipy，使用简单的numpy块平均方法
            h, w = dem_elevation.shape
            n_blocks_h = max(1, h // block_size)
            n_blocks_w = max(1, w // block_size)

            if n_blocks_h < 2 or n_blocks_w < 2:
                # 如果块太小，使用全局平均
                era5_ref_elevation = np.full_like(
                    dem_elevation,
                    np.nanmean(dem_elevation),
                    dtype=np.float64
                )
            else:
                # 计算每个块的平均值
                block_h = h // n_blocks_h
                block_w = w // n_blocks_w

                era5_ref_elevation = np.full_like(dem_elevation, np.nan, dtype=np.float64)

                for i in range(n_blocks_h):
                    for j in range(n_blocks_w):
                        i_start = i * block_h
                        i_end = min((i + 1) * block_h, h)
                        j_start = j * block_w
                        j_end = min((j + 1) * block_w, w)

                        block_data = dem_elevation[i_start:i_end, j_start:j_end]
                        block_mean = np.nanmean(block_data)

                        era5_ref_elevation[i_start:i_end, j_start:j_end] = block_mean
    else:
        # 使用提供的参考高程
        era5_ref_elevation = np.asarray(era5_reference_elevation)
        if era5_ref_elevation.shape != dem_elevation.shape:
            raise ValueError(
                f"参考高程形状 {era5_ref_elevation.shape} 与DEM形状 {dem_elevation.shape} 不匹配"
            )

    # 计算高程差
    elevation_diff = dem_elevation - era5_ref_elevation

    # 处理NaN值
    valid_mask = ~(np.isnan(era5_pressure) | np.isnan(dem_elevation) |
                   np.isnan(era5_ref_elevation) | np.isnan(elevation_diff))
    if isinstance(temperature, np.ndarray):
        valid_mask = valid_mask & ~np.isnan(temperature)

    # 气压高度订正公式（等温大气模型）
    g = 9.80665  # m/s²
    R_d = 287.0  # J/(kg·K)

    # 初始化结果数组
    pressure_adjusted = era5_pressure.copy()

    # 只在有效区域计算订正
    if isinstance(temperature, np.ndarray):
        # 温度是数组
        pressure_adjusted[valid_mask] = (
            era5_pressure[valid_mask] *
            np.exp(-g * elevation_diff[valid_mask] /
                   (R_d * temperature[valid_mask]))
        )
    else:
        # 温度是标量
        pressure_adjusted[valid_mask] = (
            era5_pressure[valid_mask] *
            np.exp(-g * elevation_diff[valid_mask] / (R_d * temperature))
        )

    return pressure_adjusted.astype(np.float32)