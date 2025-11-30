"""
栅格数据 I/O 模块 - 处理 GeoTIFF 文件的读写

职责:
- 从 GeoTIFF 文件加载 DEM 数据
- 将辐射/通量结果保存为 GeoTIFF 文件
- 处理地理变换和元数据
- 验证 DEM 数据
"""

import os
import logging
from typing import Tuple

import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logger = logging.getLogger(__name__)


def load_dem(dem_path: str) -> Tuple[np.ndarray, dict]:
    """
    加载 DEM 文件

    参数:
        dem_path: DEM 文件路径

    返回:
        (dem_array, metadata): Tuple[np.ndarray, dict]
            - dem_array: DEM 高程数组
            - metadata: 包含 transform, crs, dtype, shape, nodata, bounds

    异常:
        ImportError: 如果未安装 rasterio
        FileNotFoundError: 如果 DEM 文件不存在
    """
    if not HAS_RASTERIO:
        raise ImportError("需要安装 rasterio 才能读取栅格文件。请运行: pip install rasterio")

    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM 文件不存在: {dem_path}")

    logger.info(f"正在加载 DEM 文件: {dem_path}")

    with rasterio.open(dem_path) as src:
        dem_array = src.read(1)  # 读取第一个波段

        metadata = {
            'transform': src.transform,
            'crs': src.crs,
            'dtype': src.dtypes[0],
            'shape': dem_array.shape,
            'nodata': src.nodata,
            'bounds': src.bounds
        }

        logger.info(f"DEM 信息: shape={dem_array.shape}, CRS={src.crs}")

    return dem_array, metadata


def save_radiation(radiation_array: np.ndarray,
                   output_path: str,
                   metadata: dict):
    """
    保存辐射结果为 GeoTIFF 文件

    参数:
        radiation_array: 辐射数组 (W/m²)
        output_path: 输出文件路径
        metadata: 元数据（包含 transform, crs 等）

    注意:
        如果未安装 rasterio，将保存为 NPY 格式作为后备方案
    """
    if not HAS_RASTERIO:
        logger.warning("未安装 rasterio，无法保存 GeoTIFF 格式，仅保存为 NPY")
        npy_path = output_path.replace('.tif', '.npy')
        np.save(npy_path, radiation_array)
        logger.info(f"已保存 NPY 文件: {npy_path}")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"正在保存辐射结果: {output_path}")

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=radiation_array.shape[0],
        width=radiation_array.shape[1],
        count=1,
        dtype=rasterio.float32,
        transform=metadata['transform'],
        crs=metadata['crs'],
        nodata=np.nan
    ) as dst:
        dst.write(radiation_array, 1)

    logger.info(f"辐射数据已保存，最大值={np.nanmax(radiation_array):.2f} W/m²，"
                f"最小值={np.nanmin(radiation_array):.2f} W/m²")

