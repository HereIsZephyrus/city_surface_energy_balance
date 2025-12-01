"""
街区聚合器模块

将栅格数据聚合到街区多边形级别。
使用 exactextract 实现高效的 zonal statistics。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from typing import Dict, List, Optional, Union, TYPE_CHECKING
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

if TYPE_CHECKING:
    from rasterio import Affine


# exactextract 统计量名称映射
STAT_NAME_MAP = {
    'mean': 'mean',
    'std': 'stdev',
    'stdev': 'stdev',
    'min': 'min',
    'max': 'max',
    'sum': 'sum',
    'count': 'count',
    'median': 'median',
    'variance': 'variance',
}


class DistrictAggregator:
    """
    街区聚合器

    使用 exactextract 将栅格数据高效聚合到街区多边形
    """

    @staticmethod
    def aggregate_rasters_to_districts(
        rasters: Dict[str, Union[np.ndarray, xr.DataArray]],
        districts_gdf: gpd.GeoDataFrame,
        transform: 'Affine',
        stats: Optional[List[str]] = None,
        nodata: Optional[float] = None  # type: ignore  # 保留参数以保持接口兼容性
    ) -> pd.DataFrame:
        """
        将多个栅格聚合到街区

        参数:
            rasters: 栅格字典 {name: array}
            districts_gdf: 街区GeoDataFrame
            transform: 栅格的仿射变换
            stats: 聚合统计量列表，例如 ['mean', 'std', 'min', 'max']
            nodata: 无数据值（保留参数以保持接口兼容性，NumPyRasterSource 会自动处理 NaN）

        返回:
            DataFrame: 每个街区的聚合统计量
                columns: [district_id, var1_mean, var1_std, var2_mean, ...]
        """
        # 默认统计量
        if stats is None:
            stats = ['mean']
        
        # nodata 参数保留以保持接口兼容性，NumPyRasterSource 会自动处理 NaN
        _ = nodata
        
        # 为每个街区创建ID
        if 'district_id' not in districts_gdf.columns:
            districts_gdf = districts_gdf.copy()
            districts_gdf['district_id'] = range(len(districts_gdf))

        # 转换统计量名称为 exactextract 格式
        ee_stats = [STAT_NAME_MAP.get(s, s) for s in stats]

        # 准备结果
        results = []

        # 对每个栅格进行聚合
        for var_name, raster in rasters.items():
            # 确保是numpy数组
            if isinstance(raster, xr.DataArray):
                raster = raster.values
            
            # 确保是2D数组
            if raster.ndim != 2:
                if raster.ndim == 3 and raster.shape[0] == 1:
                    raster = raster[0, :, :]
                else:
                    raise ValueError(f"栅格 '{var_name}' 必须是2D数组，当前维度: {raster.ndim}, 形状: {raster.shape}")

            # exactextract 需要 float64 类型，不支持 float32
            # 确保是 float64 类型（exactextract 的要求）
            if raster.dtype == np.float32:
                raster = raster.astype(np.float64, copy=False)
            elif not np.issubdtype(raster.dtype, np.floating):
                # 如果不是浮点类型，转换为 float64
                raster = raster.astype(np.float64, copy=False)
            
            # 确保数组是 C-contiguous（exactextract 的要求）
            if not raster.flags['C_CONTIGUOUS']:
                raster = np.ascontiguousarray(raster, dtype=np.float64)

            # 构建 NumPyRasterSource
            # 将 Affine 对象转换为元组格式 (a, b, c, d, e, f)
            if hasattr(transform, 'a'):  # Affine 对象
                transform_tuple = (transform.a, transform.b, transform.c,
                                 transform.d, transform.e, transform.f)
            elif isinstance(transform, (tuple, list)) and len(transform) == 6:
                transform_tuple = tuple(transform)
            else:
                transform_tuple = transform

            # 从 transform 计算边界
            # Affine(a, b, c, d, e, f) 表示:
            # x = a * col + b * row + c
            # y = d * col + e * row + f
            height, width = raster.shape
            xmin = transform_tuple[2]  # c
            ymax = transform_tuple[5]  # f
            xmax = xmin + width * transform_tuple[0]  # c + width * a
            ymin = ymax + height * transform_tuple[4]  # f + height * e
            
            raster_source = NumPyRasterSource(
                raster,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax
            )

            # 使用 exactextract 进行 zonal statistics
            df = exact_extract(
                rast=raster_source,
                vec=districts_gdf,
                ops=ee_stats,
                output="pandas",
                include_cols=[]  # 不包含额外列，避免重复
            )

            # 重命名列：exactextract 输出列名为 mean, stdev 等
            # 转换为 var_name_mean, var_name_std 格式
            rename_map = {}
            for ee_stat, orig_stat in zip(ee_stats, stats):
                rename_map[ee_stat] = f"{var_name}_{orig_stat}"
            df = df.rename(columns=rename_map)

            results.append(df)

        # 合并所有结果
        result_df = pd.concat(results, axis=1)
        result_df['district_id'] = districts_gdf['district_id'].values

        # 将district_id移到第一列
        cols = ['district_id'] + [c for c in result_df.columns if c != 'district_id']
        result_df = result_df[cols]

        return result_df

    @staticmethod
    def aggregate_single_raster(
        raster: Union[np.ndarray, xr.DataArray],
        districts_gdf: gpd.GeoDataFrame,
        transform: 'Affine',
        stats: Optional[List[str]] = None,
        nodata: Optional[float] = None  # type: ignore  # 保留参数以保持接口兼容性
    ) -> pd.DataFrame:
        """
        聚合单个栅格到街区（内存优化版本）

        参数:
            raster: 栅格数组
            districts_gdf: 街区GeoDataFrame
            transform: 栅格的仿射变换
            stats: 聚合统计量列表
            nodata: 无数据值（保留参数以保持接口兼容性，NumPyRasterSource 会自动处理 NaN）

        返回:
            DataFrame: 每个街区的聚合统计量
        """
        # 默认统计量
        if stats is None:
            stats = ['mean']
        
        # nodata 参数保留以保持接口兼容性，NumPyRasterSource 会自动处理 NaN
        _ = nodata
        
        # 确保是numpy数组
        if isinstance(raster, xr.DataArray):
            raster = raster.values
        
        # 确保是2D数组
        if raster.ndim != 2:
            if raster.ndim == 3 and raster.shape[0] == 1:
                raster = raster[0, :, :]
            else:
                raise ValueError(f"栅格必须是2D数组，当前维度: {raster.ndim}, 形状: {raster.shape}")

        # exactextract 需要 float64 类型，不支持 float32
        # 确保是 float64 类型（exactextract 的要求）
        if raster.dtype == np.float32:
            raster = raster.astype(np.float64, copy=False)
        elif not np.issubdtype(raster.dtype, np.floating):
            # 如果不是浮点类型，转换为 float64
            raster = raster.astype(np.float64, copy=False)
        
        # 确保数组是 C-contiguous（exactextract 的要求）
        if not raster.flags['C_CONTIGUOUS']:
            raster = np.ascontiguousarray(raster, dtype=np.float64)

        # 转换统计量名称
        ee_stats = [STAT_NAME_MAP.get(s, s) for s in stats]

        # 构建 NumPyRasterSource
        # 将 Affine 对象转换为元组格式 (a, b, c, d, e, f)
        if hasattr(transform, 'a'):  # Affine 对象
            transform_tuple = (transform.a, transform.b, transform.c,
                             transform.d, transform.e, transform.f)
        elif isinstance(transform, (tuple, list)) and len(transform) == 6:
            transform_tuple = tuple(transform)
        else:
            transform_tuple = transform

        # 从 transform 计算边界
        height, width = raster.shape
        xmin = transform_tuple[2]  # c
        ymax = transform_tuple[5]  # f
        xmax = xmin + width * transform_tuple[0]  # c + width * a
        ymin = ymax + height * transform_tuple[4]  # f + height * e
        
        raster_source = NumPyRasterSource(
            raster,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax
        )

        # 执行 zonal statistics
        df = exact_extract(
            rast=raster_source,
            vec=districts_gdf,
            ops=ee_stats,
            output="pandas",
            include_cols=[]
        )

        return df
