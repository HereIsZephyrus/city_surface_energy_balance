"""
多源栅格数据管理模块

使用xarray管理不同来源、不同分辨率的栅格数据，包括：
- ERA5-Land数据（~11km分辨率）
- Landsat遥感数据（30m分辨率）
- DEM数据（可变分辨率）
- 土地覆盖数据

核心功能:
    1. 统一的栅格数据加载接口
    2. 自动对齐和重采样到目标分辨率
    3. 批量处理多个栅格数据
    4. 坐标系转换和投影

依赖:
    - xarray: 多维标注数组
    - rioxarray: xarray与rasterio的桥梁
    - rasterio: 栅格I/O
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple, Literal
import warnings

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    warnings.warn("未安装rasterio，栅格I/O功能受限")

try:
    import rioxarray as rxr
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False
    warnings.warn("未安装rioxarray，建议安装以获得更好的地理空间支持: pip install rioxarray")


def load_raster(
    filepath: Union[str, Path],
    name: Optional[str] = None,
    band: int = 1,
    chunks: Optional[Dict] = None
) -> xr.DataArray:
    """
    加载栅格文件为xarray.DataArray
    
    使用rioxarray加载GeoTIFF文件，自动保留空间参考信息。
    
    参数:
        filepath: 栅格文件路径（GeoTIFF等）
        name: DataArray的名称（如'LST', 'NDVI'）
               如果为None，使用文件名
        band: 波段号（从1开始）
        chunks: dask chunks配置，用于大文件延迟加载
                例如: {'x': 1000, 'y': 1000}
    
    返回:
        xr.DataArray: 带地理空间元数据的数组
        
    示例:
        >>> lst = load_raster('landsat_lst.tif', name='LST')
        >>> print(lst)
        <xarray.DataArray 'LST' (y: 1000, x: 1000)>
        ...
        Coordinates:
          * x        (x) float64 ...
          * y        (y) float64 ...
            spatial_ref  int64 0
        Attributes:
            _FillValue: nan
            scale_factor: 1.0
    """
    if not HAS_RIOXARRAY:
        raise ImportError(
            "需要安装rioxarray才能加载栅格文件\n"
            "安装命令: pip install rioxarray"
        )
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    # 使用rioxarray加载
    if chunks is not None:
        da = rxr.open_rasterio(filepath, chunks=chunks)
    else:
        da = rxr.open_rasterio(filepath)
    
    # 选择波段
    if 'band' in da.dims:
        da = da.sel(band=band)
    
    # 设置名称
    if name is None:
        name = filepath.stem
    da.name = name
    
    return da


def load_era5_band(
    filepath: Union[str, Path],
    band_name: str,
    band_index: Optional[int] = None
) -> xr.DataArray:
    """
    加载ERA5-Land数据的特定波段
    
    ERA5数据通常是NetCDF格式，包含多个变量。
    
    参数:
        filepath: ERA5文件路径（NetCDF或GeoTIFF）
        band_name: 波段/变量名称
                  例如: 'dewpoint_temperature_2m', 'surface_pressure'
        band_index: 如果是多波段GeoTIFF，指定波段索引
    
    返回:
        xr.DataArray: ERA5波段数据
        
    示例:
        >>> td = load_era5_band('era5_land.nc', 'dewpoint_temperature_2m')
        >>> pressure = load_era5_band('era5_land.nc', 'surface_pressure')
    """
    filepath = Path(filepath)
    
    # 根据文件格式选择加载方式
    if filepath.suffix in ['.nc', '.nc4']:
        # NetCDF格式
        ds = xr.open_dataset(filepath)
        if band_name not in ds.data_vars:
            raise ValueError(
                f"变量'{band_name}'不在数据集中。"
                f"可用变量: {list(ds.data_vars.keys())}"
            )
        da = ds[band_name]
        da.name = band_name
        return da
    
    elif filepath.suffix in ['.tif', '.tiff']:
        # GeoTIFF格式
        if band_index is None:
            raise ValueError("GeoTIFF格式必须指定band_index")
        da = load_raster(filepath, name=band_name, band=band_index)
        return da
    
    else:
        raise ValueError(f"不支持的文件格式: {filepath.suffix}")


class RasterCollection:
    """
    栅格数据集合管理器
    
    管理多个不同来源、不同分辨率的栅格数据，提供统一的访问接口。
    
    主要功能:
        - 添加和管理多个栅格
        - 自动对齐到目标分辨率
        - 坐标系转换
        - 批量处理
    
    示例:
        >>> collection = RasterCollection(target_resolution=30.0)
        >>> collection.add_raster('lst', 'landsat_lst.tif')
        >>> collection.add_raster('ndvi', 'landsat_ndvi.tif')
        >>> collection.add_raster('dem', 'srtm_dem.tif')
        >>> 
        >>> # 对齐所有数据
        >>> aligned = collection.align_all()
        >>> lst_array = aligned['lst'].values  # numpy数组
    """
    
    def __init__(
        self,
        target_resolution: Optional[float] = None,
        target_crs: Optional[Union[str, int]] = None,
        target_bounds: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        初始化栅格集合
        
        参数:
            target_resolution: 目标分辨率（米），如30.0
                             如果为None，使用第一个加载的栅格的分辨率
            target_crs: 目标坐标系，例如 'EPSG:4326' 或 4326
                       如果为None，使用第一个加载的栅格的CRS
            target_bounds: 目标范围 (left, bottom, right, top)
                         如果为None，使用所有栅格的交集
        """
        self.rasters: Dict[str, xr.DataArray] = {}
        self.target_resolution = target_resolution
        self.target_crs = target_crs
        self.target_bounds = target_bounds
        self._reference_raster: Optional[xr.DataArray] = None
    
    def add_raster(
        self,
        name: str,
        source: Union[str, Path, xr.DataArray],
        band: int = 1,
        **load_kwargs
    ) -> None:
        """
        添加栅格到集合
        
        参数:
            name: 栅格名称（用于后续访问）
            source: 文件路径或已加载的DataArray
            band: 如果是文件，指定波段号
            **load_kwargs: 传递给load_raster的其他参数
        """
        if isinstance(source, xr.DataArray):
            da = source
        else:
            da = load_raster(source, name=name, band=band, **load_kwargs)
        
        da.name = name
        self.rasters[name] = da
        
        # 设置参考栅格（第一个添加的）
        if self._reference_raster is None:
            self._reference_raster = da
    
    def add_era5_band(
        self,
        name: str,
        filepath: Union[str, Path],
        band_name: str,
        band_index: Optional[int] = None
    ) -> None:
        """
        添加ERA5-Land波段到集合
        
        参数:
            name: 在集合中的名称（如'dewpoint_temp'）
            filepath: ERA5文件路径
            band_name: ERA5变量名
            band_index: 如果是GeoTIFF，波段索引
        """
        da = load_era5_band(filepath, band_name, band_index)
        da.name = name
        self.rasters[name] = da
        
        if self._reference_raster is None:
            self._reference_raster = da
    
    def get_raster(self, name: str) -> xr.DataArray:
        """获取栅格数据"""
        if name not in self.rasters:
            raise KeyError(f"栅格'{name}'不存在。可用: {list(self.rasters.keys())}")
        return self.rasters[name]
    
    def list_rasters(self) -> List[str]:
        """列出所有栅格名称"""
        return list(self.rasters.keys())
    
    def align_all(
        self,
        method: Literal['nearest', 'bilinear', 'cubic'] = 'bilinear',
        fill_value: Optional[float] = np.nan
    ) -> Dict[str, xr.DataArray]:
        """
        对齐所有栅格到统一的网格
        
        参数:
            method: 重采样方法
                   'nearest': 最近邻（适合分类数据）
                   'bilinear': 双线性插值（适合连续数据）
                   'cubic': 三次卷积（更平滑）
            fill_value: 填充值
        
        返回:
            Dict[str, xr.DataArray]: 对齐后的栅格字典
        """
        if not self.rasters:
            raise ValueError("集合中没有栅格数据")
        
        # 确定目标网格
        target_grid = self._get_target_grid()
        
        aligned = {}
        for name, raster in self.rasters.items():
            aligned[name] = self._align_raster(
                raster, target_grid, method, fill_value
            )
        
        return aligned
    
    def align_to_reference(
        self,
        reference_name: str,
        method: Literal['nearest', 'bilinear', 'cubic'] = 'bilinear'
    ) -> Dict[str, xr.DataArray]:
        """
        对齐所有栅格到指定的参考栅格
        
        参数:
            reference_name: 参考栅格名称
            method: 重采样方法
        
        返回:
            Dict[str, xr.DataArray]: 对齐后的栅格字典
        """
        if reference_name not in self.rasters:
            raise KeyError(f"参考栅格'{reference_name}'不存在")
        
        reference = self.rasters[reference_name]
        aligned = {reference_name: reference}
        
        for name, raster in self.rasters.items():
            if name == reference_name:
                continue
            
            # 使用xarray的interp_like进行插值
            aligned[name] = raster.interp_like(
                reference,
                method=method
            )
        
        return aligned
    
    def to_numpy_dict(
        self,
        aligned: Optional[Dict[str, xr.DataArray]] = None,
        method: str = 'bilinear'
    ) -> Dict[str, np.ndarray]:
        """
        将栅格集合转换为numpy数组字典
        
        参数:
            aligned: 已对齐的栅格字典（如果为None，自动对齐）
            method: 对齐方法
        
        返回:
            Dict[str, np.ndarray]: numpy数组字典
        
        注意:
            确保所有输出都是numpy数组，即使输入是DataArray
        """
        if aligned is None:
            aligned = self.align_all(method=method)
        
        # 确保转换为numpy数组
        result = {}
        for name, da in aligned.items():
            if isinstance(da, xr.DataArray):
                result[name] = da.values
            elif isinstance(da, np.ndarray):
                result[name] = da
            else:
                result[name] = np.array(da)
        
        return result
    
    def to_dataset(
        self,
        aligned: Optional[Dict[str, xr.DataArray]] = None,
        method: str = 'bilinear'
    ) -> xr.Dataset:
        """
        将栅格集合转换为xarray.Dataset
        
        参数:
            aligned: 已对齐的栅格字典（如果为None，自动对齐）
            method: 对齐方法
        
        返回:
            xr.Dataset: 包含所有栅格的数据集
        """
        if aligned is None:
            aligned = self.align_all(method=method)
        
        return xr.Dataset(aligned)
    
    def _get_target_grid(self) -> xr.DataArray:
        """
        获取或创建目标网格
        
        如果设置了target_resolution/crs/bounds，创建新网格
        否则使用参考栅格
        """
        if self._reference_raster is None:
            raise ValueError("没有参考栅格")
        
        # 如果没有特殊要求，直接使用参考栅格
        if (self.target_resolution is None and 
            self.target_crs is None and 
            self.target_bounds is None):
            return self._reference_raster
        
        # TODO: 实现自定义网格创建
        # 目前简化为使用参考栅格
        warnings.warn("自定义目标网格功能尚未完全实现，使用参考栅格")
        return self._reference_raster
    
    def _align_raster(
        self,
        raster: xr.DataArray,
        target: xr.DataArray,
        method: str,
        fill_value: float
    ) -> xr.DataArray:
        """
        对齐单个栅格到目标网格
        """
        # 使用xarray的interp_like
        try:
            aligned = raster.interp_like(
                target,
                method=method,
                kwargs={'fill_value': fill_value}
            )
            aligned.name = raster.name
            return aligned
        except Exception as e:
            warnings.warn(f"对齐栅格'{raster.name}'时出错: {e}")
            # 回退：直接重采样
            return raster.interp(
                x=target.x,
                y=target.y,
                method=method
            )
    
    def save_aligned(
        self,
        output_dir: Union[str, Path],
        aligned: Optional[Dict[str, xr.DataArray]] = None,
        method: str = 'bilinear',
        driver: str = 'GTiff',
        compress: str = 'lzw'
    ) -> None:
        """
        保存对齐后的栅格到文件
        
        参数:
            output_dir: 输出目录
            aligned: 已对齐的栅格（如果为None，自动对齐）
            method: 对齐方法
            driver: 输出格式驱动（默认GeoTIFF）
            compress: 压缩方法
        """
        if not HAS_RIOXARRAY:
            raise ImportError("需要rioxarray来保存栅格文件")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if aligned is None:
            aligned = self.align_all(method=method)
        
        for name, da in aligned.items():
            output_path = output_dir / f"{name}_aligned.tif"
            
            # 使用rioxarray保存
            if hasattr(da, 'rio'):
                da.rio.to_raster(
                    output_path,
                    driver=driver,
                    compress=compress
                )
            else:
                warnings.warn(f"栅格'{name}'没有空间参考信息，跳过保存")
    
    def __repr__(self):
        raster_list = ', '.join(self.rasters.keys())
        return f"RasterCollection({len(self.rasters)} rasters: {raster_list})"


def quick_align(
    rasters: Dict[str, Union[str, Path, xr.DataArray]],
    reference: Optional[str] = None,
    method: str = 'bilinear'
) -> Dict[str, np.ndarray]:
    """
    快速对齐多个栅格的便捷函数
    
    参数:
        rasters: 栅格字典 {名称: 文件路径或DataArray}
        reference: 参考栅格名称（如果为None，使用第一个）
        method: 重采样方法
    
    返回:
        Dict[str, np.ndarray]: 对齐后的numpy数组字典
    
    示例:
        >>> aligned = quick_align({
        ...     'lst': 'landsat_lst.tif',
        ...     'ndvi': 'landsat_ndvi.tif',
        ...     'dem': 'srtm_dem.tif'
        ... })
        >>> lst_array = aligned['lst']
    """
    collection = RasterCollection()
    
    # 添加所有栅格
    for name, source in rasters.items():
        collection.add_raster(name, source)
    
    # 对齐
    if reference:
        aligned = collection.align_to_reference(reference, method=method)
    else:
        aligned = collection.align_all(method=method)
    
    # 转换为numpy
    return {name: da.values for name, da in aligned.items()}


# 向后兼容：保持与原raster_io的接口一致
def load_dem(dem_path: str) -> Tuple[np.ndarray, dict]:
    """
    加载DEM文件（兼容旧接口）
    
    参数:
        dem_path: DEM文件路径
    
    返回:
        (dem_array, metadata): numpy数组和元数据字典
    """
    da = load_raster(dem_path, name='DEM')
    
    # 提取元数据
    metadata = {
        'shape': da.shape,
        'dtype': da.dtype,
    }
    
    # 尝试提取rioxarray属性
    if hasattr(da, 'rio'):
        metadata['transform'] = da.rio.transform()
        metadata['crs'] = da.rio.crs
        metadata['bounds'] = da.rio.bounds()
        metadata['nodata'] = da.rio.nodata
    
    return da.values, metadata


def save_raster(
    array: Union[np.ndarray, xr.DataArray],
    output_path: Union[str, Path],
    metadata: Optional[dict] = None,
    compress: str = 'lzw'
) -> None:
    """
    保存栅格数据到文件
    
    参数:
        array: numpy数组或xarray.DataArray
        output_path: 输出文件路径
        metadata: 元数据字典（如果array是DataArray且有rio属性，可以省略）
        compress: 压缩方法
    """
    if isinstance(array, xr.DataArray):
        # 如果是DataArray且有空间参考，直接保存
        if hasattr(array, 'rio'):
            array.rio.to_raster(output_path, compress=compress)
            return
        else:
            # 转换为numpy
            array = array.values
    
    # numpy数组需要metadata
    if metadata is None:
        raise ValueError("保存numpy数组需要提供metadata")
    
    if not HAS_RASTERIO:
        raise ImportError("需要rasterio来保存栅格文件")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        transform=metadata.get('transform'),
        crs=metadata.get('crs'),
        nodata=metadata.get('nodata', np.nan),
        compress=compress
    ) as dst:
        dst.write(array, 1)

