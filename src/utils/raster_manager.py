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
from rasterio.enums import Resampling
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
import warnings
import rasterio
import rioxarray as rxr
from rasterio.transform import Affine

class RasterBand:
    """
    栅格波段类，封装单个波段的 DataArray 数据

    提供便捷的属性和方法来访问波段数据及其元数据。
    """

    def __init__(self, data: xr.DataArray, name: str):
        """
        初始化栅格波段对象

        参数:
            data: xarray DataArray，包含波段数据
            name: 波段名称
        """
        self.data = data
        # 确保 DataArray 的名称也设置
        self.data.name = name

    @property
    def values(self) -> np.ndarray:
        """获取 numpy 数组值"""
        return self.data.values

    @property
    def shape(self) -> Tuple[int, ...]:
        """获取数据形状"""
        return self.data.shape

    @property
    def dtype(self):
        """获取数据类型"""
        return self.data.dtype

    @property
    def crs(self):
        """获取坐标系"""
        if hasattr(self.data, 'rio'):
            return self.data.rio.crs
        return None

    @property
    def transform(self):
        """获取地理变换参数"""
        if hasattr(self.data, 'rio'):
            return self.data.rio.transform()
        return None

    @property
    def bounds(self):
        """获取边界范围"""
        if hasattr(self.data, 'rio'):
            return self.data.rio.bounds()
        return None

    @property
    def nodata(self):
        """获取 nodata 值"""
        if hasattr(self.data, 'rio'):
            return self.data.rio.nodata
        return None

    def save(
        self,
        output_path: Union[str, Path],
        compress: str = 'lzw',
        driver: str = 'GTiff'
    ) -> None:
        """
        保存波段数据到文件

        参数:
            output_path: 输出文件路径
            compress: 压缩方法
            driver: 输出格式驱动
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(self.data, 'rio'):
            self.data.rio.to_raster(output_path, compress=compress, driver=driver)
        else:
            raise ValueError("无法保存：DataArray 缺少空间参考信息（rio属性）")

    def __repr__(self) -> str:
        return f"RasterBand(name='{self.data.name}', shape={self.shape}, dtype={self.dtype})"

class RasterData:
    """
    栅格数据类，用于加载和管理多波段 TIF 文件

    可以加载完整的 TIF 文件，然后按需获取特定编号的波段。

    示例:
        >>> raster = RasterData('multiband.tif')
        >>> raster.load()  # 加载完整文件
        >>> band1 = raster.get_band(1)  # 获取第1个波段
        >>> band2 = raster.get_band(2)  # 获取第2个波段
    """

    def __init__(self, filepath: Union[str, Path]):
        """
        初始化栅格数据对象

        参数:
            filepath: TIF 文件路径
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"文件不存在: {self.filepath}")

        self._data: Optional[xr.DataArray] = None

    def load(self, chunks: Optional[Dict] = None) -> xr.DataArray:
        """
        加载完整的 TIF 文件（所有波段）

        参数:
            chunks: 分块参数，用于大文件的分块加载

        返回:
            xr.DataArray: 包含所有波段的 DataArray
        """
        if chunks is not None:
            self._data = rxr.open_rasterio(self.filepath, chunks=chunks)
        else:
            self._data = rxr.open_rasterio(self.filepath)

        return self._data

    def is_loaded(self) -> bool:
        """检查数据是否已加载"""
        return self._data is not None

    def get_band(self, band_index: int, name: Optional[str] = None) -> RasterBand:
        """
        获取指定编号的波段

        参数:
            band_index: 波段编号（从1开始）
            name: 可选的波段名称

        返回:
            RasterBand: 单个波段的 RasterBand 对象
        """
        if not self.is_loaded():
            self.load()

        # 检查是否有 band 维度
        if 'band' in self._data.dims:
            band_data = self._data.sel(band=band_index)
        else:
            # 如果没有 band 维度，可能是单波段文件
            if band_index != 1:
                raise ValueError(f"文件只有1个波段，无法获取波段 {band_index}")
            band_data = self._data

        # 设置名称
        if name is None:
            band_name = f"{self.filepath.stem}_band{band_index}"
        else:
            band_name = name

        return RasterBand(band_data, band_name)

    def save(
        self,
        array: Union[np.ndarray, xr.DataArray],
        output_path: Optional[Union[str, Path]] = None,
        compress: str = 'lzw',
        driver: str = 'GTiff'
    ) -> None:
        """
        保存数据到文件

        参数:
            array: 要保存的 numpy 数组或 xarray.DataArray
            output_path: 输出文件路径（如果为None，使用原文件路径）
            compress: 压缩方法
            driver: 输出格式驱动
        """
        if output_path is None:
            output_path = self.filepath
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(array, xr.DataArray):
            # 如果是 DataArray 且有空间参考，直接保存
            if hasattr(array, 'rio'):
                array.rio.to_raster(output_path, compress=compress, driver=driver)
                return
            else:
                # 转换为 numpy
                array = array.values

        # numpy 数组需要 metadata
        if self._data is not None and hasattr(self._data, 'rio'):
            # 从原始数据获取元数据
            metadata = {
                'transform': self._data.rio.transform(),
                'crs': self._data.rio.crs,
                'nodata': self._data.rio.nodata
            }
        else:
            raise ValueError("无法保存：缺少空间参考信息。请先加载数据或提供 metadata。")

        with rasterio.open(
            output_path, 'w',
            driver=driver,
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

    @property
    def data(self) -> Optional[xr.DataArray]:
        """获取完整的数据（所有波段）"""
        if self._data is None:
            self.load()
        return self._data

    @property
    def num_bands(self) -> int:
        """获取波段数量"""
        if self._data is None:
            self.load()

        if 'band' in self._data.dims:
            return len(self._data.band)
        else:
            return 1

    @property
    def shape(self) -> Tuple[int, ...]:
        """获取数据形状"""
        if self._data is None:
            self.load()
        return self._data.shape

    @property
    def crs(self):
        """获取坐标系"""
        if self._data is None:
            self.load()
        if hasattr(self._data, 'rio'):
            return self._data.rio.crs
        return None

    @property
    def transform(self):
        """获取地理变换参数"""
        if self._data is None:
            self.load()
        if hasattr(self._data, 'rio'):
            return self._data.rio.transform()
        return None

class RasterCollection:
    """
    栅格数据集合管理器
    """

    def __init__(
        self,
        target_resolution: float,
        target_crs: Union[str, int],
        target_bounds: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        初始化栅格集合

        参数:
            target_resolution: 目标分辨率（米），如30.0（必需）
            target_crs: 目标坐标系，例如 'EPSG:4326' 或 4326（必需）
            target_bounds: 目标范围 (left, bottom, right, top)
        """
        self.rasters: Dict[str, xr.DataArray] = {}
        self.target_resolution = target_resolution
        self.target_crs = target_crs
        self.target_bounds = target_bounds
        self._reference_bounds: Optional[Tuple[float, float, float, float]] = None
        self._reference_info: Optional[Dict] = None  # 参考图像的完整地理信息
        self._is_georeferenced: bool = False  # 标记是否已正确设置地理参考
        self.original_resolutions: Dict[str, float] = {}  # 记录各栅格的原始分辨率（米）

    def add_raster(
        self,
        name: str,
        source: Union[str, Path, xr.DataArray, RasterData, RasterBand],
        band: int = 1,
        resampling: str = 'bilinear',
        chunks: Optional[Dict] = None
    ) -> None:
        """
        添加栅格到集合，并自动转换到目标坐标系和分辨率

        参数:
            name: 栅格名称（用于后续访问）
            source: 文件路径、RasterData对象、RasterBand对象或已加载的DataArray
            band: 如果是文件或RasterData，指定波段号（从1开始）
            resampling: 重采样方法 ('bilinear', 'nearest', 'cubic')
            chunks: 分块参数，用于大文件的分块加载
        """
        if isinstance(source, RasterBand):
            # 如果传入 RasterBand，直接使用其数据
            da = source.data
        elif isinstance(source, RasterData):
            # 如果传入 RasterData，获取指定波段
            if not source.is_loaded():
                source.load(chunks=chunks)
            raster_band = source.get_band(band, name=name)
            da = raster_band.data
        elif isinstance(source, xr.DataArray):
            # 如果传入 DataArray，直接使用
            da = source
        else:
            # 如果传入文件路径，创建 RasterData 并加载
            raster = RasterData(source)
            raster.load(chunks=chunks)
            raster_band = raster.get_band(band, name=name)
            da = raster_band.data

        # 检查地理参考是否已初始化
        if not self._is_georeferenced:
            raise RuntimeError(
                "地理参考未初始化！在使用 RasterCollection 前必须先调用 set_reference() 方法。"
                "\n请使用 collection.set_reference(reference_image) 设置参考图像。"
            )

        da.name = name

        # 记录原始分辨率（在任何重采样之前）
        if hasattr(da, 'rio'):
            original_res = abs(da.rio.resolution()[0])
            self.original_resolutions[name] = original_res

        # 强制坐标系转换到目标坐标系
        if self.target_crs is not None and hasattr(da, 'rio'):
            current_crs = str(da.rio.crs)
            if current_crs != str(self.target_crs):
                print(f"  转换 {name}: {current_crs} → {self.target_crs}")
                resampling_map = {
                    'bilinear': Resampling.bilinear,
                    'nearest': Resampling.nearest,
                    'cubic': Resampling.cubic
                }
                resampling_method = resampling_map.get(resampling, Resampling.bilinear)
                da = da.rio.reproject(self.target_crs, resampling=resampling_method)
                da.name = name
                # 更新原始分辨率（坐标系转换后）
                self.original_resolutions[name] = abs(da.rio.resolution()[0])

        # 强制对齐到目标网格（无论分辨率是否匹配都需要对齐）
        if self.target_resolution is not None and self._reference_bounds is not None and self._reference_info is not None:
            if hasattr(da, 'rio'):
                current_res = abs(da.rio.resolution()[0])  # 假设正方形像素
                needs_resample = abs(current_res - self.target_resolution) > 1e-6
                
                if needs_resample:
                    print(f"  重采样 {name}: {current_res:.1f}m → {self.target_resolution:.1f}m")
                else:
                    print(f"  对齐 {name} 到目标网格")
                
                # 使用 rio.reproject 进行重采样，确保正确处理边缘插值
                # 这比 interp 更可靠，特别是对于低分辨率数据（如 ERA5）
                ref_info = self._reference_info
                resampling_map = {
                    'bilinear': Resampling.bilinear,
                    'nearest': Resampling.nearest,
                    'cubic': Resampling.cubic
                }
                resampling_method = resampling_map.get(resampling, Resampling.bilinear)
                
                # 使用 reproject 重采样到目标网格
                da = da.rio.reproject(
                    dst_crs=self.target_crs,
                    shape=(ref_info['height'], ref_info['width']),
                    transform=ref_info['transform'],
                    resampling=resampling_method
                )
                da.name = name
                
                # 对于低分辨率数据（如 ERA5），边缘可能仍有 NaN 值
                # 使用该要素的均值填充这些缺失值
                if needs_resample and current_res > self.target_resolution * 10:
                    # 只对分辨率差异大于10倍的数据进行填充（如 ERA5 ~11km vs Landsat 10m）
                    arr = da.values
                    nan_mask = np.isnan(arr)
                    nan_count = np.sum(nan_mask)
                    if nan_count > 0:
                        mean_val = np.nanmean(arr)
                        if np.isfinite(mean_val):
                            arr[nan_mask] = mean_val
                            da = da.copy(data=arr)
                            da.name = name
                            print(f"    填充 {nan_count} 个边缘 NaN 值 (使用均值 {mean_val:.4f})")

        # 确保存储的是 2D 数据（压缩 band 维度）
        if da.ndim == 3 and da.shape[0] == 1:
            da = da.squeeze('band', drop=True)
            da.name = name

        self.rasters[name] = da

    def set_reference(
        self,
        reference_raster: RasterData,
        band: int = 1
    ) -> None:
        """
        设置参考图像，用于定义所有栅格的输出范围和地理参考

        使用参考图像的 extent（范围），但输出尺寸根据 target_resolution 计算。

        参数:
            reference_raster: RasterData 对象，包含参考图像数据
            band: 参考波段编号（从1开始，默认使用第1个波段）
        """
        # 确保数据已加载
        if not reference_raster.is_loaded():
            reference_raster.load()

        # 获取指定波段的数据
        reference_band = reference_raster.get_band(band, name='reference')
        ref_da = reference_band.data

        # 转换到目标坐标系
        if self.target_crs is not None and hasattr(ref_da, 'rio'):
            current_crs = str(ref_da.rio.crs)
            if current_crs != str(self.target_crs):
                ref_da = ref_da.rio.reproject(self.target_crs, resampling=Resampling.bilinear)

        # 设置参考范围和地理信息
        if hasattr(ref_da, 'rio'):
            # 获取参考图像的范围
            bounds = ref_da.rio.bounds()  # (left, bottom, right, top)
            self.target_bounds = bounds
            self._reference_bounds = bounds
            self._is_georeferenced = True
            
            # 根据目标分辨率计算输出尺寸
            left, bottom, right, top = bounds
            width = int(np.ceil((right - left) / self.target_resolution))
            height = int(np.ceil((top - bottom) / self.target_resolution))
            
            # 根据目标分辨率构建新的 transform
            # Affine(a, b, c, d, e, f) = (pixel_width, 0, x_min, 0, -pixel_height, y_max)
            new_transform = Affine(
                self.target_resolution,  # a: pixel width
                0,                        # b: row rotation (0 for north-up)
                left,                     # c: x coordinate of upper-left corner
                0,                        # d: column rotation (0 for north-up)
                -self.target_resolution,  # e: pixel height (negative for north-up)
                top                       # f: y coordinate of upper-left corner
            )

            # 保存完整的参考信息
            self._reference_info = {
                'crs': ref_da.rio.crs,
                'transform': new_transform,
                'height': height,
                'width': width,
                'bounds': bounds,
                'nodata': ref_da.rio.nodata
            }

            # 获取参考图像的原始分辨率用于显示
            original_res = abs(ref_da.rio.resolution()[0])
            original_shape = ref_da.shape
            if len(original_shape) == 3:
                orig_h, orig_w = original_shape[1], original_shape[2]
            else:
                orig_h, orig_w = original_shape[0], original_shape[1]

            print(f"✓ 设置参考范围: {bounds}")
            print(f"  坐标系: {ref_da.rio.crs}")
            print(f"  参考图像: {orig_h} x {orig_w} @ {original_res:.1f}m")
            print(f"  目标输出: {height} x {width} @ {self.target_resolution:.1f}m")
        else:
            raise ValueError("参考图像必须包含地理空间信息（rio属性）")

    def load_multiband(
        self,
        filepath: Union[str, Path],
        bands: Dict[str, int],
        prefix: str = '',
        resampling: str = 'bilinear'
    ) -> None:
        """
        批量加载多波段数据到集合

        参数:
            filepath: 栅格文件路径
            bands: 波段名称到波段号的映射，如 {'ndvi': 9, 'lst': 19}
            prefix: 栅格名称前缀，如 'landsat' 会生成 'landsat_ndvi'
            resampling: 重采样方法
        """
        for name, band in bands.items():
            full_name = f"{prefix}_{name}" if prefix else name
            self.add_raster(full_name, filepath, band=band, resampling=resampling)

    def get_raster(self, name: str) -> RasterBand:
        """获取栅格数据"""
        if name not in self.rasters:
            raise KeyError(f"栅格'{name}'不存在。可用: {list(self.rasters.keys())}")
        return RasterBand(self.rasters[name], name)

    def get_array(self, name: str, dtype: type = np.float64) -> np.ndarray:
        """
        获取指定栅格的 numpy 数组

        参数:
            name: 栅格名称
            dtype: 目标数据类型（默认 np.float64）

        返回:
            np.ndarray: 2D numpy 数组

        抛出:
            KeyError: 如果栅格不存在
        """
        if name not in self.rasters:
            raise KeyError(f"栅格'{name}'不存在。可用: {list(self.rasters.keys())}")
        
        return self.rasters[name].values.astype(dtype)

    def list_rasters(self) -> List[str]:
        """列出所有栅格名称"""
        return list(self.rasters.keys())

    def to_numpy_dict(
        self,
        squeeze_bands: bool = True,
        dtype: Optional[type] = None,
        nodata_values: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        将栅格集合转换为numpy数组字典

        注意: 所有栅格在 add_raster 时已自动对齐，无需再次对齐

        参数:
            squeeze_bands: 是否将单波段3D数组压缩为2D（默认True）
            dtype: 目标数据类型（如 np.float64），None表示保持原类型
            nodata_values: 特定栅格的nodata值映射，如 {'dem': -999}，
                          这些值会被转换为 np.nan

        返回:
            Dict[str, np.ndarray]: numpy数组字典
        """
        if nodata_values is None:
            nodata_values = {}

        result = {}
        for name, da in self.rasters.items():
            # 转换为numpy数组
            if isinstance(da, xr.DataArray):
                values = da.values
            elif isinstance(da, np.ndarray):
                values = da
            else:
                values = np.array(da)

            # 压缩单波段维度 (band, y, x) -> (y, x)
            if squeeze_bands and values.ndim == 3 and values.shape[0] == 1:
                values = values[0, :, :]

            # 类型转换（整数转浮点以支持NaN）
            if dtype is not None:
                values = values.astype(dtype)
            elif np.issubdtype(values.dtype, np.integer):
                values = values.astype(np.float64)

            # 处理特定的nodata值
            if name in nodata_values:
                values[values == nodata_values[name]] = np.nan

            result[name] = values

        return result

    def to_dataset(self) -> xr.Dataset:
        """
        将栅格集合转换为xarray.Dataset

        注意: 所有栅格在 add_raster 时已自动对齐，无需再次对齐

        返回:
            xr.Dataset: 包含所有栅格的数据集
        """
        return xr.Dataset(self.rasters)

    def get_reference_info(self) -> Dict:
        """
        获取参考图像的地理信息

        返回:
            Dict: 包含 crs, transform, height, width, bounds, nodata 的字典

        注意:
            必须先调用 set_reference() 设置地理参考
        """
        if not self._is_georeferenced:
            raise RuntimeError(
                "地理参考未初始化！必须先调用 set_reference() 方法。"
            )

        return self._reference_info.copy()

    def get_original_resolution(self, name: str) -> float:
        """
        获取指定栅格的原始分辨率

        参数:
            name: 栅格名称

        返回:
            float: 原始分辨率（米）

        抛出:
            KeyError: 如果栅格不存在
        """
        if name not in self.original_resolutions:
            raise KeyError(f"栅格'{name}'的原始分辨率未记录。可用: {list(self.original_resolutions.keys())}")
        return self.original_resolutions[name]

    def get_original_resolution_by_prefix(self, prefix: str) -> float:
        """
        获取具有指定前缀的栅格的原始分辨率（取第一个匹配项）

        用于获取某一类数据源的原始分辨率，如 'era5_' 获取ERA5数据的原始分辨率。
        假设同一前缀的所有栅格具有相同的原始分辨率。

        参数:
            prefix: 栅格名称前缀，如 'era5_', 'landsat_'

        返回:
            float: 原始分辨率（米）

        抛出:
            KeyError: 如果没有找到匹配的栅格
        """
        for name, res in self.original_resolutions.items():
            if name.startswith(prefix):
                return res
        raise KeyError(f"没有找到前缀为'{prefix}'的栅格。可用: {list(self.original_resolutions.keys())}")

    def save_multiband(
        self,
        output_path: Union[str, Path],
        bands: List[Tuple[str, np.ndarray]],
        compress: str = 'lzw',
        dtype: type = np.float32
    ) -> None:
        """
        保存多个波段到单个GeoTIFF文件

        参数:
            output_path: 输出文件路径
            bands: [(波段名, 数组), ...] 列表，每个元素是(名称, 2D数组)
            compress: 压缩方法（默认 'lzw'）
            dtype: 输出数据类型（默认 np.float32）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 获取参考地理信息
        ref_info = self.get_reference_info()

        expected_shape = (ref_info['height'], ref_info['width'])

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=ref_info['height'],
            width=ref_info['width'],
            count=len(bands),
            dtype=dtype,
            crs=ref_info['crs'],
            transform=ref_info['transform'],
            nodata=np.nan,
            compress=compress
        ) as dst:
            for i, (name, arr) in enumerate(bands, 1):
                # 确保是2D数组
                if arr.ndim == 3:
                    arr = arr[0, :, :]
                
                # 尺寸检查
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"波段 '{name}' 尺寸不匹配！"
                        f"期望 {expected_shape}，实际 {arr.shape}"
                    )
                
                dst.write(arr.astype(dtype), i)
                dst.set_band_description(i, name)

        print(f"✓ 已保存: {output_path}")
        print(f"  波段数: {len(bands)}")

    def save(
        self,
        output_dir: Union[str, Path],
        driver: str = 'GTiff',
        compress: str = 'lzw'
    ) -> None:
        """
        保存对齐后的栅格到文件

        注意: 所有栅格在 add_raster 时已自动对齐，直接保存即可

        参数:
            output_dir: 输出目录
            driver: 输出格式驱动（默认GeoTIFF）
            compress: 压缩方法
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, da in self.rasters.items():
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

    def add_array(self, name: str, array: np.ndarray) -> None:
        """
        将 numpy 数组添加到集合（使用参考图像的地理信息）
        
        参数:
            name: 栅格名称
            array: 2D numpy 数组
        """
        if not self._is_georeferenced:
            raise RuntimeError("地理参考未初始化！必须先调用 set_reference() 方法。")
        
        ref_info = self.get_reference_info()
        expected_shape = (ref_info['height'], ref_info['width'])
        
        # 尺寸检查
        if array.shape != expected_shape:
            raise ValueError(
                f"数组 '{name}' 尺寸不匹配！"
                f"期望 {expected_shape}，实际 {array.shape}"
            )
        
        transform = ref_info['transform']
        x_coords = np.arange(ref_info['width']) * transform.a + transform.c + transform.a / 2
        y_coords = np.arange(ref_info['height']) * transform.e + transform.f + transform.e / 2
        
        da = xr.DataArray(array, dims=['y', 'x'], coords={'y': y_coords, 'x': x_coords}, name=name)
        da = da.rio.write_crs(ref_info['crs'])
        da = da.rio.write_transform(ref_info['transform'])
        self.rasters[name] = da

    def __repr__(self):
        raster_list = ', '.join(self.rasters.keys())
        return f"RasterCollection({len(self.rasters)} rasters: {raster_list})"
