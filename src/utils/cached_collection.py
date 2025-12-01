"""
缓存栅格数据集合管理器

将栅格数据保存到磁盘而非内存，按需读取，节省内存。

缓存结构:
    <cache_dir>/
    ├── metadata.pkl          # 类元信息（不含数组值）
    ├── era5_surface_pressure.npy
    ├── landsat_lst.npy
    └── ...
"""

import pickle
import numpy as np
import xarray as xr
import rasterio
import warnings
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
from rasterio.enums import Resampling
from rasterio.transform import Affine
from .raster_manager import RasterData, RasterBand


class CachedRasterCollection:
    """
    磁盘缓存的栅格数据集合管理器
    
    与 RasterCollection 接口兼容，但数据存储在磁盘而非内存。
    get_array() 按需从磁盘读取数据。
    """
    
    METADATA_FILE = 'metadata.pkl'
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        target_resolution: float,
        target_crs: Union[str, int],
        target_bounds: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        初始化缓存栅格集合
        
        参数:
            cache_dir: 缓存目录路径
            target_resolution: 目标分辨率（米）
            target_crs: 目标坐标系
            target_bounds: 目标范围 (left, bottom, right, top)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_resolution = target_resolution
        self.target_crs = target_crs
        self.target_bounds = target_bounds
        self._reference_bounds: Optional[Tuple[float, float, float, float]] = None
        self._reference_info: Optional[Dict] = None
        self._is_georeferenced: bool = False
        self.original_resolutions: Dict[str, float] = {}
        
        # 记录数组文件路径（相对于 cache_dir）
        self._array_paths: Dict[str, str] = {}
    
    def _get_array_path(self, name: str) -> Path:
        """获取数组文件的完整路径"""
        return self.cache_dir / f"{name}.npy"
    
    def _save_array(self, name: str, array: np.ndarray) -> None:
        """保存数组到磁盘"""
        path = self._get_array_path(name)
        np.save(path, array)
        self._array_paths[name] = f"{name}.npy"
    
    def _load_array(self, name: str) -> np.ndarray:
        """从磁盘加载数组"""
        if name not in self._array_paths:
            raise KeyError(f"栅格'{name}'不存在。可用: {list(self._array_paths.keys())}")
        path = self.cache_dir / self._array_paths[name]
        return np.load(path)
    
    def save_metadata(self) -> None:
        """保存元信息到 pickle（不含数组值）"""
        metadata = {
            'target_resolution': self.target_resolution,
            'target_crs': self.target_crs,
            'target_bounds': self.target_bounds,
            '_reference_bounds': self._reference_bounds,
            '_reference_info': self._serialize_reference_info(),
            '_is_georeferenced': self._is_georeferenced,
            'original_resolutions': self.original_resolutions,
            '_array_paths': self._array_paths,
        }
        
        metadata_path = self.cache_dir / self.METADATA_FILE
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ 缓存元信息已保存: {metadata_path}")
    
    def _serialize_reference_info(self) -> Optional[Dict]:
        """序列化参考信息（处理不可序列化的对象）"""
        if self._reference_info is None:
            return None
        
        info = self._reference_info.copy()
        # CRS 转为字符串
        if 'crs' in info and info['crs'] is not None:
            info['crs'] = str(info['crs'])
        # Affine transform 转为元组
        if 'transform' in info and info['transform'] is not None:
            t = info['transform']
            info['transform'] = (t.a, t.b, t.c, t.d, t.e, t.f)
        return info
    
    def _deserialize_reference_info(self, info: Optional[Dict]) -> Optional[Dict]:
        """反序列化参考信息"""
        if info is None:
            return None
        
        result = info.copy()
        # 字符串转回 CRS
        if 'crs' in result and result['crs'] is not None:
            result['crs'] = rasterio.crs.CRS.from_string(result['crs'])
        # 元组转回 Affine
        if 'transform' in result and result['transform'] is not None:
            t = result['transform']
            result['transform'] = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        return result
    
    @classmethod
    def load_from_cache(cls, cache_dir: Union[str, Path]) -> 'CachedRasterCollection':
        """
        从缓存目录加载（只读取 pickle，不加载数组）
        
        参数:
            cache_dir: 缓存目录路径
            
        返回:
            CachedRasterCollection 实例
        """
        cache_dir = Path(cache_dir)
        metadata_path = cache_dir / cls.METADATA_FILE
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"缓存元信息不存在: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 创建实例
        instance = cls(
            cache_dir=cache_dir,
            target_resolution=metadata['target_resolution'],
            target_crs=metadata['target_crs'],
            target_bounds=metadata['target_bounds']
        )
        
        # 恢复状态
        instance._reference_bounds = metadata['_reference_bounds']
        instance._reference_info = instance._deserialize_reference_info(metadata['_reference_info'])
        instance._is_georeferenced = metadata['_is_georeferenced']
        instance.original_resolutions = metadata['original_resolutions']
        instance._array_paths = metadata.get('_array_paths', {})
        
        # 自动检测并添加缺失的 .npy 文件（修复元数据不完整的问题）
        # 扫描缓存目录中的所有 .npy 文件
        npy_files = list(cache_dir.glob("*.npy"))
        added_arrays = []
        for npy_file in npy_files:
            array_name = npy_file.stem  # 去掉 .npy 后缀
            if array_name not in instance._array_paths:
                instance._array_paths[array_name] = f"{array_name}.npy"
                added_arrays.append(array_name)
        
        # 如果发现并添加了新的数组，更新元数据文件
        if added_arrays:
            print(f"  发现并添加了 {len(added_arrays)} 个缺失的数组: {added_arrays}")
            # 更新并保存元数据
            metadata['_array_paths'] = instance._array_paths
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        
        print(f"✓ 从缓存加载: {cache_dir}")
        print(f"  已缓存栅格: {list(instance._array_paths.keys())}")
        
        return instance
    
    @classmethod
    def cache_exists(cls, cache_dir: Union[str, Path]) -> bool:
        """检查缓存是否存在"""
        cache_dir = Path(cache_dir)
        return (cache_dir / cls.METADATA_FILE).exists()
    
    def set_reference(
        self,
        reference_raster: RasterData,
        band: int = 1
    ) -> None:
        """
        设置参考图像，用于定义所有栅格的输出范围和地理参考
        
        使用参考图像的 extent（范围），但输出尺寸根据 target_resolution 计算。
        """
        
        if not reference_raster.is_loaded():
            reference_raster.load()
        
        reference_band = reference_raster.get_band(band, name='reference')
        ref_da = reference_band.data
        
        # 转换到目标坐标系
        if self.target_crs is not None and hasattr(ref_da, 'rio'):
            current_crs = str(ref_da.rio.crs)
            if current_crs != str(self.target_crs):
                ref_da = ref_da.rio.reproject(self.target_crs, resampling=Resampling.bilinear)
        
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
            new_transform = Affine(
                self.target_resolution,  # a: pixel width
                0,                        # b: row rotation
                left,                     # c: x coordinate of upper-left corner
                0,                        # d: column rotation
                -self.target_resolution,  # e: pixel height (negative)
                top                       # f: y coordinate of upper-left corner
            )
            
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
    
    def add_raster(
        self,
        name: str,
        source: Union[str, Path, xr.DataArray, RasterData, RasterBand],
        band: int = 1,
        resampling: str = 'bilinear',
        chunks: Optional[Dict] = None
    ) -> None:
        """添加栅格到集合，处理对齐后保存到磁盘"""
        if isinstance(source, RasterBand):
            da = source.data
        elif isinstance(source, RasterData):
            if not source.is_loaded():
                source.load(chunks=chunks)
            raster_band = source.get_band(band, name=name)
            da = raster_band.data
        elif isinstance(source, xr.DataArray):
            da = source
        else:
            raster = RasterData(source)
            raster.load(chunks=chunks)
            raster_band = raster.get_band(band, name=name)
            da = raster_band.data
        
        if not self._is_georeferenced:
            raise RuntimeError(
                "地理参考未初始化！在使用 CachedRasterCollection 前必须先调用 set_reference() 方法。"
            )
        
        da.name = name
        
        # 记录原始分辨率
        if hasattr(da, 'rio'):
            original_res = abs(da.rio.resolution()[0])
            self.original_resolutions[name] = original_res
        
        # 坐标系转换
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
                self.original_resolutions[name] = abs(da.rio.resolution()[0])
        
        # 强制对齐到目标网格（无论分辨率是否匹配都需要对齐）
        if self.target_resolution is not None and self._reference_bounds is not None and self._reference_info is not None:
            if hasattr(da, 'rio'):
                current_res = abs(da.rio.resolution()[0])
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
        
        # 压缩 band 维度
        if da.ndim == 3 and da.shape[0] == 1:
            da = da.squeeze('band', drop=True)
            da.name = name
        
        # 保存到磁盘而非内存
        arr = da.values
        self._save_array(name, arr)
        
        # 释放内存
        del da
        del arr
    
    def load_multiband(
        self,
        filepath: Union[str, Path],
        bands: Dict[str, int],
        prefix: str = '',
        resampling: str = 'bilinear'
    ) -> None:
        """批量加载多波段数据到集合"""
        for name, band in bands.items():
            full_name = f"{prefix}_{name}" if prefix else name
            self.add_raster(full_name, filepath, band=band, resampling=resampling)
    
    def get_array(self, name: str, dtype: type = np.float64) -> np.ndarray:
        """
        从磁盘获取指定栅格的 numpy 数组
        
        参数:
            name: 栅格名称
            dtype: 目标数据类型（默认 np.float64）
            
        返回:
            np.ndarray: 2D numpy 数组
        """
        return self._load_array(name).astype(dtype)
    
    def add_array(self, name: str, array: np.ndarray) -> None:
        """
        将 numpy 数组添加到集合（保存到磁盘）
        
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
        
        self._save_array(name, array)
    
    def list_rasters(self) -> List[str]:
        """列出所有栅格名称"""
        return list(self._array_paths.keys())
    
    @property
    def rasters(self) -> Dict[str, str]:
        """返回栅格名称到路径的映射（兼容 RasterCollection 接口）"""
        return self._array_paths
    
    def get_reference_info(self) -> Dict:
        """获取参考图像的地理信息"""
        if not self._is_georeferenced:
            raise RuntimeError("地理参考未初始化！必须先调用 set_reference() 方法。")
        return self._reference_info.copy()
    
    def get_original_resolution(self, name: str) -> float:
        """获取指定栅格的原始分辨率"""
        if name not in self.original_resolutions:
            raise KeyError(f"栅格'{name}'的原始分辨率未记录。可用: {list(self.original_resolutions.keys())}")
        return self.original_resolutions[name]
    
    def get_original_resolution_by_prefix(self, prefix: str) -> float:
        """获取具有指定前缀的栅格的原始分辨率"""
        for name, res in self.original_resolutions.items():
            if name.startswith(prefix):
                return res
        raise KeyError(f"没有找到前缀为'{prefix}'的栅格。可用: {list(self.original_resolutions.keys())}")
    
    def save(
        self,
        output_path: Union[str, Path],
        driver: str = 'GTiff',
        compress: str = 'lzw',
        output_bands: Optional[List[str]] = None
    ) -> None:
        """
        保存对齐后的栅格到单个多波段GeoTIFF文件

        注意: 所有栅格在 add_raster 时已自动对齐，直接保存即可

        参数:
            output_path: 输出文件路径
            driver: 输出格式驱动（默认GeoTIFF）
            compress: 压缩方法
            output_bands: 要保存的波段名称列表（如果为None，保存所有波段）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"保存结果: {output_path}")
        print("=" * 60)

        # 获取参考地理信息
        ref_info = self.get_reference_info()
        
        # 收集所有有效的栅格数据
        valid_rasters = {}
        for name in self._array_paths.keys():
            # 如果指定了 output_bands，只处理列表中的波段
            if output_bands is not None and name not in output_bands:
                continue
                
            try:
                arr = self.get_array(name)
                # 将 numpy 数组转换为 DataArray 并添加地理信息
                da = self._array_to_dataarray(name, arr, ref_info)
                if hasattr(da, 'rio'):
                    valid_rasters[name] = da
                else:
                    warnings.warn(f"栅格'{name}'没有空间参考信息，跳过保存")
            except Exception as e:
                warnings.warn(f"无法加载栅格'{name}': {e}，跳过保存")

        if not valid_rasters:
            if output_bands is not None:
                available = list(self._array_paths.keys())
                missing = [b for b in output_bands if b not in available]
                if missing:
                    raise ValueError(
                        f"指定的波段不存在: {missing}。可用波段: {available}"
                    )
            raise ValueError("没有有效的栅格数据可保存")

        # 创建 xarray Dataset，将所有波段合并
        data_vars = {}
        ref_da = None
        
        # 如果指定了 output_bands，按照指定顺序处理；否则按照字典顺序
        band_order = output_bands if output_bands is not None else list(valid_rasters.keys())
        
        for name in band_order:
            if name not in valid_rasters:
                continue  # 跳过不存在的波段
            da = valid_rasters[name]
            # 确保是2D数组
            if da.ndim == 3:
                da = da.isel(band=0) if 'band' in da.dims else da[0]
            data_vars[name] = da
            # 保存第一个 DataArray 作为参考（用于获取地理信息）
            if ref_da is None:
                ref_da = da

        # 创建 Dataset
        ds = xr.Dataset(data_vars)
        
        # 使用 rioxarray 保存为多波段 GeoTIFF
        # 需要将 Dataset 转换为 DataArray 的堆叠形式
        # 使用 to_array() 方法将 Dataset 转换为多波段 DataArray
        ds_array = ds.to_array(dim='band')
        
        # 设置波段名称（按照指定顺序）
        ds_array['band'] = [name for name in band_order if name in valid_rasters]
        
        # 确保地理信息正确传递（从参考 DataArray 复制）
        if ref_da is not None and hasattr(ref_da, 'rio'):
            ds_array = ds_array.rio.write_crs(ref_da.rio.crs)
            ds_array = ds_array.rio.write_transform(ref_da.rio.transform())
        
        # 保存为多波段 GeoTIFF
        ds_array.rio.to_raster(
            output_path,
            driver=driver,
            compress=compress,
            nodata=np.nan
        )

        # 打印波段列表
        saved_bands = [name for name in band_order if name in valid_rasters]
        print(f"✓ 已保存: {output_path}")
        print(f"  波段数: {len(saved_bands)}")
        print(f"  保存了 {len(saved_bands)} 个波段:")
        for i, name in enumerate(saved_bands, 1):
            print(f"    {i:2d}. {name}")

    def _array_to_dataarray(
        self,
        name: str,
        array: np.ndarray,
        ref_info: Dict
    ) -> xr.DataArray:
        """
        将 numpy 数组转换为带有地理信息的 DataArray
        
        参数:
            name: 栅格名称
            array: numpy 数组
            ref_info: 参考地理信息字典
            
        返回:
            xr.DataArray: 带有地理信息的 DataArray
        """
        # 确保是2D数组
        if array.ndim == 3:
            array = array[0, :, :]
        elif array.ndim > 3:
            raise ValueError(f"不支持的数组维度: {array.ndim}")
        
        # 创建坐标
        transform = ref_info['transform']
        x_coords = np.arange(ref_info['width']) * transform.a + transform.c + transform.a / 2
        y_coords = np.arange(ref_info['height']) * transform.e + transform.f + transform.e / 2
        
        # 创建 DataArray
        da = xr.DataArray(
            array,
            dims=['y', 'x'],
            coords={'y': y_coords, 'x': x_coords},
            name=name
        )
        
        # 添加地理信息
        da = da.rio.write_crs(ref_info['crs'])
        da = da.rio.write_transform(ref_info['transform'])
        
        return da

    def __repr__(self):
        raster_list = ', '.join(self._array_paths.keys())
        return f"CachedRasterCollection({len(self._array_paths)} rasters: {raster_list}, cache={self.cache_dir})"
    
    def __contains__(self, name: str) -> bool:
        """支持 'name' in collection 语法"""
        return name in self._array_paths

