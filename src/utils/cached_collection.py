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
        instance._array_paths = metadata['_array_paths']
        
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
    
    def save_multiband(
        self,
        output_path: Union[str, Path],
        bands: List[Tuple[str, np.ndarray]],
        compress: str = 'lzw',
        dtype: type = np.float32
    ) -> None:
        """保存多个波段到单个GeoTIFF文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
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
    
    def __repr__(self):
        raster_list = ', '.join(self._array_paths.keys())
        return f"CachedRasterCollection({len(self._array_paths)} rasters: {raster_list}, cache={self.cache_dir})"
    
    def __contains__(self, name: str) -> bool:
        """支持 'name' in collection 语法"""
        return name in self._array_paths

