"""
空气动力学参数计算工作流

整合所有空气动力学参数的计算流程，提供统一接口。

计算内容:
    - 气压（DEM高程订正）
    - 空气密度
    - 风速
    - 水汽压（实际和饱和）
    - 粗糙度（基于LCZ）
    - 阻抗（空气动力学阻抗和表面阻抗）

使用示例:
    >>> from src.aerodynamics import calculate_aerodynamic_parameters
    >>> output = calculate_aerodynamic_parameters(collection)
"""

from pathlib import Path
from typing import TYPE_CHECKING, Union, Optional
import numpy as np

from .dem_downscaling import adjust_pressure_for_elevation
from .atmospheric import calculate_air_density, calculate_wind_speed
from .vapor_pressure import (
    calculate_actual_vapor_pressure_from_dewpoint,
    calculate_saturation_vapor_pressure
)
from .resistance import (
    calculate_aerodynamic_resistance,
    calculate_surface_resistance
)
from ..utils import LCZ_ROUGHNESS, RasterCollection
from ..utils.cached_collection import CachedRasterCollection

if TYPE_CHECKING:
    pass


def calculate_roughness_from_lcz(lcz: np.ndarray) -> np.ndarray:
    """
    根据LCZ数据计算粗糙度长度

    参数:
        lcz: LCZ分类栅格数组

    返回:
        粗糙度长度数组 (m)
    """
    lcz_int = lcz.astype(int)
    roughness = np.full_like(lcz_int, 0.5, dtype=np.float32)  # 默认值

    for lcz_val, z0 in LCZ_ROUGHNESS.items():
        roughness[lcz_int == lcz_val] = z0

    return roughness


def calculate_aerodynamic_parameters(
    collection: Union[RasterCollection, CachedRasterCollection],
    verbose: bool = True,
    cache_dir: Optional[str] = None,
    restart: bool = False
) -> Union[RasterCollection, CachedRasterCollection]:
    """
    计算空气动力学参数，返回只包含计算结果的新集合

    从输入集合读取数据，计算空气动力学参数，结果存入新的内存集合返回。
    输入集合不会被修改（支持 CachedRasterCollection 作为只读输入）。
    不复制原始数据到输出集合，节省内存。

    参数:
        collection: RasterCollection 或 CachedRasterCollection，需包含:
            - era5_surface_pressure, era5_temperature_2m, 
              era5_dewpoint_temperature_2m, era5_u/v_component_of_wind_10m
            - landsat_lst, landsat_ndvi
            - dem
            - lcz
        verbose: 是否打印统计信息
        cache_dir: 缓存目录路径（启用后使用磁盘缓存）
        restart: 是否强制重新计算并覆盖缓存

    返回:
        RasterCollection 或 CachedRasterCollection: 只包含计算结果的新集合（不含原始数据）
            - pressure: 订正后气压 (Pa)
            - air_density: 空气密度 (kg/m³)
            - wind_speed: 风速 (m/s)
            - actual_vapor_pressure: 实际水汽压 (kPa)
            - saturation_vapor_pressure: 饱和水汽压 (kPa)
            - aerodynamic_resistance: 空气动力学阻抗 (s/m)
            - surface_resistance: 表面阻抗 (s/m)
            - roughness_length: 粗糙度长度 (m)
    """
    # 自动从 collection 获取分辨率信息
    era5_grid_size_meters = collection.get_original_resolution_by_prefix('era5_')
    pixel_size_meters = collection.target_resolution
    
    # 如果指定缓存目录
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_exists = CachedRasterCollection.cache_exists(cache_path)
        
        # 如果缓存存在且不是重启模式，直接加载返回
        if cache_exists and not restart:
            if verbose:
                print("\n" + "=" * 60)
                print("步骤 2: 计算空气动力学参数")
                print("=" * 60)
                print(f"\n从缓存加载空气动力学参数: {cache_path}")
            return CachedRasterCollection.load_from_cache(cache_path)
        
        # 需要计算，创建缓存集合
        if verbose:
            if restart:
                print(f"\n强制重新计算空气动力学参数，覆盖缓存: {cache_path}")
            else:
                print(f"\n缓存不存在，计算空气动力学参数: {cache_path}")
        
        output = CachedRasterCollection(
            cache_dir=cache_path,
            target_resolution=collection.target_resolution,
            target_crs=collection.target_crs,
            target_bounds=collection.target_bounds
        )
    else:
        # 内存模式
        output = RasterCollection(
            target_resolution=collection.target_resolution,
            target_crs=collection.target_crs,
            target_bounds=collection.target_bounds
        )
    
    # 复制地理信息
    output._reference_bounds = collection._reference_bounds
    output._reference_info = collection._reference_info
    output._is_georeferenced = collection._is_georeferenced
    output.original_resolutions = collection.original_resolutions.copy()
    
    # 从集合中获取数据用于计算
    surface_pressure = collection.get_array('era5_surface_pressure')
    temperature_2m = collection.get_array('era5_temperature_2m')
    dewpoint_2m = collection.get_array('era5_dewpoint_temperature_2m')
    u_wind = collection.get_array('era5_u_component_of_wind_10m')
    v_wind = collection.get_array('era5_v_component_of_wind_10m')
    lst = collection.get_array('landsat_lst')
    ndvi = collection.get_array('landsat_ndvi')
    dem = collection.get_array('dem')
    lcz = collection.get_array('lcz')
    
    # DEM 特殊处理
    dem[dem == -999] = np.nan

    if verbose:
        print("\n" + "=" * 60)
        print("步骤 2: 计算空气动力学参数")
        print("=" * 60)

    # 1. 气压高程订正
    if verbose:
        print("\n计算气压（DEM高程订正）...")
    pressure = adjust_pressure_for_elevation(
        era5_pressure=surface_pressure,
        dem_elevation=dem,
        temperature=temperature_2m,
        era5_grid_size_meters=era5_grid_size_meters,
        pixel_size_meters=pixel_size_meters
    )

    # 2. 空气密度
    if verbose:
        print("计算空气密度...")
    air_density = calculate_air_density(temperature_2m, pressure)

    # 3. 风速
    if verbose:
        print("计算风速...")
    wind_speed = calculate_wind_speed(u_wind, v_wind)

    # 4. 水汽压
    if verbose:
        print("计算水汽压...")
    ea = calculate_actual_vapor_pressure_from_dewpoint(dewpoint_2m)
    es = calculate_saturation_vapor_pressure(lst)

    # 5. 粗糙度 - 合并建筑数据和LCZ数据
    if verbose:
        print("计算粗糙度...")
    
    # LCZ 查表作为基础/后备
    lcz_roughness = calculate_roughness_from_lcz(lcz)
    
    # 尝试合并建筑数据计算的粗糙度
    if 'roughness_length' in collection.rasters:
        building_roughness = collection.get_array('roughness_length')
        # 建筑数据中 <= 0.1 的区域视为无效（0.1 是默认值）
        valid_building_mask = (building_roughness > 0.1) & np.isfinite(building_roughness)
        roughness = np.where(valid_building_mask, building_roughness, lcz_roughness)
        if verbose:
            building_pixels = np.sum(valid_building_mask)
            print(f"  建筑粗糙度像元: {building_pixels}")
            print(f"  LCZ粗糙度像元: {np.sum(~valid_building_mask)}")
    else:
        roughness = lcz_roughness
    
    # 位移高度 - 合并建筑数据
    displacement_height = None
    if 'displacement_height' in collection.rasters:
        building_displacement = collection.get_array('displacement_height')
        # 建筑数据中 > 0 的区域使用建筑值
        valid_disp_mask = (building_displacement > 0) & np.isfinite(building_displacement)
        if np.any(valid_disp_mask):
            displacement_height = building_displacement.copy()
            # 无效区域会在 resistance.py 中根据 LCZ 自动计算
            if verbose:
                print(f"  建筑位移高度像元: {np.sum(valid_disp_mask)}")
    
    del surface_pressure
    del temperature_2m
    del dewpoint_2m
    del u_wind
    del v_wind
    del dem
    # 6. 阻抗
    if verbose:
        print("计算阻抗参数...")
    rah = calculate_aerodynamic_resistance(
        wind_speed=wind_speed,
        roughness_length=roughness,
        lcz=lcz.astype(int),
        displacement_height=displacement_height
    )
    rs = calculate_surface_resistance(ndvi=ndvi)


    # 保存前先将 surface_pressure 等原始数据释放以节约内存
    del ndvi
    del lcz
    del displacement_height

    # 将计算结果添加到输出集合
    output.add_array('pressure', pressure)
    output.add_array('air_density', air_density)
    output.add_array('wind_speed', wind_speed)
    output.add_array('actual_vapor_pressure', ea)
    output.add_array('saturation_vapor_pressure', es)
    output.add_array('aerodynamic_resistance', rah)
    output.add_array('surface_resistance', rs)
    output.add_array('roughness_length', roughness)

    # 如果使用缓存模式，保存元信息
    if isinstance(output, CachedRasterCollection):
        output.save_metadata()

    # 打印统计
    if verbose:
        print(f"\n参数统计:")
        print(f"  空气密度: {np.nanmean(air_density):.3f} kg/m³")
        print(f"  风速: {np.nanmean(wind_speed):.2f} m/s")
        print(f"  实际水汽压: {np.nanmean(ea):.3f} kPa")
        print(f"  饱和水汽压: {np.nanmean(es):.3f} kPa")
        print(f"  空气动力学阻抗: {np.nanmean(rah):.1f} s/m")
        print(f"  表面阻抗: {np.nanmean(rs):.1f} s/m")

    return output

