"""
能量平衡计算工作流

整合能量平衡系数计算流程，提供统一接口。

计算内容:
    - 短波辐射计算（水平面太阳辐射，适用于城市冠层模型）
    - 储热通量（基于LCZ分类计算：自然表面SEBAL / 不透水面储热系数）
    - 能量平衡系数 (f_Ta_coeff, residual)

使用示例:
    >>> from src.radiation import calculate_energy_balance
    >>> from datetime import datetime
    >>> dt = datetime(2023, 8, 15, 10, 30)
    >>> calculate_energy_balance(input_collection, aero_collection, datetime_obj=dt)
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union, Optional
import numpy as np

from .balance_equation import calculate_energy_balance_coefficients, validate_energy_balance
from .solar_radiation import calculate_dem_solar_radiation
from ..utils import LCZ_IMPERVIOUS, URBAN_LCZ_TYPES, NATURAL_LCZ_TYPES, RasterCollection
from ..utils.cached_collection import CachedRasterCollection

if TYPE_CHECKING:
    pass


def calculate_energy_balance(
    input_collection: Union['RasterCollection', 'CachedRasterCollection'],
    aero_collection: Union[RasterCollection, CachedRasterCollection],
    datetime_obj: datetime,
    std_meridian: float = 120.0,
    verbose: bool = True,
    cache_dir: Optional[str] = None,
    tif_path: Optional[str] = None,
    restart: bool = False
) -> Union[RasterCollection, CachedRasterCollection]:
    """
    计算能量平衡系数，返回只包含计算结果的新集合

    从两个集合读取数据：原始数据从 input_collection，空气动力学参数从 aero_collection。
    结果存入新的集合返回，输入集合不会被修改。

    短波辐射计算使用水平面太阳辐射，适用于城市冠层模型（Urban Canopy Layer）。

    参数:
        input_collection: 原始数据集合（RasterCollection 或 CachedRasterCollection），需包含:
            - era5_temperature_2m
            - landsat_lst, landsat_ndvi, landsat_albedo, landsat_emissivity
            - dem
            - lcz
        aero_collection: 空气动力学参数集合（RasterCollection 或 CachedRasterCollection），需包含:
            - saturation_vapor_pressure, actual_vapor_pressure
            - aerodynamic_resistance, surface_resistance
            - pressure
        datetime_obj: 观测日期时间
        std_meridian: 标准经度 (默认120度，东八区)
        verbose: 是否打印统计信息
        cache_dir: 缓存目录路径（启用后使用磁盘缓存）
        restart: 是否强制重新计算并覆盖缓存

    返回:
        RasterCollection 或 CachedRasterCollection: 只包含能量平衡计算结果的新集合
            - shortwave_down: 短波下行辐射 (W/m²)
            - f_Ta_coeff: Ta系数 (W/m²/K)
            - residual: 残差项 (W/m²)
            - soil_heat_flux: 储热通量 (W/m²)
            - latent_heat_flux: 潜热通量 (W/m²)
    """
    # 如果指定缓存目录
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_exists = CachedRasterCollection.cache_exists(cache_path)
        
        # 如果缓存存在且不是重启模式，直接加载返回
        if cache_exists and not restart:
            if verbose:
                print("\n" + "=" * 60)
                print("步骤 3: 计算能量平衡系数")
                print("=" * 60)
                print(f"\n从缓存加载能量平衡参数: {cache_path}")
            return CachedRasterCollection.load_from_cache(cache_path)
        
        # 需要计算，稍后创建缓存集合
        if verbose:
            if restart:
                print(f"\n强制重新计算能量平衡参数，覆盖缓存: {cache_path}")
            else:
                print(f"\n缓存不存在，计算能量平衡参数: {cache_path}")

    # 从原始数据集合获取数据
    lst = input_collection.get_array('landsat_lst')
    ndvi = input_collection.get_array('landsat_ndvi')
    dem = input_collection.get_array('dem')
    lcz = input_collection.get_array('lcz').astype(int)
    temperature_2m = input_collection.get_array('era5_temperature_2m')
    
    # 从空气动力学集合获取参数
    es = aero_collection.get_array('saturation_vapor_pressure')
    ea = aero_collection.get_array('actual_vapor_pressure')
    rah = aero_collection.get_array('aerodynamic_resistance')
    rs = aero_collection.get_array('surface_resistance')
    pressure = aero_collection.get_array('pressure')

    if verbose:
        print("\n" + "=" * 60)
        print("步骤 3: 计算能量平衡系数")
        print("=" * 60)
        print(f"观测时间: {datetime_obj.strftime('%Y-%m-%d %H:%M')}")

    # 获取地理参考信息，构建 geotransform
    ref_info = aero_collection.get_reference_info()
    transform = ref_info['transform']
    # Affine transform: (a, b, c, d, e, f) -> geotransform: (c, a, b, f, d, e)
    # 即 (x_min, pixel_width, 0, y_max, 0, -pixel_height)
    dem_geotransform = (transform.c, transform.a, transform.b,
                        transform.f, transform.d, transform.e)

    # 计算短波下行辐射（水平面太阳辐射，用于城市冠层模型）
    # 获取 CRS 信息用于坐标转换
    target_crs = aero_collection.target_crs
    
    if verbose:
        print("计算短波下行辐射 (S↓)...")
        print("  计算模式: 水平面太阳辐射 (Urban Canopy Layer)")
        print(f"  坐标系: {target_crs}")
        
        # 打印中心点坐标用于调试
        rows, cols = dem.shape
        x_min = transform.c
        pixel_width = transform.a
        y_max = transform.f
        pixel_height = transform.e
        center_x = x_min + pixel_width * cols / 2
        center_y = y_max + pixel_height * rows / 2
        
        if target_crs is not None:
            from pyproj import CRS as ProjCRS, Transformer
            src_crs = ProjCRS.from_string(target_crs)
            if src_crs.is_projected:
                proj_transformer = Transformer.from_crs(src_crs, ProjCRS.from_epsg(4326), always_xy=True)
                lon_deg, lat_deg = proj_transformer.transform(center_x, center_y)
                print(f"  中心点坐标: ({center_x:.1f}, {center_y:.1f}) -> ({lon_deg:.4f}°E, {lat_deg:.4f}°N)")
            else:
                print(f"  中心点坐标: ({center_x:.4f}°E, {center_y:.4f}°N)")
        else:
            print(f"  中心点坐标: ({center_x:.4f}, {center_y:.4f}) [假设为度]")

    sw_down = calculate_dem_solar_radiation(
        dem_array=dem,
        dem_geotransform=dem_geotransform,
        datetime_obj=datetime_obj,
        std_meridian=std_meridian,
        target_crs=target_crs
    )

    if verbose:
        valid_mask = ~np.isnan(sw_down) & (sw_down > 0)
        if np.any(valid_mask):
            print(f"  短波辐射统计: 均值={np.nanmean(sw_down[valid_mask]):.1f} W/m², "
                  f"范围=[{np.nanmin(sw_down[valid_mask]):.1f}, {np.nanmax(sw_down[valid_mask]):.1f}] W/m²")
        else:
            print("  警告: 短波辐射计算异常，无有效值！")
            print(f"    sw_down 范围: [{np.nanmin(sw_down):.1f}, {np.nanmax(sw_down):.1f}]")
            print(f"    NaN 比例: {np.sum(np.isnan(sw_down)) / sw_down.size * 100:.1f}%")

    emissivity = input_collection.get_array('landsat_emissivity')
    albedo = input_collection.get_array('landsat_albedo')
    # 统计 LCZ 分布
    if verbose:
        # 有效 LCZ 掩膜（1-14，排除 0/nodata）
        valid_lcz_mask = (lcz >= 1) & (lcz <= 14)
        valid_lcz_count = np.sum(valid_lcz_mask)
        
        # 不透水面统计
        impervious_mask = np.zeros_like(lcz, dtype=bool)
        for lcz_val, is_imperv in LCZ_IMPERVIOUS.items():
            if is_imperv:
                impervious_mask |= (lcz == lcz_val)
        imperv_ratio = np.sum(impervious_mask) / valid_lcz_count * 100

        # 城市/自然分布
        urban_mask = np.isin(lcz, list(URBAN_LCZ_TYPES))
        natural_mask = np.isin(lcz, list(NATURAL_LCZ_TYPES))
        urban_ratio = np.sum(urban_mask) / valid_lcz_count * 100
        natural_ratio = np.sum(natural_mask) / valid_lcz_count * 100
        
        # 显示 nodata 比例
        nodata_ratio = (1 - valid_lcz_count / lcz.size) * 100

        print("LCZ 分布 (仅统计有效值 1-14):")
        print(f"  城市建筑 (1-9): {urban_ratio:.1f}%")
        print(f"  自然地表 (10-14): {natural_ratio:.1f}%")
        print(f"  不透水面: {imperv_ratio:.1f}%")
        print(f"  (nodata/边界外: {nodata_ratio:.1f}%)")
        print("储热计算:")
        print("  自然表面: SEBAL 公式直接计算")
        print("  不透水面: 储热系数在 ALS 回归中估计")

    # 计算能量平衡系数（使用 LCZ 分类计算储热通量）
    if verbose:
        print("计算能量平衡系数...")

    coeffs = calculate_energy_balance_coefficients(
        shortwave_down=sw_down,
        surface_temperature=lst,
        elevation=dem,
        albedo=albedo,
        ndvi=ndvi,
        saturation_vapor_pressure=es,
        actual_vapor_pressure=ea,
        aerodynamic_resistance=rah,
        surface_resistance=rs,
        surface_emissivity=emissivity,
        surface_pressure=pressure,
        era5_air_temperature=temperature_2m,
        lcz=lcz
    )

    # 创建输出集合，复制地理信息
    if cache_dir is not None:
        output = CachedRasterCollection(
            cache_dir=Path(cache_dir),
            target_resolution=aero_collection.target_resolution,
            target_crs=aero_collection.target_crs,
            target_bounds=aero_collection.target_bounds
        )
    else:
        output = RasterCollection(
            target_resolution=aero_collection.target_resolution,
            target_crs=aero_collection.target_crs,
            target_bounds=aero_collection.target_bounds
        )
    # 复制地理信息（使用深拷贝避免引用问题）
    output._reference_bounds = aero_collection._reference_bounds
    output._reference_info = aero_collection._reference_info.copy() if aero_collection._reference_info else None
    output._is_georeferenced = aero_collection._is_georeferenced
    output.original_resolutions = aero_collection.original_resolutions.copy()
    
    # 将结果添加到输出集合
    output.add_array('shortwave_down', sw_down)
    output.add_array('f_Ta_coeff2', coeffs['f_Ta_coeff2'])  # 二次项系数
    output.add_array('f_Ta_coeff1', coeffs['f_Ta_coeff1'])  # 一次项系数
    output.add_array('residual', coeffs['residual'])
    output.add_array('soil_heat_flux', coeffs['soil_heat_flux'])
    output.add_array('latent_heat_flux', coeffs['latent_heat_flux'])

    # 如果使用缓存模式，保存元信息
    if isinstance(output, CachedRasterCollection):
        output.save_metadata()

    # 验证
    if verbose:
        validate_energy_balance(
            coeffs['f_Ta_coeff2'], 
            coeffs['f_Ta_coeff1'], 
            coeffs['residual']
        )
    
    return output
