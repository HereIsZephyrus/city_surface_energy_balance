"""
物理计算模块 - 城市地表能量平衡

包含数据加载、空气动力学计算、能量平衡系数计算等核心功能。

工作流程:
    1. 数据加载与对齐 (ERA5 + Landsat + DEM + LCZ)
    2. 空气动力学参数计算 (密度、风速、水汽压、阻抗)
    3. 能量平衡系数计算 (∂f/∂Ta, residual)
    4. 结果输出

使用方法:
    python -m src physics --era5 <era5.tif> --landsat <landsat.tif> --albedo <albedo.tif> --dem <dem.tif> \\
        --lcz <lcz.tif> --datetime 202308151030 -o <output.tif>
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Union

import numpy as np

from .utils import RasterCollection, RasterData, ERA5_BANDS, LANDSAT_BANDS
from .utils.cache_sanitizer import (
    DEFAULT_SANITIZATION_RULES,
    sanitize_array,
    sanitize_cache_arrays
)
from .utils.cached_collection import CachedRasterCollection
from .aerodynamics import calculate_aerodynamic_parameters
from .radiation import calculate_energy_balance
from .landscape import calculate_roughness_from_buildings, BUILDING_FIELDS


def sanitize_energy_balance_inputs(
    collection: Union[RasterCollection, CachedRasterCollection],
    verbose: bool = True
) -> None:
    """
    在能量平衡计算前，对关键输入波段做范围裁剪和 nodata 清洗。

    目标:
        - 移除 -9999/-32768 等填充值
        - 针对物理变量设置合理范围，超出范围视为缺失
        - 保证 LCZ 分类的取值在 [0, 14]

    当 collection 为 CachedRasterCollection 时，会直接修改 cache 目录下的
    ``*.npy`` 文件，实现离线修复（同独立脚本共用逻辑）。
    """

    if isinstance(collection, CachedRasterCollection):
        if verbose:
            print("\n数据合法化（缓存模式，直接写入 .npy）...")
        sanitize_cache_arrays(collection.cache_dir, verbose=verbose)
        return

    if verbose:
        print("\n数据合法化（内存模式）...")

    for rule in DEFAULT_SANITIZATION_RULES:
        try:
            arr = collection.get_array(rule.name).astype(np.float64)
        except KeyError:
            continue

        cleaned, mask = sanitize_array(arr, rule)
        if not np.any(mask):
            continue

        collection.add_array(rule.name, cleaned)
        if verbose:
            pct = mask.sum() / mask.size * 100
            range_desc = []
            if rule.min_val is not None or rule.max_val is not None:
                range_desc.append(
                    f"范围 [{rule.min_val if rule.min_val is not None else '-inf'}, "
                    f"{rule.max_val if rule.max_val is not None else 'inf'}]"
                )
            if rule.nodata_values or rule.nodata_below is not None:
                range_desc.append("nodata/sentinel")
            desc = "，".join(range_desc)
            print(f"  - {rule.name}: 清理 {mask.sum():,d} 像素 ({pct:.2f}%) {desc}")

    # LCZ 分类保证 0-14，非法值归零
    try:
        lcz = collection.get_array('lcz')
    except KeyError:
        lcz = None

    if lcz is not None:
        lcz_int = lcz.astype(np.int16, copy=True)
        invalid_mask = (lcz_int < 0) | (lcz_int > 14)
        if np.any(invalid_mask):
            lcz_int[invalid_mask] = 0
            collection.add_array('lcz', lcz_int)
            if verbose:
                pct = invalid_mask.sum() / invalid_mask.size * 100
                print(f"  - lcz: 重置 {invalid_mask.sum():,d} 像素 ({pct:.2f}%) 到 0 (无效分类)")

def load_and_align_data(
    era5_path: str,
    landsat_path: str,
    albedo_path: str,
    dem_path: str,
    lcz_path: str,
    building_height_path: str,
    building_path: str,
    voronoi_diagram_path: str,
    target_crs: str = 'EPSG:32650',
    target_resolution: float = 10.0,
    cache_dir: str = None,
    restart: bool = False
) -> Union[RasterCollection, CachedRasterCollection]:
    """
    加载并对齐多源栅格数据

    参数:
        era5_path: ERA5 tif文件路径
        landsat_path: Landsat tif文件路径（包含 NDVI/FVC/LST/发射率 等）
        albedo_path: 独立的地表反照率 tif 文件路径
        dem_path: DEM tif文件路径
        lcz_path: LCZ栅格文件路径（.tif，值为1-14的LCZ分类）
        building_height_path: 建筑高度栅格文件路径（.tif，必需，nodata处高度为0）
        building_path: 建筑数据文件路径（.gpkg，必需，用于计算粗糙度）
        voronoi_diagram_path: Voronoi图文件路径（.gpkg，必需，id与building的id对应）
        target_crs: 目标坐标系
        target_resolution: 目标分辨率(m)
        cache_dir: 缓存目录路径（启用后使用磁盘缓存）
        restart: 是否强制重新计算并覆盖缓存

    返回:
        RasterCollection 或 CachedRasterCollection: 包含所有对齐后数据的集合
    """
    print("\n" + "=" * 60)
    print("步骤 1: 加载和对齐多源数据")
    print("=" * 60)

    # 如果指定缓存目录
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_exists = CachedRasterCollection.cache_exists(cache_path)
        
        # 如果缓存存在且不是重启模式，直接加载
        if cache_exists and not restart:
            print(f"\n从缓存加载数据: {cache_path}")
            return CachedRasterCollection.load_from_cache(cache_path)
        
        # 缓存不存在或强制重启，需要创建新缓存
        if restart:
            print(f"\n强制重新计算，覆盖缓存: {cache_path}")
        else:
            print(f"\n缓存不存在，创建新缓存: {cache_path}")
        
        collection = CachedRasterCollection(
            cache_dir=cache_path,
            target_resolution=target_resolution,
            target_crs=target_crs
        )
    else:
        # 内存模式
        collection = RasterCollection(
            target_resolution=target_resolution,
            target_crs=target_crs
        )

    # 设置参考范围（Landsat NDVI）
    landsat_raster = RasterData(landsat_path)
    collection.set_reference(landsat_raster, band=LANDSAT_BANDS['ndvi'])

    # 批量加载多源数据
    print("\n加载 Landsat 数据...")
    collection.load_multiband(landsat_path, LANDSAT_BANDS, prefix='landsat')

    print("加载 Albedo 数据...")
    collection.add_raster('landsat_albedo', albedo_path, band=1, resampling='bilinear')

    print("加载 ERA5 数据...")
    collection.load_multiband(era5_path, ERA5_BANDS, prefix='era5')

    print("加载 DEM 数据...")
    collection.add_raster('dem', dem_path, band=1)

    # 添加LCZ数据（栅格输入）
    print("加载 LCZ 数据...")
    collection.add_raster('lcz', lcz_path, band=1, resampling='nearest')

    # 加载建筑高度栅格数据（必需）
    print("加载建筑高度栅格数据...")
    collection.add_raster('building_height', building_height_path, band=1, resampling='bilinear')
    
    # 处理nodata：将nodata和NaN设为0
    building_height_array = collection.get_array('building_height')
    building_height_array = np.where(
        ~np.isfinite(building_height_array),
        0.0,
        building_height_array
    )
    building_height_array = np.maximum(building_height_array, 0.0)
    collection.add_array('building_height', building_height_array)
    print(f"  建筑高度范围: {np.nanmin(building_height_array):.2f} - {np.nanmax(building_height_array):.2f} m")
    print(f"  非零像元数: {np.sum(building_height_array > 0)} / {building_height_array.size}")

    # 计算粗糙度参数（必需）
    print("\n加载建筑数据并计算粗糙度...")
    z_0, z_d = calculate_roughness_from_buildings(
        building_path=building_path,
        collection=collection,
        voronoi_diagram_path=voronoi_diagram_path,
        height_field=BUILDING_FIELDS['height'],
        footprint_field=BUILDING_FIELDS['footprint'],
        block_area_field=BUILDING_FIELDS['block_area'],
        min_proj_field=BUILDING_FIELDS['min_proj'],
        max_proj_field=BUILDING_FIELDS['max_proj'],
        district_field=BUILDING_FIELDS['district_id'],
        cluster_field=BUILDING_FIELDS['cluster']
    )
    collection.add_array('roughness_length', z_0)
    collection.add_array('displacement_height', z_d)

    # 如果使用缓存模式，保存元信息
    if isinstance(collection, CachedRasterCollection):
        collection.save_metadata()
    return collection

def main(args: argparse.Namespace = None):
    """
    物理计算主函数
    
    参数:
        args: 命令行参数（如果为None则从sys.argv解析）
    """
    if args is None:
        parser = argparse.ArgumentParser(
            description='城市地表能量平衡计算 - 物理模块',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
    # 基本用法
    python -m src physics --era5 era5.tif --landsat landsat.tif --dem dem.tif \\
        --lcz lcz.tif --datetime 202308151030 -o result.tif
    
    # 使用建筑数据计算粗糙度
    python -m src physics --era5 era5.tif --landsat landsat.tif --dem dem.tif \\
        --lcz lcz.tif --building buildings.gpkg --datetime 202308151030 -o result.tif

LCZ编码:
    1-9:  城市建筑类型 (密集高层~稀疏建筑)
    10:   裸岩/铺装 (E)    11: 密集树木 (A)
    12:   灌木/低矮植被 (C/D)  13: 裸土/沙地 (F)  14: 水体 (G)

建筑数据字段 (固定):
    height:           建筑高度 (m)
    area:             建筑平面面积 (m²)
    voroniarea:       维诺图面积 (m²)
    min_proj:         沿风向投影最小点 (m)
    max_proj:         沿风向投影最大点 (m)
    district_id:      街区ID
    spectral_cluster: 建筑群聚类标签

波段配置:
    ERA5: surface_pressure(3), dewpoint_2m(6), u_wind(5), v_wind(2), temp_2m(4)
    Landsat: ndvi(9), fvc(10), lst(19), albedo(11), emissivity(14)

注意:
    短波辐射使用水平面太阳辐射计算，适用于城市冠层模型 (Urban Canopy Layer)。
            """
        )
        parser.add_argument('--era5', required=True, help='ERA5 tif文件路径')
        parser.add_argument('--landsat', required=True, help='Landsat tif文件路径')
        parser.add_argument('--dem', required=True, help='DEM tif文件路径')
        parser.add_argument('--lcz', required=True, help='LCZ栅格文件路径 (.tif，值为1-14的LCZ分类)')
        parser.add_argument('--albedo', required=True, help='独立的地表反照率tif文件路径')
        parser.add_argument('-o', '--output', required=True, help='输出tif文件路径')
        parser.add_argument('--crs', default='EPSG:32650', help='目标坐标系 (默认: EPSG:32650)')
        parser.add_argument('--resolution', type=float, default=10.0, help='目标分辨率(m)')
        parser.add_argument('--datetime', required=True, help='观测日期时间 (格式: YYYYMMDDHHMM，如 202308151030)')
        parser.add_argument('--std-meridian', type=float, default=120.0, help='标准经度 (默认: 120.0 东八区)')
        parser.add_argument('--cachedir', type=str, help='缓存目录路径（启用后数据存储到磁盘而非内存）')
        parser.add_argument('--restart', action='store_true', help='强制重新计算并覆盖缓存（需配合 --cachedir 使用）')
        parser.add_argument('--building-height', type=str, required=True,
                            help='建筑高度栅格文件路径 (.tif，必需，nodata处高度为0)')
        parser.add_argument('--building', type=str, required=True, 
                            help='建筑数据文件路径 (.gpkg，必需，需包含 height, area, voroniarea, min_proj, max_proj, district_id, spectral_cluster 字段)')
        parser.add_argument('--voronoi-diagram', type=str, required=True,
                            help='Voronoi图文件路径 (.gpkg，必需，id与building的id对应，用于栅格化粗糙度结果)')
        args = parser.parse_args()

    # 检查输入文件
    required_files = [
        args.era5,
        args.landsat,
        args.albedo,
        args.dem,
        args.lcz,
        args.building_height,
        args.building,
        args.voronoi_diagram
    ]
    
    for path in required_files:
        if not Path(path).exists():
            print(f"错误: 文件不存在 - {path}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("城市地表能量平衡计算")
    print("=" * 60)
    print("\n输入文件:")
    print(f"  ERA5: {args.era5}")
    print(f"  Landsat: {args.landsat}")
    print(f"  DEM: {args.dem}")
    print(f"  Albedo: {args.albedo}")
    print(f"  LCZ: {args.lcz}")
    print(f"  建筑高度栅格: {args.building_height}")
    print(f"  建筑数据: {args.building}")
    print("    字段: height, area, voroniarea, min_proj, max_proj, district_id, spectral_cluster")
    print(f"  Voronoi图: {args.voronoi_diagram}")
    print(f"\n输出: {args.output}")
    if args.cachedir:
        print(f"缓存目录: {args.cachedir}")
        if args.restart:
            print("  (强制重新计算)")

    # 执行工作流
    # 1. 加载数据
    collection = load_and_align_data(
        era5_path=args.era5,
        landsat_path=args.landsat,
        albedo_path=args.albedo,
        dem_path=args.dem,
        lcz_path=args.lcz,
        building_height_path=args.building_height,
        building_path=args.building,
        voronoi_diagram_path=args.voronoi_diagram,
        target_crs=args.crs,
        target_resolution=args.resolution,
        cache_dir=args.cachedir,
        restart=args.restart
    )

    #sanitize_energy_balance_inputs(collection)
    #aligned_output = args.output.replace(".tif", "_aligned.tif")
    #display band name in collection
    #print(f"aligned collection bands: {list(collection.rasters.keys())}")
    #if aligned_output != args.output:
    #    output_bands = ("dem", "lcz", "roughness_length", "displacement_height", "era5_temperature_2m")
    #    collection.save(aligned_output, output_bands=output_bands)

    # 2. 计算空气动力学参数（返回包含原始数据和计算结果的新集合）
    # 输入的 collection 不会被修改（支持 CachedRasterCollection 作为只读输入）
    aero_cache_dir = None
    if args.cachedir:
        aero_cache_dir = str(Path(args.cachedir) / "aerodynamic")
    
    air_collection = calculate_aerodynamic_parameters(
        collection,
        cache_dir=aero_cache_dir,
        restart=args.restart
    )

    # 保存空气动力学参数
    #aero_output = args.output.replace(".tif", "_aerodynamic.tif")
    #if aero_output != args.output:
    #    air_collection.save(aero_output)

    # 解析日期时间
    try:
        datetime_obj = datetime.strptime(args.datetime, "%Y%m%d%H%M")
    except ValueError:
        print(f"错误: 无效的日期时间格式 '{args.datetime}'，请使用 YYYYMMDDHHMM 格式")
        sys.exit(1)

    print(f"\n观测时间: {datetime_obj.strftime('%Y-%m-%d %H:%M')}")

    # 3. 计算能量平衡系数
    # 从 collection 读取原始数据，从 air_collection 读取空气动力学参数
    balance_cache_dir = None
    if args.cachedir:
        balance_cache_dir = str(Path(args.cachedir) / "balance")
    
    _balance_collection = calculate_energy_balance(
        input_collection=collection,
        aero_collection=air_collection,
        datetime_obj=datetime_obj,
        std_meridian=args.std_meridian,
        cache_dir=balance_cache_dir,
        restart=args.restart
    )

    # 保存能量平衡系数
    #balance_output = args.output.replace(".tif", "_balance.tif")
    #if balance_output != args.output:
    #    _balance_collection.save(balance_output)

    print("\n" + "=" * 60)
    print("✓ 计算完成!")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 使用 regression.DistrictAggregator 聚合到街区")
    print("  2. 使用 regression.DistrictRegressionModel 进行ALS回归")
    print("  3. 求解每个街区的近地面气温 Ta")