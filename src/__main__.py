"""
城市地表能量平衡计算 - 命令行接口

基于《晴朗无风条件下城市生态空间对城市降温作用量化模型》的工作流实现。

工作流程:
    1. 数据加载与对齐 (ERA5 + Landsat + DEM + LCZ)
    2. 空气动力学参数计算 (密度、风速、水汽压、阻抗)
    3. 能量平衡系数计算 (∂f/∂Ta, residual)
    4. 街区聚合与ALS回归求解 (可选)
    5. 结果输出

使用方法:
    python -m src <era5.tif> --landsat <landsat.tif> --dem <dem.tif> \\
        --lcz <lcz.tif> -o <output.tif>

LCZ编码:
    1-9:  城市建筑类型
    10:   裸岩/铺装 (E)
    11:   密集树木 (A)
    12:   灌木/低矮植被 (C/D)
    13:   裸土/沙地 (F)
    14:   水体 (G)

参考文献:
    doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
"""

import argparse
import sys
from pathlib import Path
import traceback
from datetime import datetime

from typing import Union, Optional
from .utils import RasterCollection, RasterData, ERA5_BANDS, LANDSAT_BANDS
from .utils.cached_collection import CachedRasterCollection
from .aerodynamics import calculate_aerodynamic_parameters
from .radiation import calculate_energy_balance
from .landscape import calculate_roughness_from_buildings, BUILDING_FIELDS

# ============================================================================
# 数据加载
# ============================================================================

def load_and_align_data(
    era5_path: str,
    landsat_path: str,
    dem_path: str,
    lcz_path: str,
    building_path: Optional[str] = None,
    target_crs: str = 'EPSG:32650',
    target_resolution: float = 10.0,
    cache_dir: str = None,
    restart: bool = False
) -> Union[RasterCollection, CachedRasterCollection]:
    """
    加载并对齐多源栅格数据

    参数:
        era5_path: ERA5 tif文件路径
        landsat_path: Landsat tif文件路径
        dem_path: DEM tif文件路径
        lcz_path: LCZ栅格文件路径（.tif，值为1-14的LCZ分类）
        building_path: 建筑数据文件路径（.gpkg，可选，用于计算粗糙度）
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

    print("加载 ERA5 数据...")
    collection.load_multiband(era5_path, ERA5_BANDS, prefix='era5')

    print("加载 DEM 数据...")
    collection.add_raster('dem', dem_path, band=1)

    # 添加LCZ数据（栅格输入）
    print("加载 LCZ 数据...")
    collection.add_raster('lcz', lcz_path, band=1, resampling='nearest')

    # 如果提供了建筑数据，计算粗糙度参数
    if building_path is not None:
        print("\n加载建筑数据并计算粗糙度...")
        z_0, z_d, building_height = calculate_roughness_from_buildings(
            building_path=building_path,
            collection=collection,
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
        collection.add_array('building_height', building_height)

    # 如果使用缓存模式，保存元信息
    if isinstance(collection, CachedRasterCollection):
        collection.save_metadata()

    return collection

# ============================================================================
# 结果保存
# ============================================================================

# 输出波段配置
OUTPUT_BANDS = [
    'f_Ta_coeff2',            # 二次项系数 (W/m²/K²)
    'f_Ta_coeff1',            # 一次项系数 (W/m²/K)
    'residual',
    'shortwave_down',
    'era5_temperature_2m',
    'landsat_lst',
    'air_density',
    'wind_speed',
    'actual_vapor_pressure',
    'saturation_vapor_pressure',
    'aerodynamic_resistance',
    'surface_resistance',
    'soil_heat_flux',
    'latent_heat_flux',
    'dem',
    'landsat_ndvi',
    'roughness_length',       # 从建筑数据计算的粗糙度长度
    'displacement_height',    # 从建筑数据计算的零平面位移高度
    'building_height',        # 建筑高度
]


def save_results(
    collection: Union[RasterCollection, CachedRasterCollection],
    output_path: str,
    band_names: list = None
) -> None:
    """
    保存单个集合的计算结果到GeoTIFF
    
    参数:
        collection: 包含数据的集合
        output_path: 输出文件路径
        band_names: 要保存的波段名称列表（默认保存集合中所有波段）
    """
    print("\n" + "=" * 60)
    print(f"保存结果: {output_path}")
    print("=" * 60)

    # 确定要保存的波段
    if band_names is None:
        # 默认保存集合中所有波段
        band_names = list(collection.rasters.keys())

    # 从集合中获取波段数据
    bands = []
    for name in band_names:
        if name in collection.rasters:
            arr = collection.get_array(name)
            bands.append((name, arr))
        else:
            print(f"  警告: 波段 '{name}' 不存在，跳过")

    # 保存
    collection.save_multiband(output_path=output_path, bands=bands)

    # 打印波段列表
    print(f"  保存了 {len(bands)} 个波段:")
    for i, (name, _) in enumerate(bands, 1):
        print(f"    {i:2d}. {name}")


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='城市地表能量平衡计算',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法
    python -m src --era5 era5.tif --landsat landsat.tif --dem dem.tif \\
        --lcz lcz.tif --datetime 202308151030 -o result.tif
    
    # 使用建筑数据计算粗糙度
    python -m src --era5 era5.tif --landsat landsat.tif --dem dem.tif \\
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
    district_id:       街区ID
    spectral_cluster: 建筑群聚类标签

波段配置:
    ERA5: surface_pressure(19), dewpoint_2m(18), u_wind(14), v_wind(15), temp_2m(16)
    Landsat: ndvi(9), fvc(10), lst(19), albedo(11), emissivity(12)

注意:
    短波辐射使用水平面太阳辐射计算，适用于城市冠层模型 (Urban Canopy Layer)。
        """
    )

    parser.add_argument('--era5', required=True, help='ERA5 tif文件路径')
    parser.add_argument('--landsat', required=True, help='Landsat tif文件路径')
    parser.add_argument('--dem', required=True, help='DEM tif文件路径')
    parser.add_argument('--lcz', required=True, help='LCZ栅格文件路径 (.tif，值为1-14的LCZ分类)')
    parser.add_argument('-o', '--output', required=True, help='输出tif文件路径')
    parser.add_argument('--crs', default='EPSG:32650', help='目标坐标系 (默认: EPSG:32650)')
    parser.add_argument('--resolution', type=float, default=10.0, help='目标分辨率(m)')
    parser.add_argument('--datetime', required=True, help='观测日期时间 (格式: YYYYMMDDHHMM，如 202308151030)')
    parser.add_argument('--std-meridian', type=float, default=120.0, help='标准经度 (默认: 120.0 东八区)')
    parser.add_argument('--cachedir', type=str, help='缓存目录路径（启用后数据存储到磁盘而非内存）')
    parser.add_argument('--restart', action='store_true', help='强制重新计算并覆盖缓存（需配合 --cachedir 使用）')
    parser.add_argument('--building', type=str, default=None, help='建筑数据文件路径 (.gpkg，需包含 height, area, voroniarea, min_proj, max_proj, district_id, spectral_cluster 字段)')

    args = parser.parse_args()

    # 检查输入文件
    required_files = [args.era5, args.landsat, args.dem, args.lcz]
    if args.building:
        required_files.append(args.building)
    
    for path in required_files:
        if not Path(path).exists():
            print(f"错误: 文件不存在 - {path}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("城市地表能量平衡计算")
    print("=" * 60)
    print(f"\n输入文件:")
    print(f"  ERA5: {args.era5}")
    print(f"  Landsat: {args.landsat}")
    print(f"  DEM: {args.dem}")
    print(f"  LCZ: {args.lcz}")
    if args.building:
        print(f"  建筑数据: {args.building}")
        print(f"    字段: height, area, voroniarea, min_proj, max_proj, district_id, spectral_cluster")
    print(f"\n输出: {args.output}")
    if args.cachedir:
        print(f"缓存目录: {args.cachedir}")
        if args.restart:
            print("  (强制重新计算)")

    # 执行工作流
    try:
        # 1. 加载数据
        collection = load_and_align_data(
            era5_path=args.era5,
            landsat_path=args.landsat,
            dem_path=args.dem,
            lcz_path=args.lcz,
            building_path=args.building,
            target_crs=args.crs,
            target_resolution=args.resolution,
            cache_dir=args.cachedir,
            restart=args.restart
        )

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
        air_output_path = args.output.replace(".tif", "_aerodynamic.tif")
        save_results(air_collection, air_output_path)

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
        
        balance_collection = calculate_energy_balance(
            input_collection=collection,
            aero_collection=air_collection,
            datetime_obj=datetime_obj,
            std_meridian=args.std_meridian,
            cache_dir=balance_cache_dir,
            restart=args.restart
        )

        # 4. 保存结果（需要从两个集合获取数据）
        balance_output_path = args.output.replace(".tif", "_balance.tif")
        save_results(balance_collection, balance_output_path)

        print("\n" + "=" * 60)
        print("✓ 计算完成!")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 使用 regression.DistrictAggregator 聚合到街区")
        print("  2. 使用 regression.DistrictRegressionModel 进行ALS回归")
        print("  3. 求解每个街区的近地面气温 Ta")

    except Exception as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
