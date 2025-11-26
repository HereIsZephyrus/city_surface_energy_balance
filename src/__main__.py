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
from typing import Dict
import numpy as np

from .utils import RasterCollection, RasterData, ERA5_BANDS, LANDSAT_BANDS
from .aerodynamics import calculate_aerodynamic_parameters
from .radiation import calculate_energy_balance


# ============================================================================
# 数据加载
# ============================================================================

def load_and_align_data(
    era5_path: str,
    landsat_path: str,
    dem_path: str,
    lcz_path: str,
    target_crs: str = 'EPSG:32650',
    target_resolution: float = 10.0
) -> Dict[str, np.ndarray]:
    """
    加载并对齐多源栅格数据
    
    参数:
        era5_path: ERA5 tif文件路径
        landsat_path: Landsat tif文件路径
        dem_path: DEM tif文件路径
        lcz_path: LCZ分类tif文件路径
        target_crs: 目标坐标系
        target_resolution: 目标分辨率(m)
    
    返回:
        dict: 包含所有对齐后数据的字典
    """
    print("\n" + "=" * 60)
    print("步骤 1: 加载和对齐多源数据")
    print("=" * 60)
    
    collection = RasterCollection(
        target_resolution=target_resolution,
        target_crs=target_crs
    )
    
    # 设置参考范围（Landsat NDVI）
    landsat_raster = RasterData(landsat_path)
    collection.set_reference(landsat_raster, band=LANDSAT_BANDS['ndvi'])
    
    # 添加Landsat数据
    print("\n加载 Landsat 数据...")
    for name, band in LANDSAT_BANDS.items():
        collection.add_raster(f'landsat_{name}', landsat_path, band=band)
    
    # 添加ERA5数据
    print("加载 ERA5 数据...")
    for name, band in ERA5_BANDS.items():
        collection.add_raster(f'era5_{name}', era5_path, band=band)
    
    # 添加DEM数据
    print("加载 DEM 数据...")
    collection.add_raster('dem', dem_path, band=1)
    
    # 添加LCZ数据
    print("加载 LCZ 数据...")
    collection.add_raster('lcz', lcz_path, band=1, resampling='nearest')
    
    # 转换为numpy数组
    result = _extract_arrays(collection)
    
    print(f"\n数据形状: {result['shape']}")
    print(f"有效像素: {np.sum(~result['nodata_mask'])}")
    
    return result


def _extract_arrays(collection: RasterCollection) -> Dict:
    """
    从RasterCollection提取numpy数组并组织为业务数据结构
    
    使用 RasterCollection.to_numpy_dict() 进行通用转换，
    然后按 ERA5/Landsat 数据源组织返回结构。
    """
    # 使用增强后的通用方法提取数组
    arrays = collection.to_numpy_dict(
        squeeze_bands=True,
        dtype=np.float64,
        nodata_values={'dem': -999}  # DEM特殊值处理
    )
    
    # 构建nodata掩码（基于Landsat NDVI）
    nodata_mask = np.isnan(arrays['landsat_ndvi'])
    for arr in arrays.values():
        arr[nodata_mask] = np.nan
    
    # 按数据源组织返回结构
    return {
        'era5': {k: arrays[f'era5_{k}'] for k in ERA5_BANDS.keys()},
        'landsat': {k: arrays[f'landsat_{k}'] for k in LANDSAT_BANDS.keys()},
        'dem': arrays['dem'],
        'lcz': arrays['lcz'],
        'shape': arrays['landsat_ndvi'].shape,
        'nodata_mask': nodata_mask,
        '_collection': collection
    }


# ============================================================================
# 结果保存
# ============================================================================

def save_results(
    data: Dict,
    aero_params: Dict,
    coeffs: Dict,
    output_path: str
) -> None:
    """
    保存计算结果到GeoTIFF
    
    使用 RasterCollection.save_multiband() 进行通用保存，
    仅在此定义业务特定的波段配置。
    """
    print("\n" + "=" * 60)
    print("步骤 4: 保存结果")
    print("=" * 60)
    
    collection = data['_collection']
    
    # 业务特定：定义输出波段配置
    bands = [
        ('f_Ta_coeff', coeffs['f_Ta_coeff']),
        ('residual', coeffs['residual']),
        ('air_temperature', data['era5']['temperature_2m']),
        ('surface_temperature', data['landsat']['lst']),
        ('air_density', aero_params['air_density']),
        ('wind_speed', aero_params['wind_speed']),
        ('actual_vapor_pressure', aero_params['actual_vapor_pressure']),
        ('saturation_vapor_pressure', aero_params['saturation_vapor_pressure']),
        ('aerodynamic_resistance', aero_params['aerodynamic_resistance']),
        ('surface_resistance', aero_params['surface_resistance']),
        ('soil_heat_flux', coeffs['soil_heat_flux']),
        ('latent_heat_flux', coeffs['latent_heat_flux']),
        ('elevation', data['dem']),
        ('ndvi', data['landsat']['ndvi']),
    ]
    
    # 使用通用方法保存
    collection.save_multiband(output_path=output_path, bands=bands)
    
    # 打印波段列表
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
    python -m src era5.tif --landsat landsat.tif --dem dem.tif \\
        --lcz lcz.tif -o result.tif

LCZ编码:
    1-9:  城市建筑类型 (密集高层~稀疏建筑)
    10:   裸岩/铺装 (E)    11: 密集树木 (A)
    12:   灌木/低矮植被 (C/D)  13: 裸土/沙地 (F)  14: 水体 (G)

波段配置:
    ERA5: surface_pressure(19), dewpoint_2m(18), u_wind(14), v_wind(15), temp_2m(16)
    Landsat: ndvi(9), fvc(10), lst(19), albedo(11), emissivity(12)
        """
    )
    
    parser.add_argument('era5', help='ERA5 tif文件路径')
    parser.add_argument('--landsat', required=True, help='Landsat tif文件路径')
    parser.add_argument('--dem', required=True, help='DEM tif文件路径')
    parser.add_argument('--lcz', required=True, help='LCZ分类tif (1-9城市, 10-14自然)')
    parser.add_argument('-o', '--output', required=True, help='输出tif文件路径')
    parser.add_argument('--crs', default='EPSG:32650', help='目标坐标系 (默认: EPSG:32650)')
    parser.add_argument('--resolution', type=float, default=10.0, help='目标分辨率(m)')
    parser.add_argument('--shortwave', type=float, default=800.0, help='短波辐射(W/m²)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    for path in [args.era5, args.landsat, args.dem, args.lcz]:
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
    print(f"\n输出: {args.output}")
    
    # 执行工作流
    try:
        # 1. 加载数据
        data = load_and_align_data(
            era5_path=args.era5,
            landsat_path=args.landsat,
            dem_path=args.dem,
            lcz_path=args.lcz,
            target_crs=args.crs,
            target_resolution=args.resolution
        )
        
        # 2. 计算空气动力学参数（调用 aerodynamics 模块）
        aero_params = calculate_aerodynamic_parameters(
            data,
            pixel_size_meters=args.resolution
        )
        
        # 3. 计算能量平衡系数（调用 radiation 模块）
        coeffs = calculate_energy_balance(
            data, aero_params,
            shortwave_down=args.shortwave
        )
        
        # 4. 保存结果
        save_results(data, aero_params, coeffs, args.output)
        
        print("\n" + "=" * 60)
        print("✓ 计算完成!")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 使用 regression.DistrictAggregator 聚合到街区")
        print("  2. 使用 regression.DistrictRegressionModel 进行ALS回归")
        print("  3. 求解每个街区的近地面气温 Ta")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
