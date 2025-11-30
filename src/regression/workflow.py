"""
回归工作流 - 街区级气温反演

基于能量平衡的交替最小二乘法（ALS）回归，从栅格能量平衡系数求解街区气温。

工作流程:
    1. 加载 physics 模块输出的能量平衡系数缓存
    2. 加载街区矢量数据
    3. 栅格到街区的空间聚合
    4. ALS 回归求解各街区气温
    5. 结果输出
"""

import argparse
import sys
from pathlib import Path
import traceback
from typing import Optional, List
import numpy as np
import geopandas as gpd

from ..utils.cached_collection import CachedRasterCollection
from .district_regression import DistrictAggregator, DistrictRegressionModel


# ============================================================================
# 配置
# ============================================================================

# 默认聚合的栅格波段
DEFAULT_AGGREGATE_BANDS = [
    'f_Ta_coeff1',              # Ta一次项系数 (W/m²/K)
    'f_Ta_coeff2',              # Ta二次项系数 (W/m²/K²)
    'residual',                 # 能量平衡残差
    'era5_temperature_2m',      # ERA5 2米气温 (K)
    'landsat_lst',              # 地表温度 (K)
    'shortwave_down',           # 下行短波辐射 (W/m²)
    'soil_heat_flux',           # 土壤热通量 (W/m²)
    'latent_heat_flux',         # 潜热通量 (W/m²)
    'landsat_ndvi',             # NDVI
    'landsat_fvc',              # 植被覆盖度
]


# ============================================================================
# 工作流函数
# ============================================================================

def run_district_regression(
    cache_dir: str,
    districts_path: str,
    output_path: str,
    district_id_field: str = 'district_id',
    X_F_columns: Optional[List[str]] = None,
    X_S_columns: Optional[List[str]] = None,
    X_A_columns: Optional[List[str]] = None,
    max_iter: int = 20,
    tol: float = 1e-4,
    verbose: bool = True
) -> gpd.GeoDataFrame:
    """
    执行街区级回归分析
    
    参数:
        cache_dir: 缓存目录路径（physics模块的输出）
        districts_path: 街区矢量数据路径 (.gpkg)
        output_path: 输出文件路径 (.gpkg 或 .csv)
        district_id_field: 街区ID字段名
        X_F_columns: 人为热 Q_F 相关特征列名
        X_S_columns: 建筑储热 ΔQ_Sb 相关特征列名
        X_A_columns: 水平交换 ΔQ_A 相关特征列名
        max_iter: ALS最大迭代次数
        tol: 收敛容差
        verbose: 是否打印详细信息
        
    返回:
        包含回归结果的 GeoDataFrame
    """
    print("\n" + "=" * 60)
    print("街区级回归分析")
    print("=" * 60)
    
    # 1. 加载缓存数据
    print("\n[1] 加载能量平衡系数缓存...")
    cache_path = Path(cache_dir)
    
    # 检查不同级别的缓存
    balance_cache = cache_path / "balance"
    if balance_cache.exists() and CachedRasterCollection.cache_exists(balance_cache):
        collection = CachedRasterCollection.load_from_cache(balance_cache)
        print(f"  从 balance 子目录加载: {balance_cache}")
    elif CachedRasterCollection.cache_exists(cache_path):
        collection = CachedRasterCollection.load_from_cache(cache_path)
        print(f"  从根目录加载: {cache_path}")
    else:
        raise FileNotFoundError(f"缓存不存在: {cache_path}")
    
    print(f"  可用波段: {list(collection.rasters.keys())}")
    
    # 2. 加载街区数据
    print("\n[2] 加载街区矢量数据...")
    districts_gdf = gpd.read_file(districts_path)
    print(f"  街区数量: {len(districts_gdf)}")
    print(f"  可用字段: {list(districts_gdf.columns)}")
    
    # 确保有 district_id
    if district_id_field not in districts_gdf.columns:
        districts_gdf[district_id_field] = range(len(districts_gdf))
        print(f"  自动生成 {district_id_field}")
    
    # 3. 栅格聚合到街区
    print("\n[3] 栅格聚合到街区...")
    
    # 准备要聚合的栅格
    rasters = {}
    for band_name in DEFAULT_AGGREGATE_BANDS:
        if band_name in collection.rasters:
            rasters[band_name] = collection.get_array(band_name)
        else:
            print(f"  警告: 波段 '{band_name}' 不存在，跳过")
    
    print(f"  聚合 {len(rasters)} 个波段...")
    
    # 获取仿射变换
    if hasattr(collection, '_reference_info') and collection._reference_info:
        transform = collection._reference_info.get('transform')
    else:
        raise ValueError("缓存数据缺少地理参考信息")
    
    # 执行聚合
    aggregated_df = DistrictAggregator.aggregate_rasters_to_districts(
        rasters=rasters,
        districts_gdf=districts_gdf,
        transform=transform,
        stats=['mean', 'std'],
        nodata=np.nan
    )
    
    print(f"  聚合结果列: {list(aggregated_df.columns)}")
    
    # 4. 计算邻域特征（可选，用于水平交换项）
    if X_A_columns and 'landsat_lst_mean' in aggregated_df.columns:
        print("\n[4] 计算邻域特征...")
        # 先将聚合结果合并到街区
        districts_with_data = districts_gdf.merge(
            aggregated_df, on=district_id_field, how='left'
        )
        
        neighbor_df = DistrictAggregator.calculate_neighbor_features(
            districts_gdf=districts_with_data,
            value_column='landsat_lst_mean',
            id_column=district_id_field
        )
        
        aggregated_df = aggregated_df.merge(neighbor_df, on=district_id_field, how='left')
        print(f"  添加邻域特征: {list(neighbor_df.columns)}")
    
    # 5. 准备回归数据
    print("\n[5] 准备回归数据...")
    model = DistrictRegressionModel()
    
    # 确定可用的列名
    f_Ta_column = 'f_Ta_coeff1_mean'
    residual_column = 'residual_mean'
    era5_Ta_column = 'era5_temperature_2m_mean'
    
    # 检查必需列是否存在
    required_cols = [f_Ta_column, residual_column]
    for col in required_cols:
        if col not in aggregated_df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    X_F, X_S, X_A, f_Ta_coeffs, y_residual, era5_Ta_mean = model.prepare_regression_data(
        aggregated_df=aggregated_df,
        districts_gdf=districts_gdf,
        f_Ta_column=f_Ta_column,
        residual_column=residual_column,
        era5_Ta_column=era5_Ta_column,
        X_F_columns=X_F_columns,
        X_S_columns=X_S_columns,
        X_A_columns=X_A_columns
    )
    
    print(f"  特征维度: X_F={X_F.shape}, X_S={X_S.shape}, X_A={X_A.shape}")
    print(f"  样本数量: {len(f_Ta_coeffs)}")
    
    # 6. ALS 回归
    print("\n[6] ALS 回归求解...")
    results = model.fit_als_regression(
        X_F=X_F,
        X_S=X_S,
        X_A=X_A,
        f_Ta_coeffs=f_Ta_coeffs,
        y_residual=y_residual,
        era5_Ta_mean=era5_Ta_mean,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose
    )
    
    # 7. 整理结果
    print("\n[7] 整理结果...")
    results_df = model.get_results_dataframe(
        districts_gdf=districts_gdf,
        X_F_columns=X_F_columns,
        X_S_columns=X_S_columns,
        X_A_columns=X_A_columns
    )
    
    # 合并到街区 GeoDataFrame
    output_gdf = districts_gdf.merge(results_df, on=district_id_field, how='left')
    
    # 也添加聚合数据
    output_gdf = output_gdf.merge(
        aggregated_df.drop(columns=[district_id_field], errors='ignore'),
        left_index=True, right_index=True, how='left'
    )
    
    # 8. 保存结果
    print(f"\n[8] 保存结果: {output_path}")
    output_path = Path(output_path)
    
    if output_path.suffix == '.gpkg':
        output_gdf.to_file(output_path, driver='GPKG')
    elif output_path.suffix == '.csv':
        # CSV 需要去掉 geometry
        output_gdf.drop(columns=['geometry'], errors='ignore').to_csv(output_path, index=False)
    elif output_path.suffix == '.parquet':
        output_gdf.to_parquet(output_path)
    else:
        # 默认保存为 GeoPackage
        output_gdf.to_file(str(output_path) + '.gpkg', driver='GPKG')
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("回归结果摘要")
    print("=" * 60)
    print(f"  收敛: {'是' if results['converged'] else '否'}")
    print(f"  迭代次数: {results['n_iter']}")
    print(f"  残差范数: {results['residual_norm']:.4f}")
    Ta = results['Ta_per_district']
    print(f"  气温范围: {Ta.min()-273.15:.2f}°C ~ {Ta.max()-273.15:.2f}°C")
    print(f"  气温均值: {Ta.mean()-273.15:.2f}°C (±{Ta.std():.2f}K)")
    
    return output_gdf


def main(args: argparse.Namespace = None):
    """
    回归分析主函数
    
    参数:
        args: 命令行参数（如果为None则从sys.argv解析）
    """
    if args is None:
        parser = argparse.ArgumentParser(
            description='街区级回归分析 - 从能量平衡系数求解气温',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
    # 基本用法（使用 physics 模块的缓存）
    python -m src regression --cachedir ./cache --districts districts.gpkg -o result.gpkg
    
    # 指定回归特征
    python -m src regression --cachedir ./cache --districts districts.gpkg \\
        --x-f population,building_volume --x-s fvc_mean -o result.gpkg

物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si + γ·X^k_A = 0
    
    其中:
    - f(Ta): 能量平衡系数（关于Ta的函数）
    - X_Fi: 人为热 Q_F 相关特征
    - X_Si: 建筑储热 ΔQ_Sb 相关特征
    - X_A: 水平交换 ΔQ_A 相关特征
    - Ta: 每个街区的气温（ALS求解）
            """
        )
        parser.add_argument('--cachedir', required=True, 
                            help='缓存目录路径（physics模块输出）')
        parser.add_argument('--districts', required=True, 
                            help='街区矢量数据路径 (.gpkg)')
        parser.add_argument('-o', '--output', required=True, 
                            help='输出文件路径 (.gpkg/.csv/.parquet)')
        parser.add_argument('--district-id', default='district_id', 
                            help='街区ID字段名 (默认: district_id)')
        parser.add_argument('--x-f', type=str, default=None,
                            help='人为热Q_F特征列名，逗号分隔 (如: population,building_volume)')
        parser.add_argument('--x-s', type=str, default=None,
                            help='储热ΔQ_Sb特征列名，逗号分隔 (如: fvc_mean,building_volume)')
        parser.add_argument('--x-a', type=str, default=None,
                            help='水平交换ΔQ_A特征列名，逗号分隔 (如: landsat_lst_mean_neighbor_diff)')
        parser.add_argument('--max-iter', type=int, default=20,
                            help='ALS最大迭代次数 (默认: 20)')
        parser.add_argument('--tol', type=float, default=1e-4,
                            help='收敛容差 (默认: 1e-4)')
        parser.add_argument('--quiet', action='store_true',
                            help='静默模式，减少输出')
        args = parser.parse_args()

    # 解析特征列
    X_F_columns = args.x_f.split(',') if args.x_f else None
    X_S_columns = args.x_s.split(',') if args.x_s else None
    X_A_columns = args.x_a.split(',') if args.x_a else None
    
    # 检查输入文件
    if not Path(args.cachedir).exists():
        print(f"错误: 缓存目录不存在 - {args.cachedir}")
        sys.exit(1)
    
    if not Path(args.districts).exists():
        print(f"错误: 街区文件不存在 - {args.districts}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("街区级回归分析")
    print("=" * 60)
    print(f"\n输入:")
    print(f"  缓存目录: {args.cachedir}")
    print(f"  街区数据: {args.districts}")
    print(f"\n输出: {args.output}")
    if X_F_columns:
        print(f"  人为热特征: {X_F_columns}")
    if X_S_columns:
        print(f"  储热特征: {X_S_columns}")
    if X_A_columns:
        print(f"  水平交换特征: {X_A_columns}")

    # 执行回归
    try:
        run_district_regression(
            cache_dir=args.cachedir,
            districts_path=args.districts,
            output_path=args.output,
            district_id_field=args.district_id,
            X_F_columns=X_F_columns,
            X_S_columns=X_S_columns,
            X_A_columns=X_A_columns,
            max_iter=args.max_iter,
            tol=args.tol,
            verbose=not args.quiet
        )
        
        print("\n" + "=" * 60)
        print("✓ 回归分析完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

