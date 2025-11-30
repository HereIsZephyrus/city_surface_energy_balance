"""
城市地表能量平衡 - 街区级回归工作流

基于能量平衡的交替最小二乘法（ALS）回归，从栅格能量平衡系数求解街区气温。

工作流程:
    1. 加载 physics 模块输出的能量平衡系数缓存
    2. 加载街区矢量数据
    3. 栅格到街区的空间聚合
    4. ALS 回归求解各街区气温（Q_F, ΔQ_Sb）
    5. 空间滞后模型分析气温空间自相关（ΔQ_A）
    6. 结果输出

物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si = 0
    
    水平交换项 ΔQ_A（空间滞后模型）:
    Ta = μ + ρ·W·Ta + ε
"""

from __future__ import annotations

import sys
from pathlib import Path
import traceback
from typing import Optional, List
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd

from .utils.cached_collection import CachedRasterCollection
from .regression.aggregator import DistrictAggregator
from .regression.als_regression import ALSRegression
from .regression.spatial_analysis import SpatialWeightMatrix, analyze_spatial_autocorrelation


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


def _get_aggregation_cache_key(
    cache_dir: str,
    districts_path: str,
    aggregate_bands: List[str],
    stats: List[str]
) -> str:
    """生成聚合缓存的唯一键"""
    # 获取街区文件的修改时间作为版本标识
    districts_mtime = Path(districts_path).stat().st_mtime
    
    # 组合所有信息生成哈希
    key_str = f"{cache_dir}|{districts_path}|{districts_mtime}|{sorted(aggregate_bands)}|{sorted(stats)}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_aggregation_cache_path(cache_dir: str) -> Path:
    """获取聚合缓存目录"""
    cache_path = Path(cache_dir)
    aggregation_cache_dir = cache_path / "aggregation_cache"
    aggregation_cache_dir.mkdir(parents=True, exist_ok=True)
    return aggregation_cache_dir


def _load_aggregated_cache(cache_key: str, cache_dir: str) -> Optional[pd.DataFrame]:
    """加载聚合结果缓存"""
    cache_dir_path = _get_aggregation_cache_path(cache_dir)
    cache_file = cache_dir_path / f"{cache_key}.parquet"
    
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            print(f"✓ 从缓存加载聚合结果: {cache_file}")
            return df
        except Exception as e:
            print(f"  警告: 加载聚合缓存失败 ({e})，将重新计算")
            return None
    return None


def _save_aggregated_cache(df: pd.DataFrame, cache_key: str, cache_dir: str) -> None:
    """保存聚合结果到缓存"""
    cache_dir_path = _get_aggregation_cache_path(cache_dir)
    cache_file = cache_dir_path / f"{cache_key}.parquet"
    
    try:
        df.to_parquet(cache_file, index=False)
        print(f"✓ 聚合结果已缓存: {cache_file}")
    except Exception as e:
        print(f"  警告: 保存聚合缓存失败 ({e})")


def run_regression(
    cache_dir: str,
    districts_path: str,
    output_path: str,
    district_id_field: str = 'district_id',
    X_F_columns: Optional[List[str]] = None,
    X_S_columns: Optional[List[str]] = None,
    X_C_columns: Optional[List[str]] = None,
    distance_threshold: float = 500.0,
    distance_decay: str = 'binary',
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
        X_F_columns: 人为热 Q_F 相关特征列名（连续变量）
        X_S_columns: 建筑储热 ΔQ_Sb 相关特征列名（连续变量）
        X_C_columns: 分类特征列名（将进行 one-hot 编码）
        distance_threshold: 空间权重距离阈值（米）
        distance_decay: 距离衰减函数
        max_iter: ALS最大迭代次数
        tol: 收敛容差
        verbose: 是否打印详细信息
        
    返回:
        包含回归结果的 GeoDataFrame
    """
    print("\n" + "=" * 60)
    print("街区级回归分析")
    print("=" * 60)
    
    # 1. 检查缓存路径
    print("\n[1] 检查缓存数据...")
    cache_path = Path(cache_dir)
    
    # 收集可用的缓存路径（延迟加载）
    cache_sources = []
    
    balance_cache = cache_path / "balance"
    if balance_cache.exists() and CachedRasterCollection.cache_exists(balance_cache):
        cache_sources.append(('balance', balance_cache))
        print(f"  发现 balance 缓存: {balance_cache}")
    
    if CachedRasterCollection.cache_exists(cache_path):
        cache_sources.append(('root', cache_path))
        print(f"  发现根目录缓存: {cache_path}")
    
    if not cache_sources:
        raise FileNotFoundError(f"缓存不存在: {cache_path}")
    
    # 2. 加载街区数据
    print("\n[2] 加载街区矢量数据...")
    districts_gdf = gpd.read_file(districts_path)
    print(f"  街区数量: {len(districts_gdf)}")
    print(f"  可用字段: {list(districts_gdf.columns)}")
    
    if district_id_field not in districts_gdf.columns:
        districts_gdf[district_id_field] = range(len(districts_gdf))
        print(f"  自动生成 {district_id_field}")
    
    # 3. 栅格聚合到街区（带缓存）
    print("\n[3] 栅格聚合到街区...")
    
    # 先收集所有需要聚合的波段，用于生成缓存键
    all_bands = []
    for source_name, source_path in cache_sources:
        collection = CachedRasterCollection.load_from_cache(source_path)
        for band_name in DEFAULT_AGGREGATE_BANDS:
            if band_name in collection.rasters:
                all_bands.append(band_name)
        del collection
    
    # 检查聚合缓存
    aggregated_df = None
    if all_bands:
        cache_key = _get_aggregation_cache_key(
            cache_dir=cache_dir,
            districts_path=districts_path,
            aggregate_bands=sorted(set(all_bands)),
            stats=['mean', 'std']
        )
        aggregated_df = _load_aggregated_cache(cache_key, cache_dir)
    
    # 如果缓存不存在，执行聚合
    if aggregated_df is None:
        import gc
        
        for source_name, source_path in cache_sources:
            # 加载 collection
            print(f"  加载 {source_name} 缓存: {source_path}")
            collection = CachedRasterCollection.load_from_cache(source_path)
            print(f"    波段: {list(collection.rasters.keys())}")
            
            # 找出该 collection 中需要聚合的波段
            rasters = {}
            for band_name in DEFAULT_AGGREGATE_BANDS:
                if band_name in collection.rasters:
                    rasters[band_name] = collection.get_array(band_name)
            
            if not rasters:
                # 释放 collection 资源
                del collection
                gc.collect()
                continue
            
            print(f"    聚合 {len(rasters)} 个波段: {list(rasters.keys())}")
            
            # 获取地理参考信息
            if hasattr(collection, '_reference_info') and collection._reference_info:
                transform = collection._reference_info.get('transform')
            else:
                raise ValueError(f"缓存 {source_name} 缺少地理参考信息")
            
            # 聚合
            df = DistrictAggregator.aggregate_rasters_to_districts(
                rasters=rasters,
                districts_gdf=districts_gdf,
                transform=transform,
                stats=['mean', 'std'],
                nodata=np.nan
            )
            
            # 释放栅格数据
            del rasters
            
            # 释放 collection 资源
            del collection
            gc.collect()
            print(f"    已释放 {source_name} 缓存资源")
            
            # 合并聚合结果
            if aggregated_df is None:
                aggregated_df = df
            else:
                # 合并，避免重复列
                for col in df.columns:
                    if col not in aggregated_df.columns:
                        aggregated_df[col] = df[col]
                del df
        
        if aggregated_df is None:
            raise ValueError("没有可用的波段进行聚合")
        
        # 保存到缓存
        if all_bands:
            cache_key = _get_aggregation_cache_key(
                cache_dir=cache_dir,
                districts_path=districts_path,
                aggregate_bands=sorted(set(all_bands)),
                stats=['mean', 'std']
            )
            _save_aggregated_cache(aggregated_df, cache_key, cache_dir)
    
    if aggregated_df is None:
        raise ValueError("没有可用的波段进行聚合")
    
    print(f"  聚合结果列: {list(aggregated_df.columns)}")
    
    # 4. ALS 回归
    print("\n[4] 准备回归数据...")
    model = ALSRegression()
    
    f_Ta_column = 'f_Ta_coeff1_mean'
    residual_column = 'residual_mean'
    era5_Ta_column = 'era5_temperature_2m_mean'
    
    required_cols = [f_Ta_column, residual_column]
    for col in required_cols:
        if col not in aggregated_df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    X_F, X_S, f_Ta_coeffs, y_residual, era5_Ta_mean = model.prepare_regression_data(
        aggregated_df=aggregated_df,
        districts_gdf=districts_gdf,
        f_Ta_column=f_Ta_column,
        residual_column=residual_column,
        era5_Ta_column=era5_Ta_column,
        X_F_columns=X_F_columns,
        X_S_columns=X_S_columns,
        X_C_columns=X_C_columns
    )
    
    print(f"  特征维度: X_F={X_F.shape}, X_S={X_S.shape}")
    print(f"  样本数量: {len(f_Ta_coeffs)}")
    
    print("\n[5] ALS 回归求解...")
    results = model.fit(
        X_F=X_F,
        X_S=X_S,
        f_Ta_coeffs=f_Ta_coeffs,
        y_residual=y_residual,
        era5_Ta_mean=era5_Ta_mean,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose
    )
    
    # 5. 空间滞后模型分析（必要步骤）
    print("\n[6] 空间滞后模型分析（水平交换项 ΔQ_A）...")
    
    spatial_weights = SpatialWeightMatrix(
        districts_gdf=districts_gdf,
        distance_threshold=distance_threshold,
        decay_function=distance_decay,
        row_standardize=True
    )
    
    spatial_results = analyze_spatial_autocorrelation(
        Ta=model.Ta_per_district,
        spatial_weights=spatial_weights,
        verbose=verbose
    )
    
    # 6. 整理结果
    print("\n[7] 整理结果...")
    results_df = model.get_results_dataframe(
        districts_gdf=districts_gdf,
        X_F_columns=X_F_columns,
        X_S_columns=X_S_columns,
        spatial_analysis=spatial_results
    )
    
    # 合并到街区 GeoDataFrame
    output_gdf = districts_gdf.merge(results_df, on=district_id_field, how='left')
    
    # 添加聚合数据
    output_gdf = output_gdf.merge(
        aggregated_df.drop(columns=[district_id_field], errors='ignore'),
        left_index=True, right_index=True, how='left'
    )
    
    # 7. 保存结果
    print(f"\n[8] 保存结果: {output_path}")
    output_path = Path(output_path)
    
    if output_path.suffix == '.gpkg':
        output_gdf.to_file(output_path, driver='GPKG')
    elif output_path.suffix == '.csv':
        output_gdf.drop(columns=['geometry'], errors='ignore').to_csv(output_path, index=False)
    elif output_path.suffix == '.parquet':
        output_gdf.to_parquet(output_path)
    else:
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
    
    # 空间分析摘要
    if spatial_results is not None:
        print("\n空间自相关分析（水平交换项 ΔQ_A）:")
        print(f"  Moran's I = {spatial_results['moran_i']:.4f}")
        print(f"  空间自相关系数 ρ = {spatial_results['rho']:.4f}")
        print(f"  R² (方差解释比例): {spatial_results['r_squared']*100:.1f}%")
        if spatial_results['moran_p'] < 0.05:
            print("  结论: 存在显著的空间自相关 (p < 0.05)")
    
    return output_gdf


def main(args=None):
    """
    回归分析主函数
    """
    import argparse
    
    if args is None:
        parser = argparse.ArgumentParser(
            description='街区级回归分析 - 从能量平衡系数求解气温',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
    python -m src.workflow --cachedir ./cache --districts districts.gpkg -o result.gpkg
    
物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si = 0
    
    水平交换项 ΔQ_A（空间滞后模型）:
    Ta = μ + ρ·W·Ta + ε
            """
        )
        parser.add_argument('--cachedir', required=True, 
                            help='缓存目录路径（physics模块输出）')
        parser.add_argument('--districts', required=True, 
                            help='街区矢量数据路径 (.gpkg)')
        parser.add_argument('-o', '--output', required=True, 
                            help='输出文件路径')
        parser.add_argument('--district-id', default='district_id', 
                            help='街区ID字段名')
        parser.add_argument('--x-f', type=str, default=None,
                            help='人为热Q_F特征列名（连续变量），逗号分隔')
        parser.add_argument('--x-s', type=str, default=None,
                            help='储热ΔQ_Sb特征列名（连续变量），逗号分隔')
        parser.add_argument('--x-c', type=str, default=None,
                            help='分类特征列名（将进行one-hot编码），逗号分隔')
        parser.add_argument('--distance-threshold', type=float, default=500.0,
                            help='空间权重距离阈值(m)')
        parser.add_argument('--distance-decay', type=str, default='binary',
                            choices=['binary', 'linear', 'inverse', 'gaussian'],
                            help='距离衰减函数')
        parser.add_argument('--max-iter', type=int, default=20,
                            help='ALS最大迭代次数')
        parser.add_argument('--tol', type=float, default=1e-4,
                            help='收敛容差')
        parser.add_argument('--quiet', action='store_true',
                            help='静默模式')
        args = parser.parse_args()

    X_F_columns = [x.strip() for x in args.x_f.split(',')] if args.x_f and args.x_f.strip() else None
    X_S_columns = [x.strip() for x in args.x_s.split(',')] if args.x_s and args.x_s.strip() else None
    X_C_columns = [x.strip() for x in args.x_c.split(',')] if args.x_c and args.x_c.strip() else None
    
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
        print(f"  人为热特征（连续）: {X_F_columns}")
    if X_S_columns:
        print(f"  储热特征（连续）: {X_S_columns}")
    if X_C_columns:
        print(f"  分类特征（one-hot编码）: {X_C_columns}")
    print(f"  空间分析:")
    print(f"    距离阈值: {args.distance_threshold} m")
    print(f"    衰减函数: {args.distance_decay}")

    try:
        run_regression(
            cache_dir=args.cachedir,
            districts_path=args.districts,
            output_path=args.output,
            district_id_field=args.district_id,
            X_F_columns=X_F_columns,
            X_S_columns=X_S_columns,
            X_C_columns=X_C_columns,
            distance_threshold=args.distance_threshold,
            distance_decay=args.distance_decay,
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

