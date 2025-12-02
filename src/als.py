"""
ALS 回归模块 - 城市地表能量平衡

基于能量平衡的交替最小二乘法（ALS）回归，从栅格能量平衡系数求解街区气温。

工作流程:
    1. 加载 physics 模块输出的能量平衡系数缓存
    2. 加载街区矢量数据
    3. 栅格到街区的空间聚合
    4. ALS 回归求解各街区气温（Q_F, ΔQ_Sb）
    5. 保存 Ta 栅格和系数 CSV

物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si = 0

使用方法:
    python -m src als --cachedir <cache_dir> --districts <districts.gpkg> \\
        --voronoi <voronoi.gpkg> -o <output_prefix>
"""

from __future__ import annotations

import sys
from pathlib import Path
import traceback
from typing import Optional, List, Dict
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize

from scipy.sparse import csr_matrix

from .utils.cached_collection import CachedRasterCollection
from .regression.aggregator import DistrictAggregator
from .regression.als_regression import ALSRegression


def load_spatial_weights(cache_dir: str, verbose: bool = True) -> Optional[csr_matrix]:
    """
    从缓存加载空间权重矩阵
    
    参数:
        cache_dir: 缓存目录路径
        verbose: 是否打印信息
        
    返回:
        空间权重矩阵 (csr_matrix)，如果不存在则返回 None
    """
    cache_path = Path(cache_dir)
    matrix_path = cache_path / 'spatial_weights_matrix.npz'
    
    if not matrix_path.exists():
        if verbose:
            print(f"  未找到空间权重矩阵缓存: {matrix_path}")
        return None
    
    try:
        # 使用手动重建方式，避免 numpy 版本兼容性问题
        npz = np.load(matrix_path, allow_pickle=True)
        data = npz['data']
        indices = npz['indices']
        indptr = npz['indptr']
        shape = tuple(npz['shape'])
        W = csr_matrix((data, indices, indptr), shape=shape)
        
        if verbose:
            print(f"  加载空间权重矩阵: {W.shape}, 非零元素={W.nnz}")
        
        return W
    except Exception as e:
        if verbose:
            print(f"  加载空间权重矩阵失败: {e}")
        return None


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
    # 空气动力学参数
    'air_density',              # 空气密度 (kg/m³)
    'wind_speed',               # 风速 (m/s)
    'actual_vapor_pressure',    # 实际水汽压 (kPa)
    'saturation_vapor_pressure', # 饱和水汽压 (kPa)
    'aerodynamic_resistance',   # 空气动力学阻抗 (s/m)
    'surface_resistance',       # 表面阻抗 (s/m)
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


def rasterize_district_values(
    districts_gdf: gpd.GeoDataFrame,
    values: np.ndarray,
    transform,
    shape: tuple,
    district_id_field: str = 'district_id',
    fill_value: float = np.nan
) -> np.ndarray:
    """
    将街区值栅格化为栅格数组
    
    参数:
        districts_gdf: 街区 GeoDataFrame
        values: 每个街区对应的值（与 districts_gdf 行数对应）
        transform: 目标栅格的仿射变换
        shape: 目标栅格形状 (height, width)
        district_id_field: 街区 ID 字段名
        fill_value: 填充值（默认 NaN）
        
    返回:
        np.ndarray: 栅格化后的数组
    """
    # 创建 (geometry, value) 对
    shapes = []
    for idx, row in districts_gdf.iterrows():
        geom = row.geometry
        if geom is not None and not geom.is_empty:
            # 获取对应的值
            if district_id_field in districts_gdf.columns:
                district_id = row[district_id_field]
                # 找到对应的值索引
                value_idx = districts_gdf[districts_gdf[district_id_field] == district_id].index[0]
                value_idx = districts_gdf.index.get_loc(value_idx)
            else:
                value_idx = idx
            
            if value_idx < len(values):
                value = float(values[value_idx])
                if np.isfinite(value):
                    shapes.append((geom, value))
    
    if not shapes:
        print("  警告: 没有有效的几何图形可栅格化")
        return np.full(shape, fill_value, dtype=np.float64)
    
    # 栅格化
    raster = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=fill_value,
        dtype=np.float64
    )
    
    return raster


def run_als_regression(
    cache_dir: str,
    districts_path: str,
    output_prefix: str,
    voronoi_path: Optional[str] = None,
    district_id_field: str = 'district_id',
    X_F_columns: Optional[List[str]] = None,
    X_S_columns: Optional[List[str]] = None,
    X_C_columns: Optional[List[str]] = None,
    spatial_weights_path: Optional[str] = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    ridge_alpha: float = 0.0,
    verbose: bool = True
) -> Dict:
    """
    执行 ALS 回归分析
    
    参数:
        cache_dir: 缓存目录路径（physics模块的输出）
        districts_path: 街区矢量数据路径 (.gpkg)
        output_prefix: 输出文件前缀（将生成 _Ta.tif 和 _coefficients.csv）
        voronoi_path: Voronoi 图路径（用于栅格化，可选，默认使用 districts）
        district_id_field: 街区ID字段名
        X_F_columns: 人为热 Q_F 相关特征列名（连续变量）
        X_S_columns: 建筑储热 ΔQ_Sb 相关特征列名（连续变量）
        X_C_columns: 分类特征列名（将进行 one-hot 编码）
        spatial_weights_path: 空间权重矩阵缓存目录路径
                             如果提供，将加入水平交换项 ΔQ_A = λ·(Ta - W·Ta)
                             None 则不考虑空间效应
        max_iter: ALS最大迭代次数
        tol: 收敛容差
        ridge_alpha: 岭回归正则化参数（0表示普通最小二乘，推荐值1e3-1e6）
        verbose: 是否打印详细信息
        
    返回:
        Dict: 包含结果的字典
            - 'Ta_per_district': 每个街区的气温
            - 'alpha_coeffs': 人为热系数
            - 'beta_coeffs': 储热系数
            - 'lambda_coeff': 空间交换系数（如果使用空间权重）
            - 'spatial_lag': 邻域平均气温（如果使用空间权重）
            - 'districts_result': 街区结果 GeoDataFrame
            - 'aggregated_df': 聚合后的 DataFrame
    """
    print("\n" + "=" * 60)
    print("ALS 回归分析")
    print("=" * 60)
    
    # 1. 检查缓存路径
    print("\n[1] 检查缓存数据...")
    cache_path = Path(cache_dir)
    
    # 收集可用的缓存路径（延迟加载）
    cache_sources = []
    
    # 检查 aerodynamic 缓存（包含空气动力学参数）
    aerodynamic_cache = cache_path / "aerodynamic"
    if aerodynamic_cache.exists() and CachedRasterCollection.cache_exists(aerodynamic_cache):
        cache_sources.append(('aerodynamic', aerodynamic_cache))
        print(f"  发现 aerodynamic 缓存: {aerodynamic_cache}")
    
    # 检查 balance 缓存（包含能量平衡参数）
    balance_cache = cache_path / "balance"
    if balance_cache.exists() and CachedRasterCollection.cache_exists(balance_cache):
        cache_sources.append(('balance', balance_cache))
        print(f"  发现 balance 缓存: {balance_cache}")
    
    # 检查根目录缓存
    if CachedRasterCollection.cache_exists(cache_path):
        cache_sources.append(('root', cache_path))
        print(f"  发现根目录缓存: {cache_path}")
    
    if not cache_sources:
        raise FileNotFoundError(f"缓存不存在: {cache_path}")
    
    # 获取参考信息（用于后续栅格化）
    reference_info = None
    for _, source_path in cache_sources:
        collection = CachedRasterCollection.load_from_cache(source_path)
        if hasattr(collection, '_reference_info') and collection._reference_info:
            reference_info = collection._reference_info
            break
    
    if reference_info is None:
        raise ValueError("无法获取栅格参考信息")
    
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
        
        # 验证聚合缓存是否包含所有预期的列
        if aggregated_df is not None:
            expected_columns = set()
            for band_name in all_bands:
                expected_columns.add(f"{band_name}_mean")
                expected_columns.add(f"{band_name}_std")
            expected_columns.add('district_id')
            
            missing_columns = expected_columns - set(aggregated_df.columns)
            if missing_columns:
                print(f"  警告: 聚合缓存缺少 {len(missing_columns)} 个预期列: {sorted(missing_columns)[:5]}...")
                print(f"  将重新聚合以包含所有可用波段")
                aggregated_df = None
    
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
    
    f_Ta_coeff1_column = 'f_Ta_coeff1_mean'
    f_Ta_coeff2_column = 'f_Ta_coeff2_mean'
    residual_column = 'residual_mean'
    era5_Ta_column = 'era5_temperature_2m_mean'
    
    required_cols = [f_Ta_coeff1_column, residual_column]
    for col in required_cols:
        if col not in aggregated_df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    # 检查是否有二次项系数
    has_coeff2 = f_Ta_coeff2_column in aggregated_df.columns
    if not has_coeff2:
        print(f"  警告: 未找到二次项系数 '{f_Ta_coeff2_column}'，将使用一次近似")
    
    X_F, X_S, f_Ta_coeff1, f_Ta_coeff2, y_residual, era5_Ta_mean, Ts_mean = model.prepare_regression_data(
        aggregated_df=aggregated_df,
        districts_gdf=districts_gdf,
        f_Ta_coeff1_column=f_Ta_coeff1_column,
        f_Ta_coeff2_column=f_Ta_coeff2_column if has_coeff2 else None,
        residual_column=residual_column,
        era5_Ta_column=era5_Ta_column,
        X_F_columns=X_F_columns,
        X_S_columns=X_S_columns,
        X_C_columns=X_C_columns
    )
    
    print(f"  特征维度: X_F={X_F.shape}, X_S={X_S.shape}")
    print(f"  样本数量: {len(f_Ta_coeff1)}")
    if f_Ta_coeff2 is not None:
        print(f"  使用二次能量平衡方程: f(Ta) = a×Ta² + b×Ta + c = 0")
    else:
        print(f"  使用一次能量平衡方程: f(Ta) = b×Ta + c = 0")
    
    # 加载空间权重矩阵（如果提供）
    spatial_weights = None
    spatial_mask = None  # 标记哪些样本在空间权重矩阵覆盖范围内
    if spatial_weights_path:
        print(f"\n[4.5] 加载空间权重矩阵...")
        spatial_weights = load_spatial_weights(spatial_weights_path, verbose=verbose)
        
        if spatial_weights is not None:
            n_valid = len(f_Ta_coeff1)
            n_spatial = spatial_weights.shape[0]
            
            if n_spatial != n_valid:
                print(f"  空间权重矩阵维度 ({n_spatial}) 与有效样本数 ({n_valid}) 不匹配")
                
                # 找出哪些有效样本在空间权重矩阵范围内
                valid_ids = model.valid_district_ids
                if valid_ids is not None and len(valid_ids) > 0:
                    valid_idx = np.array(valid_ids)
                    # 创建掩码：标记哪些样本的 ID 在空间矩阵范围内
                    spatial_mask = valid_idx < n_spatial
                    n_covered = np.sum(spatial_mask)
                    
                    if n_covered > 0:
                        print(f"  空间权重矩阵覆盖 {n_covered}/{n_valid} 个样本 ({100*n_covered/n_valid:.1f}%)")
                        # 提取覆盖样本对应的子矩阵
                        covered_ids = valid_idx[spatial_mask]
                        try:
                            from scipy.sparse import diags
                            sub_W = spatial_weights[covered_ids][:, covered_ids]
                            # 重新行标准化
                            row_sums = np.array(sub_W.sum(axis=1)).flatten()
                            row_sums[row_sums == 0] = 1
                            inv_row_sums = 1.0 / row_sums
                            spatial_weights = diags(inv_row_sums, format='csr') @ sub_W
                            print(f"  已提取子矩阵: {spatial_weights.shape}")
                        except Exception as e:
                            print(f"  无法提取子矩阵: {e}，禁用空间效应")
                            spatial_weights = None
                            spatial_mask = None
                    else:
                        print(f"  没有样本在空间权重矩阵覆盖范围内，禁用空间效应")
                        spatial_weights = None
                        spatial_mask = None
                else:
                    print(f"  无法获取有效样本索引，禁用空间效应")
                    spatial_weights = None
    
    print("\n[5] ALS 回归求解...")
    if ridge_alpha > 0:
        print(f"  使用岭回归正则化 (alpha={ridge_alpha:.2e})")
    if spatial_weights is not None:
        print(f"  启用空间交换项 ΔQ_A = λ·(Ta - W·Ta)")
        if spatial_mask is not None:
            print(f"  （仅对部分样本应用空间效应）")
    results = model.fit(
        X_F=X_F,
        X_S=X_S,
        f_Ta_coeff1=f_Ta_coeff1,
        f_Ta_coeff2=f_Ta_coeff2,
        y_residual=y_residual,
        era5_Ta_mean=era5_Ta_mean,
        Ts_mean=Ts_mean,
        spatial_weights=spatial_weights,
        spatial_mask=spatial_mask,
        max_iter=max_iter,
        tol=tol,
        ridge_alpha=ridge_alpha,
        verbose=verbose
    )
    
    # 6. 整理结果
    print("\n[6] 整理结果...")
    results_df = model.get_results_dataframe(
        districts_gdf=districts_gdf,
        X_F_columns=X_F_columns,
        X_S_columns=X_S_columns,
        spatial_analysis=None  # 空间分析将在单独的模块中进行
    )
    
    # 将 'district_id' 列重命名为实际使用的字段名
    if 'district_id' in results_df.columns and district_id_field != 'district_id':
        results_df = results_df.rename(columns={'district_id': district_id_field})
    
    # 合并到街区 GeoDataFrame
    output_gdf = districts_gdf.merge(results_df, on=district_id_field, how='left')
    
    # 添加聚合数据
    output_gdf = output_gdf.merge(
        aggregated_df.drop(columns=[district_id_field], errors='ignore'),
        left_index=True, right_index=True, how='left'
    )
    
    # 7. 保存结果
    print(f"\n[7] 保存结果...")
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存街区结果 (.gpkg)
    districts_output = output_prefix.with_suffix('.gpkg')
    output_gdf.to_file(districts_output, driver='GPKG')
    print(f"  街区结果: {districts_output}")
    
    # 保存系数 CSV
    coefficients_output = str(output_prefix) + '_coefficients.csv'
    coefficients_df = pd.DataFrame({
        'type': [],
        'name': [],
        'coefficient': []
    })
    
    # 添加 alpha 系数（人为热 Q_F）
    if model.alpha_coeffs is not None and model.alpha_coeffs.size > 0:
        for i, coeff in enumerate(model.alpha_coeffs):
            col_name = X_F_columns[i] if X_F_columns and i < len(X_F_columns) else f'alpha_{i}'
            coefficients_df = pd.concat([coefficients_df, pd.DataFrame({
                'type': ['Q_F (人为热)'],
                'name': [col_name],
                'coefficient': [coeff]
            })], ignore_index=True)
    
    # 添加 beta 系数（储热 ΔQ_Sb）
    if model.beta_coeffs is not None and model.beta_coeffs.size > 0:
        for i, coeff in enumerate(model.beta_coeffs):
            col_name = X_S_columns[i] if X_S_columns and i < len(X_S_columns) else f'beta_{i}'
            coefficients_df = pd.concat([coefficients_df, pd.DataFrame({
                'type': ['ΔQ_Sb (储热)'],
                'name': [col_name],
                'coefficient': [coeff]
            })], ignore_index=True)
    
    coefficients_df.to_csv(coefficients_output, index=False)
    print(f"  系数文件: {coefficients_output}")
    
    # 保存迭代历史 CSV
    if model.history:
        history_output = str(output_prefix) + '_iteration_history.csv'
        history_df = pd.DataFrame(model.history)
        history_df.to_csv(history_output, index=False)
        print(f"  迭代历史: {history_output}")
    
    # 8. 生成 Ta 栅格
    print("\n[8] 生成 Ta 栅格...")
    
    # 使用 voronoi 或 districts 进行栅格化
    if voronoi_path and Path(voronoi_path).exists():
        rasterize_gdf = gpd.read_file(voronoi_path)
        print(f"  使用 Voronoi 图进行栅格化: {voronoi_path}")
    else:
        rasterize_gdf = output_gdf
        print(f"  使用街区边界进行栅格化")
    
    # 确保 rasterize_gdf 包含 Ta 值
    if 'Ta_optimized' not in rasterize_gdf.columns:
        # 合并 Ta 结果到栅格化 GeoDataFrame
        if district_id_field in rasterize_gdf.columns and district_id_field in output_gdf.columns:
            ta_mapping = output_gdf[[district_id_field, 'Ta_optimized']].drop_duplicates()
            rasterize_gdf = rasterize_gdf.merge(ta_mapping, on=district_id_field, how='left')
        else:
            print("  警告: 无法将 Ta 值映射到栅格化几何图形")
    
    if 'Ta_optimized' in rasterize_gdf.columns:
        # 栅格化
        transform = reference_info['transform']
        shape = (reference_info['height'], reference_info['width'])
        
        Ta_values = rasterize_gdf['Ta_optimized'].values
        Ta_raster = rasterize_district_values(
            districts_gdf=rasterize_gdf,
            values=Ta_values,
            transform=transform,
            shape=shape,
            district_id_field=district_id_field,
            fill_value=np.nan
        )
        
        # 保存 Ta 栅格
        ta_output = str(output_prefix) + '_Ta.tif'
        
        import rasterio
        
        with rasterio.open(
            ta_output,
            'w',
            driver='GTiff',
            height=shape[0],
            width=shape[1],
            count=1,
            dtype=np.float64,
            crs=str(reference_info['crs']),
            transform=transform,
            compress='lzw',
            nodata=np.nan
        ) as dst:
            dst.write(Ta_raster, 1)
        
        print(f"  Ta 栅格: {ta_output}")
        print(f"    形状: {shape}")
        print(f"    Ta 范围: {np.nanmin(Ta_raster)-273.15:.2f}°C ~ {np.nanmax(Ta_raster)-273.15:.2f}°C")
    else:
        print("  警告: 无法生成 Ta 栅格（缺少 Ta_optimized 列）")
        Ta_raster = None
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("ALS 回归结果摘要")
    print("=" * 60)
    print(f"  收敛: {'是' if results['converged'] else '否'}")
    print(f"  迭代次数: {results['n_iter']}")
    print(f"  残差范数: {results['residual_norm']:.4f}")
    Ta = results['Ta_per_district']
    print(f"  气温范围: {Ta.min()-273.15:.2f}°C ~ {Ta.max()-273.15:.2f}°C")
    print(f"  气温均值: {Ta.mean()-273.15:.2f}°C (±{Ta.std():.2f}K)")
    
    if model.alpha_coeffs is not None and model.alpha_coeffs.size > 0:
        print(f"\n人为热系数 (α):")
        for i, coeff in enumerate(model.alpha_coeffs):
            col_name = X_F_columns[i] if X_F_columns and i < len(X_F_columns) else f'alpha_{i}'
            print(f"    {col_name}: {coeff:.6f}")
    
    if model.beta_coeffs is not None and model.beta_coeffs.size > 0:
        print(f"\n储热系数 (β):")
        for i, coeff in enumerate(model.beta_coeffs):
            col_name = X_S_columns[i] if X_S_columns and i < len(X_S_columns) else f'beta_{i}'
            print(f"    {col_name}: {coeff:.6f}")
    
    return {
        'Ta_per_district': model.Ta_per_district,
        'alpha_coeffs': model.alpha_coeffs,
        'beta_coeffs': model.beta_coeffs,
        'districts_result': output_gdf,
        'aggregated_df': aggregated_df,
        'reference_info': reference_info,
        'converged': results['converged'],
        'n_iter': results['n_iter'],
        'residual_norm': results['residual_norm']
    }


def main(args=None):
    """
    ALS 回归主函数
    """
    import argparse
    
    if args is None:
        parser = argparse.ArgumentParser(
            description='ALS 回归分析 - 从能量平衡系数求解气温',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
    python -m src als --cachedir ./cache --districts districts.gpkg -o result
    
物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si = 0

输出文件:
    <output_prefix>.gpkg          - 街区结果（含 Ta 和聚合数据）
    <output_prefix>_Ta.tif        - Ta 栅格
    <output_prefix>_coefficients.csv - 回归系数
            """
        )
        parser.add_argument('--cachedir', required=True, 
                            help='缓存目录路径（physics模块输出）')
        parser.add_argument('--districts', required=True, 
                            help='街区矢量数据路径 (.gpkg)')
        parser.add_argument('--voronoi', type=str, default=None,
                            help='Voronoi图路径（用于栅格化，可选）')
        parser.add_argument('-o', '--output', required=True, 
                            help='输出文件前缀')
        parser.add_argument('--district-id', default='district_id', 
                            help='街区ID字段名')
        parser.add_argument('--x-f', type=str, default=None,
                            help='人为热Q_F特征列名（连续变量），逗号分隔')
        parser.add_argument('--x-s', type=str, default=None,
                            help='储热ΔQ_Sb特征列名（连续变量），逗号分隔')
        parser.add_argument('--x-c', type=str, default=None,
                            help='分类特征列名（将进行one-hot编码），逗号分隔')
        parser.add_argument('--spatial-weights', type=str, default=None,
                            help='空间权重矩阵缓存目录路径，启用水平交换项 ΔQ_A')
        parser.add_argument('--max-iter', type=int, default=100,
                            help='ALS最大迭代次数')
        parser.add_argument('--tol', type=float, default=1e-4,
                            help='收敛容差')
        parser.add_argument('--ridge-alpha', type=float, default=0.0,
                            help='岭回归正则化参数（0表示普通最小二乘，推荐值1e3-1e6）')
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
    print("ALS 回归分析")
    print("=" * 60)
    print(f"\n输入:")
    print(f"  缓存目录: {args.cachedir}")
    print(f"  街区数据: {args.districts}")
    if args.voronoi:
        print(f"  Voronoi图: {args.voronoi}")
    print(f"\n输出前缀: {args.output}")
    if X_F_columns:
        print(f"  人为热特征（连续）: {X_F_columns}")
    if X_S_columns:
        print(f"  储热特征（连续）: {X_S_columns}")
    if X_C_columns:
        print(f"  分类特征（one-hot编码）: {X_C_columns}")
    if args.spatial_weights:
        print(f"  空间权重矩阵: {args.spatial_weights}")
    if args.ridge_alpha > 0:
        print(f"  岭回归正则化参数: {args.ridge_alpha:.2e}")

    try:
        run_als_regression(
            cache_dir=args.cachedir,
            districts_path=args.districts,
            output_prefix=args.output,
            voronoi_path=args.voronoi,
            district_id_field=args.district_id,
            X_F_columns=X_F_columns,
            X_S_columns=X_S_columns,
            X_C_columns=X_C_columns,
            spatial_weights_path=args.spatial_weights,
            max_iter=args.max_iter,
            tol=args.tol,
            ridge_alpha=args.ridge_alpha,
            verbose=not args.quiet
        )
        
        print("\n" + "=" * 60)
        print("✓ ALS 回归分析完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

