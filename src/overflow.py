"""
溢出效应分析命令行接口

承接 spatial 模块的输出，分析自然景观对建成区的降温溢出效应。

使用方法:
    python -m src overflow --input spatial_result.gpkg -o overflow_result.gpkg

输出:
    - GeoPackage 文件：包含溢出分析结果的街区数据
    - CSV 文件：溢出效应统计摘要
"""

from pathlib import Path
import json

import geopandas as gpd
import pandas as pd

from .analysis import (
    OverflowResult,
    analyze_overflow,
    LCZ_NAMES
)


def main(args):
    """
    溢出效应分析主函数
    
    参数:
        args: 命令行参数对象，包含:
            - input: 输入文件路径（spatial 模块输出）
            - output: 输出文件路径
            - ta_column: 气温列名
            - lcz_column: LCZ 列名
            - distance_threshold: 空间权重距离阈值
            - decay_function: 距离衰减函数
            - cache_dir: 缓存目录
            - natural_lcz: 自然景观 LCZ 值
            - quiet: 静默模式
    """
    verbose = not getattr(args, 'quiet', False)
    
    if verbose:
        print("\n" + "=" * 70)
        print("溢出效应分析模块")
        print("=" * 70)
    
    # 1. 加载输入数据
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    if verbose:
        print(f"\n加载数据: {input_path}")
    
    gdf = gpd.read_file(input_path)
    
    if verbose:
        print(f"  街区数量: {len(gdf)}")
    
    # 2. 解析参数
    ta_column = getattr(args, 'ta_column', 'Ta_optimized')
    lcz_column = getattr(args, 'lcz_column', 'LCZ')
    distance_threshold = getattr(args, 'distance_threshold', 5000.0)
    decay_function = getattr(args, 'distance_decay', 'gaussian')
    cache_dir = getattr(args, 'cache_dir', None)
    
    # 解析自然景观 LCZ 值
    natural_lcz_str = getattr(args, 'natural_lcz', None)
    natural_lcz_values = None
    if natural_lcz_str:
        natural_lcz_values = [int(x.strip()) for x in natural_lcz_str.split(',')]
    
    # 检查必需列
    if ta_column not in gdf.columns:
        raise ValueError(f"气温列 '{ta_column}' 不存在于输入文件中")
    if lcz_column not in gdf.columns:
        raise ValueError(f"LCZ 列 '{lcz_column}' 不存在于输入文件中")
    
    # 3. 执行溢出效应分析
    result, gdf_with_results = analyze_overflow(
        gdf=gdf,
        ta_column=ta_column,
        lcz_column=lcz_column,
        distance_threshold=distance_threshold,
        decay_function=decay_function,
        cache_dir=cache_dir,
        natural_lcz_values=natural_lcz_values,
        verbose=verbose
    )
    
    # 4. 保存结果
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 GeoPackage
    if output_path.suffix.lower() in ['.gpkg', '.geojson', '.shp']:
        gdf_with_results.to_file(output_path, driver='GPKG')
        if verbose:
            print(f"\n✓ 街区结果已保存: {output_path}")
    
    # 保存统计摘要 CSV
    summary_path = output_path.with_suffix('.csv')
    summary_df = _create_summary_dataframe(result)
    summary_df.to_csv(summary_path, index=False)
    if verbose:
        print(f"✓ 统计摘要已保存: {summary_path}")
    
    # 保存详细结果 JSON
    json_path = output_path.with_name(output_path.stem + '_stats.json')
    stats_dict = _create_stats_dict(result)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"✓ 详细统计已保存: {json_path}")
    
    if verbose:
        print("\n" + "=" * 70)
        print("✓ 溢出效应分析完成!")
        print("=" * 70)
    
    return result, gdf_with_results


def _create_summary_dataframe(result: OverflowResult) -> pd.DataFrame:
    """创建统计摘要 DataFrame"""
    rows = [
        {'指标': '总街区数', '值': result.n_total, '单位': '个'},
        {'指标': '自然景观街区数', '值': result.n_natural, '单位': '个'},
        {'指标': '建成区街区数', '值': result.n_built, '单位': '个'},
        {'指标': '自然景观占比', '值': f"{100*result.n_natural/result.n_total:.1f}", '单位': '%'},
        {'指标': '平均自然邻居占比', '值': f"{result.natural_neighbor_ratio.mean()*100:.1f}", '单位': '%'},
        {'指标': '溢出系数', '值': f"{result.spillover_coeff:.4f}", '单位': 'K/K'},
        {'指标': '溢出回归R²', '值': f"{result.spillover_r2:.4f}", '单位': '-'},
        {'指标': '溢出系数p值', '值': f"{result.spillover_pvalue:.4e}", '单位': '-'},
        {'指标': '高自然邻居建成区平均气温', '值': f"{result.mean_Ta_high_natural-273.15:.2f}", '单位': '°C'},
        {'指标': '低自然邻居建成区平均气温', '值': f"{result.mean_Ta_low_natural-273.15:.2f}", '单位': '°C'},
        {'指标': '降温效应', '值': f"{result.cooling_effect:.2f}", '单位': 'K'},
    ]
    
    # 添加聚集类型统计
    for ctype in ['Cold-Cold', 'Hot-Hot', 'Low-High', 'High-Low']:
        count = (result.cluster_type == ctype).sum()
        rows.append({
            '指标': f'LISA {ctype} 街区数',
            '值': count,
            '单位': '个'
        })
    
    return pd.DataFrame(rows)


def _create_stats_dict(result: OverflowResult) -> dict:
    """创建详细统计字典"""
    stats = {
        'basic': {
            'n_total': int(result.n_total),
            'n_natural': int(result.n_natural),
            'n_built': int(result.n_built),
            'natural_ratio': float(result.n_natural / result.n_total)
        },
        'natural_neighbor': {
            'mean_ratio': float(result.natural_neighbor_ratio.mean()),
            'max_ratio': float(result.natural_neighbor_ratio.max()),
            'min_ratio': float(result.natural_neighbor_ratio.min()),
            'std_ratio': float(result.natural_neighbor_ratio.std())
        },
        'spillover_regression': {
            'coefficient': float(result.spillover_coeff) if not pd.isna(result.spillover_coeff) else None,
            'intercept': float(result.spillover_intercept) if not pd.isna(result.spillover_intercept) else None,
            'r_squared': float(result.spillover_r2) if not pd.isna(result.spillover_r2) else None,
            'p_value': float(result.spillover_pvalue) if not pd.isna(result.spillover_pvalue) else None,
            'significant': bool(result.spillover_pvalue < 0.05) if not pd.isna(result.spillover_pvalue) else None
        },
        'group_comparison': {
            'mean_Ta_high_natural_K': float(result.mean_Ta_high_natural) if not pd.isna(result.mean_Ta_high_natural) else None,
            'mean_Ta_low_natural_K': float(result.mean_Ta_low_natural) if not pd.isna(result.mean_Ta_low_natural) else None,
            'cooling_effect_K': float(result.cooling_effect) if not pd.isna(result.cooling_effect) else None
        },
        'lisa_clusters': {
            ctype: int((result.cluster_type == ctype).sum())
            for ctype in ['Cold-Cold', 'Hot-Hot', 'Low-High', 'High-Low', '']
        },
        'spillover_by_lcz': {}
    }
    
    # 按 LCZ 类型的统计
    for lcz_code, lcz_stats in result.spillover_by_lcz.items():
        lcz_name = LCZ_NAMES.get(lcz_code, f'LCZ {lcz_code}')
        stats['spillover_by_lcz'][f'LCZ_{lcz_code}_{lcz_name}'] = {
            'count': int(lcz_stats['count']),
            'mean_Ta_K': float(lcz_stats['mean_Ta']) if not pd.isna(lcz_stats['mean_Ta']) else None,
            'mean_Ta_C': float(lcz_stats['mean_Ta'] - 273.15) if not pd.isna(lcz_stats['mean_Ta']) else None,
            'spillover_coeff': float(lcz_stats['spillover_coeff']) if not pd.isna(lcz_stats['spillover_coeff']) else None,
            'n_affected_built': int(lcz_stats.get('n_affected_built', 0))
        }
    
    return stats

