"""
空间滞后模型模块 - 城市地表能量平衡

分析水平交换项 ΔQ_A（气温空间自相关），基于 ALS 回归求解的街区气温。

工作流程:
    1. 加载 ALS 回归输出的街区结果
    2. 构建空间权重矩阵
    3. 空间滞后模型分析
    4. 结果输出

物理背景:
    水平交换项 ΔQ_A 描述相邻街区之间因气温差异导致的热量交换。
    
空间滞后模型:
    Ta = μ + ρ·W·Ta + ε
    
    其中:
    - Ta: 各街区气温
    - W: 空间权重矩阵（行标准化）
    - ρ: 空间自相关系数
    - μ: 截距
    - ε: 残差

使用方法:
    python -m src spatial --input <als_result.gpkg> -o <output.gpkg>
"""

from __future__ import annotations

import sys
from pathlib import Path
import traceback
from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd

from .regression.spatial_analysis import SpatialWeightMatrix, analyze_spatial_autocorrelation


def run_spatial_analysis(
    input_path: str,
    output_path: str,
    ta_column: str = 'Ta_optimized',
    district_id_field: str = 'district_id',
    distance_threshold: float = 500.0,
    distance_decay: str = 'binary',
    verbose: bool = True
) -> gpd.GeoDataFrame:
    """
    执行空间滞后模型分析
    
    参数:
        input_path: ALS 回归输出文件路径 (.gpkg)
        output_path: 输出文件路径 (.gpkg 或 .csv)
        ta_column: 气温列名
        district_id_field: 街区ID字段名
        distance_threshold: 空间权重距离阈值（米）
        distance_decay: 距离衰减函数 ('binary', 'linear', 'inverse', 'gaussian')
        verbose: 是否打印详细信息
        
    返回:
        包含空间分析结果的 GeoDataFrame
    """
    print("\n" + "=" * 60)
    print("空间滞后模型分析（水平交换项 ΔQ_A）")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1] 加载 ALS 回归结果...")
    input_gdf = gpd.read_file(input_path)
    print(f"  街区数量: {len(input_gdf)}")
    print(f"  可用字段: {list(input_gdf.columns)}")
    
    # 检查必需列
    if ta_column not in input_gdf.columns:
        raise ValueError(f"缺少气温列: {ta_column}")
    
    # 2. 构建空间权重矩阵
    print("\n[2] 构建空间权重矩阵...")
    spatial_weights = SpatialWeightMatrix(
        districts_gdf=input_gdf,
        distance_threshold=distance_threshold,
        decay_function=distance_decay,
        row_standardize=True
    )
    
    # 3. 空间滞后模型分析
    print("\n[3] 空间滞后模型分析...")
    Ta = input_gdf[ta_column].values
    
    spatial_results = analyze_spatial_autocorrelation(
        Ta=Ta,
        spatial_weights=spatial_weights,
        verbose=verbose
    )
    
    # 4. 整理结果
    print("\n[4] 整理结果...")
    output_gdf = input_gdf.copy()
    
    if spatial_results is not None:
        # 添加空间分析结果列
        output_gdf['spatial_rho'] = spatial_results['rho']
        output_gdf['spatial_r_squared'] = spatial_results['r_squared']
        output_gdf['spatial_moran_i'] = spatial_results['moran_i']
        output_gdf['spatial_moran_p'] = spatial_results['moran_p']
        output_gdf['spatial_intercept'] = spatial_results['intercept']
        
        if 'spatial_lag' in spatial_results:
            output_gdf['spatial_lag_Ta'] = spatial_results['spatial_lag']
        
        if 'residuals' in spatial_results:
            output_gdf['spatial_residual'] = spatial_results['residuals']
    
    # 5. 保存结果
    print(f"\n[5] 保存结果: {output_path}")
    output_path = Path(output_path)
    
    if output_path.suffix == '.gpkg':
        output_gdf.to_file(output_path, driver='GPKG')
    elif output_path.suffix == '.csv':
        output_gdf.drop(columns=['geometry'], errors='ignore').to_csv(output_path, index=False)
    elif output_path.suffix == '.parquet':
        output_gdf.to_parquet(output_path)
    else:
        output_gdf.to_file(str(output_path) + '.gpkg', driver='GPKG')
    
    # 保存空间分析摘要到 CSV
    summary_output = str(output_path).replace(output_path.suffix, '_spatial_summary.csv')
    summary_df = pd.DataFrame({
        '指标': [
            '空间自相关系数 (ρ)',
            '截距 (μ)',
            'R² 决定系数',
            "Moran's I",
            "Moran's I 期望值",
            "Moran's I Z统计量",
            "Moran's I p值",
            '有效样本数',
            '距离阈值 (m)',
            '衰减函数'
        ],
        '值': [
            spatial_results['rho'] if spatial_results else np.nan,
            spatial_results['intercept'] if spatial_results else np.nan,
            spatial_results['r_squared'] if spatial_results else np.nan,
            spatial_results['moran_i'] if spatial_results else np.nan,
            spatial_results['moran_expected'] if spatial_results else np.nan,
            spatial_results['moran_z'] if spatial_results else np.nan,
            spatial_results['moran_p'] if spatial_results else np.nan,
            spatial_results['n_valid_samples'] if spatial_results else np.nan,
            distance_threshold,
            distance_decay
        ]
    })
    summary_df.to_csv(summary_output, index=False)
    print(f"  空间分析摘要: {summary_output}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("空间分析结果摘要")
    print("=" * 60)
    
    if spatial_results is not None:
        print(f"\n空间自相关检验 (Moran's I):")
        print(f"  Moran's I = {spatial_results['moran_i']:.4f}")
        print(f"  期望值 E[I] = {spatial_results['moran_expected']:.4f}")
        print(f"  Z 统计量 = {spatial_results['moran_z']:.2f}")
        print(f"  p 值 = {spatial_results['moran_p']:.4f}")
        
        if spatial_results['moran_p'] < 0.05:
            print("  结论: 存在显著的空间自相关 (p < 0.05)")
        else:
            print("  结论: 空间自相关不显著 (p >= 0.05)")
        
        print(f"\n空间滞后模型 (Ta = μ + ρ·W·Ta + ε):")
        print(f"  截距 μ = {spatial_results['intercept']:.2f} K ({spatial_results['intercept']-273.15:.2f}°C)")
        print(f"  空间自相关系数 ρ = {spatial_results['rho']:.4f}")
        print(f"  R² = {spatial_results['r_squared']:.4f} (方差解释比例: {spatial_results['r_squared']*100:.1f}%)")
        
        if abs(spatial_results['rho']) > 0.1:
            direction = "正" if spatial_results['rho'] > 0 else "负"
            print(f"\n物理解释:")
            print(f"  ρ > 0 表示相邻街区气温相似（热量扩散/均质化）")
            print(f"  当前 ρ = {spatial_results['rho']:.4f}，存在{direction}向空间自相关")
            print(f"  即相邻街区每 1K 温差，本街区温度变化约 {abs(spatial_results['rho']):.2f}K")
    else:
        print("\n  警告: 空间分析未能完成")
    
    return output_gdf


def main(args=None):
    """
    空间滞后模型分析主函数
    """
    import argparse
    
    if args is None:
        parser = argparse.ArgumentParser(
            description='空间滞后模型分析 - 水平交换项 ΔQ_A',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
    python -m src spatial --input als_result.gpkg -o spatial_result.gpkg
    
空间滞后模型:
    Ta = μ + ρ·W·Ta + ε
    
    其中:
    - Ta: 各街区气温
    - W: 空间权重矩阵（行标准化）
    - ρ: 空间自相关系数（水平交换项强度）
    - μ: 截距
    - ε: 残差

输出文件:
    <output>.gpkg                 - 包含空间分析结果的街区数据
    <output>_spatial_summary.csv  - 空间分析摘要
            """
        )
        parser.add_argument('--input', required=True, 
                            help='ALS 回归输出文件路径 (.gpkg)')
        parser.add_argument('-o', '--output', required=True, 
                            help='输出文件路径')
        parser.add_argument('--ta-column', default='Ta_optimized',
                            help='气温列名 (默认: Ta_optimized)')
        parser.add_argument('--district-id', default='district_id', 
                            help='街区ID字段名')
        parser.add_argument('--distance-threshold', type=float, default=500.0,
                            help='空间权重距离阈值(m)')
        parser.add_argument('--distance-decay', type=str, default='binary',
                            choices=['binary', 'linear', 'inverse', 'gaussian'],
                            help='距离衰减函数')
        parser.add_argument('--quiet', action='store_true',
                            help='静默模式')
        args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在 - {args.input}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("空间滞后模型分析（水平交换项 ΔQ_A）")
    print("=" * 60)
    print(f"\n输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"\n空间分析参数:")
    print(f"  气温列: {args.ta_column}")
    print(f"  距离阈值: {args.distance_threshold} m")
    print(f"  衰减函数: {args.distance_decay}")

    try:
        run_spatial_analysis(
            input_path=args.input,
            output_path=args.output,
            ta_column=args.ta_column,
            district_id_field=args.district_id,
            distance_threshold=args.distance_threshold,
            distance_decay=args.distance_decay,
            verbose=not args.quiet
        )
        
        print("\n" + "=" * 60)
        print("✓ 空间滞后模型分析完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

