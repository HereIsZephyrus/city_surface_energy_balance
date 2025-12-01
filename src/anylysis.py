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

注意:
    本模块为组合工作流，实际计算由以下独立模块执行：
    - als.py: ALS 回归（输出 Ta 栅格和系数）
    - spatial.py: 空间滞后模型分析
"""

from __future__ import annotations

import sys
from pathlib import Path
import traceback
from typing import Optional, List
import geopandas as gpd

from .als import run_als_regression
from .spatial import run_spatial_analysis


def run_regression(
    cache_dir: str,
    districts_path: str,
    output_path: str,
    voronoi_path: Optional[str] = None,
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
    执行完整的街区级回归分析（ALS + 空间分析）
    
    参数:
        cache_dir: 缓存目录路径（physics模块的输出）
        districts_path: 街区矢量数据路径 (.gpkg)
        output_path: 输出文件路径 (.gpkg 或 .csv)
        voronoi_path: Voronoi 图路径（用于栅格化，可选）
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
    print("街区级回归分析（ALS + 空间分析）")
    print("=" * 60)
    
    # 第一步: ALS 回归
    output_path_obj = Path(output_path)
    als_output_prefix = str(output_path_obj.with_suffix(''))
    
    print("\n>>> 阶段 1/2: ALS 回归 <<<")
    als_results = run_als_regression(
        cache_dir=cache_dir,
        districts_path=districts_path,
        output_prefix=als_output_prefix,
        voronoi_path=voronoi_path,
        district_id_field=district_id_field,
        X_F_columns=X_F_columns,
        X_S_columns=X_S_columns,
        X_C_columns=X_C_columns,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose
    )
    
    # 第二步: 空间滞后模型分析
    als_result_path = als_output_prefix + '.gpkg'
    
    print("\n>>> 阶段 2/2: 空间滞后模型分析 <<<")
    output_gdf = run_spatial_analysis(
        input_path=als_result_path,
        output_path=output_path,
        ta_column='Ta_optimized',
        district_id_field=district_id_field,
        distance_threshold=distance_threshold,
        distance_decay=distance_decay,
        verbose=verbose
    )
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("回归结果摘要")
    print("=" * 60)
    print(f"  收敛: {'是' if als_results['converged'] else '否'}")
    print(f"  迭代次数: {als_results['n_iter']}")
    print(f"  残差范数: {als_results['residual_norm']:.4f}")
    Ta = als_results['Ta_per_district']
    print(f"  气温范围: {Ta.min()-273.15:.2f}°C ~ {Ta.max()-273.15:.2f}°C")
    print(f"  气温均值: {Ta.mean()-273.15:.2f}°C (±{Ta.std():.2f}K)")
    
    print("\n输出文件:")
    print(f"  街区结果: {als_result_path}")
    print(f"  Ta 栅格: {als_output_prefix}_Ta.tif")
    print(f"  系数: {als_output_prefix}_coefficients.csv")
    print(f"  空间分析: {output_path}")
    
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
    python -m src.anylysis --cachedir ./cache --districts districts.gpkg -o result.gpkg
    
物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si = 0
    
    水平交换项 ΔQ_A（空间滞后模型）:
    Ta = μ + ρ·W·Ta + ε

输出文件:
    <output>.gpkg                 - 包含空间分析结果的街区数据
    <output>_Ta.tif               - Ta 栅格
    <output>_coefficients.csv     - 回归系数
    <output>_spatial_summary.csv  - 空间分析摘要
            """
        )
        parser.add_argument('--cachedir', required=True, 
                            help='缓存目录路径（physics模块输出）')
        parser.add_argument('--districts', required=True, 
                            help='街区矢量数据路径 (.gpkg)')
        parser.add_argument('--voronoi', type=str, default=None,
                            help='Voronoi图路径（用于栅格化，可选）')
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
    if hasattr(args, 'voronoi') and args.voronoi:
        print(f"  Voronoi图: {args.voronoi}")
    print(f"\n输出: {args.output}")
    if X_F_columns:
        print(f"  人为热特征（连续）: {X_F_columns}")
    if X_S_columns:
        print(f"  储热特征（连续）: {X_S_columns}")
    if X_C_columns:
        print(f"  分类特征（one-hot编码）: {X_C_columns}")
    print("  空间分析:")
    print(f"    距离阈值: {args.distance_threshold} m")
    print(f"    衰减函数: {args.distance_decay}")

    try:
        run_regression(
            cache_dir=args.cachedir,
            districts_path=args.districts,
            output_path=args.output,
            voronoi_path=getattr(args, 'voronoi', None),
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
