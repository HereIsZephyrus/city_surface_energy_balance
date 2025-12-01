#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
建筑群迎风面积可视化脚本

从真实建筑数据中随机选择一个建筑群，计算并可视化其迎风面积。

使用方法:
    python script/plot_building_example.py <building.gpkg> [options]

示例:
    # 随机选择一个建筑群
    python script/plot_building_example.py data/buildings.gpkg
    
    # 指定街区和建筑群
    python script/plot_building_example.py data/buildings.gpkg --district 5 --cluster 2
    
    # 保存图片
    python script/plot_building_example.py data/buildings.gpkg -o output.png
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Unifont"

from src.landscape import (
    BUILDING_FIELDS,
    calculate_cluster_roughness,
    plot_frontal_area_profile,
)


def load_and_filter_buildings(
    building_path: str,
    height_field: str = 'height',
    min_proj_field: str = 'min_proj',
    max_proj_field: str = 'max_proj',
    footprint_field: str = 'area',
    block_area_field: str = 'voroniarea'
) -> gpd.GeoDataFrame:
    """
    加载并过滤建筑数据
    
    返回:
        有效的建筑 GeoDataFrame
    """
    buildings = gpd.read_file(building_path)
    print(f"原始建筑数量: {len(buildings)}")
    
    # 过滤无效建筑
    valid_mask = (
        (buildings[height_field] > 0) &
        (buildings[max_proj_field] > buildings[min_proj_field]) &
        (buildings[footprint_field] > 0) &
        (buildings[block_area_field] > 0) &
        buildings[height_field].notna() &
        buildings[min_proj_field].notna() &
        buildings[max_proj_field].notna()
    )
    buildings = buildings[valid_mask].copy()
    print(f"有效建筑数量: {len(buildings)}")
    
    return buildings


def get_random_cluster(
    buildings: gpd.GeoDataFrame,
    district_field: str = 'district_id',
    cluster_field: str = 'spectral_cluster',
    min_buildings: int = 3
) -> tuple:
    """
    随机选择一个建筑群（至少包含 min_buildings 栋建筑）
    
    返回:
        (district_id, cluster_id, group_dataframe)
    """
    grouped = buildings.groupby([district_field, cluster_field])
    
    # 过滤掉建筑数量太少的群
    valid_groups = [(key, group) for key, group in grouped if len(group) >= min_buildings]
    
    if not valid_groups:
        raise ValueError(f"没有找到至少包含 {min_buildings} 栋建筑的建筑群")
    
    # 随机选择
    idx = np.random.randint(0, len(valid_groups))
    (district_id, cluster_id), group = valid_groups[idx]
    
    return district_id, cluster_id, group


def plot_cluster_with_stats(
    group: gpd.GeoDataFrame,
    district_id,
    cluster_id,
    height_field: str = 'height',
    footprint_field: str = 'area',
    block_area_field: str = 'voroniarea',
    min_proj_field: str = 'min_proj',
    max_proj_field: str = 'max_proj',
    save_path: str = None
):
    """
    计算建筑群粗糙度参数并绘制迎风面积剖面图
    """
    # 计算粗糙度参数
    result = calculate_cluster_roughness(
        group=group,
        height_field=height_field,
        footprint_field=footprint_field,
        block_area_field=block_area_field,
        min_proj_field=min_proj_field,
        max_proj_field=max_proj_field,
        district_id=district_id,
        cluster_id=cluster_id
    )
    
    if result is None:
        print("错误: 无法计算该建筑群的粗糙度参数")
        return None
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print(f"建筑群信息: 街区 {district_id}, 聚类 {cluster_id}")
    print("=" * 60)
    print(f"  建筑数量:       {result.n_buildings}")
    print(f"  加权平均高度:   {result.h_weighted:.2f} m")
    print(f"  总平面面积:     {result.total_footprint:.2f} m²")
    print(f"  总维诺图面积:   {result.total_block_area:.2f} m²")
    print(f"  有效迎风面积:   {result.effective_frontal_area:.2f} m²")
    print(f"  λ_p (平面密度): {result.lambda_p:.4f}")
    print(f"  λ_F (迎风密度): {result.lambda_F:.4f}")
    print(f"  z_d (位移高度): {result.z_d:.2f} m")
    print(f"  z_0 (粗糙度):   {result.z_0:.4f} m")
    print("=" * 60)
    
    # 绘制剖面图
    title = f"街区 {district_id} - 聚类 {cluster_id}\n" \
            f"n={result.n_buildings}, h̄={result.h_weighted:.1f}m, " \
            f"λ_F={result.lambda_F:.3f}, z_0={result.z_0:.3f}m"
    
    fig = plot_frontal_area_profile(
        minp=result.min_projs,
        maxp=result.max_projs,
        height=result.heights,
        title=title,
        save_path=save_path
    )
    
    return fig, result


def main():
    parser = argparse.ArgumentParser(
        description='建筑群迎风面积可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 随机选择一个建筑群
    python script/plot_building_example.py data/buildings.gpkg
    
    # 指定街区和建筑群
    python script/plot_building_example.py data/buildings.gpkg --district 5 --cluster 2
    
    # 保存图片
    python script/plot_building_example.py data/buildings.gpkg -o output.png

字段说明:
    height:           建筑高度 (m)
    area:             建筑平面面积 (m²)
    voroniarea:       维诺图面积 (m²)
    min_proj:         沿风向投影最小点 (m)
    max_proj:         沿风向投影最大点 (m)
    district_id:      街区ID
    spectral_cluster: 建筑群聚类标签
        """
    )
    
    parser.add_argument('building_path', help='建筑数据文件路径 (.gpkg)')
    parser.add_argument('--district', type=int, default=None, help='指定街区ID')
    parser.add_argument('--cluster', type=int, default=None, help='指定建筑群ID')
    parser.add_argument('-o', '--output', type=str, default=None, help='输出图片路径')
    parser.add_argument('--min-buildings', type=int, default=3, help='最少建筑数量 (默认: 3)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--no-show', action='store_true', help='不显示图片（仅保存）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.building_path).exists():
        print(f"错误: 文件不存在 - {args.building_path}")
        sys.exit(1)
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # 字段名配置
    height_field = BUILDING_FIELDS['height']
    footprint_field = BUILDING_FIELDS['footprint']
    block_area_field = BUILDING_FIELDS['block_area']
    min_proj_field = BUILDING_FIELDS['min_proj']
    max_proj_field = BUILDING_FIELDS['max_proj']
    district_field = BUILDING_FIELDS['district_id']
    cluster_field = BUILDING_FIELDS['cluster']
    
    # 加载数据
    print(f"\n加载建筑数据: {args.building_path}")
    buildings = load_and_filter_buildings(
        args.building_path,
        height_field=height_field,
        min_proj_field=min_proj_field,
        max_proj_field=max_proj_field,
        footprint_field=footprint_field,
        block_area_field=block_area_field
    )
    
    # 选择建筑群
    if args.district is not None and args.cluster is not None:
        # 使用指定的街区和建筑群
        mask = (buildings[district_field] == args.district) & \
               (buildings[cluster_field] == args.cluster)
        group = buildings[mask]
        
        if len(group) == 0:
            print(f"错误: 未找到街区 {args.district} 建筑群 {args.cluster}")
            sys.exit(1)
        
        district_id = args.district
        cluster_id = args.cluster
        print(f"\n使用指定的建筑群: 街区 {district_id}, 聚类 {cluster_id}")
    else:
        # 随机选择
        print(f"\n随机选择建筑群 (最少 {args.min_buildings} 栋建筑)...")
        district_id, cluster_id, group = get_random_cluster(
            buildings,
            district_field=district_field,
            cluster_field=cluster_field,
            min_buildings=args.min_buildings
        )
    
    # 绘制
    result = plot_cluster_with_stats(
        group=group,
        district_id=district_id,
        cluster_id=cluster_id,
        height_field=height_field,
        footprint_field=footprint_field,
        block_area_field=block_area_field,
        min_proj_field=min_proj_field,
        max_proj_field=max_proj_field,
        save_path=args.output
    )
    
    if result is None:
        sys.exit(1)
    
    # 显示
    if not args.no_show:
        plt.show()
    
    print("\n完成!")


if __name__ == '__main__':
    main()

