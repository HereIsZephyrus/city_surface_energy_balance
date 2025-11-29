"""
迎风面积计算模块

使用扫描线法计算建筑群的有效迎风面积（考虑遮挡效应）。

算法原理:
    将每栋建筑视为矩形 [minp, maxp] × [0, height]，使用扫描线法
    计算所有矩形的并集面积，即有效迎风面积。

数据要求:
    - minp: 建筑沿风向投影的起点位置 (m)
    - maxp: 建筑沿风向投影的终点位置 (m)
    - height: 建筑高度 (m)

参考文献:
    Bottema, M., & Mestayer, P. G. (1998). Urban roughness mapping.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure


def calculate_frontal_area_for_cluster(
    minp: np.ndarray,
    maxp: np.ndarray,
    height: np.ndarray
) -> float:
    """
    计算单个建筑群的有效迎风面积（考虑遮挡）
    
    使用扫描线法计算多个矩形的并集面积。每栋建筑被视为
    矩形 [minp, maxp] × [0, height]。
    
    算法步骤:
        1. 收集所有垂直边事件 (x, 类型, height)
        2. 按 x 坐标排序
        3. 从左到右扫描，维护活跃高度集合
        4. 累加每个区间的面积
    
    参数:
        minp: 建筑沿风向起点位置数组 (m)
        maxp: 建筑沿风向终点位置数组 (m)
        height: 建筑高度数组 (m)
    
    返回:
        有效迎风面积 (m²)
    
    示例:
        >>> minp = np.array([0, 5, 8])
        >>> maxp = np.array([10, 15, 12])
        >>> height = np.array([20, 15, 25])
        >>> area = calculate_frontal_area_for_cluster(minp, maxp, height)
    """
    minp = np.asarray(minp, dtype=np.float64)
    maxp = np.asarray(maxp, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    
    if len(minp) == 0:
        return 0.0
    
    # 过滤无效数据
    valid_mask = (height > 0) & (maxp > minp) & np.isfinite(minp) & np.isfinite(maxp) & np.isfinite(height)
    minp = minp[valid_mask]
    maxp = maxp[valid_mask]
    height = height[valid_mask]
    
    if len(minp) == 0:
        return 0.0
    
    # 收集所有事件: (x位置, 事件类型, 高度)
    # 事件类型: 1 = 进入(左边), -1 = 离开(右边)
    events: List[Tuple[float, int, float]] = []
    
    for i in range(len(minp)):
        events.append((minp[i], 1, height[i]))   # 进入事件
        events.append((maxp[i], -1, height[i]))  # 离开事件
    
    # 按 x 坐标排序，相同 x 时进入事件优先
    events.sort(key=lambda e: (e[0], -e[1]))
    
    # 扫描线算法
    total_area = 0.0
    active_heights: List[float] = []  # 当前活跃的高度列表
    prev_x = events[0][0]
    
    for x, event_type, h in events:
        if x > prev_x and len(active_heights) > 0:
            # 计算区间面积
            max_height = max(active_heights)
            width = x - prev_x
            total_area += width * max_height
        
        # 更新活跃高度集合
        if event_type == 1:  # 进入
            active_heights.append(h)
        else:  # 离开
            active_heights.remove(h)
        
        prev_x = x
    
    return total_area


def calculate_frontal_area_by_district(
    buildings_gdf: gpd.GeoDataFrame,
    district_field: str = 'district_id',
    cluster_field: str = 'spectral_cluster',
    minp_field: str = 'min_proj',
    maxp_field: str = 'max_proj',
    height_field: str = 'height'
) -> pd.DataFrame:
    """
    从 GeoDataFrame 按建筑群分组计算有效迎风面积
    
    建筑群定义: district_field 和 cluster_field 都相同的建筑
    
    参数:
        buildings_gdf: 建筑数据 GeoDataFrame
        district_field: 街区ID字段名
        cluster_field: 空间聚类字段名
        minp_field: 沿风向起点位置字段名
        maxp_field: 沿风向终点位置字段名
        height_field: 建筑高度字段名
    
    返回:
        DataFrame 包含:
            - district_id: 街区ID
            - spatial_cluster: 空间聚类ID
            - frontal_area: 有效迎风面积 (m²)
            - building_count: 建筑数量
    
    示例:
        >>> gdf = gpd.read_file('buildings.gpkg')
        >>> result = calculate_frontal_area_by_district(gdf)
    """
    # 验证字段存在
    required_fields = [district_field, cluster_field, minp_field, maxp_field, height_field]
    missing = [f for f in required_fields if f not in buildings_gdf.columns]
    if missing:
        raise ValueError(f"GeoDataFrame 缺少必需字段: {missing}")
    
    # 过滤无效建筑（高度 <= 0 或其他异常值）
    original_count = len(buildings_gdf)
    valid_mask = (
        (buildings_gdf[height_field] > 0) &
        (buildings_gdf[maxp_field] > buildings_gdf[minp_field]) &
        buildings_gdf[height_field].notna() &
        buildings_gdf[minp_field].notna() &
        buildings_gdf[maxp_field].notna()
    )
    buildings_gdf = buildings_gdf[valid_mask].copy()
    filtered_count = original_count - len(buildings_gdf)
    
    if filtered_count > 0:
        print(f"  过滤无效建筑: {filtered_count} 栋 (高度<=0 或位置异常)")
        print(f"  有效建筑数量: {len(buildings_gdf)} 栋")
    
    # 按建筑群分组计算
    results = []
    
    grouped = buildings_gdf.groupby([district_field, cluster_field])
    
    for (district_id, cluster_id), group in grouped:
        minp = group[minp_field].values
        maxp = group[maxp_field].values
        height = group[height_field].values
        
        frontal_area = calculate_frontal_area_for_cluster(minp, maxp, height)
        
        results.append({
            'district_id': district_id,
            'spatial_cluster': cluster_id,
            'frontal_area': frontal_area,
            'building_count': len(group)  # 现在只计算有效建筑
        })
    
    return pd.DataFrame(results)


def get_skyline_profile(
    minp: np.ndarray,
    maxp: np.ndarray,
    height: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算建筑群的天际线轮廓
    
    返回可用于绘图的 (x, y) 坐标序列。
    
    参数:
        minp: 建筑沿风向起点位置数组 (m)
        maxp: 建筑沿风向终点位置数组 (m)
        height: 建筑高度数组 (m)
    
    返回:
        (x_coords, y_coords): 轮廓线的坐标数组
    """
    minp = np.asarray(minp, dtype=np.float64)
    maxp = np.asarray(maxp, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    
    if len(minp) == 0:
        return np.array([]), np.array([])
    
    # 过滤无效数据
    valid_mask = (height > 0) & (maxp > minp) & np.isfinite(minp) & np.isfinite(maxp) & np.isfinite(height)
    minp = minp[valid_mask]
    maxp = maxp[valid_mask]
    height = height[valid_mask]
    
    if len(minp) == 0:
        return np.array([]), np.array([])
    
    # 收集事件
    events: List[Tuple[float, int, float]] = []
    for i in range(len(minp)):
        events.append((minp[i], 1, height[i]))
        events.append((maxp[i], -1, height[i]))
    
    events.sort(key=lambda e: (e[0], -e[1]))
    
    # 生成轮廓点
    x_coords = []
    y_coords = []
    active_heights: List[float] = []
    prev_height = 0.0
    
    for x, event_type, h in events:
        # 更新活跃高度
        if event_type == 1:
            active_heights.append(h)
        else:
            active_heights.remove(h)
        
        current_height = max(active_heights) if active_heights else 0.0
        
        # 如果高度变化，添加轮廓点
        if current_height != prev_height:
            # 先添加水平线段的终点
            x_coords.append(x)
            y_coords.append(prev_height)
            # 再添加垂直变化后的点
            x_coords.append(x)
            y_coords.append(current_height)
            prev_height = current_height
    
    return np.array(x_coords), np.array(y_coords)


def plot_frontal_area_profile(
    minp: np.ndarray,
    maxp: np.ndarray,
    height: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    show_buildings: bool = True,
    show_skyline: bool = True,
    show_area_fill: bool = True
) -> Figure:
    """
    绘制建筑群的迎风面积剖面图
    
    可视化内容:
        - 每栋建筑绘制为半透明矩形（不同颜色）
        - 最终轮廓线（黑色粗线）
        - 填充有效面积区域
        - 标注总面积值
    
    参数:
        minp: 建筑沿风向起点位置数组 (m)
        maxp: 建筑沿风向终点位置数组 (m)
        height: 建筑高度数组 (m)
        title: 图表标题
        save_path: 保存路径（如果提供）
        figsize: 图表尺寸
        show_buildings: 是否显示单栋建筑矩形
        show_skyline: 是否显示天际线轮廓
        show_area_fill: 是否填充有效面积区域
    
    返回:
        matplotlib Figure 对象
    
    示例:
        >>> minp = np.array([0, 5, 8])
        >>> maxp = np.array([10, 15, 12])
        >>> height = np.array([20, 15, 25])
        >>> fig = plot_frontal_area_profile(minp, maxp, height, title="建筑群 A")
        >>> plt.show()
    """
    minp = np.asarray(minp, dtype=np.float64)
    maxp = np.asarray(maxp, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    
    # 过滤无效数据
    valid_mask = (height > 0) & (maxp > minp) & np.isfinite(minp) & np.isfinite(maxp) & np.isfinite(height)
    minp = minp[valid_mask]
    maxp = maxp[valid_mask]
    height = height[valid_mask]
    
    # 计算总面积
    total_area = calculate_frontal_area_for_cluster(minp, maxp, height)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 颜色映射
    cmap = plt.get_cmap('Set3')
    colors = [cmap(i % 12) for i in range(len(minp))]
    
    # 绘制单栋建筑矩形
    if show_buildings and len(minp) > 0:
        for i, (mp, mxp, h) in enumerate(zip(minp, maxp, height)):
            rect = patches.Rectangle(
                (mp, 0), mxp - mp, h,
                linewidth=1,
                edgecolor='gray',
                facecolor=colors[i],
                alpha=0.5,
                label=f'建筑 {i+1}: [{mp:.1f}, {mxp:.1f}] h={h:.1f}m' if len(minp) <= 10 else None
            )
            ax.add_patch(rect)
    
    # 获取天际线轮廓
    skyline_x, skyline_y = get_skyline_profile(minp, maxp, height)
    
    # 填充有效面积区域
    if show_area_fill and len(skyline_x) > 0:
        # 闭合轮廓用于填充
        fill_x = np.concatenate([[skyline_x[0]], skyline_x, [skyline_x[-1]]])
        fill_y = np.concatenate([[0], skyline_y, [0]])
        ax.fill(fill_x, fill_y, color='lightblue', alpha=0.3, label='有效迎风面积')
    
    # 绘制天际线轮廓
    if show_skyline and len(skyline_x) > 0:
        ax.plot(skyline_x, skyline_y, 'k-', linewidth=2.5, label='天际线轮廓')
    
    # 设置图表属性
    if len(minp) > 0:
        x_margin = (maxp.max() - minp.min()) * 0.05
        ax.set_xlim(minp.min() - x_margin, maxp.max() + x_margin)
        ax.set_ylim(0, height.max() * 1.1)
    
    ax.set_xlabel('沿风向位置 (m)', fontsize=12)
    ax.set_ylabel('高度 (m)', fontsize=12)
    
    if title:
        ax.set_title(f'{title}\n有效迎风面积: {total_area:.2f} m²', fontsize=14)
    else:
        ax.set_title(f'建筑群迎风面积剖面\n有效迎风面积: {total_area:.2f} m²', fontsize=14)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 图例
    if len(minp) <= 10 or (show_skyline and show_area_fill):
        ax.legend(loc='upper right', fontsize=9)
    
    # 添加统计信息文本框
    stats_text = (
        f'建筑数量: {len(minp)}\n'
        f'最大高度: {height.max():.1f} m\n'
        f'跨度: {maxp.max() - minp.min():.1f} m'
    ) if len(minp) > 0 else '无有效建筑数据'
    
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    return fig


def plot_district_comparison(
    buildings_gdf: gpd.GeoDataFrame,
    district_field: str = 'district_id',
    cluster_field: str = 'spectral_cluster',
    minp_field: str = 'min_proj',
    maxp_field: str = 'max_proj',
    height_field: str = 'height',
    n_examples: int = 4,
    save_path: Optional[str] = None
) -> Figure:
    """
    绘制多个建筑群的迎风面积对比图
    
    参数:
        buildings_gdf: 建筑数据 GeoDataFrame
        district_field: 街区ID字段名
        cluster_field: 空间聚类字段名
        minp_field: 沿风向起点位置字段名
        maxp_field: 沿风向终点位置字段名
        height_field: 建筑高度字段名
        n_examples: 显示的建筑群数量
        save_path: 保存路径
    
    返回:
        matplotlib Figure 对象
    """
    grouped = buildings_gdf.groupby([district_field, cluster_field])
    groups = list(grouped)[:n_examples]
    
    n_cols = 2
    n_rows = (len(groups) + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if len(groups) == 1 else axes
    
    for idx, ((district_id, cluster_id), group) in enumerate(groups):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        minp = group[minp_field].values
        maxp = group[maxp_field].values
        height = group[height_field].values
        
        # 过滤无效数据
        valid_mask = (height > 0) & (maxp > minp)
        minp = minp[valid_mask]
        maxp = maxp[valid_mask]
        height = height[valid_mask]
        
        if len(minp) == 0:
            ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'街区 {district_id} - 聚类 {cluster_id}')
            continue
        
        # 计算面积
        total_area = calculate_frontal_area_for_cluster(minp, maxp, height)
        
        # 绘制建筑
        cmap = plt.get_cmap('Set3')
        for i, (mp, mxp, h) in enumerate(zip(minp, maxp, height)):
            rect = patches.Rectangle(
                (mp, 0), mxp - mp, h,
                linewidth=1,
                edgecolor='gray',
                facecolor=cmap(i % 12),
                alpha=0.5
            )
            ax.add_patch(rect)
        
        # 绘制轮廓
        skyline_x, skyline_y = get_skyline_profile(minp, maxp, height)
        if len(skyline_x) > 0:
            ax.plot(skyline_x, skyline_y, 'k-', linewidth=2)
        
        x_margin = (maxp.max() - minp.min()) * 0.05
        ax.set_xlim(minp.min() - x_margin, maxp.max() + x_margin)
        ax.set_ylim(0, height.max() * 1.1)
        ax.set_xlabel('沿风向位置 (m)')
        ax.set_ylabel('高度 (m)')
        ax.set_title(f'街区 {district_id} - 聚类 {cluster_id}\n面积: {total_area:.1f} m², 建筑: {len(minp)}')
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # 隐藏多余的子图
    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    return fig


# ============================================================================
# 建筑数据处理
# ============================================================================

# 建筑数据字段名配置（固定字段名）
BUILDING_FIELDS = {
    'height': 'height',               # 建筑高度字段
    'footprint': 'area',              # 建筑平面面积字段
    'block_area': 'voroniarea',       # 维诺图面积字段
    'min_proj': 'min_proj',           # 沿风向投影最小点
    'max_proj': 'max_proj',           # 沿风向投影最大点
    'district_id': 'district_id',      # 街区ID字段
    'cluster': 'spectral_cluster',    # 建筑群聚类标签
}


@dataclass
class ClusterRoughnessResult:
    """建筑群粗糙度计算结果"""
    district_id: Any                  # 街区ID
    cluster_id: Any                   # 建筑群ID
    n_buildings: int                  # 建筑数量
    h_weighted: float                 # 体积加权平均高度 (m)
    total_footprint: float            # 总平面面积 (m²)
    total_block_area: float           # 总维诺图面积 (m²)
    effective_frontal_area: float     # 有效迎风面积 (m²)
    lambda_p: float                   # 平面面积密度
    lambda_F: float                   # 迎风面积密度
    z_d: float                        # 零平面位移高度 (m)
    z_0: float                        # 粗糙度长度 (m)
    min_projs: np.ndarray             # 各建筑投影最小点
    max_projs: np.ndarray             # 各建筑投影最大点
    heights: np.ndarray               # 各建筑高度


def calculate_cluster_roughness(
    group: gpd.GeoDataFrame,
    height_field: str = 'height',
    footprint_field: str = 'area',
    block_area_field: str = 'voroniarea',
    min_proj_field: str = 'min_proj',
    max_proj_field: str = 'max_proj',
    district_id: Any = None,
    cluster_id: Any = None
) -> Optional[ClusterRoughnessResult]:
    """
    计算单个建筑群的粗糙度参数
    
    使用扫描线法计算有效迎风面积（考虑遮挡），然后计算粗糙度参数。
    
    参数:
        group: 建筑群的 GeoDataFrame（已按 district + cluster 分组）
        height_field: 建筑高度字段名
        footprint_field: 建筑平面面积字段名
        block_area_field: 维诺图面积字段名
        min_proj_field: 沿风向投影最小点字段名
        max_proj_field: 沿风向投影最大点字段名
        district_id: 街区ID（用于结果标识）
        cluster_id: 建筑群ID（用于结果标识）
    
    返回:
        ClusterRoughnessResult 或 None（如果数据无效）
    """
    from .roughness import calculate_zero_plane_displacement, calculate_roughness_length
    
    if len(group) == 0:
        return None
    
    # 提取该建筑群的数据
    heights = group[height_field].values.astype(np.float64)
    footprints = group[footprint_field].values.astype(np.float64)
    block_areas = group[block_area_field].values.astype(np.float64)
    min_projs = group[min_proj_field].values.astype(np.float64)
    max_projs = group[max_proj_field].values.astype(np.float64)
    
    # 计算建筑群的聚合统计量
    # 1. 体积加权平均高度
    volumes = footprints * heights
    total_volume = np.sum(volumes)
    if total_volume <= 0:
        return None
    h_weighted = np.sum(volumes * heights) / total_volume
    
    # 2. 总平面面积和总维诺图面积
    total_footprint = np.sum(footprints)
    total_block_area = np.sum(block_areas)
    
    # 3. λ_p = 总建筑面积 / 总维诺图面积
    lambda_p = min(total_footprint / total_block_area, 1.0) if total_block_area > 0 else 0.0
    
    # 4. 使用扫描线法计算有效迎风面积（考虑遮挡）
    effective_frontal_area = calculate_frontal_area_for_cluster(min_projs, max_projs, heights)
    
    # 5. λ_F = 有效迎风面积 / 总维诺图面积
    lambda_F = min(effective_frontal_area / total_block_area, 1.0) if total_block_area > 0 else 0.0
    
    # 如果 λ_F 太小，使用估算值
    if lambda_F < 0.001:
        lambda_F = 0.8 * lambda_p
    
    # 6. 计算粗糙度参数
    z_d = calculate_zero_plane_displacement(h_weighted, lambda_p)
    z_0 = calculate_roughness_length(h_weighted, z_d, lambda_F)
    
    return ClusterRoughnessResult(
        district_id=district_id,
        cluster_id=cluster_id,
        n_buildings=len(group),
        h_weighted=h_weighted,
        total_footprint=total_footprint,
        total_block_area=total_block_area,
        effective_frontal_area=effective_frontal_area,
        lambda_p=lambda_p,
        lambda_F=lambda_F,
        z_d=z_d,
        z_0=z_0,
        min_projs=min_projs,
        max_projs=max_projs,
        heights=heights
    )


def calculate_roughness_from_buildings(
    building_path: str,
    collection,
    height_field: str = 'height',
    footprint_field: str = 'area',
    block_area_field: str = 'voroniarea',
    min_proj_field: str = 'min_proj',
    max_proj_field: str = 'max_proj',
    district_field: str = 'district_id',
    cluster_field: str = 'spectral_cluster'
) -> tuple:
    """
    从建筑 gpkg 数据计算粗糙度长度和零平面位移高度
    
    使用扫描线法按建筑群计算有效迎风面积（考虑遮挡），然后计算粗糙度参数。
    
    基于 Bottema & Mestayer (1998) 方法:
        z_d = h × λ_p^0.6
        z_0 = (h - z_d) × exp(-κ / √(0.5 × C_Dh × λ_F))
    
    参数:
        building_path: 建筑数据文件路径 (.gpkg)
        collection: 栅格集合（用于获取目标范围和分辨率）
        height_field: 建筑高度字段名
        footprint_field: 建筑平面面积字段名
        block_area_field: 维诺图面积字段名
        min_proj_field: 沿风向投影最小点字段名
        max_proj_field: 沿风向投影最大点字段名
        district_field: 街区ID字段名
        cluster_field: 建筑群聚类标签字段名
    
    返回:
        (z_0_raster, z_d_raster, height_raster): 粗糙度长度、零平面位移高度、建筑高度栅格
    """
    from rasterio.transform import rowcol
    
    print("\n从建筑数据计算粗糙度参数（按建筑群聚合，扫描线法）...")
    print(f"  文件: {building_path}")
    print(f"  分组字段: {district_field} + {cluster_field}")
    
    # 读取建筑数据
    buildings = gpd.read_file(building_path)
    print(f"  原始建筑数量: {len(buildings)}")
    
    # 确保坐标系一致
    target_crs = collection.target_crs
    if buildings.crs != target_crs:
        buildings = buildings.to_crs(target_crs)
    
    # 验证必需字段
    required_fields = [height_field, footprint_field, block_area_field, 
                       min_proj_field, max_proj_field, district_field, cluster_field]
    missing = [f for f in required_fields if f not in buildings.columns]
    if missing:
        raise ValueError(f"建筑数据缺少必需字段: {missing}")
    
    # 过滤无效建筑（高度 <= 0 或投影位置异常）
    original_count = len(buildings)
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
    filtered_count = original_count - len(buildings)
    
    if filtered_count > 0:
        print(f"  过滤无效建筑: {filtered_count} 栋")
    print(f"  有效建筑数量: {len(buildings)} 栋")
    
    # 获取目标栅格信息
    ref_info = collection._reference_info
    shape = (ref_info['height'], ref_info['width'])
    transform = ref_info['transform']
    
    # 初始化输出栅格
    z_0_raster = np.full(shape, np.nan, dtype=np.float32)
    z_d_raster = np.full(shape, np.nan, dtype=np.float32)
    height_raster = np.full(shape, np.nan, dtype=np.float32)
    
    # 按建筑群分组计算
    grouped = buildings.groupby([district_field, cluster_field])
    cluster_count = 0
    
    print(f"  建筑群数量: {len(grouped)}")
    
    for (district_id, cluster_id), group in grouped:
        # 使用抽象的计算函数
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
            continue
        
        # 将该建筑群的粗糙度写入所有建筑所在的像元
        centroids = group.geometry.centroid
        for centroid in centroids:
            try:
                row, col = rowcol(transform, centroid.x, centroid.y)
                if 0 <= row < shape[0] and 0 <= col < shape[1]:
                    # 如果该位置已有值，取较大值（更保守）
                    if np.isnan(z_0_raster[row, col]) or result.z_0 > z_0_raster[row, col]:
                        z_0_raster[row, col] = result.z_0
                        z_d_raster[row, col] = result.z_d
                        height_raster[row, col] = result.h_weighted
            except Exception:
                continue
        
        cluster_count += 1
    
    print(f"  处理建筑群: {cluster_count}")
    
    # 对于没有建筑的区域，使用默认值
    nan_mask = np.isnan(z_0_raster)
    z_0_raster[nan_mask] = 0.1  # 默认粗糙度 0.1m
    z_d_raster[nan_mask] = 0.0  # 默认位移高度 0m
    height_raster[nan_mask] = 0.0
    
    # 限制在合理范围
    z_0_raster = np.clip(z_0_raster, 0.0001, 5.0)
    z_d_raster = np.clip(z_d_raster, 0.0, 50.0)
    
    print(f"  z_0 范围: {np.nanmin(z_0_raster):.4f} - {np.nanmax(z_0_raster):.4f} m")
    print(f"  z_d 范围: {np.nanmin(z_d_raster):.2f} - {np.nanmax(z_d_raster):.2f} m")
    
    return z_0_raster, z_d_raster, height_raster
