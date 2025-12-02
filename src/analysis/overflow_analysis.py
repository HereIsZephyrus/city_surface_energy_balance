"""
溢出效应分析模块（Overflow/Spillover Analysis）

分析自然景观对相邻建成区的降温溢出效应。

物理背景:
    自然景观（绿地、水体）通过蒸散发、遮阴等机制降低自身温度，
    同时通过热量水平交换影响相邻街区的气温。
    本模块量化这种"降温溢出效应"的强度和空间范围。

主要分析:
    1. 自然邻居占比分析：计算每个街区的自然景观邻居比例
    2. 溢出效应回归：量化自然邻居对建成区气温的影响系数
    3. 局部 Moran's I (LISA)：识别冷点/热点的空间聚集模式
    4. 按景观类型分解：分别分析树木、水体等的溢出效应

参考文献:
    Anselin, L. (1995). Local Indicators of Spatial Association—LISA.
    Geographical Analysis, 27(2), 93-115.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from scipy.stats import norm
from scipy.sparse import csr_matrix

from .constants import LCZ_NATURAL_DEFAULT, LCZ_BUILT_DEFAULT, LCZ_NAMES
from .spatial_analysis import SpatialWeightMatrix


@dataclass
class OverflowResult:
    """溢出效应分析结果"""
    
    # 基本统计
    n_total: int                          # 总街区数
    n_natural: int                        # 自然景观街区数
    n_built: int                          # 建成区街区数
    
    # 自然邻居占比
    natural_neighbor_ratio: np.ndarray    # 每个街区的自然邻居占比 [0, 1]
    
    # 溢出效应回归结果
    spillover_coeff: float                # 溢出系数
    spillover_r2: float                   # 溢出回归 R²
    spillover_pvalue: float               # 溢出系数的 p 值
    spillover_intercept: float            # 溢出回归截距
    
    # 局部 Moran's I
    local_moran_i: np.ndarray             # 局部 Moran's I 值
    cluster_type: np.ndarray              # 聚集类型
    
    # 分组统计
    mean_Ta_high_natural: float           # 高自然邻居占比街区平均气温 (K)
    mean_Ta_low_natural: float            # 低自然邻居占比街区平均气温 (K)
    cooling_effect: float                 # 降温效应 (K)
    
    # 按 LCZ 类型的溢出效应
    spillover_by_lcz: Dict[int, Dict] = field(default_factory=dict)
    
    # 原始数据引用
    Ta: Optional[np.ndarray] = None
    lcz: Optional[np.ndarray] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return pd.DataFrame({
            'natural_neighbor_ratio': self.natural_neighbor_ratio,
            'local_moran_i': self.local_moran_i,
            'cluster_type': self.cluster_type,
        })
    
    def summary(self) -> str:
        """生成摘要报告"""
        lines = [
            "",
            "=" * 70,
            "溢出效应分析结果摘要",
            "=" * 70,
            "",
            "【1. 基本统计】",
            f"  总街区数: {self.n_total}",
            f"  自然景观街区: {self.n_natural} ({100*self.n_natural/self.n_total:.1f}%)",
            f"  建成区街区: {self.n_built} ({100*self.n_built/self.n_total:.1f}%)",
            "",
            "【2. 自然邻居占比分布】",
            f"  平均自然邻居占比: {self.natural_neighbor_ratio.mean()*100:.1f}%",
            f"  最大自然邻居占比: {self.natural_neighbor_ratio.max()*100:.1f}%",
            f"  自然邻居占比 > 30% 的街区: {(self.natural_neighbor_ratio > 0.3).sum()}",
            "",
            "【3. 溢出效应回归】",
            f"  模型: Ta_built = α + β × Ta_natural_neighbors + ε",
            f"  溢出系数 β = {self.spillover_coeff:.4f}",
            f"  截距 α = {self.spillover_intercept:.2f} K ({self.spillover_intercept-273.15:.2f}°C)",
            f"  R² = {self.spillover_r2:.4f}",
            f"  p 值 = {self.spillover_pvalue:.4e}",
            "",
            "  物理解释:",
            f"    自然邻居平均气温每降低 1K，建成区街区气温降低 {self.spillover_coeff:.3f} K",
        ]
        
        if self.spillover_pvalue < 0.05:
            lines.append(f"    溢出效应显著 (p < 0.05)")
        else:
            lines.append(f"    溢出效应不显著 (p >= 0.05)")
        
        lines.extend([
            "",
            "【4. 分组比较】",
            f"  高自然邻居占比 (>30%) 建成区平均气温: {self.mean_Ta_high_natural-273.15:.2f}°C",
            f"  低自然邻居占比 (<10%) 建成区平均气温: {self.mean_Ta_low_natural-273.15:.2f}°C",
            f"  降温效应: {self.cooling_effect:.2f} K",
            "",
            "【5. 局部空间聚集 (LISA)】",
        ])
        
        for ctype in ['Cold-Cold', 'Hot-Hot', 'Low-High', 'High-Low']:
            count = (self.cluster_type == ctype).sum()
            pct = 100 * count / self.n_total
            lines.append(f"  {ctype}: {count} ({pct:.1f}%)")
        
        if self.spillover_by_lcz:
            lines.extend([
                "",
                "【6. 按自然景观类型的溢出效应】",
            ])
            for lcz_code, stats in self.spillover_by_lcz.items():
                lcz_name = LCZ_NAMES.get(lcz_code, f'LCZ {lcz_code}')
                lines.append(f"  {lcz_name} (LCZ {lcz_code}):")
                lines.append(f"    街区数: {stats['count']}")
                lines.append(f"    平均气温: {stats['mean_Ta']-273.15:.2f}°C")
                lines.append(f"    溢出系数: {stats.get('spillover_coeff', np.nan):.4f}")
        
        lines.extend(["", "=" * 70])
        
        return "\n".join(lines)


class _OverflowAnalyzer:
    """溢出效应分析器（内部使用）"""
    
    def __init__(
        self,
        natural_lcz_values: Optional[List[int]] = None,
        built_lcz_values: Optional[List[int]] = None,
        high_natural_threshold: float = 0.3,
        low_natural_threshold: float = 0.1,
        verbose: bool = True
    ):
        self.natural_lcz_values = natural_lcz_values or LCZ_NATURAL_DEFAULT
        self.built_lcz_values = built_lcz_values or LCZ_BUILT_DEFAULT
        self.high_natural_threshold = high_natural_threshold
        self.low_natural_threshold = low_natural_threshold
        self.verbose = verbose
        self.result: Optional[OverflowResult] = None
    
    def analyze(
        self,
        gdf: gpd.GeoDataFrame,
        spatial_weights: Optional[SpatialWeightMatrix] = None,
        ta_column: str = 'Ta_optimized',
        lcz_column: str = 'LCZ',
        distance_threshold: float = 5000.0,
        decay_function: str = 'gaussian',
        cache_dir: Optional[Union[str, Path]] = None
    ) -> OverflowResult:
        """执行溢出效应分析"""
        if self.verbose:
            print("\n" + "=" * 70)
            print("溢出效应分析：自然景观对建成区的降温溢出")
            print("=" * 70)
        
        Ta = gdf[ta_column].values.astype(float)
        lcz = gdf[lcz_column].values
        n = len(gdf)
        
        if lcz.dtype == object:
            lcz = pd.to_numeric(lcz, errors='coerce').fillna(0).astype(int)
        
        is_natural = np.isin(lcz, self.natural_lcz_values)
        is_built = np.isin(lcz, self.built_lcz_values)
        
        n_natural = is_natural.sum()
        n_built = is_built.sum()
        
        if self.verbose:
            print(f"\n街区分类:")
            print(f"  总街区数: {n}")
            print(f"  自然景观: {n_natural} ({100*n_natural/n:.1f}%)")
            print(f"  建成区: {n_built} ({100*n_built/n:.1f}%)")
        
        if spatial_weights is None:
            if self.verbose:
                print(f"\n构建空间权重矩阵 (阈值={distance_threshold}m, 衰减={decay_function})...")
            spatial_weights = SpatialWeightMatrix(
                gdf,
                distance_threshold=distance_threshold,
                decay_function=decay_function,
                row_standardize=True,
                cache_dir=cache_dir
            )
        
        W = spatial_weights.W
        
        # 计算分析结果
        if self.verbose:
            print("\n计算自然邻居占比...")
        natural_neighbor_ratio = self._compute_natural_neighbor_ratio(W, is_natural)
        
        if self.verbose:
            print("\n进行溢出效应回归...")
        spillover_results = self._spillover_regression(Ta, W, is_natural, is_built)
        
        if self.verbose:
            print("\n计算局部 Moran's I (LISA)...")
        local_moran_i, cluster_type = self._compute_local_moran(Ta, W)
        
        group_stats = self._group_comparison(Ta, is_built, natural_neighbor_ratio)
        
        if self.verbose:
            print("\n按自然景观类型分解溢出效应...")
        spillover_by_lcz = self._analyze_by_lcz_type(Ta, W, lcz, is_built)
        
        self.result = OverflowResult(
            n_total=n,
            n_natural=n_natural,
            n_built=n_built,
            natural_neighbor_ratio=natural_neighbor_ratio,
            spillover_coeff=spillover_results['coeff'],
            spillover_r2=spillover_results['r2'],
            spillover_pvalue=spillover_results['pvalue'],
            spillover_intercept=spillover_results['intercept'],
            local_moran_i=local_moran_i,
            cluster_type=cluster_type,
            mean_Ta_high_natural=group_stats['mean_Ta_high_natural'],
            mean_Ta_low_natural=group_stats['mean_Ta_low_natural'],
            cooling_effect=group_stats['cooling_effect'],
            spillover_by_lcz=spillover_by_lcz,
            Ta=Ta,
            lcz=lcz
        )
        
        if self.verbose:
            print(self.result.summary())
        
        return self.result
    
    def _compute_natural_neighbor_ratio(self, W: csr_matrix, is_natural: np.ndarray) -> np.ndarray:
        """计算每个街区的自然邻居占比"""
        return np.asarray(W @ is_natural.astype(float)).flatten()
    
    def _spillover_regression(
        self, Ta: np.ndarray, W: csr_matrix, 
        is_natural: np.ndarray, is_built: np.ndarray
    ) -> Dict:
        """溢出效应回归"""
        n = len(Ta)
        Ta_natural_neighbors = np.full(n, np.nan)
        
        for i in range(n):
            row = W.getrow(i)
            neighbors = row.indices
            weights = row.data
            
            if len(neighbors) == 0:
                continue
            
            natural_mask = is_natural[neighbors]
            if not natural_mask.any():
                continue
            
            natural_neighbors = neighbors[natural_mask]
            natural_weights = weights[natural_mask]
            
            if natural_weights.sum() > 0:
                Ta_natural_neighbors[i] = np.average(
                    Ta[natural_neighbors],
                    weights=natural_weights
                )
        
        valid_mask = is_built & ~np.isnan(Ta_natural_neighbors) & ~np.isnan(Ta)
        
        if valid_mask.sum() < 10:
            return {'coeff': np.nan, 'intercept': np.nan, 'r2': np.nan, 'pvalue': np.nan}
        
        X = Ta_natural_neighbors[valid_mask]
        y = Ta[valid_mask]
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            XtX = X_with_const.T @ X_with_const
            XtY = X_with_const.T @ y
            beta = np.linalg.solve(XtX, XtY)
            
            intercept = beta[0]
            coeff = beta[1]
            
            y_pred = X_with_const @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            n_obs = len(y)
            k = 2
            mse = ss_res / (n_obs - k)
            var_beta = mse * np.linalg.inv(XtX)
            se_coeff = np.sqrt(var_beta[1, 1])
            t_stat = coeff / se_coeff if se_coeff > 0 else 0
            pvalue = 2 * (1 - norm.cdf(abs(t_stat)))
            
        except np.linalg.LinAlgError:
            return {'coeff': np.nan, 'intercept': np.nan, 'r2': np.nan, 'pvalue': np.nan}
        
        return {'coeff': coeff, 'intercept': intercept, 'r2': r2, 'pvalue': pvalue}
    
    def _compute_local_moran(self, Ta: np.ndarray, W: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
        """计算局部 Moran's I (LISA)"""
        n = len(Ta)
        
        valid_mask = ~np.isnan(Ta)
        Ta_valid = np.where(valid_mask, Ta, np.nanmean(Ta))
        
        Ta_mean = np.nanmean(Ta)
        Ta_std = np.nanstd(Ta)
        
        if Ta_std == 0:
            return np.zeros(n), np.full(n, '', dtype='U10')
        
        z = (Ta_valid - Ta_mean) / Ta_std
        Wz = np.asarray(W @ z).flatten()
        
        local_moran_i = z * Wz
        
        cluster_type = np.full(n, '', dtype='U10')
        cluster_type[(z > 0) & (Wz > 0)] = 'Hot-Hot'
        cluster_type[(z < 0) & (Wz < 0)] = 'Cold-Cold'
        cluster_type[(z > 0) & (Wz < 0)] = 'High-Low'
        cluster_type[(z < 0) & (Wz > 0)] = 'Low-High'
        
        return local_moran_i, cluster_type
    
    def _group_comparison(
        self, Ta: np.ndarray, is_built: np.ndarray, 
        natural_neighbor_ratio: np.ndarray
    ) -> Dict:
        """分组比较"""
        high_natural_mask = (
            is_built &
            (natural_neighbor_ratio > self.high_natural_threshold) &
            ~np.isnan(Ta)
        )
        low_natural_mask = (
            is_built &
            (natural_neighbor_ratio < self.low_natural_threshold) &
            ~np.isnan(Ta)
        )
        
        mean_Ta_high = Ta[high_natural_mask].mean() if high_natural_mask.any() else np.nan
        mean_Ta_low = Ta[low_natural_mask].mean() if low_natural_mask.any() else np.nan
        
        cooling_effect = mean_Ta_low - mean_Ta_high if not (np.isnan(mean_Ta_high) or np.isnan(mean_Ta_low)) else np.nan
        
        return {
            'mean_Ta_high_natural': mean_Ta_high,
            'mean_Ta_low_natural': mean_Ta_low,
            'cooling_effect': cooling_effect,
        }
    
    def _analyze_by_lcz_type(
        self, Ta: np.ndarray, W: csr_matrix,
        lcz: np.ndarray, is_built: np.ndarray
    ) -> Dict[int, Dict]:
        """按自然景观 LCZ 类型分析"""
        results = {}
        n = len(Ta)
        
        for lcz_code in self.natural_lcz_values:
            is_this_lcz = (lcz == lcz_code)
            count = is_this_lcz.sum()
            
            if count == 0:
                continue
            
            mean_Ta = Ta[is_this_lcz & ~np.isnan(Ta)].mean() if (is_this_lcz & ~np.isnan(Ta)).any() else np.nan
            
            Ta_this_lcz_neighbors = np.full(n, np.nan)
            
            for i in range(n):
                if not is_built[i]:
                    continue
                
                row = W.getrow(i)
                neighbors = row.indices
                weights = row.data
                
                if len(neighbors) == 0:
                    continue
                
                this_lcz_mask = is_this_lcz[neighbors]
                if not this_lcz_mask.any():
                    continue
                
                this_neighbors = neighbors[this_lcz_mask]
                this_weights = weights[this_lcz_mask]
                
                if this_weights.sum() > 0:
                    Ta_this_lcz_neighbors[i] = np.average(
                        Ta[this_neighbors],
                        weights=this_weights
                    )
            
            valid_mask = is_built & ~np.isnan(Ta_this_lcz_neighbors) & ~np.isnan(Ta)
            
            spillover_coeff = np.nan
            if valid_mask.sum() >= 10:
                X = Ta_this_lcz_neighbors[valid_mask]
                y = Ta[valid_mask]
                
                if X.std() > 0 and y.std() > 0:
                    corr = np.corrcoef(X, y)[0, 1]
                    spillover_coeff = corr * y.std() / X.std()
            
            results[lcz_code] = {
                'count': count,
                'mean_Ta': mean_Ta,
                'spillover_coeff': spillover_coeff,
                'n_affected_built': valid_mask.sum()
            }
        
        return results
    
    def add_results_to_gdf(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """将分析结果添加到 GeoDataFrame"""
        if self.result is None:
            raise ValueError("请先运行 analyze() 方法")
        
        gdf = gdf.copy()
        gdf['natural_neighbor_ratio'] = self.result.natural_neighbor_ratio
        gdf['local_moran_i'] = self.result.local_moran_i
        gdf['cluster_type'] = self.result.cluster_type
        
        lcz = gdf.get('LCZ', pd.Series([0] * len(gdf)))
        if lcz.dtype == object:
            lcz = pd.to_numeric(lcz, errors='coerce').fillna(0).astype(int)
        gdf['is_natural'] = np.isin(lcz.values, self.natural_lcz_values)
        gdf['is_built'] = np.isin(lcz.values, self.built_lcz_values)
        
        return gdf


def analyze_overflow(
    gdf: gpd.GeoDataFrame,
    ta_column: str = 'Ta_optimized',
    lcz_column: str = 'LCZ',
    distance_threshold: float = 5000.0,
    decay_function: str = 'gaussian',
    cache_dir: Optional[Union[str, Path]] = None,
    natural_lcz_values: Optional[List[int]] = None,
    verbose: bool = True
) -> Tuple[OverflowResult, gpd.GeoDataFrame]:
    """
    执行溢出效应分析
    
    参数:
        gdf: 包含气温和 LCZ 的 GeoDataFrame
        ta_column: 气温列名
        lcz_column: LCZ 列名
        distance_threshold: 空间权重距离阈值 (m)
        decay_function: 距离衰减函数
        cache_dir: 缓存目录
        natural_lcz_values: 自然景观 LCZ 值列表
        verbose: 是否打印信息
        
    返回:
        (OverflowResult, 带结果的 GeoDataFrame)
    """
    analyzer = _OverflowAnalyzer(
        natural_lcz_values=natural_lcz_values,
        verbose=verbose
    )
    
    result = analyzer.analyze(
        gdf=gdf,
        ta_column=ta_column,
        lcz_column=lcz_column,
        distance_threshold=distance_threshold,
        decay_function=decay_function,
        cache_dir=cache_dir
    )
    
    gdf_with_results = analyzer.add_results_to_gdf(gdf)
    
    return result, gdf_with_results
