"""
空间自相关分析模块

实现空间权重矩阵和空间滞后模型，用于分析水平交换项 ΔQ_A（气温空间自相关）。

核心组件:
    - SpatialWeightMatrix: 空间权重矩阵（稀疏矩阵优化）
    - analyze_spatial_autocorrelation: Moran's I 和空间滞后模型分析

物理背景:
    水平交换项 ΔQ_A 描述相邻街区之间因气温差异导致的热量交换。
    使用空间滞后模型分析气温的空间自相关特征：
    
    Ta = μ + ρ * W * Ta + ε
    
    其中:
    - Ta: 各街区气温
    - W: 空间权重矩阵（行标准化）
    - ρ: 空间自相关系数
    - μ: 截距
    - ε: 残差
"""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import geopandas as gpd
from scipy.stats import norm
from scipy.sparse import csr_matrix, diags, save_npz, load_npz
from shapely.strtree import STRtree
from shapely import distance as shapely_distance


class SpatialWeightMatrix:
    """
    空间权重矩阵（使用稀疏矩阵优化）
    
    用于定义街区之间的空间邻接关系，支持水平交换项 ΔQ_A 的空间自相关分析。
    
    距离计算: 使用多边形边界之间的最近距离。
    使用空间索引（STRtree）和稀疏矩阵优化大规模数据，避免内存溢出。
    
    支持的距离衰减函数:
        - binary: 二值权重（距离 < 阈值为 1，否则为 0）
        - linear: 线性衰减（1 - d/threshold）
        - inverse: 反比衰减（1/d）
        - gaussian: 高斯衰减（exp(-d²/2σ²)）
    
    缓存支持:
        使用 cache_dir 参数可以将计算好的权重矩阵保存到磁盘，
        下次使用相同参数时可以直接从缓存加载，避免重复计算。
    """
    
    CACHE_METADATA_FILE = 'spatial_weights_metadata.pkl'
    CACHE_MATRIX_FILE = 'spatial_weights_matrix.npz'
    
    def __init__(
        self,
        districts_gdf: gpd.GeoDataFrame,
        distance_threshold: float = 5000.0,
        decay_function: str = 'binary',
        row_standardize: bool = True,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """
        构建空间权重矩阵（稀疏矩阵版本）
        
        参数:
            districts_gdf: 街区 GeoDataFrame
            distance_threshold: 距离阈值（米），边界距离小于此值视为邻居
            decay_function: 距离衰减函数 ('binary', 'linear', 'inverse', 'gaussian')
            row_standardize: 是否行标准化（每行权重和为1）
            cache_dir: 缓存目录路径（如果提供，会尝试从缓存加载或保存到缓存）
        """
        self.n = len(districts_gdf)
        self.distance_threshold = distance_threshold
        self.decay_function = decay_function
        self.row_standardize = row_standardize
        self._cache_dir = Path(cache_dir) if cache_dir else None
        
        # 计算几何哈希，用于验证缓存有效性
        self._geometry_hash = self._compute_geometry_hash(districts_gdf)
        
        # 尝试从缓存加载
        if self._cache_dir and self._try_load_from_cache():
            print(f"  ✓ 从缓存加载空间权重矩阵: {self._cache_dir}")
        else:
            # 使用稀疏矩阵和空间索引构建权重矩阵
            self.W = self._build_sparse_weight_matrix(districts_gdf)
            self.neighbor_counts = np.array(self.W.sum(axis=1)).flatten()
            # 保存到缓存
            if self._cache_dir:
                self.save_to_cache()
    
    @staticmethod
    def _compute_geometry_hash(gdf: gpd.GeoDataFrame) -> str:
        """计算 GeoDataFrame 几何的哈希值，用于验证缓存有效性"""
        n = len(gdf)
        indices = []
        if n <= 30:
            indices = list(range(n))
        else:
            indices = list(range(10)) + list(range(n//2 - 5, n//2 + 5)) + list(range(n-10, n))
        
        hash_data = f"n={n};"
        for i in indices:
            geom = gdf.geometry.iloc[i]
            if geom is not None:
                bounds = geom.bounds
                area = geom.area
                hash_data += f"{i}:{bounds}:{area:.6f};"
        
        return hashlib.md5(hash_data.encode()).hexdigest()[:16]
    
    def _get_cache_key(self) -> Dict:
        """获取缓存键"""
        return {
            'n': self.n,
            'distance_threshold': self.distance_threshold,
            'decay_function': self.decay_function,
            'row_standardize': self.row_standardize,
            'geometry_hash': self._geometry_hash
        }
    
    def _try_load_from_cache(self) -> bool:
        """尝试从缓存加载权重矩阵"""
        if not self._cache_dir:
            return False
        
        metadata_path = self._cache_dir / self.CACHE_METADATA_FILE
        matrix_path = self._cache_dir / self.CACHE_MATRIX_FILE
        
        if not metadata_path.exists() or not matrix_path.exists():
            return False
        
        try:
            with open(metadata_path, 'rb') as f:
                cached_metadata = pickle.load(f)
            
            current_key = self._get_cache_key()
            if cached_metadata.get('cache_key') != current_key:
                print(f"  缓存参数不匹配，将重新计算...")
                return False
            
            self.W = load_npz(matrix_path)
            self.neighbor_counts = cached_metadata['neighbor_counts']
            return True
            
        except Exception as e:
            print(f"  缓存加载失败: {e}，将重新计算...")
            return False
    
    def save_to_cache(self) -> None:
        """保存权重矩阵到缓存"""
        if not self._cache_dir:
            return
        
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = self._cache_dir / self.CACHE_METADATA_FILE
        matrix_path = self._cache_dir / self.CACHE_MATRIX_FILE
        
        metadata = {
            'cache_key': self._get_cache_key(),
            'neighbor_counts': self.neighbor_counts
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        save_npz(matrix_path, self.W)
        print(f"  ✓ 空间权重矩阵已缓存: {self._cache_dir}")
    
    @classmethod
    def cache_exists(cls, cache_dir: Union[str, Path]) -> bool:
        """检查缓存是否存在"""
        cache_dir = Path(cache_dir)
        return (
            (cache_dir / cls.CACHE_METADATA_FILE).exists() and
            (cache_dir / cls.CACHE_MATRIX_FILE).exists()
        )
    
    def _build_sparse_weight_matrix(self, districts_gdf: gpd.GeoDataFrame) -> csr_matrix:
        """使用空间索引和稀疏矩阵构建权重矩阵"""
        print(f"  构建空间权重矩阵（稀疏矩阵，n={self.n}）...")
        
        geometries = districts_gdf.geometry.values
        tree = STRtree(geometries)
        
        print(f"    使用 dwithin 谓词查询邻居（距离阈值: {self.distance_threshold}m）...")
        left_indices, right_indices = tree.query(
            geometries, 
            predicate='dwithin', 
            distance=self.distance_threshold
        )
        
        # 过滤掉自身
        mask = left_indices != right_indices
        left_indices = left_indices[mask]
        right_indices = right_indices[mask]
        
        print(f"    找到 {len(left_indices)} 对邻居关系")
        
        # 计算权重
        if self.decay_function == 'binary':
            weights = np.ones(len(left_indices), dtype=np.float64)
        else:
            print(f"    向量化计算距离并应用 {self.decay_function} 衰减函数...")
            
            geoms_left = geometries[left_indices]
            geoms_right = geometries[right_indices]
            distances = shapely_distance(geoms_left, geoms_right)
            
            sigma = self.distance_threshold / 3
            
            if self.decay_function == 'linear':
                weights = 1.0 - distances / self.distance_threshold
            elif self.decay_function == 'inverse':
                weights = 1.0 / np.maximum(distances, 1.0)
            elif self.decay_function == 'gaussian':
                weights = np.exp(-distances**2 / (2 * sigma**2))
            else:
                weights = np.zeros(len(distances), dtype=np.float64)
            
            positive_mask = weights > 0
            left_indices = left_indices[positive_mask]
            right_indices = right_indices[positive_mask]
            weights = weights[positive_mask]
        
        W = csr_matrix(
            (weights, (left_indices, right_indices)),
            shape=(self.n, self.n),
            dtype=np.float64
        )
        
        # 行标准化
        if self.row_standardize:
            row_sums = np.array(W.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1
            inv_row_sums = 1.0 / row_sums
            W = diags(inv_row_sums, format='csr') @ W
        
        print(f"    完成！非零元素: {W.nnz} ({100*W.nnz/(self.n*self.n):.4f}%)")
        
        return W
    
    def spatial_lag(self, y: np.ndarray) -> np.ndarray:
        """计算空间滞后项 Wy"""
        return self.W @ y
    
    def get_neighbors(self, idx: int) -> np.ndarray:
        """获取指定街区的邻居索引"""
        row = self.W.getrow(idx)
        return row.indices
    
    def summary(self) -> Dict:
        """权重矩阵摘要统计"""
        non_zero = self.W.nnz
        return {
            'n_districts': self.n,
            'distance_threshold': self.distance_threshold,
            'decay_function': self.decay_function,
            'row_standardized': self.row_standardize,
            'total_connections': int(non_zero),
            'avg_neighbors': float(self.neighbor_counts.mean()),
            'min_neighbors': int(self.neighbor_counts.min()),
            'max_neighbors': int(self.neighbor_counts.max()),
            'isolated_districts': int((self.neighbor_counts == 0).sum()),
            'sparsity': float(1.0 - non_zero / (self.n * self.n))
        }


def analyze_spatial_autocorrelation(
    Ta: np.ndarray,
    spatial_weights: SpatialWeightMatrix,
    verbose: bool = True
) -> Optional[Dict]:
    """
    空间滞后模型分析：水平交换项 ΔQ_A 对气温的影响
    
    空间滞后模型: Ta = μ + ρ * W * Ta + ε
    
    参数:
        Ta: 各街区气温 (n,)
        spatial_weights: 空间权重矩阵对象
        verbose: 是否打印分析结果
        
    返回:
        分析结果字典:
        {
            'rho': 空间自相关系数 ρ,
            'intercept': 截距 μ,
            'r_squared': R² 决定系数,
            'moran_i': Moran's I 统计量,
            'moran_p': Moran's I 的 p 值,
            'spatial_lag': 空间滞后项 W*Ta,
            'residuals': 模型残差
        }
    """
    W = spatial_weights.W
    n = len(Ta)
    
    if verbose:
        print("\n" + "=" * 60)
        print("空间滞后模型分析（水平交换项 ΔQ_A）")
        print("=" * 60)
        
        summary = spatial_weights.summary()
        print(f"\n空间权重矩阵:")
        print(f"  街区数量: {summary['n_districts']}")
        print(f"  距离阈值: {summary['distance_threshold']:.0f} m")
        print(f"  衰减函数: {summary['decay_function']}")
        print(f"  平均邻居数: {summary['avg_neighbors']:.1f}")
        print(f"  孤立街区数: {summary['isolated_districts']}")
    
    # 过滤有效数据
    valid_mask = ~np.isnan(Ta)
    n_valid = valid_mask.sum()
    
    if n_valid < 3:
        if verbose:
            print("有效样本数不足，无法进行空间分析")
        return None
    
    Ta_valid = Ta[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    W_valid = W[valid_indices[:, None], valid_indices]
    
    # 重新行标准化
    row_sums = np.array(W_valid.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    inv_row_sums = 1.0 / row_sums
    W_valid = diags(inv_row_sums, format='csr') @ W_valid
    
    # 计算空间滞后项
    W_Ta = W_valid @ Ta_valid
    
    # 1. Moran's I
    Ta_centered = Ta_valid - Ta_valid.mean()
    numerator = n_valid * (Ta_centered @ W_valid @ Ta_centered)
    denominator = W_valid.sum() * (Ta_centered @ Ta_centered)
    moran_i = numerator / denominator if denominator != 0 else 0
    
    E_I = -1 / (n_valid - 1)
    
    # 方差计算
    S0 = float(W_valid.sum())
    W_sym = W_valid + W_valid.T
    S1 = 0.5 * float((W_sym.multiply(W_sym)).sum())
    row_sums = np.array(W_valid.sum(axis=1)).flatten()
    col_sums = np.array(W_valid.sum(axis=0)).flatten()
    S2 = float(((row_sums + col_sums) ** 2).sum())
    
    b2 = (Ta_centered ** 4).sum() / n_valid / ((Ta_centered ** 2).sum() / n_valid) ** 2
    
    A = n_valid * ((n_valid**2 - 3*n_valid + 3) * S1 - n_valid * S2 + 3 * S0**2)
    B = b2 * ((n_valid**2 - n_valid) * S1 - 2*n_valid * S2 + 6 * S0**2)
    C = (n_valid - 1) * (n_valid - 2) * (n_valid - 3) * S0**2
    
    Var_I = (A - B) / C - E_I**2 if C != 0 else 0.01
    Var_I = max(Var_I, 1e-10)
    
    z_score = (moran_i - E_I) / np.sqrt(Var_I)
    moran_p = 2 * (1 - norm.cdf(abs(z_score)))
    
    # 2. 空间滞后模型 OLS
    X = np.column_stack([np.ones(n_valid), W_Ta])
    XtX = X.T @ X
    XtY = X.T @ Ta_valid
    
    try:
        beta = np.linalg.solve(XtX, XtY)
        intercept = beta[0]
        rho = beta[1]
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(X, Ta_valid, rcond=None)
        intercept = beta[0]
        rho = beta[1]
    
    y_pred = X @ beta
    ss_res = np.sum((Ta_valid - y_pred) ** 2)
    ss_tot = np.sum((Ta_valid - Ta_valid.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # 恢复完整数组
    spatial_lag_full = np.full(n, np.nan)
    residuals_full = np.full(n, np.nan)
    spatial_lag_full[valid_mask] = W_Ta
    residuals_full[valid_mask] = Ta_valid - y_pred
    
    if verbose:
        print(f"\n空间自相关检验 (Moran's I):")
        print(f"  Moran's I = {moran_i:.4f}")
        print(f"  期望值 E[I] = {E_I:.4f}")
        print(f"  Z 统计量 = {z_score:.2f}")
        print(f"  p 值 = {moran_p:.4f}")
        if moran_p < 0.05:
            print("  结论: 存在显著的空间自相关 (p < 0.05)")
        else:
            print("  结论: 空间自相关不显著 (p >= 0.05)")
        
        print(f"\n空间滞后模型 (Ta = μ + ρ·W·Ta + ε):")
        print(f"  截距 μ = {intercept:.2f} K ({intercept-273.15:.2f}°C)")
        print(f"  空间自相关系数 ρ = {rho:.4f}")
        print(f"  R² = {r_squared:.4f} (方差解释比例: {r_squared*100:.1f}%)")
        
        if abs(rho) > 0.1:
            direction = "正" if rho > 0 else "负"
            print(f"\n物理解释:")
            print(f"  ρ > 0 表示相邻街区气温相似（热量扩散/均质化）")
            print(f"  当前 ρ = {rho:.4f}，存在{direction}向空间自相关")
            print(f"  即相邻街区每 1K 温差，本街区温度变化约 {abs(rho):.2f}K")
        
        print("=" * 60)
    
    return {
        'rho': rho,
        'intercept': intercept,
        'r_squared': r_squared,
        'moran_i': moran_i,
        'moran_expected': E_I,
        'moran_z': z_score,
        'moran_p': moran_p,
        'spatial_lag': spatial_lag_full,
        'residuals': residuals_full,
        'n_valid_samples': n_valid,
        'spatial_weights_summary': spatial_weights.summary()
    }
