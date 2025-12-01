"""
空间自相关分析模块

实现空间权重矩阵和空间滞后模型，用于分析水平交换项 ΔQ_A（气温空间自相关）。

物理背景:
    水平交换项 ΔQ_A 描述相邻街区之间因气温差异导致的热量交换。
    由于 ΔQ_A 只与气温相关，不作为能量平衡方程的独立项，
    而是在 ALS 求解 Ta 后，使用空间滞后模型分析气温的空间自相关特征。

空间滞后模型:
    Ta = μ + ρ * W * Ta + ε
    
    其中:
    - Ta: 各街区气温
    - W: 空间权重矩阵（行标准化）
    - ρ: 空间自相关系数
    - μ: 截距
    - ε: 残差
"""

import numpy as np
import geopandas as gpd
from typing import Dict
from scipy.stats import norm
from scipy.sparse import csr_matrix, lil_matrix, diags
from shapely.strtree import STRtree


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
    """
    
    def __init__(
        self,
        districts_gdf: gpd.GeoDataFrame,
        distance_threshold: float = 5000.0,
        decay_function: str = 'binary',
        row_standardize: bool = True
    ):
        """
        构建空间权重矩阵（稀疏矩阵版本）
        
        参数:
            districts_gdf: 街区 GeoDataFrame
            distance_threshold: 距离阈值（米），边界距离小于此值视为邻居
            decay_function: 距离衰减函数 ('binary', 'linear', 'inverse', 'gaussian')
            row_standardize: 是否行标准化（每行权重和为1）
        """
        self.n = len(districts_gdf)
        self.distance_threshold = distance_threshold
        self.decay_function = decay_function
        self.row_standardize = row_standardize
        
        # 使用稀疏矩阵和空间索引构建权重矩阵（内存高效）
        self.W = self._build_sparse_weight_matrix(districts_gdf)
        
        # 计算邻居统计
        self.neighbor_counts = np.array(self.W.sum(axis=1)).flatten()
    
    def _build_sparse_weight_matrix(self, districts_gdf: gpd.GeoDataFrame) -> csr_matrix:
        """
        使用空间索引和稀疏矩阵构建权重矩阵（内存高效）
        
        使用 STRtree 的 dwithin 谓词直接查询距离内的邻居，
        避免创建缓冲区，大幅提升效率。
        """
        print(f"  构建空间权重矩阵（稀疏矩阵，n={self.n}）...")
        
        # 使用 STRtree 空间索引
        geometries = districts_gdf.geometry.values
        tree = STRtree(geometries)
        
        # 使用 dwithin 谓词批量查询距离内的所有邻居对
        # 返回 (left_indices, right_indices)，表示 geometries[left] 与 geometries[right] 距离 < threshold
        print(f"    使用 dwithin 谓词查询邻居（距离阈值: {self.distance_threshold}m）...")
        left_indices, right_indices = tree.query(
            geometries, 
            predicate='dwithin', 
            distance=self.distance_threshold
        )
        
        # 过滤掉自身（i == j 的情况）
        mask = left_indices != right_indices
        left_indices = left_indices[mask]
        right_indices = right_indices[mask]
        
        print(f"    找到 {len(left_indices)} 对邻居关系")
        
        # 使用稀疏矩阵（LIL格式便于逐元素构建）
        W = lil_matrix((self.n, self.n), dtype=np.float64)
        
        # 根据衰减函数计算权重
        if self.decay_function == 'binary':
            # 二值权重，无需计算距离
            for i, j in zip(left_indices, right_indices):
                W[i, j] = 1.0
        else:
            # 需要计算精确距离来确定权重
            print(f"    计算精确距离并应用 {self.decay_function} 衰减函数...")
            sigma = self.distance_threshold / 3  # 用于 gaussian 衰减
            
            for idx, (i, j) in enumerate(zip(left_indices, right_indices)):
                if (idx + 1) % 100000 == 0:
                    print(f"      进度: {idx+1}/{len(left_indices)} ({100*(idx+1)/len(left_indices):.1f}%)")
                
                dist = geometries[i].distance(geometries[j])
                
                if self.decay_function == 'linear':
                    weight = 1.0 - dist / self.distance_threshold
                elif self.decay_function == 'inverse':
                    weight = 1.0 / max(dist, 1.0)
                elif self.decay_function == 'gaussian':
                    weight = np.exp(-dist**2 / (2 * sigma**2))
                else:
                    weight = 0.0
                
                if weight > 0:
                    W[i, j] = weight
        
        # 转换为 CSR 格式（更高效的矩阵运算）
        W = W.tocsr()
        
        # 行标准化
        if self.row_standardize:
            row_sums = np.array(W.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1  # 避免除以0
            # 使用稀疏矩阵的逐元素除法（通过对角线矩阵）
            inv_row_sums = 1.0 / row_sums
            W = diags(inv_row_sums, format='csr') @ W
        
        print(f"    完成！非零元素: {W.nnz} ({100*W.nnz/(self.n*self.n):.4f}%)")
        
        return W
    
    def spatial_lag(self, y: np.ndarray) -> np.ndarray:
        """
        计算空间滞后项 Wy
        
        参数:
            y: 观测值向量 (n,)
            
        返回:
            空间滞后值 Wy (n,)
        """
        return self.W @ y
    
    def get_neighbors(self, idx: int) -> np.ndarray:
        """获取指定街区的邻居索引"""
        # 对于稀疏矩阵，使用 getrow 方法
        row = self.W.getrow(idx)
        return row.indices
    
    def summary(self) -> Dict:
        """权重矩阵摘要统计"""
        non_zero = self.W.nnz  # 稀疏矩阵的非零元素数量
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
            'sparsity': float(1.0 - non_zero / (self.n * self.n))  # 稀疏度
        }


def analyze_spatial_autocorrelation(
    Ta: np.ndarray,
    spatial_weights: SpatialWeightMatrix,
    verbose: bool = True
) -> Dict:
    """
    空间滞后模型分析：水平交换项 ΔQ_A 对气温的影响
    
    由于 ΔQ_A（水平交换项）是气温的空间自相关项，
    使用空间滞后模型分析相邻街区气温对本街区的影响。
    
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
            'moran_p': Moran's I 的 p 值（基于正态近似）,
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
        
        # 打印空间权重矩阵摘要
        summary = spatial_weights.summary()
        print(f"\n空间权重矩阵:")
        print(f"  街区数量: {summary['n_districts']}")
        print(f"  距离阈值: {summary['distance_threshold']:.0f} m")
        print(f"  衰减函数: {summary['decay_function']}")
        print(f"  平均邻居数: {summary['avg_neighbors']:.1f}")
        print(f"  孤立街区数: {summary['isolated_districts']}")
    
    # 过滤有效数据（非NaN）
    valid_mask = ~np.isnan(Ta)
    n_valid = valid_mask.sum()
    
    if n_valid < 3:
        if verbose:
            print("有效样本数不足，无法进行空间分析")
        return None
    
    # 对于有 NaN 的情况，需要调整权重矩阵
    Ta_valid = Ta[valid_mask]
    # 对于稀疏矩阵，使用布尔索引
    valid_indices = np.where(valid_mask)[0]
    W_valid = W[valid_indices[:, None], valid_indices]
    
    # 重新行标准化
    row_sums = np.array(W_valid.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    # 使用稀疏矩阵的逐元素除法
    inv_row_sums = 1.0 / row_sums
    W_valid = diags(inv_row_sums, format='csr') @ W_valid
    
    # 计算空间滞后项 W*Ta
    W_Ta = W_valid @ Ta_valid
    
    # 1. 计算 Moran's I
    Ta_centered = Ta_valid - Ta_valid.mean()
    numerator = n_valid * (Ta_centered @ W_valid @ Ta_centered)
    denominator = W_valid.sum() * (Ta_centered @ Ta_centered)
    moran_i = numerator / denominator if denominator != 0 else 0
    
    # Moran's I 期望值和方差（正态近似）
    E_I = -1 / (n_valid - 1)
    
    # 简化的方差计算（兼容稀疏矩阵）
    S0 = float(W_valid.sum())
    # 对于稀疏矩阵，使用 element-wise 操作
    W_sym = W_valid + W_valid.T
    S1 = 0.5 * float((W_sym.multiply(W_sym)).sum())
    # 计算行和列的和
    row_sums = np.array(W_valid.sum(axis=1)).flatten()
    col_sums = np.array(W_valid.sum(axis=0)).flatten()
    S2 = float(((row_sums + col_sums) ** 2).sum())
    
    b2 = (Ta_centered ** 4).sum() / n_valid / ((Ta_centered ** 2).sum() / n_valid) ** 2
    
    A = n_valid * ((n_valid**2 - 3*n_valid + 3) * S1 - n_valid * S2 + 3 * S0**2)
    B = b2 * ((n_valid**2 - n_valid) * S1 - 2*n_valid * S2 + 6 * S0**2)
    C = (n_valid - 1) * (n_valid - 2) * (n_valid - 3) * S0**2
    
    Var_I = (A - B) / C - E_I**2 if C != 0 else 0.01
    Var_I = max(Var_I, 1e-10)  # 避免负方差
    
    # Z 统计量和 p 值
    z_score = (moran_i - E_I) / np.sqrt(Var_I)
    moran_p = 2 * (1 - norm.cdf(abs(z_score)))  # 双尾检验
    
    # 2. 空间滞后模型: Ta = μ + ρ * W*Ta + ε
    # 使用 OLS 估计
    X = np.column_stack([np.ones(n_valid), W_Ta])
    
    # OLS: β = (X'X)^(-1) X'y
    XtX = X.T @ X
    XtY = X.T @ Ta_valid
    
    try:
        beta = np.linalg.solve(XtX, XtY)
        intercept = beta[0]
        rho = beta[1]
    except np.linalg.LinAlgError:
        # 矩阵奇异，使用最小二乘
        beta, _, _, _ = np.linalg.lstsq(X, Ta_valid, rcond=None)
        intercept = beta[0]
        rho = beta[1]
    
    # 计算 R²
    y_pred = X @ beta
    ss_res = np.sum((Ta_valid - y_pred) ** 2)
    ss_tot = np.sum((Ta_valid - Ta_valid.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # 计算完整的空间滞后项和残差
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

