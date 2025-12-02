"""
交替最小二乘法（ALS）回归模块

实现基于能量平衡的 ALS 回归，求解各街区气温。

物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si + λ·(Ta - W·Ta) = 0

    其中:
    - f(Ta): 能量平衡系数（关于Ta的函数）
    - X_Fi: 人为热 Q_F 相关特征（与 Ta 独立）
    - X_Si: 建筑储热 ΔQ_Sb 相关特征（与 Ta 独立）
    - αi, βi: 待求回归系数
    - Ta: 每个街区的气温（待优化）
    - W: 空间权重矩阵（行标准化）
    - λ: 水平交换系数（待估计）
    - W·Ta: 邻域平均气温

水平交换项 ΔQ_A:
    ΔQ_A = λ·(Ta - W·Ta)
    物理意义: 本街区与邻域的热量水平交换
    当 λ > 0 时，气温趋向于邻域平均（热量扩散）

算法:
    1. 初始化: 使用ERA5气温作为Ta初值
    2. 固定 Ta，拟合线性项 α, β, λ
    3. 固定 α, β, λ，优化各街区的 Ta
    4. 重复2-3直到收敛
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, spmatrix
import warnings


class ALSRegression:
    """
    交替最小二乘法回归模型
    
    求解能量平衡方程中各街区的气温。
    """

    def __init__(self):
        """初始化回归模型"""
        self.alpha_coeffs = None  # X_Fi的系数（人为热 Q_F）
        self.beta_coeffs = None   # X_Si的系数（储热 ΔQ_Sb）
        self.lambda_coeff = None  # 空间交换系数 λ
        self.Ta_per_district = None  # 每个街区的气温
        self.history = []  # 迭代历史
        self.valid_district_ids = None  # 有效样本的 district_id（过滤后）
        self.scaler_F = None  # X_F的标准化器（只标准化连续变量）
        self.scaler_S = None  # X_S的标准化器
        self.X_F_continuous_indices = None  # X_F中连续变量的列索引
        self.X_S_continuous_indices = None  # X_S中连续变量的列索引（X_S全部是连续的）
        # 能量平衡系数（在 fit 中设置）
        self.f_Ta_coeff1 = None  # Ta 一次项系数
        self.f_Ta_coeff2 = None  # Ta 二次项系数
        self.y_residual = None   # 残差项
        # 空间权重矩阵（在 fit 中设置）
        self.spatial_weights = None  # 空间权重矩阵 W
        self.spatial_mask = None  # 空间覆盖掩码
        self.spatial_lag = None  # 最终的空间滞后项 W·Ta

    def prepare_regression_data(
        self,
        aggregated_df: pd.DataFrame,
        districts_gdf: gpd.GeoDataFrame,
        f_Ta_coeff1_column: str = 'f_Ta_coeff1_mean',
        f_Ta_coeff2_column: Optional[str] = 'f_Ta_coeff2_mean',
        residual_column: str = 'residual_mean',
        era5_Ta_column: str = 'era5_air_temperature_mean',
        X_F_columns: Optional[List[str]] = None,
        X_S_columns: Optional[List[str]] = None,
        X_C_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        准备回归数据

        从聚合后的DataFrame和街区属性构建回归矩阵

        参数:
            aggregated_df: 从栅格聚合的DataFrame
            districts_gdf: 街区GeoDataFrame（包含属性字段）
            f_Ta_coeff1_column: Ta一次项系数栏位名
            f_Ta_coeff2_column: Ta二次项系数栏位名（可选，None则使用一次近似）
            residual_column: 残差栏位名
            era5_Ta_column: ERA5气温栏位名 - 用于Ta初始化
            X_F_columns: 人为热 Q_F 相关特征列表（连续变量）
            X_S_columns: 建筑储热 ΔQ_Sb 相关特征列表（连续变量）
            X_C_columns: 分类特征列表（将进行 one-hot 编码）

        返回:
            (X_F, X_S, f_Ta_coeff1, f_Ta_coeff2, y_residual, era5_Ta_mean, Ts_mean)
            f_Ta_coeff2 为 None 时表示使用一次近似
            Ts_mean: 地表温度均值（用于空间异质性初始化）
        """
        # 合并数据
        if 'district_id' in aggregated_df.columns:
            merged = aggregated_df.copy()
        else:
            merged = aggregated_df.reset_index()
            merged['district_id'] = merged.index

        # 添加街区属性
        for col in districts_gdf.columns:
            if col != 'geometry' and col not in merged.columns:
                merged[col] = districts_gdf[col].values

        # 提取f(Ta)系数和残差
        f_Ta_coeff1 = merged[f_Ta_coeff1_column].values
        f_Ta_coeff2 = None
        if f_Ta_coeff2_column and f_Ta_coeff2_column in merged.columns:
            f_Ta_coeff2 = merged[f_Ta_coeff2_column].values
        y_residual = merged[residual_column].values

        # 提取ERA5气温（用于Ta初始化）
        era5_Ta_mean = merged[era5_Ta_column].values if era5_Ta_column in merged.columns else None
        
        # 提取地表温度 Ts（用于空间异质性初始化）
        Ts_column = 'landsat_lst_mean'
        Ts_mean = merged[Ts_column].values if Ts_column in merged.columns else None

        # 构建有效样本掩码
        # 1. 过滤 NaN 值
        valid_mask = (
            ~np.isnan(f_Ta_coeff1) &
            ~np.isnan(y_residual)
        )
        
        # 2. 如果有二次项系数，也检查其 NaN
        if f_Ta_coeff2 is not None:
            valid_mask = valid_mask & ~np.isnan(f_Ta_coeff2)
        
        # 3. 如果 era5_Ta_mean 存在，也要检查其 NaN
        if era5_Ta_mean is not None:
            valid_mask = valid_mask & ~np.isnan(era5_Ta_mean)
        
        # 4. 如果 Ts_mean 存在，也要检查其 NaN
        if Ts_mean is not None:
            valid_mask = valid_mask & ~np.isnan(Ts_mean)
        
        # 5. 约束：landuse 在 [0, 12] 范围内
        if 'landuse' in merged.columns:
            landuse = merged['landuse'].values
            valid_mask = valid_mask & (landuse >= 0) & (landuse <= 12)
        
        # 应用掩码过滤有效样本
        if not np.all(valid_mask):
            n_filtered = np.sum(~valid_mask)
            print(f"  过滤掉 {n_filtered} 个无效样本（NaN 或 landuse 超出 [0,12] 范围）")
            print(f"  有效样本数: {np.sum(valid_mask)} / {len(valid_mask)}")
        
        merged = merged[valid_mask].copy()
        f_Ta_coeff1 = f_Ta_coeff1[valid_mask]
        if f_Ta_coeff2 is not None:
            f_Ta_coeff2 = f_Ta_coeff2[valid_mask]
        y_residual = y_residual[valid_mask]
        if era5_Ta_mean is not None:
            era5_Ta_mean = era5_Ta_mean[valid_mask]
        if Ts_mean is not None:
            Ts_mean = Ts_mean[valid_mask]
        
        # 保存有效样本的 district_id
        self.valid_district_ids = merged['district_id'].values

        # 确定需要 one-hot 编码的列
        categorical_columns = set(X_C_columns) if X_C_columns else set()

        # 构建X_F矩阵（连续变量）
        n_F_continuous = 0
        if X_F_columns:
            X_F = self._build_feature_matrix(merged, X_F_columns, categorical_columns)
            n_F_continuous = X_F.shape[1]  # 记录连续变量的数量
        else:
            X_F = np.zeros((len(merged), 0))

        # 构建X_S矩阵（连续变量）
        if X_S_columns:
            X_S = self._build_feature_matrix(merged, X_S_columns, categorical_columns)
        else:
            X_S = np.zeros((len(merged), 0))

        # 构建分类变量矩阵（one-hot编码）并合并到X_F
        if X_C_columns:
            X_C = self._build_categorical_matrix(merged, X_C_columns)
            X_F = np.hstack([X_F, X_C]) if X_F.shape[1] > 0 else X_C
        
        # 记录连续变量的索引（用于后续标准化）
        # X_F中前n_F_continuous列是连续变量，后面是one-hot编码
        self.X_F_continuous_indices = np.arange(n_F_continuous) if n_F_continuous > 0 else np.array([], dtype=int)
        # X_S全部是连续变量
        self.X_S_continuous_indices = np.arange(X_S.shape[1]) if X_S.shape[1] > 0 else np.array([], dtype=int)

        # 最终检查：确保特征矩阵中没有 NaN（额外安全措施）
        if X_F.shape[1] > 0:
            X_F_has_nan = np.isnan(X_F).any(axis=1)
            if X_F_has_nan.any():
                print(f"  警告: X_F 特征矩阵中发现 NaN，将过滤 {np.sum(X_F_has_nan)} 个样本")
                valid_mask_final = ~X_F_has_nan
                merged = merged[valid_mask_final].copy()
                X_F = X_F[valid_mask_final]
                X_S = X_S[valid_mask_final]
                f_Ta_coeff1 = f_Ta_coeff1[valid_mask_final]
                if f_Ta_coeff2 is not None:
                    f_Ta_coeff2 = f_Ta_coeff2[valid_mask_final]
                y_residual = y_residual[valid_mask_final]
                if era5_Ta_mean is not None:
                    era5_Ta_mean = era5_Ta_mean[valid_mask_final]
                if Ts_mean is not None:
                    Ts_mean = Ts_mean[valid_mask_final]
                self.valid_district_ids = merged['district_id'].values
        
        if X_S.shape[1] > 0:
            X_S_has_nan = np.isnan(X_S).any(axis=1)
            if X_S_has_nan.any():
                print(f"  警告: X_S 特征矩阵中发现 NaN，将过滤 {np.sum(X_S_has_nan)} 个样本")
                valid_mask_final = ~X_S_has_nan
                merged = merged[valid_mask_final].copy()
                X_F = X_F[valid_mask_final]
                X_S = X_S[valid_mask_final]
                f_Ta_coeff1 = f_Ta_coeff1[valid_mask_final]
                if f_Ta_coeff2 is not None:
                    f_Ta_coeff2 = f_Ta_coeff2[valid_mask_final]
                y_residual = y_residual[valid_mask_final]
                if era5_Ta_mean is not None:
                    era5_Ta_mean = era5_Ta_mean[valid_mask_final]
                if Ts_mean is not None:
                    Ts_mean = Ts_mean[valid_mask_final]
                self.valid_district_ids = merged['district_id'].values

        return X_F, X_S, f_Ta_coeff1, f_Ta_coeff2, y_residual, era5_Ta_mean, Ts_mean

    def _optimize_Ta_single(
        self,
        ta_init: float,
        coeff2: float,
        coeff1: float,
        linear_pred_i: float,
        y_residual_i: float
    ) -> float:
        """
        单个街区的 Ta 优化（当求根公式不适用时）
        
        求解: coeff2×Ta² + coeff1×Ta + (linear_pred + y_residual) = 0
        即能量平衡方程: f(Ta) + linear_pred + residual = 0
        """
        def objective(ta):
            pred = coeff2 * (ta ** 2) + coeff1 * ta + linear_pred_i
            return (pred + y_residual_i) ** 2

        bounds = [(273.15, 323.15)]  # 0°C 到 50°C
        result = minimize(
            objective,
            x0=ta_init,
            method='L-BFGS-B',
            bounds=bounds
        )
        return result.x[0]

    def _build_feature_matrix(
        self,
        df: pd.DataFrame,
        columns: List[str],
        categorical_columns: Optional[set] = None
    ) -> np.ndarray:
        """
        构建连续变量特征矩阵
        
        参数:
            df: 数据DataFrame
            columns: 要构建的列名列表
            categorical_columns: 分类变量列名集合（将被跳过，由 _build_categorical_matrix 处理）
        """
        X_list = []
        categorical_columns = categorical_columns or set()

        for col in columns:
            if col not in df.columns:
                warnings.warn(f"列'{col}'不存在，跳过")
                continue
            
            # 跳过分类变量（由 _build_categorical_matrix 单独处理）
            if col in categorical_columns:
                continue

            values = df[col].values
            X_list.append(values.reshape(-1, 1))

        if X_list:
            return np.hstack(X_list)
        else:
            return np.zeros((len(df), 0))

    def _build_categorical_matrix(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> np.ndarray:
        """
        构建分类变量特征矩阵（one-hot 编码）
        
        参数:
            df: 数据DataFrame
            columns: 需要进行 one-hot 编码的列名列表
        
        注意:
            - landuse 列只根据 0-11 的值生成编码（12 仅用于筛选）
        """
        X_list = []

        for col in columns:
            if col not in df.columns:
                warnings.warn(f"分类列'{col}'不存在，跳过")
                continue

            # 特殊处理 landuse 列：使用 0-12 的值进行编码
            if col == 'landuse':
                # landuse 范围为 0-12
                # 转为 Categorical 类型，指定类别范围 0-12
                landuse_categorical = pd.Categorical(df[col], categories=range(13))
                dummies = pd.get_dummies(landuse_categorical, prefix=col, drop_first=True)
            else:
                # One-hot 编码，drop_first=True 避免多重共线性
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            
            X_list.append(dummies.values)

        if X_list:
            return np.hstack(X_list)
        else:
            return np.zeros((len(df), 0))

    def fit(
        self,
        X_F: np.ndarray,
        X_S: np.ndarray,
        f_Ta_coeff1: np.ndarray,
        f_Ta_coeff2: Optional[np.ndarray] = None,
        y_residual: Optional[np.ndarray] = None,
        era5_Ta_mean: Optional[np.ndarray] = None,
        Ts_mean: Optional[np.ndarray] = None,
        spatial_weights: Optional[Union[np.ndarray, spmatrix]] = None,
        spatial_mask: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        ridge_alpha: float = 0.0,
        verbose: bool = True
    ) -> Dict:
        """
        交替最小二乘法（ALS）求解

        求解方程: f(Ta) + Σαi·X_Fi + Σβi·X_Si + λ·(Ta - W·Ta) = 0
        
        其中 f(Ta) = coeff2 × Ta² + coeff1 × Ta + residual（二次形式）
        或 f(Ta) = coeff1 × Ta + residual（一次近似）

        水平交换项: ΔQ_A = λ·(Ta - W·Ta)
        - W·Ta 是邻域平均气温
        - λ 是水平交换系数（待估计）

        参数:
            X_F: 人为热 Q_F 特征矩阵 (n, p_F)
            X_S: 建筑储热 ΔQ_Sb 特征矩阵 (n, p_S)
            f_Ta_coeff1: Ta一次项系数 (n,)
            f_Ta_coeff2: Ta二次项系数 (n,)，None 则使用一次近似
            y_residual: 观测残差 (n,)
            era5_Ta_mean: 各街区的ERA5气温平均 (n,) - 用于Ta初始化校准
            Ts_mean: 各街区的地表温度 Ts (n,) - 用于空间异质性初始化
            spatial_weights: 空间权重矩阵 W (m, m)，行标准化
                            用于计算水平交换项 ΔQ_A
                            None 则不考虑空间效应
                            m 可以等于 n（全覆盖）或小于 n（部分覆盖）
            spatial_mask: bool数组 (n,)，标记哪些样本在空间权重矩阵覆盖范围内
                         仅当 spatial_weights 不为 None 且部分覆盖时使用
                         None 表示全覆盖
            max_iter: 最大迭代次数
            tol: 收敛容差
            ridge_alpha: 岭回归正则化参数 (0 表示普通最小二乘)
            verbose: 是否打印迭代信息

        返回:
            结果字典，包含:
            - Ta_per_district: 各街区气温
            - alpha_coeffs: 人为热系数
            - beta_coeffs: 储热系数
            - lambda_coeff: 空间交换系数（如果使用空间权重）
            - spatial_lag: 邻域平均气温 W·Ta（如果使用空间权重）
        """
        # 保存系数用于后续计算
        self.f_Ta_coeff1 = f_Ta_coeff1
        self.f_Ta_coeff2 = f_Ta_coeff2
        self.y_residual = y_residual
        n_districts = len(y_residual)

        # 处理空间权重矩阵
        self.spatial_weights = spatial_weights
        self.spatial_mask = spatial_mask
        use_spatial = spatial_weights is not None
        
        if use_spatial:
            n_spatial = spatial_weights.shape[0]
            if spatial_mask is None:
                # 全覆盖模式：空间矩阵维度必须等于样本数
                if n_spatial != n_districts:
                    raise ValueError(
                        f"空间权重矩阵维度不匹配: "
                        f"W.shape={spatial_weights.shape}, n_districts={n_districts}"
                    )
            else:
                # 部分覆盖模式：验证掩码和矩阵维度
                n_covered = np.sum(spatial_mask)
                if n_spatial != n_covered:
                    raise ValueError(
                        f"空间权重矩阵维度与掩码不匹配: "
                        f"W.shape={spatial_weights.shape}, n_covered={n_covered}"
                    )

        # 初始化Ta：使用 Ts + (ERA5_Ta均值 - Ts均值) 以保留空间异质性
        if Ts_mean is not None and era5_Ta_mean is not None:
            # 使用 Ts 的空间分布 + ERA5 校准偏移
            # Ta_init = Ts + (mean(ERA5_Ta) - mean(Ts))
            offset = np.nanmean(era5_Ta_mean) - np.nanmean(Ts_mean)
            Ta = Ts_mean + offset
            if verbose:
                print(f"  Ta初始化: 使用 Ts + 偏移量 ({offset:.2f} K)")
                print(f"    Ts 均值: {np.nanmean(Ts_mean):.2f} K, std: {np.nanstd(Ts_mean):.2f} K")
                print(f"    ERA5_Ta 均值: {np.nanmean(era5_Ta_mean):.2f} K")
                print(f"    初始 Ta 均值: {np.nanmean(Ta):.2f} K, std: {np.nanstd(Ta):.2f} K")
        elif era5_Ta_mean is not None:
            Ta = era5_Ta_mean.copy()
            if verbose:
                print(f"  Ta初始化: 使用 ERA5 气温 (无 Ts 数据)")
        else:
            Ta = np.full(n_districts, 298.15)  # 默认25°C
            if verbose:
                print(f"  Ta初始化: 使用默认值 298.15 K")

        self.history = []

        # ========== 诊断代码 ==========
        if verbose:
            print("\n开始ALS回归迭代...")
            print("=" * 60)
            print(f"特征维度: X_F={X_F.shape[1]}, X_S={X_S.shape[1]}")
            if use_spatial:
                nnz = spatial_weights.nnz if hasattr(spatial_weights, 'nnz') else np.count_nonzero(spatial_weights)
                print(f"空间权重矩阵: {spatial_weights.shape}, 非零元素={nnz}")
            print(f"  连续变量: X_F连续={len(self.X_F_continuous_indices)}, X_S连续={len(self.X_S_continuous_indices)}")
            
            # 诊断1: 特征量级统计
            if X_F.shape[1] > 0:
                X_F_continuous = X_F[:, self.X_F_continuous_indices] if len(self.X_F_continuous_indices) > 0 else np.array([]).reshape(n_districts, 0)
                if X_F_continuous.size > 0:
                    print(f"\n[X_F 连续变量统计]")
                    print(f"  min: {X_F_continuous.min(axis=0)}")
                    print(f"  max: {X_F_continuous.max(axis=0)}")
                    print(f"  mean: {X_F_continuous.mean(axis=0)}")
                    print(f"  std: {X_F_continuous.std(axis=0)}")
                
                # 检查one-hot编码（应该是0/1）
                if len(self.X_F_continuous_indices) < X_F.shape[1]:
                    X_F_onehot = X_F[:, len(self.X_F_continuous_indices):]
                    print(f"\n[X_F One-hot编码统计]")
                    print(f"  唯一值: {np.unique(X_F_onehot)}")
                    print(f"  形状: {X_F_onehot.shape}")
            
            if X_S.shape[1] > 0:
                print(f"\n[X_S 连续变量统计]")
                print(f"  min: {X_S.min(axis=0)}")
                print(f"  max: {X_S.max(axis=0)}")
                print(f"  mean: {X_S.mean(axis=0)}")
                print(f"  std: {X_S.std(axis=0)}")
            
            # 诊断2: f_Ta_coeff1 统计
            print(f"\n[f_Ta_coeff1 统计] (一次项系数)")
            print(f"  min: {f_Ta_coeff1.min():.4f}")
            print(f"  max: {f_Ta_coeff1.max():.4f}")
            print(f"  mean: {f_Ta_coeff1.mean():.4f}")
            print(f"  std: {f_Ta_coeff1.std():.4f}")
            
            # 诊断2b: f_Ta_coeff2 统计（如果存在）
            if f_Ta_coeff2 is not None:
                print(f"\n[f_Ta_coeff2 统计] (二次项系数)")
                print(f"  min: {f_Ta_coeff2.min():.6f}")
                print(f"  max: {f_Ta_coeff2.max():.6f}")
                print(f"  mean: {f_Ta_coeff2.mean():.6f}")
                print(f"  std: {f_Ta_coeff2.std():.6f}")
            
            # 诊断3: y_residual统计
            print(f"\n[y_residual 统计]")
            print(f"  min: {y_residual.min():.2f}")
            print(f"  max: {y_residual.max():.2f}")
            print(f"  mean: {y_residual.mean():.2f}")
            print(f"  std: {y_residual.std():.2f}")
            
            # 诊断4: 特征矩阵条件数（数值稳定性）
            X_list = []
            if X_F.shape[1] > 0:
                X_list.append(X_F)
            if X_S.shape[1] > 0:
                X_list.append(X_S)
            X_combined = np.hstack(X_list) if X_list else None
            
            if X_combined is not None and X_combined.shape[1] > 0:
                try:
                    cond_num = np.linalg.cond(X_combined)
                    print(f"\n[X_combined 条件数]")
                    print(f"  cond(X): {cond_num:.2e}")
                    if cond_num > 1e12:
                        print(f"  ⚠️  警告: 条件数很大，可能存在数值不稳定问题！")
                    elif cond_num > 1e10:
                        print(f"  ⚠️  注意: 条件数较大，建议标准化特征")
                except np.linalg.LinAlgError:
                    print(f"\n[X_combined 条件数]")
                    print(f"  ⚠️  无法计算条件数（矩阵可能奇异）")
            
            print("=" * 60)
        
        # ========== 特征标准化 ==========
        # 只标准化连续变量，保留one-hot编码不变
        X_F_scaled = X_F.copy()
        X_S_scaled = X_S.copy()
        
        # 标准化X_F中的连续变量
        if len(self.X_F_continuous_indices) > 0:
            self.scaler_F = StandardScaler()
            X_F_continuous = X_F[:, self.X_F_continuous_indices]
            X_F_continuous_scaled = self.scaler_F.fit_transform(X_F_continuous)
            X_F_scaled[:, self.X_F_continuous_indices] = X_F_continuous_scaled
            
            if verbose:
                print(f"\n[特征标准化]")
                print(f"  已标准化 X_F 中的 {len(self.X_F_continuous_indices)} 个连续变量")
                print(f"  保留 X_F 中的 {X_F.shape[1] - len(self.X_F_continuous_indices)} 个 one-hot 编码变量")
        
        # 标准化X_S（全部是连续变量）
        if X_S.shape[1] > 0:
            self.scaler_S = StandardScaler()
            X_S_scaled = self.scaler_S.fit_transform(X_S)
            
            if verbose:
                print(f"  已标准化 X_S 中的 {X_S.shape[1]} 个连续变量")
        
        # 使用标准化后的特征
        X_F = X_F_scaled
        X_S = X_S_scaled
        
        if verbose:
            # 标准化后的条件数检查
            X_list_scaled = []
            if X_F.shape[1] > 0:
                X_list_scaled.append(X_F)
            if X_S.shape[1] > 0:
                X_list_scaled.append(X_S)
            X_combined_scaled = np.hstack(X_list_scaled) if X_list_scaled else None
            
            if X_combined_scaled is not None and X_combined_scaled.shape[1] > 0:
                try:
                    cond_num_scaled = np.linalg.cond(X_combined_scaled)
                    print(f"  标准化后条件数: {cond_num_scaled:.2e}")
                    if 'cond_num' in locals() and cond_num_scaled < cond_num:
                        print(f"  ✓ 条件数已改善")
                except (np.linalg.LinAlgError, ValueError):
                    pass
            print("=" * 60)

        # 初始化空间交换系数
        lambda_coeff = 0.0
        W_Ta = np.zeros(n_districts)  # 邻域平均气温

        for iteration in range(max_iter):
            # Step 1: 固定Ta，拟合线性系数 α, β, λ
            X_list = []
            if X_F.shape[1] > 0:
                X_list.append(X_F)
            if X_S.shape[1] > 0:
                X_list.append(X_S)

            # 计算空间滞后项并加入特征矩阵
            if use_spatial:
                if spatial_mask is None:
                    # 全覆盖模式
                    W_Ta = spatial_weights @ Ta
                    X_spatial = (Ta - W_Ta).reshape(-1, 1)
                else:
                    # 部分覆盖模式：只对掩码内的样本计算空间滞后
                    Ta_covered = Ta[spatial_mask]
                    W_Ta_covered = spatial_weights @ Ta_covered
                    # 填充完整的 W_Ta 数组
                    W_Ta = np.zeros(n_districts)
                    W_Ta[spatial_mask] = W_Ta_covered
                    # 空间特征：未覆盖的样本设为 0（不参与空间交换）
                    X_spatial = np.zeros((n_districts, 1))
                    X_spatial[spatial_mask, 0] = Ta_covered - W_Ta_covered
                X_list.append(X_spatial)

            X_combined = np.hstack(X_list) if X_list else None

            # 构建目标: y - f(Ta)
            # f(Ta) = coeff2 × Ta² + coeff1 × Ta（如果有二次项）
            # f(Ta) = coeff1 × Ta（如果只有一次项）
            if f_Ta_coeff2 is not None:
                f_Ta_values = f_Ta_coeff2 * (Ta ** 2) + f_Ta_coeff1 * Ta
            else:
                f_Ta_values = f_Ta_coeff1 * Ta
            # 能量平衡: f(Ta) + linear_pred + y_residual = 0
            # 即: linear_pred = -y_residual - f(Ta)
            y_adjusted = -y_residual - f_Ta_values

            if X_combined is not None and X_combined.shape[1] > 0:
                # 使用岭回归或普通最小二乘
                if ridge_alpha > 0:
                    reg = Ridge(alpha=ridge_alpha, fit_intercept=False)
                else:
                    reg = LinearRegression(fit_intercept=False)
                reg.fit(X_combined, y_adjusted)
                coeffs = reg.coef_

                n_F = X_F.shape[1]
                n_S = X_S.shape[1]

                alpha = coeffs[:n_F] if n_F > 0 else np.array([])
                beta = coeffs[n_F:n_F+n_S] if n_S > 0 else np.array([])
                
                # 提取空间交换系数 λ
                if use_spatial:
                    lambda_coeff = coeffs[n_F + n_S]
                else:
                    lambda_coeff = 0.0

                # 计算不含空间项的线性预测（用于更新 Ta）
                other_linear_pred = np.zeros(n_districts)
                if n_F > 0:
                    other_linear_pred += X_F @ alpha
                if n_S > 0:
                    other_linear_pred += X_S @ beta
                
                linear_pred = X_combined @ coeffs
            else:
                alpha = np.array([])
                beta = np.array([])
                lambda_coeff = 0.0
                linear_pred = np.zeros(n_districts)
                other_linear_pred = np.zeros(n_districts)  # 修复：确保 other_linear_pred 被定义

            # Step 2: 固定α, β, λ，优化Ta
            # 求解能量平衡方程: f(Ta) + other_linear + λ×(Ta - W_Ta) + y_residual = 0
            # 展开后: coeff2×Ta² + (coeff1 + λ)×Ta + (other_linear - λ×W_Ta + y_residual) = 0
            Ta_new = np.zeros(n_districts)
            
            # 判断是否在空间覆盖范围内
            if spatial_mask is None:
                # 全覆盖或无空间效应
                is_covered = np.ones(n_districts, dtype=bool) if use_spatial else np.zeros(n_districts, dtype=bool)
            else:
                is_covered = spatial_mask

            # 诊断输出：检查空间效应（仅在第一轮迭代）
            if use_spatial and verbose and iteration == 0:
                n_covered = np.sum(is_covered)
                Ta_covered_vals = Ta[is_covered]
                W_Ta_covered_vals = W_Ta[is_covered]
                diff = Ta_covered_vals - W_Ta_covered_vals
                print(f"\n  [空间效应诊断]")
                print(f"    覆盖样本数: {n_covered}/{n_districts}")
                print(f"    Ta (covered): {Ta_covered_vals.mean():.2f}±{Ta_covered_vals.std():.2f} K")
                print(f"    W_Ta (covered): {W_Ta_covered_vals.mean():.2f}±{W_Ta_covered_vals.std():.2f} K")
                print(f"    Ta - W_Ta: {diff.mean():.4f}±{diff.std():.4f} K")
                print(f"    λ = {lambda_coeff:.4f}")
                print(f"    λ×(Ta-W_Ta): {(lambda_coeff * diff).mean():.4f}±{(lambda_coeff * diff).std():.4f}")
                print(f"    coeff1 均值: {f_Ta_coeff1[is_covered].mean():.4f}")
                print(f"    coeff1 + λ 均值: {(f_Ta_coeff1[is_covered] + lambda_coeff).mean():.4f}")
                print(f"    -λ×W_Ta 均值: {(-lambda_coeff * W_Ta_covered_vals).mean():.2f}")
                print()

            for i in range(n_districts):
                # 对于覆盖范围内的样本，λ 影响系数
                if use_spatial and is_covered[i]:
                    lambda_i = lambda_coeff
                    W_Ta_i = W_Ta[i]
                else:
                    lambda_i = 0.0
                    W_Ta_i = 0.0
                
                # 定义能量平衡目标函数
                if f_Ta_coeff2 is not None:
                    # 二次方程: a×Ta² + b×Ta + c = 0
                    # b 需要加上 λ，c 需要减去 λ×W_Ta
                    a = f_Ta_coeff2[i]
                    b = f_Ta_coeff1[i] + lambda_i  # 加入空间项对 Ta 系数的影响
                    c = other_linear_pred[i] - lambda_i * W_Ta_i + y_residual[i]  # 空间项的常数部分
                    
                    # 尝试使用求根公式
                    discriminant = b**2 - 4*a*c
                    if discriminant >= 0 and abs(a) > 1e-12:
                        # 选择物理合理的根（更接近初始值）
                        sqrt_disc = np.sqrt(discriminant)
                        ta1 = (-b + sqrt_disc) / (2*a)
                        ta2 = (-b - sqrt_disc) / (2*a)
                        
                        # 选择在合理范围内且更接近初始值的根
                        candidates = []
                        for ta_cand in [ta1, ta2]:
                            if 273.15 <= ta_cand <= 323.15:  # 0-50°C
                                candidates.append(ta_cand)
                        
                        if candidates:
                            # 选择更接近当前值的
                            ta_current = Ta[i]  # 捕获当前值避免闭包问题
                            Ta_new[i] = min(candidates, key=lambda x: abs(x - ta_current))
                        else:
                            # 无合理解，使用优化（传入修正后的系数）
                            Ta_new[i] = self._optimize_Ta_single(
                                Ta[i], a, b, other_linear_pred[i] - lambda_i * W_Ta_i, y_residual[i]
                            )
                    else:
                        # 判别式为负或a太小，使用优化
                        Ta_new[i] = self._optimize_Ta_single(
                            Ta[i], a, b, other_linear_pred[i] - lambda_i * W_Ta_i, y_residual[i]
                        )
                else:
                    # 一次方程: b×Ta + c = 0
                    # b 需要加上 λ，c 需要减去 λ×W_Ta
                    b = f_Ta_coeff1[i] + lambda_i  # 加入空间项对 Ta 系数的影响
                    c = other_linear_pred[i] - lambda_i * W_Ta_i + y_residual[i]  # 空间项的常数部分
                    
                    if abs(b) > 1e-12:
                        ta_solution = -c / b
                        # 约束在合理范围
                        Ta_new[i] = np.clip(ta_solution, 273.15, 323.15)
                    else:
                        Ta_new[i] = Ta[i]  # 保持不变

            # 检查收敛
            delta_Ta = np.abs(Ta_new - Ta).max()
            Ta = Ta_new

            # 计算残差（能量平衡误差）
            # 理想情况: f(Ta) + linear_pred + y_residual = 0
            if f_Ta_coeff2 is not None:
                f_Ta_total = f_Ta_coeff2 * (Ta ** 2) + f_Ta_coeff1 * Ta
            else:
                f_Ta_total = f_Ta_coeff1 * Ta
            pred_total = f_Ta_total + linear_pred
            energy_balance_error = pred_total + y_residual  # 应接近 0
            residual_norm = np.linalg.norm(energy_balance_error)
            rmse = residual_norm / np.sqrt(len(energy_balance_error))  # 均方根误差

            # 记录历史
            self.history.append({
                'iteration': iteration,
                'delta_Ta': delta_Ta,
                'residual_norm': residual_norm,
                'rmse': rmse,
                'Ta_mean': Ta.mean(),
                'Ta_std': Ta.std()
            })

            if verbose:
                # 计算线性项和Ta项的贡献（用于诊断）
                linear_contribution = np.abs(linear_pred).mean() if linear_pred.size > 0 else 0
                ta_contribution = np.abs(f_Ta_total).mean()
                spatial_info = f", λ={lambda_coeff:.4f}" if use_spatial else ""
                print(f"Iter {iteration+1:2d}: "
                      f"ΔTa={delta_Ta:.4f} K, "
                      f"RMSE={rmse:.2f} W/m², "
                      f"Ta={Ta.mean():.2f}±{Ta.std():.2f} K, "
                      f"线性项贡献={linear_contribution:.2f}, Ta项贡献={ta_contribution:.2f}"
                      f"{spatial_info}")

            if delta_Ta < tol:
                if verbose:
                    print(f"\n收敛！ΔTa < {tol}")
                converged = True
                break
        else:
            converged = False
            if verbose:
                print(f"\n达到最大迭代次数 {max_iter}")

        if verbose:
            print("=" * 60)
            if use_spatial:
                print(f"\n空间交换系数 λ = {lambda_coeff:.4f}")
                if lambda_coeff > 0:
                    print(f"  物理意义: 气温趋向于邻域平均（热量扩散）")
                elif lambda_coeff < 0:
                    print(f"  物理意义: 气温偏离邻域平均（异常）")

        # 保存结果
        self.Ta_per_district = Ta
        self.alpha_coeffs = alpha
        self.beta_coeffs = beta
        self.lambda_coeff = lambda_coeff if use_spatial else None
        self.spatial_lag = W_Ta if use_spatial else None

        result = {
            'Ta_per_district': Ta,
            'alpha_coeffs': alpha,
            'beta_coeffs': beta,
            'converged': converged,
            'n_iter': iteration + 1 if converged else max_iter,
            'residual_norm': residual_norm,
            'rmse': rmse
        }
        
        # 添加空间相关结果
        if use_spatial:
            result['lambda_coeff'] = lambda_coeff
            result['spatial_lag'] = W_Ta  # 邻域平均气温
        
        return result

    def predict(
        self,
        X_F: np.ndarray,
        X_S: np.ndarray,
        f_Ta_coeff1: Optional[np.ndarray] = None,
        f_Ta_coeff2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """使用拟合的模型进行预测
        
        参数:
            X_F: 人为热特征矩阵
            X_S: 储热特征矩阵
            f_Ta_coeff1: Ta一次项系数（可选，默认使用训练时的系数）
            f_Ta_coeff2: Ta二次项系数（可选，默认使用训练时的系数）
        """
        if self.Ta_per_district is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        # 使用训练时保存的系数（如果未提供）
        if f_Ta_coeff1 is None:
            f_Ta_coeff1 = self.f_Ta_coeff1
        if f_Ta_coeff2 is None:
            f_Ta_coeff2 = self.f_Ta_coeff2

        # 标准化特征（与训练时一致）
        X_F_scaled = X_F.copy()
        X_S_scaled = X_S.copy()
        
        # 标准化X_F中的连续变量
        if self.scaler_F is not None and len(self.X_F_continuous_indices) > 0:
            X_F_continuous = X_F[:, self.X_F_continuous_indices]
            X_F_continuous_scaled = self.scaler_F.transform(X_F_continuous)
            X_F_scaled[:, self.X_F_continuous_indices] = X_F_continuous_scaled
        
        # 标准化X_S
        if self.scaler_S is not None and X_S.shape[1] > 0:
            X_S_scaled = self.scaler_S.transform(X_S)

        # 计算 f(Ta)
        if f_Ta_coeff2 is not None:
            pred = f_Ta_coeff2 * (self.Ta_per_district ** 2) + f_Ta_coeff1 * self.Ta_per_district
        else:
            pred = f_Ta_coeff1 * self.Ta_per_district

        if self.alpha_coeffs is not None and self.alpha_coeffs.size > 0:
            pred += X_F_scaled @ self.alpha_coeffs

        if self.beta_coeffs is not None and self.beta_coeffs.size > 0:
            pred += X_S_scaled @ self.beta_coeffs

        return pred

    def get_results_dataframe(
        self,
        districts_gdf: gpd.GeoDataFrame,  # noqa: ARG002 - 保留用于将来扩展
        X_F_columns: Optional[List[str]] = None,
        X_S_columns: Optional[List[str]] = None,
        spatial_analysis: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        将回归结果整理为DataFrame

        参数:
            districts_gdf: 街区GeoDataFrame（保留用于将来添加几何信息）
            X_F_columns: X_F特征名称列表
            X_S_columns: X_S特征名称列表
            spatial_analysis: 空间自相关分析结果

        返回:
            DataFrame: 每个街区的结果
        """
        if self.Ta_per_district is None:
            raise ValueError("模型尚未拟合")

        # 使用有效样本的 district_id
        if self.valid_district_ids is not None:
            district_ids = self.valid_district_ids
        else:
            # 向后兼容：如果没有保存 valid_district_ids，使用连续索引
            district_ids = range(len(self.Ta_per_district))

        result = pd.DataFrame({
            'district_id': district_ids,
            'Ta_optimized': self.Ta_per_district,
            'Ta_celsius': self.Ta_per_district - 273.15
        })

        # 添加人为热系数
        if self.alpha_coeffs is not None and self.alpha_coeffs.size > 0:
            for i, coeff in enumerate(self.alpha_coeffs):
                col_name = X_F_columns[i] if X_F_columns and i < len(X_F_columns) else f'alpha_{i}'
                result[f'coeff_F_{col_name}'] = coeff

        # 添加储热系数
        if self.beta_coeffs is not None and self.beta_coeffs.size > 0:
            for i, coeff in enumerate(self.beta_coeffs):
                col_name = X_S_columns[i] if X_S_columns and i < len(X_S_columns) else f'beta_{i}'
                result[f'coeff_S_{col_name}'] = coeff

        # 添加空间分析结果
        if spatial_analysis is not None:
            result['spatial_rho'] = spatial_analysis.get('rho', np.nan)
            result['spatial_r_squared'] = spatial_analysis.get('r_squared', np.nan)
            result['spatial_moran_i'] = spatial_analysis.get('moran_i', np.nan)
            result['spatial_moran_p'] = spatial_analysis.get('moran_p', np.nan)
            
            if 'spatial_lag' in spatial_analysis:
                result['spatial_lag_Ta'] = spatial_analysis['spatial_lag']
            
            if 'residuals' in spatial_analysis:
                result['spatial_residual'] = spatial_analysis['residuals']

        return result

