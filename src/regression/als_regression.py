"""
交替最小二乘法（ALS）回归模块

实现基于能量平衡的 ALS 回归，求解各街区气温。

物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si = 0

    其中:
    - f(Ta): 能量平衡系数（关于Ta的函数）
    - X_Fi: 人为热 Q_F 相关特征（与 Ta 独立）
    - X_Si: 建筑储热 ΔQ_Sb 相关特征（与 Ta 独立）
    - αi, βi: 待求回归系数
    - Ta: 每个街区的气温（待优化）

算法:
    1. 初始化: 使用ERA5气温作为Ta初值
    2. 固定 Ta，拟合线性项 α, β
    3. 固定 α, β，优化各街区的 Ta
    4. 重复2-3直到收敛
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
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
        self.Ta_per_district = None  # 每个街区的气温
        self.history = []  # 迭代历史
        self.valid_district_ids = None  # 有效样本的 district_id（过滤后）

    def prepare_regression_data(
        self,
        aggregated_df: pd.DataFrame,
        districts_gdf: gpd.GeoDataFrame,
        f_Ta_column: str = 'f_Ta_coeff_mean',
        residual_column: str = 'residual_mean',
        era5_Ta_column: str = 'era5_air_temperature_mean',
        X_F_columns: Optional[List[str]] = None,
        X_S_columns: Optional[List[str]] = None,
        X_C_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备回归数据

        从聚合后的DataFrame和街区属性构建回归矩阵

        参数:
            aggregated_df: 从栅格聚合的DataFrame
            districts_gdf: 街区GeoDataFrame（包含属性字段）
            f_Ta_column: Ta系数栏位名
            residual_column: 残差栏位名
            era5_Ta_column: ERA5气温栏位名 - 用于Ta初始化
            X_F_columns: 人为热 Q_F 相关特征列表（连续变量）
            X_S_columns: 建筑储热 ΔQ_Sb 相关特征列表（连续变量）
            X_C_columns: 分类特征列表（将进行 one-hot 编码）

        返回:
            (X_F, X_S, f_Ta_coeffs, y_residual, era5_Ta_mean)
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
        f_Ta_coeffs = merged[f_Ta_column].values
        y_residual = merged[residual_column].values

        # 提取ERA5气温（用于Ta初始化）
        era5_Ta_mean = merged[era5_Ta_column].values if era5_Ta_column in merged.columns else None

        # 构建有效样本掩码
        # 1. 过滤 NaN 值
        valid_mask = (
            ~np.isnan(f_Ta_coeffs) &
            ~np.isnan(y_residual)
        )
        
        # 2. 如果 era5_Ta_mean 存在，也要检查其 NaN
        if era5_Ta_mean is not None:
            valid_mask = valid_mask & ~np.isnan(era5_Ta_mean)
        
        # 3. 约束：landuse > 0
        if 'landuse' in merged.columns:
            landuse = merged['landuse'].values
            valid_mask = valid_mask & (landuse > 0)
        
        # 应用掩码过滤有效样本
        if not np.all(valid_mask):
            n_filtered = np.sum(~valid_mask)
            print(f"  过滤掉 {n_filtered} 个无效样本（NaN 或 land_type <= 0）")
            print(f"  有效样本数: {np.sum(valid_mask)} / {len(valid_mask)}")
        
        merged = merged[valid_mask].copy()
        f_Ta_coeffs = f_Ta_coeffs[valid_mask]
        y_residual = y_residual[valid_mask]
        if era5_Ta_mean is not None:
            era5_Ta_mean = era5_Ta_mean[valid_mask]
        
        # 保存有效样本的 district_id
        self.valid_district_ids = merged['district_id'].values

        # 确定需要 one-hot 编码的列
        categorical_columns = set(X_C_columns) if X_C_columns else set()

        # 构建X_F矩阵（连续变量）
        if X_F_columns:
            X_F = self._build_feature_matrix(merged, X_F_columns, categorical_columns)
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

        # 最终检查：确保特征矩阵中没有 NaN（额外安全措施）
        if X_F.shape[1] > 0:
            X_F_has_nan = np.isnan(X_F).any(axis=1)
            if X_F_has_nan.any():
                print(f"  警告: X_F 特征矩阵中发现 NaN，将过滤 {np.sum(X_F_has_nan)} 个样本")
                valid_mask_final = ~X_F_has_nan
                merged = merged[valid_mask_final].copy()
                X_F = X_F[valid_mask_final]
                X_S = X_S[valid_mask_final]
                f_Ta_coeffs = f_Ta_coeffs[valid_mask_final]
                y_residual = y_residual[valid_mask_final]
                if era5_Ta_mean is not None:
                    era5_Ta_mean = era5_Ta_mean[valid_mask_final]
                self.valid_district_ids = merged['district_id'].values
        
        if X_S.shape[1] > 0:
            X_S_has_nan = np.isnan(X_S).any(axis=1)
            if X_S_has_nan.any():
                print(f"  警告: X_S 特征矩阵中发现 NaN，将过滤 {np.sum(X_S_has_nan)} 个样本")
                valid_mask_final = ~X_S_has_nan
                merged = merged[valid_mask_final].copy()
                X_F = X_F[valid_mask_final]
                X_S = X_S[valid_mask_final]
                f_Ta_coeffs = f_Ta_coeffs[valid_mask_final]
                y_residual = y_residual[valid_mask_final]
                if era5_Ta_mean is not None:
                    era5_Ta_mean = era5_Ta_mean[valid_mask_final]
                self.valid_district_ids = merged['district_id'].values

        return X_F, X_S, f_Ta_coeffs, y_residual, era5_Ta_mean

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
        """
        X_list = []

        for col in columns:
            if col not in df.columns:
                warnings.warn(f"分类列'{col}'不存在，跳过")
                continue

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
        f_Ta_coeffs: np.ndarray,
        y_residual: np.ndarray,
        era5_Ta_mean: Optional[np.ndarray] = None,
        max_iter: int = 20,
        tol: float = 1e-4,
        verbose: bool = True
    ) -> Dict:
        """
        交替最小二乘法（ALS）求解

        求解方程: f(Ta) + Σαi·X_Fi + Σβi·X_Si = 0

        参数:
            X_F: 人为热 Q_F 特征矩阵 (n, p_F)
            X_S: 建筑储热 ΔQ_Sb 特征矩阵 (n, p_S)
            f_Ta_coeffs: Ta系数 (n,)
            y_residual: 观测残差 (n,)
            era5_Ta_mean: 各街区的ERA5气温平均 (n,) - 用于Ta初始化
            max_iter: 最大迭代次数
            tol: 收敛容差
            verbose: 是否打印迭代信息

        返回:
            结果字典
        """
        n_districts = len(y_residual)

        # 初始化Ta（使用ERA5气温）
        if era5_Ta_mean is not None:
            Ta = era5_Ta_mean.copy()
        else:
            Ta = np.full(n_districts, 298.15)  # 默认25°C

        self.history = []

        if verbose:
            print("\n开始ALS回归迭代...")
            print("=" * 60)
            print(f"特征维度: X_F={X_F.shape[1]}, X_S={X_S.shape[1]}")

        for iteration in range(max_iter):
            # Step 1: 固定Ta，拟合线性系数 α, β
            X_list = []
            if X_F.shape[1] > 0:
                X_list.append(X_F)
            if X_S.shape[1] > 0:
                X_list.append(X_S)

            X_combined = np.hstack(X_list) if X_list else None

            # 构建目标: y - f(Ta)
            y_adjusted = y_residual - f_Ta_coeffs * Ta

            if X_combined is not None and X_combined.shape[1] > 0:
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X_combined, y_adjusted)
                coeffs = reg.coef_

                n_F = X_F.shape[1]
                n_S = X_S.shape[1]

                alpha = coeffs[:n_F] if n_F > 0 else np.array([])
                beta = coeffs[n_F:n_F+n_S] if n_S > 0 else np.array([])

                linear_pred = X_combined @ coeffs
            else:
                alpha = np.array([])
                beta = np.array([])
                linear_pred = np.zeros(n_districts)

            # Step 2: 固定α, β，优化Ta
            Ta_new = np.zeros(n_districts)

            for i in range(n_districts):
                def objective(ta):
                    pred = f_Ta_coeffs[i] * ta + linear_pred[i]
                    return (pred - y_residual[i]) ** 2

                bounds = [(273.15, 323.15)]  # 0°C到50°C
                result = minimize(
                    objective,
                    x0=Ta[i],
                    method='L-BFGS-B',
                    bounds=bounds
                )
                Ta_new[i] = result.x[0]

            # 检查收敛
            delta_Ta = np.abs(Ta_new - Ta).max()
            Ta = Ta_new

            # 计算残差
            pred_total = f_Ta_coeffs * Ta + linear_pred
            residual = y_residual - pred_total
            residual_norm = np.linalg.norm(residual)

            # 记录历史
            self.history.append({
                'iteration': iteration,
                'delta_Ta': delta_Ta,
                'residual_norm': residual_norm,
                'Ta_mean': Ta.mean(),
                'Ta_std': Ta.std()
            })

            if verbose:
                print(f"Iter {iteration+1:2d}: "
                      f"ΔTa={delta_Ta:.4f} K, "
                      f"residual={residual_norm:.2f}, "
                      f"Ta={Ta.mean():.2f}±{Ta.std():.2f} K")

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

        # 保存结果
        self.Ta_per_district = Ta
        self.alpha_coeffs = alpha
        self.beta_coeffs = beta

        return {
            'Ta_per_district': Ta,
            'alpha_coeffs': alpha,
            'beta_coeffs': beta,
            'converged': converged,
            'n_iter': iteration + 1 if converged else max_iter,
            'residual_norm': residual_norm
        }

    def predict(
        self,
        X_F: np.ndarray,
        X_S: np.ndarray,
        f_Ta_coeffs: np.ndarray
    ) -> np.ndarray:
        """使用拟合的模型进行预测"""
        if self.Ta_per_district is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        pred = f_Ta_coeffs * self.Ta_per_district

        if self.alpha_coeffs is not None and self.alpha_coeffs.size > 0:
            pred += X_F @ self.alpha_coeffs

        if self.beta_coeffs is not None and self.beta_coeffs.size > 0:
            pred += X_S @ self.beta_coeffs

        return pred

    def get_results_dataframe(
        self,
        districts_gdf: gpd.GeoDataFrame,
        X_F_columns: Optional[List[str]] = None,
        X_S_columns: Optional[List[str]] = None,
        spatial_analysis: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        将回归结果整理为DataFrame

        参数:
            districts_gdf: 街区GeoDataFrame
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

