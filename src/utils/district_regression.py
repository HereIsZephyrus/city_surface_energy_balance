"""
街区级回归分析模块

提供从栅格能量平衡结果到街区级回归分析的完整工具链。

核心功能:
    1. 栅格到街区的空间聚合
    2. 回归数据准备（线性项设计矩阵）
    3. 交替最小二乘法（ALS）求解
    4. 结果导出

物理模型:
    f^k(Ta) + Σαi·X^k_Fi + Σβi·X^k_Si = C^k

    其中:
    - f(Ta): 非线性项（能量平衡，关于Ta的函数）
    - X_Fi: 人为热相关线性项（LCZ类型、不透水面面积、人口等）
    - X_Si: 建筑储热相关线性项（建筑体积、LCZ类型、植被覆盖度等）
    - αi, βi: 待求回归系数
    - Ta: 每个街区的气温（待优化）

参考文献:
    doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import warnings

try:
    from rasterio import features
    from rasterio.transform import from_bounds
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    warnings.warn("未安装rasterio，栅格处理功能受限")

try:
    from rasterstats import zonal_stats
    HAS_RASTERSTATS = True
except ImportError:
    HAS_RASTERSTATS = False
    warnings.warn("未安装rasterstats，建议安装: pip install rasterstats")


class DistrictAggregator:
    """
    街区聚合器

    将栅格数据聚合到街区多边形
    """

    @staticmethod
    def aggregate_rasters_to_districts(
        rasters: Dict[str, Union[np.ndarray, xr.DataArray]],
        districts_gdf: gpd.GeoDataFrame,
        transform: 'rasterio.Affine',
        stats: List[str] = ['mean'],
        nodata: Optional[float] = None
    ) -> pd.DataFrame:
        """
        将多个栅格聚合到街区

        参数:
            rasters: 栅格字典 {name: array}
            districts_gdf: 街区GeoDataFrame
            transform: 栅格的仿射变换
            stats: 聚合统计量列表，例如 ['mean', 'std', 'min', 'max']
            nodata: 无数据值

        返回:
            DataFrame: 每个街区的聚合统计量
                columns: [district_id, var1_mean, var1_std, var2_mean, ...]
        """
        if not HAS_RASTERSTATS:
            return DistrictAggregator._aggregate_manual(
                rasters, districts_gdf, transform, stats, nodata
            )

        # 准备结果
        results = []

        # 为每个街区创建ID
        if 'district_id' not in districts_gdf.columns:
            districts_gdf = districts_gdf.copy()
            districts_gdf['district_id'] = range(len(districts_gdf))

        # 对每个栅格进行聚合
        for var_name, raster in rasters.items():
            # 确保是numpy数组
            if isinstance(raster, xr.DataArray):
                raster = raster.values

            # 使用rasterstats进行zonal statistics
            zs = zonal_stats(
                districts_gdf.geometry,
                raster,
                affine=transform,
                stats=stats,
                nodata=nodata
            )

            # 转换为DataFrame
            df = pd.DataFrame(zs)
            # 重命名列
            df.columns = [f"{var_name}_{stat}" for stat in stats]

            results.append(df)

        # 合并所有结果
        result_df = pd.concat(results, axis=1)
        result_df['district_id'] = districts_gdf['district_id'].values

        # 将district_id移到第一列
        cols = ['district_id'] + [c for c in result_df.columns if c != 'district_id']
        result_df = result_df[cols]

        return result_df

    @staticmethod
    def _aggregate_manual(
        rasters: Dict[str, Union[np.ndarray, xr.DataArray]],
        districts_gdf: gpd.GeoDataFrame,
        transform: 'rasterio.Affine',
        stats: List[str],
        nodata: Optional[float]
    ) -> pd.DataFrame:
        """
        手动实现的聚合（rasterstats不可用时的备份方案）
        """
        if not HAS_RASTERIO:
            raise ImportError("需要安装rasterio: pip install rasterio")

        # 为每个街区创建mask
        results = []

        for idx, district in districts_gdf.iterrows():
            district_stats = {'district_id': idx}

            # 创建该街区的mask
            shape = next(iter(rasters.values())).shape
            mask = features.geometry_mask(
                [district.geometry],
                transform=transform,
                out_shape=shape,
                invert=True
            )

            # 对每个栅格计算统计量
            for var_name, raster in rasters.items():
                if isinstance(raster, xr.DataArray):
                    raster = raster.values

                # 提取该街区的值
                values = raster[mask]

                # 去除nodata
                if nodata is not None:
                    values = values[values != nodata]
                values = values[~np.isnan(values)]

                if len(values) == 0:
                    for stat in stats:
                        district_stats[f"{var_name}_{stat}"] = np.nan
                else:
                    for stat in stats:
                        if stat == 'mean':
                            district_stats[f"{var_name}_{stat}"] = np.mean(values)
                        elif stat == 'std':
                            district_stats[f"{var_name}_{stat}"] = np.std(values)
                        elif stat == 'min':
                            district_stats[f"{var_name}_{stat}"] = np.min(values)
                        elif stat == 'max':
                            district_stats[f"{var_name}_{stat}"] = np.max(values)
                        elif stat == 'median':
                            district_stats[f"{var_name}_{stat}"] = np.median(values)
                        elif stat == 'sum':
                            district_stats[f"{var_name}_{stat}"] = np.sum(values)

            results.append(district_stats)

        return pd.DataFrame(results)


class DistrictRegressionModel:
    """
    街区级回归模型

    实现基于能量平衡的交替最小二乘法（ALS）回归
    """

    def __init__(self):
        """初始化回归模型"""
        self.alpha_coeffs = None  # X_Fi的系数
        self.beta_coeffs = None   # X_Si的系数
        self.Ta_per_district = None  # 每个街区的气温
        self.history = []  # 迭代历史

    def prepare_regression_data(
        self,
        aggregated_df: pd.DataFrame,
        districts_gdf: gpd.GeoDataFrame,
        f_Ta_column: str = 'f_Ta_coeff_mean',
        residual_column: str = 'residual_mean',
        Ts_column: str = 'surface_temperature_mean',
        X_F_columns: Optional[List[str]] = None,
        X_S_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备回归数据

        从聚合后的DataFrame和街区属性构建回归矩阵

        参数:
            aggregated_df: 从栅格聚合的DataFrame
            districts_gdf: 街区GeoDataFrame（包含属性字段）
            f_Ta_column: Ta系数栏位名
            residual_column: 残差栏位名
            X_F_columns: 人为热相关特征列表
                        例如: ['lcz_type', 'impervious_area', 'population']
            X_S_columns: 建筑储热相关特征列表
                        例如: ['building_volume', 'lcz_type', 'fvc_mean']

        返回:
            (X_F, X_S, f_Ta_coeffs, y_residual, Ts_mean):
            - X_F: 人为热特征矩阵 (n_districts, n_F_features)
            - X_S: 建筑储热特征矩阵 (n_districts, n_S_features)
            - f_Ta_coeffs: Ta系数向量 (n_districts,)
            - y_residual: 残差向量 (n_districts,)
            - Ts_mean: 各街区平均地表温度 (n_districts,) - 用于Ta初始化
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

        # 提取地表温度（用于Ta初始化）
        Ts_mean = merged[Ts_column].values if Ts_column in merged.columns else None

        # 构建X_F矩阵
        if X_F_columns:
            X_F = self._build_feature_matrix(merged, X_F_columns)
        else:
            # 默认使用空矩阵（no X_F features）
            X_F = np.zeros((len(merged), 0))

        # 构建X_S矩阵
        if X_S_columns:
            X_S = self._build_feature_matrix(merged, X_S_columns)
        else:
            # 默认使用空矩阵
            X_S = np.zeros((len(merged), 0))

        return X_F, X_S, f_Ta_coeffs, y_residual, Ts_mean

    def _build_feature_matrix(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> np.ndarray:
        """
        构建特征矩阵，支持类别变量的one-hot编码

        参数:
            df: 数据DataFrame
            columns: 特征列名列表

        返回:
            特征矩阵 (n_samples, n_features)
        """
        X_list = []

        for col in columns:
            if col not in df.columns:
                warnings.warn(f"列'{col}'不存在，跳过")
                continue

            values = df[col].values

            # 检查是否是分类变量
            if pd.api.types.is_categorical_dtype(values) or pd.api.types.is_object_dtype(values):
                # One-hot编码
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                X_list.append(dummies.values)
            else:
                # 数值变量
                X_list.append(values.reshape(-1, 1))

        if X_list:
            return np.hstack(X_list)
        else:
            return np.zeros((len(df), 0))

    def fit_als_regression(
        self,
        X_F: np.ndarray,
        X_S: np.ndarray,
        f_Ta_coeffs: np.ndarray,
        y_residual: np.ndarray,
        Ts_mean: Optional[np.ndarray] = None,
        max_iter: int = 20,
        tol: float = 1e-4,
        verbose: bool = True
    ) -> Dict:
        """
        交替最小二乘法（ALS）求解

        求解方程: f(Ta) + Σαi·X_Fi + Σβi·X_Si = C

        算法:
            1. 初始化: 假设 dT = Ta - Ts = const (例如 -5K)
            2. 固定 Ta，拟合线性项 α, β
            3. 固定 α, β，优化各街区的 Ta
            4. 重复2-3直到收敛

        参数:
            X_F: 人为热特征矩阵 (n, p_F)
            X_S: 建筑储热特征矩阵 (n, p_S)
            f_Ta_coeffs: Ta系数 (n,) - f(Ta)对Ta的偏导数
            y_residual: 观测残差 (n,) - 不包含Ta项的能量收支
            Ts_mean: 各街区的平均地表温度 (n,) - 用于Ta初始化
            max_iter: 最大迭代次数
            tol: 收敛容差
            verbose: 是否打印迭代信息

        返回:
            结果字典:
            {
                'Ta_per_district': 每个街区的气温 (n,),
                'alpha_coeffs': X_F的系数 (p_F,),
                'beta_coeffs': X_S的系数 (p_S,),
                'converged': 是否收敛,
                'n_iter': 迭代次数,
                'residual_norm': 最终残差范数
            }
        """
        n_districts = len(y_residual)

        # 初始化Ta
        if Ts_mean is not None:
            Ta = Ts_mean - 5.0  # 假设气温比地表温度低5K
        else:
            Ta = np.full(n_districts, 298.15)  # 默认25°C

        self.history = []

        if verbose:
            print("\n开始ALS回归迭代...")
            print("=" * 60)

        for iteration in range(max_iter):
            # Step 1: 固定Ta，拟合线性系数 α, β
            # 构建增广特征矩阵 X = [X_F, X_S]
            X_combined = np.hstack([X_F, X_S]) if X_F.shape[1] > 0 or X_S.shape[1] > 0 else None

            # 构建目标: y - f(Ta)
            y_adjusted = y_residual - f_Ta_coeffs * Ta

            if X_combined is not None and X_combined.shape[1] > 0:
                # 线性回归
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X_combined, y_adjusted)
                coeffs = reg.coef_

                # 分离α和β
                n_F = X_F.shape[1]
                alpha = coeffs[:n_F] if n_F > 0 else np.array([])
                beta = coeffs[n_F:] if X_S.shape[1] > 0 else np.array([])

                # 计算线性项预测
                linear_pred = X_combined @ coeffs
            else:
                alpha = np.array([])
                beta = np.array([])
                linear_pred = np.zeros(n_districts)

            # Step 2: 固定α, β，优化Ta
            # 对每个街区单独优化
            Ta_new = np.zeros(n_districts)

            for i in range(n_districts):
                def objective(ta):
                    # f(Ta) + linear_terms - residual = 0
                    # 最小化: (f(Ta) + linear_terms - residual)^2
                    pred = f_Ta_coeffs[i] * ta + linear_pred[i]
                    return (pred - y_residual[i]) ** 2

                # 优化Ta（约束在合理范围）
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

            # 收敛检查
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
        """
        使用拟合的模型进行预测

        参数:
            X_F, X_S: 特征矩阵
            f_Ta_coeffs: Ta系数

        返回:
            预测的能量收支 (n,)
        """
        if self.Ta_per_district is None:
            raise ValueError("模型尚未拟合，请先调用fit_als_regression()")

        pred = f_Ta_coeffs * self.Ta_per_district

        if self.alpha_coeffs.size > 0:
            pred += X_F @ self.alpha_coeffs

        if self.beta_coeffs.size > 0:
            pred += X_S @ self.beta_coeffs

        return pred

    def get_results_dataframe(
        self,
        districts_gdf: gpd.GeoDataFrame,
        X_F_columns: Optional[List[str]] = None,
        X_S_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        将回归结果整理为DataFrame

        参数:
            districts_gdf: 街区GeoDataFrame
            X_F_columns: X_F特征名称列表
            X_S_columns: X_S特征名称列表

        返回:
            DataFrame: 每个街区的结果
                columns: [district_id, Ta, alpha_1, ..., beta_1, ...]
        """
        if self.Ta_per_district is None:
            raise ValueError("模型尚未拟合")

        result = pd.DataFrame({
            'district_id': range(len(self.Ta_per_district)),
            'Ta_optimized': self.Ta_per_district,
            'Ta_celsius': self.Ta_per_district - 273.15
        })

        # 添加回归系数（作为常数列）
        if self.alpha_coeffs.size > 0:
            for i, coeff in enumerate(self.alpha_coeffs):
                col_name = X_F_columns[i] if X_F_columns and i < len(X_F_columns) else f'alpha_{i}'
                result[f'coeff_F_{col_name}'] = coeff

        if self.beta_coeffs.size > 0:
            for i, coeff in enumerate(self.beta_coeffs):
                col_name = X_S_columns[i] if X_S_columns and i < len(X_S_columns) else f'beta_{i}'
                result[f'coeff_S_{col_name}'] = coeff

        return result


def example_usage():
    """使用示例"""
    print("""
    # 街区回归分析使用示例

    from utils.district_regression import DistrictAggregator, DistrictRegressionModel
    from radiation import calculate_energy_balance_coefficients
    import geopandas as gpd
    import numpy as np

    # Step 1: 计算能量平衡系数（栅格层面）
    coeffs = calculate_energy_balance_coefficients(
        shortwave_down=S_down,
        surface_temperature=Ts,
        elevation=elevation,
        albedo=albedo,
        ndvi=ndvi,
        saturation_vapor_pressure=es,
        actual_vapor_pressure=ea,
        aerodynamic_resistance=rah,
        surface_resistance=rs,
        surface_emissivity=epsilon_0,
        surface_pressure=P,
        ta_reference=298.15  # 参考气温
    )

    # Step 2: 聚合栅格到街区
    aggregator = DistrictAggregator()

    rasters = {
        'f_Ta_coeff': coeffs['f_Ta_coeff'],
        'residual': coeffs['residual'],
        'surface_temperature': coeffs['surface_temperature']
    }

    aggregated_df = aggregator.aggregate_rasters_to_districts(
        rasters=rasters,
        districts_gdf=districts,
        transform=transform,
        stats=['mean']
    )

    # Step 3: 准备回归数据
    model = DistrictRegressionModel()

    X_F, X_S, f_Ta_coeffs, y_residual, Ts_mean = model.prepare_regression_data(
        aggregated_df=aggregated_df,
        districts_gdf=districts,
        f_Ta_column='f_Ta_coeff_mean',
        residual_column='residual_mean',
        Ts_column='surface_temperature_mean',
        X_F_columns=['lcz_type', 'impervious_area', 'population'],
        X_S_columns=['building_volume', 'fvc_mean']
    )

    # Step 4: ALS回归（求解每个街区的Ta）
    results = model.fit_als_regression(
        X_F=X_F,
        X_S=X_S,
        f_Ta_coeffs=f_Ta_coeffs,
        y_residual=y_residual,
        Ts_mean=Ts_mean,
        max_iter=20,
        verbose=True
    )

    # Step 5: 获取结果
    results_df = model.get_results_dataframe(districts)
    print(results_df)
    # 每个街区一个Ta值！
    """)


if __name__ == "__main__":
    example_usage()

