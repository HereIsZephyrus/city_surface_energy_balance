"""
反事实分析模块（Counterfactual Analysis）

通过情景模拟估算自然景观对气温的降温贡献。

方法论:
    1. 使用ALS回归得到原始气温 Ta_original
    2. 将自然景观LCZ替换为建成区LCZ，重新求解 Ta_counterfactual
    3. 降温贡献 = Ta_counterfactual - Ta_original

物理意义:
    如果某块自然景观（绿地、水体等）变成建成区，气温会升高多少？
    这个升温量即为该自然景观的降温贡献。

参考文献:
    Stewart, I. D., & Oke, T. R. (2012). Local Climate Zones for urban
    temperature studies. Bulletin of the American Meteorological Society.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import warnings

from ..regression.als_regression import ALSRegression
from .constants import (
    LCZ_NATURAL_NUMERIC,
    LCZ_NATURAL_STRING,
    LCZ_NATURAL_NAMES,
)
from .spatial_analysis import SpatialWeightMatrix, analyze_spatial_autocorrelation


@dataclass
class CounterfactualResult:
    """反事实分析结果"""
    
    Ta_original: np.ndarray          # 原始气温 (K)
    Ta_counterfactual: np.ndarray    # 反事实气温 (K)
    cooling_contribution: np.ndarray  # 降温贡献 (K)，正值表示降温
    natural_mask: np.ndarray          # 自然景观街区掩膜
    lcz_original: np.ndarray          # 原始LCZ值
    lcz_counterfactual: np.ndarray    # 反事实LCZ值
    
    # 统计摘要
    n_total: int                      # 总街区数
    n_natural: int                    # 自然景观街区数
    mean_cooling: float               # 自然景观平均降温 (K)
    max_cooling: float                # 最大降温 (K)
    std_cooling: float                # 降温标准差 (K)
    mean_cooling_all: float           # 全域平均降温 (K)
    
    # 空间分析结果（可选）
    spatial_analysis_original: Optional[Dict] = None
    spatial_analysis_counterfactual: Optional[Dict] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame({
            'Ta_original_K': self.Ta_original,
            'Ta_original_C': self.Ta_original - 273.15,
            'Ta_counterfactual_K': self.Ta_counterfactual,
            'Ta_counterfactual_C': self.Ta_counterfactual - 273.15,
            'cooling_contribution_K': self.cooling_contribution,
            'is_natural': self.natural_mask,
            'lcz_original': self.lcz_original,
            'lcz_counterfactual': self.lcz_counterfactual,
        })
    
    def summary(self) -> str:
        """生成摘要报告"""
        lines = [
            "=" * 60,
            "反事实分析结果摘要",
            "=" * 60,
            f"总街区数: {self.n_total}",
            f"自然景观街区数: {self.n_natural} ({100*self.n_natural/self.n_total:.1f}%)",
            "",
            "自然景观降温贡献:",
            f"  平均降温: {self.mean_cooling:.2f} K ({self.mean_cooling:.2f}°C)",
            f"  最大降温: {self.max_cooling:.2f} K",
            f"  标准差: {self.std_cooling:.2f} K",
            "",
            f"全域平均降温贡献: {self.mean_cooling_all:.2f} K",
            "=" * 60,
        ]
        return "\n".join(lines)


class CounterfactualAnalyzer:
    """
    反事实分析器
    
    通过LCZ替换情景模拟，估算自然景观的降温贡献。
    """
    
    def __init__(
        self,
        natural_lcz_values: Optional[List] = None,
        replacement_lcz: Union[str, int] = 6,
        verbose: bool = True
    ):
        """
        初始化反事实分析器
        
        参数:
            natural_lcz_values: 自然景观LCZ值列表
                默认: ['A', 'B', 'C', 'D', 'G'] 或 [11, 12, 13, 14, 17]
            replacement_lcz: 替代用的建成区LCZ值
                默认: 6（开敞低层）或 'LCZ_6'
            verbose: 是否打印详细信息
        """
        self.natural_lcz_values = natural_lcz_values
        self.replacement_lcz = replacement_lcz
        self.verbose = verbose
        self.result: Optional[CounterfactualResult] = None
        
    def _detect_lcz_format(self, lcz_values: np.ndarray) -> str:
        """检测LCZ编码格式，返回 'string' 或 'numeric'"""
        sample = lcz_values[~pd.isna(lcz_values)]
        if len(sample) == 0:
            return 'unknown'
        
        sample_val = sample[0]
        
        if isinstance(sample_val, str):
            return 'string'
        elif isinstance(sample_val, (int, np.integer, float, np.floating)):
            return 'numeric'
        else:
            return 'unknown'
    
    def _get_natural_mask(
        self,
        lcz_values: np.ndarray,
        natural_lcz_values: Optional[List] = None
    ) -> np.ndarray:
        """获取自然景观掩膜"""
        if natural_lcz_values is None:
            fmt = self._detect_lcz_format(lcz_values)
            if fmt == 'string':
                natural_lcz_values = LCZ_NATURAL_STRING
            elif fmt == 'numeric':
                natural_lcz_values = LCZ_NATURAL_NUMERIC
            else:
                warnings.warn("无法检测LCZ格式，使用数值型默认值")
                natural_lcz_values = LCZ_NATURAL_NUMERIC
        
        return np.isin(lcz_values, natural_lcz_values)
    
    def _normalize_replacement_lcz(
        self,
        replacement_lcz: Union[str, int],
        lcz_format: str
    ) -> Union[str, int]:
        """标准化替代LCZ值"""
        if lcz_format == 'string':
            if isinstance(replacement_lcz, int):
                return str(replacement_lcz)
            return str(replacement_lcz).replace('LCZ_', '')
        else:
            if isinstance(replacement_lcz, str):
                return int(replacement_lcz.replace('LCZ_', ''))
            return int(replacement_lcz)
    
    def analyze(
        self,
        als_model: ALSRegression,
        districts_gdf: gpd.GeoDataFrame,
        X_F: np.ndarray,
        X_S: np.ndarray,
        f_Ta_coeff1: np.ndarray,
        y_residual: np.ndarray,
        f_Ta_coeff2: Optional[np.ndarray] = None,
        era5_Ta_mean: Optional[np.ndarray] = None,
        lcz_column: str = 'LCZ',
        natural_lcz_values: Optional[List] = None,
        replacement_lcz: Optional[Union[str, int]] = None,
        include_spatial_analysis: bool = True,
        spatial_distance_threshold: float = 500.0
    ) -> CounterfactualResult:
        """
        执行反事实分析
        
        参数:
            als_model: 已拟合的ALS模型（需要有 Ta_per_district）
            districts_gdf: 街区GeoDataFrame（含LCZ列）
            X_F: 人为热特征矩阵
            X_S: 储热特征矩阵
            f_Ta_coeff1: Ta一次项系数
            y_residual: 残差
            f_Ta_coeff2: Ta二次项系数（可选）
            era5_Ta_mean: ERA5气温初值
            lcz_column: LCZ列名
            natural_lcz_values: 自然景观LCZ值列表
            replacement_lcz: 替代建成区LCZ值
            include_spatial_analysis: 是否包含空间自相关分析
            spatial_distance_threshold: 空间权重距离阈值 (m)
            
        返回:
            CounterfactualResult 对象
        """
        if als_model.Ta_per_district is None:
            raise ValueError("ALS模型尚未拟合，请先调用 als_model.fit()")
        
        n_districts = len(districts_gdf)
        
        if natural_lcz_values is None:
            natural_lcz_values = self.natural_lcz_values
        if replacement_lcz is None:
            replacement_lcz = self.replacement_lcz
        
        # 获取原始气温
        Ta_original = als_model.Ta_per_district.copy()
        
        # 获取LCZ值
        if lcz_column not in districts_gdf.columns:
            raise ValueError(f"LCZ列 '{lcz_column}' 不存在于 districts_gdf")
        
        lcz_original = districts_gdf[lcz_column].values.copy()
        lcz_format = self._detect_lcz_format(lcz_original)
        
        # 识别自然景观街区
        natural_mask = self._get_natural_mask(lcz_original, natural_lcz_values)
        n_natural = natural_mask.sum()
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("反事实分析：自然景观降温贡献估算")
            print("=" * 60)
            print(f"总街区数: {n_districts}")
            print(f"自然景观街区数: {n_natural} ({100*n_natural/n_districts:.1f}%)")
            print(f"LCZ格式: {lcz_format}")
            print(f"自然景观LCZ: {natural_lcz_values}")
            print(f"替代建成区LCZ: {replacement_lcz}")
        
        if n_natural == 0:
            warnings.warn("没有检测到自然景观街区")
            return CounterfactualResult(
                Ta_original=Ta_original,
                Ta_counterfactual=Ta_original.copy(),
                cooling_contribution=np.zeros(n_districts),
                natural_mask=natural_mask,
                lcz_original=lcz_original,
                lcz_counterfactual=lcz_original.copy(),
                n_total=n_districts,
                n_natural=0,
                mean_cooling=0.0,
                max_cooling=0.0,
                std_cooling=0.0,
                mean_cooling_all=0.0
            )
        
        # 构建反事实LCZ
        replacement_lcz_normalized = self._normalize_replacement_lcz(
            replacement_lcz, lcz_format
        )
        lcz_counterfactual = lcz_original.copy()
        lcz_counterfactual[natural_mask] = replacement_lcz_normalized
        
        if self.verbose:
            print("\n正在进行反事实模拟...")
            print(f"  替换 {n_natural} 个自然景观街区为 LCZ {replacement_lcz_normalized}")
        
        # 重新拟合ALS模型
        als_counterfactual = ALSRegression()
        
        result_cf = als_counterfactual.fit(
            X_F=X_F,
            X_S=X_S,
            f_Ta_coeff1=f_Ta_coeff1,
            f_Ta_coeff2=f_Ta_coeff2,
            y_residual=y_residual,
            era5_Ta_mean=era5_Ta_mean,
            max_iter=20,
            tol=1e-4,
            verbose=False
        )
        
        Ta_counterfactual = result_cf['Ta_per_district']
        
        # 计算降温贡献
        cooling_contribution = Ta_counterfactual - Ta_original
        
        # 空间自相关分析（可选）
        spatial_original = None
        spatial_counterfactual = None
        
        if include_spatial_analysis:
            if self.verbose:
                print("\n进行空间自相关分析...")
            
            try:
                spatial_weights = SpatialWeightMatrix(
                    districts_gdf,
                    distance_threshold=spatial_distance_threshold
                )
                
                spatial_original = analyze_spatial_autocorrelation(
                    Ta_original, spatial_weights, verbose=False
                )
                spatial_counterfactual = analyze_spatial_autocorrelation(
                    Ta_counterfactual, spatial_weights, verbose=False
                )
                
                if self.verbose and spatial_original:
                    print(f"  原始场景 ρ = {spatial_original['rho']:.4f}")
                    print(f"  反事实场景 ρ = {spatial_counterfactual['rho']:.4f}")
            except (ValueError, np.linalg.LinAlgError, KeyError) as e:
                warnings.warn(f"空间分析失败: {e}")
        
        # 计算统计摘要
        cooling_natural = cooling_contribution[natural_mask]
        
        mean_cooling = float(np.nanmean(cooling_natural))
        max_cooling = float(np.nanmax(cooling_natural))
        std_cooling = float(np.nanstd(cooling_natural))
        mean_cooling_all = float(np.nanmean(cooling_contribution))
        
        self.result = CounterfactualResult(
            Ta_original=Ta_original,
            Ta_counterfactual=Ta_counterfactual,
            cooling_contribution=cooling_contribution,
            natural_mask=natural_mask,
            lcz_original=lcz_original,
            lcz_counterfactual=lcz_counterfactual,
            n_total=n_districts,
            n_natural=n_natural,
            mean_cooling=mean_cooling,
            max_cooling=max_cooling,
            std_cooling=std_cooling,
            mean_cooling_all=mean_cooling_all,
            spatial_analysis_original=spatial_original,
            spatial_analysis_counterfactual=spatial_counterfactual
        )
        
        if self.verbose:
            print(self.result.summary())
        
        return self.result
    
    def analyze_by_lcz_type(
        self,
        als_model: ALSRegression,
        districts_gdf: gpd.GeoDataFrame,
        X_F: np.ndarray,
        X_S: np.ndarray,
        f_Ta_coeff1: np.ndarray,
        y_residual: np.ndarray,
        f_Ta_coeff2: Optional[np.ndarray] = None,
        era5_Ta_mean: Optional[np.ndarray] = None,
        lcz_column: str = 'LCZ',
        replacement_lcz: Optional[Union[str, int]] = None
    ) -> Dict[str, CounterfactualResult]:
        """
        按LCZ类型分别进行反事实分析
        
        返回:
            Dict: {lcz_type: CounterfactualResult}
        """
        results = {}
        
        lcz_values = districts_gdf[lcz_column].values
        lcz_format = self._detect_lcz_format(lcz_values)
        
        if lcz_format == 'string':
            natural_types = LCZ_NATURAL_STRING
        else:
            natural_types = LCZ_NATURAL_NUMERIC
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("按LCZ类型进行反事实分析")
            print("=" * 60)
        
        for lcz_type in natural_types:
            if not np.any(lcz_values == lcz_type):
                continue
            
            if self.verbose:
                type_name = LCZ_NATURAL_NAMES.get(str(lcz_type), f'LCZ {lcz_type}')
                print(f"\n分析 {type_name}...")
            
            analyzer = CounterfactualAnalyzer(
                natural_lcz_values=[lcz_type],
                replacement_lcz=replacement_lcz or self.replacement_lcz,
                verbose=False
            )
            
            result = analyzer.analyze(
                als_model=als_model,
                districts_gdf=districts_gdf,
                X_F=X_F,
                X_S=X_S,
                f_Ta_coeff1=f_Ta_coeff1,
                y_residual=y_residual,
                f_Ta_coeff2=f_Ta_coeff2,
                era5_Ta_mean=era5_Ta_mean,
                lcz_column=lcz_column,
                include_spatial_analysis=False
            )
            
            results[str(lcz_type)] = result
            
            if self.verbose:
                print(f"  街区数: {result.n_natural}")
                print(f"  平均降温: {result.mean_cooling:.2f} K")
        
        if self.verbose:
            print("\n" + "-" * 60)
            print("各类型降温贡献排名:")
            sorted_results = sorted(
                results.items(),
                key=lambda x: x[1].mean_cooling,
                reverse=True
            )
            for i, (lcz_type, result) in enumerate(sorted_results, 1):
                type_name = LCZ_NATURAL_NAMES.get(lcz_type, f'LCZ {lcz_type}')
                print(f"  {i}. {type_name}: {result.mean_cooling:.2f} K "
                      f"(n={result.n_natural})")
            print("=" * 60)
        
        return results


def estimate_cooling_contribution(
    Ta_original: np.ndarray,
    Ta_counterfactual: np.ndarray,
    natural_mask: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    简化版：直接从气温数组计算降温贡献
    
    适用于已有原始和反事实气温的情况。
    
    参数:
        Ta_original: 原始气温数组 (K)
        Ta_counterfactual: 反事实气温数组 (K)
        natural_mask: 自然景观掩膜
        verbose: 是否打印信息
        
    返回:
        降温贡献统计字典
    """
    cooling = Ta_counterfactual - Ta_original
    cooling_natural = cooling[natural_mask]
    
    result = {
        'cooling_contribution': cooling,
        'mean_cooling_natural': float(np.nanmean(cooling_natural)),
        'max_cooling_natural': float(np.nanmax(cooling_natural)),
        'min_cooling_natural': float(np.nanmin(cooling_natural)),
        'std_cooling_natural': float(np.nanstd(cooling_natural)),
        'mean_cooling_all': float(np.nanmean(cooling)),
        'n_natural': int(natural_mask.sum()),
        'n_total': len(Ta_original)
    }
    
    if verbose:
        print("\n降温贡献估算结果:")
        print(f"  自然景观街区: {result['n_natural']}/{result['n_total']}")
        print(f"  平均降温: {result['mean_cooling_natural']:.2f} K")
        print(f"  最大降温: {result['max_cooling_natural']:.2f} K")
        print(f"  全域平均: {result['mean_cooling_all']:.2f} K")
    
    return result
