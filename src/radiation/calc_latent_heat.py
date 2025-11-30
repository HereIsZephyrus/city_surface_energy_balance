"""
潜热通量计算模块

基于 SEBAL 模型计算潜热通量。

公式:
    QE = ρCp(es - ea) / [γ(rah + rs)]

其中:
    - es: 地表温度对应的饱和水汽压 (kPa) - 由aerodynamics模块计算
    - ea: 近地表实际水汽压 (kPa) - 由aerodynamics模块计算
    - γ: 干湿计常数 ≈ 0.067 kPa/K
    - rah: 大气湍流热交换阻抗 (s/m) - 由aerodynamics模块计算
    - rs: 下垫面植被阻抗/表面阻抗 (s/m) - 由aerodynamics模块计算

注意:
    本模块仅负责根据已知参数计算潜热通量，所有空气动力学参数
    （如水汽压、阻抗等）应由aerodynamics模块计算后传入。
"""

import numpy as np
from .constants import (
    SPECIFIC_HEAT_AIR,
    LATENT_HEAT_VAPORIZATION,
    PSYCHROMETRIC_CONSTANT
)


class LatentHeatFluxCalculator:
    """
    潜热通量计算器

    Formula: QE = ρCp(es - ea) / [γ(rah + rs)]

    where:
    - es: 地表温度对应的饱和水汽压 (kPa)
    - ea: 近地表实际水汽压 (kPa)
    - γ: 干湿计常数 ≈ 0.067 kPa/K
    - rah: 大气湍流热交换阻抗 (s/m)
    - rs: 下垫面植被阻抗/表面阻抗 (s/m)

    注意:
        所有空气动力学参数（es, ea, rah, rs）应由外部（aerodynamics模块）
        计算后传入，本计算器仅执行潜热通量公式计算。
    """

    def __init__(self,
                 saturation_vapor_pressure: np.ndarray,
                 actual_vapor_pressure: np.ndarray,
                 aerodynamic_resistance: np.ndarray,
                 surface_resistance: np.ndarray,
                 air_density: float,
                 specific_heat: float = SPECIFIC_HEAT_AIR,
                 latent_heat: float = LATENT_HEAT_VAPORIZATION):
        """
        初始化潜热通量计算器

        Parameters:
            saturation_vapor_pressure: 地表温度对应的饱和水汽压 es (kPa) - ndarray
            actual_vapor_pressure: 近地表实际水汽压 ea (kPa) - ndarray
                                  推荐从ERA5-Land的dewpoint_temperature_2m计算
            aerodynamic_resistance: 大气湍流热交换阻抗 rah (s/m) - ndarray
            surface_resistance: 表面阻抗 rs (s/m) - ndarray
            air_density: 空气密度 ρ (kg/m³) - scalar
                        使用ERA5-Land气温计算的参考值
            specific_heat: 空气定压比热容 (J/kg/K) - scalar
            latent_heat: 水的汽化潜热 (J/kg) - scalar

        注意:
            - saturation_vapor_pressure: 通过aerodynamics.calculate_saturation_vapor_pressure(Ts)
            - actual_vapor_pressure: 推荐使用aerodynamics.calculate_actual_vapor_pressure_from_dewpoint(Td)
              从ERA5-Land的dewpoint_temperature_2m直接计算
            - air_density使用参考值（标量），numpy广播机制自动处理
        """
        self.es = saturation_vapor_pressure
        self.ea = actual_vapor_pressure
        self.rah = aerodynamic_resistance
        self.rs = surface_resistance
        self.rho = air_density
        self.Cp = specific_heat
        self.lambda_v = latent_heat

    @property
    def latent_heat_flux(self) -> np.ndarray:
        """
        计算潜热通量

        Formula: QE = ρCp(es - ea) / [γ(rah + rs)]

        Returns:
            潜热通量 QE (W/m²) - ndarray
        """
        # 避免除以零
        total_resistance = self.rah + self.rs
        total_resistance_safe = np.maximum(total_resistance, 1.0)

        # 计算潜热通量
        # QE = ρCp(es - ea) / [γ(rah + rs)]
        # 单位说明: es,ea为kPa, γ为kPa/K, 单位自洽无需额外转换
        # 注意: 不要添加 ×1000，否则结果会偏大1000倍！
        QE = (self.rho * self.Cp * (self.es - self.ea) / 
              (PSYCHROMETRIC_CONSTANT * total_resistance_safe))

        # 限制潜热通量为非负值（蒸发）
        QE = np.maximum(QE, 0.0)

        return QE.astype(np.float32)
