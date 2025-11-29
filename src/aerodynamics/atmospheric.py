"""
大气参数计算模块

提供从ERA5-Land数据计算大气物理参数的功能。

主要功能:
    1. 空气密度计算（作为气温和气压的函数）
    2. 风速计算（从u、v分量）
    3. 其他大气参数计算

数据源:
    ERA5-Land提供以下参数：
    - surface_pressure (Pa)
    - u_component_of_wind_10m (m/s)
    - v_component_of_wind_10m (m/s)
    - dewpoint_temperature_2m (K)
"""

import numpy as np

def calculate_air_density(
        air_temperature: np.ndarray,
        surface_pressure: np.ndarray,
        gas_constant: float = 287.0) -> np.ndarray:
    """
    计算空气密度（作为气温和气压的函数）

    使用理想气体状态方程:
        ρ = P / (R_d × T)

    其中:
        ρ: 空气密度 (kg/m³)
        P: 气压 (Pa)
        R_d: 干空气气体常数 = 287.0 J/(kg·K)
        T: 气温 (K)

    参数:
        air_temperature: 近地表气温 Ta (K) - scalar或ndarray
                        注意：这是待估计的未知量
        surface_pressure: 地表气压 P (Pa) - scalar或ndarray
                         来自ERA5-Land: 'surface_pressure'
        gas_constant: 干空气气体常数 (J/(kg·K)) - scalar, 默认287.0

    返回:
        空气密度 ρ(Ta) (kg/m³) - 与输入类型相同

    注意:
        1. 空气密度是气温Ta的函数: ρ(Ta) = P / (R_d × Ta)
        2. 气压P从ERA5-Land获取，Ta是模型待估计参数
        3. 标准条件下(P=101325 Pa, T=288.15 K): ρ ≈ 1.225 kg/m³

    数据源:
        ERA5-Land band: 'surface_pressure' (Pa)

    参考:
        Wallace & Hobbs, 2006. Atmospheric Science: An Introductory Survey.
    """
    # 理想气体状态方程
    rho = surface_pressure / (gas_constant * air_temperature)

    return rho.astype(np.float32)


def calculate_wind_speed(
        u_component: np.ndarray,
        v_component: np.ndarray) -> np.ndarray:
    """
    从风速分量计算风速大小

    公式: U = √(u² + v²)

    参数:
        u_component: 东西向风速分量 u (m/s) - ndarray
                    来自ERA5-Land: 'u_component_of_wind_10m'
        v_component: 南北向风速分量 v (m/s) - ndarray
                    来自ERA5-Land: 'v_component_of_wind_10m'

    返回:
        风速大小 U (m/s) - ndarray

    注意:
        1. ERA5-Land提供10米高度的风速分量
        2. 风速用于计算空气动力学阻抗rah
        3. 可能需要根据粗糙度进行高度订正

    数据源:
        ERA5-Land bands:
        - 'u_component_of_wind_10m' (m/s)
        - 'v_component_of_wind_10m' (m/s)
    """
    U = np.sqrt(u_component**2 + v_component**2)

    return U.astype(np.float32)


def adjust_wind_speed_height(
        wind_speed_measured: np.ndarray,
        height_measured: float,
        height_target: float,
        roughness_length: np.ndarray,
        displacement_height: np.ndarray = None) -> np.ndarray:
    """
    风速高度订正（对数廓线法）

    公式: U(z) = U(z_m) × ln[(z - d)/z0] / ln[(z_m - d)/z0]

    参数:
        wind_speed_measured: 测量高度的风速 (m/s) - ndarray
        height_measured: 测量高度 (m) - scalar, ERA5-Land为10m
        height_target: 目标高度 (m) - scalar
        roughness_length: 地表粗糙度 z0 (m) - ndarray
        displacement_height: 零平面位移高度 d (m) - ndarray, 可选

    返回:
        目标高度的风速 U(z) (m/s) - ndarray

    注意:
        ERA5-Land提供10米高度风速，可能需要订正到2米或树冠高度
    """
    if displacement_height is None:
        displacement_height = np.zeros_like(roughness_length)

    # von Karman常数
    k = 0.41

    # 避免对数计算问题
    roughness_safe = np.maximum(roughness_length, 0.001)
    z_m_eff = np.maximum(height_measured - displacement_height, roughness_safe * 2)
    z_t_eff = np.maximum(height_target - displacement_height, roughness_safe * 2)

    # 对数廓线订正
    U_target = wind_speed_measured * (
        np.log(z_t_eff / roughness_safe) / 
        np.log(z_m_eff / roughness_safe)
    )

    # 限制在合理范围
    U_target = np.maximum(U_target, 0.1)  # 最小风速0.1 m/s

    return U_target.astype(np.float32)
