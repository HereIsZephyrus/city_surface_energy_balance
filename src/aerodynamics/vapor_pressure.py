"""
水汽压计算模块

提供饱和水汽压、实际水汽压和水汽压差的计算功能。

公式参考:
    - Magnus公式: es = 0.6108 × exp[17.27T/(T + 237.3)]
    - 实际水汽压（从相对湿度）: ea = RH × es / 100
    - 实际水汽压（从露点温度）: ea = 0.6108 × exp[17.27Td/(Td + 237.3)]
    - 水汽压差: VPD = es - ea

数据来源:
    - ERA5-Land提供露点温度(dewpoint_temperature_2m)，可直接计算实际水汽压
    - 不需要相对湿度数据
"""

import numpy as np


def _print_array_stats(name: str, array: np.ndarray) -> None:
    """打印数组的基本统计量，方便排查异常值"""
    if array.size == 0:
        print(f"{name}: empty array")
        return
    finite_vals = array[np.isfinite(array)]
    if finite_vals.size == 0:
        print(f"{name}: all values are NaN or inf")
        return
    print(
        f"{name}: min={finite_vals.min():.3f}, "
        f"max={finite_vals.max():.3f}, "
        f"mean={finite_vals.mean():.3f}, "
        f"std={finite_vals.std():.3f}"
    )


def calculate_saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """
    计算饱和水汽压（Magnus公式）

    公式: es = 0.6108 × exp[17.27(T)/(T + 237.3)]

    参数:
        temperature: 温度 (K) - ndarray

    返回:
        饱和水汽压 es (kPa) - ndarray

    参考:
        Magnus, 1844; Murray, 1967
    """
    # 转换为摄氏度
    _print_array_stats("temperature_K", temperature)
    T_celsius = temperature - 273.15
    _print_array_stats("temperature_C", T_celsius)

    # Magnus公式
    es = 0.6108 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))

    return es


def calculate_actual_vapor_pressure(temperature: np.ndarray,
                                   relative_humidity: np.ndarray) -> np.ndarray:
    """
    计算实际水汽压

    公式: ea = RH × es / 100

    参数:
        temperature: 温度 (K) - ndarray
        relative_humidity: 相对湿度 (%) - ndarray, 范围 0-100

    返回:
        实际水汽压 ea (kPa) - ndarray
    """
    # 先计算饱和水汽压
    _print_array_stats("relative_humidity", relative_humidity)
    es = calculate_saturation_vapor_pressure(temperature)

    # 计算实际水汽压
    ea = relative_humidity * es / 100.0

    return ea


def calculate_actual_vapor_pressure_from_dewpoint(
        dewpoint_temperature: np.ndarray) -> np.ndarray:
    """
    从露点温度计算实际水汽压（推荐用于ERA5-Land数据）

    公式: ea = 0.6108 × exp[17.27(Td)/(Td + 237.3)]

    参数:
        dewpoint_temperature: 露点温度 Td (K) - ndarray, 来自ERA5-Land

    返回:
        实际水汽压 ea (kPa) - ndarray

    注意:
        这是从ERA5-Land获取实际水汽压的推荐方法，因为：
        1. ERA5-Land直接提供dewpoint_temperature_2m
        2. 不需要相对湿度数据
        3. 露点温度是水汽含量的直接度量

    数据源:
        ERA5-Land band: 'dewpoint_temperature_2m' (K)
    """
    # 转换为摄氏度
    _print_array_stats("dewpoint_temperature_K", dewpoint_temperature)
    Td_celsius = dewpoint_temperature - 273.15
    _print_array_stats("dewpoint_temperature_C", Td_celsius)

    # Magnus公式计算实际水汽压
    ea = 0.6108 * np.exp(17.27 * Td_celsius / (Td_celsius + 237.3))

    return ea


def calculate_vapor_pressure_deficit(temperature: np.ndarray,
                                     relative_humidity: np.ndarray) -> np.ndarray:
    """
    计算水汽压差 (Vapor Pressure Deficit, VPD)

    公式: VPD = es - ea = es × (1 - RH/100)

    参数:
        temperature: 温度 (K) - ndarray
        relative_humidity: 相对湿度 (%) - ndarray, 范围 0-100

    返回:
        水汽压差 VPD (kPa) - ndarray

    注意:
        VPD 是植物蒸腾和表面阻抗的重要影响因子
    """
    _print_array_stats("relative_humidity", relative_humidity)
    es = calculate_saturation_vapor_pressure(temperature)
    ea = calculate_actual_vapor_pressure(temperature, relative_humidity)

    vpd = es - ea

    return vpd


def calculate_vapor_pressure_deficit_from_dewpoint(
        temperature: np.ndarray,
        dewpoint_temperature: np.ndarray) -> np.ndarray:
    """
    从温度和露点温度计算水汽压差（推荐用于ERA5-Land数据）

    公式: VPD = es(T) - ea(Td)

    参数:
        temperature: 温度 (K) - ndarray, 可以是气温或地表温度
        dewpoint_temperature: 露点温度 Td (K) - ndarray, 来自ERA5-Land

    返回:
        水汽压差 VPD (kPa) - ndarray

    数据源:
        ERA5-Land bands:
        - 'temperature_2m' (K) 或使用地表温度
        - 'dewpoint_temperature_2m' (K)
    """
    es = calculate_saturation_vapor_pressure(temperature)
    ea = calculate_actual_vapor_pressure_from_dewpoint(dewpoint_temperature)

    vpd = es - ea

    return vpd

