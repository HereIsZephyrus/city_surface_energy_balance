"""
空气动力学参数计算工作流

整合所有空气动力学参数的计算流程，提供统一接口。

计算内容:
    - 气压（DEM高程订正）
    - 空气密度
    - 风速
    - 水汽压（实际和饱和）
    - 粗糙度（基于LCZ）
    - 阻抗（空气动力学阻抗和表面阻抗）

使用示例:
    >>> from src.aerodynamics import calculate_aerodynamic_parameters
    >>> aero_params = calculate_aerodynamic_parameters(data)
"""

from typing import Dict
import numpy as np

from .dem_downscaling import adjust_pressure_for_elevation
from .atmospheric import calculate_air_density, calculate_wind_speed
from .vapor_pressure import (
    calculate_actual_vapor_pressure_from_dewpoint,
    calculate_saturation_vapor_pressure
)
from .resistance import (
    calculate_aerodynamic_resistance,
    calculate_surface_resistance
)
from ..utils import LCZ_ROUGHNESS


def calculate_roughness_from_lcz(lcz: np.ndarray) -> np.ndarray:
    """
    根据LCZ数据计算粗糙度长度
    
    参数:
        lcz: LCZ分类栅格数组
    
    返回:
        粗糙度长度数组 (m)
    """
    lcz_int = lcz.astype(int)
    roughness = np.full_like(lcz_int, 0.5, dtype=np.float32)  # 默认值
    
    for lcz_val, z0 in LCZ_ROUGHNESS.items():
        roughness[lcz_int == lcz_val] = z0
    
    return roughness


def calculate_aerodynamic_parameters(
    data: Dict,
    era5_grid_size_meters: float = 11000.0,
    pixel_size_meters: float = 10.0,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    计算空气动力学参数
    
    整合所有空气动力学参数的计算流程。
    
    参数:
        data: 数据字典，需包含:
            - era5: ERA5数据 (surface_pressure, temperature_2m, 
                    dewpoint_temperature_2m, u/v_component_of_wind_10m)
            - landsat: Landsat数据 (lst, ndvi)
            - dem: DEM高程数据
            - lcz: LCZ分类数据
        era5_grid_size_meters: ERA5网格大小 (m)，默认11000
        pixel_size_meters: 输出像素大小 (m)，默认10
        verbose: 是否打印统计信息
    
    返回:
        dict: 包含所有空气动力学参数
            - pressure: 订正后气压 (Pa)
            - air_density: 空气密度 (kg/m³)
            - wind_speed: 风速 (m/s)
            - actual_vapor_pressure: 实际水汽压 (kPa)
            - saturation_vapor_pressure: 饱和水汽压 (kPa)
            - aerodynamic_resistance: 空气动力学阻抗 (s/m)
            - surface_resistance: 表面阻抗 (s/m)
            - roughness_length: 粗糙度长度 (m)
            - lcz: LCZ分类数据
    """
    era5 = data['era5']
    landsat = data['landsat']
    dem = data['dem']
    lcz = data['lcz']
    
    if verbose:
        print("\n" + "=" * 60)
        print("计算空气动力学参数")
        print("=" * 60)
    
    # 1. 气压高程订正
    if verbose:
        print("\n计算气压（DEM高程订正）...")
    pressure = adjust_pressure_for_elevation(
        era5_pressure=era5['surface_pressure'],
        dem_elevation=dem,
        temperature=era5['temperature_2m'],
        era5_grid_size_meters=era5_grid_size_meters,
        pixel_size_meters=pixel_size_meters
    )
    
    # 2. 空气密度
    if verbose:
        print("计算空气密度...")
    air_density = calculate_air_density(era5['temperature_2m'], pressure)
    
    # 3. 风速
    if verbose:
        print("计算风速...")
    wind_speed = calculate_wind_speed(
        era5['u_component_of_wind_10m'],
        era5['v_component_of_wind_10m']
    )
    
    # 4. 水汽压
    if verbose:
        print("计算水汽压...")
    ea = calculate_actual_vapor_pressure_from_dewpoint(
        era5['dewpoint_temperature_2m']
    )
    es = calculate_saturation_vapor_pressure(landsat['lst'])
    
    # 5. 粗糙度
    if verbose:
        print("计算粗糙度...")
    roughness = calculate_roughness_from_lcz(lcz)
    
    # 6. 阻抗
    if verbose:
        print("计算阻抗参数...")
    rah = calculate_aerodynamic_resistance(
        wind_speed=wind_speed,
        roughness_length=roughness,
        lcz=lcz.astype(int)
    )
    rs = calculate_surface_resistance(ndvi=landsat['ndvi'])
    
    result = {
        'pressure': pressure,
        'air_density': air_density,
        'wind_speed': wind_speed,
        'actual_vapor_pressure': ea,
        'saturation_vapor_pressure': es,
        'aerodynamic_resistance': rah,
        'surface_resistance': rs,
        'roughness_length': roughness,
        'lcz': lcz.astype(int)
    }
    
    # 打印统计
    if verbose:
        print(f"\n参数统计:")
        print(f"  空气密度: {np.nanmean(air_density):.3f} kg/m³")
        print(f"  风速: {np.nanmean(wind_speed):.2f} m/s")
        print(f"  实际水汽压: {np.nanmean(ea):.3f} kPa")
        print(f"  饱和水汽压: {np.nanmean(es):.3f} kPa")
        print(f"  空气动力学阻抗: {np.nanmean(rah):.1f} s/m")
        print(f"  表面阻抗: {np.nanmean(rs):.1f} s/m")
    
    return result

