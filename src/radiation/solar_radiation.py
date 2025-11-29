"""
Solar radiation calculation module

Based on SEBAL model theory, calculates shortwave downward radiation (Rs↓) for each DEM grid cell
Factors considered:
  - Solar position corresponding to date and time
  - Effect of terrain slope and aspect on solar incidence angle
  - Atmospheric transmissivity
"""
from datetime import datetime
from typing import Tuple, Optional
import numpy as np
from pyproj import CRS, Transformer

def parse_datetime(date_str: str, time_str: str) -> datetime:
    """
    解析日期和时间字符串

    参数:
        date_str: 日期字符串 (YYYY-MM-DD)
        time_str: 时间字符串 (HH:MM:SS 或 HH:MM)

    返回:
        datetime对象
    """
    try:
        datetime_str = f"{date_str} {time_str}"
        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            datetime_str = f"{date_str} {time_str}"
            dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError(f"无效的日期时间格式: {date_str} {time_str}")

    return dt
class SolarConstantsCalculator:
    """Solar constants calculator class"""

    Gsc = 1366.0
    def __init__(self, latitude: float, longitude: float, datetime_obj: datetime, std_meridian: float = 120.0):
        self.latitude = latitude
        self.longitude = longitude
        self.hour = datetime_obj.hour
        self.tm_yday = datetime_obj.timetuple().tm_yday
        self.std_meridian = std_meridian

    @property
    def earth_sun_distance(self) -> float:
        """
        equation: dr = 1 + 0.033*cos(2π*doy/365)
        """
        dr = 1 + 0.033 * np.cos(2 * np.pi * self.tm_yday / 365)
        return dr

    @property
    def solar_declination(self) -> float:
        """
        equation: δ = 0.409*sin(2π*doy/365 - 1.39)
        """
        delta = 0.409 * np.sin(2 * np.pi * self.tm_yday / 365 - 1.39)
        return delta

    @property
    def solar_time_angle(self) -> float:
        """
        calculated omega = π/12 * (t - 12 - 4*(Lz - Lon)/60)
        
        Note: self.longitude is in radians, std_meridian is in degrees
        """
        # Convert longitude from radians to degrees for time calculation
        longitude_deg = np.degrees(self.longitude)
        delta_lon = (self.std_meridian - longitude_deg) / 15.0
        solar_time = self.hour + delta_lon
        omega = np.pi / 12 * (solar_time - 12)
        return omega

    @property
    def sun_elevation_angle(self) -> float:
        """
        equation: sin(α) = sin(φ)*sin(δ) + cos(φ)*cos(δ)*cos(ω)
        """
        sin_alpha = (np.sin(self.latitude) * np.sin(self.solar_declination) +
                     np.cos(self.latitude) * np.cos(self.solar_declination) * 
                     np.cos(self.solar_time_angle))

        sin_alpha = np.clip(sin_alpha, -1, 1)
        return np.arcsin(sin_alpha)

    @property
    def sun_azimuth_angle(self) -> float:
        """
        equation: cos(A) = (sin(δ) - sin(α)*sin(φ)) / (cos(α)*cos(φ))
        return azimuth = -np.arccos(cos_azimuth) if solar_time_angle >= 0 else np.arccos(cos_azimuth)
        """
        if np.cos(self.sun_elevation_angle) == 0:
            return 0.0

        cos_azimuth = ((np.sin(self.solar_declination) - 
                       np.sin(self.sun_elevation_angle) * np.sin(self.latitude)) /
                      (np.cos(self.sun_elevation_angle) * np.cos(self.latitude)))

        cos_azimuth = np.clip(cos_azimuth, -1, 1)

        if self.solar_time_angle >= 0:
            azimuth = -np.arccos(cos_azimuth)
        else:
            azimuth = np.arccos(cos_azimuth)

        return azimuth

class SolarRadiationCalculator:
    """
    Solar radiation calculator for horizontal surfaces (Urban Canopy Layer)

    Calculates shortwave radiation on horizontal surfaces without terrain effects.
    Suitable for urban energy balance models where radiation is input to the urban canopy.

    Responsibilities:
    - Calculate atmospheric transmissivity based on elevation array
    - Calculate direct and diffuse radiation for horizontal surfaces
    - Work with numpy ndarrays
    """

    def __init__(self, solar_constants: SolarConstantsCalculator, elevation: np.ndarray):
        """
        Initialize radiation calculator

        Parameters:
            solar_constants: SolarConstantsCalculator instance for sun geometry
            elevation: elevation array in meters (ndarray)
        """
        self.constants = solar_constants
        self.elevation = elevation

    @property
    def atmospheric_transmissivity(self) -> np.ndarray:
        """
        Calculate atmospheric transmissivity based on elevation

        equation: τ_sw = 0.75 + 2×10⁻⁵ × Ele
        reference: Laipelt et al., 2021

        Returns:
            transmissivity array (ndarray, clipped to 0.1-1.0)
        """
        tau_sw = 0.75 + 2e-5 * self.elevation
        return np.clip(tau_sw, 0.1, 1.0).astype(np.float32)

    @property
    def direct_radiation(self) -> np.ndarray:
        """
        Calculate direct solar radiation on horizontal surface

        equation: S↓_direct = Gsc × sin(α) × dr × τ_sw
        where α is solar elevation angle

        Returns:
            direct radiation array in W/m² (ndarray)
        """
        sun_elevation = self.constants.sun_elevation_angle
        
        if sun_elevation <= 0:
            return np.zeros_like(self.elevation, dtype=np.float32)
        
        # Direct radiation on horizontal surface = Gsc * sin(solar_elevation) * dr * tau
        s_down = (self.constants.Gsc * np.sin(sun_elevation) * 
                  self.constants.earth_sun_distance * self.atmospheric_transmissivity)

        return np.maximum(s_down, 0.0).astype(np.float32)

    @property
    def global_radiation(self) -> np.ndarray:
        """
        Calculate global shortwave downward radiation on horizontal surface

        SEBAL公式: Rs↓ = Gsc × dr × sin(θ) × τsw

        注意: τsw 是大气总透射率，已经包含了散射的影响，
        因此不需要额外添加散射辐射项。

        Returns:
            global radiation array in W/m² (ndarray)
        """
        if self.constants.sun_elevation_angle <= 0:
            return np.zeros_like(self.elevation, dtype=np.float32)

        # 直接使用 SEBAL 公式，τsw 已包含散射影响
        return self.direct_radiation


def calculate_dem_solar_radiation(dem_array: np.ndarray,
                                  dem_geotransform: Tuple,
                                  datetime_obj: datetime,
                                  std_meridian: float = 120.0,
                                  target_crs: Optional[str] = None) -> np.ndarray:
    """
    Calculate shortwave downward radiation on horizontal surfaces (Urban Canopy Layer)

    This function calculates solar radiation for horizontal surfaces without terrain effects.
    Suitable for urban energy balance models.

    Parameters:
        dem_array: elevation array in meters (ndarray) - used for atmospheric transmissivity
        dem_geotransform: geotransform tuple (x_min, pixel_width, 0, y_max, 0, -pixel_height)
        datetime_obj: calculation datetime
        std_meridian: standard meridian in degrees (default 120 for China)
        target_crs: coordinate reference system string (e.g., 'EPSG:32650')
                   If provided, coordinates will be transformed to WGS84 for solar calculations

    Returns:
        global radiation array in W/m² (ndarray)
    """
    rows, cols = dem_array.shape

    # Extract coordinates (in source CRS units)
    x_min = dem_geotransform[0]
    pixel_width = dem_geotransform[1]
    y_max = dem_geotransform[3]
    pixel_height = dem_geotransform[5]  # Usually negative

    # Calculate center coordinates in source CRS
    center_x = x_min + pixel_width * cols / 2
    center_y = y_max + pixel_height * rows / 2

    # Transform coordinates to WGS84 (EPSG:4326) if a projected CRS is provided
    if target_crs is not None:
        src_crs = CRS.from_string(target_crs)
        # Check if the CRS is projected (not geographic)
        if src_crs.is_projected:
            # Transform to WGS84
            transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
            center_lon_deg, center_lat_deg = transformer.transform(center_x, center_y)
        else:
            # Already in geographic coordinates
            center_lon_deg = center_x
            center_lat_deg = center_y
    else:
        # Assume coordinates are already in degrees (legacy behavior)
        center_lon_deg = center_x
        center_lat_deg = center_y
    
    # Convert to radians for trigonometric calculations
    center_lat = np.radians(center_lat_deg)
    center_lon = np.radians(center_lon_deg)

    # Create solar constants calculator
    solar_constants = SolarConstantsCalculator(
        latitude=center_lat,
        longitude=center_lon,
        datetime_obj=datetime_obj,
        std_meridian=std_meridian
    )

    # Create radiation calculator (horizontal surface only)
    rad_calculator = SolarRadiationCalculator(
        solar_constants=solar_constants,
        elevation=dem_array
    )

    # Calculate and return global radiation
    return rad_calculator.global_radiation


def calculate_slope_aspect(dem: np.ndarray, 
                          cell_width: float,
                          cell_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    equation: slope = arctan(sqrt(dz_x^2 + dz_y^2))
    equation: aspect = arctan2(dz_x, -dz_y)
    """
    rows, cols = dem.shape
    slopes = np.zeros_like(dem, dtype=np.float32)
    aspects = np.zeros_like(dem, dtype=np.float32)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            z = dem[i-1:i+2, j-1:j+2]

            dz_x = ((z[0, 2] + 2*z[1, 2] + z[2, 2]) - 
                    (z[0, 0] + 2*z[1, 0] + z[2, 0])) / (8 * cell_width)

            dz_y = ((z[2, 0] + 2*z[2, 1] + z[2, 2]) - 
                    (z[0, 0] + 2*z[0, 1] + z[0, 2])) / (8 * cell_height)

            slope = np.arctan(np.sqrt(dz_x**2 + dz_y**2))
            slopes[i, j] = slope

            if dz_x == 0 and dz_y == 0:
                aspect = 0
            else:
                aspect = np.arctan2(dz_x, -dz_y)
                if aspect < 0:
                    aspect += 2 * np.pi

            aspects[i, j] = aspect

    slopes[0, :] = slopes[1, :]
    slopes[-1, :] = slopes[-2, :]
    slopes[:, 0] = slopes[:, 1]
    slopes[:, -1] = slopes[:, -2]

    aspects[0, :] = aspects[1, :]
    aspects[-1, :] = aspects[-2, :]
    aspects[:, 0] = aspects[:, 1]
    aspects[:, -1] = aspects[:, -2]

    return slopes, aspects

