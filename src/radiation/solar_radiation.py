"""
Solar radiation calculation module

Based on SEBAL model theory, calculates shortwave downward radiation (Rs↓) for each DEM grid cell
Factors considered:
  - Solar position corresponding to date and time
  - Effect of terrain slope and aspect on solar incidence angle
  - Atmospheric transmissivity
"""
from datetime import datetime
from typing import Tuple
import numpy as np

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
        """
        delta_lon = (self.std_meridian - self.longitude) / 15.0
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
    Solar radiation calculator - calculates radiation for entire DEM arrays

    Responsibilities:
    - Calculate atmospheric transmissivity based on elevation array
    - Calculate incident angles for terrain
    - Calculate direct, diffuse, and global radiation arrays
    - Work with numpy ndarrays after DEM is loaded
    """

    def __init__(self, solar_constants: SolarConstantsCalculator, elevation: np.ndarray, slope: np.ndarray, aspect: np.ndarray):
        """
        Initialize radiation calculator

        Parameters:
            solar_constants: SolarConstantsCalculator instance for sun geometry
            elevation: elevation array in meters (ndarray)
            slope: slope array in radians (ndarray)
            aspect: aspect array in radians (ndarray)
        """
        self.constants = solar_constants
        self.elevation = elevation
        self.slope = slope
        self.aspect = aspect

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
    def incident_angle(self) -> np.ndarray:
        """
        Calculate solar incident angle on inclined terrain

        equation: cos(θ) = sin(α)*cos(s) + cos(α)*sin(s)*cos(A_sun - A_slope)

        Returns:
            incident angle array in radians (ndarray)
        """
        cos_incident = (np.sin(self.constants.sun_elevation_angle) * np.cos(self.slope) +
                       np.cos(self.constants.sun_elevation_angle) * np.sin(self.slope) *
                       np.cos(self.constants.sun_azimuth_angle - self.aspect))

        cos_incident = np.clip(cos_incident, -1, 1)
        return np.arccos(cos_incident)

    @property
    def direct_radiation(self) -> np.ndarray:
        """
        Calculate direct solar radiation

        equation: S↓ = Gsc × cosθ × dr × τ_sw
        reference: Laipelt et al., 2021 公式(3)

        Returns:
            direct radiation array in W/m² (ndarray)
        """
        cos_incident = np.cos(self.incident_angle)

        # No radiation when sun is below horizon
        s_down = np.where(
            cos_incident > 0,
            self.constants.Gsc * cos_incident * self.constants.earth_sun_distance * self.atmospheric_transmissivity,
            0.0
        )

        return np.maximum(s_down, 0.0).astype(np.float32)

    @property
    def diffuse_radiation(self) -> np.ndarray:
        """
        Calculate diffuse radiation (simplified model)

        equation: Rd = 0.25 * Gsc * τ_sw * sin(α)

        Returns:
            diffuse radiation array in W/m² (ndarray)
        """
        sun_elevation = np.pi / 2 - self.incident_angle

        diffuse = np.where(
            sun_elevation > 0,
            0.25 * self.constants.Gsc * self.atmospheric_transmissivity * np.sin(sun_elevation),
            0.0
        )

        return np.maximum(diffuse, 0.0).astype(np.float32)

    @property
    def global_radiation(self) -> np.ndarray:
        """
        Calculate global shortwave downward radiation

        equation: Rs↓ = Rd + Rd_diffuse

        Returns:
            global radiation array in W/m² (ndarray)
        """
        if self.constants.sun_elevation_angle <= 0:
            return np.zeros_like(self.elevation, dtype=np.float32)

        return (self.direct_radiation + self.diffuse_radiation).astype(np.float32)


def calculate_dem_solar_radiation(dem_array: np.ndarray,
                                  dem_geotransform: Tuple,
                                  datetime_obj: datetime,
                                  std_meridian: float = 120.0,
                                  consider_terrain: bool = True) -> np.ndarray:
    """
    Calculate shortwave downward radiation for entire DEM array

    This is a helper function that orchestrates the calculation workflow:
    1. Calculate slope and aspect from DEM
    2. Create solar constants calculator (center coordinates)
    3. Create radiation calculator with arrays
    4. Return global radiation array

    Parameters:
        dem_array: elevation array in meters (ndarray)
        dem_geotransform: geotransform tuple (x_min, pixel_width, 0, y_max, 0, -pixel_height)
        datetime_obj: calculation datetime
        std_meridian: standard meridian in degrees (default 120 for China)
        consider_terrain: whether to consider slope and aspect effects

    Returns:
        global radiation array in W/m² (ndarray)
    """
    rows, cols = dem_array.shape

    # Extract coordinates
    x_min = dem_geotransform[0]
    pixel_width = dem_geotransform[1]
    y_max = dem_geotransform[3]
    pixel_height = dem_geotransform[5]  # Usually negative

    # Calculate slope and aspect if needed
    if consider_terrain:
        slopes, aspects = calculate_slope_aspect(dem_array, pixel_width, abs(pixel_height))
    else:
        slopes = np.zeros_like(dem_array, dtype=np.float32)
        aspects = np.zeros_like(dem_array, dtype=np.float32)

    # Use center coordinates for sun position
    center_lat = np.radians(y_max + pixel_height * rows / 2)
    center_lon = np.radians(x_min + pixel_width * cols / 2)

    # Create solar constants calculator
    solar_constants = SolarConstantsCalculator(
        latitude=center_lat,
        longitude=center_lon,
        datetime_obj=datetime_obj,
        std_meridian=std_meridian
    )

    # Create radiation calculator
    rad_calculator = SolarRadiationCalculator(
        solar_constants=solar_constants,
        elevation=dem_array,
        slope=slopes,
        aspect=aspects
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

