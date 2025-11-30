# 使用指南：能量平衡计算工作流

## 概述

本指南说明如何使用重构后的 `radiation` 和 `aerodynamics` 模块进行城市地表能量平衡计算。

## 完整能量平衡方程

```
Q* + QF = QH + QE + ΔQSb + ΔQSg + ΔQA
```

其中：
- **可直接量化**（`radiation` 模块计算）：
  - `Q*`: 净辐射 (W/m²) - 依赖Ta
  - `ΔQSg`: 土壤热通量 (W/m²) - 依赖Ta
  - `QH`: 感热通量 (W/m²) - 依赖Ta
  - `QE`: 潜热通量 (W/m²) - 依赖Ta

- **需要回归估计**（外部模块处理）：
  - `QF`: 人为热/建筑热 (W/m²)
  - `ΔQSb`: 建筑储热 (W/m²)
  - `ΔQA`: 水平热交换 (W/m²)

## 工作流程

### 第一步：准备遥感和气象数据

```python
import numpy as np
from datetime import datetime

# 1. 遥感数据
surface_temperature = ...  # 地表温度 Ts (K) - 热红外反演
albedo = ...               # 地表反照率 (0-1) - 多光谱计算
ndvi = ...                 # 归一化植被指数 - 多光谱计算
surface_emissivity = ...   # 地表发射率 (0-1) - NDVI阈值法计算
elevation = ...            # 高程 (m) - DEM

# 2. 气象数据（ERA5-Land）
air_temperature = ...      # 近地表气温 Ta (K) - 初始估计值
relative_humidity = ...    # 相对湿度 (%)
wind_speed = ...           # 近地表风速 (m/s)

# 3. 太阳辐射数据（使用 solar_radiation 模块计算）
from radiation import calculate_dem_solar_radiation

shortwave_down = calculate_dem_solar_radiation(
    dem_array=elevation,
    dem_geotransform=geotransform,
    datetime_obj=datetime(2025, 6, 21, 12, 0, 0),
    std_meridian=120.0,
    consider_terrain=True
)
```

### 第二步：计算空气动力学参数

```python
from aerodynamics import (
    calculate_saturation_vapor_pressure,
    calculate_actual_vapor_pressure,
    calculate_aerodynamic_resistance,
    calculate_surface_resistance
)

# 计算水汽压
saturation_vapor_pressure = calculate_saturation_vapor_pressure(surface_temperature)
actual_vapor_pressure = calculate_actual_vapor_pressure(
    temperature=air_temperature * np.ones_like(surface_temperature),  # 转为数组
    relative_humidity=relative_humidity
)

# 计算阻抗参数
# TODO: 这些函数需要根据实际的LCZ分类、建筑数据等进一步完善
roughness_length = ...  # 从建筑数据计算，或根据LCZ查表

aerodynamic_resistance = calculate_aerodynamic_resistance(
    wind_speed=wind_speed,
    roughness_length=roughness_length
)

surface_resistance = calculate_surface_resistance(
    ndvi=ndvi
)
```

### 第三步：计算可量化的能量通量

```python
from radiation import calculate_quantifiable_fluxes

# 计算所有依赖Ta的可量化通量
result = calculate_quantifiable_fluxes(
    shortwave_down=shortwave_down,
    surface_temperature=surface_temperature,
    air_temperature=air_temperature,  # 待估计参数
    elevation=elevation,
    albedo=albedo,
    ndvi=ndvi,
    saturation_vapor_pressure=saturation_vapor_pressure,
    actual_vapor_pressure=actual_vapor_pressure,
    aerodynamic_resistance=aerodynamic_resistance,
    surface_resistance=surface_resistance,
    surface_emissivity=surface_emissivity
)

# 提取结果
Q_star = result['net_radiation']           # 净辐射
delta_QSg = result['soil_heat_flux']       # 土壤热通量
QH = result['sensible_heat_flux']          # 感热通量
QE = result['latent_heat_flux']            # 潜热通量
quantified_total = result['quantified_total']  # Q* - ΔQSg - QH - QE
```

### 第四步：迭代最小二乘估计（外部模块实现）

```python
# 这部分需要在外部模块中实现
# 基本思路：

# 1. 准备影响因子矩阵
# X_F: QF的影响因子（LCZ类型、不透水面面积、人口等）
# X_S: ΔQSb的影响因子（建筑体积、LCZ类型、植被覆盖度等）

# 2. 对每个LCZ区域k，构建方程：
# f^k(Ta) + Σ(αi * X_Fi^k) + Σ(βi * X_Si^k) = C^k
# 其中 f^k(Ta) = quantified_total (已量化部分的残差)

# 3. 使用交替最小二乘法（ALS）迭代估计：
#    a) 固定dT = Ta - Ts，估计线性系数 α, β
#    b) 固定α, β，更新每个区域的dT
#    c) 重复直到收敛

# 4. 估计出的Ta可用于进一步分析
```

## 关键点说明

### 1. Ta（近地表气温）的处理

- Ta 是关键的未知量，贯穿整个能量平衡方程
- 所有可量化的通量（Q*, ΔQSg, QH, QE）都依赖Ta
- Ta 需要通过迭代最小二乘法结合影响因子回归估计

### 2. 空气动力学参数的来源

- **水汽压**：由 `aerodynamics` 模块从气象数据计算
- **阻抗参数**：需要根据LCZ分类、建筑数据、风速等计算
  - 目前 `aerodynamics.resistance` 提供了简化实现
  - 完整实现需要集成LCZ分类结果和建筑数据

### 3. 能量平衡的闭合性

- `quantified_total = Q* - ΔQSg - QH - QE` 代表未量化部分
- 理论上：`quantified_total ≈ QF + ΔQSb + ΔQA`
- 通过回归估计这三项，可以分析城市热环境的驱动因素

### 4. LCZ（局地气候区）的作用

- 每个LCZ内部假设Ta一致
- LCZ边界作为区域划分的依据
- LCZ类型影响阻抗参数和影响因子

## 示例：完整计算流程

```python
import numpy as np
from datetime import datetime
from radiation import calculate_dem_solar_radiation, calculate_quantifiable_fluxes
from radiation import load_dem, save_radiation
from aerodynamics import (
    calculate_saturation_vapor_pressure,
    calculate_actual_vapor_pressure,
    calculate_aerodynamic_resistance,
    calculate_surface_resistance
)

# ========== 1. 加载数据 ==========
# 加载DEM
dem_array, metadata = load_dem('./data/dem/wuhan_dem.tif')
elevation = dem_array

# 加载其他遥感数据
surface_temperature = ...  # 从热红外数据
albedo = ...
ndvi = ...
surface_emissivity = ...

# ========== 2. 计算短波辐射 ==========
transform = metadata['transform']
if hasattr(transform, 'c'):
    geotransform = (transform.c, transform.a, 0, transform.f, 0, transform.e)
else:
    geotransform = transform

shortwave_down = calculate_dem_solar_radiation(
    dem_array=elevation,
    dem_geotransform=geotransform,
    datetime_obj=datetime(2025, 6, 21, 12, 0, 0),
    std_meridian=120.0,
    consider_terrain=True
)

# ========== 3. 准备气象数据和阻抗参数 ==========
# 初始气温估计（可以从ERA5获取）
air_temperature = 298.0  # K

# 气象参数
relative_humidity = ...  # 从ERA5
wind_speed = ...

# 计算空气动力学参数
es = calculate_saturation_vapor_pressure(surface_temperature)
ea_values = np.ones_like(surface_temperature) * 2.0  # kPa，从ERA5或湿度计算

# 阻抗参数（需要根据实际数据改进）
roughness = 0.5 * np.ones_like(elevation)  # m
rah = calculate_aerodynamic_resistance(wind_speed, roughness)
rs = calculate_surface_resistance(ndvi)

# ========== 4. 计算可量化通量 ==========
result = calculate_quantifiable_fluxes(
    shortwave_down=shortwave_down,
    surface_temperature=surface_temperature,
    air_temperature=air_temperature,
    elevation=elevation,
    albedo=albedo,
    ndvi=ndvi,
    saturation_vapor_pressure=es,
    actual_vapor_pressure=ea_values,
    aerodynamic_resistance=rah,
    surface_resistance=rs,
    surface_emissivity=surface_emissivity
)

# ========== 5. 分析结果 ==========
print(f"净辐射 Q*: {result['net_radiation'].mean():.2f} W/m²")
print(f"土壤热通量: {result['soil_heat_flux'].mean():.2f} W/m²")
print(f"感热通量: {result['sensible_heat_flux'].mean():.2f} W/m²")
print(f"潜热通量: {result['latent_heat_flux'].mean():.2f} W/m²")
print(f"未量化部分: {result['quantified_total'].mean():.2f} W/m²")

# ========== 6. 保存结果 ==========
save_radiation(result['net_radiation'], './output/Q_star.tif', metadata)
save_radiation(result['quantified_total'], './output/residual.tif', metadata)
```

## 后续工作

### aerodynamics 模块需要完善的部分：

1. **阻抗计算**：
   - 集成LCZ分类数据
   - 根据建筑数据计算粗糙度
   - 实现不同地表类型的参数化方案
   - 考虑大气稳定度的影响

2. **水汽压计算**：
   - 当前实现已基本完善
   - 可以考虑更精确的公式（如Tetens公式）

### 外部模块需要实现的部分：

1. **迭代最小二乘估计模块**：
   - 实现ALS算法估计Ta
   - 构建影响因子矩阵（QF, ΔQSb的影响因子）
   - 实现收敛判断和迭代优化

2. **LCZ分类模块**：
   - 提供LCZ边界数据
   - 提供LCZ类型标签
   - 支持区域级别的聚合计算

3. **降温效益量化模块**：
   - 分析ΔQA（水平热交换）
   - 量化相邻景观的影响
   - 构建因果推断模型

## 参考文献

1. 《晴朗无风条件下城市生态空间对城市降温作用量化模型》
2. Bastiaanssen et al., 1998. SEBAL: 1. Formulation. Journal of Hydrology.
3. Laipelt et al., 2021. Long-term monitoring of evapotranspiration using SEBAL.

---

**最后更新**: 2025-11-21

