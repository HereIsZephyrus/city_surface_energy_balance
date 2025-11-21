# 城市地表能量平衡计算模块

## 模块概述

本模块基于 **SEBAL** (Surface Energy Balance Algorithm for Land) 模型实现，提供完整的地表能量平衡计算功能。模块采用清晰的职责分离设计，以计算器类库的形式提供核心功能。

## 模块结构

```
src/radiation/
├── __init__.py                  # 模块统一导出接口
├── constants.py                 # 物理常量定义
├── solar_radiation.py           # 太阳辐射计算
├── calc_net_radiation.py        # 净辐射计算器
├── calc_soil_heat.py            # 土壤热通量计算器
├── calc_sensible_heat.py        # 感热通量计算器
├── calc_latent_heat.py          # 潜热通量计算器
├── balance_equation.py          # 能量平衡协调函数
├── raster_io.py                 # GeoTIFF 文件 I/O
└── README.md                    # 本文档
```

## 核心功能

### 1. `constants.py` - 物理常量

定义所有物理常量，避免硬编码和重复定义：

```python
from radiation import (
    STEFAN_BOLTZMANN,          # Stefan-Boltzmann 常数
    SOLAR_CONSTANT,            # 太阳辐射常数
    AIR_DENSITY,               # 空气密度
    SPECIFIC_HEAT_AIR,         # 空气比热容
    LATENT_HEAT_VAPORIZATION,  # 水汽化潜热
    PSYCHROMETRIC_CONSTANT     # 干湿计常数
)
```

### 2. `solar_radiation.py` - 太阳辐射计算

**职责**：
- 计算太阳几何常量（日地距离、太阳赤纬、时角、高度角、方位角）
- 基于 DEM 数组计算短波下行辐射
- 考虑地形效应（坡度、坡向）
- 高程修正的大气透射率计算

**核心类**：
- `SolarConstantsCalculator`: 太阳几何常量计算
- `SolarRadiationCalculator`: DEM 数组辐射计算

**关键公式**：
```python
# 大气透射率（高程修正）
τ_sw = 0.75 + 2×10⁻⁵ × Elevation

# 短波下行辐射
S↓ = Gsc × cosθ × dr × τ_sw
```

### 3. `calc_net_radiation.py` - 净辐射计算

**公式**: `Q* = (1 - α)S↓ + ε₀L↓ - L↑`

**计算组件**:
- `L↓ = εₐσTₐ⁴` - 长波下行辐射
- `L↑ = ε₀σTₛ⁴` - 长波上行辐射
- `εₐ = 0.85 × (-ln τsw)^0.09` - 大气发射率

**输入参数**:
- 短波下行辐射 S↓ (W/m²)
- 地表温度 Ts (K)
- 近地表气温 Ta (K)
- 高程 (m)
- 地表反照率 α (0-1)
- 地表发射率 ε₀ (0-1) - **作为输入参数**

### 4. `calc_soil_heat.py` - 土壤热通量计算

**公式**: `ΔQSg/Q* = (Ts-273.15)/α × (0.0038α + 0.0074α²) × (1-0.98NDVI⁴)`

**特殊处理**: 不透水面的土壤热通量 ≈ 0

### 5. `calc_sensible_heat.py` - 感热通量计算

**公式**: `QH = ρCp(Ta - Ts)/rah`

**输入参数**:
- 大气湍流热交换阻抗 rah (s/m) - **作为输入参数**

### 6. `calc_latent_heat.py` - 潜热通量计算

**公式**: `QE = ρCp(es - ea)/[γ(rah + rs)]`

**输入参数**:
- 大气湍流热交换阻抗 rah (s/m) - **作为输入参数**
- 表面阻抗 rs (s/m) - **作为输入参数**

### 7. `balance_equation.py` - 能量平衡协调

提供高层函数协调所有计算器：
- `calculate_energy_balance()` - 完整能量平衡计算
- `calculate_evaporative_fraction()` - 蒸散发比计算

**能量平衡方程**: `Q* = ΔQSg + QH + QE`

### 8. `raster_io.py` - 栅格数据 I/O

**功能**：
- `load_dem()` - 读取 GeoTIFF 格式的 DEM 文件
- `save_radiation()` - 保存计算结果为 GeoTIFF 文件

## 使用示例

### 示例 1: 计算短波下行辐射

```python
from datetime import datetime
import numpy as np
from radiation import calculate_dem_solar_radiation
from radiation import load_dem, save_radiation

# 加载 DEM
dem_array, metadata = load_dem('./data/dem/wuhan_dem.tif')

# 提取地理变换信息
transform = metadata['transform']
if hasattr(transform, 'c'):  # Affine 对象
    geotransform = (transform.c, transform.a, 0, transform.f, 0, transform.e)
else:
    geotransform = transform

# 计算短波辐射
radiation_array = calculate_dem_solar_radiation(
    dem_array=dem_array,
    dem_geotransform=geotransform,
    datetime_obj=datetime(2025, 6, 21, 12, 0, 0),
    std_meridian=120.0,
    consider_terrain=True
)

# 保存结果
save_radiation(
    radiation_array,
    './output/Rs_down_20250621.tif',
    metadata
)

print(f"最大辐射: {radiation_array.max():.2f} W/m²")
print(f"平均辐射: {radiation_array[radiation_array>0].mean():.2f} W/m²")
```

### 示例 2: 使用单个计算器

```python
import numpy as np
from radiation import NetRadiationCalculator

# 准备输入数据（示例数据）
shortwave_down = np.random.rand(100, 100) * 800  # W/m²
surface_temperature = np.random.rand(100, 100) * 20 + 290  # K
air_temperature = 300.0  # K
elevation = np.random.rand(100, 100) * 1000  # m
albedo = np.random.rand(100, 100) * 0.3  # 0-1
surface_emissivity = np.random.rand(100, 100) * 0.1 + 0.9  # 0-1

# 创建计算器
net_rad_calc = NetRadiationCalculator(
    shortwave_down=shortwave_down,
    surface_temperature=surface_temperature,
    air_temperature=air_temperature,
    elevation=elevation,
    albedo=albedo,
    surface_emissivity=surface_emissivity
)

# 获取计算结果
Q_star = net_rad_calc.net_radiation
L_down = net_rad_calc.longwave_down
L_up = net_rad_calc.longwave_up

print(f"净辐射: {Q_star.mean():.2f} W/m²")
print(f"长波下行: {L_down.mean():.2f} W/m²")
print(f"长波上行: {L_up.mean():.2f} W/m²")
```

### 示例 3: 完整能量平衡计算

```python
import numpy as np
from radiation import calculate_energy_balance

# 准备所有输入数据
shortwave_down = ...              # 短波下行辐射 (W/m²) - 从示例1获得
surface_temperature = ...         # 地表温度 (K) - 遥感反演
air_temperature = 300.0           # 近地表气温 (K) - 气象数据
elevation = ...                   # 高程 (m) - DEM
albedo = ...                      # 地表反照率 (0-1) - 遥感计算
ndvi = ...                        # 归一化植被指数 - 遥感计算
aerodynamic_resistance = ...      # rah (s/m) - 外部估算或输入
surface_resistance = ...          # rs (s/m) - 外部估算或输入
actual_vapor_pressure = ...       # ea (kPa) - 气象数据
surface_emissivity = ...          # ε₀ (0-1) - 外部计算或输入

# 计算完整能量平衡
result = calculate_energy_balance(
    shortwave_down=shortwave_down,
    surface_temperature=surface_temperature,
    air_temperature=air_temperature,
    elevation=elevation,
    albedo=albedo,
    ndvi=ndvi,
    aerodynamic_resistance=aerodynamic_resistance,
    surface_resistance=surface_resistance,
    actual_vapor_pressure=actual_vapor_pressure,
    surface_emissivity=surface_emissivity
)

# 结果包含所有能量通量
print(f"净辐射 Q*: {result['net_radiation'].mean():.2f} W/m²")
print(f"土壤热通量 ΔQSg: {result['soil_heat_flux'].mean():.2f} W/m²")
print(f"感热通量 QH: {result['sensible_heat_flux'].mean():.2f} W/m²")
print(f"潜热通量 QE: {result['latent_heat_flux'].mean():.2f} W/m²")
print(f"能量平衡残差: {result['energy_balance_residual'].mean():.2f} W/m²")
print(f"可用能量: {result['available_energy'].mean():.2f} W/m²")
```

## 数据流程图

```
┌─────────────────┐
│   DEM.tif       │
│  (GeoTIFF)      │
└────────┬────────┘
         │
         ↓ raster_io.load_dem()
┌─────────────────┐
│  DEM ndarray    │
│  + metadata     │
└────────┬────────┘
         │
         ↓ solar_radiation.calculate_dem_solar_radiation()
         │  ├─ SolarConstantsCalculator (太阳几何)
         │  ├─ calculate_slope_aspect (地形)
         │  └─ SolarRadiationCalculator (辐射计算)
         │
┌─────────────────┐
│  S↓ ndarray     │
│  (短波下行辐射) │
└────────┬────────┘
         │
         ↓ 准备其他输入数据
         │  ├─ 地表温度 Ts (遥感反演)
         │  ├─ 地表发射率 ε₀ (NDVI 阈值法)
         │  ├─ 阻抗参数 rah, rs (外部估算)
         │  └─ 其他气象/遥感数据
         │
         ↓ balance_equation.calculate_energy_balance()
         │  ├─ NetRadiationCalculator (Q*)
         │  ├─ SoilHeatFluxCalculator (ΔQSg)
         │  ├─ SensibleHeatFluxCalculator (QH)
         │  └─ LatentHeatFluxCalculator (QE)
         │
┌─────────────────┐
│  Energy Fluxes  │
│  (各通量数组)   │
└────────┬────────┘
         │
         ↓ raster_io.save_radiation()
┌─────────────────┐
│  Output.tif     │
│  (GeoTIFF)      │
└─────────────────┘
```

## 理论基础

### SEBAL 模型

本模块基于 SEBAL (Surface Energy Balance Algorithm for Land) 模型实现。

**核心思想**：
- 利用遥感数据计算地表能量平衡
- 通过能量平衡方程计算蒸散发
- 考虑高程、地形对辐射的影响

**能量平衡方程**：
```
Q* = ΔQSg + QH + QE
```

其中：
- Q*: 净辐射 (W/m²)
- ΔQSg: 土壤热通量 (W/m²)
- QH: 感热通量 (W/m²)
- QE: 潜热通量 (W/m²)

**参考文献**：
1. Bastiaanssen et al., 1998. SEBAL: 1. Formulation. Journal of Hydrology, 212-213.
2. Laipelt et al., 2021. Long-term monitoring of evapotranspiration using the SEBAL algorithm.

### 城市生态空间降温模型

针对城市环境的特殊处理：
- 不透水面的土壤热通量近似为 0
- 城市地表异质性对阻抗的影响需要特殊考虑
- 支持 LCZ (Local Climate Zone) 分类处理

## 输入数据要求

### 必需的遥感数据
1. **DEM** (数字高程模型)
   - 格式：GeoTIFF
   - 坐标系：地理坐标系统（经纬度）

2. **地表温度 Ts**
   - 来源：热红外遥感反演
   - 单位：K (开尔文)

3. **地表反照率 α**
   - 来源：多光谱遥感计算
   - 范围：0-1

4. **NDVI**
   - 来源：多光谱遥感计算
   - 范围：-1 到 1

### 必需的气象数据
1. **近地表气温 Ta** (K)
2. **实际水汽压 ea** (kPa)

### 必需的外部估算参数
1. **地表发射率 ε₀** (0-1)
   - 可使用 NDVI 阈值法计算

2. **大气湍流热交换阻抗 rah** (s/m)
   - 与地表粗糙度、风速相关
   - 需要根据 LCZ 分类或地表特征估算

3. **表面阻抗 rs** (s/m)
   - 与植被类型、土壤湿度相关
   - 需要根据植被覆盖度和类型估算

## 输出结果

### 能量通量输出
各能量通量均为 ndarray 格式：
- `net_radiation`: 净辐射 Q* (W/m²)
- `soil_heat_flux`: 土壤热通量 ΔQSg (W/m²)
- `sensible_heat_flux`: 感热通量 QH (W/m²)
- `latent_heat_flux`: 潜热通量 QE (W/m²)
- `longwave_down`: 长波下行辐射 L↓ (W/m²)
- `longwave_up`: 长波上行辐射 L↑ (W/m²)
- `available_energy`: 可用能量 Q* - ΔQSg (W/m²)
- `energy_balance_residual`: 能量平衡残差 (W/m²)

### 文件格式
- 格式：GeoTIFF (.tif)
- 数据类型：Float32
- 单位：W/m²
- 空间分辨率：与输入 DEM 相同
- 坐标系统：与输入 DEM 相同

## 依赖包

```
numpy>=1.20.0
rasterio>=1.2.0  # 用于 GeoTIFF 读写
```

安装：
```bash
pip install -r requirements.txt
```

## 注意事项

1. **坐标系统**：DEM 必须使用地理坐标系统（经纬度）

2. **数据质量**：
   - DEM 质量直接影响坡度/坡向计算
   - 地表温度数据需要大气校正
   - 气象数据应与遥感数据时间同步

3. **参数估算**：
   - 地表发射率 ε₀ 可使用 NDVI 阈值法计算
   - 阻抗参数 rah 和 rs 需要根据地表特征和气象条件估算
   - 不同地表类型需要不同的参数化方案

4. **计算效率**：
   - 大尺寸 DEM 建议分块处理
   - 使用 numpy 矢量化操作提高速度

5. **物理合理性**：
   - 检查能量平衡残差（通常 <10% 可接受）
   - 验证各通量的合理范围
   - 对比实测数据验证

## 常见问题

**Q: 为什么透射率使用高程修正？**

A: 高海拔地区大气更稀薄，透射率更高。公式 `τ_sw = 0.75 + 2×10⁻⁵ × Ele` 基于 Laipelt et al. (2021) 的研究。

**Q: 不透水面的土壤热通量为什么设为 0？**

A: 不透水面（如混凝土、沥青）的热量主要通过建筑物传递，而非土壤传导。

**Q: 如何选择标准子午线？**

A: 根据时区选择：中国使用 120°（UTC+8），美国东部使用 -75°（UTC-5）。

**Q: 能量平衡为什么不闭合？**

A: 可能原因包括：
- 输入数据误差
- 阻抗参数估算不准
- 模型假设的局限性
- 数值计算误差

通常残差 <10% 可接受。

**Q: 如何估算阻抗参数 rah 和 rs？**

A: 这是 SEBAL 模型中最具挑战性的部分：
- **rah**: 需要根据地表粗糙度、风速、大气稳定度等参数计算，通常与 LCZ 分类相关
- **rs**: 需要根据植被类型、LAI、土壤水分等参数估算，参考 Penman-Monteith 方程

建议查阅相关文献获取特定地表类型的参数化方案。

**Q: 大气发射率公式中的指数为什么是 0.09 而不是 0.99？**

A: 根据实际应用和文献验证，指数 0.09 提供了更合理的结果。指数 0.99 可能是文献中的打字错误。

## 开发路线图

- [x] 短波辐射计算
- [x] 净辐射计算
- [x] 土壤热通量计算
- [x] 感热通量计算
- [x] 潜热通量计算
- [x] 模块结构重构
- [ ] 阻抗参数自动估算
- [ ] 地表发射率自动计算（NDVI 阈值法）
- [ ] LCZ 分类集成
- [ ] 并行计算支持
- [ ] 时间序列批处理
- [ ] 可视化工具

## 版本历史

- **v0.2.0** (2025-11-21): 模块结构重构，移除工作流代码，提供清晰的计算器类库
- **v0.1.0** (2025-11-12): 初始版本，包含基础 SEBAL 计算功能

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**最后更新**: 2025-11-21
