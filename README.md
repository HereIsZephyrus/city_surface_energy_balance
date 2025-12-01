# City Surface Energy Balance Model

!This Doc is archived by AI
[中文版本](#中文版本)

A quantitative model for estimating near-surface air temperature and analyzing the cooling effect of urban ecological spaces under clear and windless conditions.

## Overview

This project implements a surface energy balance model based on remote sensing data to estimate near-surface (2m) air temperature in urban areas. The model addresses three key challenges in urban heat island research:

1. **LST vs. Air Temperature**: Remote sensing-derived Land Surface Temperature (LST) cannot fully represent thermal comfort. This model estimates near-surface air temperature from LST using energy balance equations.

2. **Semantic Gap in Urban Landscape Description**: Traditional pixel-level LCZ (Local Climate Zone) classification creates a "semantic gap" with the actual LCZ concept (which is block-scale, not pixel-scale). This model uses graph neural networks and spectral clustering to create block-scale LCZ boundaries.

3. **Urban-Specific Energy Balance**: Most remote sensing energy balance models (like SEBAL) are designed for natural surfaces and lack urban-specific considerations. This model introduces semi-empirical terms for anthropogenic heat, building heat storage, and horizontal heat exchange.

## Key Innovations

### 1. Block-Scale LCZ Classification
- Uses Voronoi diagrams to build building adjacency graphs
- Graph Attention Networks (GAT) for building-level LCZ prediction
- Spectral clustering for LCZ boundary identification
- Results in block-scale LCZ with clear boundaries

### 2. Near-Surface Air Temperature Estimation
- Based on surface energy balance equations
- Uses iterative least squares (ALS) to estimate air temperature
- Incorporates high-precision LCZ classification
- Estimates unquantified terms (anthropogenic heat, building storage, horizontal exchange) through regression

### 3. Cooling Effect Quantification
- Statistical method to quantify cooling effects of urban ecological spaces
- Considers adjacent LCZ influences
- Quantifies the intensity of cooling effects (variance explained)

## Model Architecture

The model consists of three main stages:

### Stage 1: Physical Calculation (Raster Level)
- Calculate energy balance coefficients (∂f/∂Ta, residual)
- Compute aerodynamic parameters (resistance, vapor pressure, etc.)
- **No air temperature Ta is needed** - only coefficients

### Stage 2: Spatial Aggregation (Block Level)
- Aggregate raster coefficients to urban blocks (districts)
- Average coefficients within each block
- Each block gets one set of coefficients

### Stage 3: Block Regression (Block Level)
- Use Alternating Least Squares (ALS) to solve for air temperature
- Each block has **one air temperature value**
- Estimate linear coefficients for QF, ΔQSb, ΔQA

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd city_surface_energy_balance

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Physical calculation mode (raster-level coefficients)
python -m src physics \
    --era5 era5.tif \
    --landsat landsat.tif \
    --albedo albedo.tif \
    --dem dem.tif \
    --lcz lcz.tif \
    --datetime 202308151030 \
    -o result.tif

# Block regression mode (solve for block-level Ta)
python -m src regression \
    --cachedir ./cache \
    --districts districts.gpkg \
    -o result.gpkg

# Full workflow (physics + regression)
python -m src full \
    --era5 era5.tif \
    --landsat landsat.tif \
    --albedo albedo.tif \
    --dem dem.tif \
    --lcz lcz.tif \
    --datetime 202308151030 \
    --districts districts.gpkg \
    --cachedir ./cache \
    -o result.gpkg
```

### Preparing Albedo

Use the helper script to derive broadband albedo from multispectral Landsat reflectance:

```bash
python script/compute_albedo.py \
    --landsat raw_landsat_stack.tif \
    --blue-band 2 --green-band 3 --red-band 4 \
    --nir-band 5 --swir1-band 6 --swir2-band 7 \
    --scale 2.75e-05 --offset -0.2 \
    -o albedo.tif
```

Adjust the band indices and scale/offset parameters to match your data product.

### Python API

```python
from src.radiation import calculate_energy_balance_coefficients
from src.regression import DistrictAggregator, DistrictRegressionModel
from src.utils import RasterCollection

# Step 1: Load data
collection = RasterCollection()
collection.add_raster('LST', 'landsat_lst.tif')
# ... load other data

# Step 2: Calculate energy balance coefficients (raster level)
coeffs = calculate_energy_balance_coefficients(...)

# Step 3: Aggregate to blocks
aggregator = DistrictAggregator()
aggregated = aggregator.aggregate_rasters_to_districts(...)

# Step 4: Regression (one Ta per block)
model = DistrictRegressionModel()
results = model.fit_als_regression(...)
```

## Data Requirements

### Required Remote Sensing Data
- **DEM**: Digital Elevation Model (GeoTIFF)
- **Landsat**: Multi-band surface products (GeoTIFF, contains LST/NDVI/FVC/emissivity)
- **Albedo**: Broadband surface albedo (GeoTIFF, 单独文件，可用 `script/compute_albedo.py` 生成)
- **LCZ**: Local Climate Zone classification (GeoTIFF, 1-14)
- **Building Data**: Optional, for roughness calculation (GeoPackage)

### Required Meteorological Data (ERA5-Land)
- **Surface Pressure**: For air density calculation
- **Dewpoint Temperature**: For actual vapor pressure
- **Wind Speed Components**: For aerodynamic resistance
- **Note**: Air temperature is **NOT used** - it's the model's target variable

### Required Vector Data
- **Districts**: Urban block polygons (GeoPackage/Shapefile)

## Output

### Physical Calculation Output
- `f_Ta_coeff`: Coefficient of Ta in energy balance equation (W/m²/K)
- `residual`: Constant term (independent of Ta) (W/m²)
- `era5_air_temperature`: ERA5 temperature (for reference only)

### Regression Output
- `Ta_optimized`: Estimated near-surface air temperature (K)
- `Ta_celsius`: Air temperature in Celsius (°C)
- Regression coefficients for QF, ΔQSb, ΔQA

## Project Structure

```
city_surface_energy_balance/
├── src/
│   ├── radiation/          # Radiation and energy balance calculation
│   ├── aerodynamics/       # Aerodynamic parameters (resistance, vapor pressure)
│   ├── regression/         # Block-level regression (ALS)
│   ├── landscape/          # Landscape parameters (roughness, frontal area)
│   ├── utils/              # Utilities (raster management, mapping)
│   └── physics.py          # Main physics workflow
├── doc/                    # Documentation (Chinese)
│   ├── 算法思想.md         # Algorithm ideas
│   ├── 数学模型.md         # Mathematical models
│   ├── 具体实现.md         # Implementation details
│   └── 使用指南.md         # User guide
├── doc-old/                # Legacy documentation
├── examples/               # Example scripts
└── test/                   # Test scripts
```

## Documentation

### English
- This README
- Code docstrings and inline comments

### Chinese (中文)
Comprehensive Chinese documentation is available in the `doc/` directory:
- **算法思想.md**: Algorithm concepts and design philosophy
- **数学模型.md**: Mathematical formulations and equations
- **具体实现.md**: Implementation details and code structure
- **使用指南.md**: User guide with examples

## Key Features

- **Coefficient Decomposition**: Calculates Ta coefficients without knowing Ta values
- **Block-Scale Processing**: Each urban block has one air temperature value
- **ALS Regression**: Alternating Least Squares for parameter estimation
- **ERA5 Integration**: Uses ERA5-Land for meteorological parameters
- **LCZ Support**: Full support for Local Climate Zone classification
- **Caching System**: Efficient data caching for large-scale processing

## Energy Balance Equation

The complete energy balance equation is:

```
Q* + QF = QH + QE + ΔQSb + ΔQSg + ΔQA
```

Where:
- **Q***: Net radiation (quantifiable)
- **QF**: Anthropogenic heat (estimated via regression)
- **QH**: Sensible heat flux (quantifiable, depends on Ta)
- **QE**: Latent heat flux (quantifiable, depends on Ta)
- **ΔQSb**: Building heat storage (estimated via regression)
- **ΔQSg**: Soil heat flux (quantifiable)
- **ΔQA**: Horizontal heat exchange (estimated via regression)

The model calculates the quantifiable part:
```
Quantified(Ta) = Q*(Ta) - ΔQSg(Ta) - QH(Ta) - QE(Ta)
```

And estimates the unquantified part through regression:
```
Unquantified = QF + ΔQSb + ΔQA ≈ Quantified(Ta)
```

## LCZ Classification

The model uses a simplified LCZ scheme:

**Urban Building Types (1-9)**:
- 1: Compact High-rise
- 2: Compact Mid-rise
- 3: Compact Low-rise
- 4: Open High-rise
- 5: Open Mid-rise
- 6: Open Low-rise
- 7: Lightweight Low-rise
- 8: Large Low-rise
- 9: Sparsely Built

**Natural/Surface Types (10-14)**:
- 10: Bare Rock/Paved (E)
- 11: Dense Trees (A)
- 12: Bush/Grass (C/D)
- 13: Bare Soil/Sand (F)
- 14: Water (G)

## Assumptions

1. **Clear and Windless Conditions**: The model assumes clear sky and low wind speed (< 3-4 m/s)
2. **Block-Scale Uniformity**: Air temperature is uniform within each urban block
3. **Local Effects**: Cooling effects are local and first-order (only affect adjacent blocks)
4. **Unstable Atmosphere**: Daytime clear conditions lead to unstable atmospheric conditions

## Limitations

- Requires clear sky conditions (no clouds)
- Low wind speed assumption may limit applicability
- LCZ classification accuracy depends on input data quality
- Energy balance closure depends on parameter estimation accuracy

## Citation

If you use this model in your research, please cite:

```
City Surface Energy Balance Model
A quantitative model for urban heat island analysis under clear and windless conditions
```

## References

1. Bastiaanssen et al., 1998. SEBAL: Surface Energy Balance Algorithm for Land. Journal of Hydrology.
2. Oke & Stewart, 2012. Local Climate Zone classification for urban studies.
3. ERA5-Land Dataset Documentation

## License

See LICENSE file for details.

## Contact

For questions or issues, please open an issue on the repository.

---

## 中文版本

### 项目概述

本项目实现了《晴朗无风条件下城市生态空间对城市降温作用量化模型》，用于估算城市近地表（2米）气温并分析生态空间的降温效益。

### 核心创新

1. **街区尺度LCZ分类**：基于图神经网络和谱聚类的街区级LCZ划分
2. **近地表气温估算**：基于地表能量平衡方程，使用迭代最小二乘法估算气温
3. **降温效益量化**：统计方法量化城市生态空间的降温作用

### 快速开始

```bash
# 物理计算模式
python -m src physics --era5 era5.tif --landsat landsat.tif --dem dem.tif \
    --lcz lcz.tif --datetime 202308151030 -o result.tif

# 街区回归模式
python -m src regression --cachedir ./cache --districts districts.gpkg \
    -o result.gpkg

# 完整工作流
python -m src full --era5 era5.tif --landsat landsat.tif --dem dem.tif \
    --lcz lcz.tif --datetime 202308151030 --districts districts.gpkg \
    --cachedir ./cache -o result.gpkg
```

### 详细文档

中文详细文档位于 `doc/` 目录：
- **算法思想.md**：算法设计思路和核心思想
- **数学模型.md**：数学公式和物理方程
- **具体实现.md**：代码实现细节
- **使用指南.md**：使用说明和示例

### 数据需求

- **遥感数据**：DEM、Landsat（地表温度、反照率、NDVI）、LCZ分类
- **气象数据**：ERA5-Land（气压、露点温度、风速）
- **矢量数据**：城市街区多边形

### 输出结果

- 栅格级能量平衡系数（∂f/∂Ta, residual）
- 街区级近地表气温（每个街区一个Ta值）
- 回归系数（QF, ΔQSb, ΔQA的影响因子）

---

**Note**: For detailed Chinese documentation, please refer to the `doc/` directory.
