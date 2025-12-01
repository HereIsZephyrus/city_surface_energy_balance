#!/usr/bin/env python
"""
基于多光谱 Landsat 数据计算宽波段地表反照率 (albedo)。

默认使用 Liang (2001) 提供的经验系数，支持手动指定波段索引、缩放系数以及输出
范围，方便与现有 wuhanthermal 能量平衡流程集成。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import rasterio

# 默认假设输入按照 SR 缩放系数（scale=2.75e-05, offset=-0.2）存储。
DEFAULT_SCALE = 2.75e-05
DEFAULT_OFFSET = -0.2
DEFAULT_NODATA = -9999.0

# Liang (2001) 宽波段反照率系数（适用于 Landsat 7/8 反射率）
BROADBAND_COEFFS: Dict[str, float] = {
    'blue': 0.356,
    'green': 0.130,
    'red': 0.373,
    'nir': 0.085,
    'swir1': 0.072,
    'swir2': 0.115,
}
INTERCEPT = -0.0018


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从多波段 Landsat GeoTIFF 计算宽波段地表反照率 (albedo)。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--landsat', required=True, help='多波段 Landsat GeoTIFF 输入路径')
    parser.add_argument('-o', '--output', required=True, help='输出 albedo.tif 路径')

    parser.add_argument('--blue-band', type=int, default=2, help='蓝光波段 (1-based)')
    parser.add_argument('--green-band', type=int, default=3, help='绿波段 (1-based)')
    parser.add_argument('--red-band', type=int, default=4, help='红光波段 (1-based)')
    parser.add_argument('--nir-band', type=int, default=5, help='近红外波段 (1-based)')
    parser.add_argument('--swir1-band', type=int, default=6, help='短波红外1波段 (1-based)')
    parser.add_argument('--swir2-band', type=int, default=7, help='短波红外2波段 (1-based)')

    parser.add_argument(
        '--scale',
        type=float,
        default=DEFAULT_SCALE,
        help='输入数据到表面反射率的缩放系数 (设为1表示输入已是0-1反射率)',
    )
    parser.add_argument(
        '--offset',
        type=float,
        default=DEFAULT_OFFSET,
        help='输入数据到表面反射率的偏移量 (设为0表示无偏移)',
    )
    parser.add_argument('--clip-min', type=float, default=0.0, help='输出裁剪下限')
    parser.add_argument('--clip-max', type=float, default=1.0, help='输出裁剪上限')
    parser.add_argument('--nodata', type=float, default=DEFAULT_NODATA, help='输出nodata值')
    parser.add_argument('--overwrite', action='store_true', help='若输出已存在则覆盖')

    return parser.parse_args()


def _read_reflectance(
    dataset: rasterio.io.DatasetReader,
    band_index: int,
    scale: float,
    offset: float,
) -> np.ndarray:
    if band_index < 1 or band_index > dataset.count:
        raise ValueError(f"波段 {band_index} 超出范围 (1-{dataset.count})")

    raw = dataset.read(band_index).astype(np.float32)
    nodata = dataset.nodatavals[band_index - 1]
    invalid = ~np.isfinite(raw)
    if nodata is not None:
        invalid |= (raw == nodata)

    reflectance = raw * scale + offset
    reflectance[invalid] = np.nan
    return reflectance


def compute_albedo(bands: Dict[str, np.ndarray]) -> np.ndarray:
    albedo = np.full_like(next(iter(bands.values())), INTERCEPT, dtype=np.float32)
    for name, coeff in BROADBAND_COEFFS.items():
        if name not in bands:
            raise KeyError(f"缺少必需波段: {name}")
        albedo += coeff * bands[name]
    return albedo


def main():
    args = parse_args()

    landsat_path = Path(args.landsat)
    output_path = Path(args.output)

    if not landsat_path.exists():
        raise FileNotFoundError(f"Landsat 文件不存在: {landsat_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在: {output_path}，如需覆盖请使用 --overwrite")

    band_indices = {
        'blue': args.blue_band,
        'green': args.green_band,
        'red': args.red_band,
        'nir': args.nir_band,
        'swir1': args.swir1_band,
        'swir2': args.swir2_band,
    }

    with rasterio.open(landsat_path) as src:
        bands = {
            name: _read_reflectance(src, index, args.scale, args.offset)
            for name, index in band_indices.items()
        }

        albedo = compute_albedo(bands)
        if args.clip_min is not None or args.clip_max is not None:
            albedo = np.clip(albedo, args.clip_min, args.clip_max)

        valid_mask = np.isfinite(albedo)
        if valid_mask.any():
            albedo_min = float(np.nanmin(albedo[valid_mask]))
            albedo_max = float(np.nanmax(albedo[valid_mask]))
            albedo_mean = float(np.nanmean(albedo[valid_mask]))
            print(f"✓ Albedo 统计: 均值={albedo_mean:.3f}, 范围=[{albedo_min:.3f}, {albedo_max:.3f}]")
        else:
            print("⚠ 未得到有效的 albedo 像元，可能输入全为 nodata。")

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype='float32',
            nodata=args.nodata,
            driver='GTiff',
            compress=profile.get('compress', 'deflate')
        )
        data_to_write = albedo.astype(np.float32)
        mask = ~np.isfinite(data_to_write)
        if args.nodata is None:
            data_to_write[mask] = np.nan
        elif np.isnan(args.nodata):
            data_to_write[mask] = np.nan
        else:
            data_to_write[mask] = args.nodata

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data_to_write, 1)
            dst.set_band_description(1, 'albedo')

    print(f"✓ 已保存 albedo 至: {output_path}")


if __name__ == '__main__':
    main()

