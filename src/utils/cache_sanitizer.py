"""
Cache sanitization helpers for raster collections.

These routines make it possible to fix already-generated cache folders
by directly editing the ``*.npy`` arrays in-place, applying the same
thresholding logic that used to run inline in ``physics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SanitizationRule:
    """Configuration for clamping a raster band to a valid range."""

    name: str
    min_val: float | None
    max_val: float | None
    nodata_values: Tuple[float, ...] = ()
    nodata_below: float | None = None


DEFAULT_SANITIZATION_RULES: Tuple[SanitizationRule, ...] = (
    SanitizationRule("landsat_albedo", 0.0, 1.0, (-9999.0, -999.0, -32768.0)),
    SanitizationRule("landsat_emissivity", 0.0, 1.2, (-9999.0, -32768.0)),
    SanitizationRule("landsat_ndvi", -1.2, 1.2, (-9999.0,)),
    SanitizationRule("landsat_fvc", 0.0, 1.0, (-9999.0,)),
    SanitizationRule("landsat_lst", 150.0, 400.0, (-9999.0, 0.0, -32768.0), -100.0),
    SanitizationRule("dem", -500.0, 9000.0, (-9999.0, -32768.0)),
    SanitizationRule("era5_temperature_2m", 150.0, 350.0, (-9999.0, -32768.0)),
)


def _build_mask(
    arr: np.ndarray,
    rule: SanitizationRule,
) -> np.ndarray:
    """Return a boolean mask indicating positions that should be nulled."""
    mask = ~np.isfinite(arr)
    if rule.nodata_values:
        for nodata in rule.nodata_values:
            mask |= np.isclose(arr, nodata)
    if rule.nodata_below is not None:
        mask |= arr < rule.nodata_below
    if rule.min_val is not None:
        mask |= arr < rule.min_val
    if rule.max_val is not None:
        mask |= arr > rule.max_val
    return mask


def sanitize_array(
    arr: np.ndarray,
    rule: SanitizationRule,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the rule to the provided array.

    Returns the cleaned array (float32) and the boolean mask indicating which
    pixels were touched.
    """
    working = arr.astype(np.float64, copy=True)
    mask = _build_mask(working, rule)
    if np.any(mask):
        working[mask] = np.nan
    return working.astype(np.float32), mask


def sanitize_cache_arrays(
    cache_dir: Path,
    rules: Sequence[SanitizationRule] = DEFAULT_SANITIZATION_RULES,
    *,
    verbose: bool = True,
    include_bands: Iterable[str] | None = None,
) -> None:
    """
    Sanitize cached ``*.npy`` arrays in-place.

    Args:
        cache_dir: Folder containing ``metadata.pkl`` and ``*.npy`` tiles.
        rules: Rules to apply; defaults to the standard energy-balance bands.
        verbose: Emit per-band statistics.
        include_bands: Optional whitelist; if provided, only listed bands
            (plus ``lcz``) are processed.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"缓存目录不存在: {cache_dir}")

    band_filter = set(include_bands) if include_bands is not None else None

    for rule in rules:
        if band_filter is not None and rule.name not in band_filter:
            continue

        npy_path = cache_dir / f"{rule.name}.npy"
        if not npy_path.exists():
            continue

        arr = np.load(npy_path)
        cleaned, mask = sanitize_array(arr, rule)
        if np.any(mask):
            np.save(npy_path, cleaned)
            if verbose:
                pct = mask.sum() / mask.size * 100
                print(
                    f"  - {rule.name}: 清理 {mask.sum():,d} 像素 "
                    f"({pct:.2f}%) -> {npy_path.name}"
                )

    if band_filter is not None and "lcz" not in band_filter:
        return

    _sanitize_lcz(cache_dir, verbose=verbose)


def _sanitize_lcz(cache_dir: Path, *, verbose: bool) -> None:
    """Ensure LCZ classes remain inside [0, 14]."""
    lcz_path = cache_dir / "lcz.npy"
    if not lcz_path.exists():
        return

    lcz = np.load(lcz_path)
    lcz_int = lcz.astype(np.int16, copy=True)
    invalid_mask = (lcz_int < 0) | (lcz_int > 14)
    if not np.any(invalid_mask):
        return

    lcz_int[invalid_mask] = 0
    np.save(lcz_path, lcz_int)
    if verbose:
        pct = invalid_mask.sum() / invalid_mask.size * 100
        print(
            f"  - lcz: 重置 {invalid_mask.sum():,d} 像素 "
            f"({pct:.2f}%) -> lcz.npy"
        )

