#!/usr/bin/env python
"""
离线修复缓存 .npy 栅格的脚本。

使用方式:
    python script/sanitize_cache.py --cache-dir /path/to/cache

可选参数:
    --bands landsat_lst landsat_ndvi   # 仅处理指定波段
    --quiet                            # 禁止打印详细信息
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.cache_sanitizer import sanitize_cache_arrays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对已生成的缓存 .npy 栅格进行范围裁剪与 nodata 清洗"
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="缓存目录路径（包含 metadata.pkl 和若干 .npy 文件）"
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        help="可选，仅处理指定波段（默认全部标准波段 + lcz）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，不输出详细统计"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(os.path.expanduser(args.cache_dir)).resolve()

    if not cache_dir.exists():
        raise FileNotFoundError(f"缓存目录不存在: {cache_dir}")

    if not args.quiet:
        print("=" * 60)
        print(f"缓存合法化: {cache_dir}")
        print("=" * 60)

    sanitize_cache_arrays(
        cache_dir,
        verbose=not args.quiet,
        include_bands=args.bands
    )

    if not args.quiet:
        print("\n✓ 处理完成")


if __name__ == "__main__":
    main()

