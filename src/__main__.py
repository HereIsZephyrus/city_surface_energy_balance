"""
城市地表能量平衡计算 - 命令行接口

基于《晴朗无风条件下城市生态空间对城市降温作用量化模型》的工作流实现。

模式:
    physics:    物理计算（能量平衡系数）
    regression: 街区回归（ALS求解气温）
    full:       完整工作流（physics + regression）

工作流程:
    1. 数据加载与对齐 (ERA5 + Landsat + DEM + LCZ)
    2. 空气动力学参数计算 (密度、风速、水汽压、阻抗)
    3. 能量平衡系数计算 (∂f/∂Ta, residual)
    4. 街区聚合与ALS回归求解 (可选)
    5. 结果输出

使用方法:
    # 物理计算模式
    python -m src physics --era5 <era5.tif> --landsat <landsat.tif> --dem <dem.tif> \\
        --lcz <lcz.tif> --datetime 202308151030 -o <output.tif>
    
    # 街区回归模式（使用physics模块的缓存）
    python -m src regression --cachedir <cache_dir> --districts <districts.gpkg> \\
        -o <result.gpkg>
    
    # 完整工作流
    python -m src full --era5 <era5.tif> --landsat <landsat.tif> --dem <dem.tif> \\
        --lcz <lcz.tif> --datetime 202308151030 --districts <districts.gpkg> \\
        --cachedir <cache_dir> -o <result.gpkg>

LCZ编码:
    1-9:  城市建筑类型
    10:   裸岩/铺装 (E)
    11:   密集树木 (A)
    12:   灌木/低矮植被 (C/D)
    13:   裸土/沙地 (F)
    14:   水体 (G)

参考文献:
    doc/晴朗无风条件下城市生态空间对城市降温作用量化模型.md
"""

import argparse
import sys


def create_physics_parser(subparsers):
    """创建物理计算模式的参数解析器"""
    parser = subparsers.add_parser(
        'physics',
        help='物理计算模式 - 计算能量平衡系数',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='计算栅格级能量平衡系数，包括空气动力学参数和辐射平衡'
    )
    
    # 输入文件
    parser.add_argument('--era5', required=True, help='ERA5 tif文件路径')
    parser.add_argument('--landsat', required=True, help='Landsat tif文件路径')
    parser.add_argument('--dem', required=True, help='DEM tif文件路径')
    parser.add_argument('--lcz', required=True, help='LCZ栅格文件路径 (.tif)')
    parser.add_argument('--building', type=str, default=None,
                        help='建筑数据文件路径 (.gpkg，可选)')
    
    # 时间参数
    parser.add_argument('--datetime', required=True,
                        help='观测日期时间 (格式: YYYYMMDDHHMM)')
    parser.add_argument('--std-meridian', type=float, default=120.0,
                        help='标准经度 (默认: 120.0 东八区)')
    
    # 坐标系和分辨率
    parser.add_argument('--crs', default='EPSG:32650', help='目标坐标系')
    parser.add_argument('--resolution', type=float, default=10.0,
                        help='目标分辨率(m)')
    
    # 缓存选项
    parser.add_argument('--cachedir', type=str,
                        help='缓存目录路径（启用后数据存储到磁盘）')
    parser.add_argument('--restart', action='store_true',
                        help='强制重新计算并覆盖缓存')
    
    # 输出
    parser.add_argument('-o', '--output', required=True,
                        help='输出tif文件路径')
    
    return parser


def create_regression_parser(subparsers):
    """创建回归分析模式的参数解析器"""
    parser = subparsers.add_parser(
        'regression',
        help='街区回归模式 - ALS求解气温',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='从能量平衡系数求解各街区气温'
    )
    
    # 输入
    parser.add_argument('--cachedir', required=True,
                        help='缓存目录路径（physics模块输出）')
    parser.add_argument('--districts', required=True,
                        help='街区矢量数据路径 (.gpkg)')
    
    # 回归特征
    parser.add_argument('--district-id', default='district_id',
                        help='街区ID字段名 (默认: district_id)')
    parser.add_argument('--x-f', type=str, default=None,
                        help='人为热Q_F特征列名（连续变量），逗号分隔')
    parser.add_argument('--x-s', type=str, default=None,
                        help='储热ΔQ_Sb特征列名（连续变量），逗号分隔')
    parser.add_argument('--x-c', type=str, default=None,
                        help='分类特征列名（将进行one-hot编码），逗号分隔')
    
    # 空间自相关参数（水平交换项 ΔQ_A）
    parser.add_argument('--distance-threshold', type=float, default=5000.0,
                        help='空间权重距离阈值(m)，街区边界距离小于此值视为邻居 (默认: 5000)')
    parser.add_argument('--distance-decay', type=str, default='binary',
                        choices=['binary', 'linear', 'inverse', 'gaussian'],
                        help='距离衰减函数 (默认: binary)')
    
    # 回归参数
    parser.add_argument('--max-iter', type=int, default=20,
                        help='ALS最大迭代次数 (默认: 20)')
    parser.add_argument('--tol', type=float, default=1e-4,
                        help='收敛容差 (默认: 1e-4)')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式')
    
    # 输出
    parser.add_argument('-o', '--output', required=True,
                        help='输出文件路径 (.gpkg/.csv)')
    
    return parser


def create_full_parser(subparsers):
    """创建完整工作流模式的参数解析器"""
    parser = subparsers.add_parser(
        'full',
        help='完整工作流 - physics + regression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='执行完整工作流：物理计算 + 街区回归'
    )
    
    # === 物理计算参数 ===
    input_group = parser.add_argument_group('输入数据')
    input_group.add_argument('--era5', required=True, help='ERA5 tif文件路径')
    input_group.add_argument('--landsat', required=True, help='Landsat tif文件路径')
    input_group.add_argument('--dem', required=True, help='DEM tif文件路径')
    input_group.add_argument('--lcz', required=True, help='LCZ栅格文件路径')
    input_group.add_argument('--building', type=str, default=None,
                             help='建筑数据文件路径 (.gpkg)')
    input_group.add_argument('--districts', required=True,
                             help='街区矢量数据路径 (.gpkg)')
    
    time_group = parser.add_argument_group('时间参数')
    time_group.add_argument('--datetime', required=True,
                            help='观测日期时间 (格式: YYYYMMDDHHMM)')
    time_group.add_argument('--std-meridian', type=float, default=120.0,
                            help='标准经度 (默认: 120.0)')
    
    geo_group = parser.add_argument_group('地理参数')
    geo_group.add_argument('--crs', default='EPSG:32650', help='目标坐标系')
    geo_group.add_argument('--resolution', type=float, default=10.0,
                           help='目标分辨率(m)')
    
    # === 回归参数 ===
    reg_group = parser.add_argument_group('回归参数')
    reg_group.add_argument('--district-id', default='district_id',
                           help='街区ID字段名')
    reg_group.add_argument('--x-f', type=str, default=None,
                           help='人为热Q_F特征列名（连续变量），逗号分隔')
    reg_group.add_argument('--x-s', type=str, default=None,
                           help='储热ΔQ_Sb特征列名（连续变量），逗号分隔')
    reg_group.add_argument('--x-c', type=str, default=None,
                           help='分类特征列名（将进行one-hot编码），逗号分隔')
    reg_group.add_argument('--max-iter', type=int, default=20,
                           help='ALS最大迭代次数')
    reg_group.add_argument('--tol', type=float, default=1e-4,
                           help='收敛容差')
    
    # === 空间自相关参数（水平交换项 ΔQ_A）===
    spatial_group = parser.add_argument_group('空间自相关参数')
    spatial_group.add_argument('--distance-threshold', type=float, default=5000.0,
                               help='空间权重距离阈值(m) (默认: 5000)')
    spatial_group.add_argument('--distance-decay', type=str, default='binary',
                               choices=['binary', 'linear', 'inverse', 'gaussian'],
                               help='距离衰减函数 (默认: binary)')
    
    # === 缓存和输出 ===
    cache_group = parser.add_argument_group('缓存和输出')
    cache_group.add_argument('--cachedir', type=str, required=True,
                             help='缓存目录路径')
    cache_group.add_argument('--restart', action='store_true',
                             help='强制重新计算')
    cache_group.add_argument('-o', '--output', required=True,
                             help='输出文件路径')
    cache_group.add_argument('--quiet', action='store_true',
                             help='静默模式')
    
    return parser


def run_physics(args):
    """执行物理计算模式"""
    from .physics import main as physics_main
    physics_main(args)


def run_regression(args):
    """执行回归分析模式"""
    from .anylysis import main as anylysis_main
    anylysis_main(args)


def run_full(args):
    """执行完整工作流"""
    from pathlib import Path
    from types import SimpleNamespace
    
    print("\n" + "=" * 60)
    print("完整工作流: 物理计算 + 街区回归")
    print("=" * 60)
    
    # 第一步: 物理计算
    print("\n>>> 阶段 1/2: 物理计算 <<<")
    
    # 构建 physics 参数
    physics_output = str(Path(args.output).with_suffix('.tif'))
    if '_regression' in physics_output:
        physics_output = physics_output.replace('_regression', '_physics')
    elif not physics_output.endswith('_physics.tif'):
        physics_output = physics_output.replace('.tif', '_physics.tif')
    
    # 创建一个模拟的 args 对象用于 physics
    physics_args = SimpleNamespace(
        era5=args.era5,
        landsat=args.landsat,
        dem=args.dem,
        lcz=args.lcz,
        building=args.building,
        datetime=args.datetime,
        std_meridian=getattr(args, 'std_meridian', 120.0),
        crs=args.crs,
        resolution=args.resolution,
        cachedir=args.cachedir,
        restart=args.restart,
        output=physics_output
    )
    
    from .physics import main as physics_main
    physics_main(physics_args)
    
    # 第二步: 街区回归
    print("\n>>> 阶段 2/2: 街区回归 <<<")
    
    # 构建 regression 参数
    anylysis_args = SimpleNamespace(
        cachedir=args.cachedir,
        districts=args.districts,
        output=args.output,
        district_id=getattr(args, 'district_id', 'district_id'),
        x_f=getattr(args, 'x_f', None),
        x_s=getattr(args, 'x_s', None),
        x_c=getattr(args, 'x_c', None),
        distance_threshold=getattr(args, 'distance_threshold', 5000.0),
        distance_decay=getattr(args, 'distance_decay', 'binary'),
        max_iter=getattr(args, 'max_iter', 20),
        tol=getattr(args, 'tol', 1e-4),
        quiet=getattr(args, 'quiet', False)
    )
    
    from .anylysis import main as anylysis_main
    anylysis_main(anylysis_args)
    
    print("\n" + "=" * 60)
    print("✓ 完整工作流完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  物理计算: {physics_output}")
    print(f"  回归结果: {args.output}")


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        prog='python -m src',
        description='城市地表能量平衡计算',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式说明:
    physics     计算栅格级能量平衡系数
    regression  从能量平衡系数求解街区气温（需要先运行physics）
    full        执行完整工作流（physics + regression）

示例:
    # 物理计算
    python -m src physics --era5 era5.tif --landsat landsat.tif --dem dem.tif \\
        --lcz lcz.tif --datetime 202308151030 -o result.tif
    
    # 街区回归
    python -m src regression --cachedir ./cache --districts districts.gpkg \\
        -o result.gpkg
    
    # 完整工作流
    python -m src full --era5 era5.tif --landsat landsat.tif --dem dem.tif \\
        --lcz lcz.tif --datetime 202308151030 --districts districts.gpkg \\
        --cachedir ./cache -o result.gpkg
        """
    )
    
    # 创建子命令
    subparsers = parser.add_subparsers(
        dest='mode',
        title='运行模式',
        description='选择运行模式',
        metavar='MODE'
    )
    
    # 添加各模式的解析器
    create_physics_parser(subparsers)
    create_regression_parser(subparsers)
    create_full_parser(subparsers)
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有指定模式，显示帮助
    if args.mode is None:
        parser.print_help()
        sys.exit(0)
    
    # 根据模式执行
    if args.mode == 'physics':
        run_physics(args)
    elif args.mode == 'regression':
        run_regression(args)
    elif args.mode == 'full':
        run_full(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
