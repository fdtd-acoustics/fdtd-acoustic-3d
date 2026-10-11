"""`fdtd run <config.yaml>` — single headless simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

def add_subparser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "run",
        help="Run a single headless simulation from a YAML config",
        description="Run one FDTD simulation defined by a YAML config and write the results to disk.",
    )
    parser.add_argument("config", type=Path, help="Path to scene YAML or .npz file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: output/runs/run_<timestamp>/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plot generation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar",
    )
    parser.add_argument(
        "--2d",
        dest="dim_2d",
        action="store_true",
        help="Run simulation in 2D mode (requires .npz config file)"
    )
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    is_npz = args.config.suffix.lower() == ".npz"

    if args.dim_2d and not is_npz:
        sys.exit("Error: 2D mode (--2d) requires a .npz file as input")
    if is_npz and not args.dim_2d:
        sys.exit("Error: Loading .npz files is currently only supported in 2D mode. Add the --2d flag.")

    _init_taichi()
    from simulation.headless_runner import HeadlessRunner

    if is_npz:
        runner = HeadlessRunner.from_npz(args.config)
    else:
        runner = HeadlessRunner.from_yaml(args.config)

    save_plots = False if args.no_plots else None
    dim = 2 if args.dim_2d else 3

    result = runner.run(
        output_dir=args.output,
        save_plots=save_plots,
        progress=not args.quiet,
        dim = dim,
    )

    print(f"Run completed. Output: {result.output_dir}")
    print(f"  receivers: {list(result.audio.keys())}")
    print(f"  sample_rate: {result.sample_rate} Hz")
    print(f"  wall_clock: {result.metadata['wall_clock_s']} s")
    return 0


def _init_taichi() -> None:
    import taichi as ti

    try:
        from visualization.vis_config import MEMORY_LIMIT_GB
    except Exception:
        MEMORY_LIMIT_GB = 4.0

    ti.init(arch=ti.gpu, device_memory_GB=MEMORY_LIMIT_GB)
