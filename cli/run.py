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
    parser.add_argument("config", type=Path, help="Path to scene YAML")
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
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    _init_taichi()
    from simulation.headless_runner import HeadlessRunner

    runner = HeadlessRunner.from_yaml(args.config)
    save_plots = False if args.no_plots else None
    result = runner.run(
        output_dir=args.output,
        save_plots=save_plots,
        progress=not args.quiet,
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
