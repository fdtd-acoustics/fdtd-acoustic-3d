"""argparse subparser dispatcher for the `fdtd` CLI.

Subcommands:
    gui        — launch the existing tkinter GUI (no args)
    run        — single headless simulation from a YAML config
    sweep      — ??
    validate   — validate a YAML config without running todo
    voxelize   — pre-build a scene .npz cache todo
"""

from __future__ import annotations

import argparse
import sys

from . import run as run_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fdtd",
        description="FDTD acoustic simulator — GUI, headless and batch modes.",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    gui = sub.add_parser("gui", help="Launch the interactive GUI")
    gui.set_defaults(handler=_run_gui)

    run_cmd.add_subparser(sub)

    # sweep = sub.add_parser("sweep", help="Run a parametric / random sweep ")
    # sweep.set_defaults(handler=_not_implemented("sweep"))
    #
    # validate = sub.add_parser("validate", help="Validate a YAML config without running")
    # validate.set_defaults(handler=_not_implemented("validate"))
    #
    # voxelize = sub.add_parser("voxelize", help="Pre-voxelize a scene to .npz cache")
    # voxelize.set_defaults(handler=_not_implemented("voxelize"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        return _run_gui(args)

    return args.handler(args)


def _run_gui(args: argparse.Namespace) -> int:
    import taichi as ti

    from gui import MainMenuWindow
    from visualization.vis_config import MEMORY_LIMIT_GB

    ti.init(arch=ti.gpu, device_memory_GB=MEMORY_LIMIT_GB)
    app = MainMenuWindow()
    app.mainloop()
    return 0


def _not_implemented(name: str):
    def handler(args: argparse.Namespace) -> int:
        print(f"`{name}` subcommand is not yet implemented", file=sys.stderr)
        return 2

    return handler


if __name__ == "__main__":
    raise SystemExit(main())
