"""Run a single FDTD simulation from a normalized config dict, no GUI.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from fdtd import ReceiverManager, SourceManager, waveform_factory, FDTD_Simulation

from .config_loader import load_config, validate_config
from .simulation_builder import SimulationBuilder
from .simulation_config import SimulationConfig


@dataclass
class RunResult:
    output_dir: Path
    audio: dict[str, np.ndarray]
    sample_rate: int
    metadata: dict
    config: dict


class HeadlessRunner:
    def __init__(self, cfg: dict):
        self._cfg = cfg

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict | None = None) -> "HeadlessRunner":
        cfg = load_config(path)
        if overrides:
            cfg = _apply_overrides(cfg, overrides)
        return cls(cfg)

    @classmethod
    def from_dict(cls, cfg: dict, overrides: dict | None = None) -> "HeadlessRunner":
        normalized = validate_config(cfg)
        if overrides:
            normalized = _apply_overrides(normalized, overrides)
        return cls(normalized)

    @classmethod
    def from_npz(cls, path: str | Path, overrides: dict | None = None) -> "HeadlessRunner":
        path = Path(path)
        cfg = _npz_to_cfg(path)
        if overrides:
            cfg = _apply_overrides(cfg, overrides)
        return cls(cfg, npz_path=path)

    def run(
        self,
        output_dir: str | Path | None = None,
        save_plots: bool | None = None,
        progress: bool = True,
    ) -> RunResult:
        cfg = self._cfg
        scene = cfg["scene"]
        physics = cfg["physics"]
        sources = cfg["sources"]
        receivers = cfg["receivers"]
        run_cfg = cfg["run"]

        resolved_save_plots = run_cfg["save_plots"] if save_plots is None else save_plots
        out_dir = self._resolve_output_dir(output_dir, run_cfg)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            sim_config = SimulationConfig(
                obj_filepath=scene["obj_file"],
                pml_thick=scene["pml_thickness"],
                alpha_max=scene["alpha_max"],
                record_time=scene["record_time"],
                sound_speed=physics["sound_speed"],
                nodes_per_wavelength=physics["nodes_per_wavelength"],
                safety_factor=physics["safety_factor"],
            )

            builder = SimulationBuilder(sim_config)

            waveform_specs = [s["waveform"] for s in sources]
            grid = builder.compute_grid(waveform_specs)
            total_steps = int(math.ceil(scene["record_time"] / grid.dt))

            #Todo: to pewnie bedzie do zmiany bo to dziala jak cache ktory sie nie zmienia jak zmienisz
            # freq wiec pewnie cos w wokselizerze do dodania zeby hashowal config z jakim wokselizuje
            if not Path(sim_config.npz_filepath).is_file():
                builder.voxelize(grid)

            grid_inner = (grid.Nx, grid.Ny, grid.Nz)

            source_manager = self._build_source_manager(
                sources, grid.dt, total_steps, grid.dx, grid_inner
            )
            receiver_manager = self._build_receiver_manager(
                receivers, grid.dt, total_steps, grid.dx, grid_inner, out_dir, resolved_save_plots
            )

            fdtd_sim = builder.build_fdtd(
                grid=grid,
                source_manager=source_manager,
                receiver_manager=receiver_manager
            )

            wall_clock_start = time.perf_counter()
            self._run_loop(fdtd_sim, total_steps, progress)
            audio = receiver_manager.save_all()
            wall_clock_s = time.perf_counter() - wall_clock_start

            print(f"Wall clock: {wall_clock_s:.3f} s")

            sample_rate = int(round(1.0 / grid.dt))
            metadata = self._build_metadata(
                cfg=cfg,
                dx=grid.dx,
                dt=grid.dt,
                total_steps=total_steps,
                grid_inner=grid_inner,
                pml=scene["pml_thickness"],
                sources=sources,
                receivers=receivers,
                wall_clock_s=wall_clock_s,
            )

            self._write_artifacts(out_dir, cfg, metadata)
        except BaseException as exc:
            failed_dir = self._mark_run_failed(out_dir, cfg, exc)
            print(f"Run failed. Artifacts moved to: {failed_dir}")
            raise

        return RunResult(
            output_dir=out_dir,
            audio=audio,
            sample_rate=sample_rate,
            metadata=metadata,
            config=cfg,
        )

    @staticmethod
    def _mark_run_failed(out_dir: Path, cfg: dict, exc: BaseException) -> Path:
        """Dump traceback to error.txt, rename folder with _FAILED suffix.
        """
        try:
            try:
                with (out_dir / "config.yaml").open("w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
            except Exception:
                pass

            with (out_dir / "error.txt").open("w") as f:
                f.write(f"failed_at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
                f.write(f"exception: {type(exc).__name__}: {exc}\n\n")
                f.write(traceback.format_exc())

            failed = out_dir.with_name(out_dir.name + "_FAILED")
            counter = 1
            while failed.exists():
                failed = out_dir.with_name(f"{out_dir.name}_FAILED_{counter}")
                counter += 1
            out_dir.rename(failed)
            return failed
        except Exception:
            return out_dir

    @staticmethod
    def _resolve_output_dir(override: str | Path | None, run_cfg: dict) -> Path:
        if override is not None:
            return Path(override)
        base = Path(run_cfg.get("output", "output/runs/"))
        if base.suffix or any(part.startswith("run_") for part in base.parts):
            return base
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base / f"run_{timestamp}"

    @staticmethod
    def _build_source_manager(
        sources: list[dict],
        dt: float,
        total_steps: int,
        dx: float,
        grid_inner: tuple[int, int, int],
    ) -> SourceManager:
        manager = SourceManager(max_sources=len(sources), max_steps=total_steps)
        for idx, src in enumerate(sources):
            waveform_array = waveform_factory.synthesize(src["waveform"], dt, total_steps)
            waveform_array = np.ascontiguousarray(waveform_array, dtype=np.float32)
            x, y, z = _world_to_voxel(src["position"], grid_inner, dx)
            manager.add_source(
                name=f"src_{idx}",
                x=x,
                y=y,
                z=z,
                waveform_array=waveform_array,
            )
        return manager

    @staticmethod
    def _build_receiver_manager(
        receivers: list[dict],
        dt: float,
        total_steps: int,
        dx: float,
        grid_inner: tuple[int, int, int],
        output_dir: Path,
        save_plots: bool,
    ) -> ReceiverManager:
        manager = ReceiverManager(
            max_receivers=len(receivers),
            max_steps=total_steps,
            dt=dt,
            output_dir=output_dir,
            save_plots=save_plots,
        )
        for rec in receivers:
            x, y, z = _world_to_voxel(rec["position"], grid_inner, dx)
            manager.add_receiver(x=x, y=y, z=z, name=rec["name"])
        return manager

    @staticmethod
    def _run_loop(fdtd_sim: FDTD_Simulation, total_steps: int, progress: bool) -> None:
        if progress:
            try:
                from tqdm import tqdm
                #pasek postepu
                iterator = tqdm(range(total_steps), desc="FDTD", unit="step")
            except ImportError:
                iterator = range(total_steps)
        else:
            iterator = range(total_steps)

        for _ in iterator:
            fdtd_sim.update()

    @staticmethod
    def _build_metadata(
        cfg: dict,
        dx: float,
        dt: float,
        total_steps: int,
        grid_inner: tuple[int, int, int],
        pml: int,
        sources: list[dict],
        receivers: list[dict],
        wall_clock_s: float,
    ) -> dict:
        Nx_inner, Ny_inner, Nz_inner = grid_inner

        config_bytes = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
        config_hash = "sha256:" + hashlib.sha256(config_bytes).hexdigest()

        return {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_clock_s": round(wall_clock_s, 3),
            "config_hash": config_hash,
            "grid": {
                "Nx_inner": Nx_inner,
                "Ny_inner": Ny_inner,
                "Nz_inner": Nz_inner,
                "Nx_total": Nx_inner + 2 * pml,
                "Ny_total": Ny_inner + 2 * pml,
                "Nz_total": Nz_inner + 2 * pml,
                "pml_thickness": pml,
                "dx": float(dx),
                "dt": float(dt),
                "total_steps": int(total_steps),
            },
            "sources": [
                {
                    "index": i,
                    "position": list(src["position"]),
                    "waveform_type": src["waveform"]["type"],
                    "waveform": src["waveform"],
                }
                for i, src in enumerate(sources)
            ],
            "receivers": [
                {
                    "index": i,
                    "name": rec["name"],
                    "position": list(rec["position"]),
                    "file": f"{rec['name']}.wav",
                }
                for i, rec in enumerate(receivers)
            ],
        }

    @staticmethod
    def _write_artifacts(out_dir: Path, cfg: dict, metadata: dict) -> None:
        with (out_dir / "config.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        with (out_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

def _world_to_voxel(
    position_m: list[float],
    grid_inner: tuple[int, int, int],
    dx: float,
) -> tuple[int, int, int]:
    """Convert room-centered meters to voxel indices.
    """
    pos = np.asarray(position_m, dtype=np.float64)
    grid_center = (np.asarray(grid_inner, dtype=np.float64) - 1.0) / 2.0
    idx = np.floor(pos / dx + grid_center + 0.5).astype(int)
    return int(idx[0]), int(idx[1]), int(idx[2])


def _apply_overrides(cfg: dict, overrides: dict) -> dict:
    """
    for parametrized yaml todo
    """
    out = json.loads(json.dumps(cfg))
    for path, value in overrides.items():
        _set_by_path(out, path, value)
    return validate_config(out)


def _set_by_path(obj, path: str, value):
    parts = _parse_path(path)
    cursor = obj
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _parse_path(path: str) -> list:
    parts: list = []
    buf = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if buf:
                parts.append(buf)
                buf = ""
        elif ch == "[":
            if buf:
                parts.append(buf)
                buf = ""
            end = path.index("]", i)
            parts.append(int(path[i + 1 : end]))
            i = end
        else:
            buf += ch
        i += 1
    if buf:
        parts.append(buf)
    return parts
