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
    output_dir: Path | None
    audio: dict[str, np.ndarray]
    raw_pressures: np.ndarray | None
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
    def from_npz(cls, path: str | Path, overrides: dict | None = None) -> "HeadlessRunner":
        with np.load(path, allow_pickle=True) as data:
            loaded_dict = dict(data)
        return cls.from_data(loaded_dict, overrides=overrides, npz_path=str(path))

    @classmethod
    def from_data(cls, data: dict, overrides: dict | None = None, npz_path: str = "in_memory",) -> "HeadlessRunner":
        sources_list = data['sources'].tolist()
        receivers_list = data['receivers'].tolist()

        cfg = {
            "scene": {
                "obj_file": str(data.get('obj_filepath', 'in_memory')),
                "pml_thickness": int(data['pml_thick']),
                "alpha_max": float(data['alpha_max']),
                "record_time": float(data['record_time']),
            },
            "physics": {
                "sound_speed": float(data['sound_speed']),
                "nodes_per_wavelength": float(data['nodes_per_wavelength']),
                "safety_factor": float(data['safety_factor']),
            },
            "sources": sources_list,
            "receivers": receivers_list,
            "run": {
                "save_plots": True,
                "output": "output/runs/"
            },

            "_is_npz": True,
            "_npz_path": str(npz_path),
            "_grid_params": {
                "dx": float(data['dx']),
                "dt": float(data['dt']),
                "Nx": int(data['Nx']),
                "Ny": int(data['Ny']),
                "Nz": int(data['Nz']),
            }
        }

        if overrides:
            cfg = _apply_overrides(cfg, overrides)

        runner = cls(cfg)
        runner._raw_data = data
        return runner

    def run(
        self,
        output_dir: str | Path | None = None,
        save_plots: bool | None = None,
        progress: bool = True,
        dim: int = 3,
        write_files: bool = True,
    ) -> RunResult:
        cfg = self._cfg
        scene = cfg["scene"]
        physics = cfg["physics"]
        sources = cfg["sources"]
        receivers = cfg["receivers"]
        run_cfg = cfg["run"]

        resolved_save_plots = run_cfg["save_plots"] if save_plots is None else save_plots
        if not write_files:
            resolved_save_plots = False

        out_dir = None
        if write_files:
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
                dim=dim,
                safety_factor=physics["safety_factor"],
            )

            is_npz_mode = cfg.get("_is_npz", False)
            if is_npz_mode:
                sim_config.npz_filepath = cfg["_npz_path"]

            builder = SimulationBuilder(sim_config)
            if is_npz_mode:
                if hasattr(self, "_raw_data") and self._raw_data is not None:
                    loaded_data = self._raw_data
                else:
                    with np.load(cfg["_npz_path"], allow_pickle=True) as data:
                        loaded_data = dict(data)

                grid = builder.grid_from_data(loaded_data)

                if not builder.validate_dx(grid, sources):
                    max_freq = builder.get_max_safe_frequency(grid.dx)
                    raise ValueError(
                        f"Frequency Too High: The grid max safe frequency is {max_freq:.0f} Hz"
                    )
            else:
                waveform_specs = [s["waveform"] for s in sources]
                grid = builder.compute_grid(waveform_specs)

                #Todo: to pewnie bedzie do zmiany bo to dziala jak cache ktory sie nie zmienia jak zmienisz
                # freq wiec pewnie cos w wokselizerze do dodania zeby hashowal config z jakim wokselizuje
                # if not Path(sim_config.npz_filepath).is_file():
                builder.voxelize(grid)

            total_steps = int(math.ceil(scene["record_time"] / grid.dt))

            grid_inner = (grid.Nx, grid.Ny, grid.Nz)

            source_manager = self._build_source_manager(
                sources, grid.dt, total_steps, grid.dx, grid_inner, sim_config.pml_thick, dim
            )
            receiver_manager = self._build_receiver_manager(
                receivers, grid.dt, total_steps, grid.dx, grid_inner, out_dir, resolved_save_plots, sim_config.pml_thick, dim
            )

            fdtd_sim = builder.build_fdtd(
                grid=grid,
                source_manager=source_manager,
                receiver_manager=receiver_manager,
                data=loaded_data if is_npz_mode else None
            )

            wall_clock_start = time.perf_counter()
            self._run_loop(fdtd_sim, total_steps, progress)
            raw_pressures = receiver_manager.get_raw_pressures()

            audio = receiver_manager.save_all() if write_files else {}
            wall_clock_s = time.perf_counter() - wall_clock_start

            if progress:
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
                dim=dim
            )

            if write_files:
                self._write_artifacts(out_dir, cfg, metadata)
        except BaseException as exc:
            print(f"Run failed")
            if write_files:
                failed_dir = self._mark_run_failed(out_dir, cfg, exc)
                print(f"Artifacts moved to: {failed_dir}")
            raise

        return RunResult(
            output_dir=out_dir,
            audio=audio,
            raw_pressures=raw_pressures,
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
        pml_thick: int,
        dim: int = 3,
    ) -> SourceManager:
        manager = SourceManager(max_sources=len(sources), max_steps=total_steps, dim=dim)
        for idx, src in enumerate(sources):
            emit_steps = min(int(math.ceil(src["time"] / dt)), total_steps)
            waveform_array = waveform_factory.synthesize(src["waveform"], dt, emit_steps)
            if emit_steps < total_steps:
                waveform_array = np.concatenate([
                    waveform_array,
                    np.zeros(total_steps - emit_steps, dtype=np.float32),
                ])
            waveform_array = np.ascontiguousarray(waveform_array, dtype=np.float32)
            coords = _world_to_voxel(src["coords"], grid_inner, dx)

            x_val = coords[0] + pml_thick
            y_val = coords[1] + pml_thick
            z_val = coords[2] + pml_thick if dim == 3 else None

            manager.add_source(
                name=f"src_{idx}",
                waveform_array=waveform_array,
                x=x_val,
                y=y_val,
                z=z_val
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
        pml_thick: int,
        dim: int = 3,
    ) -> ReceiverManager:
        manager = ReceiverManager(
            max_receivers=len(receivers),
            max_steps=total_steps,
            dt=dt,
            output_dir=output_dir,
            save_plots=save_plots,
            dim=dim
        )
        for rec in receivers:
            coords = _world_to_voxel(rec["coords"], grid_inner, dx)

            x_val = coords[0] + pml_thick
            y_val = coords[1] + pml_thick
            z_val = coords[2] + pml_thick if dim == 3 else None

            manager.add_receiver(name=rec["name"], x=x_val, y=y_val, z=z_val)
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
        dim: int = 3
    ) -> dict:
        Nx_inner, Ny_inner, Nz_inner = grid_inner

        config_bytes = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
        config_hash = "sha256:" + hashlib.sha256(config_bytes).hexdigest()

        grid_meta = {
            "Nx_inner": Nx_inner,
            "Ny_inner": Ny_inner,
            "Nx_total": Nx_inner + 2 * pml,
            "Ny_total": Ny_inner + 2 * pml,
            "pml_thickness": pml,
            "dx": float(dx),
            "dt": float(dt),
            "total_steps": int(total_steps),
        }

        if dim == 3:
            grid_meta["Nz_inner"] = Nz_inner
            grid_meta["Nz_total"] = Nz_inner + 2 * pml

        return {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_clock_s": round(wall_clock_s, 3),
            "config_hash": config_hash,
            "grid": grid_meta,
            "sources": [
                {
                    "index": i,
                    "coords": list(src["coords"]),
                    "waveform_type": src["waveform"]["type"],
                    "waveform": src["waveform"],
                }
                for i, src in enumerate(sources)
            ],
            "receivers": [
                {
                    "index": i,
                    "name": rec["name"],
                    "coords": list(rec["coords"]),
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

    ndim = len(pos)
    grid_inner_trimmed = np.asarray(grid_inner[:ndim], dtype=np.float64)

    grid_center = (grid_inner_trimmed - 1.0) / 2.0
    idx = np.floor(pos / dx + grid_center + 0.5).astype(int)

    if len(pos) == 2:
        return int(idx[0]), int(idx[1])
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

# mysle ze jest lepsze miejsce na tą funkcje niz tu, ale no trzeba by bylo w ogole zając sie struktura projektu
FOAM_MATERIAL_ID = 16


def simulate_2d(data: dict, foam: list[int] | None = None, save_plots: bool = False) -> RunResult:
    if foam:
        material_core = np.array(data['material_core'], copy=True)
        foam_slots = data['foam_slots']
        for slot_idx in foam:
            for cx, cy in foam_slots[slot_idx]['cells']:
                material_core[cx, cy] = FOAM_MATERIAL_ID
        data = {**data, 'material_core': material_core}

    runner = HeadlessRunner.from_data(data)
    result = runner.run(save_plots=save_plots, progress=False, dim=2, write_files=False)

    return result