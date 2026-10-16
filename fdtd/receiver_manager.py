import os
from pathlib import Path

import numpy as np
import taichi as ti
from scipy.io import wavfile

import config

@ti.data_oriented
class ReceiverManager:
    def __init__(
        self,
        max_receivers: int,
        max_steps: int,
        dt,
        output_dir: str | Path | None = None,
        save_plots: bool = True,
        dim: int = 3
    ):
        self.max_receivers = max(1, max_receivers)
        self.max_steps = max_steps
        self.dt = dt
        self.saved = 0

        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._save_plots = save_plots

        self.count = ti.field(ti.i32, shape=())
        self.count[None] = 0

        self.names = []
        self.pos = ti.Vector.field(dim, dtype=ti.i32, shape=self.max_receivers)
        self.history = ti.field(dtype=ti.f32, shape=(self.max_receivers, max_steps))

    def add_receiver(self, name: str, x: int, y: int, z: int | None = None):
        idx = self.count[None]
        if idx < self.max_receivers:
            if z is None:
                self.pos[idx] = [x, y]
            else:
                self.pos[idx] = [x, y, z]

            self.names.append(name)
            self.count[None] += 1

    @classmethod
    def build_receiver_manager(
        cls,
        receivers: list[dict],
        max_time: float,
        dt: float,
        pml_thick: int,
        output_dir: str | Path | None = None,
        save_plots: bool = True,
        dim: int = 3,
    ) -> "ReceiverManager":
        max_steps = int(np.ceil(max_time / dt))
        manager = cls(
            max_receivers=len(receivers),
            max_steps=max_steps,
            dt=dt,
            output_dir=output_dir,
            save_plots=save_plots,
            dim = dim,
        )

        for idx, rec in enumerate(receivers):
            coords = rec.get('coords')

            if coords is None or len(coords) != dim:
                raise ValueError(f"Missing or invalid coordinates for source at index {idx}. Expected {dim} coords, got {coords}")

            rec_x = int(coords[0]) + pml_thick
            rec_y = int(coords[1]) + pml_thick
            rec_z = int(coords[2]) + pml_thick if dim == 3 else None

            manager.add_receiver(
                name = rec.get('name', f"mic_{idx + 1}"),
                x = rec_x,
                y = rec_y,
                z = rec_z
            )

        return manager

    def update_receivers(self, p_field: ti.template(), step: ti.i32):
        if step < self.max_steps:
            self.record_step_kernel(p_field, step)
        elif self.saved == 0:
            self.save_all()

    def save_all(self, use_global_normalization: bool = True) -> dict[str, np.ndarray]:
        from visualization.plotter import save_spl_plot, save_plot, save_fft_plot
        if self.saved:
            return {}
        full_history = self.history.to_numpy()
        result: dict[str, np.ndarray] = {}
        positions_np = self.pos.to_numpy()

        global_max = None
        if use_global_normalization:
            centered_history = full_history - np.mean(full_history, axis=1, keepdims=True)
            global_max = np.max(np.abs(centered_history))
            if global_max == 0: global_max = 1.0

        from visualization.plotter import save_spl_plot, save_plot, save_fft_plot

        if self._output_dir is not None:
            directory = self._output_dir / "plots"
        else:
            directory = Path(config.PLOT_DIR)

        for i in range(self.count[None]):
            name = self.names[i]
            pos = positions_np[i]
            data = full_history[i, :]

            coords_str = f"({', '.join(map(str, pos))})"
            display_name = f"{name} {coords_str}"

            self.save_to_wav(name, data, global_max=global_max)
            if self._save_plots:
                save_plot(display_name, data, self.dt, directory, global_max=global_max)
                save_spl_plot(display_name, data, self.dt, directory, global_max=global_max)
                save_fft_plot(display_name, data, self.dt, directory)
            result[name] = data
        self.saved = 1
        return result

    def get_raw_pressures(self) -> np.ndarray:
        num_mics = self.count[None]
        full_history = self.history.to_numpy()

        return full_history[:num_mics, :]

    @ti.kernel
    def record_step_kernel(self, p_field: ti.template(), step: ti.i32):
        for i in range(self.count[None]):
            pos = self.pos[i]
            self.history[i, step] = p_field[pos]

    def save_to_wav(self, filename: str, pressure_data: np.ndarray, global_max: float):
        directory = self._output_dir if self._output_dir is not None else Path(config.WAV_DIR)
        os.makedirs(directory, exist_ok=True)

        if not filename.endswith('.wav'):
            filename += '.wav'
        file_path = os.path.join(directory, filename)

        fs = int(1.0 / self.dt)

        pressure = pressure_data.copy()
        pressure -= np.mean(pressure)

        max = global_max if global_max is not None else np.max(np.abs(pressure))
        if max > 0:
            pressure = pressure / max

        pressure_int16 = (pressure * 32767).astype(np.int16)
        wavfile.write(file_path, fs, pressure_int16)

        mode = "Global" if global_max is not None else "Local"
        print(f"WAV SAVED: {file_path} (FS: {fs} Hz, Mode: {mode})")





