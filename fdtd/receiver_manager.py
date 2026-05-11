import os
from pathlib import Path

import matplotlib
import numpy as np
import taichi as ti
from scipy.io import wavfile

import config

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt

@ti.data_oriented
class ReceiverManager:
    def __init__(
        self,
        max_receivers: int,
        max_steps: int,
        dt,
        output_dir: str | Path | None = None,
        save_plots: bool = True,
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
        self.pos = ti.Vector.field(3, dtype=ti.i32, shape=self.max_receivers)
        self.history = ti.field(dtype=ti.f32, shape=(self.max_receivers, max_steps))

    def add_receiver(self, x, y, z, name):
        idx = self.count[None]
        if idx < self.max_receivers:
            self.pos[idx] = [x, y, z]
            self.names.append(name)
            self.count[None] += 1

    @classmethod
    def build_receiver_manager(
        cls,
        receivers: list[dict],
        max_time: float,
        dt: float,
        output_dir: str | Path | None = None,
        save_plots: bool = True,
    ) -> "ReceiverManager":
        max_steps = int(np.ceil(max_time / dt))
        manager = cls(
            max_receivers=len(receivers),
            max_steps=max_steps,
            dt=dt,
            output_dir=output_dir,
            save_plots=save_plots,
        )

        for r in receivers:
            x = int(r.get('x', 50))
            y = int(r.get('y', 50))
            z = int(r.get('z', 50))

            name = r.get('name', f"mic_{manager.count[None] + 1}")

            manager.add_receiver(x, y, z, name)

        return manager

    def update_receivers(self, p_field: ti.template(), step: ti.i32):
        if step < self.max_steps:
            self.record_step_kernel(p_field, step)
        elif self.saved == 0:
            self.save_all()

    def save_all(self) -> dict[str, np.ndarray]:
        if self.saved:
            return {}
        full_history = self.history.to_numpy()
        result: dict[str, np.ndarray] = {}
        for i in range(self.count[None]):
            name = self.names[i]
            data = full_history[i, :]
            self.save_to_wav(name, data)
            if self._save_plots:
                self.save_plot(name, data)
            result[name] = data
        self.saved = 1
        return result

    @ti.kernel
    def record_step_kernel(self, p_field: ti.template(), step: ti.i32):
        for i in range(self.count[None]):
            pos = self.pos[i]
            self.history[i, step] = p_field[pos[0], pos[1], pos[2]]

    def save_to_wav(self, filename: str, pressure_data: np.ndarray):
        directory = self._output_dir if self._output_dir is not None else Path(config.WAV_DIR)
        os.makedirs(directory, exist_ok=True)

        if not filename.endswith('.wav'):
            filename += '.wav'
        file_path = os.path.join(directory, filename)

        fs = int(1.0 / self.dt)

        pressure = pressure_data.copy()
        pressure -= np.mean(pressure)

        max_val = np.max(np.abs(pressure))
        if max_val > 0:
            pressure = pressure / max_val

        pressure_int16 = (pressure * 32767).astype(np.int16)
        wavfile.write(file_path, fs, pressure_int16)
        print(f"WAV SAVED: {file_path} (FS: {fs} Hz)")

    def save_plot(self, filename: str, pressure_data: np.ndarray):
        if self._output_dir is not None:
            directory = self._output_dir / "plots"
        else:
            directory = Path(config.PLOT_DIR)
        os.makedirs(directory, exist_ok=True)

        if not filename.endswith(".png"):
            filename += ".png"
        file_path = os.path.join(directory, filename)

        pressure = pressure_data.copy()
        pressure -= np.mean(pressure)

        max_val = np.max(np.abs(pressure))
        if max_val > 0:
            pressure = pressure / max_val

        time_axis = np.arange(len(pressure)) * self.dt

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, pressure, color="#1f77b4", linewidth=1)
        plt.title(f"Pressure Signal - {filename} (Normalized)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.ylim(-1.1, 1.1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        plt.close()

        print(f"PLOT SAVED: {file_path}")
