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
        pml_thick: int,
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
            x_val, y_val, z_val = r.get('x'), r.get('y'), r.get('z')

            x = int(x_val) + pml_thick
            y = int(y_val) + pml_thick
            z = int(z_val) + pml_thick

            name = r.get('name', f"mic_{manager.count[None] + 1}")

            manager.add_receiver(x, y, z, name)

        return manager

    def update_receivers(self, p_field: ti.template(), step: ti.i32):
        if step < self.max_steps:
            self.record_step_kernel(p_field, step)
        elif self.saved == 0:
            self.save_all()

    def save_all(self, use_global_normalization: bool = True) -> dict[str, np.ndarray]:
        if self.saved:
            return {}
        full_history = self.history.to_numpy()
        result: dict[str, np.ndarray] = {}

        global_max = None
        if use_global_normalization:
            centered_history = full_history - np.mean(full_history, axis=1, keepdims=True)
            global_max = np.max(np.abs(centered_history))
            if global_max == 0: global_max = 1.0

        for i in range(self.count[None]):
            name = self.names[i]
            data = full_history[i, :]
            self.save_to_wav(name, data, global_max=global_max)
            if self._save_plots:
                self.save_plot(name, data, global_max=global_max)
                self.save_spl_plot(name, data, global_max=global_max)
                self.save_fft_plot(name, data)
            result[name] = data
        self.saved = 1
        return result

    @ti.kernel
    def record_step_kernel(self, p_field: ti.template(), step: ti.i32):
        for i in range(self.count[None]):
            pos = self.pos[i]
            self.history[i, step] = p_field[pos[0], pos[1], pos[2]]

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

    def save_plot(self, filename: str, pressure_data: np.ndarray, global_max: float = None):
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

        max = global_max if global_max is not None else np.max(np.abs(pressure))
        if max > 0:
            pressure = pressure / max

        time_axis = np.arange(len(pressure)) * self.dt

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, pressure, color="#1f77b4", linewidth=1)

        mode = "Globally" if global_max is not None else "Locally"
        plt.title(f"Pressure Signal - {filename} ({mode} Normalized)")

        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.ylim(-1.1, 1.1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        plt.close()

        print(f"PLOT SAVED: {file_path} (Mode: {mode})")

    def save_fft_plot(self, filename: str, pressure_data: np.ndarray):
        if self._output_dir is not None:
            directory = self._output_dir / "plots"
        else:
            directory = Path(config.PLOT_DIR)
        os.makedirs(directory, exist_ok=True)

        fft_filename = filename.replace('.png', '_fft.png') if filename.endswith('.png') else f"{filename}_fft.png"
        file_path = os.path.join(directory, fft_filename)

        fs = int(1.0 / self.dt)
        n = len(pressure_data)

        data = pressure_data - np.mean(pressure_data)
        yf = np.fft.rfft(data)   # real fast fourier transform - zmiana dziedziny czasu na dziedzinę czestotliowsci
        xf = np.fft.rfftfreq(n, self.dt)

        magnitude = 20 * np.log10(np.abs(yf) + 1e-10) # tu zdobywamy względne dB danych częstotliwości

        plt.figure(figsize=(10, 4))
        plt.semilogx(xf, magnitude, color="#ff7f0e", linewidth=1)
        plt.title(f"Frequency Spectrum (FFT) - {filename}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.xlim(20, fs / 2)  # pasmo słyszalne
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        plt.close()
        print(f"FFT PLOT SAVED: {file_path}")

    def save_spl_plot(self, filename: str, pressure_data: np.ndarray, global_max: float = None):
        if self._output_dir is not None:
            directory = self._output_dir / "plots"
        else:
            directory = Path(config.PLOT_DIR)
        os.makedirs(directory, exist_ok=True)

        spl_filename = filename.replace('.png', '_spl.png') if filename.endswith('.png') else f"{filename}_spl.png"
        file_path = os.path.join(directory, spl_filename)

        pressure = pressure_data.copy()
        pressure -= np.mean(pressure)

        max = global_max if global_max is not None else np.max(np.abs(pressure))
        if max > 0:
            pressure = pressure / max

        pressure_db = 20 * np.log10(np.abs(pressure) + 1e-10) # tutaj nie dzielimy ciśnienia, ponieważ chcemy żeby wynik był względny
        # ponieważ nie jesteśmy w stanie określić fizycznej wartości ciśnienia w Pascalach
        pressure_db = np.maximum(pressure_db, -80)

        time_axis = np.arange(len(pressure)) * self.dt

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, pressure_db, color="#d62728", linewidth=1)

        mode = "Globally" if global_max is not None else "Locally"
        plt.title(f"Sound Pressure Level (SPL) - {filename} ({mode} Normalized)")

        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude [dB relative to peak]")
        plt.ylim(-80, 5)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        plt.close()

        print(f"SPL PLOT SAVED: {file_path} (Mode: {mode})")



