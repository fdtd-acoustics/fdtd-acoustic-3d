import taichi as ti
from scipy.io import wavfile
import numpy as np
import config
import os
import matplotlib.pyplot as plt

@ti.data_oriented
class ReceiverManager:
    def __init__(self,max_receivers: int, max_steps: int, dt):
        self.max_receivers = max_receivers
        self.max_steps = max_steps
        self.dt = dt
        self.saved = 0
        self.count = ti.field(ti.i32, shape=())
        self.count[None] = 0

        self.names = []
        self.pos = ti.Vector.field(3, dtype=ti.i32, shape=max_receivers)
        self.history = ti.field(dtype=ti.f32, shape=(max_receivers, max_steps))

    def add_receiver(self, x, y, z, name):
        idx = self.count[None]
        if idx < self.max_receivers:
            self.pos[idx] = [x, y, z]
            self.names.append(name)
            self.count[None] += 1

    def update_receivers(self, p_field: ti.template(), step: ti.i32):
        if step < self.max_steps:
            self.record_step_kernel(p_field, step)
        elif self.saved ==0:             # to jest tymczasowo
            self.save_to_wav("mik4", 0)
            self.save_plot("mik4", 0)
            self.save_to_wav("src1", 1)
            self.save_plot("src1", 1)
            self.save_to_wav("mic5", 2)
            self.save_plot("mic5", 2)
            self.saved = 1

    @ti.kernel
    def record_step_kernel(self, p_field: ti.template(), step: ti.i32):
        for i in range(self.count[None]):
            pos = self.pos[i]
            self.history[i, step] = p_field[pos[0], pos[1], pos[2]]

    def save_to_wav(self, filename: str, index: int):
        directory = config.WAV_DIR
        os.makedirs(directory, exist_ok=True)

        if not filename.endswith('.wav'):
            filename += '.wav'

        file_path = os.path.join(directory, filename)

        history_np = self.history.to_numpy()
        pressure = history_np[index, :]

        fs = int(1.0 / self.dt)

        pressure = pressure - np.mean(pressure)

        max_val = np.max(np.abs(pressure))
        if max_val > 0:
            pressure = pressure / max_val

        pressure_int16 = (pressure * 32767).astype(np.int16)

        wavfile.write(file_path, fs, pressure_int16)
        print(f"WAV SAVED: {filename} (FS: {fs} Hz)")

    def save_plot(self, filename: str, index: int):
        directory = config.PLOT_DIR
        os.makedirs(directory, exist_ok=True)

        if not filename.endswith('.png'):
            filename += '.png'

        file_path = os.path.join(directory, filename)

        history_np = self.history.to_numpy()
        pressure = history_np[index, :]

        pressure = pressure - np.mean(pressure)

        max_val = np.max(np.abs(pressure))

        if max_val > 0:
            pressure = pressure / max_val

        time_axis = np.arange(len(pressure)) * self.dt

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, pressure, color='#1f77b4', linewidth=1)

        plt.title(f"Pressure Signal - {filename} (Normalized)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        plt.ylim(-1.1, 1.1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(file_path, dpi=150)
        plt.close()

        print(f"PLOT SAVED: {file_path}")