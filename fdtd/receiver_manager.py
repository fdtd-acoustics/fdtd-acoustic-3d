import taichi as ti
from visualization import config, PML_THICK
from scipy.io import wavfile
import numpy as np

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
            self.pos[idx] = [x + PML_THICK, y + PML_THICK, z+ PML_THICK]
            self.names.append(name)
            self.count[None] += 1

    def update_receivers(self, p_field: ti.template(), step: ti.i32):
        if step < self.max_steps:
            self.record_step_kernel(p_field, step)
        elif self.saved ==0:             # to jest tymczasowo
            self.save_to_wav("Test.wav", 0)
            self.saved = 1

    @ti.kernel
    def record_step_kernel(self, p_field: ti.template(), step: ti.i32):
        for i in range(self.count[None]):
            pos = self.pos[i]
            self.history[i, step] = p_field[pos[0], pos[1], pos[2]]

    def save_to_wav(self, filename: str, index: int):
        history_np = self.history.to_numpy()
        pressure = history_np[index, :]

        fs = int(1.0/self.dt)
        pressure = pressure - np.mean(pressure)

        max_val = np.max(np.abs(pressure))
        if max_val > 0:
            pressure = pressure / max_val # normalizacja

        wavfile.write(filename, fs, pressure)
        print(f"SAVED: {filename} (FS: {fs} Hz)")