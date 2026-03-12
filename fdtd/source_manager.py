import math
import taichi as ti
from visualization import config, PML_THICK


@ti.data_oriented
class SourceManager:
    def __init__(self, max_sources: int):
        self.max_sources = max_sources
        self.count = ti.field(ti.i32, shape=())
        self.count[None] = 0

        self.data = ti.Struct.field({
            "pos": ti.types.vector(3, ti.i32),
            "amp": ti.f32,
            "freq": ti.f32,
            "sigma": ti.f32,
            "delay_t": ti.f32,
        }, shape=max_sources)

    def add_source(self, x, y, z, freq, amplitude):
        idx = self.count[None]

        if idx < self.max_sources:
            sigma = math.sqrt(2 * math.log(2)) / (2 * math.pi * freq)
            self.data[idx].pos = [x + config.PML_THICK, y + PML_THICK, z + PML_THICK] # uwzgledniamy warstwe pml
            self.data[idx].amp = amplitude
            self.data[idx].freq = freq
            self.data[idx].sigma = sigma
            self.data[idx].delay_t = 4 * sigma
            self.count[None] += 1

    @ti.kernel
    def update_sources(self, p_field: ti.template(), steps: ti.i32, dt:ti.f32):
        if steps < 10: #tymczasowo dodalem, tylko przez 10 krokow zrodlo dziala
            for i in range(self.count[None]):
                s = self.data[i]
                val = s.amp * ti.exp(- ((dt * steps - dt * s.delay_t) ** 2) / (2 * s.sigma ** 2))
                p_field[s.pos[0],s.pos[1], s.pos[2]] += val

    def get_max_freq(self) -> float:
        max_f = 0.0
        for i in range(self.count[None]):
            current_f = self.data[i].freq
            if current_f > max_f:
                max_f = current_f
        return max_f