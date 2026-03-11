import math
import taichi as ti

@ti.data_oriented
class Source:
    def __init__(self, x, y, z, freq, amplitude):
        self.x = x
        self.y = y
        self.z = z
        self.freq = freq
        self.amplitude = amplitude
        self.sigma = math.sqrt(2 * math.log(2)) / (2 * math.pi * freq)
        self.delay = 4 * self.sigma

    @ti.func
    def get_pulse(self, steps: ti.i32, DT: ti.f32) -> ti.f32:
        return self.amplitude * ti.exp(- ((DT * steps - DT * self.delay) ** 2) / (2 * self.sigma ** 2))