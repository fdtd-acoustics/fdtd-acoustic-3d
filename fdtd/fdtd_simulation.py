import math

import taichi as ti
from visualization import config
from .source import Source


def get_time_step(dimensions, dx, speed, safety_factor):
    courant_limit = 1.0 / math.sqrt(dimensions)
    return (dx / speed) * courant_limit * safety_factor


@ti.data_oriented
class FDTD_Simulation:
    def __init__(self,DX: float, N: int, c: float,source_object: Source):
        self.source = source_object
        wavelength = c / self.source.freq
        self.C = c
        self.DX = DX
        self.N = N

        self.courant = (1.0 / math.sqrt(config.DIM)) * config.safety_factor
        self.courant_sq = self.courant ** 2
        self.dt = get_time_step(config.DIM, self.DX, c, config.safety_factor)
        self.steps = 0

        self.alpha_field = ti.field(dtype=ti.f32, shape=(self.N, self.N, self.N)) # pozniej jako parametr
        self.alpha_A = ti.field(dtype=ti.f32, shape=(self.N, self.N, self.N))
        self.alpha_B = ti.field(dtype=ti.f32, shape=(self.N, self.N, self.N))

        self.k_field = ti.field(dtype=ti.i32, shape=(self.N, self.N, self.N))
        self.bk_field = ti.field(dtype=ti.f32, shape=(self.N, self.N, self.N))
        self.p_prev = ti.field(ti.f32, shape=(self.N, self.N, self.N))  # macierz ciśnień, do niej wsadzane są najnowsze wartosci
        self.p_curr = ti.field(ti.f32, shape=(self.N, self.N, self.N))  # macierz ciśnień

        self.buffers = [self.p_prev, self.p_curr]

        self.generate_simulation_map()

    @ti.func
    def calculate_alpha_A(self, c: ti.f32, alpha: ti.f32, density: ti.f32) -> ti.f32:
        return alpha * (density * c ** 2 + 1.0 / density)

    @ti.func
    def calculate_alpha_B(sellf, c: ti.f32, alpha: ti.f32) -> ti.f32:
        return (c ** 2) * (alpha ** 2)

    @ti.func
    def beta_from_alpha(self, alpha: ti.f32) -> ti.f32:
        Beta = 0.0
        if alpha >= 1.0:  # otwarta przestrzeń
            Beta = 1.0
        elif alpha <= 0.0:  # idealna ściana
            Beta = 0.0
        else:
            R = ti.sqrt(1.0 - alpha)
            Beta = (1.0 - R) / (1.0 + R)

        return Beta

    @ti.kernel
    def generate_simulation_map(self):
        for i, j, k in self.alpha_field:  # domyslnie wszedzie 4 sąsiadów i otwarta przestrzen
            self.alpha_field[i, j, k] = 0.0
            self.alpha_A[i, j, k] = 0.0
            self.alpha_B[i, j, k] = 0.0

            count = 0
            if i > 1: count += 1
            if i < self.N - 2: count += 1
            if j > 1: count += 1
            if j < self.N - 2: count += 1
            if k > 1: count += 1
            if k < self.N - 2: count += 1
            self.k_field[i, j, k] = count

        for i, j, k in self.alpha_A:  # sciany
            if (i == config.PML_THICK or i == self.N - config.PML_THICK - 1 or j == config.PML_THICK or
                    j == self.N - config.PML_THICK - 1 or k == config.PML_THICK or k == self.N - config.PML_THICK - 1):
                self.alpha_field[i, j, k] = 0 #0.03  # brick_alpha  "brick": (0.03,1800.0),
                self.alpha_A[i, j, k] = 0 #self.calculate_alpha_A(self.C, 0.03, 1800.0)
                self.alpha_B[i, j, k] = 0 #self.calculate_alpha_B(self.C, 0.03)

        # tworzymy sciany pml
        for i, j, k in self.alpha_A:
            dist_x = 0.0
            if i < config.PML_THICK:
                dist_x = ti.cast(config.PML_THICK - i, ti.f32) / config.PML_THICK
            elif i >= self.N - config.PML_THICK:
                dist_x = ti.cast(i - (self.N - config.PML_THICK) + 1, ti.f32) / config.PML_THICK

            dist_y = 0.0
            if j < config.PML_THICK:
                dist_y = ti.cast(config.PML_THICK - j, ti.f32) / config.PML_THICK
            elif j >= self.N - config.PML_THICK:
                dist_y = ti.cast(j - (self.N - config.PML_THICK) + 1, ti.f32) / config.PML_THICK

            dist_z = 0.0
            if k < config.PML_THICK:
                dist_z = ti.cast(config.PML_THICK - k, ti.f32) / config.PML_THICK
            elif k >= self.N - config.PML_THICK:
                dist_z = ti.cast(k - (self.N - config.PML_THICK) + 1, ti.f32) / config.PML_THICK

            dist_final = ti.max(dist_x, dist_y, dist_z)
            if dist_final > 0.0:
                alpha_val = config.alpha_max * (dist_final ** 3)
                self.alpha_field[i, j, k] = alpha_val
                self.alpha_A[i, j, k] = self.calculate_alpha_A(self.C, alpha_val, 1.21) # gestosc powietrza
                self.alpha_B[i, j, k] = self.calculate_alpha_B(self. C, alpha_val)

        for i, j, k in self.bk_field:
            k_val = self.k_field[i, j, k]
            beta_val = self.beta_from_alpha(self.alpha_field[i, j, k])
            self.bk_field[i, j, k] = (6.0 - float(k_val)) * self.courant * beta_val
            if i == 0 or i == self.N - 1 or j == 0 or j == self.N - 1 or k == 0 or k == self.N - 1:
                self.p_curr[i, j, k] = 0.0
                self.p_prev[i, j, k] = 0.0


    @ti.kernel
    def step(self, p_prev: ti.template(), p_curr: ti.template(), steps: ti.i32):
        for i, j, k in ti.ndrange((1, self.N - 1), (1, self.N - 1),(1, self.N - 1)):
            k_val = self.k_field[i, j, k]
            bk_val = self.bk_field[i, j, k]
            alpha_a = self.alpha_A[i, j, k]
            alpha_b = self.alpha_B[i, j, k]

            w1 = 1.0 + bk_val + alpha_a / 2 * self.dt + alpha_b * (self.dt ** 2)
            w2 = self.courant_sq * (p_curr[i + 1, j, k] + p_curr[i - 1, j, k] + p_curr[i, j + 1, k] + p_curr[i, j - 1, k] + p_curr[i, j, k + 1] + p_curr[i, j, k - 1])
            w3 = (2.0 - k_val * self.courant_sq) * p_curr[i, j, k]
            w4 = (bk_val - 1.0 - (alpha_a / 2.0) * self.dt) * p_prev[i, j, k]

            p_prev[i, j, k] = (1.0 / w1) * (w2 + w3 + w4)


        # ZRODLO
        pulse = self.source.get_pulse(steps, self.dt)
        self.p_prev[self.source.x, self.source.y, self.source.z] += pulse  # soft source
        print(f"{self.p_prev[self.source.x, self.source.y, self.source.z]}")

    def update(self):
        self.steps += 1
        p_past = self.buffers[self.steps % 2]
        p_present = self.buffers[(self.steps + 1) % 2]

        self.step(p_past, p_present, self.steps)

    def get_current_pressure(self):
        return self.buffers[self.steps % 2]