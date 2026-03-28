from doctest import NORMALIZE_WHITESPACE

import config
import taichi as ti
import math
import numpy as np


@ti.data_oriented
class FDTD_Simulation:
    def __init__(self, alpha_map_np: np.ndarray, density_map_np: np.ndarray):
        self.sound_speed = config.SOUND_SPEED

        self.courant = (1.0 / math.sqrt(config.DIM)) * config.SAFETY_FACTOR # Courant number
        self.courant_sq = self.courant ** 2 # squared Courant number

        #source
        self.src_x = config.SRC_X
        self.src_y = config.SRC_Y
        self.amplitude = config.AMPLITUDE
        self.freq_max = config.FREQ_MAX
        self.sigma = math.sqrt(2 * math.log(2)) / (2 * math.pi * self.freq_max)
        self.delay = 4 * self.sigma

        # receiver
        self.rec_x = config.REC_X
        self.rec_y = config.REC_Y
        self.history = ti.field(dtype=ti.f32, shape=(1, config.MAX_STEPS))

        wavelength = self.sound_speed / self.freq_max # wavelength [m]
        self.dx = wavelength / config.NODES_PER_WAVELENGTH # spatial step [m]

        # time step [s]
        self.dt = self.get_time_step(config.DIM, self.dx, self.sound_speed, config.SAFETY_FACTOR)

        self.pml_thickness = config.PML_THICKNESS
        self.alpha_max = config.ALPHA_MAX

        self.N = alpha_map_np.shape[0] + self.pml_thickness*2 # interior grid points in one dimension

        N_without_pml = alpha_map_np.shape[0]
        self._mat_alpha_field = ti.field(ti.f32, shape=(N_without_pml, N_without_pml))
        self._mat_density_field = ti.field(ti.f32, shape=(N_without_pml, N_without_pml))

        self._p_prev = ti.field(ti.f32, shape=(self.N, self.N))
        self._p_curr = ti.field(ti.f32, shape=(self.N, self.N))

        self._k_field = ti.field(dtype=ti.i32, shape=(self.N, self.N))
        self._bk_field = ti.field(dtype=ti.f32, shape=(self.N, self.N))

        self.alpha_field = ti.field(dtype=ti.f32, shape=(self.N, self.N))
        self.density_field = ti.field(dtype=ti.f32, shape=(self.N, self.N))

        self._alpha_A = ti.field(dtype=ti.f32, shape=(self.N, self.N))
        self._alpha_B = ti.field(dtype=ti.f32, shape=(self.N, self.N))

        self._alpha_field = ti.field(dtype=ti.f32, shape=(self.N, self.N))

        self.steps = 0

        self.buffers = [self._p_prev, self._p_curr]

        self._prepare_data(alpha_map_np, density_map_np)

    # calculates the maximum stable time step (DT) to satisfy the CFL condition and prevent simulation instability

    def get_time_step(self, dimensions, dx, speed, safety_factor):
        courant_limit = 1.0 / math.sqrt(dimensions)
        return (dx / speed) * courant_limit * safety_factor

    @ti.func
    def _calculate_alpha_A(self,sound_speed, alpha, density):
        return alpha * (density * sound_speed ** 2 + 1.0 / density)

    @ti.func
    def _calculate_alpha_B(self,sound_speed, alpha):
        return (sound_speed** 2) * (alpha ** 2)

    @ti.func
    def _beta_from_alpha(self, alpha: ti.f32) -> ti.f32:
        Beta = 0.0
        if alpha >= 1.0:
            Beta = 1.0
        elif alpha <= 0.0:
            Beta = 0.0
        else:
            R = ti.sqrt(1.0 - alpha)
            Beta = (1.0 - R) / (1.0 + R)

        return Beta

    def _prepare_data(self, alpha_map_np: np.ndarray, density_map_np: np.ndarray):
        self._mat_alpha_field.from_numpy(alpha_map_np)
        self._mat_density_field.from_numpy(density_map_np)

        self._prepare_simulation_data()

    @ti.kernel
    def _prepare_simulation_data(self):
        for i, j in self.alpha_field:
            count = 0
            if i > 1: count += 1
            if i < self.N - 2: count += 1
            if j > 1: count += 1
            if j < self.N - 2: count += 1
            self._k_field[i, j] = count
            self._bk_field[i, j] = 0.0

            alpha_val = config.DEFAULT_ALPHA
            density_val = config.DEFAULT_DENSITY
            beta_val = 0.0

            if (i >= self.pml_thickness and i < self.N - self.pml_thickness and
                    j >= self.pml_thickness and j < self.N - self.pml_thickness):
                ii = i - self.pml_thickness
                jj = j - self.pml_thickness

                alpha_val = self._mat_alpha_field[ii, jj]
                density_val = self._mat_density_field[ii,jj]

                self._alpha_A[i, j] = self._calculate_alpha_A(self.sound_speed, alpha_val, density_val)
                self._alpha_B[i, j] = self._calculate_alpha_B(self.sound_speed, alpha_val)

                beta_val = self._beta_from_alpha(alpha_val)

            else:
                dist_x = 0.0
                if i < self.pml_thickness:
                    dist_x = ti.cast(self.pml_thickness - i, ti.f32) / self.pml_thickness
                elif i >= self.N - self.pml_thickness:
                    dist_x = ti.cast(i - (self.N - self.pml_thickness) + 1, ti.f32) / self.pml_thickness

                dist_y = 0.0
                if j < self.pml_thickness:
                    dist_y = ti.cast(self.pml_thickness - j, ti.f32) / self.pml_thickness
                elif j >= self.N - self.pml_thickness:
                    dist_y = ti.cast(j - (self.N - self.pml_thickness) + 1, ti.f32) / self.pml_thickness

                dist_final = ti.max(dist_x, dist_y)

                if dist_final > 0.0:
                    alpha_val = self.alpha_max * (dist_final ** 3)
                    self._alpha_A[i, j] = self._calculate_alpha_A(self.sound_speed, alpha_val, density_val)  # gestosc powietrza
                    self._alpha_B[i, j] = self._calculate_alpha_B(self.sound_speed, alpha_val)

                beta_val = 0

            self._k_field[i, j] = alpha_val
            k_val = self._k_field[i, j]
            self._bk_field[i, j] = (4.0 - ti.cast(k_val, ti.f32)) * self.courant * beta_val

            if i == 0 or i == self.N - 1 or j == 0 or j == self.N - 1:
                self._p_curr[i, j] = 0.0
                self._p_prev[i, j] = 0.0


    @ti.kernel
    def _step(self, p_prev: ti.template(), p_now: ti.template(), steps: int):
        for i, j in ti.ndrange((1, self.N - 1), (1, self.N - 1)):
            k_val = self._k_field[i, j]
            bk_val = self._bk_field[i, j]
            alpha_a = self._alpha_A[i, j]
            alpha_b = self._alpha_B[i, j]

            w1 = 1.0 + bk_val + alpha_a / 2 * self.dt + alpha_b * (self.dt ** 2)
            w2 = self.courant_sq * (p_now[i + 1, j] + p_now[i - 1, j] + p_now[i, j + 1] + p_now[i, j - 1])
            w3 = (2.0 - k_val * self.courant_sq) * p_now[i, j]
            w4 = (bk_val - 1.0 + (alpha_a / 2.0) * self.dt) * p_prev[i, j]

            p_prev[i, j] = (1.0 / w1) * (w2 + w3 + w4)

        if self.dt * steps < (self.dt * self.delay) + 4 * self.sigma:
            pulse = self.amplitude * ti.exp(- ((self.dt * steps - self.dt * self.delay) ** 2) / (2 * self.sigma ** 2))
            p_prev[self.src_x, self.src_y] += pulse  # soft source


    def update(self):
        p_past = self.buffers[self.steps % 2]
        p_present = self.buffers[(self.steps + 1) % 2]

        self._step(p_past, p_present, self.steps)

        self.steps += 1

    def get_current_pressure(self):
        return self.buffers[self.steps % 2].to_numpy()

    def get_alpha_map(self):
        return self._alpha_field.to_numpy()

    def print_config(self):
        print("-" * 30)
        print(f"FDTD SIMULATION CONFIGURATION {config.DIM}D")
        print("-" * 30)
        print(f"Grid Size(with PML):  {self.N} x {self.N} nodes")
        print(f"Spatial Step (DX):         {self.dx * 1000:.2f} mm")
        print(f"Time Step (DT):         {self.dt * 1e6:.2f} us")
        print(f"Courant Number: {self.courant:.4f} (Limit: {1 / math.sqrt(config.DIM):.4f})")
        print(f"Max Frequency:     {self.freq_max} Hz")
        print("-" * 30)

