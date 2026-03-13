import math

import taichi as ti
from visualization import config
from .source_manager import SourceManager
from .receiver_manager import ReceiverManager


@ti.data_oriented
class FDTD_Simulation:
    def __init__(self, c: float, dx: float, dt:float, sources: SourceManager,
                 receivers: ReceiverManager, material_core: ti.template()):
        self.sources = sources
        self.receivers = receivers
        self.C = c
        self.dx= dx
        self.dt = dt

        self.courant = (1.0 / math.sqrt(config.DIM)) * config.SAFETY_FACTOR
        self.courant_sq = self.courant ** 2

        self.Nx = material_core.shape[0] + config.PML_THICK*2
        self.Ny = material_core.shape[1] + config.PML_THICK*2
        self.Nz = material_core.shape[2] + config.PML_THICK*2

        self.steps = 0

        self.alpha_A = ti.field(dtype=ti.f32, shape=(self.Nx, self.Ny, self.Nz))
        self.alpha_B = ti.field(dtype=ti.f32, shape=(self.Nx, self.Ny, self.Nz))

        self.k_field = ti.field(dtype=ti.i32, shape=(self.Nx, self.Ny, self.Nz))
        self.bk_field = ti.field(dtype=ti.f32, shape=(self.Nx, self.Ny, self.Nz))

        self.p_prev = ti.field(ti.f32, shape=(self.Nx, self.Ny, self.Nz))  # macierz ciśnień, do niej wsadzane są najnowsze wartosci
        self.p_curr = ti.field(ti.f32, shape=(self.Nx, self.Ny, self.Nz))  # macierz ciśnień

        self.buffers = [self.p_prev, self.p_curr]

        self.prepare_simulation_data(material_core)

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
    def prepare_simulation_data(self, material_core: ti.template()):
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # ile sasiadow posiada punkt
            count = 0
            if i > 1: count += 1
            if i < self.Nx - 2: count += 1
            if j > 1: count += 1
            if j < self.Ny - 2: count += 1
            if k > 1: count += 1
            if k < self.Nz - 2: count += 1
            self.k_field[i, j, k] = count

            alpha_val = config.DEFAULT_ALPHA
            density_val = config.DEFAULT_DENSITY

            if (i >= config.PML_THICK and i < self.Nx - config.PML_THICK and
                    j >= config.PML_THICK and j < self.Ny - config.PML_THICK and
                    k >= config.PML_THICK and k < self.Nz - config.PML_THICK):

                ii = i - config.PML_THICK
                jj = j - config.PML_THICK
                kk = k - config.PML_THICK

                mat_data = material_core[ii, jj, kk]
                alpha_val = mat_data[0]
                density_val = mat_data[1]

                self.alpha_A[i, j, k] = self.calculate_alpha_A(self.C, alpha_val, density_val)
                self.alpha_B[i, j, k] = self.calculate_alpha_B(self.C, alpha_val)

            else:
                #PML
                dist_x = 0.0
                if i < config.PML_THICK:
                    dist_x = ti.cast(config.PML_THICK - i, ti.f32) / config.PML_THICK
                elif i >= self.Nx - config.PML_THICK:
                    dist_x = ti.cast(i - (self.Nx - config.PML_THICK) + 1, ti.f32) / config.PML_THICK

                dist_y = 0.0
                if j < config.PML_THICK:
                    dist_y = ti.cast(config.PML_THICK - j, ti.f32) / config.PML_THICK
                elif j >= self.Ny - config.PML_THICK:
                    dist_y = ti.cast(j - (self.Ny - config.PML_THICK) + 1, ti.f32) / config.PML_THICK

                dist_z = 0.0
                if k < config.PML_THICK:
                    dist_z = ti.cast(config.PML_THICK - k, ti.f32) / config.PML_THICK
                elif k >= self.Nz - config.PML_THICK:
                    dist_z = ti.cast(k - (self.Nz - config.PML_THICK) + 1, ti.f32) / config.PML_THICK

                dist_final = ti.max(dist_x, dist_y, dist_z)
                if dist_final > 0.0:
                    alpha_val = config.alpha_max * (dist_final ** 3)
                    self.alpha_A[i, j, k] = self.calculate_alpha_A(self.C, alpha_val, density_val)  # gestosc powietrza
                    self.alpha_B[i, j, k] = self.calculate_alpha_B(self.C, alpha_val)

            k_val = self.k_field[i, j, k]
            # beta_val = self.beta_from_alpha(alpha_val) # wczesniej mialem tak, ale raczej zle
            beta_val = 0

            self.bk_field[i, j, k] = (6.0 - ti.cast(k_val, ti.f32)) * self.courant * beta_val

            # brzegi maja zawsze cisnienie 0
            if i == 0 or i == self.Nx - 1 or j == 0 or j == self.Ny - 1 or k == 0 or k == self.Nz - 1:
                self.p_curr[i, j, k] = 0.0
                self.p_prev[i, j, k] = 0.0





    @ti.kernel
    def step(self, p_prev: ti.template(), p_curr: ti.template(), steps: ti.i32):
        for i, j, k in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1),(1, self.Nz - 1)):
            k_val = self.k_field[i, j, k]
            bk_val = self.bk_field[i, j, k]
            alpha_a = self.alpha_A[i, j, k]
            alpha_b = self.alpha_B[i, j, k]

            w1 = 1.0 + bk_val + alpha_a / 2 * self.dt + alpha_b * (self.dt ** 2)
            w2 = self.courant_sq * (p_curr[i + 1, j, k] + p_curr[i - 1, j, k] + p_curr[i, j + 1, k] + p_curr[i, j - 1, k] + p_curr[i, j, k + 1] + p_curr[i, j, k - 1])
            w3 = (2.0 - k_val * self.courant_sq) * p_curr[i, j, k]
            w4 = (bk_val - 1.0 - (alpha_a / 2.0) * self.dt) * p_prev[i, j, k]

            p_prev[i, j, k] = (1.0 / w1) * (w2 + w3 + w4)


    def update(self):
        p_past = self.buffers[self.steps % 2]
        p_present = self.buffers[(self.steps + 1) % 2]

        self.step(p_past, p_present, self.steps)
        self.sources.update_sources(p_past, self.steps, self.dt) # ZRODLO
        self.receivers.update_receivers(p_past, self.steps) # MIKROFONY
        self.steps += 1


    def get_current_pressure(self):
        return self.buffers[self.steps % 2]