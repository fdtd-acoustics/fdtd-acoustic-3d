import math

import taichi as ti

import config
from .source_manager import SourceManager

#todo:konwencja w pytonie jest taka ze internal methody zacznymay od _ a publiczne normalnie

@ti.data_oriented
class FDTD_Simulation:
    def __init__(self, sound_speed: float, dx: float, dt: float, pml_thick: int,
                 alpha_max: float, safety_factor: float,
                 sources: SourceManager, material_core: ti.template()):
        self.sources = sources
        self.sound_speed = sound_speed
        self.dx= dx
        self.dt = dt
        self.pml_thick = pml_thick
        self.alpha_max = alpha_max

        self.courant = (1.0 / math.sqrt(config.DIM)) * safety_factor
        self.courant_sq = self.courant ** 2

        self.Nx = material_core.shape[0] + self.pml_thick*2
        self.Ny = material_core.shape[1] + self.pml_thick*2
        self.Nz = material_core.shape[2] + self.pml_thick*2

        self.steps = 0

        self._alpha_A = ti.field(dtype=ti.f32, shape=(self.Nx, self.Ny, self.Nz))
        self._alpha_B = ti.field(dtype=ti.f32, shape=(self.Nx, self.Ny, self.Nz))

        self._k_field = ti.field(dtype=ti.i32, shape=(self.Nx, self.Ny, self.Nz))
        self._bk_field = ti.field(dtype=ti.f32, shape=(self.Nx, self.Ny, self.Nz))

        self._p_prev = ti.field(ti.f32, shape=(self.Nx, self.Ny, self.Nz))  # macierz ciśnień
        self._p_curr = ti.field(ti.f32, shape=(self.Nx, self.Ny, self.Nz))  # macierz ciśnień

        self.buffers = [self._p_prev, self._p_curr]

        self._prepare_simulation_data(material_core)

    @ti.func
    def _calculate_alpha_A(self, sound_speed: ti.f32, alpha: ti.f32, density: ti.f32) -> ti.f32:
        return alpha * (density * sound_speed ** 2 + 1.0 / density)

    @ti.func
    def _calculate_alpha_B(self, sound_speed: ti.f32, alpha: ti.f32) -> ti.f32:
        return (sound_speed ** 2) * (alpha ** 2)

    @ti.func
    def _beta_from_alpha(self, alpha: ti.f32) -> ti.f32:
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
    def _prepare_simulation_data(self, material_core: ti.template()):
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # ile sasiadow posiada punkt
            count = 0
            if i > 1: count += 1
            if i < self.Nx - 2: count += 1
            if j > 1: count += 1
            if j < self.Ny - 2: count += 1
            if k > 1: count += 1
            if k < self.Nz - 2: count += 1
            self._k_field[i, j, k] = count

            alpha_val = config.DEFAULT_ALPHA
            density_val = config.DEFAULT_DENSITY

            if (i >= self.pml_thick and i < self.Nx - self.pml_thick and
                    j >= self.pml_thick and j < self.Ny - self.pml_thick and
                    k >= self.pml_thick and k < self.Nz - self.pml_thick):

                ii = i - self.pml_thick
                jj = j - self.pml_thick
                kk = k - self.pml_thick

                mat_data = material_core[ii, jj, kk]
                alpha_val = mat_data[0]
                density_val = mat_data[1]

                self._alpha_A[i, j, k] = self._calculate_alpha_A(self.sound_speed, alpha_val, density_val)
                self._alpha_B[i, j, k] = self._calculate_alpha_B(self.sound_speed, alpha_val)

            else:
                #PML
                dist_x = 0.0
                if i < self.pml_thick:
                    dist_x = ti.cast(self.pml_thick - i, ti.f32) / self.pml_thick
                elif i >= self.Nx - self.pml_thick:
                    dist_x = ti.cast(i - (self.Nx - self.pml_thick) + 1, ti.f32) / self.pml_thick

                dist_y = 0.0
                if j < self.pml_thick:
                    dist_y = ti.cast(self.pml_thick - j, ti.f32) / self.pml_thick
                elif j >= self.Ny - self.pml_thick:
                    dist_y = ti.cast(j - (self.Ny - self.pml_thick) + 1, ti.f32) / self.pml_thick

                dist_z = 0.0
                if k < self.pml_thick:
                    dist_z = ti.cast(self.pml_thick - k, ti.f32) / self.pml_thick
                elif k >= self.Nz - self.pml_thick:
                    dist_z = ti.cast(k - (self.Nz - self.pml_thick) + 1, ti.f32) / self.pml_thick

                dist_final = ti.max(dist_x, dist_y, dist_z)
                if dist_final > 0.0:
                    alpha_val = self.alpha_max * (dist_final ** 3)
                    self._alpha_A[i, j, k] = self._calculate_alpha_A(self.sound_speed, alpha_val, density_val)  # gestosc powietrza
                    self._alpha_B[i, j, k] = self._calculate_alpha_B(self.sound_speed, alpha_val)

            k_val = self._k_field[i, j, k]
            beta_val = self._beta_from_alpha(alpha_val)

            self._bk_field[i, j, k] = (6.0 - ti.cast(k_val, ti.f32)) * self.courant * beta_val

            # brzegi maja zawsze cisnienie 0
            if i == 0 or i == self.Nx - 1 or j == 0 or j == self.Ny - 1 or k == 0 or k == self.Nz - 1:
                self._p_curr[i, j, k] = 0.0
                self._p_prev[i, j, k] = 0.0





    @ti.kernel
    def _step(self, p_prev: ti.template(), p_curr: ti.template(), steps: ti.i32):
        for i, j, k in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1),(1, self.Nz - 1)):
            k_val = self._k_field[i, j, k]
            bk_val = self._bk_field[i, j, k]
            alpha_a = self._alpha_A[i, j, k]
            alpha_b = self._alpha_B[i, j, k]

            w1 = 1.0 + bk_val + alpha_a / 2 * self.dt + alpha_b * (self.dt ** 2)
            w2 = self.courant_sq * (p_curr[i + 1, j, k] + p_curr[i - 1, j, k] + p_curr[i, j + 1, k] + p_curr[i, j - 1, k] + p_curr[i, j, k + 1] + p_curr[i, j, k - 1])
            w3 = (2.0 - k_val * self.courant_sq) * p_curr[i, j, k]
            w4 = (bk_val - 1.0 + (alpha_a / 2.0) * self.dt) * p_prev[i, j, k]

            p_prev[i, j, k] = (1.0 / w1) * (w2 + w3 + w4)


    def update(self):
        p_past = self.buffers[self.steps % 2]
        p_present = self.buffers[(self.steps + 1) % 2]

        self._step(p_past, p_present, self.steps)
        self.sources.update_sources(p_past, self.steps, self.dt) # ZRODLO
        # self.receivers.update_receivers(p_past, self.steps) # MIKROFONY
        self.steps += 1


    def get_current_pressure(self):
        return self.buffers[self.steps % 2]