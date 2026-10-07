import math

import taichi as ti

import config
from .receiver_manager import ReceiverManager
from .source_manager import SourceManager

@ti.data_oriented
class FDTD_Simulation:
    def __init__(self, sound_speed: float, dx: float, dt: float, pml_thick: int,
                 alpha_max: float, safety_factor: float,
                 source_manager: SourceManager, receiver_manager: ReceiverManager, material_core: ti.template(), dim: int = 3):
        self.source_manager = source_manager
        self.receiver_manager = receiver_manager
        self.sound_speed = sound_speed
        self.dx= dx
        self.dt = dt
        self.pml_thick = pml_thick
        self.alpha_max = alpha_max
        self.dim = dim

        self.courant = (1.0 / math.sqrt(self.dim)) * safety_factor
        self.courant_sq = self.courant ** 2

        if self.dim == 2:
            self.Nx = material_core.shape[0] + self.pml_thick * 2
            self.Ny = material_core.shape[1] + self.pml_thick * 2
            self.Nz = None
            self.field_shape = (self.Nx, self.Ny)
            self._step_kernel = self._step_2d
            self._is_air = self._is_air_2d
        elif self.dim == 3:
            self.Nx = material_core.shape[0] + self.pml_thick * 2
            self.Ny = material_core.shape[1] + self.pml_thick * 2
            self.Nz = material_core.shape[2] + self.pml_thick * 2
            self.field_shape = (self.Nx, self.Ny, self.Nz)
            self._step_kernel = self._step_3d
            self._is_air = self._is_air_3d
        else:
            raise ValueError("Only dimensions dim=2 or dim=3 are supported")

        self._steps = 0
        self._current_time = 0

        self._alpha_A = ti.field(dtype=ti.f32, shape=self.field_shape)
        self._alpha_B = ti.field(dtype=ti.f32, shape=self.field_shape)

        self._k_field = ti.field(dtype=ti.i32, shape=self.field_shape)
        self._bk_field = ti.field(dtype=ti.f32, shape=self.field_shape)

        self._p_prev = ti.field(ti.f32, shape=self.field_shape)  # macierz ciśnień
        self._p_curr = ti.field(ti.f32, shape=self.field_shape)  # macierz ciśnień

        self.buffers = [self._p_prev, self._p_curr]

        if dim == 2:
            self._prepare_simulation_data_2d(material_core)
        else:
            self._prepare_simulation_data_3d(material_core)

    @ti.func
    def _calculate_alpha_A(self, sound_speed: ti.f32, alpha: ti.f32, density: ti.f32) -> ti.f32:
        return alpha * (density * sound_speed ** 2 + 1.0 / density)

    @ti.func
    def _calculate_alpha_B(self, sound_speed: ti.f32, alpha: ti.f32) -> ti.f32:
        return (sound_speed ** 2) * (alpha ** 2)

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

    @ti.func
    def _is_air_2d(self, i, j, material_core: ti.template()) -> ti.i32:
        air_flag = 1

        if (self.pml_thick <= i < self.Nx - self.pml_thick and
                self.pml_thick <= j < self.Ny - self.pml_thick):
            ii = i - self.pml_thick
            jj = j - self.pml_thick

            alpha_val = material_core[ii, jj][0]

            if alpha_val != 0.0:
                air_flag = 0
            else:
                air_flag = 1

        return air_flag

    @ti.func
    def _is_air_3d(self, i, j, k, material_core: ti.template()) -> ti.i32:
        air_flag = 1

        if (self.pml_thick <= i < self.Nx - self.pml_thick and
                self.pml_thick <= j < self.Ny - self.pml_thick and
                self.pml_thick <= k < self.Nz - self.pml_thick):
            ii = i - self.pml_thick
            jj = j - self.pml_thick
            kk = k - self.pml_thick

            alpha_val = material_core[ii, jj, kk][0]

            if alpha_val != 0.0:
                air_flag = 0
            else:
                air_flag = 1

        return air_flag

    @ti.kernel
    def _prepare_simulation_data_2d(self, material_core: ti.template()):
        for i, j in ti.ndrange(self.Nx, self.Ny):
            # ile sasiadow posiada punkt
            count = 0

            if self._is_air(i, j, material_core):
                if i + 1 < self.Nx: count += self._is_air(i + 1, j, material_core)
                if i - 1 >= 0:      count += self._is_air(i - 1, j, material_core)

                if j + 1 < self.Ny: count += self._is_air(i, j + 1, material_core)
                if j - 1 >= 0:      count += self._is_air(i, j - 1, material_core)

            alpha_val = config.DEFAULT_ALPHA
            density_val = config.DEFAULT_DENSITY
            beta_val = 0.0

            if (self.pml_thick <= i < self.Nx - self.pml_thick and
                    self.pml_thick <= j < self.Ny - self.pml_thick):

                ii = i - self.pml_thick
                jj = j - self.pml_thick

                mat_data = material_core[ii, jj]
                alpha_val = mat_data[0]
                density_val = mat_data[1]

                self._alpha_A[i, j] = self._calculate_alpha_A(self.sound_speed, alpha_val, density_val)
                self._alpha_B[i, j] = self._calculate_alpha_B(self.sound_speed, alpha_val)

                surface_alpha = alpha_val
                if alpha_val == 0.0:
                    nx_m = material_core.shape[0]
                    ny_m = material_core.shape[1]
                    nz_m = material_core.shape[2]
                    best = 0.0
                    if ii + 1 < nx_m:
                        a = material_core[ii + 1, jj][0]
                        if a > best: best = a
                    if ii - 1 >= 0:
                        a = material_core[ii - 1, jj][0]
                        if a > best: best = a
                    if jj + 1 < ny_m:
                        a = material_core[ii, jj + 1][0]
                        if a > best: best = a
                    if jj - 1 >= 0:
                        a = material_core[ii, jj - 1][0]
                        if a > best: best = a
                    surface_alpha = best

                beta_val = self._beta_from_alpha(surface_alpha)

            else:
                # PML
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

                dist_final = ti.max(dist_x, dist_y)
                if dist_final > 0.0:
                    alpha_val = self.alpha_max * (dist_final ** 2)
                    self._alpha_A[i, j] = self._calculate_alpha_A(self.sound_speed, alpha_val,
                                                                     density_val)  # gestosc powietrza
                    self._alpha_B[i, j] = self._calculate_alpha_B(self.sound_speed, alpha_val)

                beta_val = 0

            self._k_field[i, j] = count

            self._bk_field[i, j] = (4 - count) * self.courant * beta_val * 0.5

            # brzegi maja zawsze cisnienie 0
            if i == 0 or i == self.Nx - 1 or j == 0 or j == self.Ny - 1:
                self._p_curr[i, j] = 0.0
                self._p_prev[i, j] = 0.0

    @ti.kernel
    def _prepare_simulation_data_3d(self, material_core: ti.template()):
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # ile sasiadow posiada punkt
            count = 0

            if self._is_air(i, j, k, material_core):
                if i + 1 < self.Nx: count += self._is_air(i + 1, j, k, material_core)
                if i - 1 >= 0:      count += self._is_air(i - 1, j, k, material_core)

                if j + 1 < self.Ny: count += self._is_air(i, j + 1, k, material_core)
                if j - 1 >= 0:      count += self._is_air(i, j - 1, k, material_core)

                if k + 1 < self.Nz: count += self._is_air(i, j, k + 1, material_core)
                if k - 1 >= 0:      count += self._is_air(i, j, k - 1, material_core)


            alpha_val = config.DEFAULT_ALPHA
            density_val = config.DEFAULT_DENSITY
            beta_val = 0.0

            if (self.pml_thick <= i < self.Nx - self.pml_thick and
                    self.pml_thick <= j < self.Ny - self.pml_thick and
                    self.pml_thick <= k < self.Nz - self.pml_thick):

                ii = i - self.pml_thick
                jj = j - self.pml_thick
                kk = k - self.pml_thick

                mat_data = material_core[ii, jj, kk]
                alpha_val = mat_data[0]
                density_val = mat_data[1]

                self._alpha_A[i, j, k] = self._calculate_alpha_A(self.sound_speed, alpha_val, density_val)
                self._alpha_B[i, j, k] = self._calculate_alpha_B(self.sound_speed, alpha_val)

                surface_alpha = alpha_val
                if alpha_val == 0.0:
                    nx_m = material_core.shape[0]
                    ny_m = material_core.shape[1]
                    nz_m = material_core.shape[2]
                    best = 0.0
                    if ii + 1 < nx_m:
                        a = material_core[ii + 1, jj, kk][0]
                        if a > best: best = a
                    if ii - 1 >= 0:
                        a = material_core[ii - 1, jj, kk][0]
                        if a > best: best = a
                    if jj + 1 < ny_m:
                        a = material_core[ii, jj + 1, kk][0]
                        if a > best: best = a
                    if jj - 1 >= 0:
                        a = material_core[ii, jj - 1, kk][0]
                        if a > best: best = a
                    if kk + 1 < nz_m:
                        a = material_core[ii, jj, kk + 1][0]
                        if a > best: best = a
                    if kk - 1 >= 0:
                        a = material_core[ii, jj, kk - 1][0]
                        if a > best: best = a
                    surface_alpha = best

                beta_val = self._beta_from_alpha(surface_alpha)

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
                    alpha_val = self.alpha_max * (dist_final**2)
                    self._alpha_A[i, j, k] = self._calculate_alpha_A(self.sound_speed, alpha_val, density_val)  # gestosc powietrza
                    self._alpha_B[i, j, k] = self._calculate_alpha_B(self.sound_speed, alpha_val)

                beta_val = 0

            self._k_field[i,j,k] = count

            self._bk_field[i, j, k] = (6 - count) * self.courant * beta_val * 0.5

            # brzegi maja zawsze cisnienie 0
            if i == 0 or i == self.Nx - 1 or j == 0 or j == self.Ny - 1 or k == 0 or k == self.Nz - 1:
                self._p_curr[i, j, k] = 0.0
                self._p_prev[i, j, k] = 0.0

    @ti.kernel
    def _step_2d(self, p_prev: ti.template(), p_curr: ti.template(), steps: ti.i32):
        for i, j, k in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1)):
            k_val = self._k_field[i, j]
            if k_val > 0:
                bk_val = self._bk_field[i, j]
                alpha_a = self._alpha_A[i, j]
                alpha_b = self._alpha_B[i, j]

                w1 = 1.0 + bk_val + alpha_a / 2 * self.dt + alpha_b * (self.dt ** 2)
                w2 = self.courant_sq * (p_curr[i + 1, j] + p_curr[i - 1, j] + p_curr[i, j + 1] + p_curr[i, j - 1])
                w3 = (2.0 - k_val * self.courant_sq) * p_curr[i, j]
                w4 = (bk_val - 1.0 + (alpha_a / 2.0) * self.dt) * p_prev[i, j]

                p_prev[i, j] = (1.0 / w1) * (w2 + w3 + w4)
            else:
                p_prev[i, j] = 0.0

    @ti.kernel
    def _step_3d(self, p_prev: ti.template(), p_curr: ti.template(), steps: ti.i32):
        for i, j, k in ti.ndrange((1, self.Nx - 1), (1, self.Ny - 1),(1, self.Nz - 1)):
            k_val = self._k_field[i, j, k]
            if k_val > 0:
                bk_val = self._bk_field[i, j, k]
                alpha_a = self._alpha_A[i, j, k]
                alpha_b = self._alpha_B[i, j, k]

                w1 = 1.0 + bk_val + alpha_a / 2 * self.dt + alpha_b * (self.dt ** 2)
                w2 = self.courant_sq * (p_curr[i + 1, j, k] + p_curr[i - 1, j, k] + p_curr[i, j + 1, k] + p_curr[i, j - 1, k] + p_curr[i, j, k + 1] + p_curr[i, j, k - 1])
                w3 = (2.0 - k_val * self.courant_sq) * p_curr[i, j, k]
                w4 = (bk_val - 1.0 + (alpha_a / 2.0) * self.dt) * p_prev[i, j, k]

                p_prev[i, j, k] = (1.0 / w1) * (w2 + w3 + w4)
            else:
                p_prev[i, j, k] = 0.0

    def update(self):
        p_past = self.buffers[self._steps % 2]
        p_present = self.buffers[(self._steps + 1) % 2]

        self._step_kernel(p_past, p_present, self._steps)
        self.source_manager.update_sources(p_present, self._steps) # ZRODLO
        self.receiver_manager.update_receivers(p_past, self._steps) # MIKROFONY

        self._steps += 1
        self._current_time += self.dt

    def get_current_pressure(self):
        return self.buffers[self._steps % 2]

    def get_time(self):
        return self._current_time

    def get_steps(self):
        return self._steps