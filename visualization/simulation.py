import math

import taichi as ti
from . import config

@ti.data_oriented
class Simulation:
    def __init__(self):
        self.t = ti.field(float, shape=())

        # 3D Matrix containing pressure
        self.pressure_field = ti.field(dtype=ti.f32, shape=(config.N, config.N, config.N))

        # Slice 2D
        self.plane_v_1 = ti.Vector.field(3, dtype=ti.f32, shape=config.N * config.N)
        self.plane_c_1 = ti.Vector.field(3, dtype=ti.f32, shape=config.N * config.N)
        self.plane_v_2 = ti.Vector.field(3, dtype=ti.f32, shape=config.N * config.N)
        self.plane_c_2 = ti.Vector.field(3, dtype=ti.f32, shape=config.N * config.N)



    # @ti.kernel
    # def update_wave(self):
    #     self.t[None] += config.DELTA_TIME
    #     current_t = self.t[None]
    #     source = ti.Vector([config.SOURCE_POS[0], config.SOURCE_POS[1], config.SOURCE_POS[2]])
    #
    #     for i, j, k in ti.ndrange(config.N, config.N, config.N):
    #         pos = ti.Vector([float(i), float(j), float(k)])
    #         dist = (pos - source).norm()
    #
    #         self.pressure_field[i, j, k] = (ti.sin(current_t - dist * 0.5) / (dist * 0.1 + 1.0)) * 2.0
    #


    @ti.kernel
    def update_planes(self, slice_y: ti.i32, slice_z: ti.i32, pressure_field: ti.template()):

        norm_val = 1.0
        inv_n = 1.0 / config.N

        for i, k in ti.ndrange(config.N, config.N):
            idx = i * config.N + k

            # Horizontal Slice
            self.plane_v_1[idx] = ti.Vector([i * inv_n, slice_y * inv_n, k * inv_n])
            pres1 = pressure_field[i, slice_y, k]

            p1_norm = ti.math.clamp(pres1 / norm_val, -1.0, 1.0)

            # Horizontal Slice color mapping
            color1 = ti.Vector([1.0, 1.0, 1.0])
            if ti.abs(pres1) > 0.001:
                if p1_norm > 0:
                    color1 = ti.Vector([1.0, 1.0 - p1_norm, 1.0 - p1_norm])
                else:
                    color1 = ti.Vector([1.0 + p1_norm, 1.0 + p1_norm, 1.0])
                # color = ti.Vector([
                #     1.0 - ti.max(0.0, -pres1),
                #     1.0 - ti.abs(pres1),
                #     1.0 - ti.max(0.0, pres1)
                # ])
            self.plane_c_1[idx] = color1

            # Vertical Slice
            self.plane_v_2[idx] = ti.Vector([i * inv_n, k * inv_n, slice_z * inv_n])
            pres2 = pressure_field[i, k, slice_z]

            p2_norm = ti.math.clamp(pres2 / norm_val, -1.0, 1.0)

            # Vertical Slice color mapping
            color2 = ti.Vector([1.0, 1.0, 1.0])
            if ti.abs(pres2) > 0.01:
                if p2_norm > 0:
                    color2 = ti.Vector([1.0, 1.0 - p2_norm, 1.0 - p2_norm])
                else:
                    color2 = ti.Vector([1.0 + p2_norm, 1.0 + p2_norm, 1.0])
                # color = ti.Vector([
                #     1.0 - ti.max(0.0, -pres2),
                #     1.0 - ti.abs(pres2),
                #     1.0 - ti.max(0.0, pres2)
                # ])
            self.plane_c_2[idx] = color2