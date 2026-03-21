import math

import numpy as np
import taichi as ti
from . import vis_config as config


#todo:ta klasa korzysta jeszcze z N to trzeba przerobic na nx,ny,nz
# plus mozna to jakos inaczej nazwac bo to simulation nie jest
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

        self.voxels_pos = None
        self.voxels_color = None
        self.num_voxels = 0

    def init_voxels(self, file_path):
        data = np.load(file_path)
        space_matrix = data['material_core']

        base_mask = space_matrix > 0
        indices = np.argwhere(base_mask)

        self.num_voxels = len(indices)
        print(f"===>Prepared {self.num_voxels} voxels to show.")

        if self.num_voxels == 0:
            return

        #inv_n = 1.0 / config.N
        #points = indices * inv_n

        points = indices

        colors = np.zeros((self.num_voxels, 3), dtype=np.float32)
        colors[space_matrix[base_mask] == 1] = [0.5, 0.5, 0.5]  # Gray (wall)
        colors[space_matrix[base_mask] == 2] = [0.9, 0.7, 0.1]  # Yellow (other material)
        colors[colors.sum(axis=1) == 0] = [0.7, 0.7, 0.7]

        self.voxels_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.num_voxels)
        self.voxels_color = ti.Vector.field(3, dtype=ti.f32, shape=self.num_voxels)

        self.voxels_pos.from_numpy(points.astype(np.float32))
        self.voxels_color.from_numpy(colors)


    @ti.kernel
    def update_planes(self, slice_y: ti.i32, slice_z: ti.i32, pressure_field: ti.template()):

        norm_val = 1.0
        # inv_n = 1.0 / config.N

        for i, k in ti.ndrange(config.N, config.N):
            idx = i * config.N + k

            # Horizontal Slice
            #self.plane_v_1[idx] = ti.Vector([i * inv_n, slice_y * inv_n, k * inv_n])
            self.plane_v_1[idx] = ti.Vector([i, slice_y, k])
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
            self.plane_v_2[idx] = ti.Vector([i, k, slice_z])
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