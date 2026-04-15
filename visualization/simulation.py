import math

import numpy as np
import taichi as ti

from simulation import GridParams
from . import vis_config as config


@ti.data_oriented
class Simulation:
    def __init__(self, grid: GridParams, pml_thick: int):
        self.t = ti.field(float, shape=())

        self.Nx = grid.Nx
        self.Ny = grid.Ny
        self.Nz = grid.Nz
        self.pml_thick = pml_thick

        # 3D Matrix containing pressure
        self.pressure_field = ti.field(dtype=ti.f32, shape=(self.Nx, self.Ny, self.Nz))

        # Slice 2D
        self.plane_v_1 = ti.Vector.field(3, dtype=ti.f32, shape=self.Nx * self.Nz) # Slice pressure horizontal
        self.plane_c_1 = ti.Vector.field(3, dtype=ti.f32, shape=self.Nx * self.Nz) # Slice colors horizontal
        self.plane_v_2 = ti.Vector.field(3, dtype=ti.f32, shape=self.Nx * self.Ny) # Slice pressure vertical
        self.plane_c_2 = ti.Vector.field(3, dtype=ti.f32, shape=self.Nx * self.Ny) # Slice colors vertical

        self.voxels_pos = None
        self.voxels_color = None
        self.num_voxels = 0

    def init_voxels(self, space_matrix: np.ndarray):
        # data = np.load(file_path)
        # space_matrix = data['material_core']

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

        # Horizontal Slice
        for i, k in ti.ndrange(self.Nx, self.Nz):
            idx_1 = i * self.Nz + k

            self.plane_v_1[idx_1] = ti.Vector([i, slice_y, k])
            pres1 = pressure_field[i + self.pml_thick , slice_y + self.pml_thick , k + self.pml_thick ]

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
            self.plane_c_1[idx_1] = color1

        # Vertical Slice
        for i, j in ti.ndrange(self.Nx, self.Ny):
            idx_2 = i * self.Ny + j

            self.plane_v_2[idx_2] = ti.Vector([i, j, slice_z])
            pres2 = pressure_field[i + self.pml_thick, j + self.pml_thick, slice_z + self.pml_thick]

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

            self.plane_c_2[idx_2] = color2