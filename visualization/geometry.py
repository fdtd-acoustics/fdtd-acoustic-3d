import taichi as ti

from simulation import GridParams
from . import vis_config as config

# This class gets dimensions for specific plane, not for all the planes
@ti.data_oriented
class PlaneGeometry:
    #TODO Dodać uwzględnienie PML THICK
    def __init__(self, dim_x, dim_y):
        self.dim_x = dim_x  # local X dimension of a specific plane
        self.dim_y = dim_y  # local Y dimension of a specific plane

        self.num_indices = (self.dim_x - 1) * (self.dim_y - 1) * 6
        self.indices = ti.field(dtype=ti.i32, shape=self.num_indices)

        self.init_mesh()

    @ti.kernel
    def init_mesh(self):
        for i, j in ti.ndrange(self.dim_x - 1, self.dim_y - 1):
            idx = (i * (self.dim_y - 1) + j) * 6

            v0 = i * self.dim_y + j
            v1 = (i + 1) * self.dim_y + j
            v2 = i * self.dim_y + (j + 1)
            v3 = (i + 1) * self.dim_y + (j + 1)

            self.indices[idx + 0] = v0
            self.indices[idx + 1] = v1
            self.indices[idx + 2] = v2

            self.indices[idx + 3] = v2
            self.indices[idx + 4] = v1
            self.indices[idx + 5] = v3