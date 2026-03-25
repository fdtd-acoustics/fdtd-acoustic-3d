import taichi as ti

from simulation import GridParams
from . import vis_config as config

"""
@ti.data_oriented
class CubeGeometry:
    def __init__(self):
        #Tutaj tworzymy szescian w pamieci - raczej logiczne jak to dziala
        #Zmienione jest na 24 wierzcholki bo przy 8 te sciany sie brzydko naciagaly
        self.verts =  ti.Vector.field(3, dtype=ti.f32, shape=24)
        self.indices = ti.field(dtype=ti.i32, shape=36)

        self.init_mesh_kernel()

    @ti.kernel
    def init_mesh_kernel(self):
        ds = config.VOXEL_WIDTH

        # przód
        self.verts[0] = [0, 0, ds];
        self.verts[1] = [ds, 0, ds]
        self.verts[2] = [ds, ds, ds];
        self.verts[3] = [0, ds, ds]
        # tyl
        self.verts[4] = [ds, 0, 0];
        self.verts[5] = [0, 0, 0]
        self.verts[6] = [0, ds, 0];
        self.verts[7] = [ds, ds, 0]
        # gora
        self.verts[8] = [0, ds, ds];
        self.verts[9] = [ds, ds, ds]
        self.verts[10] = [ds, ds, 0];
        self.verts[11] = [0, ds, 0]
        # dol
        self.verts[12] = [0, 0, 0];
        self.verts[13] = [ds, 0, 0]
        self.verts[14] = [ds, 0, ds];
        self.verts[15] = [0, 0, ds]
        # prawo
        self.verts[16] = [ds, 0, ds];
        self.verts[17] = [ds, 0, 0]
        self.verts[18] = [ds, ds, 0];
        self.verts[19] = [ds, ds, ds]
        # lewo
        self.verts[20] = [0, 0, 0];
        self.verts[21] = [0, 0, ds]
        self.verts[22] = [0, ds, ds];
        self.verts[23] = [0, ds, 0]

        #wypelnianie scian
        for i in range(6):
            offset = i * 4
            self.indices[i * 6 + 0] = offset + 0
            self.indices[i * 6 + 1] = offset + 1
            self.indices[i * 6 + 2] = offset + 2
            self.indices[i * 6 + 3] = offset + 0
            self.indices[i * 6 + 4] = offset + 2
            self.indices[i * 6 + 5] = offset + 3
"""

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