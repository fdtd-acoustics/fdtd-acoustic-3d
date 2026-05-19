import config
import numpy as np
import taichi as ti
from simulation import GridParams, SimulationBuilder

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
        self.pressure_field_db = ti.field(dtype=ti.f32, shape=(self.Nx, self.Ny, self.Nz))

        # Slice 2D
        self.plane_v_1 = ti.Vector.field(3, dtype=ti.f32, shape=self.Nx * self.Nz) # Slice pressure horizontal
        self.plane_c_1 = ti.Vector.field(3, dtype=ti.f32, shape=self.Nx * self.Nz) # Slice colors horizontal
        self.plane_v_2 = ti.Vector.field(3, dtype=ti.f32, shape=self.Nx * self.Ny) # Slice pressure vertical
        self.plane_c_2 = ti.Vector.field(3, dtype=ti.f32, shape=self.Nx * self.Ny) # Slice colors vertical
        self.plane_v_3 = ti.Vector.field(3, dtype=ti.f32, shape=self.Ny * self.Nz) # Slice pressure vertical
        self.plane_c_3 = ti.Vector.field(3, dtype=ti.f32, shape=self.Ny * self.Nz) # Slice colors vertical

        # --- Voxels ---
        self.voxels_pos = None
        self.voxels_color = None
        self.num_voxels = 0

        # --- Mesh ---
        self.v_mesh = None
        self.f_mesh = None
        self.c_mesh = None
        self.has_mesh = False

    @ti.kernel
    def compute_spl(self):
        p0 = 2e-5

        for I in ti.grouped(self.pressure_field):
            p_abs = ti.abs(self.pressure_field[I])
            p_safe = ti.max(p_abs, 1e-10)

            #self.pressure_field_db[I] = 10.0 * (ti.log(p_safe / p0) / 2.30258509299) # według prezentacji
            self.pressure_field_db[I] = 20.0 * (ti.log(p_safe / p0) / 2.30258509299) # według normalnego wzoru

    def init_voxels(self, space_matrix: np.ndarray):
        base_mask = space_matrix > 0
        indices = np.argwhere(base_mask)

        self.num_voxels = len(indices)
        print(f"===>Prepared {self.num_voxels} voxels to show.")

        if self.num_voxels == 0:
            return

        points = indices

        colors = np.zeros((self.num_voxels, 3), dtype=np.float32)

        material_map = SimulationBuilder.get_material_map_from_csv(config.MAIN_MATERIAL_LIBRARY)

        if material_map:
            for mat_id, mat_info in material_map.items():
                if "color" in mat_info:
                    rgb_normalized = SimulationBuilder.hex_to_rgb_normalized(mat_info["color"])
                    colors[space_matrix[base_mask] == mat_id] = rgb_normalized

        self.voxels_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.num_voxels)
        self.voxels_color = ti.Vector.field(3, dtype=ti.f32, shape=self.num_voxels)

        self.voxels_pos.from_numpy(points.astype(np.float32))
        self.voxels_color.from_numpy(colors)

    def init_mesh(self, mesh_verts_np: np.ndarray, mesh_faces_np: np.ndarray, mesh_colors_np: np.ndarray, dx: float):
        if mesh_verts_np is None or len(mesh_verts_np) == 0:
            self.has_mesh = False
            return

        print(f"===>Prepared Mesh to show.")
        self.has_mesh = True

        mesh_verts_grid = mesh_verts_np / dx

        self.v_mesh = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh_verts_grid))
        self.f_mesh = ti.field(dtype=ti.i32, shape=len(mesh_faces_np.flatten()))
        self.c_mesh = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh_colors_np))

        self.v_mesh.from_numpy(mesh_verts_grid.astype(np.float32))
        self.f_mesh.from_numpy(mesh_faces_np.astype(np.int32).flatten())
        self.c_mesh.from_numpy(mesh_colors_np.astype(np.float32))

    @ti.kernel
    def update_planes(self, slice_x: int, slice_y: int, slice_z: int, pressure_field: ti.template(), use_log_scale: int, max_db: float):

        p_ref = 0.00002
        norm_val = 1.0
        min_db = 40.0
        #max_db = 160.0

        # X Slice
        for j, k in ti.ndrange(self.Ny, self.Nz):
            idx_3 = j * self.Nz + k
            self.plane_v_3[idx_3] = ti.Vector([float(slice_x), float(j), float(k)])
            pres1 = pressure_field[slice_x + self.pml_thick, j + self.pml_thick, k + self.pml_thick]

            p1_norm = 0.0
            if use_log_scale == 1:
                p1_norm = get_signed_db_norm(pres1, p_ref, min_db, max_db)
            else:
                p1_norm = ti.math.clamp(pres1 / norm_val, -1.0, 1.0)

            self.plane_c_3[idx_3] = get_color_jet(p1_norm)

        # Y Slice
        for i, k in ti.ndrange(self.Nx, self.Nz):
            idx_1 = i * self.Nz + k
            self.plane_v_1[idx_1] = ti.Vector([float(i), float(slice_y), float(k)])
            pres1 = pressure_field[i + self.pml_thick, slice_y + self.pml_thick, k + self.pml_thick]

            p1_norm = 0.0
            if use_log_scale == 1:
                p1_norm = get_signed_db_norm(pres1, p_ref, min_db, max_db)
            else:
                p1_norm = ti.math.clamp(pres1 / norm_val, -1.0, 1.0)

            self.plane_c_1[idx_1] = get_color_jet(p1_norm)

        # Z Slice
        for i, j in ti.ndrange(self.Nx, self.Ny):
            idx_2 = i * self.Ny + j
            self.plane_v_2[idx_2] = ti.Vector([float(i), float(j), float(slice_z)])

            pres2 = pressure_field[i + self.pml_thick, j + self.pml_thick, slice_z + self.pml_thick]
            p2_norm = 0.0
            if use_log_scale == 1:
                p2_norm = get_signed_db_norm(pres2, p_ref, min_db, max_db)
            else:
                p2_norm = ti.math.clamp(pres2 / norm_val, -1.0, 1.0)

            self.plane_c_2[idx_2] = get_color_jet(p2_norm)


@ti.func
def get_color_jet(val):
    v = (val + 1.0) / 2.0

    r = ti.math.clamp(1.5 - ti.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = ti.math.clamp(1.5 - ti.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = ti.math.clamp(1.5 - ti.abs(4.0 * v - 1.0), 0.0, 1.0)

    return ti.Vector([r, g, b])

@ti.func
def get_signed_db_norm(pressure, p_ref, min_db, max_db):
    p_abs = ti.abs(pressure)
    p_safe = ti.max(p_abs, p_ref * 1e-6)

    spl = 20.0 * (ti.log(p_safe / p_ref) / 2.30258509299)

    spl_clamped = ti.math.clamp(spl, min_db, max_db)
    spl_norm = (spl_clamped - min_db) / (max_db - min_db)

    sign = 1.0
    if pressure < 0.0:
        sign = -1.0

    return sign * spl_norm