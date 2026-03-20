import math
import numpy as np
import taichi as ti
from fdtd import SourceManager, ReceiverManager, FDTD_Simulation
from visualization import *
from visualization.config import  *
import trimesh

from voxelization.voxelizer import Voxelizer

MATERIAL_MAP = {
    0: [0.0, 1.225],
    1: [0.5, 1000.0],
    2: [0.1, 500.0]
}

def print_information(dt, Nx, Ny, Nz, dx):
    print("Simulation parameters:")
    print(f"dt: {dt}")
    print(f"Nx: {Nx}")
    print(f"Ny: {Ny}")
    print(f"Nz: {Nz}")
    print(f"dx: {dx}")


@ti.kernel
def fill_material_kernel(
        material_field: ti.template(),
        data_np: ti.types.ndarray(),
        map_alpha: ti.types.ndarray(),
        map_density: ti.types.ndarray()
):
    for i, j, k in material_field:
        mat_id = data_np[i, j, k]
        material_field[i, j, k][0] = map_alpha[mat_id]
        material_field[i, j, k][1] = map_density[mat_id]


class SimulationConfig:
    def __init__(self,
                 obj_filepath: str,
                 pml_thick: int,
                 alpha_max: float,
                 source_manager: SourceManager,
                ):
        self.source_manager = source_manager
        # self.receiver_manager = receiver_manager
        self.obj_filepath = obj_filepath
        self.pml_thick = pml_thick
        self.alpha_max = alpha_max
        self.nx = 0
        self.ny = 0
        self.nz = 0

        self.sound_speed = C
        self.nodes_per_wavelength = NODES_PER_WAVELENGTH
        self.dim = DIM
        self.safety_factor = SAFETY_FACTOR

        npz_filepath = SCENES_OUT_DIR / obj_filepath.split("/")[-1].replace('.obj', '.npz')

        with np.load(npz_filepath) as data:
            self.matrix_3d = data[data.files[0]].astype(np.int32)


    def get_time_step(self, dimensions, dx, speed, safety_factor):
        courant_limit = 1.0 / math.sqrt(dimensions)
        return (dx / speed) * courant_limit * safety_factor

    def prepare_material_core(self):
        self.nx, self.ny, self.nz = self.matrix_3d.shape

        material_core = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx, self.ny, self.nz))

        ids = sorted(MATERIAL_MAP.keys())
        alphas = np.array([MATERIAL_MAP[i][0] for i in ids], dtype=np.float32)
        densities = np.array([MATERIAL_MAP[i][1] for i in ids], dtype=np.float32)

        fill_material_kernel(material_core, self.matrix_3d, alphas, densities)

        return material_core



    def get_obj_dimensions(self):
        try:
            mesh = trimesh.load(self.obj_filepath, force='mesh')

            bounds = mesh.bounds

            dimensions = bounds[1] - bounds[0]

            return dimensions

        except Exception as e:
            return np.array([0.0, 0.0, 0.0])



    def builder(self):
        # trzeba dostac metry

        max_freq = self.source_manager.get_max_freq()
        wavelength = self.sound_speed / max_freq
        dx = wavelength / self.nodes_per_wavelength

        x_meters, y_meters, z_meters = self.get_obj_dimensions()

        Nx = int(x_meters/ dx)
        Ny = int(y_meters / dx)
        Nz = int(z_meters / dx)

        voxelizer = Voxelizer(Nx, Ny, Nz, dx, self.obj_filepath)
        voxelizer.load_scene()

        material_core = self.prepare_material_core()

        dt = self.get_time_step(self.dim, dx, self.sound_speed, self.safety_factor)

        print_information(dt, Nx, Ny, Nz, dx)

        fdtd_sim = FDTD_Simulation(self.sound_speed, dx, dt, self.source_manager, material_core)

        sim = Simulation()
        renderer = SceneRenderer()
        # geo = CubeGeometry()
        plane_geo = PlaneGeometry()

        slice_y = Ny // 2
        slice_z = Nz // 2


        while renderer.is_running:
            gui = renderer.window.get_gui()
            with gui.sub_window("2D Slices", 0.05, 0.05, 0.5, 0.2):
                slice_y = gui.slider_int("Horizontal Slice", slice_y, 0, Ny - 1)
                slice_z = gui.slider_int("Vertical Slice", slice_z, 0, Nz - 1)


            fdtd_sim.update()
            current_pressure = fdtd_sim.get_current_pressure()

            sim.update_planes(slice_y, slice_z, current_pressure)
            renderer.render_frame(simulation=sim, plane_geo=plane_geo)

        return fdtd_sim, dx, dt


