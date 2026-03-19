import math
import yaml
import numpy as np
import taichi as ti
from fdtd import SourceManager, ReceiverManager, FDTD_Simulation
from visualization import *
from visualization.config import  SCENES_OUT_DIR
import trimesh

from voxelization.voxelizer import Voxelizer

MATERIAL_MAP = {
    0: [0.0, 1.225],
    1: [0.5, 1000.0],
    2: [0.1, 500.0]
}

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
                 receiver_manager: ReceiverManager):
        self.source_manager = source_manager
        self.receiver_manager = receiver_manager
        self.obj_filepath = obj_filepath
        self.pml_thick = pml_thick
        self.alpha_max = alpha_max
        self.npz_filepath = SCENES_OUT_DIR/obj_filepath.replace('.obj', '.npz')

        self.nx = 0
        self.ny = 0
        self.nz = 0

        with open("config.yaml", 'r') as file:
            self.yaml_data = yaml.safe_load(file)

        self.c = self.yaml_data['sound_speed']
        self.nodes_per_wavelength = self.yaml_data['nodes_per_wavelength']
        self.dim = self.yaml_data['dimensions']
        self.safety_factor = self.yaml_data['safety_factor']


    def get_time_step(self, dimensions, dx, speed, safety_factor):
        courant_limit = 1.0 / math.sqrt(dimensions)
        return (dx / speed) * courant_limit * safety_factor

    def prepare_material_core(self):
        with np.load(self.npz_filepath) as data:
            matrix_3d = data[data.files[0]].astype(np.int32)

        self.nx, self.ny, self.nz = matrix_3d.shape

        material_core = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx, self.ny, self.nz))

        ids = sorted(MATERIAL_MAP.keys())
        alphas = np.array([MATERIAL_MAP[i][0] for i in ids], dtype=np.float32)
        densities = np.array([MATERIAL_MAP[i][1] for i in ids], dtype=np.float32)

        fill_material_kernel(material_core, matrix_3d, alphas, densities)

        return material_core



    def get_obj_dimensions(self):
        try:
            mesh = trimesh.load(self.filepath, force='mesh')

            bounds = mesh.bounds

            dimensions = bounds[1] - bounds[0]

            return dimensions

        except Exception as e:
            return np.array([0.0, 0.0, 0.0])



    def builder(self):
        # trzeba dostac metry

        max_freq = self.source_manager.get_max_freq()
        wavelength = self.c / max_freq
        dx = wavelength / self.nodes_per_wavelength

        x_meters, y_meters, z_meters = self.get_obj_dimensions()

        Nx = int(x_meters/ dx)
        Ny = int(y_meters / dx)
        Nz = int(z_meters / dx)

        voxelizer = Voxelizer(Nx, Ny, Nz, dx, self.obj_filepath)
        voxelizer.load_scene()
        material_core = self.prepare_material_core()

        dt = self.get_time_step(self.dim, dx, self.c, self.safety_factor)

        fdtd_sim = FDTD_Simulation(self.c, dx, dt, self.source_manager, self.receiver_manager, material_core)



        # hujostwo

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


