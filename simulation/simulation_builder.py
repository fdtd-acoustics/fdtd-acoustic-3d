import math

import numpy as np
import taichi as ti
import trimesh

from config import MAIN_MATERIAL_LIBRARY
from fdtd import FDTD_Simulation, SourceManager, ReceiverManager
from voxelization import Voxelizer

from .grid_params import GridParams
from .simulation_config import SimulationConfig

import os
import csv

@ti.kernel
def _fill_material_kernel(
        material_field: ti.template(),
        data_np: ti.types.ndarray(),
        map_alpha: ti.types.ndarray(),
        map_density: ti.types.ndarray()
):
    for i, j, k in material_field:
        mat_id = data_np[i, j, k]
        material_field[i, j, k][0] = map_alpha[mat_id]
        material_field[i, j, k][1] = map_density[mat_id]

class SimulationBuilder:
    """Main Builder which gives access to simulation building methods"""

    def __init__(self, cfg: SimulationConfig):
        self._cfg = cfg

    def compute_dx(self, sources_cfg: list[dict]) -> float:
        max_freq = SourceManager.get_highest_frequency(sources=sources_cfg)
        wavelength = self._cfg.sound_speed / max_freq
        dx = wavelength / self._cfg.nodes_per_wavelength
        return dx

    def compute_grid(self, sources_cfg: list[dict]) -> GridParams:
        dx = self.compute_dx(sources_cfg)
        dt = self._compute_dt(dx)

        x_meters, y_meters, z_meters = self._load_obj_dimensions()

        Nx = int(x_meters / dx)
        Ny = int(y_meters / dx)
        Nz = int(z_meters / dx)

        grid = GridParams(dx=dx, dt=dt, Nx=Nx, Ny=Ny, Nz=Nz)
        self._log_grid(grid)
        return grid

    def grid_from_data(self, loaded_data) -> GridParams:
        try:
            dx = float(loaded_data['dx'])
            dt = float(loaded_data['dt'])
            Nx = int(loaded_data['Nx'])
            Ny = int(loaded_data['Ny'])
            Nz = int(loaded_data['Nz'])

            return GridParams(
                dx=dx,
                dt=dt,
                Nx=Nx,
                Ny=Ny,
                Nz=Nz
            )
        except KeyError as e:
            print(f"Error: Missing grid parameter in loaded data: {e}")
            raise ValueError(f"Incomplete grid data in NPZ file. Missing: {e}")

    def voxelize(self, grid: GridParams) -> None:
        """Runs voxelization and saves to .npz"""
        voxelizer = Voxelizer(
            grid.Nx, grid.Ny, grid.Nz,
            grid.dx,
            self._cfg.obj_filepath
        )
        voxelizer.load_scene()

    def build_fdtd(self, grid: GridParams, source_manager: SourceManager, receiver_manager: ReceiverManager) -> FDTD_Simulation:
        """Builds FDTD_Simulation"""
        material_core = self._prepare_material_core()
        return FDTD_Simulation(
            sound_speed=self._cfg.sound_speed,
            dx=grid.dx,
            dt=grid.dt,
            pml_thick = self._cfg.pml_thick,
            alpha_max= self._cfg.alpha_max,
            safety_factor = self._cfg.safety_factor,
            source_manager=source_manager,
            receiver_manager = receiver_manager,
            material_core=material_core,
        )

    def _prepare_material_core(self) -> ti.Vector:
        with np.load(self._cfg.npz_filepath) as data:
            matrix_3d = data['material_core'].astype(np.int32)

        nx, ny, nz = matrix_3d.shape

        material_core = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny, nz))

        material_map = self.get_material_map_from_csv(MAIN_MATERIAL_LIBRARY)

        ids = sorted(material_map.keys())
        alphas = np.array([material_map[i]['alpha'] for i in ids], dtype=np.float32)
        densities = np.array([material_map[i]['density'] for i in ids], dtype=np.float32)

        _fill_material_kernel(material_core, matrix_3d, alphas, densities)

        return material_core

    @staticmethod
    def get_material_map_from_csv(csv_path):
        if not os.path.exists(csv_path):
            print(f"Note: File {csv_path} does not exist.")
            return
        materials_map ={}

        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')

            try:
                next(reader)
            except StopIteration:
                return

            for row in reader:
                if not row or len(row) < 4:
                    continue

                try:
                    m_id = int(row[0])
                    materials_map[m_id] = {
                        'name': str(row[1]),
                        'alpha': float(row[2]),
                        'density': float(row[3]),
                        'color': str(row[4])
                    }
                except ValueError as e:
                    print(f"Error skipping row {row}: {e}")
                    continue

        return materials_map

    @staticmethod
    def hex_to_rgb_normalized(hex_color):
        hex_color = hex_color.lstrip('#')
        return [int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


    def _load_obj_dimensions(self):
        try:
            mesh = trimesh.load(self._cfg.obj_filepath, force='mesh')

            bounds = mesh.bounds

            dimensions = bounds[1] - bounds[0]

            return dimensions
        except Exception as e:
            print(f"[SimulationBuilder] Error loading obj file {e}")
            return np.array([0.0, 0.0, 0.0])

    def _compute_dt(self, dx: float) -> float:
        courant_limit = 1.0 / math.sqrt(self._cfg.dim)
        return (dx / self._cfg.sound_speed) * courant_limit * self._cfg.safety_factor

    def validate_dx(self, grid: GridParams, sources_cfg: list[dict]) -> bool:
        required_dx = self.compute_dx(sources_cfg)

        return grid.dx <= required_dx

    def get_max_safe_frequency(self, dx: float) -> float:
        wavelength_min = dx * self._cfg.nodes_per_wavelength
        max_freq = self._cfg.sound_speed / wavelength_min
        return max_freq


    @staticmethod
    def _log_grid(grid: GridParams) -> None:
        print("Simulation parameters:")
        print(f"  dx={grid.dx:.6f}  dt={grid.dt:.6f}")
        print(f"  Nx={grid.Nx}  Ny={grid.Ny}  Nz={grid.Nz}")

    @staticmethod
    def save_full_configuration(self, path, grid, sim_config, sources, receivers):
        try:
            base_npz_path = sim_config.npz_filepath
            with np.load(base_npz_path, allow_pickle=True) as data:
                data_to_save = dict(data)  # mapa materialow

            data_to_save.update({
                'obj_filepath': np.array(str(sim_config.obj_filepath)),
                'sources': np.array(sources, dtype=object),
                'receivers': np.array(receivers, dtype=object),
                'pml_thick': np.array(sim_config.pml_thick),
                'alpha_max': np.array(sim_config.alpha_max),
                'record_time': np.array(sim_config.record_time),
                'sound_speed': np.array(sim_config.sound_speed),

                'dx': np.array(grid.dx),
                'dt': np.array(grid.dt),
                'Nx': np.array(grid.Nx),
                'Ny': np.array(grid.Ny),
                'Nz': np.array(grid.Nz),

                'safety_factor': np.array(sim_config.safety_factor),
                'nodes_per_wavelength': np.array(sim_config.nodes_per_wavelength)
            })

            np.savez(path, **data_to_save)
            print(f"Full configuration merged and saved to: {base_npz_path}")

        except FileNotFoundError:
            print(f"Error: Base voxel file not found at {path}")
            raise
        except Exception as e:
            print(f"An error occurred during saving: {e}")
            raise

    @staticmethod
    def load_project_data(path: str) -> dict:
        data = np.load(path, allow_pickle=True)
        return dict(data)
