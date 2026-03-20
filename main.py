import time

import taichi as ti
import math

from simulation_config import SimulationConfig
from visualization import Simulation, SceneRenderer, config, MAX_VOXELS
from gui import MainMenuWindow

from visualization import PlaneGeometry
from fdtd import FDTD_Simulation, ReceiverManager, SourceManager
import test


# do liczenia dt
def get_time_step(dimensions, dx, speed, safety_factor):
    courant_limit = 1.0 / math.sqrt(dimensions)
    return (dx / speed) * courant_limit * safety_factor

def print_information(dt, Nx, Ny, Nz, dx, receiver_steps):
    print("Simulation parameters:")
    print(f"dt: {dt}")
    print(f"Nx: {Nx}")
    print(f"Ny: {Ny}")
    print(f"Nz: {Nz}")
    print(f"dx: {dx}")
    print(f"receiver steps: {receiver_steps}")


def run_pipeline(config: dict):
    source_manager = SourceManager(len(config['sources']))
    for source in config['sources']:
        if source['type'] == 'Gauss':
            source_manager.add_source(
                name="gauss_source_",
                freq=source['freq'],
                amplitude=source['amp'],
            )
        elif source['type'] == 'Custom':
            source_manager.add_source(
                name="custom_source_",
                freq=None,
                amplitude=None,
            )
    simulation_conf = SimulationConfig(
        obj_filepath=config['obj_file'],
        pml_thick=config['pml_thickness'],
        alpha_max=config['alpha_max'],
        source_manager=source_manager,
    )
    simulation_conf.builder()


def main():
    ti.init(arch=ti.gpu, device_memory_GB=config.MEMORY_LIMIT_GB)

    app = MainMenuWindow(on_start=run_pipeline)
    app.mainloop()

    # simulation_conf = SimulationConfig()





if __name__ == "__main__":
    main()