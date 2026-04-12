import taichi as ti

from fdtd import SourceManager, ReceiverManager, receiver_manager
from gui import MainMenuWindow
from simulation import SimulationConfig, SimulationBuilder
from visualization import SceneRenderer, Simulation
from visualization.render_loop import RenderLoop
from visualization.vis_config import MEMORY_LIMIT_GB
from gui import SetupLoop


def run_pipeline(cfg: dict) -> None:
    sim_config = SimulationConfig.from_dict(cfg)

    builder = SimulationBuilder(sim_config)
    grid = builder.compute_grid(cfg['sources'])
    builder.voxelize(grid)

    sim = Simulation(grid, sim_config.pml_thick)
    sim.init_voxels(sim_config.npz_filepath)

    renderer = SceneRenderer(grid) # ustawianie zrodel i mikrofonow

    sources, receivers = SetupLoop(renderer, grid,sim ,cfg['sources'] ).run()

    source_manager = SourceManager.build_source_manager(sources, grid.dt )
    receiver_manager = ReceiverManager.build_receiver_manager(receivers, 2.0, grid.dt)

    fdtd_sim = builder.build_fdtd(grid, source_manager, receiver_manager)

    RenderLoop(fdtd_sim, grid, sim, renderer).run()


def main():
    ti.init(arch=ti.gpu, device_memory_GB=MEMORY_LIMIT_GB)
    app = MainMenuWindow(on_start=run_pipeline)
    app.mainloop()


if __name__ == "__main__":
    main()
