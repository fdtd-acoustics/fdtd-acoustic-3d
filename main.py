import taichi as ti

from fdtd import SourceManager
from gui import MainMenuWindow
from simulation import SimulationConfig, SimulationBuilder
from visualization import SceneRenderer, Simulation
from visualization.render_loop import RenderLoop
from visualization.vis_config import MEMORY_LIMIT_GB


def run_pipeline(cfg: dict) -> None:
    sim_config = SimulationConfig.from_dict(cfg)
    source_manager = SourceManager.from_dict(cfg['sources']) # to trzeba tworzyc pozniej

    builder = SimulationBuilder(sim_config, source_manager) # tu nie przekazjemy source_manager
    grid = builder.compute_grid()   # tu trzeba przekazac max_freq, to odda dt wiec po tym mozna stworzyc juz source manager
    builder.voxelize(grid)
    fdtd_sim = builder.build_fdtd(grid) # tu trzeba bedzie przekazac source managero oraz receiver manager

    sim = Simulation(grid, sim_config.pml_thick)
    sim.init_voxels(sim_config.npz_filepath)

    renderer = SceneRenderer(grid)

    #tutaj bedziemy dorzucac stawianie Sourcow, ale ten sam renderer i powinno byc git
    #po prostu mozemy stworzyc cos na wzor RenderLoopa doslownie tylko inne rzeczy bedzie robil

    RenderLoop(fdtd_sim, grid, sim, renderer).run()


def main():
    ti.init(arch=ti.gpu, device_memory_GB=MEMORY_LIMIT_GB)
    app = MainMenuWindow(on_start=run_pipeline)
    app.mainloop()


if __name__ == "__main__":
    main()
