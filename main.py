import taichi as ti

from fdtd import SourceManager, ReceiverManager
from gui import MainMenuWindow
from simulation import SimulationConfig, SimulationBuilder
from visualization import SceneRenderer, Simulation
from visualization.render_loop import RenderLoop
from visualization.vis_config import MEMORY_LIMIT_GB


def run_pipeline(cfg: dict) -> None:
    sim_config = SimulationConfig.from_dict(cfg)

    builder = SimulationBuilder(sim_config)
    grid = builder.compute_grid(cfg['sources'])
    builder.voxelize(grid)

    sim = Simulation(grid, sim_config.pml_thick)
    sim.init_voxels(sim_config.npz_filepath)

    renderer = SceneRenderer(grid)

    #tutaj bedziemy dorzucac stawianie Sourcow, ale ten sam renderer i powinno byc git
    #po prostu mozemy stworzyc cos na wzor RenderLoopa doslownie tylko inne rzeczy bedzie robil

    max_steps = 10000 # pozniej w gui bedzie to zdobywac
    source_manager = SourceManager.from_dict(cfg['sources'], max_steps, grid.dt ) #TODO tu trzeba bedzie tez przekazac tablice position i od razu przy dodawaniu zrodel ustawiac
    source_manager.set_pos(0, 30, 30, 30)  # nie uwzglednia pml
    receiver_manager = ReceiverManager(5, 20000, grid.dt)
    receiver_manager.add_receiver(35,31,32, "mik1") # nie uwzilednia pmla
    receiver_manager.add_receiver(30, 30, 30, "src")  # nie uwzilednia pmla
    receiver_manager.add_receiver(22, 22, 22, "mic2")  # nie uwzilednia pmla
    fdtd_sim = builder.build_fdtd(grid, source_manager, receiver_manager)


    RenderLoop(fdtd_sim, grid, sim, renderer).run()


def main():
    ti.init(arch=ti.gpu, device_memory_GB=MEMORY_LIMIT_GB)
    app = MainMenuWindow(on_start=run_pipeline)
    app.mainloop()


if __name__ == "__main__":
    main()
