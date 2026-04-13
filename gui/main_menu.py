import tkinter as tk

from fdtd import SourceManager, ReceiverManager
from gui import NewSimulationWindow
from simulation import SimulationConfig, SimulationBuilder
from visualization import SceneRenderer, Simulation
from visualization.render_loop import RenderLoop
from .setup_loop import SetupLoop

class MainMenuWindow(tk.Tk):
    def __init__(self, on_start=None) -> None:
        super().__init__()
        self.title("FDTD simulator")
        self.geometry("300x400")
        self.resizable(False, False)

        tk.Label(self, text="FDTD 3D Solver", font=("Arial", 14, "bold")).pack(pady=20)

        tk.Button(self, text="New Simulation", width=25, height=2,
                  command=self.create_new_sim).pack(pady=10)

        tk.Button(self, text="Load Simulation (.npz)", width=25, height=2,
                  command=self.load_simulation).pack(pady=10)

        tk.Button(self, text="Material Library", width=25, height=2,
                  command=self.open_materials).pack(pady=10)



    def create_new_sim(self):
        self.withdraw()
        new_sim_window = NewSimulationWindow(on_start=self.run_pipeline)
        new_sim_window.protocol("WM_DELETE_WINDOW", lambda: self.on_close_subwindow(new_sim_window))

    def on_close_subwindow(self, window):
        window.destroy()
        self.deiconify()

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
        receiver_manager = ReceiverManager.build_receiver_manager(receivers, sim_config.record_time, grid.dt)

        fdtd_sim = builder.build_fdtd(grid, source_manager, receiver_manager)

        RenderLoop(fdtd_sim, grid, sim, renderer).run()



    def load_simulation(self):
        pass

    def open_materials(self):
        pass