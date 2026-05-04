import tkinter as tk

from fdtd import SourceManager, ReceiverManager, FDTD_Simulation
from .new_simulation_menu import NewSimulationWindow
from .material_library import MaterialLibraryWindow
from simulation import SimulationConfig, SimulationBuilder
from visualization import SceneRenderer, Simulation
from visualization.render_loop import RenderLoop
from .material_library import MaterialLibraryWindow
from .setup_loop import SetupLoop
from tkinter import messagebox, filedialog, Grid
import config
import numpy as np

class MainMenuWindow(tk.Tk):
    def __init__(self, on_start=None) -> None:
        super().__init__()
        self.title("FDTD simulator")
        self.geometry("300x400")
        self.resizable(False, False)

        tk.Button(self, text="New Simulation", width=25, height=2,
                  command=self.create_new_sim).pack(pady=10)

        tk.Button(self, text="Load Simulation (.npz)", width=25, height=2,
                  command=self.load_simulation).pack(pady=10)

        tk.Button(self, text="Material Library", width=25, height=2,
                  command=self.open_materials).pack(pady=10)

        bottom_frame = tk.Frame(self, padx=10, pady=10)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.info_btn = tk.Button(
            bottom_frame,
            text="ⓘ",
            width=4,
            command=self.show_info,
            font=("Arial", 12)
        )
        self.info_btn.pack(side=tk.LEFT)


        self.exit_btn = tk.Button(
            bottom_frame,
            text="Exit",
            width=8,
            command=self.quit
        )
        self.exit_btn.pack(side=tk.RIGHT)

    def show_info(self):
        messagebox.showinfo(
            "Information",
            "Version 1.0".center(50)
        )

    def create_new_sim(self):
        self.withdraw()
        new_sim_window = NewSimulationWindow(on_start=self.run_pipeline)
        new_sim_window.protocol("WM_DELETE_WINDOW", lambda: self.on_close_subwindow(new_sim_window))

    def on_close_subwindow(self, window):
        window.destroy()
        self.deiconify()

    def run_pipeline(self, cfg: dict, loaded_data: dict | None = None) -> bool:
        sim_config = SimulationConfig.from_dict(cfg)

        builder = SimulationBuilder(sim_config)
        if loaded_data is not None:
            grid = builder.grid_from_data(loaded_data)

            # check if source frequencies are consistent with the grid resolution
            if not builder.validate_dx(grid, cfg['sources']):
                from tkinter import messagebox

                max_frequency = builder.get_max_safe_frequency(grid.dx)

                messagebox.showerror(
                    "Frequency Too High",
            f"The grid is configured for a maximum frequency of {max_frequency:.0f} Hz, "
                    f"but one of your sources exceeds this limit.\n\n"
                    f"This may cause numerical instability. Please ensure all sources are set "
                    f"to a frequency no higher than {max_frequency:.0f} Hz."
                )
                return False

            space_matrix = loaded_data['material_core']

            mesh_verts = loaded_data.get('mesh_vertices')
            mesh_faces = loaded_data.get('mesh_faces')
            mesh_colors = loaded_data.get('mesh_colors')

            initial_sources = cfg['sources']
            initial_receivers = loaded_data.get('receivers')
        else:
            grid = builder.compute_grid(cfg['sources'])
            builder.voxelize(grid)

            data = np.load(sim_config.npz_filepath)
            space_matrix = data['material_core']

            mesh_verts = data.get('mesh_vertices')
            mesh_faces = data.get('mesh_faces')
            mesh_colors = data.get('mesh_colors')

            initial_sources = cfg['sources']
            initial_receivers = []

        sim = Simulation(grid, sim_config.pml_thick)
        sim.init_voxels(space_matrix)
        sim.init_mesh(mesh_verts, mesh_faces, mesh_colors, dx=grid.dx)

        renderer = SceneRenderer(grid)

        sources, receivers = SetupLoop(renderer, grid,sim ,initial_sources, initial_receivers).run() # ustawianie zrodel i mikrofonow

        action = self.show_post_setup_dialog(sim_config,grid, sources, receivers)  # mozliwosc zapisu konfiguracji do .npz

        if action == "start":
            source_manager = SourceManager.build_source_manager(sources, grid.dt )
            receiver_manager = ReceiverManager.build_receiver_manager(receivers, sim_config.record_time, grid.dt)

            fdtd_sim = builder.build_fdtd(grid, source_manager, receiver_manager)
            RenderLoop(fdtd_sim, grid, sim, renderer).run()
        else:
            self.deiconify()
        return True


    def load_simulation(self):
        loaded_data = self.load_data_from_npz()

        self.withdraw()
        new_sim_window = NewSimulationWindow(on_start=self.run_pipeline, loaded_data = loaded_data)
        new_sim_window.protocol("WM_DELETE_WINDOW", lambda: self.on_close_subwindow(new_sim_window))


    def load_data_from_npz(self):
        path = filedialog.askopenfilename(
            initialdir=config.PROJECTS_DIR,
            title="Select Simulation Project (.npz)",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )

        if not path:
            return None

        try:
            data = SimulationBuilder.load_project_data(path)
            return data

        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load project:\n{e}")
            return None

    def open_materials(self):
        self.withdraw()
        library_material_windows = MaterialLibraryWindow(on_close=lambda: self.on_close_subwindow(library_material_windows))
        library_material_windows.protocol("WM_DELETE_WINDOW", lambda: self.on_close_subwindow(library_material_windows))


    def show_post_setup_dialog(self, sim_config,grid, sources, receivers):
        dialog = tk.Toplevel(self)
        dialog.title("Setup Complete")
        dialog.geometry("400x250")
        dialog.grab_set()

        tk.Label(dialog, text="Scene Configuration Finished!", font=('Arial', 12, 'bold')).pack(pady=10)

        info_text = (
            f"Sources: {len(sources)}\n"
            f"Microphones: {len(receivers)}\n"
        )
        tk.Label(dialog, text=info_text, justify="left").pack(pady=5)

        status_label = tk.Label(dialog, text="", font=('Arial', 10, 'bold'))
        status_label.pack(pady=5)

        result = {"action": "cancel"}

        def on_save():
            path = filedialog.asksaveasfilename(
                initialdir=config.PROJECTS_DIR,
                defaultextension=".npz",
                filetypes=[("NPZ files", "*.npz")],
                title="Save Full Simulation Project"
            )
            if path:
                try:
                    self.save_full_configuration(path, grid, sim_config, sources, receivers)
                    filename = path.split('/')[-1]
                    status_label.config(text=f"Saved: {filename}", fg="#28a745")
                    dialog.update_idletasks()
                except Exception as e:
                    status_label.config(text=f"Save Error!", fg="red")
                    print(f"Error details: {e}")

        def on_start():
            result["action"] = "start"
            dialog.destroy()

        def on_close():
            result["action"] = "cancel"
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_close)

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="Save Project (.npz)", command=on_save, width=18).pack(side="left", padx=5)

        tk.Button(btn_frame, text="START SIMULATION", command=on_start, width=18, bg="#28a745", fg="white",
                  font=('Arial', 10, 'bold')).pack(side="left", padx=5)

        self.wait_window(dialog)
        return result["action"]

