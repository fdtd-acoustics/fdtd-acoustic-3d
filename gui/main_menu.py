import tkinter as tk

from fdtd import SourceManager, ReceiverManager, FDTD_Simulation
from gui import NewSimulationWindow
from simulation import SimulationConfig, SimulationBuilder
from visualization import SceneRenderer, Simulation
from visualization.render_loop import RenderLoop
from .setup_loop import SetupLoop
from tkinter import messagebox, filedialog, Grid
import config
import numpy as np
from simulation import GridParams
import taichi as ti

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

    def run_pipeline(self, cfg: dict) -> None:
        sim_config = SimulationConfig.from_dict(cfg)

        builder = SimulationBuilder(sim_config)
        grid = builder.compute_grid(cfg['sources'])
        builder.voxelize(grid)

        sim = Simulation(grid, sim_config.pml_thick)
        data = np.load(sim_config.npz_filepath)

        space_matrix = data['material_core']
        sim.init_voxels(space_matrix)

        renderer = SceneRenderer(grid)

        sources, receivers = SetupLoop(renderer, grid,sim ,cfg['sources'] ).run() # ustawianie zrodel i mikrofonow

        action = self.show_post_setup_dialog(sim_config,grid, sources, receivers)  # mozliwosc zapisu konfiguracji do .npz

        if action == "start":
            source_manager = SourceManager.build_source_manager(sources, grid.dt )
            receiver_manager = ReceiverManager.build_receiver_manager(receivers, sim_config.record_time, grid.dt)

            fdtd_sim = builder.build_fdtd(grid, source_manager, receiver_manager)

            RenderLoop(fdtd_sim, grid, sim, renderer).run()

        else:
            self.deiconify()



    def load_simulation(self):
        loaded_data = self.load_data_from_npz()

        fdtd_sim, grid, sim = self.prepare_data(loaded_data)

        renderer = SceneRenderer(grid)
        RenderLoop(fdtd_sim, grid, sim, renderer).run()



    def prepare_data(self, data):
        space_matrix = data['material_core']

        source_manager = data['sources']
        receiver_manager = data['receivers']

        pml_thick = int(data['pml_thick'])
        alpha_max = float(data['alpha_max'])

        sound_speed = float(data['sound_speed'])
        record_time = float(data['record_time'])

        nodes_per_wavelength = int(data['nodes_per_wavelength'])
        safety_factor = float(data['safety_factor'])

        dx = float(data['dx'])
        dt = float(data['dt'])

        Nx = int(data['Nx'])
        Ny = int(data['Ny'])
        Nz = int(data['Nz'])

        grid = GridParams(dx, dt, Nx, Ny, Nz)

        sim = Simulation(grid, pml_thick)
        sim.init_voxels(space_matrix)

        fdtd_sim = FDTD_Simulation(sound_speed, dx, dt, pml_thick, alpha_max, safety_factor, source_manager, receiver_manager, space_matrix)

        return fdtd_sim, grid, sim


    def load_data_from_npz(self):
        path = filedialog.askopenfilename(
            initialdir=config.PROJECTS_DIR,
            title="Select Simulation Project (.npz)",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if not path:
            return None

        try:
            data = np.load(path, allow_pickle=True)

            if 'sources' not in data.files:
                messagebox.showwarning(
                    "Incomplete Project",
                    "This file only contains geometry. You will need to set up sources and microphones again."
                )

            print(f"Project loaded successfully: {path}")
            return data

        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load the file:\n{e}")
            return None

    def open_materials(self):
        #TODO
        pass

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



    def save_full_configuration(self, path, grid, sim_config, sources, receivers):
        try:
            base_npz_path = sim_config.npz_filepath
            with np.load(base_npz_path, allow_pickle=True) as data:
                data_to_save = dict(data)  # mapa materialow

            data_to_save.update({
                'sources': np.array(sources, dtype=object),
                'receivers': np.array(receivers, dtype=object),
                'pml_thick': np.array(sim_config.pml_thick),
                'alpha_max': np.array(sim_config.alpha_max),
                'record_time': np.array(sim_config.record_time),
                'sound_speed': np.array(sim_config.sound_speed),

                'dx': np.array(grid.dx),
                'dt': np.array(grid.dt),

                'safety_factor': np.array(sim_config.safety_factor),
                'nodes_per_wavelength': np.array(sim_config.nodes_per_wavelength),

                'Nx':np.array(grid.Nx),
                'Ny':np.array(grid.Ny),
                'Nz':np.array(grid.Nz)
            })

            np.savez(path, **data_to_save)
            print(f"Full configuration merged and saved to: {base_npz_path}")

        except FileNotFoundError:
            print(f"Error: Base voxel file not found at {path}")
            raise
        except Exception as e:
            print(f"An error occurred during saving: {e}")
            raise
