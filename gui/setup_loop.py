import taichi as ti
from simulation.grid_params import GridParams
from visualization import Simulation, SceneRenderer, PlaneGeometry

class SetupLoop():
    def __init__(
            self,
            renderer: SceneRenderer,
            grid: GridParams,
            sim: Simulation,
            initial_sources: list[dict],
            initial_receivers: list[dict] = None
    ):
        self._renderer = renderer
        self._grid = grid
        self._sim = sim

        self._sources_to_setup = initial_sources

        self._microphones = initial_receivers if initial_receivers is not None else []
        self._microphone_id = len(self._microphones) + 1

        self._is_running = True
        self._edit_type: str | None = "SOURCE" # can be "SOURCE", "MIC", or None
        self._edit_index: int = 0

        self.show_voxels = False
        self.show_mesh = True

    def run(self) -> tuple[list[dict], list[dict]]:

        setup_vis = {
            "sources_pos": ti.Vector.field(3, dtype=ti.f32, shape=len(self._sources_to_setup)),
            "mics_pos": ti.Vector.field(3, dtype=ti.f32, shape=20)
        }

        while self._renderer.is_running and self._is_running:
            self._handle_setup_gui(setup_vis)

            self._renderer.render_frame(
                simulation=self._sim,
                plane_geo_1=None,
                plane_geo_2=None,
                plane_geo_3=None,
                render_enabled=True,
                setup_data=setup_vis,
                show_voxels=self.show_voxels,
                show_mesh=self.show_mesh
            )

        return self._sources_to_setup, self._microphones

    def _handle_setup_gui(self, setup_vis):
        gui = self._renderer.window.get_gui()
        with gui.sub_window("Setup Mode", 0.05, 0.05, 0.5, 0.6):
            gui.text("=== VIEW SETTINGS ===")
            self.show_voxels = gui.checkbox("Show Voxels", self.show_voxels)
            self.show_mesh = gui.checkbox("Show Mesh", self.show_mesh)
            gui.text("")

            self._gui_draw_sources_list(gui)
            self._gui_draw_microphones_list(gui, setup_vis)
            self._gui_draw_editor(gui, setup_vis)

            if gui.button("FINISH CONFIGURATION"):
                self._is_running = False

    def _gui_draw_sources_list(self, gui):
        gui.text("--- SOURCES ---")
        for i, src in enumerate(self._sources_to_setup):
            src_name = src.get('name', f'Source {i}')
            gui.text(f"Source: {src_name}")
            if gui.button(f"Edit##src_{i}"):
                self._edit_type = "SOURCE"
                self._edit_index = i

        gui.text("")

    def _gui_draw_microphones_list(self, gui, setup_vis):
        gui.text("--- MICROPHONES ---")

        if gui.button("+ Add Microphone"):
            new_mic = {"name": f"mic_{self._microphone_id}", "x": 50, "y": 50, "z": 50}
            self._microphone_id += 1
            self._microphones.append(new_mic)
            self._edit_type = "MIC"
            self._edit_index = len(self._microphones) - 1

        mic_to_delete = -1
        for i, mic in enumerate(self._microphones):
            gui.text(f"- {mic['name']}")

            if gui.button(f"Edit##mic_edit_{i}"):
                self._edit_type = "MIC"
                self._edit_index = i

            if gui.button(f"Delete##mic_del_{i}"):
                mic_to_delete = i

        if mic_to_delete != -1:
            self._microphones.pop(mic_to_delete)

            if self._edit_type == "MIC" and self._edit_index == mic_to_delete:
                self._edit_type = None
            elif self._edit_type == "MIC" and self._edit_index > mic_to_delete:
                self._edit_index -= 1

            self._sync_all_mics_to_vis(setup_vis)

        gui.text("")

    def _sync_all_mics_to_vis(self, setup_vis):
        max_mics = setup_vis["mics_pos"].shape[0]
        for i in range(max_mics):
            setup_vis["mics_pos"][i] = ti.Vector([-10000, -10000, -10000])

        for i, mic in enumerate(self._microphones):
            setup_vis["mics_pos"][i] = ti.Vector([mic['x'], mic['y'], mic['z']])

    def _gui_draw_editor(self, gui, setup_vis):
        if self._edit_type is None:
            return

        gui.text("=== EDITOR ===")

        if self._edit_type == "SOURCE" and self._edit_index < len(self._sources_to_setup):
            current_src = self._sources_to_setup[self._edit_index]
            gui.text(f"Configuring: {current_src.get('name', 'Source')}")

            c = list(current_src.get('coords', (self._grid.Nx // 2, self._grid.Ny // 2, self._grid.Nz // 2)))
            c[0] = gui.slider_int("X", int(c[0]), 0, self._grid.Nx - 1)
            c[1] = gui.slider_int("Y", int(c[1]), 0, self._grid.Ny - 1)
            c[2] = gui.slider_int("Z", int(c[2]), 0, self._grid.Nz - 1)

            current_src['coords'] = tuple(c)
            setup_vis["sources_pos"][self._edit_index] = ti.Vector([c[0], c[1], c[2]])

        elif self._edit_type == "MIC" and self._edit_index < len(self._microphones):
            m = self._microphones[self._edit_index]
            gui.text(f"Configuring: {m['name']}")

            m['x'] = gui.slider_int("Mic X", m['x'], 0, self._grid.Nx - 1)
            m['y'] = gui.slider_int("Mic Y", m['y'], 0, self._grid.Ny - 1)
            m['z'] = gui.slider_int("Mic Z", m['z'], 0, self._grid.Nz - 1)

            setup_vis["mics_pos"][self._edit_index] = ti.Vector([m['x'], m['y'], m['z']])

        gui.text("")