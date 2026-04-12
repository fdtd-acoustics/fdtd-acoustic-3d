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
    ):
        self._renderer = renderer
        self._grid = grid
        self._sim = sim

        self._sources_to_setup = initial_sources
        self._microphones = []

        self._is_running = True
        self._current_mode = "SOURCES"  # Lub "RECEIVERS"
        self._active_index = 0

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
                render_enabled=True,
                setup_data=setup_vis
            )

        return self._sources_to_setup, self._microphones

    def _handle_setup_gui(self, setup_vis) -> tuple[int, int]:
        gui = self._renderer.window.get_gui()
        with gui.sub_window("Setup Mode", 0.05, 0.05, 0.5, 0.5):
            if self._current_mode == "SOURCES":
                current_src = self._sources_to_setup[self._active_index]
                gui.text(f"Configuring Source: {current_src['name']}")

                c = list(current_src.get('coords', (self._grid.Nx // 2, self._grid.Ny // 2, self._grid.Nz // 2)))
                c[0] = gui.slider_int("X", int(c[0]), 0, self._grid.Nx - 1)
                c[1] = gui.slider_int("Y", int(c[1]), 0, self._grid.Ny - 1)
                c[2] = gui.slider_int("Z", int(c[2]), 0, self._grid.Nz - 1)

                current_src['coords'] = tuple(c)
                setup_vis["sources_pos"][self._active_index] = ti.Vector([c[0], c[1], c[2]])

                if gui.button("Confirm Source"):
                    if self._active_index < len(self._sources_to_setup) - 1:
                        self._active_index += 1
                    else:
                        self._current_mode = "MICROPHONE"
                        self._active_index = 0

            elif self._current_mode == "MICROPHONE":
                gui.text(f"Microphones: {len(self._microphones)}")
                if gui.button("Add Microphone"):
                    self._microphones.append({"name": f"mic_{len(self._microphones) + 1}", "x": 50, "y": 50, "z": 50})
                    self._active_index = len(self._microphones) - 1

                if self._microphones:
                    m = self._microphones[self._active_index]
                    m['x'] = gui.slider_int("Mic X", m['x'], 0, self._grid.Nx - 1)
                    m['y'] = gui.slider_int("Mic Y", m['y'], 0, self._grid.Ny - 1)
                    m['z'] = gui.slider_int("Mic Z", m['z'], 0, self._grid.Nz - 1)
                    setup_vis["mics_pos"][self._active_index] = ti.Vector([m['x'], m['y'], m['z']])

                if gui.button("START SIMULATION"):
                    self._is_running = False