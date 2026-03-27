from fdtd import FDTD_Simulation
from simulation.grid_params import GridParams
from visualization import Simulation, SceneRenderer, PlaneGeometry


class RenderLoop:

    def __init__(
            self,
            fdtd_sim: FDTD_Simulation,
            grid: GridParams,
            sim: Simulation,
            renderer: SceneRenderer,
    ):
        self._fdtd_sim = fdtd_sim
        self._grid = grid
        self._sim = sim
        self._renderer = renderer
        self._is_paused = False

    def run(self) -> None:
        plane_geo_1 = PlaneGeometry(self._grid.Nx, self._grid.Nz)
        plane_geo_2   = PlaneGeometry(self._grid.Nx, self._grid.Ny)

        slice_y = self._grid.Ny // 5
        slice_z = self._grid.Nz // 5

        while self._renderer.is_running:
            slice_y, slice_z = self._handle_gui(slice_y, slice_z)

            if not self._is_paused:
                self._fdtd_sim.update()
            current_pressure = self._fdtd_sim.get_current_pressure()

            # self._sim.update_planes(slice_y, slice_z, current_pressure)
            # self._renderer.render_frame(simulation=self._sim, plane_geo_1=plane_geo_1, plane_geo_2=plane_geo_2)

    def _handle_gui(self,
                    slice_y: int,
                    slice_z: int,
                    ) -> tuple[int, int]:
        gui = self._renderer.window.get_gui()
        with gui.sub_window("2D Slices", 0.05, 0.05, 0.5, 0.2):
            slice_y = gui.slider_int("Horizontal Slice", slice_y, 0, self._grid.Ny - 1)
            slice_z = gui.slider_int("Vertical Slice", slice_z, 0, self._grid.Nz - 1)

            btn_label = "START" if self._is_paused else "PAUSE"
            if gui.button(btn_label):
                self._is_paused = not self._is_paused

        return slice_y, slice_z
