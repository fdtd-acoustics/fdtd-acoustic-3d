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
        self._render_enabled = True

        self.show_voxels = False
        self.show_mesh = True

    def run(self) -> None:
        plane_geo_1 = PlaneGeometry(self._grid.Nx, self._grid.Nz)
        plane_geo_2 = PlaneGeometry(self._grid.Nx, self._grid.Ny)
        plane_geo_3 = PlaneGeometry(self._grid.Ny, self._grid.Nz)

        slice_y = self._grid.Ny // 5 # plane_geo_1
        slice_z = self._grid.Nz // 5 # plane_geo_2
        slice_x = self._grid.Nx // 5 # plane_geo_3

        while self._renderer.is_running:
            slice_x, slice_y, slice_z = self._handle_gui(slice_x, slice_y, slice_z)

            if not self._is_paused:
                self._fdtd_sim.update()

            if self._render_enabled:
                current_pressure = self._fdtd_sim.get_current_pressure()
                self._sim.update_planes(slice_x, slice_y, slice_z, current_pressure)

            self._renderer.render_frame(
                simulation=self._sim,
                plane_geo_1=plane_geo_1,
                plane_geo_2=plane_geo_2,
                plane_geo_3=plane_geo_3,
                render_enabled=self._render_enabled,
                show_voxels=self.show_voxels,
                show_mesh=self.show_mesh
            )

    def _handle_gui(self,
                    slice_x: int,
                    slice_y: int,
                    slice_z: int,
                    ) -> tuple[int, int, int]:
        gui = self._renderer.window.get_gui()
        with gui.sub_window("2D Slices", 0.05, 0.05, 0.5, 0.35):
            gui.text(f"Step: {self._fdtd_sim.get_steps()}")
            gui.text(f"Time: {self._fdtd_sim.get_time():.4f}")

            pause_label = "RESUME" if self._is_paused else "PAUSE"
            if gui.button(pause_label):
                self._is_paused = not self._is_paused

            if not self._render_enabled:
                gui.text("Running in FAST MODE (No Rendering)")

            slice_x = gui.slider_int("X Slice", slice_x, 0, self._grid.Nx - 1)
            slice_y = gui.slider_int("Y Slice", slice_y, 0, self._grid.Ny - 1)
            slice_z = gui.slider_int("Z Slice", slice_z, 0, self._grid.Nz - 1)

            render_label = "ENABLE 3D VIEW" if not self._render_enabled else "DISABLE 3D VIEW"
            if gui.button(render_label):
                self._render_enabled = not self._render_enabled
            self.show_voxels = gui.checkbox("Show Voxels", self.show_voxels)
            self.show_mesh = gui.checkbox("Show Mesh", self.show_mesh)

        return slice_x, slice_y, slice_z
