from fdtd import FDTD_Simulation
from simulation.grid_params import GridParams
from visualization import Simulation, SceneRenderer, PlaneGeometry
from visualization.hud import ColorScaleHUD, VectorTextHUD

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
        self.show_scale = True
        self.use_log_scale = True
        self.max_db = 120.0

        self.color_hud = ColorScaleHUD(x_start=0.2, x_end=0.8, y_start=0.05, y_end=0.1)
        self.text_hud = VectorTextHUD(max_chars=50)

    def run(self) -> None:
        plane_geo_1 = PlaneGeometry(self._grid.Nx, self._grid.Nz)
        plane_geo_2 = PlaneGeometry(self._grid.Nx, self._grid.Ny)
        plane_geo_3 = PlaneGeometry(self._grid.Ny, self._grid.Nz)

        slice_y = self._grid.Ny // 5 # plane_geo_1
        slice_z = self._grid.Nz // 5 # plane_geo_2
        slice_x = self._grid.Nx // 5 # plane_geo_3

        use_log_scale_int = 1 if self.use_log_scale else 0

        while self._renderer.is_running:
            slice_x, slice_y, slice_z = self._handle_gui(slice_x, slice_y, slice_z)

            if not self._is_paused:
                self._fdtd_sim.update()

            if self._render_enabled:
                current_pressure = self._fdtd_sim.get_current_pressure()
                use_log_scale_int = 1 if self.use_log_scale else 0

                self._sim.update_planes(slice_x, slice_y, slice_z, current_pressure, use_log_scale_int, self.max_db)

            if self.show_scale and self._render_enabled:
                if self.use_log_scale:
                    max_val = int(self.max_db)
                else:
                    max_val = 1000

                texts = [
                    (f"-{max_val}", 0.18, 0.12),
                    ("0", 0.49, 0.12),
                    (f"{max_val}", 0.78, 0.12)
                ]
                self.text_hud.set_text(texts, size=0.025)

            self._renderer.render_frame(
                simulation=self._sim,
                plane_geo_1=plane_geo_1,
                plane_geo_2=plane_geo_2,
                plane_geo_3=plane_geo_3,
                render_enabled=self._render_enabled,
                show_voxels=self.show_voxels,
                show_mesh=self.show_mesh,
                show_scale=self.show_scale,
                color_hud = self.color_hud if self.show_scale else None,
                text_hud = self.text_hud if self.show_scale else None
            )

    def _handle_gui(self,
                    slice_x: int,
                    slice_y: int,
                    slice_z: int,
                    ) -> tuple[int, int, int]:

        gui = self._renderer.window.get_gui()
        with gui.sub_window("2D Slices", 0.025, 0.025, 0.3, 0.45):
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

            scale_label = "HIDE LEGEND" if self.show_scale else "SHOW LEGEND"
            if gui.button(scale_label):
                self.show_scale = not self.show_scale

            self.use_log_scale = gui.checkbox("Use decibel scale (recommended)", self.use_log_scale)
            if self.use_log_scale:
                self.max_db = gui.slider_float("Max decibel limit", self.max_db, 60.0, 200.0)

        return slice_x, slice_y, slice_z
