import taichi as ti

from . import vis_config as config
from .hud import ColorScaleHUD, VectorTextHUD

class SceneRenderer:
    def __init__(self, grid):
        self.window = ti.ui.Window(name="FDTD 3D Slices", res=config.SCREEN_RESOLUTION, fps_limit=config.FPS_LIMIT)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(grid.Nx - grid.Nx//10, grid.Ny - grid.Ny//10, grid.Nz - grid.Nz//10)
        self.camera.lookat(grid.Nx//2, grid.Ny//2, grid.Nz//2)
        self.camera.fov(60)
        self.light_pos = (grid.Nx / 2, grid.Ny - grid.Ny / 5, grid.Nz / 2)

        self.hud = ColorScaleHUD()
        self.text_hud = VectorTextHUD()

        # pressure scale
        # self.text_hud.set_text([
        #     ("-2", 0.20, 0.02),
        #     ("-1", 0.35, 0.02),
        #     ("0", 0.50, 0.02),
        #     ("1", 0.64, 0.02),
        #     ("2", 0.79, 0.02)
        # ], size=0.02)

        # decibel scale
        self.text_hud.set_text([
            ("-120 dB", 0.16, 0.02),
            ("-80 dB", 0.31, 0.02),
            (" 0 dB", 0.49, 0.02),
            ("+80 dB", 0.63, 0.02),
            ("+120 dB", 0.77, 0.02)
        ], size=0.02)

    def render_frame(self, simulation, plane_geo_1, plane_geo_2, plane_geo_3, render_enabled, setup_data=None, show_voxels=False, show_mesh=False, show_scale=False, color_hud=None, text_hud=None):

        if not render_enabled:
            return

        self.camera.track_user_inputs(self.window, movement_speed=config.CAMERA_SPEED, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        self.canvas.set_background_color(config.BG_COLOR)
        self.scene.ambient_light((0.6, 0.6, 0.6))
        self.scene.point_light(pos=self.light_pos, color=(1.0, 1.0, 1.0))

        if render_enabled:
            # Voxels Rendering
            if show_voxels and simulation.voxels_pos is not None:
                self.scene.particles(centers=simulation.voxels_pos, radius=0.5, per_vertex_color=simulation.voxels_color)

            # Mesh Rendering
            if show_mesh and simulation.has_mesh:
                self.scene.mesh(simulation.v_mesh, indices=simulation.f_mesh, per_vertex_color=simulation.c_mesh)

            # Setup data
            if setup_data is not None:
                self.scene.particles(centers=setup_data["sources_pos"], radius=0.6, color=(1, 0, 0))
                self.scene.particles(centers=setup_data["mics_pos"], radius=0.6, color=(0, 0.5, 1))

            # 2D Slices
            elif plane_geo_1 is not None and plane_geo_2 is not None and plane_geo_3 is not None:
                self.scene.mesh(simulation.plane_v_1, indices=plane_geo_1.indices, per_vertex_color=simulation.plane_c_1)
                self.scene.mesh(simulation.plane_v_2, indices=plane_geo_2.indices, per_vertex_color=simulation.plane_c_2)
                self.scene.mesh(simulation.plane_v_3, indices=plane_geo_3.indices, per_vertex_color=simulation.plane_c_3)

            canvas = self.window.get_canvas()
            canvas.scene(self.scene)

            if color_hud:
                color_hud.draw(canvas)
            if text_hud:
                text_hud.draw(canvas, color=(1.0, 1.0, 1.0))
        else:
            pass

        self.window.show()

    @property
    def is_running(self):
        return self.window.running

