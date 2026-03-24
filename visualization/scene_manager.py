import taichi as ti

from . import vis_config as config

class SceneRenderer:
    def __init__(self, grid):
        self.window = ti.ui.Window(name="FDTD 3D Slices", res=config.SCREEN_RESOLUTION, fps_limit=config.FPS_LIMIT)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(grid.Nx - grid.Nx//10, grid.Ny - grid.Ny//10, grid.Nz - grid.Nz//10)
        self.camera.lookat(grid.Nx//2, grid.Ny//2, grid.Nz//2)

    def render_frame(self, simulation, plane_geo_1, plane_geo_2):
        self.camera.track_user_inputs(self.window, movement_speed=config.CAMERA_SPEED, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        self.canvas.set_background_color(config.BG_COLOR)
        #self.scene.ambient_light((1, 1, 1))
        self.scene.ambient_light((0.6, 0.6, 0.6))
        self.scene.point_light(pos=(4.6, 10.0, 4.6), color=(1.0, 1.0, 1.0))

        if simulation.voxels_pos is not None and simulation.voxels_color is not None:
            self.scene.particles(
                centers=simulation.voxels_pos,
                radius=0.5,
                per_vertex_color=simulation.voxels_color
            )

        self.scene.mesh(simulation.plane_v_1, indices=plane_geo_1.indices, per_vertex_color=simulation.plane_c_1)
        self.scene.mesh(simulation.plane_v_2, indices=plane_geo_2.indices, per_vertex_color=simulation.plane_c_2)

        self.canvas.scene(self.scene)
        self.window.show()

    @property
    def is_running(self):
        return self.window.running

