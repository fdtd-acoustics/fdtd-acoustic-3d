import taichi as ti

from . import config

class SceneRenderer:
    def __init__(self):
        self.window = ti.ui.Window(name="FDTD 3D Slices", res=config.SCREEN_RESOLUTION, fps_limit=config.FPS_LIMIT)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(config.CAMERA_POS_X, config.CAMERA_POS_Y, config.CAMERA_POS_Z)
        self.camera.lookat(0.5, 0.5, 0.5) # tu zmienilem

    def render_frame(self, simulation, plane_geo):
        self.camera.track_user_inputs(self.window, movement_speed=config.CAMERA_SPEED, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        self.canvas.set_background_color(config.BG_COLOR)
        self.scene.ambient_light((1, 1, 1))

        self.scene.mesh(simulation.plane_v_1, indices=plane_geo.indices, per_vertex_color=simulation.plane_c_1)
        self.scene.mesh(simulation.plane_v_2, indices=plane_geo.indices, per_vertex_color=simulation.plane_c_2)

        self.canvas.scene(self.scene)
        self.window.show()

    @property
    def is_running(self):
        return self.window.running

