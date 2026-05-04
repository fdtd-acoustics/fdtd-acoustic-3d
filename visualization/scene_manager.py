import taichi as ti
import cv2
from . import vis_config as config

class SceneRenderer:
    def __init__(self, grid):
        # Main Window
        self.window = ti.ui.Window(name="FDTD 3D Slices", res=config.SCREEN_RESOLUTION, fps_limit=config.FPS_LIMIT)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(grid.Nx - grid.Nx//10, grid.Ny - grid.Ny//10, grid.Nz - grid.Nz//10)
        self.camera.lookat(grid.Nx//2, grid.Ny//2, grid.Nz//2)
        self.camera.fov(60)
        self.light_pos = (grid.Nx / 2, grid.Ny - grid.Ny / 5, grid.Nz / 2)

        # Slices Windows
        self.win_name_x = "Slice X (YZ Plane)"
        self.win_name_y = "Slice Y (XZ Plane)"
        self.win_name_z = "Slice Z (XY Plane)"

        scale = 2
        w_x, h_x = grid.Nz * scale, grid.Ny * scale
        w_y, h_y = grid.Nz * scale, grid.Nx * scale
        w_z, h_z = grid.Ny * scale, grid.Nx * scale

        spacing = 30
        start_x = 100
        start_y = config.SCREEN_RESOLUTION[1] + 4 * spacing

        self.cv_windows = {
            'x': {"name": "Slice X (YZ Plane)", "w": w_x, "h": h_x, "x": start_x, "y": start_y, "active": False},
            'y': {"name": "Slice Y (XZ Plane)", "w": w_y, "h": h_y, "x": start_x + w_x + spacing, "y": start_y, "active": False},
            'z': {"name": "Slice Z (XY Plane)", "w": w_z, "h": h_z, "x": start_x + w_x + w_y + 2 * spacing, "y": start_y, "active": False}
        }

    def _manage_cv_window(self, key, should_show, img_tensor):
        win = self.cv_windows[key]

        if should_show:
            if not win["active"]:
                cv2.namedWindow(win["name"], cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win["name"], win["w"], win["h"])
                cv2.moveWindow(win["name"], win["x"], win["y"])
                win["active"] = True

            img = img_tensor.to_numpy()
            cv2.imshow(win["name"], img[:, :, ::-1])
        else:
            if win["active"]:
                try:
                    cv2.destroyWindow(win["name"])
                except Exception:
                    pass
                win["active"] = False

    def render_frame(self, simulation, plane_geo_1, plane_geo_2, plane_geo_3, render_enabled,
                     setup_data=None, show_voxels=False, show_mesh=False,
                     show_slice_x=False, show_slice_y=False, show_slice_z=False):

        # Main Window
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

            self.canvas.scene(self.scene)
        else:
            pass

        self.window.show()

        # Slice Windows
        if render_enabled:
            if hasattr(simulation, 'image_slice_x'):
                self._manage_cv_window('x', show_slice_x, simulation.image_slice_x)

            if hasattr(simulation, 'image_slice_y'):
                self._manage_cv_window('y', show_slice_y, simulation.image_slice_y)

            if hasattr(simulation, 'image_slice_z'):
                self._manage_cv_window('z', show_slice_z, simulation.image_slice_z)

            cv2.waitKey(1)

    @property
    def is_running(self):
        return self.window.running

