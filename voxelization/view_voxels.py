import numpy as np
import taichi as ti
from visualization.config import SCENES_OUT_DIR

DX = 0.34000

ti.init(arch=ti.gpu)

output_file_path = SCENES_OUT_DIR / "test_room4.npz"

def show_taichi_3d():
    data = np.load(output_file_path)
    space_matrix = data['material_core']

    #usuniecie powietrza
    base_mask = space_matrix > 0
    indices = np.argwhere(base_mask)

    points = indices * DX
    num_particles = len(points)

    #kolory dla materialow
    colors = np.zeros((num_particles, 3), dtype=np.float32)
    colors[space_matrix[base_mask] == 1] = [0.5, 0.5, 0.5]
    colors[space_matrix[base_mask] == 2] = [0.9, 0.7, 0.1]

    print(f"===>Prepared {num_particles} voxels to show.")

    pos_field = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
    color_field = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)

    pos_field.from_numpy(points.astype(np.float32))
    color_field.from_numpy(colors)

    window = ti.ui.Window("Taichi FDTD - Voxel Tester", (1024, 768), fps_limit=500)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    camera.position(4.6, 4.6, 15.0)
    camera.lookat(4.6, 4.6, 4.6)

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(4.6, 10.0, 4.6), color=(1.0, 1.0, 1.0))

        scene.particles(pos_field, radius=0.02, per_vertex_color=color_field)
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    show_taichi_3d()