import numpy as np
import taichi as ti
import trimesh
import config

# Settings
ti.init(arch=ti.gpu)

DX = 0.034300
output_file_path = config.VOXELS_DIR / "Untitled.npz"
input_mesh_path = config.MODELS_DIR / "Untitled.obj"

def show_taichi_3d():
    # --- Preparing Voxels from .npz ---
    data = np.load(output_file_path)
    space_matrix = data['material_core']

    # Removing air
    base_mask = space_matrix > 0
    indices = np.argwhere(base_mask)
    #points = indices * DX
    points = indices
    num_particles = len(points)

    # Material Colors
    colors = np.zeros((num_particles, 3), dtype=np.float32)
    for mat_id, mat_info in config.MATERIAL_MAP.items():
        if "color" in mat_info:
            colors[space_matrix[base_mask] == mat_id] = mat_info["color"]

    pos_field = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
    color_field = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
    pos_field.from_numpy(points.astype(np.float32))
    color_field.from_numpy(colors)

    print(f"===>Prepared {num_particles} voxels to show.")

    # --- Preparing Mesh from .npz ---
    mesh_verts_np = data['mesh_vertices'] / DX
    mesh_faces_np = data['mesh_faces']
    mesh_colors_np = data['mesh_colors']

    v_mesh = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh_verts_np))
    f_mesh = ti.field(dtype=ti.i32, shape=len(mesh_faces_np.flatten()))
    c_mesh = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh_colors_np))

    v_mesh.from_numpy(mesh_verts_np.astype(np.float32))
    f_mesh.from_numpy(mesh_faces_np.astype(np.int32).flatten())
    c_mesh.from_numpy(mesh_colors_np.astype(np.float32))

    # --- Window and GUI Configuration
    window = ti.ui.Window("Taichi FDTD - Voxel & Mesh Tester", (1024, 768), fps_limit=500)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    center = (np.array(space_matrix.shape) / 2)
    camera.position(center[0], center[1], center[2] + 20)
    camera.lookat(center[0], center[1], center[2])

    # --- Default GUI variables
    show_voxels = True
    show_mesh = False

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(center[0], space_matrix.shape[1] - space_matrix.shape[1]//5, center[2]), color=(1, 1, 1))

        if show_voxels:
            scene.particles(pos_field, radius=0.5, per_vertex_color=color_field)

        if show_mesh:
            scene.mesh(v_mesh, indices=f_mesh, per_vertex_color=c_mesh)

        canvas.scene(scene)

        with window.GUI.sub_window("Control Panel", 0.02, 0.02, 0.2, 0.15) as w:
            show_voxels = w.checkbox("Show Voxels", show_voxels)
            show_mesh = w.checkbox("Show Mesh", show_mesh)

        window.show()


if __name__ == "__main__":
    show_taichi_3d()