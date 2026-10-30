# ==============================================================================
# File:         room_2d_visualizer.py
# Description:  Provides 2D visualization utilities for FDTD simulation rooms and grids.
#               Features dynamic material mapping from 'materials.csv', custom side legends,
#               coordinate systems anchored at the bottom-left (0,0), and support for plotting
#               in both voxels and physical meters.

# Functions:
#   - visualize_room_data():
#       Renders room layout, sources, receivers, and available foam slots.
#   - visualize_simulation_grid():
#       Renders simulation grid states without slots.

# Usage Example:
#   from assets.rooms_2d.room_2d_visualizer import visualize_room_data
#   visualize_room_data(room_data, title="Room Preview", save_path="room.png", use_meters=False)
#   visualize_room_data(room_data, title="Room Preview", save_path="room_m.png", use_meters=True)
# ==============================================================================

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RASTERIZED_FOLDER = 'rasterized_rooms'
IMAGES_FOLDER = 'visualized_rooms'


def _load_material_library():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(current_dir, '..', 'material_library', 'materials.csv')

    id_to_color = {}
    id_to_name = {}

    if csv_path:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    mat_id = int(row['id'])
                    id_to_color[mat_id] = row['color']
                    id_to_name[mat_id] = row['name']
                except (ValueError, KeyError):
                    continue

    if not id_to_color:
        print("Wrong path to .csv file")


    return id_to_color, id_to_name


def visualize_room_data(data, title="Room Preview", save_path=None, show=False, use_meters: bool = False):
    grid = data['material_core']
    dx = data['dx'].item() if isinstance(data['dx'], np.ndarray) else data['dx']
    nx, ny = grid.shape

    src_data = data['sources'] if 'sources' in data else []
    sources = src_data.tolist() if isinstance(src_data, np.ndarray) else src_data

    recv_data = data['receivers'] if 'receivers' in data else []
    receivers = recv_data.tolist() if isinstance(recv_data, np.ndarray) else recv_data

    slot_data = data['foam_slots'] if 'foam_slots' in data else []
    foam_slots = slot_data.tolist() if isinstance(slot_data, np.ndarray) else slot_data

    plt.figure(figsize=(7, 6))

    id_to_color, id_to_name = _load_material_library()

    unique_ids = sorted([int(uid) for uid in np.unique(grid)])
    colors_list = [id_to_color.get(uid, '#888888') for uid in unique_ids]
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    mapped_grid = np.vectorize(id_to_idx.get)(grid)

    cmap = mcolors.ListedColormap(colors_list)

    half_width_m = (nx * dx) / 2.0
    half_height_m = (ny * dx) / 2.0
    grid_center_vox_x = (nx - 1.0) / 2.0
    grid_center_vox_y = (ny - 1.0) / 2.0

    if use_meters:
        extent = [-0.5 * dx, (nx - 0.5) * dx, -0.5 * dx, (ny - 0.5) * dx]
        plt.imshow(mapped_grid.T, cmap=cmap, origin='lower', interpolation='nearest', extent=extent)
        plt.xlabel("X Axis (meters)")
        plt.ylabel("Y Axis (meters)")
    else:
        plt.imshow(mapped_grid.T, cmap=cmap, origin='lower', interpolation='nearest')
        plt.xlabel("X Axis (voxels)")
        plt.ylabel("Y Axis (voxels)")

    if sources:
        src = sources[0]
        if use_meters:
            sx = src['coords'][0] + half_width_m
            sy = src['coords'][1] + half_height_m
        else:
            sx = src['coords'][0] / dx + grid_center_vox_x
            sy = src['coords'][1] / dx + grid_center_vox_y
        plt.scatter(sx, sy, color='red', marker='*', s=128, label='Source', zorder=5)

    if receivers:
        rec_x, rec_y = [], []
        for r in receivers:
            if use_meters:
                rx = r['coords'][0] + half_width_m
                ry = r['coords'][1] + half_height_m
            else:
                rx = r['coords'][0] / dx + grid_center_vox_x
                ry = r['coords'][1] / dx + grid_center_vox_y
            rec_x.append(rx)
            rec_y.append(ry)
        plt.scatter(rec_x, rec_y, color='blue', marker='o', s=64, label='Receivers', zorder=4)
        for i, (x, y) in enumerate(zip(rec_x, rec_y)):
            plt.annotate(str(i), (x, y), color='white', fontweight='bold', fontsize=6, ha='center', va='center', zorder=6)

    if foam_slots:
        slot_x, slot_y = [], []
        for s in foam_slots:
            gx = s['grid_x']
            gy = s['grid_y']
            sx = gx * dx if use_meters else gx
            sy = gy * dx if use_meters else gy
            slot_x.append(sx)
            slot_y.append(sy)
        plt.scatter(slot_x, slot_y, color='green', marker='s', s=40, label='Foam Slots', zorder=3)

    plt.title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    for uid in unique_ids:
        mat_name = id_to_name.get(uid, f"Material {uid}")
        hex_color = id_to_color.get(uid, '#888888')
        patch = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=hex_color, markersize=8, label=f"{uid}: {mat_name}")
        handles.append(patch)

    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Generated room image: {save_path}")

    if show:
        plt.show()

    plt.close()


def visualize_simulation_grid(data, title="Simulation Grid Preview", save_path=None, show=False, use_meters: bool = False, chosen_foams=None):
    grid = data['material_core']
    dx = data['dx'].item() if isinstance(data['dx'], np.ndarray) else data['dx']
    nx, ny = grid.shape

    if chosen_foams is not None and 'foam_slots' in data:
        grid = np.array(grid, copy=True)
        foam_slots_data = data['foam_slots']
        for slot_idx in chosen_foams:
            if slot_idx < len(foam_slots_data):
                for cx, cy in foam_slots_data[slot_idx]['cells']:
                    grid[cx, cy] = 16

    src_data = data['sources'] if 'sources' in data else []
    sources = src_data.tolist() if isinstance(src_data, np.ndarray) else src_data

    recv_data = data['receivers'] if 'receivers' in data else []
    receivers = recv_data.tolist() if isinstance(recv_data, np.ndarray) else recv_data

    plt.figure(figsize=(7, 6))

    id_to_color, id_to_name = _load_material_library()

    unique_ids = sorted([int(uid) for uid in np.unique(grid)])
    colors_list = [id_to_color.get(uid, '#888888') for uid in unique_ids]
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    mapped_grid = np.vectorize(id_to_idx.get)(grid)

    cmap = mcolors.ListedColormap(colors_list)

    half_width_m = (nx * dx) / 2.0
    half_height_m = (ny * dx) / 2.0
    grid_center_vox_x = (nx - 1.0) / 2.0
    grid_center_vox_y = (ny - 1.0) / 2.0

    if use_meters:
        extent = [-0.5 * dx, (nx - 0.5) * dx, -0.5 * dx, (ny - 0.5) * dx]
        plt.imshow(mapped_grid.T, cmap=cmap, origin='lower', interpolation='nearest', extent=extent)
        plt.xlabel("X Axis (meters)")
        plt.ylabel("Y Axis (meters)")
    else:
        plt.imshow(mapped_grid.T, cmap=cmap, origin='lower', interpolation='nearest')
        plt.xlabel("X Axis (voxels)")
        plt.ylabel("Y Axis (voxels)")

    if sources:
        src = sources[0]
        if use_meters:
            sx = src['coords'][0] + half_width_m
            sy = src['coords'][1] + half_height_m
        else:
            sx = src['coords'][0] / dx + grid_center_vox_x
            sy = src['coords'][1] / dx + grid_center_vox_y
        plt.scatter(sx, sy, color='red', marker='*', s=128, label='Source', zorder=5)

    if receivers:
        rec_x, rec_y = [], []
        for r in receivers:
            if use_meters:
                rx = r['coords'][0] + half_width_m
                ry = r['coords'][1] + half_height_m
            else:
                rx = r['coords'][0] / dx + grid_center_vox_x
                ry = r['coords'][1] / dx + grid_center_vox_y
            rec_x.append(rx)
            rec_y.append(ry)
        plt.scatter(rec_x, rec_y, color='blue', marker='o', s=64, label='Receivers', zorder=4)
        for i, (x, y) in enumerate(zip(rec_x, rec_y)):
            plt.annotate(str(i), (x, y), color='white', fontweight='bold', fontsize=6, ha='center', va='center', zorder=6)

    plt.title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    for uid in unique_ids:
        mat_name = id_to_name.get(uid, f"Material {uid}")
        hex_color = id_to_color.get(uid, '#888888')
        patch = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=hex_color, markersize=8, label=f"{uid}: {mat_name}")
        handles.append(patch)

    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def visualize_and_save_rooms():
    print("Visualizing rooms...")

    if not os.path.exists(RASTERIZED_FOLDER):
        print(f"Folder {RASTERIZED_FOLDER} does not exist. Generate rooms first.")
        return

    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    npz_files = [f for f in os.listdir(RASTERIZED_FOLDER) if f.endswith('.npz')]

    if not npz_files:
        print(f"No NPZ files in {RASTERIZED_FOLDER}.")
        return

    for filename in npz_files:
        filepath = os.path.join(RASTERIZED_FOLDER, filename)
        data = np.load(filepath, allow_pickle=True)

        base_name = os.path.splitext(filename)[0]

        img_path = os.path.join(IMAGES_FOLDER, f"{base_name}.png")
        visualize_room_data(data, title=filename, save_path=img_path, show=False, use_meters=False)

        img_path = os.path.join(IMAGES_FOLDER, f"{base_name}_m.png")
        visualize_room_data(data, title=filename, save_path=img_path, show=False, use_meters=True)

    print(f"Visualized {len(npz_files)} rooms.")


if __name__ == "__main__":
    visualize_and_save_rooms()