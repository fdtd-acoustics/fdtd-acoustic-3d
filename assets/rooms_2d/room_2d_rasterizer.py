import os
import json
import numpy as np
import math

INPUT_FOLDER = 'generated_rooms'
OUTPUT_FOLDER = 'rasterized_rooms'
WALL_WIDTH = 2
SOUND_SPEED = 343.0
NODES_PER_WAVELENGTH = 10
SAFETY_FACTOR = 0.95

PML_THICK = 10
ALPHA_MAX = 0.15
RECORD_TIME = 0.5 # tymczasowo, trzeba bedzie zrobic to z zewnatrz mysle

def to_centered_coords(x_m, y_m, nx, ny, dx):
    voxel_x = int(round(x_m / dx))
    voxel_y = int(round(y_m / dx))

    grid_center_x = (nx - 1.0) / 2.0
    grid_center_y = (ny - 1.0) / 2.0

    centered_x_m = (voxel_x - grid_center_x) * dx
    centered_y_m = (voxel_y - grid_center_y) * dx

    return round(float(centered_x_m), 6), round(float(centered_y_m), 6)

def rasterize_room(json_filepath, dx):
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    room_shape = data["room_shape"]
    material_id = data["room_material_id"]
    max_width_m = data["width_m"]
    max_height_m = data["height_m"]
    vertices = data["room_vertices"]

    grid_w = int(np.ceil(max_width_m / dx)) + WALL_WIDTH
    grid_h = int(np.ceil(max_height_m / dx)) + WALL_WIDTH

    grid = np.ones((grid_w, grid_h), dtype=np.int8) * material_id

    if room_shape == "L-shape":
        w2 = vertices[1][0]
        h2 = vertices[2][1]
        w1 = vertices[3][0]
        h1 = vertices[4][1]

        x_idx = int(round(0.0 / dx)) + 1
        y_idx = int(round(0.0 / dx)) + 1

        w1_cells = int(round(w1 / dx))
        h1_cells = int(round(h1 / dx))
        grid[x_idx: x_idx + w1_cells, y_idx: y_idx + h1_cells] = 0

        w2_cells = int(round(w2 / dx))
        h2_cells = int(round(h2 / dx))
        grid[x_idx: x_idx + w2_cells, y_idx: y_idx + h2_cells] = 0  # 0 to AIR

    else:
        raise NotImplementedError(f"Room Shape: '{room_shape}' not implemented.")

    source_data_json = data["speaker"]
    microphones_json = data["microphones"]

    foam_slots_grid = []
    for slot in data["foam_slots"]:
        foam_slots_grid.append({
            "grid_x": int(round(slot["x_m"] / dx)) + 1,
            "grid_y": int(round(slot["y_m"] / dx)) + 1,
            "distance_to_corner": slot["distance_to_corner"],
            "distance_to_speaker": slot["distance_to_speaker"],
            "orientation": slot["orientation"]
        })

    return grid, source_data_json, microphones_json, foam_slots_grid


def rasterize_all_rooms(dx, dt, max_freq):
    if not os.path.exists(INPUT_FOLDER):
        print(f"Folder {INPUT_FOLDER} does not exist. Generate rooms first.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    json_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files in {INPUT_FOLDER}.")
        return

    for filename in json_files:
        json_path = os.path.join(INPUT_FOLDER, filename)

        grid, source_json, mics_json, foam_slots_grid = rasterize_room(json_path, dx=dx)

        nx, ny = grid.shape
        src_cx, src_cy = to_centered_coords(source_json["x_m"], source_json["y_m"], nx, ny, dx)

        sources_data = [
            {
                "name": "source_main",
                "coords": (src_cx, src_cy),
                "time": 0.5,
                "waveform": {
                    "type": "Gauss",
                    "freq": float(max_freq),
                    "amp": 100.0,
                    "vol": 1.0
                },
                "type": "Gauss",  # tymczasowo, trzeba zmienic headless mode
                "freq": float(max_freq)   # tymczasowo, trzeba zmienic headless mode
            }
        ]

        receivers_data = [
            {
                "name": f"mic_{idx}",
                "coords": list(to_centered_coords(mic["x_m"], mic["y_m"], nx, ny, dx))
            }
            for idx, mic in enumerate(mics_json)
        ]

        nx , ny = grid.shape

        base_name = os.path.splitext(filename)[0]
        output_npz_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.npz")

        np.savez_compressed(
            output_npz_path,
            material_core=grid,
            dx=dx,
            dt=dt,
            Nx=nx,
            Ny=ny,
            Nz=1,
            pml_thick=PML_THICK,
            alpha_max=ALPHA_MAX,
            sound_speed=SOUND_SPEED,
            nodes_per_wavelength=NODES_PER_WAVELENGTH,
            safety_factor=SAFETY_FACTOR,
            record_time=RECORD_TIME,
            obj_filepath=json_path,
            sources=np.array(sources_data, dtype=object),
            receivers=np.array(receivers_data, dtype=object),
            foam_slots=np.array(foam_slots_grid, dtype=object)
        )

        print(f"Rasterized {filename} -> {base_name}.npz (Grid Shape: {grid.shape})")

    print(f"Rasterized {len(json_files)} rooms.")

# Copied function and consts from simulation/simulation_builder.py
def compute_dx(max_freq):
    wavelength = SOUND_SPEED / max_freq
    dx = wavelength / NODES_PER_WAVELENGTH
    return dx

def compute_dt(dx):
    courant_limit = 1.0 / math.sqrt(2)
    return (dx / SOUND_SPEED) * courant_limit * SAFETY_FACTOR

if __name__ == "__main__":
    max_freq = int(input("Max Frequency [Hz] (default 1000) ") or 1000)

    dx = compute_dx(max_freq)
    dt = compute_dt(dx)

    rasterize_all_rooms(dx, dt, max_freq)