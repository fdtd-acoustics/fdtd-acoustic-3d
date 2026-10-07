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

    grid = np.ones((grid_h, grid_w), dtype=np.int8) * material_id

    if room_shape == "L-shape":
        w2 = vertices[1][0]
        h2 = vertices[2][1]
        w1 = vertices[3][0]
        h1 = vertices[4][1]

        x_idx = int(round(0.0 / dx)) + 1
        y_idx = int(round(0.0 / dx)) + 1

        w1_cells = int(round(w1 / dx))
        h1_cells = int(round(h1 / dx))
        grid[y_idx: y_idx + h1_cells, x_idx: x_idx + w1_cells] = 0

        w2_cells = int(round(w2 / dx))
        h2_cells = int(round(h2 / dx))
        grid[y_idx: y_idx + h2_cells, x_idx: x_idx + w2_cells] = 0  # 0 to AIR

    else:
        raise NotImplementedError(f"Room Shape: '{room_shape}' not implemented.")

    speaker_grid = {
        "grid_x": int(round(data["speaker"]["x_m"] / dx)) + 1,
        "grid_y": int(round(data["speaker"]["y_m"] / dx)) + 1
    }

    microphones_grid = []
    for mic in data["microphones"]:
        microphones_grid.append({
            "grid_x": int(round(mic["x_m"] / dx)) + 1,
            "grid_y": int(round(mic["y_m"] / dx)) + 1
        })

    foam_slots_grid = []
    for slot in data["foam_slots"]:
        foam_slots_grid.append({
            "grid_x": int(round(slot["x_m"] / dx)) + 1,
            "grid_y": int(round(slot["y_m"] / dx)) + 1,
            "is_near_corner": slot["is_near_corner"],
            "distance_to_speaker": slot["distance_to_speaker"]
        })

    return grid, speaker_grid, microphones_grid, foam_slots_grid


def rasterize_all_rooms(dx, dt):
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

        grid, speaker_grid, microphones_grid, foam_slots_grid = rasterize_room(json_path, dx=dx)

        base_name = os.path.splitext(filename)[0]
        output_npz_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.npz")

        np.savez_compressed(
            output_npz_path,
            grid=grid,
            dx=dx,
            dt=dt,
            speaker=speaker_grid,
            microphones=microphones_grid,
            foam_slots=foam_slots_grid
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
    max_freq = int(input("Max Frequency [Hz]: "))

    dx = compute_dx(max_freq)
    dt = compute_dt(dx)

    rasterize_all_rooms(dx, dt)