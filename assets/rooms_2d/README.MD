# 2D Rooms Data Structure

## 1. Vector Format (`generated_rooms/room_X.json`)
`.json` files contain the metric definition of the environment (dimensions in meters).

### JSON Structure:
- `room_shape` (string) - Room Shape (e.g.,`"L-shape"`)
- `room_material_id` (int) - Wall material ID (e.g., `1` - concrete).
- `width_m` / `height_m` (float) - Maximum width/height of the entire area in meters.
- `room_vertices` (list of lists of floats) - Room wall vertices. Coordinates `[x, y]` in meters (origin `[0.0, 0.0]`).
- `speaker` (dict) - Dictionary containing the sound source position: `{"x_m": float, "y_m": float}`.
- `microphones` (list of dicts) - List of microphones. Each element is `{"x_m": float, "y_m": float}`.
- `foam_slots` (list of dicts) - List of calculated, mounting slots for acoustic panels. 
  - `"x_m": float`, `"y_m": float` - Coordinates of the slot's center in meters.
  - `"distance_to_corner"` (float) - Distance from the slot to the nearest corner [m].
  - `"distance_to_speaker"` (float) - Distance from the slot to the speaker [m].
  - `"orientation"` (string) - `"horizontal"` or `"vertical"`.

---

## 2. FDTD Format (`rasterized_rooms/room_X.npz`)
`.npz` files are created by passing JSONs through `room_2d_rasterizer.py`. \
Vector dimensions are converted into cell indices based on the grid step parameter `dx`.

### NPZ Structure (accessible via `data['key']`):
- `material_core` (`ndarray`, dtype `int8`) - 2D simulation matrix. `0` means air, `1` means concrete.
- `dx` (float) - The spatial grid step in meters (e.g., `0.05`).
- `dt` (float) - The time step in seconds, calculated via Courant condition.
- `Nx`, `Ny`, `Nz` (int) - Number of cells in X, Y, and Z axes (`Nz` is `1` for 2D).
- `pml_thick` (int) - Thickness of the Perfectly Matched Layer (in cells).
- `alpha_max` (float) - Maximum attenuation for PML.
- `sound_speed` (float) - Speed of sound used (e.g., `343.0` m/s).
- `nodes_per_wavelength` (int) - Target nodes per wavelength (e.g., `10`).
- `safety_factor` (float) - Courant condition safety factor (e.g., `0.95`).
- `record_time` (float) - Total simulation recording time in seconds.
- `obj_filepath` (string) - Path to the original `.json` room file.
- `sources` (list of dicts) - Array containing sound sources (speakers).
  - `"name"` (string) - Source identifier.
  - `"position"` (tuple of floats) - A 3D position in meters `(x_m, y_m, 0.0)`.
  - `"time"` (float) - Start time of the signal.
  - `"waveform"` (dict) - Signal properties (`"type"`, `"freq"`, `"amp"`, `"vol"`).
  - `"type"`, `"freq"` - Additional redundant flat keys for headless mode setup.
- `receivers` (list of dicts) - Array containing microphones (listeners).
  - `"name"` (string) - Receiver identifier.
  - `"position"` (tuple of floats) - A 3D position in meters `(x_m, y_m, 0.0)`.
- `foam_slots` (list of dicts) - Array containing acoustic foam mounting slots.
  - `"grid_x"`, `"grid_y"` (int) - Matrix column and row indices on the `material_core` grid.
  - `"distance_to_corner"` (float) - Distance from the slot to the nearest corner [m].
  - `"distance_to_speaker"` (float) - Distance from the slot to the speaker [m].
  - `"orientation"` (string) - Wall orientation (`"horizontal"` or `"vertical"`).
  - `"normal_vector"` (list of floats) - Normal vector of the wall pointing inside the room `[nx, ny]`.
  
---
