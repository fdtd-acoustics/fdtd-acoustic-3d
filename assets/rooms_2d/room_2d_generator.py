import random
import os
import json
import math

MIN_MIC_SPEAKER_DISTANCE = 1.0      # [m]
MIN_MIC_MIC_DISTANCE = 0.8          # [m]
MIN_MIC_FOAM_DISTANCE = 0.4         # [m]
MIN_SPEAKER_FOAM_DISTANCE = 0.4     # [m]

WALL_MARGIN = 0.3  # [m]
MIC_MAX_ATTEMPTS = 10000

OUTPUT_FOLDER = 'generated_rooms'


class Room:
    def __init__(self, room_shape, room_material_id, width_m, height_m, room_vertices, speaker, microphones,
                 foam_slots):
        self.room_shape = room_shape
        self.room_material_id = room_material_id
        self.width_m = width_m
        self.height_m = height_m
        self.room_vertices = room_vertices
        self.speaker = speaker
        self.microphones = microphones
        self.foam_slots = foam_slots

    def to_dict(self):
        return {
            "room_shape": self.room_shape,
            "room_material_id": self.room_material_id,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "room_vertices": self.room_vertices,
            "speaker": self.speaker,
            "microphones": self.microphones,
            "foam_slots": self.foam_slots
        }


def is_point_in_shape(x, y, shape, params, margin=0.0):
    if shape == "L-shape":
        w1, h1, w2, h2 = params
        in_bottom = (margin <= x <= w2 - margin) and (margin <= y <= h2 - margin)
        in_left = (margin <= x <= w1 - margin) and (margin <= y <= h1 - margin)
        return in_bottom or in_left

    elif shape == "C-shape":
        W, H, cut_w, cut_h, cut_y = params
        in_main = (margin <= x <= W - margin) and (margin <= y <= H - margin)
        in_cut = (W - cut_w + margin <= x <= W - margin) and (cut_y + margin <= y <= cut_y + cut_h - margin)
        return in_main and not in_cut

    elif shape == "T-shape":
        W, H, stem_w, stem_h, bar_h = params
        stem_x_min = (W - stem_w) / 2.0
        in_stem = (stem_x_min + margin <= x <= stem_x_min + stem_w - margin) and (margin <= y <= H - bar_h - margin)
        in_bar = (margin <= x <= W - margin) and (H - bar_h + margin <= y <= H - margin)
        return in_stem or in_bar

    elif shape == "S-shape":
        W, H, w_block, h_half, overlap = params
        in_bottom = (margin <= x <= w_block - margin) and (margin <= y <= h_half - margin)
        in_top = (W - w_block + margin <= x <= W - margin) and (h_half + margin <= y <= H - margin)
        in_overlap = (w_block - overlap + margin <= x <= W - w_block + overlap - margin) and (
                    h_half - 0.5 + margin <= y <= h_half + 0.5 - margin)
        return in_bottom or in_top or in_overlap

    return False


def generate_room(material_id=5, foam_size=0.25, num_mics=3, seed=None, shape_type=None, return_dict=True,
                  room_width_min=6, room_width_max=12,
                  room_length_min=6, room_length_max=12):
    rng = random.Random(seed)

    if shape_type is None:
        shape_type = rng.choice(["L-shape", "C-shape", "S-shape", "T-shape"])

    total_width = round(rng.uniform(room_width_min, room_width_max), 1)
    total_height = round(rng.uniform(room_length_min, room_length_max), 1)

    params = None
    vertices = []

    if shape_type == "L-shape":
        w1 = round(total_width * rng.uniform(0.4, 0.7), 1)
        w2 = total_width
        h1 = total_height
        h2 = round(total_height * rng.uniform(0.4, 0.7), 1)
        params = (w1, h1, w2, h2)
        vertices = [[0.0, 0.0], [w2, 0.0], [w2, h2], [w1, h2], [w1, h1], [0.0, h1]]

    elif shape_type == "C-shape":
        W, H = total_width, total_height
        cut_w = round(W * rng.uniform(0.3, 0.5), 1)
        cut_h = round(H * rng.uniform(0.4, 0.6), 1)
        cut_y = round((H - cut_h) * rng.uniform(0.2, 0.5), 1)
        params = (W, H, cut_w, cut_h, cut_y)
        vertices = [
            [0.0, 0.0], [W, 0.0], [W, cut_y], [W - cut_w, cut_y],
            [W - cut_w, cut_y + cut_h], [W, cut_y + cut_h], [W, H], [0.0, H]
        ]

    elif shape_type == "T-shape":
        W, H = total_width, total_height
        stem_w = round(W * rng.uniform(0.3, 0.5), 1)
        bar_h = round(H * rng.uniform(0.3, 0.4), 1)
        stem_x1 = round((W - stem_w) / 2.0, 1)
        stem_x2 = round(stem_x1 + stem_w, 1)
        stem_h = H - bar_h
        params = (W, H, stem_w, stem_h, bar_h)
        vertices = [
            [stem_x1, 0.0], [stem_x2, 0.0], [stem_x2, stem_h], [W, stem_h],
            [W, H], [0.0, H], [0.0, stem_h], [stem_x1, stem_h]
        ]

    elif shape_type == "S-shape":
        W, H = total_width, total_height
        w_block = round(W * rng.uniform(0.55, 0.75), 1)
        h_half = round(H / 2.0, 1)
        overlap = round(w_block * rng.uniform(0.2, 0.4), 1)

        params = (W, H, w_block, h_half, overlap)

        vertices = [
            [0.0, 0.0],
            [w_block, 0.0],
            [w_block, h_half],
            [W, h_half],
            [W, H],
            [W - w_block, H],
            [W - w_block, h_half],
            [0.0, h_half]
        ]

    max_width = total_width
    max_height = total_height

    foam_slots = []
    segments = []
    if shape_type == "S-shape":
        top_left_x = W - w_block
        segments = [
            {"p1": [0.0, 0.0], "p2": [w_block, 0.0]},
            {"p1": [w_block, 0.0], "p2": [w_block, h_half]},
            {"p1": [w_block, h_half], "p2": [W, h_half]},
            {"p1": [W, h_half], "p2": [W, H]},
            {"p1": [W, H], "p2": [top_left_x, H]},
            {"p1": [top_left_x, H], "p2": [top_left_x, h_half]},
            {"p1": [top_left_x, h_half], "p2": [0.0, h_half]},
            {"p1": [0.0, h_half], "p2": [0.0, 0.0]}
        ]
    else:
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            segments.append({"p1": p1, "p2": p2})

    for segment in segments:
        p1 = segment["p1"]
        p2 = segment["p2"]
        segment_length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        num_slots_per_segment = int(segment_length // foam_size)

        if num_slots_per_segment > 0:
            offset = (segment_length - (num_slots_per_segment * foam_size)) / 2.0
            dx = (p2[0] - p1[0]) / segment_length
            dy = (p2[1] - p1[1]) / segment_length
            nx = -dy
            ny = dx
            orientation_str = "horizontal" if abs(ny) > 0.5 else "vertical"

            for j in range(num_slots_per_segment):
                start_dist = offset + j * foam_size
                center_dist = start_dist + (foam_size / 2.0)
                slot_x = round(p1[0] + dx * center_dist, 3)
                slot_y = round(p1[1] + dy * center_dist, 3)

                dist_to_p1 = math.hypot(slot_x - p1[0], slot_y - p1[1])
                dist_to_p2 = math.hypot(slot_x - p2[0], slot_y - p2[1])
                distance_to_corner = round(min(dist_to_p1, dist_to_p2), 3)

                foam_slots.append({
                    "x_m": slot_x,
                    "y_m": slot_y,
                    "distance_to_corner": distance_to_corner,
                    "distance_to_speaker": 0.0,
                    "orientation": orientation_str,
                    "normal_vector": [nx, ny],
                    "wall_direction": [dx, dy]
                })

    speaker = {}
    for attempt in range(MIC_MAX_ATTEMPTS + 1):
        if attempt == MIC_MAX_ATTEMPTS:
            raise ValueError(f"Couldn't place speaker in {shape_type} due to distance constraints (seed={seed})")
        sx = round(rng.uniform(WALL_MARGIN, max_width - WALL_MARGIN), 2)
        sy = round(rng.uniform(WALL_MARGIN, max_height - WALL_MARGIN), 2)

        if not is_point_in_shape(sx, sy, shape_type, params, margin=WALL_MARGIN):
            continue

        too_close = False
        for slot in foam_slots:
            if math.hypot(sx - slot["x_m"], sy - slot["y_m"]) < MIN_SPEAKER_FOAM_DISTANCE:
                too_close = True
                break
        if too_close:
            continue

        speaker = {"x_m": sx, "y_m": sy}
        break

    for slot in foam_slots:
        slot["distance_to_speaker"] = round(math.hypot(slot["x_m"] - speaker["x_m"], slot["y_m"] - speaker["y_m"]), 3)

    microphones = []
    attempts = 0
    while len(microphones) < num_mics:
        attempts += 1
        if attempts > MIC_MAX_ATTEMPTS:
            raise ValueError(
                f"Couldn't place {num_mics} microphones in {shape_type} due to distance constraints (seed={seed})")

        mx = round(rng.uniform(WALL_MARGIN, max_width - WALL_MARGIN), 2)
        my = round(rng.uniform(WALL_MARGIN, max_height - WALL_MARGIN), 2)

        if not is_point_in_shape(mx, my, shape_type, params, margin=WALL_MARGIN):
            continue

        if math.hypot(mx - speaker["x_m"], my - speaker["y_m"]) < MIN_MIC_SPEAKER_DISTANCE:
            continue

        too_close_to_mic = False
        for mic in microphones:
            if math.hypot(mx - mic["x_m"], my - mic["y_m"]) < MIN_MIC_MIC_DISTANCE:
                too_close_to_mic = True
                break
        if too_close_to_mic:
            continue

        too_close_to_foam = False
        for slot in foam_slots:
            if math.hypot(mx - slot["x_m"], my - slot["y_m"]) < MIN_MIC_FOAM_DISTANCE:
                too_close_to_foam = True
                break
        if too_close_to_foam:
            continue

        candidate = {"x_m": mx, "y_m": my}
        if candidate not in microphones:
            microphones.append(candidate)

    room_obj = Room(
        room_shape=shape_type,
        room_material_id=material_id,
        width_m=max_width,
        height_m=max_height,
        room_vertices=vertices,
        speaker=speaker,
        microphones=microphones,
        foam_slots=foam_slots
    )

    return room_obj.to_dict() if return_dict else room_obj


def save_room_to_json(room_data, filename):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    data_to_save = room_data if isinstance(room_data, dict) else room_data.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)
    print(f"Saved room: {filepath}")


if __name__ == "__main__":
    n = int(input("How many rooms to generate? (default 10) ") or 10)
    foam_size = int(input("Size of a single acoustic foam [cm] (default 25) ") or 25)
    foam_size_m = float(foam_size) / 100.0
    mics_count = int(input("Number of microphones per room (default 3) ") or 3)
    seed_start = int(input("Start seed (default 0) ") or 0)

    for i in range(seed_start, seed_start + n, 1):
        room_dict = generate_room(material_id=5, foam_size=foam_size_m, num_mics=mics_count, seed=i, shape_type="L-shape",
                                  room_width_min=6, room_width_max=12,
                                  room_length_min=6, room_length_max=12)
        save_room_to_json(room_dict, filename=f"room_{i}.json")
    print(f"Generated {n} rooms.")