import random
import os
import json
import math

ROOM_WIDTH_MIN = 2.0
ROOM_WIDTH_MAX = 8.0
ROOM_LENGTH_MIN = 2.0
ROOM_LENGTH_MAX = 8.0

OUTPUT_FOLDER = 'generated_rooms'

class Room:
    def __init__(self, room_shape, room_material_id, width_m, height_m, room_vertices, speaker, microphones, foam_slots):
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

def is_point_in_l_shape(x, y, w1, h1, w2, h2):
    in_bottom_block = (0.0 <= x <= w2) and (0.0 <= y <= h2)
    in_top_block = (0.0 <= x <= w1) and (h2 <= y <= h1)
    return in_bottom_block or in_top_block

def generate_l_shape_room(material_id=1, foam_size=0.25, num_mics=3):
    total_width = round(random.uniform(ROOM_WIDTH_MIN, ROOM_WIDTH_MAX), 1)
    total_height = round(random.uniform(ROOM_LENGTH_MIN, ROOM_LENGTH_MAX), 1)

    w1 = round(total_width * random.uniform(0.25, 0.75), 1)
    w2 = total_width

    h1 = total_height
    h2 = round(total_height * random.uniform(0.25, 0.75), 1)

    vertices = [
        [0.0, 0.0],
        [w2, 0.0],
        [w2, h2],
        [w1, h2],
        [w1, h1],
        [0.0, h1]
    ]

    max_width = max(w1, w2)
    max_height = max(h1, h2)

    # Speaker position
    speaker = {}
    while True:
        sx = round(random.uniform(0.3, max_width - 0.3), 2)
        sy = round(random.uniform(0.3, max_height - 0.3), 2)
        if is_point_in_l_shape(sx, sy, w1, h1, w2, h2):
            speaker = {"x_m": sx, "y_m": sy}
            break

    # Microphones
    microphones = []
    while len(microphones) < num_mics:
        mx = round(random.uniform(0.3, max_width - 0.3), 2)
        my = round(random.uniform(0.3, max_height - 0.3), 2)
        if is_point_in_l_shape(mx, my, w1, h1, w2, h2):
            if [mx, my] != [speaker["x_m"], speaker["y_m"]] and [mx, my] not in microphones:
                microphones.append({"x_m": mx, "y_m": my})

    # Foam Slots
    segments = [
        {"p1": [0.0, 0.0], "p2": [w2, 0.0]},
        {"p1": [w2, 0.0], "p2": [w2, h2]},
        {"p1": [w2, h2], "p2": [w1, h2]},
        {"p1": [w1, h2], "p2": [w1, h1]},
        {"p1": [w1, h1], "p2": [0.0, h1]},
        {"p1": [0.0, h1], "p2": [0.0, 0.0]}
    ]

    foam_slots = []
    #corner_threshold = 0.5

    for segment in segments:
        p1 = segment["p1"]
        p2 = segment["p2"]

        segment_length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        num_slots_per_segment = int(segment_length // foam_size)

        if num_slots_per_segment > 0:
            offset = (segment_length - (num_slots_per_segment * foam_size))/2.0

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

                dist_to_speaker = round(math.hypot(slot_x - speaker["x_m"], slot_y - speaker["y_m"]), 3)

                foam_slots.append({
                    "x_m": slot_x,
                    "y_m": slot_y,
                    "distance_to_corner": distance_to_corner,
                    "distance_to_speaker": dist_to_speaker,
                    "orientation": orientation_str
                })

    return Room(
        room_shape="L-shape",
        room_material_id=material_id,
        width_m=max_width,
        height_m=max_height,
        room_vertices=vertices,
        speaker=speaker,
        microphones=microphones,
        foam_slots=foam_slots
    )


def save_room_to_json(room_obj, filename):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(room_obj.to_dict(), f, indent=4)

    print(f"Saved room: {filepath}")


if __name__ == "__main__":
    n = int(input("How many rooms to generate? (default 10) ") or 10)
    foam_size = int(input("Size of a single acoustic foam [cm] (default 25) ") or 25)
    foam_size_m = float(foam_size)/100.0
    mics_count = int(input("Number of microphones per room (default 5) ") or 3)

    for i in range(n):
        my_room = generate_l_shape_room(material_id=1, num_mics=mics_count, foam_size=foam_size_m)
        save_room_to_json(my_room, filename=f"room_{i}.json")
    print(f"Generated {n} rooms.")