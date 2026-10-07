import random
import os
import json

ROOM_WIDTH_MIN = 3.0
ROOM_WIDTH_MAX = 7.0
ROOM_LENGTH_MIN = 3.0
ROOM_LENGTH_MAX = 7.0

OUTPUT_FOLDER = 'generated_rooms'

class Room:
    def __init__(self, room_shape, room_material_id, room_material, width_m, height_m, room_vertices):
        self.room_shape = room_shape
        self.room_material_id = room_material_id
        self.room_material = room_material
        self.width_m = width_m
        self.height_m = height_m
        self.room_vertices = room_vertices

    def to_dict(self):
        return {
            "room_shape": self.room_shape,
            "room_material_id": self.room_material_id,
            "room_material": self.room_material,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "room_vertices": self.room_vertices
        }

def generate_l_shape_room(material_id=1, material_name="smooth_concrete"):
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

    return Room(
        room_shape="L-shape",
        room_material_id=material_id,
        room_material=material_name,
        width_m=max_width,
        height_m=max_height,
        room_vertices=vertices
    )


def save_room_to_json(room_obj, filename):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(room_obj.to_dict(), f, indent=4)

    print(f"Saved room: {filepath}")


if __name__ == "__main__":
    n = int(input("How many rooms to generate? "))
    for i in range(n):
        my_room = generate_l_shape_room(material_id=1, material_name="smooth_concrete")
        save_room_to_json(my_room, filename=f"room_{i}.json")
    print(f"Generated {n} rooms.")