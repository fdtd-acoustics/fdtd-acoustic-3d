from pathlib import Path

import numpy as np
import trimesh

from visualization.config import SCENES_MODELS_DIR, SCENES_OUT_DIR

file_name = 'test_room4.obj'

# TODO: Test czy materialy tez sa
obj_file_path = SCENES_MODELS_DIR / file_name

NX, NY, NZ = 185,185,185
DX = 0.05 # krok przestrzenny dx w m

MATERIALS_IDS = {
    "air": 0,
    "wall": 1,
    "metal": 2
}
DEFAULT_MATERIAL = 1

space_matrix = np.zeros((NX,NY,NZ), dtype=np.int8)

core_x, core_y, core_z = NX // 2, NY // 2, NZ // 2

def get_material_id(obj_name):
    obj_name_lower = obj_name.lower()
    for material_name, id in MATERIALS_IDS.items():
        if material_name in obj_name_lower:
            return id
    else:
        return DEFAULT_MATERIAL

def get_material_name(obj_name):
    obj_name_lower = obj_name.lower()
    for material_name, id in MATERIALS_IDS.items():
        if material_name in obj_name_lower:
            return material_name
    else:
        return DEFAULT_MATERIAL

def save_to_file():
    save_file_name = SCENES_OUT_DIR / Path(obj_file_path).with_suffix(".npz").name
    np.savez(save_file_name, material_core=space_matrix)
    print("Saved scene to file: ", save_file_name)

def fill_normal_objects(voxelized_object):
    filled_object = voxelized_object.fill()
    if len(filled_object.points) > 0:
        print(f" -> fill() OK: {len(filled_object.points)} voxels")
        return filled_object
    else:
        print(f" -> fill() ERROR: could not fill object. Returning unfilled object.")
        return voxelized_object


def voxelize_geometry(geom, geom_name):

    #wokselizacja obiektu
    voxelized_object = geom.voxelized(pitch=DX)

    #wypelniania jesli nie jest sciana
    is_wall = "wall" in geom_name.lower()
    if not is_wall:
        voxelized_object = fill_normal_objects(voxelized_object)


    points = voxelized_object.points
    #zamiana metrow na indexy siatki
    indices = np.round(points / DX).astype(int) + np.array([core_x, core_y, core_z])

    #walidacja czy punkty sa na siatce
    valid_mask = np.all(
        (indices >= 0) & (indices < [NX, NY, NZ]),
        axis=1
    )

    valid_indices = indices[valid_mask]

    return valid_indices

def load_scene():
    scene = trimesh.load(obj_file_path, force='scene', process=True)

    print(f"Number of objects detected in scene: {len(scene.geometry.items())}")

    for geom_name, geom in scene.geometry.items():
        print(f"Processing ==> {geom_name}")
        print(f" -> Is it watertight? {geom.is_watertight}")

        indices = voxelize_geometry(geom, geom_name)

        if len(indices) == 0:
            print(f" -> Object is outside of domain - skipping")
            continue
        material_id = get_material_id(geom_name)

        print(f" -> Material ={get_material_name(geom_name)}, found voxels: {len(indices)}")

        #wpisywanie do glownej macierzy
        x,y,z = indices.T
        values = space_matrix[x,y,z]
        space_matrix[x,y,z] = np.maximum(values, material_id)

    save_to_file()

if __name__ == "__main__":
    load_scene()









