from pathlib import Path

import numpy as np
import trimesh
import config


#Todo: refactor tej klasy
class Voxelizer:
    def __init__(self, NX, NY, NZ, dx, file_path):
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.DX = dx
        self.file_path = file_path
        self.core_x = NX //2
        self.core_y = NY //2
        self.core_z = NZ // 2
        self.space_matrix = np.zeros((NX,NY,NZ), dtype=np.int8)

# TODO: Test czy materialy tez sa

    def get_material_id(self, obj_name: str) -> int:
        obj_name_lower = obj_name.lower()
        for mat_id, mat in config.MATERIAL_MAP.items():
            if mat["name"] in obj_name_lower:
                return mat_id
        return config.DEFAULT_MATERIAL_ID

    def get_material_name(self, obj_name: str) -> str:
        obj_name_lower = obj_name.lower()
        for mat in config.MATERIAL_MAP.values():
            if mat["name"] in obj_name_lower:
                return mat["name"]
        return config.MATERIAL_MAP[config.DEFAULT_MATERIAL_ID]["name"]

    def save_to_file(self):
        save_file_name = config.SCENES_OUT_DIR / Path(self.file_path).with_suffix(".npz").name
        np.savez(save_file_name, material_core=self.space_matrix)
        print("Saved scene to file: ", save_file_name)

    def fill_normal_objects(self,voxelized_object):
        filled_object = voxelized_object.fill()
        if len(filled_object.points) > 0:
            print(f" -> fill() OK: {len(filled_object.points)} voxels")
            return filled_object
        else:
            print(f" -> fill() ERROR: could not fill object. Returning unfilled object.")
            return voxelized_object

    def voxelize_geometry(self,geom, geom_name):

        #wokselizacja obiektu
        voxelized_object = geom.voxelized(pitch=self.DX)

        #wypelniania jesli nie jest sciana
        is_wall = "wall" in geom_name.lower()
        if not is_wall:
            voxelized_object = self.fill_normal_objects(voxelized_object)


        points = voxelized_object.points

        #zamiana metrow na indexy siatki
        indices = np.round(points / self.DX).astype(int) + np.array([self.core_x, self.core_y, self.core_z])

        #walidacja czy punkty sa na siatce
        valid_mask = np.all(
            (indices >= 0) & (indices < [self.NX, self.NY, self.NZ]),
            axis=1
        )

        valid_indices = indices[valid_mask]

        return valid_indices

    def load_scene(self):
        scene = trimesh.load(self.file_path, force='scene', process=True)

        print(f"Number of objects detected in scene: {len(scene.geometry.items())}")

        for geom_name, geom in scene.geometry.items():
            print(f"Processing ==> {geom_name}")
            print(f" -> Is it watertight? {geom.is_watertight}")

            indices = self.voxelize_geometry(geom, geom_name)

            if len(indices) == 0:
                print(f" -> Object is outside of domain - skipping")
                continue
            material_id = self.get_material_id(geom_name)

            print(f" -> Material ={self.get_material_name(geom_name)}, found voxels: {len(indices)}")

            #wpisywanie do glownej macierzy
            x,y,z = indices.T
            values = self.space_matrix[x,y,z]
            self.space_matrix[x,y,z] = np.maximum(values, material_id)

        self.save_to_file()










