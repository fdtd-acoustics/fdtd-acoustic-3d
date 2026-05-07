from pathlib import Path

import numpy as np
import trimesh
import config

SAFETY_MARGIN_FOR_ITER = 2

#Todo: refactor tej klasy
class Voxelizer:
    def __init__(self, NX, NY, NZ, dx, file_path):
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.DX = dx
        self.file_path = file_path
        #self.core_x = NX //2
        #self.core_y = NY //2
        #self.core_z = NZ // 2
        self.grid_center = np.array([self.NX - 1, self.NY - 1, self.NZ - 1]) / 2.0
        self.space_matrix = np.zeros((NX,NY,NZ), dtype=np.int8)

        # 3D Mesh Variables
        self.mesh_vertices = []
        self.mesh_faces = []
        self.mesh_colors = []
        self.vertex_offset = 0

        self.scene_center = np.zeros(3)

# TODO: Test czy materialy tez sa

    def get_material_id(self, obj_name: str) -> int:
        obj_name_lower = obj_name.lower()
        from simulation.simulation_builder import SimulationBuilder
        material_map = SimulationBuilder.get_material_map_from_csv(config.MAIN_MATERIAL_LIBRARY)

        if not material_map:
            raise RuntimeError("Material library is empty or could not be loaded. Check your CSV file.")

        for mat_id, mat in material_map.items():
            if mat["name"] in obj_name_lower:
                return mat_id

        raise ValueError(
            f"\n[Material Mapping Error]\n"
            f"Object name '{obj_name}' does not contain any known material name.\n"
        )

    def get_material_name(self, obj_name: str) -> str:
        obj_name_lower = obj_name.lower()
        from simulation.simulation_builder import SimulationBuilder
        material_map = SimulationBuilder.get_material_map_from_csv(config.MAIN_MATERIAL_LIBRARY)

        if not material_map:
            raise RuntimeError("Material library is empty or could not be loaded. Check your CSV file.")

        for mat in material_map.values():
            if mat["name"] in obj_name_lower:
                return mat["name"]

        raise ValueError(
            f"\n[Material Mapping Error]\n"
            f"Object name '{obj_name}' does not contain any known material name.\n"
        )

    def get_material_color(self, obj_name: str) -> str:
        obj_name_lower = obj_name.lower()
        from simulation.simulation_builder import SimulationBuilder
        material_map = SimulationBuilder.get_material_map_from_csv(config.MAIN_MATERIAL_LIBRARY)

        if not material_map:
            raise RuntimeError("Material library is empty or could not be loaded. Check your CSV file.")

        for mat in material_map.values():
            if mat["name"] in obj_name_lower:
                return mat["color"]

        raise ValueError(
            f"\n[Material Mapping Error]\n"
            f"Object name '{obj_name}' does not contain any known material name.\n"
        )

    def save_to_file(self):
        #save_file_name = config.SCENES_OUT_DIR / Path(self.file_path).with_suffix(".npz").name
        #np.savez(save_file_name, material_core=self.space_matrix)
        #print("Saved scene to file: ", save_file_name)

        save_file_name = config.VOXELS_DIR / Path(self.file_path).with_suffix(".npz").name

        combined_vertices = np.vstack(self.mesh_vertices) if self.mesh_vertices else np.array([])
        combined_faces = np.vstack(self.mesh_faces) if self.mesh_faces else np.array([])
        combined_colors = np.vstack(self.mesh_colors) if self.mesh_colors else np.array([])

        save_file_name.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_file_name,
            material_core=self.space_matrix,
            mesh_vertices=combined_vertices,
            mesh_faces=combined_faces,
            mesh_colors=combined_colors
        )
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

        # Voxelization
        # max_iter liczone z najdluzszej krawedzi mesha
        max_edge_len = float(geom.edges_unique_length.max()) if len(geom.edges_unique_length) else 1.0
        max_iter = max(10, int(np.ceil(np.log2(max_edge_len / self.DX))) + SAFETY_MARGIN_FOR_ITER)
        print(f" -> Voxelizing with max_iter={max_iter} based on max edge length {max_edge_len:.4f}m and DX={self.DX:.4f}m")

        # sprawdzalem dla metody ray ale jest niedokladna - niektore sciany sie nie wokselizuja
        voxelized_object = geom.voxelized(pitch=self.DX, method="subdivide", max_iter=max_iter)

        # Fill inside if not wall
        mat_name = self.get_material_name(geom_name)
        is_wall = "wall" in geom_name.lower() or "wall" in mat_name.lower()
        if not is_wall:
            voxelized_object = self.fill_normal_objects(voxelized_object)

        points = voxelized_object.points - self.scene_center

        # Meters to grid indexes
        #indices = np.round(points / self.DX).astype(int) + np.array([self.core_x, self.core_y, self.core_z])
        grid_points = (points / self.DX) + self.grid_center
        indices = np.floor(grid_points + 0.5).astype(int)

        # Validation if points are on the grid
        valid_mask = np.all(
            (indices >= 0) & (indices < [self.NX, self.NY, self.NZ]),
            axis=1
        )

        valid_indices = indices[valid_mask]

        return valid_indices

    def load_scene(self):
        scene = trimesh.load(self.file_path, force='scene')
        geometries = scene.geometry
        scene_bounds = scene.bounds

        # Finding center of the .obj model for later centering if obj is not centered
        self.scene_center = np.mean(scene_bounds, axis=0)
        print(f"Number of objects detected in scene: {len(geometries)}")

        for geom_name, geom in geometries.items():
            print(f"Processing ==> {geom_name}")

            # --- Voxels ---
            indices = self.voxelize_geometry(geom, geom_name)

            if len(indices) == 0:
                print(f" -> Object is outside of domain - skipping")
                continue

            material_id = self.get_material_id(geom_name)
            mat_name = self.get_material_name(geom_name)
            print(f" -> Material ={mat_name}, found voxels: {len(indices)}")

            # Adding Voxels to the Matrix
            x, y, z = indices.T
            values = self.space_matrix[x, y, z]
            self.space_matrix[x, y, z] = np.maximum(values, material_id)

            # --- Mesh ---
            verts = geom.vertices
            faces = geom.faces

            # Shift mesh verticies to the calculated scene center
            shifted_verts = verts - self.scene_center
            offset_vector = self.grid_center * self.DX
            aligned_verts = shifted_verts + offset_vector

            from simulation.simulation_builder import SimulationBuilder
            # Mesh Colors for materials
            hex_color = self.get_material_color(geom_name)
            rgb_color = SimulationBuilder.hex_to_rgb_normalized(hex_color)
            vert_colors = np.tile(rgb_color, (len(aligned_verts), 1))

            self.mesh_vertices.append(aligned_verts)
            self.mesh_faces.append(faces + self.vertex_offset)
            self.mesh_colors.append(vert_colors)

            self.vertex_offset += len(aligned_verts)

        self.save_to_file()
