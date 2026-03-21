import taichi as ti


@ti.kernel
def generate_simulation_map(material_core: ti.template()):
    for i, j, k in material_core:
        material_core[i, j, k] = ti.Vector([0.0, 1.21])


