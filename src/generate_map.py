from taichi import j

import config
import numpy as np
def create_material_map():
    alpha_air, density_air = config.MATERIAL_PROPS["air"]
    alpha_brick, density_brick = config.MATERIAL_PROPS["brick"]
    alpha_wood, density_wood = config.MATERIAL_PROPS["wooden_bench"]

    N = 200 # number of point(without PML!)

    alpha_map = np.full((N, N), alpha_air, dtype=np.float32)
    density_map = np.full((N, N), density_air, dtype=np.float32)

    # example obstacle
    alpha_map[51:100, 51:100] = alpha_brick
    density_map[51:100, 51:100] = density_brick


    return alpha_map, density_map