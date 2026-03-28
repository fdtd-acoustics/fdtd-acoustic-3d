import config
import numpy as np
def create_material_map():
    alpha_air, density_air = config.MATERIAL_PROPS["air"]
    alpha_brick, density_brick = config.MATERIAL_PROPS["brick"]
    alpha_wood, density_wood = config.MATERIAL_PROPS["wooden_bench"]

    N = 500 # number of point(without PML!)

    alpha_map = np.full((N, N), alpha_air, dtype=np.float32)
    density_map = np.full((N, N), density_air, dtype=np.float32)

    # example obstacle
    # alpha_map[201:202, 0:250] = alpha_wood
    # density_map[201:202, 0:250] = density_wood
    #
    # alpha_map[201:202, 254:320] = alpha_wood
    # density_map[201:202, 254:320] = density_wood
    #
    # alpha_map[201:202, 325:499] = alpha_wood
    # density_map[201:202, 325:499] = density_wood

    alpha_map[201:202, 0:500] = alpha_wood
    density_map[201:202, 0:500] = density_wood


    return alpha_map, density_map