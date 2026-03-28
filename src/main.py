import taichi as ti
import numpy as np
import time
import config

from generate_map import create_material_map
from fdtd_simulation import FDTD_Simulation


def render_frame(gui, sim, is_wall):
    p_np = sim.get_current_pressure()

    # normalize
    fixed_max = sim.amplitude * 0.5
    p_normalized = p_np / fixed_max
    gray_val = p_normalized * 0.5 + 0.5
    gray_val = np.clip(gray_val, 0.0, 1.0)

    img = np.dstack((gray_val, gray_val, gray_val))

    img[is_wall, 1] *= 0.3
    img[is_wall, 2] *= 0.3

    gui.set_image(img)
    gui.show()



def main():
    ti.init(arch=ti.gpu)

    # 1. Create matrices
    alpha_map, density_map = create_material_map()

    # 2. FDTD engine initialization
    sim = FDTD_Simulation(alpha_map, density_map)

    # 3. GUI setup
    gui = ti.GUI("2d - fdtd (pml & neumann)", res=(sim.N, sim.N))

    final_alpha_map = sim.get_alpha_map()
    is_wall = final_alpha_map > 0.001

    prev_time = time.time()
    accumulator = 0.0
    is_paused = False

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.SPACE:
                is_paused = not is_paused

        now = time.time()
        delta_real_time = now - prev_time
        prev_time = now
        accumulator += delta_real_time

        if accumulator >= config.FRAME_DURATION:
            if not is_paused:
                sim.update()

            render_frame(gui, sim, is_wall)
            accumulator = 0


if __name__ == "__main__":
    main()