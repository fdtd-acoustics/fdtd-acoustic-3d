import taichi as ti
import numpy as np
import time
import config

from generate_map import create_material_map
from fdtd_simulation import FDTD_Simulation


def render_frame(gui, sim, is_wall):
    pml = sim.pml_thickness
    p_np = sim.get_current_pressure()

    # --- Linear Scalling ---
    # fixed_max = sim.amplitude * 0.1
    # p_normalized = p_np / fixed_max
    # gray_val = p_normalized * 0.5 + 0.5
    # gray_val = np.clip(gray_val, 0.0, 1.0)

    # --- Logarithmic Scaling ---
    p0 = 2e-5 # # reference pressure for the human threshold of hearing (20 micropascals)
    epsilon = 1e-10 # safety offset to prevent log10(0) math errors during silenc
    max_db = 100.0  # the range we want to see

    p_abs = np.abs(p_np) + epsilon
    spl_db = 20.0 * np.log10(p_abs / p0)
    spl_normalized = np.clip(spl_db / max_db, 0.0, 1.0)
    p_signed_log = np.sign(p_np) * spl_normalized

    gray_val = p_signed_log * 0.5 + 0.5


    img = np.dstack((gray_val, gray_val, gray_val))

    pml_mask = np.zeros((sim.N, sim.N), dtype=bool)
    pml_mask[:pml, :] = True
    pml_mask[-pml:, :] = True
    pml_mask[:, :pml] = True
    pml_mask[:, -pml:] = True

    img[pml_mask, 0] *= 0.6
    img[pml_mask, 1] *= 0.4
    img[pml_mask, 2] *= 0.8

    pure_wall_mask = is_wall & ~pml_mask

    img[pure_wall_mask, 1] *= 0.3
    img[pure_wall_mask, 2] *= 0.3

    gui.set_image(img)

    # show real time
    time_text = f"Time: {sim.get_time():.4f} s"
    gui.text(time_text, pos=[0.05, 0.95], color=0xFFFFFF)
    # show step number
    step_text = f"Step: {sim.steps}"
    gui.text(step_text, pos=[0.05, 0.92], color=0x00FF00)

    gui.show()



def main():
    ti.init(arch=ti.gpu)

    # 1. Create matrices
    alpha_map, density_map = create_material_map()

    # 2. FDTD engine initialization
    sim = FDTD_Simulation(alpha_map, density_map)
    sim.print_config()

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