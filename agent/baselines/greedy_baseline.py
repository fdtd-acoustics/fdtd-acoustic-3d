# ==============================================================================
# File:         greedy_baseline.py
# Description:  Implements a greedy optimization baseline for acoustic foam placement
#               inside the FDTD simulation environment. Iteratively evaluates available
#               slots to maximize the C50 acoustic clarity metric and saves grid visualizations
#               both in voxels and physical meters at each optimization step.
# Usage Example:
#               greedy_baseline(room_seed=0, max_foams=8)
# ==============================================================================

import taichi as ti
import numpy as np
import time
from tqdm import tqdm
from agent.foam_env import FoamEnv
from assets.rooms_2d.room_2d_visualizer import visualize_simulation_grid

def greedy_baseline(room_seed: int = 0, max_foams: int = 8):
    ti.init(arch=ti.gpu, log_level=ti.ERROR)
    env = FoamEnv(material=6, foam_size=0.25, mic_count=3, max_freq=1000)

    _, info = env.reset(options={"room_seed": room_seed})
    c50_empty = info['c50_empty']

    print(f"=====================================")
    print(f"  Greedy Algorithm | Room seed: {room_seed}")
    print(f"-------------------------------------")
    print(f"  Empty room C50: {c50_empty:.2f} dB")
    print(f"  Starting optimization for {max_foams} foams")
    print(f"=====================================")

    time.sleep(1)

    initial_slots = len(np.flatnonzero(env.action_masks()))
    total_simulations = sum(initial_slots - i for i in range(max_foams))

    pbar = tqdm(total=total_simulations, desc="Greedy Optimization", unit="simulation")

    for step in range(1, max_foams + 1):
        masks = env.action_masks()
        available_slots = np.flatnonzero(masks)

        best_action = None
        best_c50 = -float('inf')

        saved_foams = list(env.chosen_foams)
        saved_c50 = env.c50_prev

        for slot in available_slots:
            env.chosen_foams = list(saved_foams)
            env.c50_prev = saved_c50

            _, _, _, _, step_info = env.step(slot)
            current_c50 = step_info['c50']

            if current_c50 > best_c50:
                best_c50 = current_c50
                best_action = slot

            pbar.update(1)
            pbar.set_postfix(
                foam=f"{step}/{max_foams}",
                best_C50=f"{best_c50:.2f}dB"
            )

        env.chosen_foams = list(saved_foams)
        env.c50_prev = saved_c50
        _, _, _, _, step_info = env.step(best_action)

        room_data = env.rasterized_room
        title_str = f"Greedy - Foam {step} (C50={step_info['c50']:.2f}dB)"
        save_path = f"../../output/agent/greedy_baseline/greedy_seed{room_seed}_foams{step}.png"
        save_path_m = f"../../output/agent/greedy_baseline/greedy_seed{room_seed}_foams{step}_m.png"

        visualize_simulation_grid(room_data, title=title_str, save_path=save_path, show=False, use_meters=False, chosen_foams=env.chosen_foams)
        visualize_simulation_grid(room_data, title=title_str, save_path=save_path_m, show=False, use_meters=True, chosen_foams=env.chosen_foams)

        tqdm.write(f" -> [Saved image] Foams {step}: Slot {best_action} | C50: {step_info['c50']:.2f} dB")

    pbar.close()

    print(f"=====================================")
    print(f"  Finished optimization!")
    print(f"  End C50 result: {env.c50_prev:.2f} dB")
    print(f"  Selected slots: {env.chosen_foams}")
    print(f"=====================================")


if __name__ == "__main__":
    greedy_baseline(room_seed=0, max_foams=8)