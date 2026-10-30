# ==============================================================================
# File:         random_baseline.py
# Description:  Implements a random baseline for acoustic foam placement
#              inside the FDTD simulation environment. Iteratively picks random
#              available slots until max_foams are placed, saving visualizations
#              at each step.
# Usage Example:
#               random_baseline(room_seed=0, max_foams=8)
# ==============================================================================

import taichi as ti
import numpy as np
import time
from tqdm import tqdm
from agent.foam_env import FoamEnv
from assets.rooms_2d.room_2d_visualizer import visualize_simulation_grid

def random_baseline(room_seed: int = 0, max_foams: int = 8):
    ti.init(arch=ti.gpu, log_level=ti.ERROR)
    env = FoamEnv(material=6, foam_size=0.25, mic_count=3, max_freq=1000)

    _, info = env.reset(options={"room_seed": room_seed})
    c50_empty = info['c50_empty']

    print(f"=====================================")
    print(f"  Random Algorithm | Room seed: {room_seed}")
    print(f"-------------------------------------")
    print(f"  Empty room C50: {c50_empty:.2f} dB")
    print(f"  Starting random placement for {max_foams} foams")
    print(f"=====================================")

    time.sleep(1)

    pbar = tqdm(total=max_foams, desc="Random Placement", unit="foam")

    for step in range(1, max_foams + 1):
        masks = env.action_masks()
        available_slots = np.flatnonzero(masks)

        if len(available_slots) == 0:
            print("No more available slots for foam placement!")
            break

        chosen_action = np.random.choice(available_slots)

        _, _, _, _, step_info = env.step(chosen_action)

        room_data = env.rasterized_room
        title_str = f"Random - Foam {step} (C50={step_info['c50']:.2f}dB)"
        save_path = f"../../output/agent/random_baseline/random_seed{room_seed}_foams{step}.png"
        save_path_m = f"../../output/agent/random_baseline/random_seed{room_seed}_foams{step}_m.png"

        visualize_simulation_grid(room_data, title=title_str, save_path=save_path, show=False, use_meters=False, chosen_foams=env.chosen_foams)
        visualize_simulation_grid(room_data, title=title_str, save_path=save_path_m, show=False, use_meters=True, chosen_foams=env.chosen_foams)

        pbar.update(1)
        pbar.set_postfix(C50=f"{step_info['c50']:.2f}dB")
        tqdm.write(f" -> [Saved image] Foams {step}: Slot {chosen_action} | C50: {step_info['c50']:.2f} dB")

    pbar.close()

    print(f"=====================================")
    print(f"  Finished random baseline!")
    print(f"  End C50 result: {env.c50_prev:.2f} dB")
    print(f"  Selected slots: {env.chosen_foams}")
    print(f"=====================================")


if __name__ == "__main__":
    random_baseline(room_seed=0, max_foams=8)