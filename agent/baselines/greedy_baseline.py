# ==============================================================================
# File:         greedy_baseline.py
# Description:  Implements a greedy optimization baseline for acoustic foam placement
#               inside the FDTD simulation environment. Iteratively evaluates available
#               slots to maximize the C50 acoustic clarity metric and saves grid visualizations
#               both in voxels and physical meters at each optimization step.
# Usage Example:
#               python -m agent.baselines.greedy_baseline
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
    c50_empty_mean = np.mean(info['c50_empty'])

    print(f"=====================================")
    print(f"  Greedy Algorithm | Room seed: {room_seed}")
    print(f"-------------------------------------")
    print(f"  Empty room Mean C50: {c50_empty_mean:.2f} dB")
    print(f"  Starting optimization for {max_foams} foams")
    print(f"=====================================")

    time.sleep(1)

    initial_slots = len(np.flatnonzero(env.action_masks()))
    total_simulations = sum(initial_slots - i for i in range(max_foams))

    with tqdm(total=total_simulations, desc="  [Greedy]", leave=True, unit="sim", ncols=100) as pbar:
        terminated = False
        step = 0

        while not terminated and len(env.chosen_foams) < max_foams:
            step += 1
            masks = env.action_masks()
            available_slots = np.flatnonzero(masks)

            if len(available_slots) == 0:
                break

            best_action = None
            best_deficit = float('inf')

            saved_foams = list(env.chosen_foams)
            saved_c50 = env.c50_per_mic.copy()
            saved_deficit = env.deficit

            for slot in available_slots:
                env.chosen_foams = list(saved_foams)
                env.c50_per_mic = saved_c50.copy()
                env.deficit = saved_deficit

                _, _, _, _, step_info = env.step(slot)
                current_deficit = step_info['deficit']

                if current_deficit < best_deficit:
                    best_deficit = current_deficit
                    best_action = slot

                pbar.update(1)
                pbar.set_postfix(deficit=f"{best_deficit:.2f}dB")

            env.chosen_foams = list(saved_foams)
            env.c50_per_mic = saved_c50.copy()
            env.deficit = saved_deficit
            _, _, terminated, _, step_info = env.step(best_action)

            mean_c50 = np.mean(step_info['c50'])
            room_data = env.rasterized_room
            title_str = f"Greedy - Foam {step} (Mean C50={mean_c50:.2f}dB)"
            save_path = f"../../output/greedy_baseline/greedy_seed{room_seed}_foams{step}.png"
            save_path_m = f"../../output/greedy_baseline/greedy_seed{room_seed}_foams{step}_m.png"

            visualize_simulation_grid(room_data, title=title_str, save_path=save_path, show=False, use_meters=False, chosen_foams=env.chosen_foams)
            visualize_simulation_grid(room_data, title=title_str, save_path=save_path_m, show=False, use_meters=True, chosen_foams=env.chosen_foams)

            #tqdm.write(f" -> [Saved image] Foams {step}: Slot {best_action} | Mean C50: {mean_c50:.2f} dB")

            final_c50 = np.mean(env.c50_per_mic)
            if env.deficit == 0.0:
                if pbar.n < max_foams:
                    pbar.update(max_foams - pbar.n)
                pbar.set_postfix(foams=f"{len(env.chosen_foams)}", C50=f"{final_c50:.2f}dB")
            else:
                pbar.set_postfix(foams=f"{len(env.chosen_foams)}", deficit=f"{env.deficit:.2f}dB")

        pbar.close()

        final_mean_c50 = np.mean(env.c50_per_mic)

        time.sleep(1)
        print(f"=====================================")
        print(f"  Finished greedy baseline!")
        print(f"  End Mean C50 result: {final_mean_c50:.2f} dB")
        print(f"  Selected slots: {env.chosen_foams}")
        print(f"  Saved images: /output/greedy_baseline/*")
        print(f"=====================================")


if __name__ == "__main__":
    greedy_baseline(room_seed=0, max_foams=8)