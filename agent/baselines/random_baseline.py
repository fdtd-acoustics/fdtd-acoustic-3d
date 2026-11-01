# ==============================================================================
# File:         random_baseline.py
# Description:  Implements a random baseline for acoustic foam placement
#               inside the FDTD simulation environment. Iteratively picks random
#               available slots until max_foams are placed, saving visualizations
#               at each step.
# Usage Example:
#               python -m agent.baselines.random_baseline
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
    c50_empty_mean = np.mean(info['c50_empty'])

    print(f"=====================================")
    print(f"  Random Algorithm | Room seed: {room_seed}")
    print(f"-------------------------------------")
    print(f"  Empty room Mean C50: {c50_empty_mean:.2f} dB")
    print(f"  Starting random placement for {max_foams} foams")
    print(f"=====================================")

    time.sleep(1)

    with tqdm(total=max_foams, desc="  [Random]", leave=True, unit="foam", ncols=100) as pbar:
        terminated = False
        step = 0

        while not terminated and len(env.chosen_foams) < max_foams:
            step += 1
            masks = env.action_masks()
            available_slots = np.flatnonzero(masks)

            if len(available_slots) == 0:
                tqdm.write("No more available slots for foam placement!")
                break

            pbar.update(1)

            chosen_action = int(env.np_random.choice(available_slots))
            _, _, terminated, _, step_info = env.step(chosen_action)

            mean_c50 = np.mean(step_info['c50'])
            room_data = env.rasterized_room
            title_str = f"Random - Foam {step} (Mean C50={mean_c50:.2f}dB)"
            save_path = f"../../output/random_baseline/random_seed{room_seed}_foams{step}.png"
            save_path_m = f"../../output/random_baseline/random_seed{room_seed}_foams{step}_m.png"

            visualize_simulation_grid(room_data, title=title_str, save_path=save_path, show=False, use_meters=False, chosen_foams=env.chosen_foams)
            visualize_simulation_grid(room_data, title=title_str, save_path=save_path_m, show=False, use_meters=True, chosen_foams=env.chosen_foams)

            pbar.set_postfix(deficit=f"{step_info['deficit']:.2f}dB")

            #tqdm.write(f" -> [Saved image] Foams {step}: Slot {chosen_action} | Mean C50: {mean_c50:.2f} dB")

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
        print(f"  Finished random baseline!")
        print(f"  End Mean C50 result: {final_mean_c50:.2f} dB")
        print(f"  Selected slots: {env.chosen_foams}")
        print(f"  Saved images: /output/random_baseline/*")
        print(f"=====================================")


if __name__ == "__main__":
    random_baseline(room_seed=0, max_foams=8)