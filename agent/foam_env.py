from typing import Any, SupportsFloat

import numpy as np
import taichi as ti
import gymnasium
from gymnasium.core import ObsType, ActType

from agent.metrics import mean_c50
from assets.rooms_2d.room_2d_generator import generate_l_shape_room
from assets.rooms_2d.room_2d_rasterizer import rasterize_room_data
from simulation.headless_runner import simulate_2d

MAX_SLOTS = 200
FOAMS_PER_EPISODE = 8
RESET_TAICHI_EVERY = 200

NUM_OF_FEATURES = 9

TRAIN_ROOM_SEEDS = 10_000  # training rooms use seeds [0, TRAIN_ROOM_SEEDS), test rooms use seeds above


class FoamEnv(gymnasium.Env):
    def __init__(self,
                 material: int,
                 foam_size: float,
                 mic_count: int,
                 max_freq: float
        ):
        super().__init__()
        self.action_space = gymnasium.spaces.Discrete(MAX_SLOTS)
        self.observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(MAX_SLOTS, NUM_OF_FEATURES),
                                                      dtype=np.float32)

        self.material = material
        self.foam_size = foam_size
        self.mic_count = mic_count
        self.max_freq = max_freq

        self.sim_count = 0
        self.chosen_foams = []
        self.steps_taken = 0
        self.c50_prev = 0.0

        self.generated_room = None
        self.rasterized_room = None

    def increase_sim_count(self):
        self.sim_count += 1

    def _observation(self) -> np.ndarray:
        room = self.generated_room
        width, height = room['width_m'], room['height_m']
        diagonal = np.hypot(width, height)

        mics = np.array([[m['x_m'], m['y_m']] for m in room['microphones']])

        observation_arr = np.zeros((MAX_SLOTS, NUM_OF_FEATURES), dtype=np.float32)

        foams_left_to_choose = (FOAMS_PER_EPISODE - len(self.chosen_foams)) / FOAMS_PER_EPISODE

        for i, slot in enumerate(room['foam_slots']):
            # how far slot is from each mic (mean)
            mic_dist = np.hypot(mics[:, 0] - slot['x_m'], mics[:, 1] - slot['y_m']).mean()

            observation_arr[i] = [
                slot['x_m'] / width,
                slot['y_m'] / height,
                slot['normal_vector'][0],
                slot['normal_vector'][1],
                slot['distance_to_speaker'] / diagonal,
                mic_dist / diagonal,
                slot['distance_to_corner'] / diagonal,
                1.0 if i in self.chosen_foams else 0.0,
                foams_left_to_choose
            ]

        return observation_arr

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        if self.sim_count >= RESET_TAICHI_EVERY:
            ti.reset()
            ti.init(arch=ti.gpu, log_level=ti.ERROR)
            self.sim_count = 0

        if options and "room_seed" in options:
            room_seed = int(options["room_seed"])
        else:
            room_seed = int(self.np_random.integers(0, TRAIN_ROOM_SEEDS))

        self.chosen_foams = []
        self.steps_taken = 0

        self.generated_room = generate_l_shape_room(
            material_id=self.material,
            foam_size=self.foam_size,
            num_mics=self.mic_count,
            seed=room_seed
        )

        self.rasterized_room = rasterize_room_data(
            self.generated_room,
            max_freq=self.max_freq,
            foam_size=self.foam_size
        )

        if len(self.rasterized_room['foam_slots']) > MAX_SLOTS:
            raise ValueError(f"Room {room_seed} has more than {MAX_SLOTS} foam slots")

        # run simulation on empty room
        run_result = simulate_2d(self.rasterized_room, self.chosen_foams)
        self.increase_sim_count()

        self.c50_prev = mean_c50(run_result.raw_pressures, run_result.sample_rate)

        info = {"room_seed": room_seed, "c50_empty": self.c50_prev}
        return self._observation(), info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = int(action)
        self.steps_taken += 1
        terminated = self.steps_taken == FOAMS_PER_EPISODE

        if not self.action_masks()[action]:
            return self._observation(), 0.0, terminated, False, {"c50": self.c50_prev, "invalid_action": True}

        self.chosen_foams.append(action)

        run_result = simulate_2d(self.rasterized_room, self.chosen_foams)
        self.increase_sim_count()
        c50_now = mean_c50(run_result.raw_pressures, run_result.sample_rate)

        reward = c50_now - self.c50_prev
        self.c50_prev = c50_now

        info = {"c50": c50_now}
        return self._observation(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        slot_count = len(self.rasterized_room['foam_slots'])
        mask = np.zeros(MAX_SLOTS, dtype=bool)
        mask[:slot_count] = True  # existing slots
        mask[self.chosen_foams] = False  # without occupied
        return mask


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    ti.init(arch=ti.gpu, log_level=ti.ERROR)

    env = FoamEnv(material=6, foam_size=0.25, mic_count=3, max_freq=1000)
    check_env(env, skip_render_check=True)
    print("check_env OK")

    # one episode with random valid actions
    obs, info = env.reset(seed=0)
    c50_empty = info['c50_empty']
    print(f"room {info['room_seed']}, empty C50 = {c50_empty:.2f} dB")
    total_reward = 0.0
    terminated = False
    while not terminated:
        action = int(env.np_random.choice(np.flatnonzero(env.action_masks())))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  slot {action:3d}  reward {reward:+.2f}  C50 {info['c50']:.2f}")

    print(f"sum of rewards {total_reward:.2f} == final - empty {info['c50'] - c50_empty:.2f}")
