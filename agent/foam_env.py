import math
from typing import Any, SupportsFloat

import numpy as np
import taichi as ti
import gymnasium
from gymnasium.core import ObsType, ActType

from agent.metrics import c50
from assets.rooms_2d.room_2d_generator import generate_room
from assets.rooms_2d.room_2d_rasterizer import rasterize_room_data
from simulation.headless_runner import simulate_2d

MAX_SLOTS = 200
RESET_TAICHI_EVERY = 256

TARGET_C50 = 3.0          # [dB] every microphone should reach at least this
FOAM_COST = 2.0           # [dB of deficit] penalty for every placed foam, well above the critic error (~0.8)
MAX_FOAM_FRACTION = 0.3   # foam budget = this fraction of the room's slots
MAX_ROOM_DRAWS = 50       # rooms already meeting the target are skipped in training

TRAIN_ROOM_SEEDS = 10_000  # training rooms use seeds [0, TRAIN_ROOM_SEEDS), test rooms use seeds above

OBS_ROOM_SIDE = 12.0
MAX_DIAGONAL = math.hypot(OBS_ROOM_SIDE, OBS_ROOM_SIDE)

CROWD_RADIUS = 1.0   # [m] foams closer than this to a slot count as its neighbours
CROWD_FULL = 8       # this many neighbours (a full metre of wall on both sides) gives the feature value 1

# observation columns (one row per slot)
# no absolute x, y on purpose: with them the agent memorised "start in the (0, 0) corner and go along the wall"
F_EXISTS = 0              # 1 for a real slot, 0 for padding
F_NORMAL_X, F_NORMAL_Y = 1, 2
F_DIST_SPEAKER = 3
F_DIST_MICS_MEAN = 4
F_DIST_UNSATISFIED_MIC = 5  # nearest microphone still below target
F_DIST_CORNER = 6
F_EXCESS_PATH = 7           # speaker -> slot -> mic minus speaker -> mic, mean over mics below target
F_DIST_NEAREST_FOAM = 8
F_FOAMS_NEARBY = 9          # foams within CROWD_RADIUS
F_WALL_LENGTH = 10
F_WALL_COVERAGE = 11        # fraction of this wall already covered with foam
F_OPPOSITE_COVERAGE = 12    # fraction of the walls facing this one already covered with foam
# per microphone, mics sorted from the largest deficit, so the first block is always the mic that needs most help
OBS_MICS = 3                # rooms with fewer mics get padding blocks (far away, no deficit)
F_MIC_START = 13
F_MIC_DIST, F_MIC_EXCESS_PATH, F_MIC_DEFICIT = 0, 1, 2  # offsets inside one mic block
MIC_BLOCK = 3
F_OCCUPIED = F_MIC_START + OBS_MICS * MIC_BLOCK
F_DEFICIT = F_OCCUPIED + 1      # global, same in every row
F_BUDGET_LEFT = F_OCCUPIED + 2  # global, same in every row
NUM_OF_FEATURES = F_OCCUPIED + 3


def c50_deficit(c50_per_mic: np.ndarray) -> float:
    """Sum over microphones of how many dB are missing to the target (0 when all reach it)."""
    return float(np.sum(np.maximum(0.0, TARGET_C50 - c50_per_mic)))


def nearest_distance(points: np.ndarray, pos: np.ndarray) -> float:
    """Normalised distance from pos to the closest point (1.0 = as far as possible, also when there are no points)."""
    if len(points) == 0:
        return 1.0
    return np.linalg.norm(points - pos, axis=1).min() / MAX_DIAGONAL


class FoamEnv(gymnasium.Env):
    """Each episode: new random room, agent places foams until every microphone has C50 >= TARGET_C50
    or the foam budget runs out.

    Reward per step = decrease of the C50 deficit - FOAM_COST, so the agent learns to fix
    every listener with as few foams as possible. Over-damping above the target gives nothing.

    """
    def __init__(self,
                 material: int,
                 foam_size: float,
                 mic_count: int,
                 max_freq: float,
                 room_min: float,
                 room_max: float
        ):
        super().__init__()
        self.action_space = gymnasium.spaces.Discrete(MAX_SLOTS)
        self.observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(MAX_SLOTS, NUM_OF_FEATURES),
                                                      dtype=np.float32)

        self.material = material
        self.foam_size = foam_size
        self.mic_count = mic_count
        self.max_freq = max_freq
        self.room_min = room_min
        self.room_max = room_max

        self.sim_count = 0
        self.chosen_foams = []
        self.steps_taken = 0
        self.foam_budget = 0
        self.c50_per_mic = np.zeros(mic_count)
        self.deficit = 0.0

        self.generated_room = None
        self.rasterized_room = None

    def increase_sim_count(self):
        self.sim_count += 1

    def _simulate_c50(self) -> np.ndarray:
        run_result = simulate_2d(self.rasterized_room, self.chosen_foams)
        self.increase_sim_count()
        return np.array([c50(mic, run_result.sample_rate) for mic in run_result.raw_pressures])

    def _observation(self) -> np.ndarray:
        room = self.generated_room
        mics = np.array([[m['x_m'], m['y_m']] for m in room['microphones']])
        speaker = np.array([room['speaker']['x_m'], room['speaker']['y_m']])
        unsatisfied = mics[self.c50_per_mic < TARGET_C50]
        slots = room['foam_slots']
        positions = np.array([[s['x_m'], s['y_m']] for s in slots])
        foams = positions[self.chosen_foams].reshape(-1, 2)

        occupied = np.zeros(len(slots), dtype=bool)
        occupied[self.chosen_foams] = True
        wall_coverage = np.array([occupied[self.slot_wall == w].mean() for w in range(len(self.wall_normals))])
        opposite_coverage = np.zeros(len(self.wall_normals))
        for w, normal in enumerate(self.wall_normals):
            facing = np.flatnonzero(self.wall_normals @ normal < -0.5)
            if len(facing):
                opposite_coverage[w] = occupied[np.isin(self.slot_wall, facing)].mean()

        # reflection path detour through the slot, for mics that still need help (all mics once every one is fine)
        path_mics = unsatisfied if len(unsatisfied) else mics
        direct = np.linalg.norm(path_mics - speaker, axis=1)

        mic_deficits = np.maximum(0.0, TARGET_C50 - self.c50_per_mic)
        mic_order = np.argsort(-mic_deficits, kind="stable")
        mics_sorted = mics[mic_order]
        direct_sorted = np.linalg.norm(mics_sorted - speaker, axis=1)

        # global features normalised to [0, 1]
        deficit_norm = min(1.0, self.deficit / (len(mics) * 10.0))
        budget_left = (self.foam_budget - len(self.chosen_foams)) / self.foam_budget

        observation_arr = np.zeros((MAX_SLOTS, NUM_OF_FEATURES), dtype=np.float32)

        for i, slot in enumerate(slots):
            pos = positions[i]
            wall = self.slot_wall[i]
            mic_dist = np.linalg.norm(mics - pos, axis=1)
            excess = slot['distance_to_speaker'] + np.linalg.norm(path_mics - pos, axis=1) - direct
            foams_nearby = np.sum(np.linalg.norm(foams - pos, axis=1) < CROWD_RADIUS) - occupied[i]

            row = observation_arr[i]
            row[F_EXISTS] = 1.0
            row[F_NORMAL_X] = slot['normal_vector'][0]
            row[F_NORMAL_Y] = slot['normal_vector'][1]
            row[F_DIST_SPEAKER] = slot['distance_to_speaker'] / MAX_DIAGONAL
            row[F_DIST_MICS_MEAN] = mic_dist.mean() / MAX_DIAGONAL
            row[F_DIST_UNSATISFIED_MIC] = nearest_distance(unsatisfied, pos)
            row[F_DIST_CORNER] = slot['distance_to_corner'] / MAX_DIAGONAL
            row[F_EXCESS_PATH] = excess.mean() / (2 * MAX_DIAGONAL)
            row[F_DIST_NEAREST_FOAM] = nearest_distance(foams, pos)
            row[F_FOAMS_NEARBY] = foams_nearby / CROWD_FULL
            row[F_WALL_LENGTH] = self.wall_lengths[wall] / OBS_ROOM_SIDE
            row[F_WALL_COVERAGE] = wall_coverage[wall]
            row[F_OPPOSITE_COVERAGE] = opposite_coverage[wall]
            dist_sorted = np.linalg.norm(mics_sorted - pos, axis=1)
            for k in range(OBS_MICS):
                block = F_MIC_START + k * MIC_BLOCK
                if k < len(mics):
                    row[block + F_MIC_DIST] = dist_sorted[k] / MAX_DIAGONAL
                    row[block + F_MIC_EXCESS_PATH] = (slot['distance_to_speaker'] + dist_sorted[k]
                                                      - direct_sorted[k]) / (2 * MAX_DIAGONAL)
                    row[block + F_MIC_DEFICIT] = min(1.0, mic_deficits[mic_order[k]] / 10.0)
                else:
                    row[block + F_MIC_DIST] = 1.0
                    row[block + F_MIC_EXCESS_PATH] = 1.0
            row[F_OCCUPIED] = 1.0 if occupied[i] else 0.0
            row[F_DEFICIT] = deficit_norm
            row[F_BUDGET_LEFT] = budget_left

        return np.clip(observation_arr, -1.0, 1.0)

    def _split_slots_into_walls(self):
        """Wall index of every slot. Slots on one wall share the direction and the coordinate across the wall."""
        slots = self.generated_room['foam_slots']
        keys = [(round(s['normal_vector'][0]), round(s['normal_vector'][1]),
                 round(s['y_m'] if abs(s['normal_vector'][1]) > 0.5 else s['x_m'], 3)) for s in slots]
        walls = list(dict.fromkeys(keys))
        self.slot_wall = np.array([walls.index(k) for k in keys])
        self.wall_normals = np.array([[nx, ny] for nx, ny, _ in walls], dtype=float)
        self.wall_lengths = np.array([np.sum(self.slot_wall == w) * self.foam_size for w in range(len(walls))])

    def _load_room(self, room_seed: int):
        #generate and rasterize the room
        self.generated_room = generate_room(
            material_id=self.material,
            foam_size=self.foam_size,
            num_mics=self.mic_count,
            seed=room_seed,
            room_width_min=self.room_min,
            room_width_max=self.room_max,
            room_length_min=self.room_min,
            room_length_max=self.room_max,
            shape_type="L-shape"
        )
        self.rasterized_room = rasterize_room_data(
            self.generated_room,
            max_freq=self.max_freq,
            foam_size=self.foam_size
        )
        slot_count = len(self.rasterized_room['foam_slots'])
        if slot_count > MAX_SLOTS:
            raise ValueError(f"Room {room_seed} has more than {MAX_SLOTS} foam slots")
        self._split_slots_into_walls()

        self.chosen_foams = []
        self.steps_taken = 0
        self.foam_budget = max(1, math.ceil(MAX_FOAM_FRACTION * slot_count))

        # simulation of the empty room
        self.c50_per_mic = self._simulate_c50()
        self.deficit = c50_deficit(self.c50_per_mic)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # only between rooms, never inside an episode
        if self.sim_count >= RESET_TAICHI_EVERY:
            ti.reset()
            ti.init(arch=ti.gpu, log_level=ti.ERROR)
            self.sim_count = 0

        if options and "room_seed" in options:
            # evaluation: use exactly this room, even if it already meets the target
            room_seed = int(options["room_seed"])
            self._load_room(room_seed)
        else:
            # training: rooms that already meet the target teach nothing
            for _ in range(MAX_ROOM_DRAWS):
                room_seed = int(self.np_random.integers(0, TRAIN_ROOM_SEEDS))
                try:
                    self._load_room(room_seed)
                except ValueError:
                    continue  # generator could not place speaker/microphones in this room
                if self.deficit > 0.0:
                    break

        info = {
            "room_seed": room_seed,
            "c50_empty": self.c50_per_mic.copy(),
            "deficit_empty": self.deficit,
            "foam_budget": self.foam_budget,
        }
        return self._observation(), info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = int(action)
        self.steps_taken += 1
        out_of_budget = self.steps_taken >= self.foam_budget

        # masked-out slot (missing or occupied) - only possible without action masking
        if not self.action_masks()[action]:
            info = self._info(invalid_action=True)
            return self._observation(), -FOAM_COST, out_of_budget, False, info

        self.chosen_foams.append(action)

        deficit_before = self.deficit
        self.c50_per_mic = self._simulate_c50()
        self.deficit = c50_deficit(self.c50_per_mic)

        reward = (deficit_before - self.deficit) - FOAM_COST

        success = self.deficit == 0.0
        terminated = success or out_of_budget
        return self._observation(), reward, terminated, False, self._info()

    def _info(self, invalid_action: bool = False) -> dict[str, Any]:
        return {
            "c50": self.c50_per_mic.copy(),
            "deficit": self.deficit,
            "foams_used": len(self.chosen_foams),
            "success": self.deficit == 0.0,
            "invalid_action": invalid_action,
        }

    def action_masks(self) -> np.ndarray:
        slot_count = len(self.rasterized_room['foam_slots'])
        mask = np.zeros(MAX_SLOTS, dtype=bool)
        mask[:slot_count] = True  # existing slots
        mask[self.chosen_foams] = False  # without occupied
        return mask


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    ti.init(arch=ti.gpu, log_level=ti.ERROR)

    env = FoamEnv(material=5, foam_size=0.25, mic_count=3, max_freq=1000, room_min=6, room_max=12)
    check_env(env, skip_render_check=True)
    print("check_env OK")

    # one episode with random valid actions
    obs, info = env.reset(seed=0)
    deficit_empty = info['deficit_empty']
    print(f"room {info['room_seed']}, empty C50 per mic = {np.round(info['c50_empty'], 1)}, "
          f"deficit = {info['deficit_empty']:.2f}, budget = {info['foam_budget']}")
    total_reward = 0.0
    terminated = False
    while not terminated:
        action = int(env.np_random.choice(np.flatnonzero(env.action_masks())))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  slot {action:3d}  reward {reward:+.2f}  C50 {np.round(info['c50'], 1)}  deficit {info['deficit']:.2f}")

    expected = deficit_empty - info['deficit'] - FOAM_COST * info['foams_used']
    print(f"success={info['success']}  foams={info['foams_used']}  "
          f"sum of rewards {total_reward:.2f} == deficit drop - cost {expected:.2f}")
