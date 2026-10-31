import argparse
from pathlib import Path

import taichi as ti
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from agent.foam_env import FoamEnv
from agent.slot_policy import SlotPolicy

MATERIAL_ID = 5  # wood_50mm_thick: empty rooms start below the C50 target
FOAM_SIZE = 0.25 # [M]
MIC_COUNT = 3
MAX_FREQ = 1000

OUTPUT_DIR = Path("output/agent")

def make_env() -> Monitor:
    env = FoamEnv(material=MATERIAL_ID, foam_size=FOAM_SIZE, mic_count=MIC_COUNT, max_freq=MAX_FREQ)
    # Monitor logs episode reward and length (= foams used), plus success flag
    return Monitor(env, filename=str(OUTPUT_DIR / "monitor"), info_keywords=("success",))


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO agent for foam placement")
    parser.add_argument("--steps", type=int, default=5_000, help="Total agent steps (1 step = 1 simulation)")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a saved model .zip to continue from")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ti.init(arch=ti.gpu, log_level=ti.ERROR)

    env = make_env()

    if args.resume:
        model = MaskablePPO.load(args.resume, env=env, device="cpu")
    else:
        model = MaskablePPO(
            SlotPolicy,
            env,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=1.0,
            learning_rate=3e-4,
            seed=args.seed,
            verbose=1,
            device="cpu",
        )

    checkpoint = CheckpointCallback(
        save_freq=2_000,
        save_path=str(OUTPUT_DIR / "checkpoints"),
        name_prefix="ppo_foam",
    )

    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint,
        reset_num_timesteps=args.resume is None,
        progress_bar=True
    )

    final_path = OUTPUT_DIR / "ppo_foam_final"
    model.save(final_path)
    print(f"Saved model: {final_path}.zip")


if __name__ == "__main__":
    main()
