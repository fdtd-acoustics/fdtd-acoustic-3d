import argparse
import functools
from pathlib import Path

import numpy as np
import taichi as ti
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import agent.foam_env as foam_env
from agent.foam_env import FoamEnv
from agent.slot_policy import SlotPolicy
from assets.rooms_2d.room_2d_generator import generate_room

MATERIAL_ID = 5  # wood_50mm_thick: empty rooms start below the C50 target
FOAM_SIZE = 0.25 # [M]
MIC_COUNT = 3
MAX_FREQ = 1000

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output" / "agent"


class PolicyStatsCallback(BaseCallback):
    """Logs how decided the policy is, independent of how many slots the rooms have.

    custom/entropy_norm  entropy / ln(free slots): 1 = uniform random choice, 0 = always the same slot
    custom/max_prob      mean probability of the favourite slot
    custom/random_prob   the same for a uniform random choice (1 / free slots)
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer
        obs = torch.as_tensor(buffer.observations.reshape(-1, *buffer.obs_shape), device=self.model.device)
        masks = buffer.action_masks.reshape(-1, buffer.action_masks.shape[-1])
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs, action_masks=masks)
            entropy = dist.entropy().cpu().numpy()
            max_prob = dist.distribution.probs.max(dim=1).values.cpu().numpy()
        free_slots = masks.sum(axis=1)
        self.logger.record("custom/entropy_norm", float(np.mean(entropy / np.log(np.maximum(free_slots, 2)))))
        self.logger.record("custom/max_prob", float(np.mean(max_prob)))
        self.logger.record("custom/random_prob", float(np.mean(1.0 / free_slots)))


class SlotPPO(MaskablePPO):
    """MaskablePPO that keeps the critic's own learning rate: SB3 sets the same rate in every parameter group
    before each update, here every group is then multiplied by its "lr_scale" (see SlotPolicy.make_optimizer)."""

    def _update_learning_rate(self, optimizers) -> None:
        super()._update_learning_rate(optimizers)
        for group in self.policy.optimizer.param_groups:
            group["lr"] *= group.get("lr_scale", 1.0)


def make_env(output_dir: Path, room_size: tuple[float, float]) -> Monitor:
    env = FoamEnv(material=MATERIAL_ID, foam_size=FOAM_SIZE, mic_count=MIC_COUNT, max_freq=MAX_FREQ,
                  room_size=room_size)
    # Monitor logs episode reward and length (= foams used), plus success flag
    return Monitor(env, filename=str(output_dir / "monitor"), info_keywords=("success",))


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO agent for foam placement")
    parser.add_argument("--steps", type=int, default=30_000, help="Total agent steps (1 step = 1 simulation), "
                                                                  "multiple of 256 to end exactly")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a saved model .zip to continue from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (also replaces it when resuming)")
    parser.add_argument("--room-min", type=float, default=6.0, help="Smallest room side [m]")
    parser.add_argument("--room-max", type=float, default=12.0, help="Largest room side [m]")
    parser.add_argument("--run", type=str, default="default", help="Run name, output goes to output/agent/<run>")
    parser.add_argument("--shape", type=str, default="any", help="Room shape, 'any' draws L, C, T or S")
    parser.add_argument("--critic-lr-scale", type=float, default=None,
                        help="Critic learning rate = --lr * this (default 1, when resuming: what the model had)")
    args = parser.parse_args()

    # the environment draws a random shape per room, here it can be fixed without touching foam_env
    if args.shape != "any":
        foam_env.generate_room = functools.partial(generate_room, shape_type=args.shape)

    output_dir = OUTPUT_ROOT / args.run
    output_dir.mkdir(parents=True, exist_ok=True)
    ti.init(arch=ti.gpu, log_level=ti.ERROR)

    env = make_env(output_dir, (args.room_min, args.room_max))

    if args.resume:
        model = SlotPPO.load(args.resume, env=env, device="cpu", verbose=1,
                             custom_objects={"learning_rate": args.lr, "lr_schedule": lambda _: args.lr})
        if args.critic_lr_scale is not None and args.critic_lr_scale != model.policy.critic_lr_scale:
            model.policy.critic_lr_scale = args.critic_lr_scale
            model.policy_kwargs["critic_lr_scale"] = args.critic_lr_scale
            model.policy.optimizer = model.policy.make_optimizer(args.lr)
    else:
        model = SlotPPO(
            SlotPolicy,
            env,
            policy_kwargs={"critic_lr_scale": args.critic_lr_scale or 1.0},
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=1.0,
            learning_rate=args.lr,
            seed=args.seed,
            verbose=1,
            device="cpu",
        )

    checkpoint = CheckpointCallback(
        save_freq=2_000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_foam",
    )

    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint, PolicyStatsCallback()],
        reset_num_timesteps=args.resume is None,
        progress_bar=True
    )

    final_path = output_dir / "ppo_foam_final"
    model.save(final_path)
    print(f"Saved model: {final_path}.zip")


if __name__ == "__main__":
    main()
