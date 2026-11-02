# ==============================================================================
# File:         evaluate.py
# Description:  Comprehensive evaluation script comparing Random, Greedy, and
#               trained PPO Agent baselines across identical unseen test rooms.
#               Exports detailed evaluation metrics to JSON and CSV formats
#               with per-room checkpointing and resume support.
# Usage:
#               python -m agent.baselines.evaluate --model output/agent/ppo_dummy.zip --rooms 30
# ==============================================================================

import argparse
import os
import json
import csv
from pathlib import Path
import numpy as np
import taichi as ti
from tqdm import tqdm
from sb3_contrib import MaskablePPO

from agent.foam_env import FoamEnv

MATERIAL_ID = 6
FOAM_SIZE = 0.25
MIC_COUNT = 3
MAX_FREQ = 1000

TEST_SEED_START = 20000
TEST_SEED_COUNT = 30
MAX_FOAMS_PER_ROOM = 8

GREEDY_SLOT_STEP = 3

OUTPUT_DIR = Path("output/evaluation")


def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj


def save_room_checkpoint(room_results: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_results.json"
    csv_summary_path = output_dir / "evaluation_summary.csv"
    csv_history_path = output_dir / "evaluation_step_history.csv"

    existing_results = []
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []

    existing_seeds = {r["room_seed"] for r in existing_results}
    for res in room_results:
        seed = res["room_seed"]
        if seed in existing_seeds:
            existing_results = [r if r["room_seed"] != seed else res for r in existing_results]
        else:
            existing_results.append(res)

    safe_results = convert_to_native(existing_results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safe_results, f, indent=4)

    with open(csv_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "RoomSeed", "Success", "FoamsUsed", "FinalMeanC50_dB", "FinalDeficit_dB"])
        for res in safe_results:
            writer.writerow([
                res["method"],
                int(res["room_seed"]),
                bool(res["success"]),
                int(res["foams_used"]),
                round(float(res["final_mean_c50"]), 3),
                round(float(res["final_deficit"]), 3)
            ])

    with open(csv_history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "RoomSeed", "Step", "ChosenSlot", "MeanC50_dB", "Deficit_dB"])
        for res in safe_results:
            for h in res["history"]:
                writer.writerow([
                    res["method"],
                    int(res["room_seed"]),
                    int(h["step"]),
                    int(h["action"]) if h["action"] is not None else -1,
                    round(float(h["mean_c50"]), 3),
                    round(float(h["deficit"]), 3)
                ])


def run_random_baseline(env: FoamEnv, room_seed: int, max_foams: int, pbar_desc: str) -> dict:
    _, info = env.reset(options={"room_seed": room_seed})
    history = []

    history.append({
        "step": 0, "action": None,
        "mean_c50": float(np.mean(info["c50_empty"])),
        "deficit": float(info["deficit_empty"]),
    })

    with tqdm(total=max_foams, desc=pbar_desc, leave=True, unit="foam", ncols=100) as pbar:
        terminated = False
        while not terminated and len(env.chosen_foams) < max_foams:
            masks = env.action_masks()
            available = np.flatnonzero(masks)
            if len(available) == 0:
                break

            action = int(env.np_random.choice(available))
            _, _, terminated, _, step_info = env.step(action)

            history.append({
                "step": len(env.chosen_foams), "action": action,
                "mean_c50": float(np.mean(step_info["c50"])),
                "deficit": float(step_info["deficit"]),
            })
            pbar.update(1)
            pbar.set_postfix(deficit=f"{step_info['deficit']:.2f}dB")

        final_c50 = np.mean(env.c50_per_mic)
        if env.deficit == 0.0:
            if pbar.n < max_foams:
                pbar.update(max_foams - pbar.n)
            pbar.set_postfix(foams=f"{len(env.chosen_foams)}", C50=f"{final_c50:.2f}dB")
        else:
            pbar.set_postfix(foams=f"{len(env.chosen_foams)}", deficit=f"{env.deficit:.2f}dB")

    return {
        "method": "Random", "room_seed": room_seed,
        "success": bool(env.deficit == 0.0), "foams_used": len(env.chosen_foams),
        "final_mean_c50": float(np.mean(env.c50_per_mic)), "final_deficit": float(env.deficit),
        "history": history
    }


def run_greedy_baseline(env: FoamEnv, room_seed: int, max_foams: int, pbar_desc: str) -> dict:
    _, info = env.reset(options={"room_seed": room_seed})
    history = []

    history.append({
        "step": 0, "action": None,
        "mean_c50": float(np.mean(info["c50_empty"])),
        "deficit": float(info["deficit_empty"]),
    })

    initial_slots = len(np.flatnonzero(env.action_masks()))
    sampled_initial_slots = len(np.flatnonzero(env.action_masks())[::GREEDY_SLOT_STEP])
    total_simulations = sum(max(1, sampled_initial_slots - i) for i in range(max_foams))

    with tqdm(total=total_simulations, desc=pbar_desc, leave=True, unit="sim", ncols=100) as pbar:
        while len(env.chosen_foams) < max_foams and env.deficit > 0.0:
            masks = env.action_masks()
            available = np.flatnonzero(masks)
            if len(available) == 0:
                break

            sampled_available = available[::GREEDY_SLOT_STEP]

            best_action = None
            best_deficit = float('inf')

            saved_foams = list(env.chosen_foams)
            saved_c50 = env.c50_per_mic.copy()
            saved_deficit = env.deficit

            for slot in sampled_available:
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

            if best_action is None:
                break

            env.chosen_foams = list(saved_foams)
            env.c50_per_mic = saved_c50.copy()
            env.deficit = saved_deficit
            _, _, _, _, step_info = env.step(best_action)

            history.append({
                "step": len(env.chosen_foams), "action": int(best_action),
                "mean_c50": float(np.mean(step_info["c50"])),
                "deficit": float(step_info["deficit"]),
            })

        final_c50 = np.mean(env.c50_per_mic)
        if env.deficit == 0.0:
            if pbar.n < total_simulations:
                pbar.update(total_simulations - pbar.n)
            pbar.set_postfix(foams=f"{len(env.chosen_foams)}", C50=f"{final_c50:.2f}dB")
        else:
            pbar.set_postfix(foams=f"{len(env.chosen_foams)}", deficit=f"{env.deficit:.2f}dB")

    return {
        "method": "Greedy", "room_seed": room_seed,
        "success": bool(env.deficit == 0.0), "foams_used": len(env.chosen_foams),
        "final_mean_c50": float(np.mean(env.c50_per_mic)), "final_deficit": float(env.deficit),
        "history": history
    }


def run_ppo_agent(env: FoamEnv, model: MaskablePPO, room_seed: int, max_foams: int, pbar_desc: str) -> dict:
    obs, info = env.reset(options={"room_seed": room_seed})
    history = []

    history.append({
        "step": 0, "action": None,
        "mean_c50": float(np.mean(info["c50_empty"])),
        "deficit": float(info["deficit_empty"]),
    })

    with tqdm(total=max_foams, desc=pbar_desc, leave=True, unit="foam", ncols=100) as pbar:
        terminated = False
        while not terminated and len(env.chosen_foams) < max_foams:
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, _, terminated, _, step_info = env.step(action)

            history.append({
                "step": len(env.chosen_foams), "action": int(action),
                "mean_c50": float(np.mean(step_info["c50"])),
                "deficit": float(step_info["deficit"]),
            })
            pbar.update(1)
            pbar.set_postfix(deficit=f"{step_info['deficit']:.2f}dB")

        final_c50 = np.mean(env.c50_per_mic)
        if env.deficit == 0.0:
            if pbar.n < max_foams:
                pbar.update(max_foams - pbar.n)
            pbar.set_postfix(foams=f"{len(env.chosen_foams)}", C50=f"{final_c50:.2f}dB")
        else:
            pbar.set_postfix(foams=f"{len(env.chosen_foams)}", deficit=f"{env.deficit:.2f}dB")

    return {
        "method": "PPO", "room_seed": room_seed,
        "success": bool(env.deficit == 0.0), "foams_used": len(env.chosen_foams),
        "final_mean_c50": float(np.mean(env.c50_per_mic)), "final_deficit": float(env.deficit),
        "history": history
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Random, Greedy, and PPO agent on test rooms")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained PPO model .zip")
    parser.add_argument("--rooms", type=int, default=TEST_SEED_COUNT, help="Number of rooms to evaluate")
    parser.add_argument("--seed_start", type=int, default=TEST_SEED_START, help="Starting room seed")
    parser.add_argument("--max_foams", type=int, default=MAX_FOAMS_PER_ROOM, help="Max foam count budget")
    args = parser.parse_args()

    ti.init(arch=ti.gpu, log_level=ti.ERROR)

    print(f"\nLoading PPO model from {args.model}...")
    model = MaskablePPO.load(args.model, device="cpu")
    env = FoamEnv(material=MATERIAL_ID, foam_size=FOAM_SIZE, mic_count=MIC_COUNT, max_freq=MAX_FREQ)

    json_path = OUTPUT_DIR / "evaluation_results.json"

    completed_seeds = set()
    all_results = []
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                seed_counts = {}
                for r in loaded:
                    s = r["room_seed"]
                    seed_counts[s] = seed_counts.get(s, 0) + 1
                completed_seeds = {s for s, count in seed_counts.items() if count >= 3}
                if completed_seeds:
                    print(
                        f"[Resume] Found saved results for seeds: {sorted(list(completed_seeds))}.")
                    all_results = [r for r in loaded if r["room_seed"] in completed_seeds]
        except Exception:
            completed_seeds = set()

    seeds = [args.seed_start + i for i in range(args.rooms)]
    print(
        f"Starting evaluation across {len(seeds)} rooms (seeds {seeds[0]}..{seeds[-1]}, Greedy stride: {GREEDY_SLOT_STEP})")

    for room_seed in seeds:
        if room_seed in completed_seeds:
            print(f"\n[Skipped] Room (Seed: {room_seed}) already evaluated.")
            continue

        print(f"\nEvaluating room (Seed: {room_seed})")

        res_ppo = run_ppo_agent(env, model, room_seed, args.max_foams, pbar_desc=f"  [PPO Agent]")
        res_rand = run_random_baseline(env, room_seed, args.max_foams, pbar_desc=f"  [Random]")
        res_greedy = run_greedy_baseline(env, room_seed, args.max_foams, pbar_desc=f"  [Greedy]")

        room_res = [res_rand, res_greedy, res_ppo]
        all_results.extend(room_res)

        save_room_checkpoint(room_res, OUTPUT_DIR)
        print(f" -> [Checkpoint Saved] Room {room_seed} stored to disk.")

    print("\n==================================================")
    print("                EVALUATION SUMMARY                ")
    print("==================================================")
    for method in ["Random", "Greedy", "PPO"]:
        m_results = [r for r in all_results if r["method"] == method]
        if m_results:
            acc = np.mean([r["success"] for r in m_results]) * 100
            avg_foams = np.mean([r["foams_used"] for r in m_results])
            avg_c50 = np.mean([r["final_mean_c50"] for r in m_results])
            avg_def = np.mean([r["final_deficit"] for r in m_results])

            print(
                f"[{method:6s}] Success Rate: {acc:5.1f}% | Avg Foams: {avg_foams:.2f} | Avg Mean C50: {avg_c50:.2f} dB | Avg Deficit: {avg_def:.2f} dB")
    print("==================================================")


if __name__ == "__main__":
    main()