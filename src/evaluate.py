"""Evaluate a trained crowd navigation agent with optional visual rendering."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO

from crowd_env import CrowdNavEnv
from wrappers import ObservationStackWrapper

ALGORITHMS = {
    "dqn": DQN,
    "ppo": PPO,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained crowd navigation agent")
    p.add_argument("--model", type=str, default=None,
                    help="Path to model .zip (default: training_output/best_model/best_model.zip)")
    p.add_argument("--algo", type=str, choices=sorted(ALGORITHMS), default=None,
                    help="Algorithm used to train the model (auto-detected when possible)")
    p.add_argument("--frame-stack", type=int, default=None,
                    help="Observation stack size used during training (auto-detected when possible)")
    p.add_argument("--scenario", type=str, default="airport",
                    help="Scenario id")
    p.add_argument("--pedestrians", type=int, default=8,
                    help="Number of pedestrians")
    p.add_argument("--episodes", type=int, default=10,
                    help="Number of episodes to run")
    p.add_argument("--max-steps", type=int, default=1000,
                    help="Max steps per episode")
    p.add_argument("--seed", type=int, default=123,
                    help="Random seed")
    p.add_argument("--deterministic", action="store_true", default=True,
                    help="Use deterministic actions (no exploration)")
    p.add_argument("--no-render", action="store_true",
                    help="Disable rendering (for batch evaluation)")
    p.add_argument("--fps", type=int, default=30,
                    help="Rendering FPS (lower = easier to watch)")
    return p.parse_args()


def load_run_config(model_path: Path) -> dict:
    candidate_paths = [
        model_path.parent / "run_config.json",
        model_path.parent.parent / "run_config.json",
    ]
    for path in candidate_paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # Resolve model path
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = repo_root / "training_output" / "best_model" / "best_model.zip"

    if not model_path.exists():
        # Try without .zip
        alt = model_path.with_suffix("")
        if alt.with_suffix(".zip").exists():
            model_path = alt.with_suffix(".zip")
        else:
            print(f"  Error: Model not found at {model_path}")
            print(f"  Train a model first:  python src/train.py")
            sys.exit(1)

    print("=" * 60)
    print("  Crowd Navigation — Agent Evaluation")
    print("=" * 60)
    run_config = load_run_config(model_path)
    algo_name = args.algo or run_config.get("algo", "dqn")
    frame_stack = args.frame_stack or int(run_config.get("frame_stack", 1))
    model_cls = ALGORITHMS[algo_name]

    print(f"  Model:       {model_path}")
    print(f"  Algorithm:   {algo_name.upper()}")
    print(f"  Frame stack: {frame_stack}")
    print(f"  Scenario:    {args.scenario}")
    print(f"  Pedestrians: {args.pedestrians}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Render:      {not args.no_render}")
    print("=" * 60)

    # Load model
    model = model_cls.load(str(model_path))

    # Create environment
    render_mode = None if args.no_render else "human"
    env = CrowdNavEnv(
        scenario_id=args.scenario,
        num_pedestrians=args.pedestrians,
        max_steps=args.max_steps,
        seed=args.seed,
        render_mode=render_mode,
    )
    if frame_stack > 1:
        env = ObservationStackWrapper(env, stack_size=frame_stack)

    # Patch render FPS
    env.metadata["render_fps"] = args.fps

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episode_collisions = []
    episode_near_misses = []
    episode_intrusions = []
    episode_slowdown = []
    episode_blocking = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0.0
        total_collisions = 0
        total_near_misses = 0
        total_intrusions = 0
        total_slowdown = 0.0
        total_blocking = 0.0
        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            total_collisions += info.get("collisions", 0)
            total_near_misses += info.get("near_misses", 0)
            total_intrusions += info.get("personal_space_intrusions", 0)
            total_slowdown += info.get("pedestrian_slowdown", 0.0)
            total_blocking += info.get("blocking_pressure", 0.0)
            step += 1
            done = terminated or truncated

            if render_mode:
                env.render()

        success = terminated  # reached goal
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        episode_successes.append(success)
        episode_collisions.append(total_collisions)
        episode_near_misses.append(total_near_misses)
        episode_intrusions.append(total_intrusions)
        episode_slowdown.append(total_slowdown)
        episode_blocking.append(total_blocking)

        status = "✓ GOAL" if success else "✗ timeout"
        dist = info.get("distance_to_goal", -1)
        print(
            f"  Episode {ep + 1:>3d}/{args.episodes} | "
            f"{status} | "
            f"Steps: {step:>4d} | "
            f"Reward: {total_reward:>8.1f} | "
            f"Dist: {dist:>6.1f} | "
            f"Collisions: {total_collisions} | "
            f"Near misses: {total_near_misses}"
        )

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    success_rate = np.mean(episode_successes) * 100
    print(f"  Success rate:    {success_rate:.1f}% ({sum(episode_successes)}/{args.episodes})")
    print(f"  Avg reward:      {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"  Avg steps:       {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f}")
    print(f"  Avg collisions:  {np.mean(episode_collisions):.1f} ± {np.std(episode_collisions):.1f}")
    print(f"  Avg near misses: {np.mean(episode_near_misses):.1f} ± {np.std(episode_near_misses):.1f}")
    print(f"  Avg intrusions:  {np.mean(episode_intrusions):.1f} ± {np.std(episode_intrusions):.1f}")
    print(f"  Avg slowdown:    {np.mean(episode_slowdown):.1f} ± {np.std(episode_slowdown):.1f}")
    print(f"  Avg blocking:    {np.mean(episode_blocking):.1f} ± {np.std(episode_blocking):.1f}")

    if episode_successes and any(episode_successes):
        success_idx = [i for i, s in enumerate(episode_successes) if s]
        print(f"\n  Successful episodes ({len(success_idx)}):")
        print(f"    Avg steps:     {np.mean([episode_lengths[i] for i in success_idx]):.0f}")
        print(f"    Avg reward:    {np.mean([episode_rewards[i] for i in success_idx]):.1f}")
        print(f"    Avg collisions: {np.mean([episode_collisions[i] for i in success_idx]):.1f}")
        print(f"    Avg near misses: {np.mean([episode_near_misses[i] for i in success_idx]):.1f}")
    print()


if __name__ == "__main__":
    main()
