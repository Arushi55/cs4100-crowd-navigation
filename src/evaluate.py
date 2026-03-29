"""Evaluate a trained DQN agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from crowd_env import CrowdNavEnv
from wrappers import ObservationStackWrapper
from dqn import QNetwork


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a DQN agent")
    p.add_argument("--model", type=str, default=None, help="Path to checkpoint (.pt)")
    p.add_argument("--frame-stack", type=int, default=None, help="Override frame stack (defaults to run_config)")
    p.add_argument("--scenario", type=str, default=None, help="Scenario id (defaults to run_config)")
    p.add_argument("--pedestrians", type=int, default=None, help="Pedestrian count (defaults to run_config or 12)")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    p.add_argument("--max-steps", type=int, default=None, help="Max steps per episode (defaults to run_config)")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--no-render", action="store_true", help="Disable rendering")
    p.add_argument("--fps", type=int, default=30, help="Rendering FPS")
    return p.parse_args()


def load_run_config(model_path: Path) -> dict:
    for candidate in [model_path.parent / "run_config.json", model_path.parent.parent / "run_config.json"]:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


@torch.no_grad()
def greedy_action(q_net: QNetwork, obs: np.ndarray, device: torch.device) -> int:
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    return int(torch.argmax(q_net(obs_t), dim=1).item())


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_root = Path(__file__).resolve().parent.parent
    model_path = Path(args.model) if args.model else repo_root / "training_output" / "dqn" / "dqn.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Train first: python src/dqn.py")
        sys.exit(1)

    run_config = load_run_config(model_path)
    frame_stack = args.frame_stack or int(run_config.get("frame_stack", 1))
    scenario = args.scenario or run_config.get("scenario", "airport")
    pedestrians = args.pedestrians or int(run_config.get("pedestrians", 12))
    max_steps = args.max_steps or int(run_config.get("max_steps", 1000))

    render_mode = None if args.no_render else "human"
    env = CrowdNavEnv(
        scenario_id=scenario,
        num_pedestrians=pedestrians,
        max_steps=max_steps,
        seed=args.seed,
        render_mode=render_mode,
    )
    if frame_stack > 1:
        env = ObservationStackWrapper(env, stack_size=frame_stack)
    env.metadata["render_fps"] = args.fps

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    ckpt = torch.load(model_path, map_location=device)
    q_net = QNetwork(obs_dim, action_dim).to(device)
    q_net.load_state_dict(ckpt["q_net"])
    q_net.eval()

    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episode_collisions = []
    episode_near_misses = []
    episode_intrusions = []
    episode_slowdown = []
    episode_blocking = []

    print("=" * 60)
    print("  Crowd Navigation — DQN Evaluation")
    print("=" * 60)
    print(f"  Model:       {model_path}")
    print(f"  Frame stack: {frame_stack}")
    print(f"  Scenario:    {scenario}")
    print(f"  Pedestrians: {pedestrians}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Render:      {not args.no_render}")
    print("=" * 60)

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
            action = greedy_action(q_net, obs, device)
            obs, reward, terminated, truncated, info = env.step(action)
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

        success = terminated
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
            f"  Episode {ep + 1:>3d}/{args.episodes} | {status} | "
            f"Steps: {step:>4d} | Reward: {total_reward:>8.1f} | Dist: {dist:>6.1f} | "
            f"Collisions: {total_collisions} | Near misses: {total_near_misses}"
        )

    env.close()

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
    print()


if __name__ == "__main__":
    main()
