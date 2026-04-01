"""Evaluate a trained crowd navigation agent with optional visual rendering."""

from __future__ import annotations

import argparse
import json
import sys
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
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (.zip for SB3, .pt for PyTorch DQN)",
    )
    p.add_argument(
        "--algo",
        type=str,
        choices=sorted(ALGORITHMS),
        default=None,
        help="Algorithm used to train the SB3 model (auto-detected when possible)",
    )
    p.add_argument(
        "--frame-stack",
        type=int,
        default=None,
        help="Observation stack size used during training (auto-detected when possible)",
    )
    p.add_argument("--scenario", type=str, default=None, help="Scenario id")
    p.add_argument("--pedestrians", type=int, default=None, help="Number of pedestrians")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    p.add_argument("--max-steps", type=int, default=None, help="Max steps per episode")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions for SB3 models",
    )
    p.add_argument("--no-render", action="store_true", help="Disable rendering (for batch evaluation)")
    p.add_argument("--fps", type=int, default=30, help="Rendering FPS (lower = easier to watch)")
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


def _parse_hidden_sizes(raw: object, default: tuple[int, ...] = (256, 256)) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = [x.strip() for x in raw.split(",") if x.strip()]
        if not values:
            return default
        return tuple(int(x) for x in values)
    if isinstance(raw, (list, tuple)):
        values = [int(x) for x in raw]
        return tuple(values) if values else default
    return default


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = repo_root / "training_output" / "best_model" / "best_model.zip"

    if not model_path.exists():
        alt_zip = model_path.with_suffix(".zip")
        alt_pt = model_path.with_suffix(".pt")
        if alt_zip.exists():
            model_path = alt_zip
        elif alt_pt.exists():
            model_path = alt_pt
        else:
            print(f"  Error: Model not found at {model_path}")
            print("  Train a model first:  python src/train.py  or  python src/dqn.py")
            sys.exit(1)

    run_config = load_run_config(model_path)

    frame_stack = args.frame_stack if args.frame_stack is not None else int(run_config.get("frame_stack", 1))
    scenario = args.scenario or run_config.get("scenario", "airport")
    pedestrians = args.pedestrians if args.pedestrians is not None else int(run_config.get("pedestrians", 12))
    max_steps = args.max_steps if args.max_steps is not None else int(
        run_config.get("max_steps", run_config.get("max_episode_steps", 1000))
    )

    is_torch_dqn_checkpoint = model_path.suffix == ".pt"
    algo_name = args.algo or run_config.get("algo", "dqn")

    print("=" * 60)
    print("  Crowd Navigation — Agent Evaluation")
    print("=" * 60)
    print(f"  Model:       {model_path}")
    if is_torch_dqn_checkpoint:
        print("  Algorithm:   DQN (PyTorch)")
    else:
        print(f"  Algorithm:   {algo_name.upper()}")
    print(f"  Frame stack: {frame_stack}")
    print(f"  Scenario:    {scenario}")
    print(f"  Pedestrians: {pedestrians}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Render:      {not args.no_render}")
    print("=" * 60)

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

    model = None
    q_net = None
    device = None

    if is_torch_dqn_checkpoint:
        try:
            import torch
            from dqn import QNetwork
        except Exception as exc:
            env.close()
            print(f"  Error loading PyTorch DQN dependencies: {exc}")
            print("  Install dependencies from src/requirements.txt and retry.")
            sys.exit(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = env.action_space.n
        checkpoint = torch.load(model_path, map_location=device)

        hidden_sizes = _parse_hidden_sizes(run_config.get("hidden_sizes", "256,256"))
        activation = str(run_config.get("hidden_activation", "relu"))
        dueling = bool(run_config.get("dueling_dqn", False))

        q_net = QNetwork(
            obs_dim,
            action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dueling=dueling,
        ).to(device)
        q_net.load_state_dict(checkpoint["q_net"])
        q_net.eval()
    else:
        model_cls = ALGORITHMS.get(algo_name)
        if model_cls is None:
            env.close()
            print(f"  Error: unsupported algorithm '{algo_name}'.")
            print(f"  Supported: {', '.join(sorted(ALGORITHMS.keys()))}")
            sys.exit(1)
        model = model_cls.load(str(model_path))

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
            if is_torch_dqn_checkpoint:
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                action = int(torch.argmax(q_net(obs_t), dim=1).item())
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                action = int(action)

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
            f"  Episode {ep + 1:>3d}/{args.episodes} | "
            f"{status} | "
            f"Steps: {step:>4d} | "
            f"Reward: {total_reward:>8.1f} | "
            f"Dist: {dist:>6.1f} | "
            f"Collisions: {total_collisions} | "
            f"Near misses: {total_near_misses}"
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
