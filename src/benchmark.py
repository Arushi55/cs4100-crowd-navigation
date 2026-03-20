"""Benchmark: test trained agent against varying pedestrian count and speed."""

from __future__ import annotations

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


def run_benchmark(
    model_path: str,
    scenario: str = "airport",
    ped_counts: list[int] | None = None,
    speed_multipliers: list[float] | None = None,
    episodes: int = 20,
    max_steps: int = 1200,
    seed: int = 99,
) -> None:
    if ped_counts is None:
        ped_counts = [4, 8, 12, 16, 24, 32]
    if speed_multipliers is None:
        speed_multipliers = [1.0, 1.5, 2.0, 3.0]

    model_type, frame_stack = load_model_config(Path(model_path))
    model = ALGORITHMS[model_type].load(model_path)

    print("=" * 80)
    print("  Crowd Navigation — Scalability Benchmark")
    print("=" * 80)
    print(f"  Model:    {model_path}")
    print(f"  Algo:     {model_type.upper()}  |  Frame stack: {frame_stack}")
    print(f"  Scenario: {scenario}  |  Episodes per config: {episodes}")
    print(f"  Max steps: {max_steps}")
    print("=" * 80)

    # ── Sweep pedestrian count (default speed) ──
    print(f"\n{'─' * 80}")
    print("  Part 1: Scaling Pedestrian Count  (speed = 1.0×)")
    print(f"{'─' * 80}")
    print(
        f"  {'Peds':>6s} | {'Success':>8s} | {'Avg Steps':>10s} | "
        f"{'Avg Reward':>11s} | {'Collisions':>10s} | {'NearMiss':>8s} | {'Slowdown':>8s}"
    )
    print(f"  {'─'*6}-+-{'─'*8}-+-{'─'*10}-+-{'─'*11}-+-{'─'*10}-+-{'─'*8}-+-{'─'*8}")

    count_results = []
    for n_peds in ped_counts:
        stats = _evaluate(model, scenario, n_peds, 1.0, episodes, max_steps, seed, frame_stack)
        count_results.append((n_peds, stats))
        print(
            f"  {n_peds:>6d} | {stats['success_rate']:>7.0f}% | "
            f"{stats['avg_steps']:>10.0f} | {stats['avg_reward']:>11.1f} | "
            f"{stats['avg_collisions']:>10.1f} | {stats['avg_near_misses']:>8.1f} | "
            f"{stats['avg_pedestrian_slowdown']:>8.1f}"
        )

    # ── Sweep speed (8 pedestrians) ──
    print(f"\n{'─' * 80}")
    print("  Part 2: Scaling Pedestrian Speed  (count = 8)")
    print(f"{'─' * 80}")
    print(
        f"  {'Speed':>6s} | {'Success':>8s} | {'Avg Steps':>10s} | "
        f"{'Avg Reward':>11s} | {'Collisions':>10s} | {'NearMiss':>8s} | {'Slowdown':>8s}"
    )
    print(f"  {'─'*6}-+-{'─'*8}-+-{'─'*10}-+-{'─'*11}-+-{'─'*10}-+-{'─'*8}-+-{'─'*8}")

    speed_results = []
    for speed_mult in speed_multipliers:
        stats = _evaluate(model, scenario, 8, speed_mult, episodes, max_steps, seed, frame_stack)
        speed_results.append((speed_mult, stats))
        print(
            f"  {speed_mult:>5.1f}× | {stats['success_rate']:>7.0f}% | "
            f"{stats['avg_steps']:>10.0f} | {stats['avg_reward']:>11.1f} | "
            f"{stats['avg_collisions']:>10.1f} | {stats['avg_near_misses']:>8.1f} | "
            f"{stats['avg_pedestrian_slowdown']:>8.1f}"
        )

    # ── Combined stress test ──
    print(f"\n{'─' * 80}")
    print("  Part 3: Combined Stress Test  (high count + high speed)")
    print(f"{'─' * 80}")
    stress_configs = [
        (16, 1.5),
        (16, 2.0),
        (24, 1.5),
        (24, 2.0),
        (32, 2.0),
    ]
    print(
        f"  {'Peds':>6s} {'Speed':>6s} | {'Success':>8s} | {'Avg Steps':>10s} | "
        f"{'Avg Reward':>11s} | {'Collisions':>10s} | {'NearMiss':>8s} | {'Slowdown':>8s}"
    )
    print(f"  {'─'*6}-{'─'*6}-+-{'─'*8}-+-{'─'*10}-+-{'─'*11}-+-{'─'*10}-+-{'─'*8}-+-{'─'*8}")

    for n_peds, speed_mult in stress_configs:
        stats = _evaluate(model, scenario, n_peds, speed_mult, episodes, max_steps, seed, frame_stack)
        print(
            f"  {n_peds:>6d} {speed_mult:>5.1f}× | {stats['success_rate']:>7.0f}% | "
            f"{stats['avg_steps']:>10.0f} | {stats['avg_reward']:>11.1f} | "
            f"{stats['avg_collisions']:>10.1f} | {stats['avg_near_misses']:>8.1f} | "
            f"{stats['avg_pedestrian_slowdown']:>8.1f}"
        )

    print(f"\n{'=' * 80}")
    print("  Benchmark complete!")
    print(f"{'=' * 80}")


def load_model_config(model_path: Path) -> tuple[str, int]:
    candidate_paths = [
        model_path.parent / "run_config.json",
        model_path.parent.parent / "run_config.json",
    ]
    for path in candidate_paths:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("algo", "dqn"), int(data.get("frame_stack", 1))
    return "dqn", 1


def _evaluate(
    model,
    scenario: str,
    n_peds: int,
    speed_mult: float,
    episodes: int,
    max_steps: int,
    seed: int,
    frame_stack: int,
) -> dict:
    env = CrowdNavEnv(
        scenario_id=scenario,
        num_pedestrians=n_peds,
        max_steps=max_steps,
        seed=seed,
        render_mode=None,
    )
    if frame_stack > 1:
        env = ObservationStackWrapper(env, stack_size=frame_stack)

    rewards, lengths, successes, collisions = [], [], [], []
    near_misses, personal_space_intrusions = [], []
    pedestrian_slowdowns, blocking_pressures = [], []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)

        # Apply speed multiplier to all pedestrians
        if speed_mult != 1.0:
            for ped in env.unwrapped.pedestrians:
                ped.desired_speed *= speed_mult
                ped.max_speed *= speed_mult

        total_reward = 0.0
        total_collisions = 0
        total_near_misses = 0
        total_personal_space_intrusions = 0
        total_pedestrian_slowdown = 0.0
        total_blocking_pressure = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            total_collisions += info.get("collisions", 0)
            total_near_misses += info.get("near_misses", 0)
            total_personal_space_intrusions += info.get("personal_space_intrusions", 0)
            total_pedestrian_slowdown += info.get("pedestrian_slowdown", 0.0)
            total_blocking_pressure += info.get("blocking_pressure", 0.0)
            done = terminated or truncated

        rewards.append(total_reward)
        lengths.append(info.get("steps", 0))
        successes.append(terminated)
        collisions.append(total_collisions)
        near_misses.append(total_near_misses)
        personal_space_intrusions.append(total_personal_space_intrusions)
        pedestrian_slowdowns.append(total_pedestrian_slowdown)
        blocking_pressures.append(total_blocking_pressure)

    env.close()

    return {
        "success_rate": np.mean(successes) * 100,
        "avg_reward": np.mean(rewards),
        "avg_steps": np.mean(lengths),
        "avg_collisions": np.mean(collisions),
        "avg_near_misses": np.mean(near_misses),
        "avg_personal_space_intrusions": np.mean(personal_space_intrusions),
        "avg_pedestrian_slowdown": np.mean(pedestrian_slowdowns),
        "avg_blocking_pressure": np.mean(blocking_pressures),
    }


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    default_model = repo_root / "training_output" / "best_model" / "best_model.zip"

    model_path = sys.argv[1] if len(sys.argv) > 1 else str(default_model)

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Train first:  python train.py")
        sys.exit(1)

    for scenario in ["airport", "home", "shopping_center"]:
        print(f"\n\n{'#' * 80}")
        print(f"  SCENARIO: {scenario.upper()}")
        print(f"{'#' * 80}")
        run_benchmark(model_path, scenario=scenario, episodes=15)
