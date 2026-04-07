"""Benchmark: test trained agent against varying pedestrian count and speed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from crowd_env import CrowdNavEnv
from wrappers import ObservationStackWrapper

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

    import torch
    from dqn import QNetwork

    model_path_obj = Path(model_path)
    run_config = load_run_config(model_path_obj)
    frame_stack = int(run_config.get("frame_stack", 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_env = CrowdNavEnv(
        scenario_id=scenario,
        num_pedestrians=ped_counts[0] if ped_counts else 8,
        max_steps=max_steps,
        seed=seed,
        render_mode=None,
    )
    if frame_stack > 1:
        tmp_env = ObservationStackWrapper(tmp_env, stack_size=frame_stack)
    obs_dim = int(np.prod(tmp_env.observation_space.shape))
    action_dim = tmp_env.action_space.n
    tmp_env.close()

    checkpoint = torch.load(model_path_obj, map_location=device)
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

    def action_fn(obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(q_net(obs_t), dim=1).item())

    print("=" * 80)
    print("  Crowd Navigation — Scalability Benchmark")
    print("=" * 80)
    print(f"  Model:    {model_path}")
    print(f"  Algo:     DQN (PyTorch)  |  Frame stack: {frame_stack}")
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
        stats = _evaluate(action_fn, scenario, n_peds, 1.0, episodes, max_steps, seed, frame_stack)
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
        stats = _evaluate(action_fn, scenario, 8, speed_mult, episodes, max_steps, seed, frame_stack)
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
        stats = _evaluate(action_fn, scenario, n_peds, speed_mult, episodes, max_steps, seed, frame_stack)
        print(
            f"  {n_peds:>6d} {speed_mult:>5.1f}× | {stats['success_rate']:>7.0f}% | "
            f"{stats['avg_steps']:>10.0f} | {stats['avg_reward']:>11.1f} | "
            f"{stats['avg_collisions']:>10.1f} | {stats['avg_near_misses']:>8.1f} | "
            f"{stats['avg_pedestrian_slowdown']:>8.1f}"
        )

    print(f"\n{'=' * 80}")
    print("  Benchmark complete!")
    print(f"{'=' * 80}")


def load_run_config(model_path: Path) -> dict:
    candidate_paths = [
        model_path.parent / "run_config.json",
        model_path.parent.parent / "run_config.json",
    ]
    for path in candidate_paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _evaluate(
    action_fn,
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
            action = action_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
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
    default_model = repo_root / "training_output" / "dqn" / "dqn.pt"

    model_path = sys.argv[1] if len(sys.argv) > 1 else str(default_model)

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Train first:  python dqn.py")
        sys.exit(1)

    for scenario in ["airport", "home", "shopping_center"]:
        print(f"\n\n{'#' * 80}")
        print(f"  SCENARIO: {scenario.upper()}")
        print(f"{'#' * 80}")
        run_benchmark(model_path, scenario=scenario, episodes=15)
