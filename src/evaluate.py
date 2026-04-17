import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from crowd_env import ACTION_VECTORS, CrowdNavEnv
from wrappers import ObservationStackWrapper

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained crowd navigation agent")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt)",
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
    p.add_argument("--no-render", action="store_true", help="Disable rendering (for batch evaluation)")
    p.add_argument("--fps", type=int, default=30, help="Rendering FPS (lower = easier to watch)")
    p.add_argument("--save-metrics", type=str, default=None, help="Optional path to write per-episode CSV metrics")
    return p.parse_args()


def load_run_config(model_path):
    candidate_paths = [
        model_path.parent / "run_config.json",
        model_path.parent.parent / "run_config.json",
    ]
    for path in candidate_paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _parse_hidden_sizes(raw, default = (256, 256)):
    if isinstance(raw, str):
        values = [x.strip() for x in raw.split(",") if x.strip()]
        if not values:
            return default
        return tuple(int(x) for x in values)
    if isinstance(raw, (list, tuple)):
        values = [int(x) for x in raw]
        return tuple(values) if values else default
    return default


def _safe_div(num, denom):
    return float(num / denom) if denom > 1e-9 else 0.0


def _path_length(points):
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
    return float(total)


def _shortest_path_length(env_unwrapped):
    if env_unwrapped.nav_grid is None or env_unwrapped.robot is None or env_unwrapped.goal_pos is None:
        return float("nan")
    start = (env_unwrapped.robot.x, env_unwrapped.robot.y)
    goal = (float(env_unwrapped.goal_pos[0]), float(env_unwrapped.goal_pos[1]))
    path = env_unwrapped.nav_grid.find_path(start, goal)
    return _path_length([start, *path])


def _action_heading(action):
    if action < 0 or action >= len(ACTION_VECTORS):
        return None
    dx, dy = ACTION_VECTORS[action]
    if dx == 0 and dy == 0:
        return None
    return float(math.atan2(dy, dx))


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = repo_root / "training_output" / "dqn" / "dqn.pt"

    if not model_path.exists():
        alt_pt = model_path.with_suffix(".pt")
        if alt_pt.exists():
            model_path = alt_pt
        else:
            print(f"  Error: Model not found at {model_path}")
            print("  Train a model first:  python src/dqn.py")
            sys.exit(1)

    run_config = load_run_config(model_path)

    frame_stack = args.frame_stack if args.frame_stack is not None else int(run_config.get("frame_stack", 1))
    scenario = args.scenario or run_config.get("scenario", "airport")
    pedestrians = args.pedestrians if args.pedestrians is not None else int(run_config.get("pedestrians", 12))
    max_steps = args.max_steps if args.max_steps is not None else int(
        run_config.get("max_steps", run_config.get("max_episode_steps", 1000))
    )

    print("=" * 60)
    print("  Crowd Navigation — Agent Evaluation")
    print("=" * 60)
    print(f"  Model:       {model_path}")
    print("  Algorithm:   DQN (PyTorch)")
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

    try:
        import torch
        from dqn import QNetwork
    except Exception as exc:
        env.close()
        print(f"  Error loading PyTorch DQN dependencies: {exc}")
        print("  Install dependencies from requirements.txt and retry.")
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

    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episode_collisions = []
    episode_near_misses = []
    episode_intrusions = []
    episode_slowdown = []
    episode_blocking = []
    episode_wall_contacts = []
    episode_no_progress_max = []
    episode_path_length = []
    episode_shortest_path = []
    episode_path_efficiency = []
    episode_action_switch_rate = []
    episode_mean_turn_deg = []
    episode_max_turn_deg = []
    episode_rows = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        env_unwrapped = env.unwrapped
        prev_robot_x = float(env_unwrapped.robot.x) if env_unwrapped.robot is not None else float("nan")
        prev_robot_y = float(env_unwrapped.robot.y) if env_unwrapped.robot is not None else float("nan")
        shortest_path = _shortest_path_length(env_unwrapped)
        actual_path = 0.0
        action_switches = 0
        prev_action = None
        prev_heading = None
        turn_angles_deg = []

        total_reward = 0.0
        total_collisions = 0
        total_near_misses = 0
        total_intrusions = 0
        total_slowdown = 0.0
        total_blocking = 0.0
        total_wall_contacts = 0
        max_no_progress = 0
        done = False
        step = 0

        while not done:
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            action = int(torch.argmax(q_net(obs_t), dim=1).item())

            if prev_action is not None and action != prev_action:
                action_switches += 1
            prev_action = action

            heading = _action_heading(action)
            if heading is not None:
                if prev_heading is not None:
                    delta = abs(math.atan2(math.sin(heading - prev_heading), math.cos(heading - prev_heading)))
                    turn_angles_deg.append(math.degrees(delta))
                prev_heading = heading

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_collisions += info.get("collisions", 0)
            total_near_misses += info.get("near_misses", 0)
            total_intrusions += info.get("personal_space_intrusions", 0)
            total_slowdown += info.get("pedestrian_slowdown", 0.0)
            total_blocking += info.get("blocking_pressure", 0.0)
            total_wall_contacts += int(info.get("wall_contacts", 0))
            max_no_progress = max(max_no_progress, int(info.get("no_progress_steps", 0)))

            if env_unwrapped.robot is not None and math.isfinite(prev_robot_x) and math.isfinite(prev_robot_y):
                curr_x = float(env_unwrapped.robot.x)
                curr_y = float(env_unwrapped.robot.y)
                actual_path += math.hypot(curr_x - prev_robot_x, curr_y - prev_robot_y)
                prev_robot_x = curr_x
                prev_robot_y = curr_y

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
        episode_wall_contacts.append(total_wall_contacts)
        episode_no_progress_max.append(max_no_progress)
        episode_path_length.append(actual_path)
        episode_shortest_path.append(shortest_path)
        path_eff = _safe_div(actual_path, shortest_path) if np.isfinite(shortest_path) else float("nan")
        episode_path_efficiency.append(path_eff)
        switch_rate = _safe_div(action_switches, max(1, step - 1))
        episode_action_switch_rate.append(switch_rate)
        mean_turn_deg = float(np.mean(turn_angles_deg)) if turn_angles_deg else 0.0
        max_turn_deg = float(np.max(turn_angles_deg)) if turn_angles_deg else 0.0
        episode_mean_turn_deg.append(mean_turn_deg)
        episode_max_turn_deg.append(max_turn_deg)

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

        if success:
            failure_mode = "success"
        elif info.get("user_quit", False):
            failure_mode = "user_quit"
        elif truncated:
            failure_mode = "timeout"
        else:
            failure_mode = "other"

        episode_rows.append(
            {
                "episode": ep + 1,
                "success": int(success),
                "failure_mode": failure_mode,
                "steps": step,
                "reward": float(total_reward),
                "distance_to_goal": float(dist),
                "collisions": int(total_collisions),
                "near_misses": int(total_near_misses),
                "intrusions": int(total_intrusions),
                "slowdown": float(total_slowdown),
                "blocking": float(total_blocking),
                "wall_contacts": int(total_wall_contacts),
                "max_no_progress_steps": int(max_no_progress),
                "collisions_per_100_steps": _safe_div(total_collisions * 100.0, step),
                "near_misses_per_100_steps": _safe_div(total_near_misses * 100.0, step),
                "intrusions_per_100_steps": _safe_div(total_intrusions * 100.0, step),
                "slowdown_per_100_steps": _safe_div(total_slowdown * 100.0, step),
                "blocking_per_100_steps": _safe_div(total_blocking * 100.0, step),
                "path_length": float(actual_path),
                "shortest_path_length": float(shortest_path) if np.isfinite(shortest_path) else float("nan"),
                "path_efficiency": float(path_eff) if np.isfinite(path_eff) else float("nan"),
                "action_switches": int(action_switches),
                "action_switch_rate": float(switch_rate),
                "mean_turn_deg": float(mean_turn_deg),
                "max_turn_deg": float(max_turn_deg),
                "scenario": scenario,
                "pedestrians": int(pedestrians),
                "frame_stack": int(frame_stack),
                "seed": int(args.seed),
                "model_path": str(model_path),
            }
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
    print(f"  Avg wall contacts: {np.mean(episode_wall_contacts):.1f} ± {np.std(episode_wall_contacts):.1f}")
    print(
        "  Avg max no-progress steps: "
        f"{np.mean(episode_no_progress_max):.1f} ± {np.std(episode_no_progress_max):.1f}"
    )
    if any(np.isfinite(x) for x in episode_path_efficiency):
        finite_eff = np.array([x for x in episode_path_efficiency if np.isfinite(x)], dtype=np.float64)
        print(f"  Avg path efficiency: {np.mean(finite_eff):.2f} ± {np.std(finite_eff):.2f}")
    print(
        "  Avg action switch rate: "
        f"{np.mean(episode_action_switch_rate):.3f} ± {np.std(episode_action_switch_rate):.3f}"
    )
    print(f"  Avg mean turn (deg): {np.mean(episode_mean_turn_deg):.1f} ± {np.std(episode_mean_turn_deg):.1f}")

    if episode_successes and any(episode_successes):
        success_idx = [i for i, s in enumerate(episode_successes) if s]
        print(f"\n  Successful episodes ({len(success_idx)}):")
        print(f"    Avg steps:     {np.mean([episode_lengths[i] for i in success_idx]):.0f}")
        print(f"    Avg reward:    {np.mean([episode_rewards[i] for i in success_idx]):.1f}")
        print(f"    Avg collisions: {np.mean([episode_collisions[i] for i in success_idx]):.1f}")
        print(f"    Avg near misses: {np.mean([episode_near_misses[i] for i in success_idx]):.1f}")
    print()

    if args.save_metrics:
        out_path = Path(args.save_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(episode_rows[0].keys()) if episode_rows else []
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_rows)
        print(f"Saved episode metrics CSV to: {out_path}")


if __name__ == "__main__":
    main()
