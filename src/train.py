"""Train a crowd navigation agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from crowd_env import CrowdNavEnv
from multi_env import MultiScenarioEnv, VariablePedestrianEnv
from wrappers import ObservationStackWrapper

ALGORITHMS = {
    "dqn": DQN,
    "ppo": PPO,
}


# ---------------------------------------------------------------------------
# Custom callback: prints training progress and tracks success rate
# ---------------------------------------------------------------------------
class TrainingMetricsCallback(BaseCallback):
    """Log custom metrics during training."""

    def __init__(self, print_freq: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_successes: list[bool] = []

    def _on_step(self) -> bool:
        # Collect episode stats from Monitor wrapper
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                success = bool(info.get("is_success", False))
                self.episode_successes.append(success)

        if self.num_timesteps % self.print_freq == 0 and len(self.episode_rewards) > 0:
            recent_n = min(20, len(self.episode_rewards))
            avg_reward = np.mean(self.episode_rewards[-recent_n:])
            avg_length = np.mean(self.episode_lengths[-recent_n:])
            success_rate = np.mean(self.episode_successes[-recent_n:]) * 100

            print(
                f"  Step {self.num_timesteps:>8d} | "
                f"Avg reward: {avg_reward:>8.1f} | "
                f"Avg length: {avg_length:>6.0f} | "
                f"Success: {success_rate:>5.1f}% | "
                f"Episodes: {len(self.episode_rewards)}"
            )

            # Log to tensorboard
            self.logger.record("custom/avg_reward_20ep", avg_reward)
            self.logger.record("custom/avg_length_20ep", avg_length)
            self.logger.record("custom/success_rate_20ep", success_rate)

        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a crowd navigation agent")
    p.add_argument("--algo", type=str, choices=sorted(ALGORITHMS), default="ppo",
                    help="RL algorithm to use")
    p.add_argument("--scenario", type=str, default="airport",
                    help="Scenario id (home, airport, shopping_center)")
    p.add_argument("--pedestrians", type=int, default=8,
                    help="Number of pedestrians")
    p.add_argument("--total-timesteps", type=int, default=500_000,
                    help="Total training timesteps")
    p.add_argument("--max-episode-steps", type=int, default=1000,
                    help="Max steps per episode before truncation")
    p.add_argument("--learning-rate", type=float, default=1e-4,
                    help="Learning rate")
    p.add_argument("--batch-size", type=int, default=128,
                    help="Batch size for training")
    p.add_argument("--frame-stack", type=int, default=4,
                    help="Number of consecutive observations to stack")
    p.add_argument("--num-envs", type=int, default=4,
                    help="Number of parallel training environments (PPO only)")
    p.add_argument("--buffer-size", type=int, default=100_000,
                    help="Replay buffer size (DQN only)")
    p.add_argument("--exploration-fraction", type=float, default=0.3,
                    help="Fraction of training for exploration decay (DQN only)")
    p.add_argument("--exploration-final-eps", type=float, default=0.05,
                    help="Final exploration epsilon (DQN only)")
    p.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor")
    p.add_argument("--n-steps", type=int, default=1024,
                    help="Rollout length per environment before PPO update")
    p.add_argument("--ent-coef", type=float, default=0.01,
                    help="Entropy coefficient (PPO only)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    p.add_argument("--eval-freq", type=int, default=10_000,
                    help="Evaluate every N steps")
    p.add_argument("--checkpoint-freq", type=int, default=25_000,
                    help="Save checkpoint every N steps")
    p.add_argument("--output-dir", type=str, default="training_output",
                    help="Directory for logs, checkpoints, and final model")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a saved model .zip to resume training from")
    p.add_argument("--multi-scenario", action="store_true",
                    help="Train across all scenarios with random ped count & speed")
    p.add_argument("--vary-pedestrians", action="store_true",
                    help="For a single scenario, randomize pedestrian count each episode")
    p.add_argument("--pedestrians-min", type=int, default=6,
                    help="Minimum pedestrian count when varying pedestrians")
    p.add_argument("--pedestrians-max", type=int, default=20,
                    help="Maximum pedestrian count when varying pedestrians")
    p.add_argument("--ped-speed-min", type=float, default=1.0,
                    help="Minimum pedestrian speed multiplier when using variable environments")
    p.add_argument("--ped-speed-max", type=float, default=1.0,
                    help="Maximum pedestrian speed multiplier when using variable environments")
    return p.parse_args()


def make_base_env(args: argparse.Namespace, seed: int, render_mode: str | None = None):
    if args.multi_scenario:
        return MultiScenarioEnv(
            max_steps=args.max_episode_steps,
            seed=seed,
            render_mode=render_mode,
        )
    if args.vary_pedestrians:
        return VariablePedestrianEnv(
            scenario_id=args.scenario,
            ped_count_range=(args.pedestrians_min, args.pedestrians_max),
            speed_range=(args.ped_speed_min, args.ped_speed_max),
            max_steps=args.max_episode_steps,
            seed=seed,
            render_mode=render_mode,
        )
    return CrowdNavEnv(
        scenario_id=args.scenario,
        num_pedestrians=args.pedestrians,
        max_steps=args.max_episode_steps,
        seed=seed,
        render_mode=render_mode,
    )


def make_monitored_env(
    args: argparse.Namespace,
    seed: int,
    render_mode: str | None = None,
):
    env = make_base_env(args, seed=seed, render_mode=render_mode)
    if args.frame_stack > 1:
        env = ObservationStackWrapper(env, stack_size=args.frame_stack)
    return Monitor(env)


def make_vec_env(
    args: argparse.Namespace,
    num_envs: int,
    seed_start: int,
    render_mode: str | None = None,
):
    return DummyVecEnv([
        (
            lambda env_seed=seed_start + idx: make_monitored_env(
                args,
                seed=env_seed,
                render_mode=render_mode,
            )
        )
        for idx in range(num_envs)
    ])


def save_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    run_config = {
        "algo": args.algo,
        "scenario": args.scenario,
        "pedestrians": args.pedestrians,
        "frame_stack": args.frame_stack,
        "num_envs": args.num_envs,
        "multi_scenario": args.multi_scenario,
        "vary_pedestrians": args.vary_pedestrians,
        "pedestrians_min": args.pedestrians_min,
        "pedestrians_max": args.pedestrians_max,
        "ped_speed_min": args.ped_speed_min,
        "ped_speed_max": args.ped_speed_max,
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    model_cls = ALGORITHMS[args.algo]
    if args.algo == "dqn" and args.num_envs != 1:
        print("  DQN works best here with a single environment; forcing num_envs=1.")
        args.num_envs = 1

    # Resolve output paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / args.output_dir
    log_dir = output_dir / "logs"
    checkpoint_dir = output_dir / "checkpoints"
    best_model_dir = output_dir / "best_model"
    final_model_path = output_dir / "final_model"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    save_run_config(output_dir, args)

    print("=" * 60)
    print(f"  Crowd Navigation — {args.algo.upper()} Training")
    print("=" * 60)
    print(f"  Algorithm:       {args.algo.upper()}")
    print(f"  Frame stack:     {args.frame_stack}")
    print(f"  Train envs:      {args.num_envs if args.algo == 'ppo' else 1}")
    if args.multi_scenario:
        print("  Mode:            MULTI-SCENARIO")
        print("  Scenarios:       airport, home, shopping_center")
        print("  Ped count:       6–20 (random per episode)")
        print("  Ped speed:       1.0×–2.5× (random per episode)")
    elif args.vary_pedestrians:
        print("  Mode:            SINGLE-SCENARIO VARIABLE CROWD")
        print(f"  Scenario:        {args.scenario}")
        print(f"  Ped count:       {args.pedestrians_min}–{args.pedestrians_max} (random per episode)")
        print(f"  Ped speed:       {args.ped_speed_min:.1f}×–{args.ped_speed_max:.1f}×")
    else:
        print(f"  Scenario:        {args.scenario}")
        print(f"  Pedestrians:     {args.pedestrians}")
    print(f"  Total steps:     {args.total_timesteps:,}")
    print(f"  Max ep. steps:   {args.max_episode_steps}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  Batch size:      {args.batch_size}")
    if args.algo == "dqn":
        print(f"  Buffer size:     {args.buffer_size:,}")
        print(f"  Exploration:     {args.exploration_fraction} → ε={args.exploration_final_eps}")
    else:
        print(f"  PPO n_steps:     {args.n_steps}")
        print(f"  Entropy coef:    {args.ent_coef}")
    print(f"  Seed:            {args.seed}")
    print(f"  Output:          {output_dir}")
    print("=" * 60)

    # ---- Create environments ----
    train_env_count = args.num_envs if args.algo == "ppo" else 1
    train_env = make_vec_env(args, num_envs=train_env_count, seed_start=args.seed)
    eval_env = make_vec_env(args, num_envs=1, seed_start=args.seed + 1000)

    # ---- Create or resume model ----
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        model = model_cls.load(
            args.resume,
            env=train_env,
            tensorboard_log=str(log_dir),
        )
    else:
        if args.algo == "dqn":
            model = DQN(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                learning_starts=1_000,
                target_update_interval=500,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=args.exploration_fraction,
                exploration_initial_eps=1.0,
                exploration_final_eps=args.exploration_final_eps,
                gamma=args.gamma,
                policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=str(log_dir),
                seed=args.seed,
                verbose=0,
            )
        else:
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                gae_lambda=0.95,
                ent_coef=args.ent_coef,
                clip_range=0.2,
                vf_coef=0.5,
                policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=str(log_dir),
                seed=args.seed,
                verbose=0,
            )

    # ---- Callbacks ----
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=f"{args.algo}_crowd",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    metrics_cb = TrainingMetricsCallback(print_freq=5_000)

    # ---- Train ----
    print("\n  Training started...\n")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        progress_bar=True,
    )

    # ---- Save final model ----
    model.save(str(final_model_path))
    print(f"\n  Final model saved to: {final_model_path}.zip")

    # ---- Summary ----
    if metrics_cb.episode_successes:
        total_episodes = len(metrics_cb.episode_successes)
        total_successes = sum(metrics_cb.episode_successes)
        print(f"\n  Training summary:")
        print(f"    Total episodes:  {total_episodes}")
        print(f"    Total successes: {total_successes}")
        print(f"    Overall rate:    {total_successes / total_episodes * 100:.1f}%")

        # Last 100 episodes
        last_n = min(100, total_episodes)
        recent_rate = np.mean(metrics_cb.episode_successes[-last_n:]) * 100
        recent_reward = np.mean(metrics_cb.episode_rewards[-last_n:])
        print(f"    Last {last_n} episodes:")
        print(f"      Success rate:  {recent_rate:.1f}%")
        print(f"      Avg reward:    {recent_reward:.1f}")

    print(f"\n  TensorBoard logs: tensorboard --logdir {log_dir}")
    print("  Done!")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
