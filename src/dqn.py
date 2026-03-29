"""Minimal DQN training loop using PyTorch (no stable-baselines3)."""

from __future__ import annotations

import argparse
import os
import random
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from crowd_env import CrowdNavEnv
from multi_env import MultiScenarioEnv, VariablePedestrianEnv
from wrappers import ObservationStackWrapper


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = 1.0 if done else 0.0
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=batch_size)
        batch_obs = torch.as_tensor(self.obs[idxs], device=self.device)
        batch_next_obs = torch.as_tensor(self.next_obs[idxs], device=self.device)
        batch_actions = torch.as_tensor(self.actions[idxs], device=self.device)
        batch_rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        batch_dones = torch.as_tensor(self.dones[idxs], device=self.device)
        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def make_base_env(args: argparse.Namespace, seed: int, render_mode: str | None = None):
    if args.multi_scenario:
        return MultiScenarioEnv(
            ped_count_range=(args.pedestrians_min, args.pedestrians_max),
            speed_range=(args.ped_speed_min, args.ped_speed_max),
            max_steps=args.max_steps,
            seed=seed,
            render_mode=render_mode,
        )
    if args.vary_pedestrians:
        return VariablePedestrianEnv(
            scenario_id=args.scenario,
            ped_count_range=(args.pedestrians_min, args.pedestrians_max),
            speed_range=(args.ped_speed_min, args.ped_speed_max),
            max_steps=args.max_steps,
            seed=seed,
            render_mode=render_mode,
        )
    return CrowdNavEnv(
        scenario_id=args.scenario,
        num_pedestrians=args.pedestrians,
        max_steps=args.max_steps,
        seed=seed,
        render_mode=render_mode,
    )


def make_env(args: argparse.Namespace, seed: int, render_mode: str | None = None):
    env = make_base_env(args, seed=seed, render_mode=render_mode)
    if args.frame_stack > 1:
        env = ObservationStackWrapper(env, stack_size=args.frame_stack)
    return env


def save_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    run_config = {
        "algo": "dqn",
        "scenario": args.scenario,
        "pedestrians": args.pedestrians,
        "frame_stack": args.frame_stack,
        "multi_scenario": args.multi_scenario,
        "vary_pedestrians": args.vary_pedestrians,
        "pedestrians_min": args.pedestrians_min,
        "pedestrians_max": args.pedestrians_max,
        "ped_speed_min": args.ped_speed_min,
        "ped_speed_max": args.ped_speed_max,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def select_action(q_net: QNetwork, obs: np.ndarray, action_dim: int, epsilon: float, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(action_dim)
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    q_values = q_net(obs_t)
    return int(torch.argmax(q_values, dim=1).item())


def linear_schedule(start: float, end: float, current_step: int, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    fraction = min(1.0, current_step / float(decay_steps))
    return start + fraction * (end - start)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    env = make_env(args, seed=args.seed, render_mode=None)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    q_net = QNetwork(obs_dim, action_dim).to(device)
    target_net = QNetwork(obs_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.learning_rate)

    buffer = ReplayBuffer(args.buffer_size, obs_dim, device)

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "dqn.pt"
    save_run_config(run_dir, args)

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_steps = 0
    best_reward = -float("inf")

    progress = trange(1, args.total_steps + 1, desc="Training", dynamic_ncols=True)

    for step in progress:
        epsilon = linear_schedule(args.eps_start, args.eps_end, step, args.eps_decay_steps)
        action = select_action(q_net, obs, action_dim, epsilon, device)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_reward += reward
        episode_steps += 1

        if step >= args.warmup and len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            loss = compute_td_loss(q_net, target_net, optimizer, batch, args.gamma, args.max_grad_norm)

        if step % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            if episode_reward > best_reward:
                best_reward = episode_reward
            max_steps_hint = getattr(env.unwrapped, "max_steps", None)
            if step % args.log_interval <= (max_steps_hint or args.max_steps):
                progress.write(
                    f"Step {step:>7d} | Episode {info.get('steps', episode_steps):>5d} | "
                    f"Reward {episode_reward:>8.1f} | Epsilon {epsilon:>.3f} | Buffer {len(buffer):>6d}"
                )
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        if step % args.log_interval == 0 and not done:
            progress.set_postfix(
                eps=f"{epsilon:.3f}", reward=f"{episode_reward:>8.1f}", buffer=len(buffer)
            )
            progress.write(
                f"Step {step:>7d} | Episode step {episode_steps:>5d} | "
                f"Reward so far {episode_reward:>8.1f} | Epsilon {epsilon:>.3f} | Buffer {len(buffer):>6d}"
            )

        if step % args.save_interval == 0:
            torch.save({"q_net": q_net.state_dict(), "step": step}, model_path)
            progress.write(f"Saved checkpoint to {model_path}")

    torch.save({"q_net": q_net.state_dict(), "step": args.total_steps}, model_path)
    progress.write(f"Training complete. Final model saved to {model_path}")
    env.close()
    progress.close()


def compute_td_loss(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, ...],
    gamma: float,
    max_grad_norm: float,
) -> torch.Tensor:
    obs, actions, rewards, next_obs, dones = batch

    q_values = q_net(obs).gather(1, actions.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_obs).max(dim=1)[0]
        target = rewards + gamma * (1.0 - dones) * next_q

    loss = F.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_grad_norm)
    optimizer.step()
    return loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DQN trainer with PyTorch")
    p.add_argument("--scenario", type=str, default="airport", help="Scenario id")
    p.add_argument("--pedestrians", type=int, default=12, help="Number of pedestrians")
    p.add_argument("--multi-scenario", action="store_true", help="Train across all scenarios with random ped count/speed")
    p.add_argument("--vary-pedestrians", action="store_true", help="Randomize pedestrian count/speed each episode in a single scenario")
    p.add_argument("--pedestrians-min", type=int, default=6, help="Min pedestrian count when varying")
    p.add_argument("--pedestrians-max", type=int, default=20, help="Max pedestrian count when varying")
    p.add_argument("--ped-speed-min", type=float, default=1.0, help="Min pedestrian speed multiplier when varying")
    p.add_argument("--ped-speed-max", type=float, default=2.5, help="Max pedestrian speed multiplier when varying")
    p.add_argument("--total-steps", type=int, default=200_000, help="Total training steps")
    p.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    p.add_argument("--frame-stack", type=int, default=4, help="Observation frame stack")
    p.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer capacity")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate")
    p.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    p.add_argument("--eps-start", type=float, default=1.0, help="Starting epsilon")
    p.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon")
    p.add_argument("--eps-decay-steps", type=int, default=120_000, help="Steps over which epsilon decays")
    p.add_argument("--warmup", type=int, default=5_000, help="Steps before starting updates")
    p.add_argument("--target-update", type=int, default=2_000, help="Steps between target network syncs")
    p.add_argument("--max-grad-norm", type=float, default=5.0, help="Gradient clipping norm (0 to disable)")
    p.add_argument("--log-interval", type=int, default=5_000, help="Steps between console logs")
    p.add_argument("--save-interval", type=int, default=25_000, help="Steps between checkpoints")
    p.add_argument("--output-dir", type=str, default="training_output/dqn", help="Where to save model")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()