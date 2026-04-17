import argparse
import collections
import os
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from crowd_env import CrowdNavEnv
from multi_env import MultiScenarioEnv, VariablePedestrianEnv
from wrappers import ObservationStackWrapper

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, device, prioritized = False, alpha = 0.6, n_step = 1, gamma = 0.99):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.prioritized = prioritized
        self.alpha = alpha
        self.epsilon = 1e-6
        self.priorities = np.zeros((capacity,), dtype=np.float32) if prioritized else None

        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = collections.deque(maxlen=n_step)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        if self.n_step == 1:
            self._store(obs, action, reward, next_obs, done)
            if done:
                self.n_step_buffer.clear()
            return

        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        if len(self.n_step_buffer) < self.n_step and not done:
            return

        # store main n-step contiguous transition
        self._store_n_step_transition()

        if done:
            while len(self.n_step_buffer) > 1:
                self._store_n_step_transition()

    def _store_n_step_transition(self):
        if not self.n_step_buffer:
            return

        reward = 0.0
        next_obs = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]

        for idx, transition in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * transition[2]

        first_obs, first_action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        self._store(first_obs, first_action, reward, next_obs, done)
        self.n_step_buffer.popleft()

    def _store(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = 1.0 if done else 0.0

        if self.prioritized:
            max_priority = self.priorities.max() if self.full or self.idx > 0 else 1.0
            self.priorities[self.idx] = max_priority

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size, beta = 0.4):
        max_idx = self.capacity if self.full else self.idx
        if max_idx == 0:
            raise ValueError("Trying to sample from empty buffer")

        if self.prioritized:
            p = self.priorities[:max_idx] + self.epsilon
            probs = p ** self.alpha
            probs /= probs.sum()
            idxs = np.random.choice(max_idx, size=batch_size, p=probs)
            weights = (max_idx * probs[idxs]) ** (-beta)
            weights = weights / weights.max()
            weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
        else:
            idxs = np.random.randint(0, max_idx, size=batch_size)
            weights = torch.ones((batch_size,), device=self.device, dtype=torch.float32)

        batch_obs = torch.as_tensor(self.obs[idxs], device=self.device)
        batch_next_obs = torch.as_tensor(self.next_obs[idxs], device=self.device)
        batch_actions = torch.as_tensor(self.actions[idxs], device=self.device)
        batch_rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        batch_dones = torch.as_tensor(self.dones[idxs], device=self.device)

        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, weights, idxs

    def update_priorities(self, idxs, priorities):
        if not self.prioritized:
            return
        self.priorities[idxs] = priorities


# Q-network
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes = (256, 256), activation = "relu", dueling = False):
        super().__init__()

        self.dueling = dueling
        if not dueling:
            layers = []
            last_size = obs_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(last_size, h))
                if activation.lower() == "relu":
                    layers.append(nn.ReLU())
                elif activation.lower() == "tanh":
                    layers.append(nn.Tanh())
                elif activation.lower() == "leaky_relu":
                    layers.append(nn.LeakyReLU())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
                last_size = h
            layers.append(nn.Linear(last_size, action_dim))
            self.net = nn.Sequential(*layers)
        else:
            fc_layers = []
            last_size = obs_dim
            for h in hidden_sizes:
                fc_layers.append(nn.Linear(last_size, h))
                if activation.lower() == "relu":
                    fc_layers.append(nn.ReLU())
                elif activation.lower() == "tanh":
                    fc_layers.append(nn.Tanh())
                elif activation.lower() == "leaky_relu":
                    fc_layers.append(nn.LeakyReLU())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
                last_size = h
            self.feature = nn.Sequential(*fc_layers)
            self.value_stream = nn.Sequential(nn.Linear(last_size, 1))
            self.adv_stream = nn.Sequential(nn.Linear(last_size, action_dim))

    def forward(self, x):
        if not self.dueling:
            return self.net(x)

        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.adv_stream(features)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        return value + advantage - advantage_mean


# Environment helpers
def _parse_anchor_counts(raw):
    text = str(raw).strip()
    if not text:
        return ()
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    # Preserve order but remove duplicates.
    seen = set()
    deduped = []
    for v in values:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return tuple(deduped)


def make_base_env(args, seed, render_mode = None):
    ped_count_anchors = _parse_anchor_counts(args.ped_count_anchors)
    if args.multi_scenario:
        return MultiScenarioEnv(
            ped_count_range=(args.pedestrians_min, args.pedestrians_max),
            ped_count_anchors=ped_count_anchors,
            ped_count_anchor_prob=args.ped_count_anchor_prob,
            ped_count_anchor_jitter=args.ped_count_anchor_jitter,
            speed_range=(args.ped_speed_min, args.ped_speed_max),
            max_steps=args.max_steps,
            seed=seed,
            render_mode=render_mode,
        )
    if args.vary_pedestrians:
        return VariablePedestrianEnv(
            scenario_id=args.scenario,
            ped_count_range=(args.pedestrians_min, args.pedestrians_max),
            ped_count_anchors=ped_count_anchors,
            ped_count_anchor_prob=args.ped_count_anchor_prob,
            ped_count_anchor_jitter=args.ped_count_anchor_jitter,
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


def make_env(args, seed, render_mode = None):
    env = make_base_env(args, seed=seed, render_mode=render_mode)
    if args.frame_stack > 1:
        env = ObservationStackWrapper(env, stack_size=args.frame_stack)
    return env


def save_run_config(output_dir, args):
    run_config = {
        "algo": "dqn",
        "scenario": args.scenario,
        "pedestrians": args.pedestrians,
        "frame_stack": args.frame_stack,
        "multi_scenario": args.multi_scenario,
        "vary_pedestrians": args.vary_pedestrians,
        "pedestrians_min": args.pedestrians_min,
        "pedestrians_max": args.pedestrians_max,
        "ped_count_anchors": args.ped_count_anchors,
        "ped_count_anchor_prob": args.ped_count_anchor_prob,
        "ped_count_anchor_jitter": args.ped_count_anchor_jitter,
        "ped_speed_min": args.ped_speed_min,
        "ped_speed_max": args.ped_speed_max,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "hidden_sizes": args.hidden_sizes,
        "hidden_activation": args.hidden_activation,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "loss_fn": args.loss_fn,
        "eps_schedule": args.eps_schedule,
        "prioritized": args.prioritized,
        "priority_alpha": args.priority_alpha,
        "priority_beta_start": args.priority_beta_start,
        "priority_beta_frames": args.priority_beta_frames,
        "n_step": args.n_step,
        "double_dqn": args.double_dqn,
        "dueling_dqn": args.dueling_dqn,
        "tau": args.tau,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )


# Training loop
@torch.no_grad()
def select_action(q_net, obs, action_dim, epsilon, device):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    q_values = q_net(obs_t)
    return int(torch.argmax(q_values, dim=1).item())


def linear_schedule(start, end, current_step, decay_steps):
    if decay_steps <= 0:
        return end
    fraction = min(1.0, current_step / float(decay_steps))
    return start + fraction * (end - start)


def exponential_schedule(start, end, current_step, decay_steps):
    if decay_steps <= 0:
        return end
    decay = float(current_step) / float(decay_steps)
    return end + (start - end) * np.exp(-5.0 * decay)


def epsilon_schedule(args, step):
    if args.eps_schedule == "linear":
        return linear_schedule(args.eps_start, args.eps_end, step, args.eps_decay_steps)
    if args.eps_schedule == "exponential":
        return exponential_schedule(args.eps_start, args.eps_end, step, args.eps_decay_steps)
    if args.eps_schedule == "constant":
        return args.eps_end
    if args.eps_schedule == "cosine":
        if args.eps_decay_steps <= 0:
            return args.eps_end
        fraction = min(1.0, step / float(args.eps_decay_steps))
        cosine = 0.5 * (1.0 + np.cos(np.pi * fraction))
        return args.eps_end + (args.eps_start - args.eps_end) * cosine
    raise ValueError(f"Unsupported eps schedule: {args.eps_schedule}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    env = make_env(args, seed=args.seed, render_mode=None)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(",") if x.strip())
    q_net = QNetwork(
        obs_dim,
        action_dim,
        hidden_sizes=hidden_sizes,
        activation=args.hidden_activation,
        dueling=args.dueling_dqn,
    ).to(device)
    target_net = QNetwork(
        obs_dim,
        action_dim,
        hidden_sizes=hidden_sizes,
        activation=args.hidden_activation,
        dueling=args.dueling_dqn,
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(q_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(q_net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(q_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    buffer = ReplayBuffer(
        args.buffer_size,
        obs_dim,
        device,
        prioritized=args.prioritized,
        alpha=args.priority_alpha,
        n_step=args.n_step,
        gamma=args.gamma,
    )

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
        epsilon = epsilon_schedule(args, step)
        action = select_action(q_net, obs, action_dim, epsilon, device)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_reward += reward
        episode_steps += 1

        if step >= args.warmup and len(buffer) >= args.batch_size:
            beta = min(1.0, args.priority_beta_start + step * (1.0 - args.priority_beta_start) / max(1, args.priority_beta_frames))
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, weights, idxs = buffer.sample(args.batch_size, beta=beta)
            loss, td_errors = compute_td_loss(
                q_net,
                target_net,
                optimizer,
                (batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones),
                args.gamma,
                args.max_grad_norm,
                args.loss_fn,
                double_dqn=args.double_dqn,
                weights=weights,
            )
            if args.prioritized:
                new_priorities = (td_errors.abs().cpu().numpy() + buffer.epsilon)
                buffer.update_priorities(idxs, new_priorities)

        if args.tau < 1.0:
            for target_param, q_param in zip(target_net.parameters(), q_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - args.tau) + q_param.data * args.tau)
        elif step % args.target_update == 0:
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


def compute_td_loss(q_net, target_net, optimizer, batch, gamma, max_grad_norm, loss_fn, double_dqn = False, weights = None):
    obs, actions, rewards, next_obs, dones = batch

    q_values = q_net(obs).gather(1, actions.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        if double_dqn:
            next_actions = q_net(next_obs).argmax(dim=1)
            next_q = target_net(next_obs).gather(1, next_actions.view(-1, 1)).squeeze(1)
        else:
            next_q = target_net(next_obs).max(dim=1)[0]
        target = rewards + gamma * (1.0 - dones) * next_q

    if loss_fn == "huber":
        per_sample_loss = F.smooth_l1_loss(q_values, target, reduction="none")
    elif loss_fn == "mse":
        per_sample_loss = F.mse_loss(q_values, target, reduction="none")
    elif loss_fn == "l1":
        per_sample_loss = F.l1_loss(q_values, target, reduction="none")
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

    if weights is not None:
        loss = (per_sample_loss * weights).mean()
    else:
        loss = per_sample_loss.mean()

    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_grad_norm)
    optimizer.step()

    td_errors = (target - q_values).detach()
    return loss, td_errors


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="DQN trainer with PyTorch")
    p.add_argument("--scenario", type=str, default="airport", help="Scenario id")
    p.add_argument("--pedestrians", type=int, default=12, help="Number of pedestrians")
    p.add_argument("--multi-scenario", action="store_true", help="Train across all scenarios with random ped count/speed")
    p.add_argument("--vary-pedestrians", action="store_true", help="Randomize pedestrian count/speed each episode in a single scenario")
    p.add_argument("--pedestrians-min", type=int, default=6, help="Min pedestrian count when varying")
    p.add_argument("--pedestrians-max", type=int, default=20, help="Max pedestrian count when varying")
    p.add_argument(
        "--ped-count-anchors",
        type=str,
        default="",
        help="Comma-separated anchor pedestrian counts for biased sampling (e.g., 0,30,100)",
    )
    p.add_argument(
        "--ped-count-anchor-prob",
        type=float,
        default=0.0,
        help="Probability of sampling near an anchor count instead of uniform range",
    )
    p.add_argument(
        "--ped-count-anchor-jitter",
        type=int,
        default=5,
        help="Integer jitter (+/-) applied around sampled anchor counts",
    )
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
    p.add_argument("--eps-schedule", type=str, default="linear", choices=["linear", "exponential", "cosine", "constant"], help="Epsilon decay schedule")
    p.add_argument("--hidden-sizes", type=str, default="256,256", help="Comma-separated hidden layer sizes")
    p.add_argument("--hidden-activation", type=str, default="relu", choices=["relu", "tanh", "leaky_relu"], help="Activation for hidden layers")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop"], help="Optimizer")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2) for optimizer")
    p.add_argument("--momentum", type=float, default=0.9, help="Momentum (for SGD)")
    p.add_argument("--loss-fn", type=str, default="huber", choices=["huber", "mse", "l1"], help="TD loss function")
    p.add_argument("--prioritized", action="store_true", help="Use prioritized replay buffer")
    p.add_argument("--priority-alpha", type=float, default=0.6, help="Prioritized replay alpha")
    p.add_argument("--priority-beta-start", type=float, default=0.4, help="Prioritized replay beta start")
    p.add_argument("--priority-beta-frames", type=int, default=200_000, help="Frames to anneal beta to 1")
    p.add_argument("--n-step", type=int, default=1, help="N-step returns")
    p.add_argument("--double-dqn", action="store_true", help="Use Double DQN")
    p.add_argument("--dueling-dqn", action="store_true", help="Use Dueling DQN")
    p.add_argument("--tau", type=float, default=1.0, help="Soft target update coefficient (1.0 = hard update)")
    p.add_argument("--warmup", type=int, default=5_000, help="Steps before starting updates")
    p.add_argument("--target-update", type=int, default=2_000, help="Steps between target network syncs")
    p.add_argument("--max-grad-norm", type=float, default=5.0, help="Gradient clipping norm (0 to disable)")
    p.add_argument("--log-interval", type=int, default=5_000, help="Steps between console logs")
    p.add_argument("--save-interval", type=int, default=25_000, help="Steps between checkpoints")
    p.add_argument("--output-dir", type=str, default="training_output/dqn", help="Where to save model")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return p.parse_args()


def main():
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
