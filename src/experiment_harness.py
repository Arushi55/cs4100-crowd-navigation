import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENARIOS = ("airport", "home", "shopping_center")
DEFAULT_EVAL_PEDS = (0, 30, 100)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train one DQN model per scenario, then evaluate each model at "
            "multiple pedestrian counts and write CSV metrics."
        )
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        help="Scenario ids to run (e.g., airport home shopping_center).",
    )
    p.add_argument(
        "--eval-pedestrians",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVAL_PEDS),
        help="Pedestrian counts for evaluation sweeps.",
    )
    p.add_argument("--total-steps", type=int, default=500_000, help="Training steps per scenario.")
    p.add_argument("--pedestrians-min", type=int, default=0, help="Minimum pedestrians during training.")
    p.add_argument("--pedestrians-max", type=int, default=100, help="Maximum pedestrians during training.")
    p.add_argument(
        "--ped-count-anchors",
        type=str,
        default="0,30,100",
        help="Comma-separated anchor counts to bias around while still mixing uniform samples.",
    )
    p.add_argument(
        "--ped-count-anchor-prob",
        type=float,
        default=0.65,
        help="Probability of sampling near an anchor count (remaining samples are uniform).",
    )
    p.add_argument(
        "--ped-count-anchor-jitter",
        type=int,
        default=8,
        help="Integer jitter (+/-) around sampled anchor counts.",
    )
    p.add_argument("--ped-speed-min", type=float, default=0.95, help="Minimum pedestrian speed multiplier.")
    p.add_argument("--ped-speed-max", type=float, default=1.15, help="Maximum pedestrian speed multiplier.")
    p.add_argument("--frame-stack", type=int, default=4, help="Observation stack depth.")
    p.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer size.")
    p.add_argument("--batch-size", type=int, default=256, help="DQN batch size.")
    p.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    p.add_argument("--target-update", type=int, default=2_000, help="Target network sync interval.")
    p.add_argument("--eval-episodes", type=int, default=25, help="Episodes per eval run.")
    p.add_argument("--seed", type=int, default=42, help="Base seed used for train/eval.")
    p.add_argument(
        "--train-output-root",
        type=Path,
        default=Path("training_outputs") / "dqn",
        help="Directory where per-scenario model folders are written.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for evaluation CSV metrics.",
    )
    p.add_argument("--skip-train", action="store_true", help="Skip training and only run eval.")
    p.add_argument("--skip-eval", action="store_true", help="Skip eval and only run training.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain even if a scenario model checkpoint already exists.",
    )
    return p.parse_args()


def _run(cmd, cwd):
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n$ {printable}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _resolve_output_dir(repo_root, raw_path):
    """
    Resolve user-provided output directories relative to this script's folder.

    When run from repo root as `python3 src/experiment_harness.py`, users often
    pass paths like `src/results/smoke`; since this script already lives in
    `src/`, strip a leading `src/` to avoid `src/src/...`.
    """
    if raw_path.is_absolute():
        return raw_path

    parts = raw_path.parts
    if repo_root.name == "src" and parts and parts[0] == "src":
        stripped = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        return (repo_root / stripped).resolve()

    return (repo_root / raw_path).resolve()


def train_scenario(args, repo_root, scenario, model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "dqn.pt"

    if model_path.exists() and not args.overwrite:
        print(f"Skipping train for '{scenario}' (existing model: {model_path})")
        return model_path

    cmd = [
        sys.executable,
        str(repo_root / "dqn.py"),
        "--scenario",
        scenario,
        "--vary-pedestrians",
        "--pedestrians-min",
        str(args.pedestrians_min),
        "--pedestrians-max",
        str(args.pedestrians_max),
        "--ped-count-anchors",
        str(args.ped_count_anchors),
        "--ped-count-anchor-prob",
        str(args.ped_count_anchor_prob),
        "--ped-count-anchor-jitter",
        str(args.ped_count_anchor_jitter),
        "--ped-speed-min",
        str(args.ped_speed_min),
        "--ped-speed-max",
        str(args.ped_speed_max),
        "--frame-stack",
        str(args.frame_stack),
        "--total-steps",
        str(args.total_steps),
        "--buffer-size",
        str(args.buffer_size),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--target-update",
        str(args.target_update),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(model_dir),
    ]
    _run(cmd, cwd=repo_root)

    if not model_path.exists():
        raise RuntimeError(f"Training finished but model not found: {model_path}")
    return model_path


def eval_scenario(args, repo_root, scenario, model_path, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)

    for ped_count in args.eval_pedestrians:
        csv_path = results_dir / f"dqn_{scenario}_{ped_count}.csv"
        cmd = [
            sys.executable,
            str(repo_root / "evaluate.py"),
            "--model",
            str(model_path),
            "--scenario",
            scenario,
            "--pedestrians",
            str(ped_count),
            "--episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--no-render",
            "--save-metrics",
            str(csv_path),
        ]
        _run(cmd, cwd=repo_root)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    train_output_root = _resolve_output_dir(repo_root, args.train_output_root)
    results_dir = _resolve_output_dir(repo_root, args.results_dir)

    train_output_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== DQN Experiment Harness ===")
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print(f"Eval pedestrian counts: {args.eval_pedestrians}")
    print(
        "Ped-count sampling: "
        f"range=[{args.pedestrians_min},{args.pedestrians_max}], "
        f"anchors={args.ped_count_anchors!r}, "
        f"anchor_prob={args.ped_count_anchor_prob}, "
        f"anchor_jitter=+/-{args.ped_count_anchor_jitter}"
    )
    print(f"Train output root: {train_output_root}")
    print(f"Results dir: {results_dir}")

    for scenario in args.scenarios:
        print(f"\n--- Scenario: {scenario} ---")
        model_dir = train_output_root / scenario
        model_path = model_dir / "dqn.pt"

        if not args.skip_train:
            model_path = train_scenario(args, repo_root, scenario, model_dir)
        elif not model_path.exists():
            raise FileNotFoundError(
                f"--skip-train was set, but model does not exist: {model_path}"
            )

        if not args.skip_eval:
            eval_scenario(args, repo_root, scenario, model_path, results_dir)

    print("\nHarness complete.")


if __name__ == "__main__":
    main()
