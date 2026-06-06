#!/usr/bin/env python3
"""Test a saved model checkpoint with the CURRENT env.py.

Runs N episodes with the current reward function and reports metrics.
Used for fair comparison: always evaluates against the same reward function.

Usage:
    python test_best_model.py <model_path> [--episodes 30]
"""

import sys
import json
import traceback
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from stable_baselines3 import TD3


def test_model(model_path: Path, n_episodes: int = 30) -> dict:
    """Load a saved model and evaluate it for N episodes."""
    import importlib
    import env as env_module
    importlib.reload(env_module)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TD3.load(str(model_path), device=device)

    episode_rewards = []
    episode_lengths = []
    angle_errors = []
    completed = 0

    for ep in range(n_episodes):
        env_instance = env_module.Attitude_control_stage1(render=False)
        obs, _ = env_instance.reset()
        done = False
        total_reward = 0.0
        steps = 0
        ep_angle_errors = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_instance.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            ep_angle_errors.append(getattr(env_instance, 'angle_error', 0.0))

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if ep_angle_errors:
            angle_errors.append(float(np.mean(ep_angle_errors)))
        if steps >= env_instance.max_step_num:
            completed += 1

        env_instance.close()

    completion_rate = completed / n_episodes
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_lateral_error = float(np.mean(angle_errors)) if angle_errors else 0.0

    return {
        "completion_rate": completion_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "lateral_error": mean_lateral_error,
        "completed_episodes": completed,
        "total_episodes": n_episodes,
        "episode_rewards": episode_rewards,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_best_model.py <model_path> [--episodes N]")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    n_episodes = 30
    for i, arg in enumerate(sys.argv):
        if arg == "--episodes" and i + 1 < len(sys.argv):
            n_episodes = int(sys.argv[i + 1])

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    print(f"[TEST] Testing model: {model_path}", flush=True)
    print(f"[TEST] Episodes: {n_episodes}", flush=True)

    try:
        metrics = test_model(model_path, n_episodes)
    except Exception as e:
        print(f"[ERROR] Test failed: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    # Output JSON
    print("\n[METRICS]")
    print(json.dumps(metrics, indent=2))

    # Output parseable format (for research-agent metric_regex)
    print(f"\ncompletion_rate = {metrics['completion_rate']:.4f}")
    print(f"reward = {metrics['mean_reward']:.4f}")
    print(f"lateral_error = {metrics['lateral_error']:.4f}")

    return metrics


if __name__ == "__main__":
    main()
