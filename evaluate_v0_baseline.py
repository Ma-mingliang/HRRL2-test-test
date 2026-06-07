#!/usr/bin/env python3
"""Evaluate best model with v0-baseline env.py for N episodes."""
import sys
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def evaluate(model_path: Path, n_episodes: int = 30):
    import importlib
    import env as env_module
    importlib.reload(env_module)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TD3.load(str(model_path), device=device)

    episode_rewards = []
    for ep in range(n_episodes):
        env_instance = env_module.Attitude_control_stage1(render=False)
        obs, _ = env_instance.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env_instance.step(action)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
        env_instance.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "n_episodes": n_episodes,
        "episode_rewards": episode_rewards,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    print(f"[EVAL] Model: {model_path}", flush=True)
    print(f"[EVAL] Episodes: {args.episodes}", flush=True)

    metrics = evaluate(model_path, args.episodes)

    print(f"\n[RESULT]", flush=True)
    print(f"  mean_reward: {metrics['mean_reward']:.2f}", flush=True)
    print(f"  std_reward:  {metrics['std_reward']:.2f}", flush=True)
    print(f"  min_reward:  {metrics['min_reward']:.2f}", flush=True)
    print(f"  max_reward:  {metrics['max_reward']:.2f}", flush=True)

    print(f"\n[METRICS]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
