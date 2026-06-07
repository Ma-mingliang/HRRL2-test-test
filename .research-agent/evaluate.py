#!/usr/bin/env python3
"""Evaluation script for HRRL2 optimizer.

Runs model for N episodes and outputs metrics in standard format.

CLI:
  python .research-agent/evaluate.py <checkpoint_path> [--episodes 30]
"""
import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate(checkpoint_path: str, num_episodes: int = 30):
    import importlib
    import env as env_module
    importlib.reload(env_module)

    env = env_module.Attitude_control_stage1(render=False)
    env.record_flag = 0

    model = TD3.load(checkpoint_path, env=env)

    episode_rewards = []
    episode_steps = []
    episode_tilt_errors = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0
        ep_tilt_errors = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1
            # Track tilt error (dis_angle, normalized)
            tilt_error = abs(obs[0] * 1.57)  # de-normalize
            ep_tilt_errors.append(tilt_error)

        episode_rewards.append(ep_reward)
        episode_steps.append(ep_steps)
        episode_tilt_errors.append(float(np.mean(ep_tilt_errors)))

    env.close()

    mean_reward = float(np.mean(episode_rewards))
    mean_tilt_error = float(np.mean(episode_tilt_errors))
    completion_rate = float(np.mean([1.0 if s >= 300 else 0.0 for s in episode_steps]))

    return {
        "reward": mean_reward,
        "completion_rate": completion_rate,
        "lateral_error": mean_tilt_error,  # tilt error as lateral_error for framework
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate HRRL2 model")
    parser.add_argument("checkpoint", type=str, help="Path to model .zip")
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    print(f"[EVAL] checkpoint={args.checkpoint}", flush=True)
    print(f"[EVAL] episodes={args.episodes}", flush=True)

    metrics = evaluate(args.checkpoint, args.episodes)

    # Output in standard format (parsed by metric_parser)
    print(f"\n[METRICS]", flush=True)
    print(json.dumps(metrics), flush=True)
    print(f"\ncompletion_rate = {metrics['completion_rate']:.4f}", flush=True)
    print(f"reward = {metrics['reward']:.4f}", flush=True)
    print(f"lateral_error = {metrics['lateral_error']:.4f}", flush=True)


if __name__ == "__main__":
    main()
