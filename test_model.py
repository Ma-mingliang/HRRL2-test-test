#!/usr/bin/env python3
"""Evaluate a trained TD3 model over 30 episodes using v0-baseline env reward."""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def evaluate(model_path: str, num_episodes: int = 30, verbose: bool = True):
    import importlib
    import env as env_module
    importlib.reload(env_module)

    env = env_module.Attitude_control_stage1(render=False)
    env.record_flag = 0

    model = TD3.load(model_path, env=env)

    episode_rewards = []
    episode_steps = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1

        episode_rewards.append(ep_reward)
        episode_steps.append(ep_steps)
        if verbose:
            print(f"  Episode {ep+1:2d}/{num_episodes} | Steps: {ep_steps:4d} | Reward: {ep_reward:.2f}")

    env.close()

    mean_r = np.mean(episode_rewards)
    std_r = np.std(episode_rewards)
    min_r = np.min(episode_rewards)
    max_r = np.max(episode_rewards)
    mean_steps = np.mean(episode_steps)

    return {
        "mean_reward": mean_r,
        "std_reward": std_r,
        "min_reward": min_r,
        "max_reward": max_r,
        "mean_steps": mean_steps,
        "episodes": num_episodes,
        "episode_rewards": episode_rewards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Evaluating: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"{'='*60}")

    result = evaluate(args.model, num_episodes=args.episodes, verbose=not args.quiet)

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model}")
    print(f"{'='*60}")
    print(f"  Mean Reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
    print(f"  Min Reward:  {result['min_reward']:.2f}")
    print(f"  Max Reward:  {result['max_reward']:.2f}")
    print(f"  Mean Steps:  {result['mean_steps']:.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
