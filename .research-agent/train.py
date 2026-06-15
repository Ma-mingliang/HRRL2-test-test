#!/usr/bin/env python3
"""Training script for HRRL2 optimizer.

Convention:
  - RA_CHECKPOINT_DIR environment variable is set by the optimizer framework
  - Best model is saved to {RA_CHECKPOINT_DIR}/best_model.zip
  - Best model = highest average reward over last 20 episodes

CLI:
  python .research-agent/train.py <seed> [--timesteps N]
"""
import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class BestModelCallback(BaseCallback):
    """Saves the best model by 20-episode average reward."""

    def __init__(self, save_path: Path, last_n: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.last_n = last_n
        self.best_mean_reward = -float("inf")
        self.last_episode_count = 0

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        while hasattr(env, 'env'):
            env = env.env
        epoch_r_list = getattr(env, 'epoch_r_list', None)
        if epoch_r_list is None:
            return True
        current_count = len(epoch_r_list)
        if current_count > self.last_episode_count and current_count >= self.last_n:
            self.last_episode_count = current_count
            recent = epoch_r_list[-self.last_n:]
            mean_reward = float(np.mean(recent))
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.save_path))
                if self.verbose:
                    print(f"  [BestModel] ep={current_count} mean={mean_reward:.2f}", flush=True)
        return True


def train(seed: int, checkpoint_dir: Path, timesteps: int) -> dict:
    import importlib
    import env as env_module
    importlib.reload(env_module)

    best_model_path = checkpoint_dir / "best_model.zip"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_instance = env_module.Attitude_control_stage1(render=False)
    env_instance.record_flag = 0

    log_dir = str(PROJECT_ROOT / "model" / "train_logs")
    os.makedirs(log_dir, exist_ok=True)
    env_monitored = Monitor(env_instance, log_dir)

    n_actions = env_monitored.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TD3(
        "MlpPolicy", env=env_monitored, device=device,
        gamma=0.99, learning_rate=0.001, batch_size=128,
        buffer_size=100000, action_noise=action_noise,
        learning_starts=128, train_freq=(1, "step"),
        gradient_steps=-1, policy_delay=2, seed=seed, verbose=0
    )

    callback = BestModelCallback(save_path=best_model_path, last_n=20, verbose=1)

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    elapsed = time.time() - t0

    env_monitored.close()

    return {
        "best_mean_reward": callback.best_mean_reward,
        "training_time_s": round(elapsed, 1),
        "timesteps": timesteps,
        "checkpoint": str(best_model_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train HRRL2 model")
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    # RA_MAX_STEPS env var overrides --timesteps (used by smoke train)
    ra_max_steps = os.environ.get("RA_MAX_STEPS")
    if ra_max_steps:
        try:
            args.timesteps = int(ra_max_steps)
        except ValueError:
            pass

    checkpoint_dir = None
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    elif os.environ.get("RA_CHECKPOINT_DIR"):
        checkpoint_dir = Path(os.environ["RA_CHECKPOINT_DIR"])
    else:
        checkpoint_dir = PROJECT_ROOT / "model" / "checkpoints"

    print(f"[TRAIN] seed={args.seed}, timesteps={args.timesteps}", flush=True)
    print(f"[TRAIN] checkpoint_dir={checkpoint_dir}", flush=True)

    metrics = train(args.seed, checkpoint_dir, args.timesteps)

    print(f"\n[TRAIN COMPLETE]", flush=True)
    print(f"  best_mean_reward: {metrics['best_mean_reward']:.2f}", flush=True)
    print(f"  checkpoint: {metrics['checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
