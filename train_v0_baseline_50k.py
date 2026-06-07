#!/usr/bin/env python3
"""Train TD3 with v0-baseline env.py for 50k steps. Best model = highest avg reward over 20 episodes."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class BestModelCallback(BaseCallback):
    """Save best model based on average reward over last N episodes."""

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


def main():
    import importlib
    import env as env_module
    importlib.reload(env_module)

    checkpoint_dir = Path("model/checkpoints/v0_baseline")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model_v0_baseline_20k.zip"

    env_instance = env_module.Attitude_control_stage1(render=False)
    env_instance.record_flag = 1
    log_dir = str(project_root / "model" / "train_logs_v0_20k")
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
        gradient_steps=-1, policy_delay=2, seed=42, verbose=0
    )

    callback = BestModelCallback(save_path=best_path, last_n=20, verbose=1)

    print("[TRAIN] Starting 20k step training (20-episode window)...", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=20000, callback=callback)
    elapsed = time.time() - t0
    print(f"[TRAIN] Done in {elapsed:.0f}s", flush=True)
    print(f"[TRAIN] Best model saved to: {best_path}", flush=True)
    print(f"[TRAIN] Best mean reward (20ep): {callback.best_mean_reward:.2f}", flush=True)

    env_monitored.close()


if __name__ == "__main__":
    main()
