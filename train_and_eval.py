#!/usr/bin/env python3
"""Train and evaluate HRRL2 model for research-agent optimization."""

import os
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model_evaluator import ModelEvaluator


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_and_eval.py <seed> [--screening] [--checkpoint-dir DIR]")
        sys.exit(1)

    seed = int(sys.argv[1])
    screening = "--screening" in sys.argv

    checkpoint_dir = None
    for i, arg in enumerate(sys.argv):
        if arg == "--checkpoint-dir" and i + 1 < len(sys.argv):
            checkpoint_dir = Path(sys.argv[i + 1])
            break
    if checkpoint_dir is None:
        env_dir = os.environ.get("RA_CHECKPOINT_DIR")
        if env_dir:
            checkpoint_dir = Path(env_dir)
    if checkpoint_dir is None:
        checkpoint_dir = project_root / "model" / "checkpoints"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.zip"

    evaluator = ModelEvaluator(project_root)

    if screening:
        print(f"[SCREENING] Running quick evaluation with seed {seed}", flush=True)
        print(f"[CHECKPOINT] Saving best model to: {best_model_path}", flush=True)
        metrics = evaluator.quick_evaluate(timesteps=5000, save_path=best_model_path)
    else:
        print(f"[TRAINING] Training model with seed {seed}", flush=True)
        print(f"[CHECKPOINT] Saving best model to: {best_model_path}", flush=True)
        train_metrics = evaluator.train_model(timesteps=20000, save_path=best_model_path)
        if train_metrics is None:
            print("[ERROR] Training failed", flush=True)
            sys.exit(1)

        if not best_model_path.exists():
            print(f"[ERROR] Checkpoint not found: {best_model_path}", flush=True)
            sys.exit(1)

        print(f"[EVALUATING] Evaluating best model from: {best_model_path}", flush=True)
        eval_metrics = evaluator.evaluate_model(best_model_path)
        if eval_metrics is None:
            print("[ERROR] Evaluation failed", flush=True)
            sys.exit(1)

        metrics = {**train_metrics, **eval_metrics}

    print("\n[METRICS]")
    print(json.dumps(metrics, indent=2))

    reward_val = metrics.get('eval_mean_reward') or metrics.get('mean_reward') or 0.0
    if reward_val != reward_val:
        reward_val = 0.0
    print(f"\nreward = {reward_val:.4f}")
    print(f"completion_rate = {metrics.get('completion_rate', 0):.4f}")
    print(f"lateral_error = {metrics.get('lateral_error', metrics.get('mean_episode_length', 0)):.4f}")


if __name__ == "__main__":
    main()
