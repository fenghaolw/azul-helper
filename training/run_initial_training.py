"""
Quick start script for initial Azul neural network training.

This script runs a short training session with reasonable defaults
to demonstrate the full self-play and training loop.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import training modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.full_training_run import get_default_hyperparameters, run_full_training


def run_initial_training():
    """Run initial training with optimized settings for quick demonstration."""
    print("\n" + "=" * 80)
    print("AZUL NEURAL NETWORK - INITIAL TRAINING RUN")
    print("=" * 80)
    print("\nThis will run a short training session to demonstrate the full pipeline.")
    print("For production training, use: python -m training.full_training_run")
    print("=" * 80 + "\n")

    # Define hyperparameters for initial run
    # These are optimized for faster iteration while still being meaningful
    initial_hyperparameters = {
        # Self-play parameters
        "self_play_games_per_iteration": 25,  # Reduced for faster iteration
        "mcts_simulations": 200,  # Reduced for speed
        "temperature": 1.0,
        "temperature_threshold": 20,
        "num_players": 2,
        # Replay buffer parameters
        "buffer_capacity": 10000,  # Smaller buffer
        "min_buffer_size": 1000,  # Lower threshold to start training sooner
        # Training parameters
        "batch_size": 128,  # Smaller batch for faster training
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "training_steps_per_iteration": 200,  # Fewer steps per iteration
        "gradient_clip_norm": 1.0,
        # Loss function parameters
        "policy_loss_weight": 1.0,
        "value_loss_weight": 1.0,
        "loss_temperature": 1.0,
        # Model parameters
        "network_config": "small",  # Use small network for speed
        # Evaluation parameters
        "eval_games": 10,  # Fewer evaluation games
        "eval_frequency": 2,  # Evaluate more frequently
        # Saving parameters
        "save_frequency": 5,
        "save_best_model": True,
        "keep_latest_checkpoints": 3,
        # General parameters
        "max_iterations": 10,  # Short run for demonstration
        "device": "auto",
        "verbose": True,
        # ETA tracking parameters
        "enable_eta_tracking": True,
        "eta_update_frequency": 1,
        "eta_detailed_display": True,  # Show detailed ETA info for demo
    }

    # Run training
    results = run_full_training(
        hyperparameters=initial_hyperparameters,
        save_dir="models/initial_training",
        log_dir="logs/initial_training",
    )

    # Print summary
    print("\n" + "=" * 80)
    print("INITIAL TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nTotal time: {results['total_training_time']:.2f} seconds")
    print(f"Best evaluation score: {results['best_eval_score']:.4f}")
    print(f"Iterations completed: {results['iterations_completed']}")

    # Get log directory (most recent)
    log_dirs = sorted(Path("logs").glob("initial_training_*"))
    if log_dirs:
        latest_log_dir = log_dirs[-1]
        log_file = latest_log_dir / "training_log.json"

        print(f"\nTraining logs saved to: {latest_log_dir}")
        print(f"\nTo monitor training progress:")
        print(f"  python -m training.monitor_dashboard {log_file}")
        print(f"\nTo view in TensorBoard:")
        print(f"  tensorboard --logdir={latest_log_dir}/tensorboard")

    # Get model directory
    model_dirs = sorted(Path("models").glob("initial_training_*"))
    if model_dirs:
        latest_model_dir = model_dirs[-1]
        print(f"\nModels saved to: {latest_model_dir}")

        # List checkpoints
        checkpoints = sorted(latest_model_dir.glob("checkpoints/*.pt"))
        if checkpoints:
            print(f"\nAvailable checkpoints:")
            for cp in checkpoints[-3:]:  # Show last 3
                print(f"  {cp.name}")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Run extended training:")
    print("   python -m training.full_training_run --iterations 100")
    print("\n2. Evaluate trained model:")
    print("   python run_evaluation.py --agent1 checkpoint --agent2 heuristic")
    print("\n3. Monitor ongoing training:")
    print("   python -m training.monitor_dashboard logs/<log_dir>/training_log.json")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Run initial training
    run_initial_training()
