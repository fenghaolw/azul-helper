"""
Full Training Run Script for Azul Neural Network

This script implements the continuous self-play and training loop with:
- Self-play game generation
- Neural network training
- Real-time monitoring and logging
- TensorBoard integration
- Model checkpointing
- Performance tracking
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from training.neural_network import AzulNeuralNetwork
from training.training_loop import AzulTrainer, TrainingConfig


class TrainingMonitor:
    """Handles logging and monitoring for the training process."""

    def __init__(self, log_dir: str):
        """
        Initialize the training monitor.

        Args:
            log_dir: Directory for logs and tensorboard
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        tensorboard_dir = self.log_dir / "tensorboard"
        self.writer = SummaryWriter(tensorboard_dir)

        # JSON logging
        self.log_file = self.log_dir / "training_log.json"
        self.training_logs = []

        # Performance tracking
        self.start_time = time.time()
        self.iteration_times = []

        print(f"Training monitor initialized:")
        print(f"  Log directory: {self.log_dir}")
        print(f"  TensorBoard: {tensorboard_dir}")
        print(f"  Log file: {self.log_file}")

    def log_iteration(self, iteration: int, stats: Dict[str, Any]) -> None:
        """
        Log statistics for a training iteration.

        Args:
            iteration: Current iteration number
            stats: Dictionary of statistics to log
        """
        # Add timestamp
        stats["timestamp"] = datetime.now().isoformat()
        stats["elapsed_time"] = time.time() - self.start_time

        # Log to TensorBoard
        self._log_to_tensorboard(iteration, stats)

        # Log to JSON
        self.training_logs.append(stats)
        self._save_json_log()

        # Print summary
        self._print_summary(iteration, stats)

    def _log_to_tensorboard(self, iteration: int, stats: Dict[str, Any]) -> None:
        """Log metrics to TensorBoard."""
        # Training metrics
        if "total_loss" in stats:
            self.writer.add_scalar("Loss/Total", stats["total_loss"], iteration)
            self.writer.add_scalar("Loss/Policy", stats["policy_loss"], iteration)
            self.writer.add_scalar("Loss/Value", stats["value_loss"], iteration)

        # Self-play metrics
        if "buffer_size" in stats:
            self.writer.add_scalar(
                "SelfPlay/BufferSize", stats["buffer_size"], iteration
            )
        if "games_played" in stats:
            self.writer.add_scalar(
                "SelfPlay/GamesPlayed", stats["games_played"], iteration
            )
        if "avg_moves_per_game" in stats:
            self.writer.add_scalar(
                "SelfPlay/AvgMovesPerGame", stats["avg_moves_per_game"], iteration
            )

        # Evaluation metrics
        if "eval_score" in stats:
            self.writer.add_scalar("Evaluation/Score", stats["eval_score"], iteration)
        if "eval_time" in stats:
            self.writer.add_scalar("Evaluation/Time", stats["eval_time"], iteration)

        # Training efficiency
        if "iteration_time" in stats:
            self.writer.add_scalar(
                "Efficiency/IterationTime", stats["iteration_time"], iteration
            )
        if "training_time" in stats:
            self.writer.add_scalar(
                "Efficiency/TrainingTime", stats["training_time"], iteration
            )

    def _save_json_log(self) -> None:
        """Save training logs to JSON file."""
        with open(self.log_file, "w") as f:
            json.dump(self.training_logs, f, indent=2)

    def _print_summary(self, iteration: int, stats: Dict[str, Any]) -> None:
        """Print formatted summary of current iteration."""
        print(f"\n{'='*80}")
        print(
            f"Iteration {iteration} Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"{'='*80}")

        # Self-play stats
        print("\nSelf-Play:")
        print(f"  Buffer Size: {stats.get('buffer_size', 'N/A')}")
        print(f"  Games Played: {stats.get('games_played', 'N/A')}")
        print(
            f"  Avg Moves/Game: {stats.get('avg_moves_per_game', 'N/A'):.1f}"
            if "avg_moves_per_game" in stats
            else "  Avg Moves/Game: N/A"
        )

        # Training stats
        if "total_loss" in stats:
            print("\nTraining:")
            print(f"  Total Loss: {stats['total_loss']:.6f}")
            print(f"  Policy Loss: {stats['policy_loss']:.6f}")
            print(f"  Value Loss: {stats['value_loss']:.6f}")
            print(f"  Successful Steps: {stats.get('successful_steps', 'N/A')}")

        # Evaluation stats
        if "eval_score" in stats:
            print("\nEvaluation:")
            print(f"  Score: {stats['eval_score']:.4f}")
            print(f"  Games: {stats.get('eval_games', 'N/A')}")
            if stats.get("is_best", False):
                print("  🏆 NEW BEST MODEL! 🏆")

        # Timing stats
        print("\nTiming:")
        print(f"  Iteration Time: {stats.get('iteration_time', 0):.2f}s")
        print(f"  Total Elapsed: {stats.get('elapsed_time', 0):.2f}s")

        print(f"{'='*80}\n")

    def close(self) -> None:
        """Close the monitor and save final logs."""
        self.writer.close()
        self._save_json_log()
        print(f"\nTraining logs saved to: {self.log_file}")
        print(f"TensorBoard logs saved to: {self.log_dir / 'tensorboard'}")


def get_default_hyperparameters() -> Dict[str, Any]:
    """
    Get default hyperparameters for training.

    Returns:
        Dictionary of hyperparameter values
    """
    return {
        # Self-play parameters
        "self_play_games_per_iteration": 100,
        "mcts_simulations": 800,
        "temperature": 1.0,
        "temperature_threshold": 30,
        "num_players": 2,
        # Replay buffer parameters
        "buffer_capacity": 100000,
        "min_buffer_size": 5000,
        # Training parameters
        "batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "training_steps_per_iteration": 1000,
        "gradient_clip_norm": 1.0,
        # Loss function parameters
        "policy_loss_weight": 1.0,
        "value_loss_weight": 1.0,
        "loss_temperature": 1.0,
        # Model parameters
        "network_config": "medium",
        # Evaluation parameters
        "eval_games": 50,
        "eval_frequency": 5,
        # Saving parameters
        "save_frequency": 10,
        "save_best_model": True,
        "keep_latest_checkpoints": 5,
        # General parameters
        "max_iterations": 1000,
        "device": "auto",
        "verbose": True,
    }


class EnhancedAzulTrainer(AzulTrainer):
    """Enhanced trainer with integrated monitoring."""

    def __init__(self, config: TrainingConfig, save_dir: str, monitor: TrainingMonitor):
        """
        Initialize enhanced trainer.

        Args:
            config: Training configuration
            save_dir: Directory to save models
            monitor: Training monitor for logging
        """
        super().__init__(config, save_dir)
        self.monitor = monitor

    def _print_iteration_summary(self, stats: Dict[str, Any]) -> None:
        """Override to use monitor for logging."""
        # Let monitor handle all logging
        self.monitor.log_iteration(stats["iteration"], stats)


def run_full_training(
    hyperparameters: Optional[Dict[str, Any]] = None,
    save_dir: str = "models/full_training",
    log_dir: str = "logs/full_training",
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full training loop with monitoring.

    Args:
        hyperparameters: Custom hyperparameters (uses defaults if None)
        save_dir: Directory to save models
        log_dir: Directory for logs
        resume_from: Path to model to resume from

    Returns:
        Training results
    """
    # Get hyperparameters
    if hyperparameters is None:
        hyperparameters = get_default_hyperparameters()

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    log_dir = f"{log_dir}_{timestamp}"

    # Create configuration
    config = TrainingConfig(**hyperparameters)

    # Create monitor
    monitor = TrainingMonitor(log_dir)

    # Save hyperparameters
    hyperparam_file = Path(log_dir) / "hyperparameters.json"
    with open(hyperparam_file, "w") as f:
        json.dump(hyperparameters, f, indent=2)

    print("\n" + "=" * 80)
    print("AZUL NEURAL NETWORK TRAINING")
    print("=" * 80)
    print(f"\nHyperparameters saved to: {hyperparam_file}")
    print("\nKey hyperparameters:")
    print(f"  Network: {hyperparameters['network_config']}")
    print(f"  MCTS Simulations: {hyperparameters['mcts_simulations']}")
    print(f"  Batch Size: {hyperparameters['batch_size']}")
    print(f"  Learning Rate: {hyperparameters['learning_rate']}")
    print(f"  Buffer Capacity: {hyperparameters['buffer_capacity']}")
    print(f"  Max Iterations: {hyperparameters['max_iterations']}")

    # Create trainer
    trainer = EnhancedAzulTrainer(config=config, save_dir=save_dir, monitor=monitor)

    print(f"\nStarting training run...")
    print(f"  Models will be saved to: {save_dir}")
    print(f"  Logs will be saved to: {log_dir}")
    print(f"  To monitor training, run: tensorboard --logdir={log_dir}/tensorboard")
    print("\n" + "=" * 80 + "\n")

    try:
        # Run training
        results = trainer.train()

        # Save final results
        results_file = Path(log_dir) / "final_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nFinal results saved to: {results_file}")

    finally:
        # Close monitor
        monitor.close()

    return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run full Azul neural network training with monitoring"
    )

    # Hyperparameter overrides
    parser.add_argument("--iterations", type=int, help="Maximum training iterations")
    parser.add_argument(
        "--games-per-iter", type=int, help="Self-play games per iteration"
    )
    parser.add_argument("--mcts-sims", type=int, help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument(
        "--network", choices=["small", "medium", "large"], help="Network size"
    )

    # Directories
    parser.add_argument(
        "--save-dir", default="models/full_training", help="Model save directory"
    )
    parser.add_argument("--log-dir", default="logs/full_training", help="Log directory")

    # Other options
    parser.add_argument("--resume", help="Path to model to resume from")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device to use",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    return parser.parse_args()


def main():
    """Main entry point for training script."""
    args = parse_arguments()

    # Build hyperparameters from defaults and overrides
    hyperparameters = get_default_hyperparameters()

    if args.iterations is not None:
        hyperparameters["max_iterations"] = args.iterations
    if args.games_per_iter is not None:
        hyperparameters["self_play_games_per_iteration"] = args.games_per_iter
    if args.mcts_sims is not None:
        hyperparameters["mcts_simulations"] = args.mcts_sims
    if args.batch_size is not None:
        hyperparameters["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        hyperparameters["learning_rate"] = args.learning_rate
    if args.network is not None:
        hyperparameters["network_config"] = args.network

    hyperparameters["device"] = args.device
    hyperparameters["verbose"] = not args.quiet

    # Run training
    results = run_full_training(
        hyperparameters=hyperparameters,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_from=args.resume,
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total time: {results['total_training_time']:.2f}s")
    print(f"Best eval score: {results['best_eval_score']:.4f}")
    print(f"Iterations completed: {results['iterations_completed']}")


if __name__ == "__main__":
    main()
