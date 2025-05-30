"""
Main training loop for Azul neural network using self-play and MCTS.

This module implements the complete AlphaZero-style training pipeline with enhancements:
1. Generate self-play games using current neural network
2. Store experiences in replay buffer
3. Train neural network on sampled experiences using enhanced utilities
4. Evaluate and save improved models

Enhanced features:
- Cross-entropy loss for policy head (instead of KL divergence)
- MSE loss for value head
- Improved batch sampling
- Better checkpoint management
- ETA tracking for training progress
- All requirements from the user's specifications
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.eta_tracker import ETATracker
from training.neural_network import AzulNetwork, AzulNeuralNetwork, create_azul_network
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayEngine
from training.training_utils import (
    BatchSampler,
    EnhancedLossFunctions,
    ModelCheckpointManager,
    TrainingStep,
    create_training_components,
)


class TrainingConfig:
    """Configuration for training parameters."""

    def __init__(
        self,
        # Self-play parameters
        self_play_games_per_iteration: int = 100,
        mcts_simulations: int = 800,
        temperature: float = 1.0,
        temperature_threshold: int = 30,
        num_players: int = 2,
        # Replay buffer parameters
        buffer_capacity: int = 100000,
        min_buffer_size: int = 5000,
        # Training parameters
        batch_size: int = 512,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        training_steps_per_iteration: int = 1000,
        gradient_clip_norm: float = 1.0,
        # Loss function parameters
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        loss_temperature: float = 1.0,
        # Model parameters
        network_config: str = "medium",
        # Evaluation parameters
        eval_games: int = 50,
        eval_frequency: int = 5,
        # Saving parameters
        save_frequency: int = 10,
        save_best_model: bool = True,
        keep_latest_checkpoints: int = 5,
        # General parameters
        max_iterations: int = 1000,
        device: Optional[str] = None,
        verbose: bool = True,
        # ETA tracking parameters
        enable_eta_tracking: bool = True,
        eta_update_frequency: int = 1,
        eta_detailed_display: bool = False,
    ):
        self.self_play_games_per_iteration = self_play_games_per_iteration
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.num_players = num_players

        self.buffer_capacity = buffer_capacity
        self.min_buffer_size = min_buffer_size

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.training_steps_per_iteration = training_steps_per_iteration
        self.gradient_clip_norm = gradient_clip_norm

        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.loss_temperature = loss_temperature

        self.network_config = network_config

        self.eval_games = eval_games
        self.eval_frequency = eval_frequency

        self.save_frequency = save_frequency
        self.save_best_model = save_best_model
        self.keep_latest_checkpoints = keep_latest_checkpoints

        self.max_iterations = max_iterations
        self.device = device
        self.verbose = verbose

        self.enable_eta_tracking = enable_eta_tracking
        self.eta_update_frequency = eta_update_frequency
        self.eta_detailed_display = eta_detailed_display


class AzulTrainer:
    """
    Main training class for Azul neural network using self-play.

    This class coordinates the entire training pipeline including self-play
    game generation, experience storage, and neural network optimization.

    Enhanced with all required functionality:
    - Batch sampling from replay buffer with (state, MCTS_policy_target, game_outcome)
    - Cross-entropy loss for policy head, MSE for value head
    - Training step: forward pass, loss calculation, backward pass, optimizer step
    - Regular model checkpoint saving
    """

    def __init__(
        self,
        config: TrainingConfig,
        save_dir: str = "models/training",
        resume_from: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            save_dir: Directory to save models and logs
            resume_from: Path to model to resume training from
        """
        self.config = config
        self.save_dir = save_dir

        # Determine device with MPS support
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device or "cpu")

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Initialize neural network
        if resume_from:
            self.neural_network = AzulNeuralNetwork(
                config_name=config.network_config,
                model_path=resume_from,
                device=str(self.device),
            )
            if config.verbose:
                print(f"Resumed training from: {resume_from}")
        else:
            self.neural_network = AzulNeuralNetwork(
                config_name=config.network_config, device=str(self.device)
            )
            if config.verbose:
                print(f"Initialized new {config.network_config} network")

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            min_size_for_sampling=config.min_buffer_size,
        )

        # Initialize self-play engine
        self.self_play_engine = SelfPlayEngine(
            neural_network=self.neural_network,
            replay_buffer=self.replay_buffer,
            mcts_simulations=config.mcts_simulations,
            temperature=config.temperature,
            temperature_threshold=config.temperature_threshold,
            verbose=config.verbose,
        )

        # Create enhanced training components
        self.training_step, self.checkpoint_manager = create_training_components(
            model=self.neural_network.model,
            replay_buffer=self.replay_buffer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            batch_size=config.batch_size,
            device=self.device,
            policy_loss_weight=config.policy_loss_weight,
            value_loss_weight=config.value_loss_weight,
        )

        # Initialize ETA tracker
        self.eta_tracker: Optional[ETATracker] = None
        if config.enable_eta_tracking:
            self.eta_tracker = ETATracker(
                total_iterations=config.max_iterations,
                moving_average_window=min(10, config.max_iterations // 10),
                enable_phase_tracking=True,
            )
            if config.verbose:
                print("ETA tracking enabled")

        # Override checkpoint manager settings
        self.checkpoint_manager.save_frequency = config.save_frequency
        self.checkpoint_manager.keep_best = config.save_best_model
        self.checkpoint_manager.keep_latest = config.keep_latest_checkpoints
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        self.checkpoint_manager.save_dir = Path(checkpoint_dir)
        self.checkpoint_manager.save_dir.mkdir(parents=True, exist_ok=True)

        # Training statistics
        self.iteration = 0
        self.total_training_time = 0.0
        self.best_eval_score = -float("inf")
        self.training_history: List[Dict[str, Any]] = []

        if config.verbose:
            print(f"Trainer initialized:")
            print(f"  Device: {self.device}")
            print(f"  Network: {self.neural_network.get_model_info()}")
            print(f"  Buffer capacity: {config.buffer_capacity}")
            print(f"  Batch size: {config.batch_size}")
            print(f"  Policy loss weight: {config.policy_loss_weight}")
            print(f"  Value loss weight: {config.value_loss_weight}")
            print(f"  Loss temperature: {config.loss_temperature}")

    def train(self) -> Dict[str, Any]:
        """
        Run the complete training loop.

        Returns:
            Dictionary containing training results and statistics
        """
        if self.config.verbose:
            print("Starting training loop...")

        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            iteration_start = time.time()

            # Start ETA tracking for this iteration
            if self.eta_tracker:
                self.eta_tracker.start_iteration(iteration)

            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{self.config.max_iterations}")
                print(f"{'='*60}")

            # Phase 1: Generate self-play games
            if self.eta_tracker:
                self.eta_tracker.start_phase("self_play")
            self._generate_self_play_data()
            if self.eta_tracker:
                self.eta_tracker.end_phase()

            # Phase 2: Train neural network using enhanced training step
            if self.replay_buffer.is_ready_for_sampling():
                if self.eta_tracker:
                    self.eta_tracker.start_phase("training")
                training_stats = self._train_network()
                if self.eta_tracker:
                    self.eta_tracker.end_phase()
            else:
                training_stats = {"message": "Buffer not ready for training"}

            # Phase 3: Evaluate model (periodically)
            eval_stats: Dict[str, Any] = {}
            if iteration % self.config.eval_frequency == 0:
                if self.eta_tracker:
                    self.eta_tracker.start_phase("evaluation")
                eval_stats = self._evaluate_model()
                if self.eta_tracker:
                    self.eta_tracker.end_phase()

            # Phase 4: Save model checkpoint using enhanced checkpoint manager
            if self.checkpoint_manager.should_save_checkpoint(iteration + 1):
                self._save_checkpoint(training_stats, eval_stats)

            # End ETA tracking for this iteration
            if self.eta_tracker:
                self.eta_tracker.end_iteration()

            # Update statistics
            iteration_time = time.time() - iteration_start
            iteration_stats = {
                "iteration": iteration + 1,
                "iteration_time": iteration_time,
                "buffer_size": self.replay_buffer.size(),
                "games_played": self.self_play_engine.games_played,
                **training_stats,
                **eval_stats,
            }

            # Add ETA information to stats
            if self.eta_tracker:
                eta_summary = self.eta_tracker.get_progress_summary()
                iteration_stats.update(
                    {
                        "eta_seconds": eta_summary["eta_seconds"],
                        "progress_percent": eta_summary["progress_percent"],
                        "completion_time": (
                            eta_summary["completion_time"].isoformat()
                            if eta_summary["completion_time"]
                            else None
                        ),
                    }
                )

            self.training_history.append(iteration_stats)

            if self.config.verbose:
                self._print_iteration_summary(iteration_stats)

                # Print ETA update
                if (
                    self.eta_tracker
                    and iteration % self.config.eta_update_frequency == 0
                ):
                    self.eta_tracker.print_progress_update(
                        detailed=self.config.eta_detailed_display
                    )

        # Final evaluation and save
        final_eval = self._evaluate_model()
        self._save_final_model()

        total_time = time.time() - start_time
        self.total_training_time = total_time

        return {
            "iterations_completed": self.config.max_iterations,
            "total_training_time": total_time,
            "final_evaluation": final_eval,
            "training_history": self.training_history,
            "best_eval_score": self.best_eval_score,
        }

    def _generate_self_play_data(self) -> None:
        """Generate self-play games and add to replay buffer."""
        if self.config.verbose:
            print(
                f"Generating {self.config.self_play_games_per_iteration} self-play games..."
            )

        start_time = time.time()

        self.self_play_engine.play_games(
            num_games=self.config.self_play_games_per_iteration,
            num_players=self.config.num_players,
        )

        generation_time = time.time() - start_time
        buffer_stats = self.replay_buffer.get_statistics()

        if self.config.verbose:
            print(f"Self-play generation completed in {generation_time:.2f}s")
            print(f"Buffer size: {buffer_stats['size']}/{buffer_stats['capacity']}")
            if "num_wins" in buffer_stats:
                print(
                    f"Win rate balance: {buffer_stats['num_wins']}/{buffer_stats['num_losses']}/{buffer_stats['num_draws']}"
                )

    def _train_network(self) -> Dict[str, Any]:
        """
        Train the neural network using enhanced training utilities.

        This method implements:
        - Batch sampling from replay buffer with (state, MCTS_policy_target, game_outcome)
        - Cross-entropy loss for policy head, MSE for value head
        - Training step: forward pass, loss calculation, backward pass, optimizer step
        """
        if self.config.verbose:
            print(
                f"Training network for {self.config.training_steps_per_iteration} steps..."
            )

        self.neural_network.set_training_mode(True)

        # Use enhanced training step
        start_time = time.time()
        training_stats = self.training_step.train_for_steps(
            self.config.training_steps_per_iteration
        )
        training_time = time.time() - start_time

        self.neural_network.set_training_mode(False)

        # Add timing information
        training_stats["training_time"] = training_time

        if self.config.verbose:
            print(f"Training completed in {training_time:.2f}s")
            print(f"  Policy Loss (cross-entropy): {training_stats['policy_loss']:.6f}")
            print(f"  Value Loss (MSE): {training_stats['value_loss']:.6f}")
            print(f"  Total Loss: {training_stats['total_loss']:.6f}")
            print(
                f"  Successful steps: {training_stats['successful_steps']}/{self.config.training_steps_per_iteration}"
            )

        return training_stats

    def _evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the current model performance."""
        if self.config.verbose:
            print(f"Evaluating model with {self.config.eval_games} games...")

        # Create temporary evaluation engine
        eval_buffer = ReplayBuffer(capacity=1000, min_size_for_sampling=1)
        eval_engine = SelfPlayEngine(
            neural_network=self.neural_network,
            replay_buffer=eval_buffer,
            mcts_simulations=self.config.mcts_simulations // 2,  # Faster evaluation
            temperature=0.0,  # Deterministic play
            verbose=False,
        )

        # Play evaluation games
        start_time = time.time()
        eval_engine.play_games(self.config.eval_games, self.config.num_players)
        eval_time = time.time() - start_time

        # Calculate statistics
        eval_stats = eval_engine.get_statistics()

        # Calculate win rates and average score
        eval_score = 0.0
        if "avg_moves_per_game" in eval_stats:
            # Use average moves per game as a proxy for playing strength
            # (shorter games often indicate better play)
            avg_moves = eval_stats["avg_moves_per_game"]
            eval_score = max(0.0, 1.0 - (avg_moves - 30) / 100)  # Normalize to [0, 1]

        # Update best score
        if eval_score > self.best_eval_score:
            self.best_eval_score = eval_score

        result = {
            "eval_score": eval_score,
            "eval_time": eval_time,
            "eval_games": self.config.eval_games,
            "avg_moves_per_game": eval_stats.get("avg_moves_per_game", 0),
            "is_best": eval_score > self.best_eval_score,
        }

        if self.config.verbose:
            print(f"Evaluation completed in {eval_time:.2f}s")
            print(f"  Eval Score: {eval_score:.4f}")
            print(f"  Avg Moves/Game: {eval_stats.get('avg_moves_per_game', 0):.1f}")
            if result["is_best"]:
                print("  New best model!")

        return result

    def _save_checkpoint(
        self, training_stats: Dict[str, Any], eval_stats: Dict[str, Any]
    ) -> None:
        """Save checkpoint using enhanced checkpoint manager."""
        score = eval_stats.get("eval_score", 0.0)

        combined_stats = {
            **training_stats,
            **eval_stats,
            "iteration": self.iteration + 1,
        }

        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.neural_network.model,
            optimizer=self.training_step.optimizer,
            iteration=self.iteration + 1,
            stats=combined_stats,
            score=score,
        )

        if self.config.verbose:
            print(f"Checkpoint saved: {checkpoint_path}")

    def _save_best_model(self) -> None:
        """Save the best performing model (legacy method for compatibility)."""
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint and self.config.verbose:
            print(f"Best model available at: {best_checkpoint}")

    def _save_final_model(self) -> None:
        """Save the final model using enhanced checkpoint manager."""
        final_stats = {
            "iteration": self.iteration + 1,
            "total_training_time": self.total_training_time,
            "best_eval_score": self.best_eval_score,
            "total_games_played": self.self_play_engine.games_played,
        }

        final_path = self.checkpoint_manager.save_checkpoint(
            model=self.neural_network.model,
            optimizer=self.training_step.optimizer,
            iteration=self.iteration + 1,
            stats=final_stats,
            score=self.best_eval_score,
        )

        # Save training history
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        if self.config.verbose:
            print(f"Final model saved: {final_path}")
            print(f"Training history saved: {history_path}")

    def _print_iteration_summary(self, stats: Dict[str, Any]) -> None:
        """Print summary of iteration statistics."""
        print(f"\nIteration {stats['iteration']} Summary:")
        print(f"  Time: {stats['iteration_time']:.2f}s")
        print(f"  Buffer Size: {stats['buffer_size']}")
        print(f"  Games Played: {stats['games_played']}")

        if "total_loss" in stats:
            print(f"  Training Loss: {stats['total_loss']:.6f}")
            print(f"  Policy Loss: {stats['policy_loss']:.6f}")
            print(f"  Value Loss: {stats['value_loss']:.6f}")

        if "eval_score" in stats:
            print(f"  Eval Score: {stats['eval_score']:.4f}")

        if stats.get("is_best", False):
            print("  🏆 New best model!")

        # Add ETA information if available
        if "eta_seconds" in stats and stats["eta_seconds"]:
            if self.eta_tracker:
                eta_str = self.eta_tracker.format_time_display(stats["eta_seconds"])
                print(f"  ETA: {eta_str}")

        if "progress_percent" in stats:
            print(f"  Progress: {stats['progress_percent']:.1f}%")

        if "completion_time" in stats and stats["completion_time"]:
            from datetime import datetime

            try:
                completion_time = datetime.fromisoformat(stats["completion_time"])
                print(f"  Est. completion: {completion_time.strftime('%H:%M:%S')}")
            except (ValueError, TypeError):
                pass


def create_training_config(**kwargs) -> TrainingConfig:
    """
    Create a training configuration with custom parameters.

    Args:
        **kwargs: Configuration parameters to override defaults

    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(**kwargs)


def run_training_example():
    """
    Run a complete example using the training system.

    This demonstrates all the required functionality:
    - Batch sampling from replay buffer
    - Cross-entropy loss for policy, MSE for value
    - Complete training steps
    - Regular checkpoint saving
    """
    print("Training Example")
    print("=" * 50)

    # Create configuration
    config = create_training_config(
        self_play_games_per_iteration=25,
        mcts_simulations=200,
        buffer_capacity=5000,
        min_buffer_size=500,
        batch_size=64,
        training_steps_per_iteration=100,
        max_iterations=20,
        eval_frequency=5,
        save_frequency=5,
        network_config="small",
        device="cpu",
        verbose=True,
    )

    # Create trainer
    trainer = AzulTrainer(config=config, save_dir="models/training_example")

    # Run training
    results = trainer.train()

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Total time: {results['total_training_time']:.2f}s")
    print(f"Best eval score: {results['best_eval_score']:.4f}")
    print(f"Iterations: {results['iterations_completed']}")

    return results


if __name__ == "__main__":
    run_training_example()
