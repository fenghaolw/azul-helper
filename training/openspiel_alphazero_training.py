#!/usr/bin/env python3
"""
AlphaZero training using OpenSpiel's implementation.

This script trains an AlphaZero model for Azul using OpenSpiel's
mature and optimized implementation instead of our custom one.
"""

import os
import sys
import time
from typing import Any, Dict, Union

import numpy as np
from absl import app, flags

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# OpenSpiel imports
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.utils import data_logger

# Import agents for type annotations
from agents.openspiel_agents import OpenSpielMCTSAgent, RandomAgent

# Our game implementation
from game.azul_openspiel import AzulGame

FLAGS = flags.FLAGS

# Training configuration
flags.DEFINE_string(
    "checkpoint_dir", "models/openspiel_alphazero", "Directory to save checkpoints"
)
flags.DEFINE_string("model_type", "resnet", "Model type (mlp, conv2d, resnet)")
flags.DEFINE_float("learning_rate", 2e-4, "Learning rate")
flags.DEFINE_integer(
    "nn_width", 64, "Neural network width (reduced for faster training)"
)
flags.DEFINE_integer(
    "nn_depth", 3, "Neural network depth (reduced for faster training)"
)
flags.DEFINE_integer(
    "num_mcts_simulations",
    100,
    "Number of MCTS simulations per move (reduced for faster training)",
)
flags.DEFINE_float("c_puct", 1.0, "Exploration constant for PUCT")
flags.DEFINE_float("dirichlet_alpha", 0.3, "Alpha parameter for Dirichlet noise")
flags.DEFINE_float("dirichlet_epsilon", 0.25, "Epsilon for Dirichlet noise")
flags.DEFINE_integer(
    "num_iterations", 10, "Number of training iterations (reduced for testing)"
)
flags.DEFINE_integer(
    "num_self_play_games",
    10,
    "Number of self-play games per iteration (reduced for testing)",
)
flags.DEFINE_integer("batch_size", 16, "Training batch size (reduced for memory)")
flags.DEFINE_integer(
    "train_steps", 10, "Training steps per iteration (reduced for testing)"
)
flags.DEFINE_integer("checkpoint_freq", 1, "Checkpoint frequency (iterations)")
flags.DEFINE_integer("eval_freq", 1, "Evaluation frequency (iterations)")
flags.DEFINE_integer(
    "num_eval_games", 5, "Number of evaluation games (reduced for testing)"
)
flags.DEFINE_integer(
    "num_actors", 2, "Number of self-play actors (reduced for resource constraints)"
)
flags.DEFINE_integer(
    "num_evaluators", 1, "Number of evaluators (reduced for resource constraints)"
)
flags.DEFINE_integer("num_players", 2, "Number of players")
flags.DEFINE_integer("seed", 42, "Random seed")


def create_azul_config() -> Dict[str, Any]:
    """Create configuration for Azul AlphaZero training."""
    return {
        # Game configuration
        "game": "azul",
        "game_params": {
            "players": FLAGS.num_players,
        },
        # Model configuration
        "model_type": FLAGS.model_type,
        "nn_width": FLAGS.nn_width,
        "nn_depth": FLAGS.nn_depth,
        "learning_rate": FLAGS.learning_rate,
        # MCTS configuration
        "num_mcts_simulations": FLAGS.num_mcts_simulations,
        "c_puct": FLAGS.c_puct,
        "dirichlet_alpha": FLAGS.dirichlet_alpha,
        "dirichlet_epsilon": FLAGS.dirichlet_epsilon,
        # Training configuration
        "num_iterations": FLAGS.num_iterations,
        "num_self_play_games": FLAGS.num_self_play_games,
        "batch_size": FLAGS.batch_size,
        "train_steps": FLAGS.train_steps,
        "checkpoint_freq": FLAGS.checkpoint_freq,
        "eval_freq": FLAGS.eval_freq,
        "num_eval_games": FLAGS.num_eval_games,
        # Other
        "seed": FLAGS.seed,
        "verbose": True,
    }


class AzulAlphaZeroTrainer:
    """AlphaZero trainer for Azul using OpenSpiel."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the trainer."""
        self.config = config

        # Create game
        self.game = AzulGame(config["game_params"])
        print(f"Created game: {self.game}")

        # Create model
        self.model = self._create_model()
        print(f"Created model with {self._count_parameters()} parameters")

        # Create data logger using DataLoggerJsonLines
        log_dir = os.path.join(FLAGS.checkpoint_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.data_logger = data_logger.DataLoggerJsonLines(
            log_dir, "azul_alphazero", True  # Write to disk
        )

        # Ensure checkpoint directory exists
        os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)

    def _create_model(self) -> az_model.Model:
        """Create the neural network model."""
        # Create a temporary directory for the model
        import os
        import tempfile

        temp_dir = tempfile.mkdtemp(prefix="azul_az_")
        model_path = os.path.join(temp_dir, "model")

        # Get the observation shape and ensure it's a list
        obs_shape = list(self.game.observation_tensor_shape())

        return az_model.Model.build_model(
            model_type=self.config["model_type"],
            input_shape=obs_shape,
            output_size=self.game.num_distinct_actions(),
            nn_width=self.config["nn_width"],
            nn_depth=self.config["nn_depth"],
            weight_decay=self.config.get(
                "weight_decay", 0.0001
            ),  # Default weight decay
            learning_rate=self.config["learning_rate"],
            path=model_path,
        )

    def _count_parameters(self) -> int:
        """Count the number of parameters in the model."""
        try:
            return sum(p.numel() for p in self.model.parameters())
        except AttributeError:
            return 0  # If model doesn't have parameters method

    def train(self):
        """Run the main training loop."""
        print("Starting AlphaZero training for Azul...")
        print("Configuration:")
        for k, v in self.config.items():
            print(f"  {k}: {v}")
        print("\n")

        # Create checkpoint directory
        import os

        checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")

        # Print system information
        import sys

        import tensorflow as tf

        print("\nSystem information:")
        print(f"Python version: {sys.version}")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")
        print("\n")

        # Create AlphaZero config with reduced parameters for testing
        try:
            print("Creating AlphaZero configuration...")
            config = alpha_zero.Config(
                game=self.game.get_type().short_name,
                path=checkpoint_dir,
                learning_rate=self.config["learning_rate"],
                weight_decay=self.config.get("weight_decay", 0.0001),
                train_batch_size=self.config["batch_size"],
                replay_buffer_size=10000,  # Reduced for testing
                replay_buffer_reuse=1,
                max_steps=self.config["num_iterations"],
                checkpoint_freq=self.config["checkpoint_freq"],
                actors=min(2, self.config.get("num_actors", 2)),  # Reduced for testing
                evaluators=min(
                    1, self.config.get("num_evaluators", 1)
                ),  # Reduced for testing
                evaluation_window=5,  # Reduced for testing
                eval_levels=3,  # Reduced for testing
                uct_c=self.config["c_puct"],
                max_simulations=self.config["num_mcts_simulations"],
                policy_alpha=self.config["dirichlet_alpha"],
                policy_epsilon=self.config["dirichlet_epsilon"],
                temperature=1.0,
                temperature_drop=5,  # Reduced for testing
                nn_model=self.config["model_type"],
                nn_width=self.config["nn_width"],
                nn_depth=self.config["nn_depth"],
                observation_shape=list(self.game.observation_tensor_shape()),
                output_size=self.game.num_distinct_actions(),
                quiet=False,  # Enable verbose logging
            )
            print("Configuration created successfully!")
            print("\nStarting training...")

            # Start training - this will run the full AlphaZero training loop
            alpha_zero.alpha_zero(config)

        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback

            traceback.print_exc()
            print("\nTraining failed. Please check the error message above.")
            print("Common issues:")
            print("1. Out of memory: Try reducing batch_size, nn_width, or nn_depth")
            print("2. TensorFlow version mismatch: Check OpenSpiel's requirements")
            print(
                "3. GPU issues: Try running with CPU only (set CUDA_VISIBLE_DEVICES='')"
            )
            raise

    def _save_checkpoint(self, path: str, iteration: int, learner):
        """Save model checkpoint."""
        try:
            # Save model state
            checkpoint_data = {
                "iteration": iteration,
                "model_state": (
                    learner.model.state_dict()
                    if hasattr(learner.model, "state_dict")
                    else None
                ),
                "config": self.config,
            }

            # Save using pickle or torch.save depending on model type
            import pickle

            with open(path, "wb") as f:
                pickle.dump(checkpoint_data, f)

        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")

    def _evaluate_model(self, learner) -> Dict[str, Union[float, int, str]]:
        """Evaluate the current model."""
        try:
            # Simple evaluation: play against random player
            # Create agents
            alphazero_agent = OpenSpielMCTSAgent(
                num_simulations=self.config["num_mcts_simulations"]
                // 4,  # Faster for eval
                evaluator=learner.evaluator,  # Use trained evaluator
            )
            random_agent = RandomAgent(seed=self.config["seed"])

            wins = 0
            games_played = 0
            total_game_length = 0

            for game_num in range(self.config["num_eval_games"]):
                from game.game_state import GameState

                game_state = GameState(
                    num_players=self.config["game_params"]["players"],
                    seed=self.config["seed"] + game_num,
                )

                # Alternate who goes first
                if game_num % 2 == 0:
                    agents: list[Union[OpenSpielMCTSAgent, RandomAgent]] = [
                        alphazero_agent,
                        random_agent,
                    ]
                else:
                    agents = [random_agent, alphazero_agent]

                moves = 0
                max_moves = 200

                while not game_state.game_over and moves < max_moves:
                    current_player = game_state.current_player
                    agent = agents[current_player]

                    action = agent.select_action(game_state, deterministic=True)
                    game_state.apply_action(action)
                    moves += 1

                if game_state.game_over:
                    scores = game_state.get_scores()
                    alphazero_player = 0 if game_num % 2 == 0 else 1

                    if scores[alphazero_player] > scores[1 - alphazero_player]:
                        wins += 1

                    games_played += 1
                    total_game_length += moves

            win_rate = wins / max(1, games_played)
            avg_game_length = total_game_length / max(1, games_played)

            return {
                "win_rate_vs_random": win_rate,
                "games_played": games_played,
                "avg_game_length": avg_game_length,
            }

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {"error": str(e)}


def main(argv):
    """Main training function."""
    print("OpenSpiel AlphaZero Training for Azul")
    print("====================================")

    # Create configuration
    config = create_azul_config()

    # Create and run trainer
    trainer = AzulAlphaZeroTrainer(config)
    trainer.train()

    print("Training completed successfully!")
    print(f"Model saved in: {FLAGS.checkpoint_dir}")
    print(
        f"To use the trained model, load it from: {FLAGS.checkpoint_dir}/final_model.pkl"
    )


if __name__ == "__main__":
    app.run(main)
