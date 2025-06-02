#!/usr/bin/env python3
"""
AlphaZero training using OpenSpiel's implementation.

This script trains an AlphaZero model for Azul using OpenSpiel's
mature and optimized implementation instead of our custom one.
"""

import os
import time
from typing import Any, Dict

import numpy as np
from absl import app, flags

# Our game implementation
from game.azul_openspiel import AzulGame

# OpenSpiel imports
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.utils import data_logger

FLAGS = flags.FLAGS

# Training configuration
flags.DEFINE_string(
    "checkpoint_dir", "models/openspiel_alphazero", "Directory to save checkpoints"
)
flags.DEFINE_integer("num_iterations", 100, "Number of training iterations")
flags.DEFINE_integer(
    "num_self_play_games", 100, "Number of self-play games per iteration"
)
flags.DEFINE_integer("num_mcts_simulations", 400, "Number of MCTS simulations per move")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for neural network")
flags.DEFINE_integer("batch_size", 32, "Batch size for training")
flags.DEFINE_integer("train_steps", 100, "Training steps per iteration")
flags.DEFINE_integer("checkpoint_freq", 10, "Checkpoint frequency")
flags.DEFINE_integer("eval_freq", 10, "Evaluation frequency")
flags.DEFINE_integer("num_eval_games", 20, "Number of evaluation games")
flags.DEFINE_float("c_puct", 1.0, "PUCT exploration constant")
flags.DEFINE_float("dirichlet_alpha", 0.3, "Dirichlet noise alpha")
flags.DEFINE_float("dirichlet_epsilon", 0.25, "Dirichlet noise epsilon")
flags.DEFINE_string("model_type", "mlp", "Neural network architecture (mlp, resnet)")
flags.DEFINE_integer("nn_width", 256, "Neural network width")
flags.DEFINE_integer("nn_depth", 4, "Neural network depth")
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

        # Create data logger
        self.data_logger = data_logger.DataLogger(
            FLAGS.checkpoint_dir, "azul_alphazero", True  # Write to disk
        )

        # Ensure checkpoint directory exists
        os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)

    def _create_model(self) -> az_model.Model:
        """Create the neural network model."""
        return az_model.Model.build_model(
            model_type=self.config["model_type"],
            observation_shape=self.game.observation_tensor_shape(),
            num_actions=self.game.num_distinct_actions(),
            nn_width=self.config["nn_width"],
            nn_depth=self.config["nn_depth"],
            learning_rate=self.config["learning_rate"],
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
        print(f"Configuration: {self.config}")

        # Create AlphaZero learner
        learner = alpha_zero.AlphaZero(
            game=self.game,
            model=self.model,
            replay_buffer_capacity=1000000,
            **{
                k: v
                for k, v in self.config.items()
                if k
                in [
                    "num_mcts_simulations",
                    "c_puct",
                    "dirichlet_alpha",
                    "dirichlet_epsilon",
                    "num_self_play_games",
                    "batch_size",
                    "train_steps",
                ]
            },
        )

        start_time = time.time()

        for iteration in range(self.config["num_iterations"]):
            iter_start = time.time()

            print(
                f"\n=== Iteration {iteration + 1}/{self.config['num_iterations']} ==="
            )

            # Self-play
            print("Running self-play...")
            self_play_start = time.time()
            learner.generate_self_play_games(self.config["num_self_play_games"])
            self_play_time = time.time() - self_play_start
            print(f"Self-play completed in {self_play_time:.2f}s")

            # Training
            print("Training neural network...")
            train_start = time.time()
            for _ in range(self.config["train_steps"]):
                learner.train_network()
            train_time = time.time() - train_start
            print(f"Training completed in {train_time:.2f}s")

            # Logging
            iter_time = time.time() - iter_start
            total_time = time.time() - start_time

            self.data_logger.write(
                {
                    "iteration": iteration + 1,
                    "self_play_time": self_play_time,
                    "train_time": train_time,
                    "iteration_time": iter_time,
                    "total_time": total_time,
                }
            )

            print(
                f"Iteration {iteration + 1} completed in {iter_time:.2f}s "
                f"(total: {total_time:.2f}s)"
            )

            # Checkpointing
            if (iteration + 1) % self.config["checkpoint_freq"] == 0:
                checkpoint_path = os.path.join(
                    FLAGS.checkpoint_dir, f"checkpoint_{iteration + 1}.pkl"
                )
                self._save_checkpoint(checkpoint_path, iteration + 1, learner)
                print(f"Saved checkpoint: {checkpoint_path}")

            # Evaluation
            if (iteration + 1) % self.config["eval_freq"] == 0:
                print("Running evaluation...")
                eval_results = self._evaluate_model(learner)
                self.data_logger.write(
                    {f"eval_iteration_{iteration + 1}": eval_results}
                )
                print(f"Evaluation results: {eval_results}")

        # Final checkpoint
        final_checkpoint = os.path.join(FLAGS.checkpoint_dir, "final_model.pkl")
        self._save_checkpoint(final_checkpoint, self.config["num_iterations"], learner)
        print(f"Training completed! Final model saved: {final_checkpoint}")

        return learner

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

    def _evaluate_model(self, learner) -> Dict[str, float]:
        """Evaluate the current model."""
        try:
            # Simple evaluation: play against random player
            from agents.openspiel_agents import OpenSpielMCTSAgent, RandomAgent

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
                    agents = [alphazero_agent, random_agent]
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
