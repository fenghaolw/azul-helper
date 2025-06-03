#!/usr/bin/env python3
"""
AlphaZero training using OpenSpiel's implementation.

This script trains an AlphaZero model for Azul using OpenSpiel's
mature and optimized implementation.
"""

import os
import sys
import time
from typing import Any, Dict

from absl import app, flags

# Add parent directory to path so we can import from game
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Register our custom Azul game with OpenSpiel
from game import register_azul

# Our game implementation
from game.azul_openspiel import AzulGame

# OpenSpiel imports
from open_spiel.python.algorithms.alpha_zero import alpha_zero

FLAGS = flags.FLAGS

# Training configuration
flags.DEFINE_string(
    "checkpoint_dir", "models/openspiel_alphazero", "Directory to save checkpoints"
)
flags.DEFINE_integer("max_steps", 1000, "Maximum number of training steps")
flags.DEFINE_integer("actors", 2, "Number of actor processes")
flags.DEFINE_integer("evaluators", 1, "Number of evaluator processes")
flags.DEFINE_integer("max_simulations", 400, "Number of MCTS simulations per move")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for neural network")
flags.DEFINE_integer("train_batch_size", 32, "Batch size for training")
flags.DEFINE_integer("replay_buffer_size", 100000, "Size of replay buffer")
flags.DEFINE_integer("checkpoint_freq", 100, "Checkpoint frequency")
flags.DEFINE_float("uct_c", 1.4, "UCT exploration constant")
flags.DEFINE_float("policy_alpha", 0.3, "Dirichlet noise alpha")
flags.DEFINE_float("policy_epsilon", 0.25, "Dirichlet noise epsilon")
flags.DEFINE_float("temperature", 1.0, "MCTS temperature")
flags.DEFINE_integer("temperature_drop", 20, "Move when temperature drops")
flags.DEFINE_string("nn_model", "resnet", "Neural network architecture (mlp, resnet)")
flags.DEFINE_integer("nn_width", 256, "Neural network width")
flags.DEFINE_integer("nn_depth", 6, "Neural network depth")
flags.DEFINE_float("weight_decay", 1e-4, "Weight decay for training")
flags.DEFINE_integer("evaluation_window", 100, "Evaluation window")
flags.DEFINE_integer("eval_levels", 3, "Number of evaluation levels")
flags.DEFINE_integer("replay_buffer_reuse", 10, "Times to reuse replay buffer")
flags.DEFINE_boolean("quiet", False, "Suppress output")


def create_azul_config() -> alpha_zero.Config:
    """Create configuration for Azul AlphaZero training."""

    # Load our registered Azul game
    import pyspiel

    game = pyspiel.load_game("azul")

    # Ensure checkpoint directory exists
    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)

    return alpha_zero.Config(
        game="azul",  # Use our registered Azul game
        path=FLAGS.checkpoint_dir,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        train_batch_size=FLAGS.train_batch_size,
        replay_buffer_size=FLAGS.replay_buffer_size,
        replay_buffer_reuse=FLAGS.replay_buffer_reuse,
        max_steps=FLAGS.max_steps,
        checkpoint_freq=FLAGS.checkpoint_freq,
        actors=FLAGS.actors,
        evaluators=FLAGS.evaluators,
        evaluation_window=FLAGS.evaluation_window,
        eval_levels=FLAGS.eval_levels,
        uct_c=FLAGS.uct_c,
        max_simulations=FLAGS.max_simulations,
        policy_alpha=FLAGS.policy_alpha,
        policy_epsilon=FLAGS.policy_epsilon,
        temperature=FLAGS.temperature,
        temperature_drop=FLAGS.temperature_drop,
        nn_model=FLAGS.nn_model,
        nn_width=FLAGS.nn_width,
        nn_depth=FLAGS.nn_depth,
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions(),
        quiet=FLAGS.quiet,
    )


def main(argv):
    """Main training function."""
    print("OpenSpiel AlphaZero Training for Azul")
    print("====================================")

    # Create configuration
    config = create_azul_config()

    print(f"Game: {config.game}")
    print(f"Observation shape: {config.observation_shape}")
    print(f"Number of actions: {config.output_size}")
    print(f"Model type: {config.nn_model}")
    print(f"Network width: {config.nn_width}, depth: {config.nn_depth}")
    print(f"Training steps: {config.max_steps}")
    print(f"Actors: {config.actors}, Evaluators: {config.evaluators}")
    print(f"Checkpoint directory: {config.path}")
    print()

    # Start training
    print("Starting AlphaZero training...")
    start_time = time.time()

    try:
        alpha_zero.alpha_zero(config)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds!")
    print(f"Model saved in: {FLAGS.checkpoint_dir}")


if __name__ == "__main__":
    app.run(main)
