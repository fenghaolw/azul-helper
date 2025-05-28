#!/usr/bin/env python3
"""
Test script for the Azul self-play training system.

This script runs tests to verify that all self-play training components work together correctly.
"""

import os
import sys

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training import (
    AzulNeuralNetwork,
    AzulTrainer,
    ReplayBuffer,
    SelfPlayEngine,
    TrainingConfig,
    create_training_config,
)


def test_replay_buffer():
    """Test replay buffer functionality."""
    print("Testing Replay Buffer...")

    buffer = ReplayBuffer(capacity=100, min_size_for_sampling=5)

    # Check initial state
    assert buffer.size() == 0
    assert not buffer.is_ready_for_sampling()

    # Test statistics
    stats = buffer.get_statistics()
    assert stats["size"] == 0
    assert stats["games_added"] == 0

    print("✓ Replay buffer basic functionality works")


def test_self_play_engine():
    """Test self-play engine functionality."""
    print("Testing Self-Play Engine...")

    # Create components
    neural_network = AzulNeuralNetwork(config_name="small", device="cpu")
    replay_buffer = ReplayBuffer(capacity=100, min_size_for_sampling=5)

    engine = SelfPlayEngine(
        neural_network=neural_network,
        replay_buffer=replay_buffer,
        mcts_simulations=10,  # Very low for quick test
        temperature=1.0,
        verbose=False,
    )

    # Play a single game
    experiences = engine.play_game(num_players=2, seed=42)

    # Verify we got experiences
    assert len(experiences) > 0
    assert all(exp.outcome is not None for exp in experiences)

    # Verify buffer was updated
    assert replay_buffer.size() > 0

    # Check statistics
    stats = engine.get_statistics()
    assert stats["games_played"] == 1
    assert stats["total_moves"] > 0

    print("✓ Self-play engine functionality works")


def test_training_config():
    """Test training configuration."""
    print("Testing Training Configuration...")

    # Test default config
    config = create_training_config()
    assert config.self_play_games_per_iteration > 0
    assert config.mcts_simulations > 0
    assert config.max_iterations > 0

    # Test custom config
    custom_config = create_training_config(
        self_play_games_per_iteration=5, mcts_simulations=20, max_iterations=2
    )
    assert custom_config.self_play_games_per_iteration == 5
    assert custom_config.mcts_simulations == 20
    assert custom_config.max_iterations == 2

    print("✓ Training configuration works")


def test_minimal_training():
    """Test minimal training loop."""
    print("Testing Minimal Training Loop...")

    # Create very minimal config for quick test
    config = create_training_config(
        self_play_games_per_iteration=2,
        mcts_simulations=5,
        max_iterations=2,
        training_steps_per_iteration=3,
        batch_size=4,
        min_buffer_size=5,
        eval_games=2,
        eval_frequency=1,
        save_frequency=10,  # Don't save during test
        network_config="small",
        verbose=False,
    )

    # Create trainer with models directory
    trainer = AzulTrainer(config=config, save_dir="models/test_models")

    # Run minimal training
    results = trainer.train()

    # Verify results
    assert results["iterations_completed"] == 2
    assert "training_history" in results
    assert len(results["training_history"]) == 2

    print("✓ Minimal training loop works")


def main():
    """Run all tests."""
    print("Testing Azul Self-Play Training System")
    print("=" * 50)

    try:
        test_replay_buffer()
        test_self_play_engine()
        test_training_config()
        test_minimal_training()

        print("\n🎉 All tests passed!")
        print("\nThe self-play training system is working correctly.")
        print("You can now run the full example with:")
        print("  python azul_rl/examples/self_play_training_example.py")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
