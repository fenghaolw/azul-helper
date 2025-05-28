"""
Comprehensive MCTS Example for Azul Game.

This module demonstrates how to use the MCTS algorithm with PyTorch neural networks
for the Azul board game, including gameplay, benchmarking, and advanced features.
"""

import time
from typing import List, Tuple

import numpy as np

from azul_rl.agents.mcts import MCTS, AzulNeuralNetwork, MCTSAgent
from azul_rl.game.game_state import Action
from azul_rl.game.game_state import GameState as AzulGameState
from azul_rl.training.neural_network import create_azul_network


def run_basic_mcts_example():
    """
    Run a basic example of MCTS with Azul game states.
    """
    print("MCTS with Azul Game - Basic Example")
    print("=" * 60)

    # Create initial Azul game state
    azul_state = AzulGameState(num_players=2, seed=42)

    print(f"Initial state: {azul_state}")
    print(f"Current player: {azul_state.current_player}")
    print(f"Legal actions: {len(azul_state.get_legal_actions())} available")

    # Show some example actions
    legal_actions = azul_state.get_legal_actions()
    print(f"\nFirst few legal actions: {legal_actions[:5]}")

    # Create PyTorch neural network for Azul
    neural_network = AzulNeuralNetwork(config_name="small", device="cpu")
    print(f"Neural network info: {neural_network.get_model_info()}")

    # Test neural network evaluation
    action_probs, value = neural_network.evaluate(azul_state)
    print("\nNeural network evaluation:")
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"State value: {value:.3f}")
    print(f"Probabilities sum: {action_probs.sum():.3f}")

    print("Top 3 action probabilities:")
    top_indices = np.argsort(action_probs)[-3:][::-1]
    for i in top_indices:
        action = legal_actions[i]
        print(f"  Action {i} ({action}): {action_probs[i]:.3f}")

    # Create MCTS instance
    mcts = MCTS(
        neural_network=neural_network,
        c_puct=1.0,
        num_simulations=50,  # Moderate number for demonstration
        temperature=1.0,
    )

    # Run MCTS search
    print(f"\nRunning MCTS with {mcts.num_simulations} simulations...")
    improved_probs, root_node = mcts.search(azul_state)

    print("\nMCTS Results:")
    print(f"Root node statistics: {root_node}")
    print(f"Number of children explored: {len(root_node.children)}")

    # Show top actions after MCTS
    print("\nTop 3 actions after MCTS:")
    top_mcts_indices = np.argsort(improved_probs)[-3:][::-1]
    for i in top_mcts_indices:
        if i < len(legal_actions) and improved_probs[i] > 0:
            action = legal_actions[i]
            print(f"  Action {i} ({action}): {improved_probs[i]:.3f}")


def demonstrate_gameplay():
    """
    Demonstrate MCTS agent playing an Azul game.
    """
    print("\nMCTS Agent Gameplay Demonstration")
    print("=" * 60)

    # Create neural network and agent
    neural_network = AzulNeuralNetwork(config_name="small", device="cpu")
    agent = MCTSAgent(neural_network, num_simulations=20)

    # Start a new game
    azul_state = AzulGameState(num_players=2, seed=42)
    move_count = 0

    print("Playing Azul game with MCTS agent:")

    while not azul_state.game_over and move_count < 10:
        print(f"\n--- Move {move_count + 1} ---")
        print(f"Round: {azul_state.round_number}")
        print(f"Current player: {azul_state.current_player}")
        print(f"Available actions: {len(azul_state.get_legal_actions())}")

        # Get action from agent
        action = agent.select_action(azul_state, deterministic=False)
        print(f"Selected action: {action}")

        # Apply action
        success = azul_state.apply_action(action)
        if not success:
            print(f"Failed to apply action: {action}")
            break

        move_count += 1

        # Show some game state info
        scores = azul_state.get_scores()
        print(f"Current scores: Player 0: {scores[0]}, Player 1: {scores[1]}")

    print(f"\nFinal state: {azul_state}")
    if azul_state.game_over:
        scores = azul_state.get_scores()
        print(f"Game Over! Final scores: {scores}")
        winner_idx = scores.index(max(scores))
        print(f"Player {winner_idx} wins!")
    else:
        print("Game still in progress after 10 moves.")


def compare_network_configurations():
    """
    Compare different neural network configurations.
    """
    print("\nNeural Network Configuration Comparison")
    print("=" * 60)

    azul_state = AzulGameState(num_players=2, seed=42)
    configs = ["small", "medium", "large"]

    for config in configs:
        print(f"\n{config.capitalize()} Network:")
        neural_network = AzulNeuralNetwork(config_name=config, device="cpu")

        # Get model info
        model_info = neural_network.get_model_info()
        print(f"  Parameters: {model_info.get('total_parameters', 'N/A')}")
        print(f"  Hidden sizes: {model_info.get('hidden_sizes', 'N/A')}")

        # Test evaluation performance
        start_time = time.time()
        action_probs, value = neural_network.evaluate(azul_state)
        eval_time = time.time() - start_time

        print(f"  Evaluation time: {eval_time:.4f}s")
        print(f"  Probabilities shape: {action_probs.shape}")
        print(f"  Value: {value:.3f}")
        print(f"  Top 3 probs: {np.sort(action_probs)[-3:][::-1]}")


def benchmark_mcts_performance():
    """
    Benchmark MCTS performance with different configurations.
    """
    print("\nMCTS Performance Benchmark")
    print("=" * 60)

    azul_state = AzulGameState(num_players=2, seed=42)

    # Test different network sizes and simulation counts
    configs = ["small", "medium"]
    simulation_counts = [5, 10, 20, 50]

    for config in configs:
        print(f"\nTesting {config} network:")
        neural_network = AzulNeuralNetwork(config_name=config, device="cpu")

        for num_sims in simulation_counts:
            mcts = MCTS(
                neural_network=neural_network, num_simulations=num_sims, temperature=1.0
            )

            start_time = time.time()
            action_probs, root_node = mcts.search(azul_state)
            end_time = time.time()

            elapsed = end_time - start_time
            sims_per_sec = num_sims / elapsed if elapsed > 0 else float("inf")

            print(
                f"  Simulations: {num_sims:2d} | "
                f"Time: {elapsed:.3f}s | "
                f"Sims/sec: {sims_per_sec:.1f} | "
                f"Root visits: {root_node.N} | "
                f"Children: {len(root_node.children)}"
            )


def demonstrate_mcts_features():
    """
    Demonstrate key MCTS features and capabilities.
    """
    print("\nMCTS Features Demonstration")
    print("=" * 60)

    azul_state = AzulGameState(num_players=2, seed=42)
    neural_network = AzulNeuralNetwork(config_name="small", device="cpu")

    # Temperature effects
    print("Temperature Effects:")
    print("Note: Temperature effects are only visible when MCTS visits create")
    print("      uneven visit counts. Using more simulations to show this effect.")

    # Use more simulations to get meaningful visit count differences
    num_sims_for_temp_demo = 100  # Increased from 20

    # High temperature (more exploration)
    agent_hot = MCTSAgent(
        neural_network, num_simulations=num_sims_for_temp_demo, temperature=2.0
    )
    probs_hot = agent_hot.get_action_probabilities(azul_state)

    # Low temperature (more exploitation)
    agent_cold = MCTSAgent(
        neural_network, num_simulations=num_sims_for_temp_demo, temperature=0.1
    )
    probs_cold = agent_cold.get_action_probabilities(azul_state)

    entropy_hot = -sum(p * np.log(p + 1e-10) for p in probs_hot if p > 0)
    entropy_cold = -sum(p * np.log(p + 1e-10) for p in probs_cold if p > 0)

    print(
        f"  High temp (2.0) - max prob: {max(probs_hot):.3f}, entropy: {entropy_hot:.3f}"
    )
    print(
        f"  Low temp (0.1) - max prob: {max(probs_cold):.3f}, entropy: {entropy_cold:.3f}"
    )

    # Explain what we expect to see
    if entropy_cold < entropy_hot:
        print(
            "  ✓ Temperature working correctly: lower temp has lower entropy (more focused)"
        )
    else:
        print(
            "  ⚠ Temperature effect minimal - may need more simulations for uneven visit counts"
        )

    # Deterministic vs stochastic selection
    print("\nAction Selection:")

    agent = MCTSAgent(neural_network, num_simulations=10)

    # Deterministic selection
    det_action1 = agent.select_action(azul_state, deterministic=True)
    det_action2 = agent.select_action(azul_state, deterministic=True)
    print(f"  Deterministic: {det_action1 == det_action2} (same action selected)")

    # Stochastic selection
    stoch_actions = [
        agent.select_action(azul_state, deterministic=False) for _ in range(5)
    ]
    unique_actions = len(set(str(action) for action in stoch_actions))
    print(f"  Stochastic: {unique_actions}/5 unique actions selected")


def demonstrate_numerical_state():
    """
    Demonstrate the numerical state representation integration.
    """
    print("\nNumerical State Representation Integration")
    print("=" * 60)

    azul_state = AzulGameState(num_players=2, seed=42)

    # Get numerical representation
    numerical_state = azul_state.get_numerical_state()
    print(f"Numerical state type: {type(numerical_state)}")

    # Get flat vector
    state_vector = numerical_state.get_flat_state_vector(normalize=True)
    print(f"State vector shape: {state_vector.shape}")
    print(f"State vector size: {len(state_vector)}")
    print(f"Value range: [{state_vector.min():.3f}, {state_vector.max():.3f}]")

    # Show how neural network uses this
    neural_network = AzulNeuralNetwork(config_name="small", device="cpu")
    action_probs, value = neural_network.evaluate(azul_state)

    print("\nNeural network leverages numerical state:")
    print("  Input: Azul GameState object")
    print("  Internal: Uses get_numerical_state() for consistent evaluation")
    print(f"  Output: {len(action_probs)} action probabilities, value = {value:.3f}")

    # Show consistency
    action_probs2, value2 = neural_network.evaluate(azul_state)
    print("\nConsistency check:")
    print("  Same state evaluated twice")
    print(f"  Probabilities match: {np.allclose(action_probs, action_probs2)}")
    print(f"  Values match: {np.isclose(value, value2)}")


def demonstrate_model_saving():
    """
    Demonstrate saving and loading PyTorch models.
    """
    print("\nModel Saving/Loading Demonstration")
    print("=" * 60)

    # Create and save a model
    print("Creating and saving model...")
    neural_network = AzulNeuralNetwork(config_name="small", device="cpu")

    # Save model
    model_path = "/tmp/azul_mcts_model.pth"
    neural_network.save_model(model_path, epoch=0, training_step=0)

    # Load model
    print("Loading model...")
    loaded_network = AzulNeuralNetwork(
        config_name="small", model_path=model_path, device="cpu"
    )

    # Test that they produce the same results
    azul_state = AzulGameState(num_players=2, seed=42)

    original_probs, original_value = neural_network.evaluate(azul_state)
    loaded_probs, loaded_value = loaded_network.evaluate(azul_state)

    print(f"Original model value: {original_value:.6f}")
    print(f"Loaded model value: {loaded_value:.6f}")
    print(f"Values match: {np.isclose(original_value, loaded_value)}")
    print(f"Probabilities match: {np.allclose(original_probs, loaded_probs)}")


def demonstrate_custom_network():
    """
    Demonstrate creating custom neural network configurations.
    """
    print("\nCustom Neural Network Configuration")
    print("=" * 60)

    # Create custom model
    print("Creating custom neural network...")
    custom_model = create_azul_network("medium", hidden_sizes=(256, 128, 64))
    neural_network = AzulNeuralNetwork(model=custom_model, device="cpu")

    print(f"Custom network info: {neural_network.get_model_info()}")

    # Test evaluation
    azul_state = AzulGameState(num_players=2, seed=42)
    action_probs, value = neural_network.evaluate(azul_state)

    print("Custom network evaluation:")
    print(f"  Probabilities shape: {action_probs.shape}")
    print(f"  Value: {value:.3f}")
    print(f"  Probabilities sum: {action_probs.sum():.3f}")


if __name__ == "__main__":
    # Run all demonstrations
    run_basic_mcts_example()
    demonstrate_gameplay()
    compare_network_configurations()
    benchmark_mcts_performance()
    demonstrate_mcts_features()
    demonstrate_numerical_state()
    demonstrate_model_saving()
    demonstrate_custom_network()

    print("\n" + "=" * 60)
    print("All MCTS demonstrations completed successfully!")
    print("=" * 60)
