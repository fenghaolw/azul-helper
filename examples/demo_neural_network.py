#!/usr/bin/env python3
"""
Demonstration of the Azul Neural Network Architecture.

This script shows how to use the AzulNetwork with real game states,
including state preprocessing, action prediction, and value estimation.
"""

import os
import sys

# Add parent directory to path so we can import from game
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402,I201

from game.game_state import GameState  # noqa: E402,I100
from game.pettingzoo_env import AzulAECEnv  # noqa: E402
from game.state_representation import AzulStateRepresentation  # noqa: E402
from training.neural_network import create_azul_network  # noqa: E402


def demonstrate_network_with_real_game():
    """Demonstrate the neural network with real game states."""
    print("=== Azul Neural Network Demonstration ===\n")

    # Create a game and play a few moves to get interesting states
    print("1. Creating game and generating states...")
    game = GameState(num_players=2, seed=42)

    # Collect states from actual gameplay
    states = []
    legal_action_masks = []

    for move_num in range(5):
        # Get current state representation
        state_repr = AzulStateRepresentation(game)
        state_vector = state_repr.get_flat_state_vector(normalize=True)
        states.append(state_vector)

        # Get legal actions and create mask
        legal_actions = game.get_legal_actions()
        action_mask = np.zeros(500)  # Action space size from PettingZoo env

        # Create PettingZoo environment to encode actions
        env = AzulAECEnv(num_players=2, seed=42)
        for action in legal_actions:
            action_idx = env._encode_action(action)
            if 0 <= action_idx < 500:
                action_mask[action_idx] = 1.0

        legal_action_masks.append(action_mask)

        print(f"  Move {move_num + 1}: {len(legal_actions)} legal actions")
        print(f"    Current player: {game.current_player}")
        print(f"    Round: {game.round_number}")
        print(f"    State vector size: {len(state_vector)}")

        # Apply a random legal action
        if legal_actions:
            action = np.random.choice(legal_actions)
            game.apply_action(action)
            print(f"    Applied action: {action}")

        if game.game_over:
            break
        print()

    print(f"\nCollected {len(states)} game states for demonstration.\n")

    # Convert to tensors
    states_tensor = torch.FloatTensor(np.array(states))
    masks_tensor = torch.FloatTensor(np.array(legal_action_masks))

    return states_tensor, masks_tensor


def test_different_network_configurations():
    """Test different network configurations."""
    print("2. Testing different network configurations...")

    configs = ["small", "medium", "large", "deep"]

    # Create dummy input for testing - use actual state size
    from game.game_state import GameState
    from game.state_representation import AzulStateRepresentation

    dummy_game = GameState(num_players=2, seed=42)
    dummy_repr = AzulStateRepresentation(dummy_game)
    actual_state_size = dummy_repr.flat_state_size
    dummy_state = torch.randn(1, actual_state_size)

    for config_name in configs:
        print(f"\n  Testing '{config_name}' configuration:")

        try:
            network = create_azul_network(config_name)
            network.eval()  # Set to eval mode to avoid BatchNorm issues
            info = network.get_model_info()

            print(f"    Input size: {info['input_size']}")
            print(f"    Hidden layers: {info['hidden_sizes']}")
            print(f"    Total parameters: {info['total_parameters']:,}")
            print(f"    Trainable parameters: {info['trainable_parameters']:,}")

            # Test forward pass
            with torch.no_grad():
                policy_probs, value = network(dummy_state)
                print(
                    f"    Output shapes - Policy: {policy_probs.shape}, "
                    f"Value: {value.shape}"
                )
                print(
                    f"    Value range: [{value.min().item():.3f}, "
                    f"{value.max().item():.3f}]"
                )
                print(f"    Policy sum: {policy_probs.sum().item():.3f}")

        except Exception as e:
            print(f"    Error: {e}")


def demonstrate_action_prediction(states_tensor, masks_tensor):
    """Demonstrate action prediction with legal action masking."""
    print("\n3. Demonstrating action prediction with legal action masking...")

    # Create a medium-sized network
    network = create_azul_network("medium")
    network.eval()

    print(f"Network info: {network.get_model_info()}")
    print()

    # Test with different temperatures
    temperatures = [0.1, 1.0, 2.0]

    for temp in temperatures:
        print(f"  Temperature: {temp}")

        with torch.no_grad():
            # Get predictions without masking
            policy_probs_raw, values = network(states_tensor, temperature=temp)

            # Get predictions with legal action masking
            policy_probs_masked = network.get_action_probabilities(
                states_tensor, masks_tensor, temperature=temp
            )

            # Analyze the results
            for i in range(min(3, len(states_tensor))):
                legal_actions_count = int(masks_tensor[i].sum().item())
                entropy_raw = -torch.sum(
                    policy_probs_raw[i] * torch.log(policy_probs_raw[i] + 1e-8)
                )
                entropy_masked = -torch.sum(
                    policy_probs_masked[i] * torch.log(policy_probs_masked[i] + 1e-8)
                )

                print(f"    State {i+1}: {legal_actions_count} legal actions")
                print(f"      Value: {values[i].item():.3f}")
                print(f"      Entropy (raw): {entropy_raw.item():.3f}")
                print(f"      Entropy (masked): {entropy_masked.item():.3f}")
                max_prob_raw = policy_probs_raw[i].max().item()
                print(f"      Max prob (raw): {max_prob_raw:.3f}")
                max_prob_masked = policy_probs_masked[i].max().item()
                print(f"      Max prob (masked): {max_prob_masked:.3f}")
        print()


def demonstrate_single_prediction():
    """Demonstrate single state prediction (convenience method)."""
    print("4. Demonstrating single state prediction...")

    # Create a game state
    game = GameState(num_players=2, seed=123)
    state_repr = AzulStateRepresentation(game)
    state_vector = state_repr.get_flat_state_vector(normalize=True)

    # Get legal actions
    legal_actions = game.get_legal_actions()
    action_mask = np.zeros(500)

    env = AzulAECEnv(num_players=2, seed=123)
    for action in legal_actions:
        action_idx = env._encode_action(action)
        if 0 <= action_idx < 500:
            action_mask[action_idx] = 1.0

    # Create network and make prediction
    network = create_azul_network("medium")

    # Test the convenience method
    pred_probs, pred_value = network.predict(state_vector, action_mask, temperature=1.0)

    print(f"  State vector size: {len(state_vector)}")
    print(f"  Number of legal actions: {len(legal_actions)}")
    print(f"  Predicted value: {pred_value:.3f}")
    print(f"  Action probabilities shape: {pred_probs.shape}")
    print(f"  Action probabilities sum: {pred_probs.sum():.3f}")
    print(f"  Max action probability: {pred_probs.max():.3f}")
    print(f"  Entropy: {-np.sum(pred_probs * np.log(pred_probs + 1e-8)):.3f}")

    # Show top 5 actions
    top_actions = np.argsort(pred_probs)[-5:][::-1]
    print("  Top 5 action indices and probabilities:")
    for i, action_idx in enumerate(top_actions):
        print(f"    {i+1}. Action {action_idx}: {pred_probs[action_idx]:.4f}")


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n5. Demonstrating batch processing...")

    # Create multiple game states
    batch_size = 8
    games = [GameState(num_players=2, seed=i) for i in range(batch_size)]

    # Play a few random moves in each game
    for game in games:
        for _ in range(np.random.randint(1, 4)):
            actions = game.get_legal_actions()
            if actions and not game.game_over:
                action = np.random.choice(actions)
                game.apply_action(action)

    # Collect states and masks
    states = []
    masks = []
    env = AzulAECEnv(num_players=2)

    for game in games:
        state_repr = AzulStateRepresentation(game)
        state_vector = state_repr.get_flat_state_vector(normalize=True)
        states.append(state_vector)

        legal_actions = game.get_legal_actions()
        action_mask = np.zeros(500)
        for action in legal_actions:
            action_idx = env._encode_action(action)
            if 0 <= action_idx < 500:
                action_mask[action_idx] = 1.0
        masks.append(action_mask)

    # Convert to tensors
    states_tensor = torch.FloatTensor(np.array(states))
    masks_tensor = torch.FloatTensor(np.array(masks))

    # Process batch
    network = create_azul_network("medium")

    with torch.no_grad():
        policy_probs, values = network(states_tensor)
        # Note: masked_probs could be used for further analysis if needed
        _ = network.get_action_probabilities(states_tensor, masks_tensor)

    print(f"  Processed batch of {batch_size} states")
    print(f"  Input shape: {states_tensor.shape}")
    print(f"  Policy output shape: {policy_probs.shape}")
    print(f"  Value output shape: {values.shape}")
    print("  Value statistics:")
    print(f"    Mean: {values.mean().item():.3f}")
    print(f"    Std: {values.std().item():.3f}")
    print(f"    Min: {values.min().item():.3f}")
    print(f"    Max: {values.max().item():.3f}")

    # Show per-game statistics
    legal_action_counts = masks_tensor.sum(dim=1)
    print(f"  Legal actions per game: {legal_action_counts.tolist()}")


def demonstrate_feature_extraction():
    """Demonstrate feature extraction capabilities."""
    print("\n6. Demonstrating feature extraction...")

    # Create a game state
    game = GameState(num_players=2, seed=456)
    state_repr = AzulStateRepresentation(game)
    state_vector = state_repr.get_flat_state_vector(normalize=True)
    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)

    # Create network and extract features
    network = create_azul_network("medium")
    network.eval()  # Set to eval mode for single sample

    with torch.no_grad():
        policy_probs, value, features = network(state_tensor, return_features=True)

    print(f"  Input shape: {state_tensor.shape}")
    print(f"  Features shape: {features.shape}")
    print("  Feature statistics:")
    print(f"    Mean: {features.mean().item():.3f}")
    print(f"    Std: {features.std().item():.3f}")
    print(f"    Min: {features.min().item():.3f}")
    print(f"    Max: {features.max().item():.3f}")
    print(f"    Sparsity (zeros): {(features == 0).float().mean().item():.3f}")


def main():
    """Run all demonstrations."""
    print("Starting Azul Neural Network Demonstrations...\n")

    try:
        # Generate real game states
        states_tensor, masks_tensor = demonstrate_network_with_real_game()

        # Test different configurations
        test_different_network_configurations()

        # Demonstrate action prediction
        demonstrate_action_prediction(states_tensor, masks_tensor)

        # Demonstrate single prediction
        demonstrate_single_prediction()

        # Demonstrate batch processing
        demonstrate_batch_processing()

        # Demonstrate feature extraction
        demonstrate_feature_extraction()

        print("\n=== All demonstrations completed successfully! ===")
        print("\nThe AzulNetwork is ready for:")
        print("- Training with reinforcement learning algorithms")
        print("- Integration with MCTS for tree search")
        print("- Self-play training scenarios")
        print("- Real-time game AI inference")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
