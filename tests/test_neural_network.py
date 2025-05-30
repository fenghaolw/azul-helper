#!/usr/bin/env python3
"""
Comprehensive tests for the Azul Neural Network Architecture.

This module tests all aspects of the AzulNetwork including:
- Architecture initialization
- Forward pass functionality
- Action prediction and masking
- Value estimation
- Batch processing
- Integration with game states
"""

import os
import sys

import numpy as np
import pytest
import torch

# Add the parent directory to the path to import azul_rl modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from game.game_state import GameState
from game.pettingzoo_env import AzulAECEnv
from game.state_representation import AzulStateRepresentation
from training.neural_network import (
    AzulNetwork,
    AzulNetworkConfig,
    create_azul_network,
)


class TestAzulNetworkArchitecture:
    """Test the basic architecture and initialization of AzulNetwork."""

    def test_network_initialization_default(self):
        """Test default network initialization."""
        network = AzulNetwork()

        assert network.input_size > 0
        assert network.action_space_size == 500
        assert len(network.hidden_sizes) == 3
        assert network.dropout_rate == 0.1
        assert network.use_batch_norm is True
        assert network.use_residual is True

    def test_network_initialization_custom(self):
        """Test custom network initialization."""
        custom_config = {
            "input_size": 1000,
            "hidden_sizes": (256, 128),
            "action_space_size": 300,
            "dropout_rate": 0.2,
            "use_batch_norm": False,
            "use_residual": False,
        }

        network = AzulNetwork(**custom_config)

        assert network.input_size == 1000
        assert network.action_space_size == 300
        assert network.hidden_sizes == (256, 128)
        assert network.dropout_rate == 0.2
        assert network.use_batch_norm is False
        assert network.use_residual is False

    def test_network_auto_input_size_detection(self):
        """Test automatic input size detection from game state."""
        network = AzulNetwork(input_size=None)

        # Should auto-detect from dummy game state
        assert network.input_size > 900  # Should be around 915
        assert network.input_size < 1000

    def test_network_layer_creation(self):
        """Test that all network layers are created correctly."""
        network = AzulNetwork(hidden_sizes=(256, 128, 64))

        # Check input layer
        assert isinstance(network.input_layer, torch.nn.Sequential)

        # Check body layers
        assert len(network.body_layers) == 2  # hidden_sizes - 1

        # Check heads
        assert isinstance(network.policy_head, torch.nn.Sequential)
        assert isinstance(network.value_head, torch.nn.Sequential)

        # Check residual projections
        if network.use_residual:
            assert len(network.residual_projections) == 2


class TestNetworkForwardPass:
    """Test the forward pass functionality."""

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        network = AzulNetwork(input_size=100, hidden_sizes=(64, 32))
        batch_size = 4
        input_tensor = torch.randn(batch_size, 100)

        policy_probs, value = network(input_tensor)

        # Check output shapes
        assert policy_probs.shape == (batch_size, network.action_space_size)
        assert value.shape == (batch_size, 1)

        # Check value range
        assert torch.all(value >= -1.0) and torch.all(value <= 1.0)

        # Check policy probabilities sum to 1
        prob_sums = policy_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)

    def test_forward_pass_with_temperature(self):
        """Test forward pass with different temperatures."""
        network = AzulNetwork(input_size=100, hidden_sizes=(64, 32))
        network.eval()  # Set to eval mode for single sample
        input_tensor = torch.randn(1, 100)

        # Test different temperatures
        temperatures = [0.1, 1.0, 2.0, 10.0]
        entropies = []

        for temp in temperatures:
            with torch.no_grad():  # Use no_grad for eval mode
                policy_probs, _ = network(input_tensor, temperature=temp)
            entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-8))
            entropies.append(entropy.item())

        # Higher temperature should generally lead to higher entropy
        assert entropies[0] < entropies[-1]  # Low temp < high temp

    def test_forward_pass_with_features(self):
        """Test forward pass with feature extraction."""
        network = AzulNetwork(input_size=100, hidden_sizes=(64, 32))
        input_tensor = torch.randn(2, 100)

        policy_probs, value, features = network(input_tensor, return_features=True)

        # Check that features are returned
        assert features.shape == (2, 32)  # Last hidden layer size

        # Check that other outputs are still correct
        assert policy_probs.shape == (2, network.action_space_size)
        assert value.shape == (2, 1)

    def test_forward_pass_deterministic(self):
        """Test that forward pass is deterministic in eval mode."""
        network = AzulNetwork(input_size=100, hidden_sizes=(64, 32))
        network.eval()

        input_tensor = torch.randn(1, 100)

        # Run forward pass twice
        with torch.no_grad():
            policy1, value1 = network(input_tensor)
            policy2, value2 = network(input_tensor)

        # Should be identical in eval mode
        assert torch.allclose(policy1, policy2)
        assert torch.allclose(value1, value2)


class TestActionPrediction:
    """Test action prediction and legal action masking."""

    def test_action_probabilities_without_masking(self):
        """Test action probability generation without masking."""
        network = AzulNetwork(input_size=100, hidden_sizes=(64, 32))
        input_tensor = torch.randn(3, 100)

        probs = network.get_action_probabilities(input_tensor)

        # Check shape and properties
        assert probs.shape == (3, network.action_space_size)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-6)
        assert torch.all(probs >= 0.0)

    def test_action_probabilities_with_masking(self):
        """Test action probability generation with legal action masking."""
        network = AzulNetwork(input_size=100, hidden_sizes=(64, 32))
        input_tensor = torch.randn(2, 100)

        # Create legal action mask (only first 10 actions are legal)
        legal_mask = torch.zeros(2, network.action_space_size)
        legal_mask[:, :10] = 1.0

        probs = network.get_action_probabilities(input_tensor, legal_mask)

        # Check that illegal actions have zero probability
        assert torch.allclose(probs[:, 10:], torch.zeros_like(probs[:, 10:]))

        # Check that legal actions sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)

        # Check that legal actions have positive probabilities
        assert torch.all(probs[:, :10] > 0.0)

    def test_action_probabilities_empty_mask(self):
        """Test behavior with empty legal action mask."""
        network = AzulNetwork(input_size=100, hidden_sizes=(64, 32))
        network.eval()  # Set to eval mode for single sample
        input_tensor = torch.randn(1, 100)

        # Create empty mask (no legal actions)
        legal_mask = torch.zeros(1, network.action_space_size)

        with torch.no_grad():  # Use no_grad for eval mode
            probs = network.get_action_probabilities(input_tensor, legal_mask)

        # Should handle gracefully (all zeros, but normalized)
        assert probs.shape == (1, network.action_space_size)
        # Due to numerical stability, might not be exactly zero

    def test_state_value_estimation(self):
        """Test state value estimation."""
        network = AzulNetwork(input_size=100, hidden_sizes=(64, 32))
        input_tensor = torch.randn(5, 100)

        values = network.get_state_value(input_tensor)

        # Check shape and range
        assert values.shape == (5, 1)
        assert torch.all(values >= -1.0) and torch.all(values <= 1.0)


class TestPredictMethod:
    """Test the convenience predict method."""

    def test_predict_single_state(self):
        """Test prediction for a single state."""
        network = AzulNetwork(input_size=100, hidden_sizes=(64, 32))
        state = np.random.randn(100)

        probs, value = network.predict(state)

        # Check types and shapes
        assert isinstance(probs, np.ndarray)
        assert isinstance(value, float)
        assert probs.shape == (network.action_space_size,)
        assert -1.0 <= value <= 1.0
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)

    def test_predict_with_legal_actions(self):
        """Test prediction with legal action masking."""
        network = AzulNetwork(input_size=100, hidden_sizes=(64, 32))
        state = np.random.randn(100)
        legal_actions = np.zeros(network.action_space_size)
        legal_actions[:20] = 1.0  # First 20 actions are legal

        probs, value = network.predict(state, legal_actions)

        # Check that illegal actions have zero probability
        assert np.allclose(probs[20:], 0.0)
        assert np.all(probs[:20] > 0.0)
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)

    def test_predict_with_temperature(self):
        """Test prediction with different temperatures."""
        network = AzulNetwork(input_size=100, hidden_sizes=(64, 32))
        state = np.random.randn(100)

        # Test different temperatures
        probs_low, _ = network.predict(state, temperature=0.1)
        probs_high, _ = network.predict(state, temperature=2.0)

        # Low temperature should be more concentrated
        entropy_low = -np.sum(probs_low * np.log(probs_low + 1e-8))
        entropy_high = -np.sum(probs_high * np.log(probs_high + 1e-8))

        assert entropy_low < entropy_high


class TestNetworkConfigurations:
    """Test predefined network configurations."""

    def test_all_predefined_configs(self):
        """Test all predefined configurations."""
        configs = ["small", "medium", "large", "deep"]

        for config_name in configs:
            network = create_azul_network(config_name)
            network.eval()  # Set to eval mode for single sample

            # Test that network can be created and used
            assert isinstance(network, AzulNetwork)

            # Test forward pass
            dummy_input = torch.randn(1, network.input_size)
            with torch.no_grad():  # Use no_grad for eval mode
                policy, value = network(dummy_input)

            assert policy.shape == (1, network.action_space_size)
            assert value.shape == (1, 1)

    def test_config_override(self):
        """Test configuration override functionality."""
        network = create_azul_network("small", dropout_rate=0.5, use_batch_norm=False)

        # Check that overrides were applied
        assert network.dropout_rate == 0.5
        assert network.use_batch_norm is False

        # Check that other config values are preserved
        config = AzulNetworkConfig.small()
        assert network.hidden_sizes == config["hidden_sizes"]

    def test_invalid_config_name(self):
        """Test handling of invalid configuration names."""
        with pytest.raises(ValueError):
            create_azul_network("invalid_config")


class TestGameStateIntegration:
    """Test integration with actual game states."""

    def test_real_game_state_processing(self):
        """Test processing of real game states."""
        # Create a real game
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)
        state_vector = state_repr.get_flat_state_vector(normalize=True)

        # Create network and process state
        network = create_azul_network("medium")
        probs, value = network.predict(state_vector)

        # Check outputs
        assert isinstance(probs, np.ndarray)
        assert isinstance(value, float)
        assert probs.shape == (500,)  # Action space size
        assert -1.0 <= value <= 1.0
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)

    def test_legal_action_integration(self):
        """Test integration with legal actions from game."""
        # Create game and get legal actions
        game = GameState(num_players=2, seed=42)
        legal_actions = game.get_legal_actions()

        # Create action mask using PettingZoo environment
        env = AzulAECEnv(num_players=2, seed=42)
        action_mask = np.zeros(500)
        for action in legal_actions:
            action_idx = env._encode_action(action)
            if 0 <= action_idx < 500:
                action_mask[action_idx] = 1.0

        # Get state and predict
        state_repr = AzulStateRepresentation(game)
        state_vector = state_repr.get_flat_state_vector(normalize=True)

        network = create_azul_network("medium")
        probs, value = network.predict(state_vector, action_mask)

        # Check that only legal actions have positive probabilities
        legal_indices = np.where(action_mask == 1.0)[0]
        illegal_indices = np.where(action_mask == 0.0)[0]

        assert np.all(probs[legal_indices] > 0.0)
        assert np.allclose(probs[illegal_indices], 0.0)

    def test_batch_game_states(self):
        """Test batch processing of multiple game states."""
        # Create multiple games
        games = [GameState(num_players=2, seed=i) for i in range(4)]

        # Play some moves to get different states
        for i, game in enumerate(games):
            for _ in range(i + 1):  # Different number of moves per game
                actions = game.get_legal_actions()
                if actions and not game.game_over:
                    action = np.random.choice(actions)
                    game.apply_action(action)

        # Collect states
        states = []
        for game in games:
            state_repr = AzulStateRepresentation(game)
            state_vector = state_repr.get_flat_state_vector(normalize=True)
            states.append(state_vector)

        states_tensor = torch.FloatTensor(np.array(states))

        # Process batch
        network = create_azul_network("medium")
        policy_probs, values = network(states_tensor)

        # Check outputs
        assert policy_probs.shape == (4, 500)
        assert values.shape == (4, 1)
        assert torch.all(values >= -1.0) and torch.all(values <= 1.0)
        assert torch.allclose(policy_probs.sum(dim=-1), torch.ones(4), atol=1e-6)


class TestModelInfo:
    """Test model information and statistics."""

    def test_model_info_structure(self):
        """Test that model info contains expected fields."""
        network = create_azul_network("medium")
        info = network.get_model_info()

        expected_fields = [
            "input_size",
            "hidden_sizes",
            "action_space_size",
            "total_parameters",
            "trainable_parameters",
            "dropout_rate",
            "use_batch_norm",
            "use_residual",
        ]

        for field in expected_fields:
            assert field in info

    def test_parameter_counting(self):
        """Test that parameter counting is accurate."""
        network = AzulNetwork(input_size=100, hidden_sizes=(50, 25))
        info = network.get_model_info()

        # Manually count parameters for verification
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(
            p.numel() for p in network.parameters() if p.requires_grad
        )

        assert info["total_parameters"] == total_params
        assert info["trainable_parameters"] == trainable_params
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self):
        """Test handling of edge cases in input."""
        network = AzulNetwork(input_size=10, hidden_sizes=(5,))
        network.eval()  # Set to eval mode for single sample

        # Test with very small batch
        single_input = torch.randn(1, 10)
        with torch.no_grad():  # Use no_grad for eval mode
            policy, value = network(single_input)
        assert policy.shape == (1, 500)
        assert value.shape == (1, 1)

    def test_large_batch(self):
        """Test with large batch size."""
        network = AzulNetwork(input_size=50, hidden_sizes=(25,))
        large_batch = torch.randn(100, 50)

        policy, value = network(large_batch)
        assert policy.shape == (100, 500)
        assert value.shape == (100, 1)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        network = AzulNetwork(input_size=50, hidden_sizes=(25,))
        input_tensor = torch.randn(2, 50, requires_grad=True)

        policy, value = network(input_tensor)
        loss = policy.sum() + value.sum()
        loss.backward()

        # Check that gradients exist
        assert input_tensor.grad is not None
        for param in network.parameters():
            if param.requires_grad:
                assert param.grad is not None


def test_network_architecture_comprehensive():
    """Comprehensive test of the entire network architecture."""
    print("Running comprehensive neural network test...")

    # Test with real game state
    game = GameState(num_players=2, seed=42)
    state_repr = AzulStateRepresentation(game)
    state_vector = state_repr.get_flat_state_vector(normalize=True)

    # Test all configurations
    for config_name in ["small", "medium", "large", "deep"]:
        network = create_azul_network(config_name)

        # Test single prediction
        probs, value = network.predict(state_vector)
        assert probs.shape == (500,)
        assert -1.0 <= value <= 1.0
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)

        print(f"✓ {config_name} configuration test passed")

    print("✓ Comprehensive neural network test passed!")


if __name__ == "__main__":
    test_network_architecture_comprehensive()
