"""
Core Neural Network Architecture for Azul Game AI.

This module implements the neural network architecture for the Azul board game AI,
including input processing, shared body layers, policy head, and value head.
"""

from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..game.state_representation import AzulStateRepresentation


class AzulNetwork(nn.Module):
    """
    Core neural network for Azul game AI.

    Architecture:
    - Input layer: Processes the flattened game state representation
    - Shared body: Fully connected layers with residual connections
    - Policy head: Outputs action probabilities with softmax
    - Value head: Outputs state value estimation with tanh
    """

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_sizes: Tuple[int, ...] = (512, 512, 256),
        action_space_size: int = 500,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_residual: bool = True,
    ):
        """
        Initialize the Azul neural network.

        Args:
            input_size: Size of input state vector (auto-detected if None)
            hidden_sizes: Tuple of hidden layer sizes for shared body
            action_space_size: Number of possible actions (from PettingZoo env)
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections in body
        """
        super().__init__()

        # Auto-detect input size if not provided
        if input_size is None:
            # Create dummy state representation to get size
            from ..game.game_state import GameState

            dummy_game = GameState(num_players=2, seed=42)
            dummy_repr = AzulStateRepresentation(dummy_game)
            input_size = dummy_repr.flat_state_size

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.action_space_size = action_space_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # Input processing layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Shared body layers with optional residual connections
        self.body_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                (
                    nn.BatchNorm1d(hidden_sizes[i + 1])
                    if use_batch_norm
                    else nn.Identity()
                ),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            self.body_layers.append(layer)

        # Residual projection layers (for when dimensions don't match)
        if use_residual:
            self.residual_projections = nn.ModuleList()
            for i in range(len(hidden_sizes) - 1):
                if hidden_sizes[i] != hidden_sizes[i + 1]:
                    self.residual_projections.append(
                        nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
                    )
                else:
                    self.residual_projections.append(nn.Identity())

        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.BatchNorm1d(hidden_sizes[-1] // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[-1] // 2, action_space_size),
            # Note: Softmax is applied in forward() with temperature
        )

        # Value head (state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.BatchNorm1d(hidden_sizes[-1] // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[-1] // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output in [-1, 1] range
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(  # type: ignore[misc]
        self,
        state: torch.Tensor,
        temperature: float = 1.0,
        return_features: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor of shape (batch_size, input_size)
            temperature: Temperature for softmax in policy head
            return_features: Whether to return intermediate features

        Returns:
            Tuple of (policy_logits, value) where:
            - policy_logits: Action probabilities (batch_size, action_space_size)
            - value: State value estimation (batch_size, 1)
            - features: Intermediate features if return_features=True
        """
        # Input processing
        x = self.input_layer(state)

        # Shared body with optional residual connections
        for i, layer in enumerate(self.body_layers):
            if self.use_residual:
                residual = self.residual_projections[i](x)
                x = layer(x) + residual
            else:
                x = layer(x)

        # Store features for potential return
        features = x

        # Policy head - output action logits
        policy_logits = self.policy_head(x)

        # Apply temperature scaling and softmax
        policy_probs = F.softmax(policy_logits / temperature, dim=-1)

        # Value head - output state value
        value = self.value_head(x)

        if return_features:
            return policy_probs, value, features
        else:
            return policy_probs, value

    def get_action_probabilities(
        self,
        state: torch.Tensor,
        legal_actions: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Get action probabilities with optional legal action masking.

        Args:
            state: Input state tensor
            legal_actions: Binary mask for legal actions (1=legal, 0=illegal)
            temperature: Temperature for softmax

        Returns:
            Action probabilities with illegal actions masked to 0
        """
        policy_probs, _ = self.forward(  # type: ignore[misc]
            state, temperature=temperature, return_features=False
        )

        if legal_actions is not None:
            # Mask illegal actions
            policy_probs = policy_probs * legal_actions
            # Renormalize to ensure probabilities sum to 1
            policy_probs = policy_probs / (
                policy_probs.sum(dim=-1, keepdim=True) + 1e-8
            )

        return policy_probs

    def get_state_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get state value estimation.

        Args:
            state: Input state tensor

        Returns:
            State value in [-1, 1] range
        """
        _, value = self.forward(state, return_features=False)  # type: ignore[misc]
        return value

    def predict(
        self,
        state: np.ndarray,
        legal_actions: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Make predictions for a single state (convenience method).

        Args:
            state: NumPy array of game state
            legal_actions: NumPy array of legal action mask
            temperature: Temperature for action selection
            device: Device to run inference on

        Returns:
            Tuple of (action_probabilities, state_value)
        """
        if device is None:
            device = next(self.parameters()).device

        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        legal_actions_tensor = None
        if legal_actions is not None:
            legal_actions_tensor = (
                torch.FloatTensor(legal_actions).unsqueeze(0).to(device)
            )

        # Set to evaluation mode
        self.eval()
        with torch.no_grad():
            policy_probs = self.get_action_probabilities(
                state_tensor, legal_actions_tensor, temperature
            )
            value = self.get_state_value(state_tensor)

        return policy_probs.cpu().numpy()[0], value.cpu().item()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "action_space_size": self.action_space_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "use_residual": self.use_residual,
        }


class AzulNetworkConfig:
    """Configuration class for AzulNetwork with predefined architectures."""

    @staticmethod
    def small() -> Dict[str, Any]:
        """Small network configuration for fast training/inference."""
        return {
            "hidden_sizes": (256, 256),
            "dropout_rate": 0.1,
            "use_batch_norm": True,
            "use_residual": False,
        }

    @staticmethod
    def medium() -> Dict[str, Any]:
        """Medium network configuration (default)."""
        return {
            "hidden_sizes": (512, 512, 256),
            "dropout_rate": 0.1,
            "use_batch_norm": True,
            "use_residual": True,
        }

    @staticmethod
    def large() -> Dict[str, Any]:
        """Large network configuration for maximum performance."""
        return {
            "hidden_sizes": (1024, 512, 512, 256),
            "dropout_rate": 0.15,
            "use_batch_norm": True,
            "use_residual": True,
        }

    @staticmethod
    def deep() -> Dict[str, Any]:
        """Deep network configuration with many layers."""
        return {
            "hidden_sizes": (512, 512, 512, 256, 256),
            "dropout_rate": 0.2,
            "use_batch_norm": True,
            "use_residual": True,
        }


def create_azul_network(config_name: str = "medium", **kwargs) -> AzulNetwork:
    """
    Create an AzulNetwork with a predefined configuration.

    Args:
        config_name: Name of configuration ('small', 'medium', 'large', 'deep')
        **kwargs: Additional arguments to override configuration

    Returns:
        Configured AzulNetwork instance
    """
    config_map = {
        "small": AzulNetworkConfig.small(),
        "medium": AzulNetworkConfig.medium(),
        "large": AzulNetworkConfig.large(),
        "deep": AzulNetworkConfig.deep(),
    }

    if config_name not in config_map:
        raise ValueError(
            f"Unknown config: {config_name}. Available: {list(config_map.keys())}"
        )

    config = config_map[config_name]
    config.update(kwargs)  # Override with any provided kwargs

    return AzulNetwork(**config)


# Example usage and testing functions
def test_network_architecture():
    """Test the network architecture with dummy data."""
    print("Testing AzulNetwork architecture...")

    # Create network
    network = create_azul_network("medium")
    print(f"Created network with config: {network.get_model_info()}")

    # Create dummy input
    batch_size = 4
    dummy_state = torch.randn(batch_size, network.input_size)
    dummy_legal_actions = torch.randint(
        0, 2, (batch_size, network.action_space_size)
    ).float()

    # Test forward pass
    policy_probs, value = network(dummy_state)
    print(f"Policy output shape: {policy_probs.shape}")
    print(f"Value output shape: {value.shape}")
    print(f"Policy probabilities sum: {policy_probs.sum(dim=-1)}")
    print(f"Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")

    # Test with legal action masking
    masked_probs = network.get_action_probabilities(dummy_state, dummy_legal_actions)
    print(f"Masked probabilities sum: {masked_probs.sum(dim=-1)}")

    # Test single prediction
    single_state = dummy_state[0].numpy()
    single_legal = dummy_legal_actions[0].numpy()
    pred_probs, pred_value = network.predict(single_state, single_legal)
    print(
        f"Single prediction - probs shape: {pred_probs.shape}, value: {pred_value:.3f}"
    )

    print("✓ Network architecture test passed!")


if __name__ == "__main__":
    test_network_architecture()
