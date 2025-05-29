"""
Checkpoint agent for Azul game.

This module provides an agent that loads and uses a previous model checkpoint.
It allows comparing new models against previous training iterations to measure improvement over time.
"""

import time
from typing import Any, Dict, Optional

from game.game_state import Action, GameState


class CheckpointAgent:
    """
    Agent for loading and using a previous model checkpoint.

    This allows comparing new models against previous training iterations
    to measure improvement over time.
    """

    def __init__(self, checkpoint_path: str, player_id: int = 0):
        self.player_id = player_id
        self.name = f"CheckpointAgent"
        self.checkpoint_path = checkpoint_path
        self.agent: Optional[Any] = None
        self.checkpoint_info: Dict[str, Any] = {}
        self.nodes_evaluated = 0
        self.total_time_taken = 0.0
        self.total_moves = 0

        # Load the checkpoint
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load the agent from checkpoint."""
        try:
            # Import here to avoid circular dependencies
            import torch

            from agents.mcts import MCTSAgent
            from training.neural_network import AzulNeuralNetwork
            from training.training_utils import ModelCheckpointManager

            # Load checkpoint
            checkpoint_manager = ModelCheckpointManager("temp")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Create model (you may need to adjust network config based on checkpoint)
            neural_network = AzulNeuralNetwork(config_name="medium", device=str(device))

            # Load checkpoint data
            checkpoint_data = checkpoint_manager.load_checkpoint(
                self.checkpoint_path, neural_network.model, device=device
            )

            # Store checkpoint information
            self.checkpoint_info = {
                "iteration": checkpoint_data.get("iteration", "unknown"),
                "stats": checkpoint_data.get("stats", {}),
                "checkpoint_path": self.checkpoint_path,
            }

            # Create MCTS agent with loaded model
            mcts_agent = MCTSAgent(
                neural_network=neural_network,
                c_puct=1.0,
                num_simulations=100,  # Reduced for faster evaluation
                temperature=0.0,  # Deterministic for evaluation
            )
            self.agent = mcts_agent

            self.name = f"Checkpoint_{checkpoint_data.get('iteration', 'unknown')}"

        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {self.checkpoint_path}: {e}"
            )

    def select_action(self, game_state: GameState) -> Action:
        """Select action using the loaded checkpoint agent."""
        if self.agent is None:
            raise RuntimeError("Checkpoint agent not properly loaded")

        start_time = time.time()

        action = self.agent.select_action(game_state, deterministic=True)

        # Update statistics
        self.total_time_taken += time.time() - start_time
        self.total_moves += 1

        # Try to get MCTS statistics if available
        if hasattr(self.agent, "last_search_stats"):
            stats = self.agent.last_search_stats
            self.nodes_evaluated += stats.get("simulations", 0)

        return action

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics for reporting."""
        avg_time = self.total_time_taken / max(self.total_moves, 1)
        avg_nodes = self.nodes_evaluated / max(self.total_moves, 1)

        return {
            "name": self.name,
            "player_id": self.player_id,
            "total_moves": self.total_moves,
            "total_time_taken": self.total_time_taken,
            "average_time_per_move": avg_time,
            "nodes_evaluated": self.nodes_evaluated,
            "average_nodes_per_move": avg_nodes,
        }

    def reset_stats(self):
        """Reset all statistics."""
        self.nodes_evaluated = 0
        self.total_time_taken = 0.0
        self.total_moves = 0

    def get_info(self) -> Dict[str, Any]:
        """Get agent information for evaluation metadata."""
        return {
            "agent_type": self.__class__.__name__,
            "name": self.name,
            "player_id": self.player_id,
            "description": f"Previous model checkpoint from {self.checkpoint_path}",
            "checkpoint_info": self.checkpoint_info,
        }


def create_checkpoint_agent(
    checkpoint_path: str, player_id: int = 0
) -> CheckpointAgent:
    """
    Factory function to create a checkpoint agent.

    Args:
        checkpoint_path: Path to the model checkpoint
        player_id: The player ID for this agent

    Returns:
        Configured checkpoint agent
    """
    return CheckpointAgent(checkpoint_path, player_id)
