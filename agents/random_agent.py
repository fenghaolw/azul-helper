"""
Random agent for Azul game.

This module provides a random agent that selects actions uniformly at random
from legal actions. It serves as a baseline for comparison with more sophisticated agents.
"""

import random
import time
from typing import Any, Dict, Optional

from game.game_state import Action, GameState


class RandomAgent:
    """
    A random agent that selects actions uniformly at random from legal actions.

    This is the simplest baseline agent and provides a lower bound on performance.
    Any reasonable agent should easily outperform the random agent.
    """

    def __init__(self, player_id: int = 0, seed: Optional[int] = None):
        self.player_id = player_id
        self.name = "RandomAgent"
        self.random_state = random.Random(seed)
        self.nodes_evaluated = 0
        self.total_time_taken = 0.0
        self.total_moves = 0

    def select_action(self, game_state: GameState) -> Action:
        """
        Select a random action from legal actions.

        Args:
            game_state: Current game state

        Returns:
            Randomly selected legal action
        """
        start_time = time.time()
        legal_actions = game_state.get_legal_actions()

        if not legal_actions:
            # Check if game is over before raising error
            if game_state.game_over:
                # Game has ended, this is expected
                raise ValueError("No legal actions available - game is over")
            else:
                # Game is not over but no actions available - this indicates a bug
                raise ValueError(
                    "No legal actions available - possible game state inconsistency"
                )

        action = self.random_state.choice(legal_actions)

        # Update statistics
        self.actions_taken = getattr(self, "actions_taken", 0) + 1
        self.last_action = action
        self.nodes_evaluated += len(legal_actions)  # "Evaluated" all options randomly
        self.total_time_taken += time.time() - start_time
        self.total_moves += 1

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
            "description": "Selects actions uniformly at random from legal actions",
        }


def create_random_agent(player_id: int = 0, seed: Optional[int] = None) -> RandomAgent:
    """
    Factory function to create a random agent.

    Args:
        player_id: The player ID for this agent
        seed: Optional seed for reproducible randomness

    Returns:
        Configured random agent
    """
    return RandomAgent(player_id, seed)
