"""
Baseline agents for evaluation purposes.

This module provides various baseline agents that can be used to evaluate
the performance of trained models. These include random agents, heuristic
agents, and wrapper for previous model checkpoints.
"""

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agents.checkpoint_agent import CheckpointAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from game.game_state import Action, GameState, TileColor


class BaselineAgent(ABC):
    """
    Abstract base class for baseline agents used in evaluation.

    All baseline agents should inherit from this class and implement
    the required methods.
    """

    def __init__(self, player_id: int = 0, name: str = "BaselineAgent"):
        self.player_id = player_id
        self.name = name
        self.nodes_evaluated = 0
        self.total_time_taken = 0.0
        self.total_moves = 0

    @abstractmethod
    def select_action(self, game_state: GameState) -> Action:
        """Select an action given the current game state."""
        pass

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
        }


class RandomBaselineAgent(BaselineAgent):
    """
    Wrapper for RandomAgent to use as baseline with consistent interface.
    """

    def __init__(self, player_id: int = 0, seed: Optional[int] = None):
        super().__init__(player_id, "RandomAgent")
        self.random_agent = RandomAgent(player_id, seed)

    def select_action(self, game_state: GameState) -> Action:
        """Select action using the random agent."""
        action = self.random_agent.select_action(game_state)

        # Copy statistics from random agent
        self.nodes_evaluated = self.random_agent.nodes_evaluated
        self.total_time_taken = self.random_agent.total_time_taken
        self.total_moves = self.random_agent.total_moves

        return action

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return self.random_agent.get_stats()

    def reset_stats(self):
        """Reset statistics for both agents."""
        super().reset_stats()
        self.random_agent.reset_stats()

    def get_info(self) -> Dict[str, Any]:
        """Get agent information for evaluation metadata."""
        return self.random_agent.get_info()


class HeuristicBaselineAgent(BaselineAgent):
    """
    Wrapper for the full HeuristicAgent to use as baseline.

    This allows using the existing sophisticated heuristic agent
    as a baseline for comparison with trained models.
    """

    def __init__(self, player_id: int = 0):
        super().__init__(player_id, "HeuristicAgent")
        self.heuristic_agent = HeuristicAgent(player_id)

    def select_action(self, game_state: GameState) -> Action:
        """Select action using the full heuristic agent."""
        start_time = time.time()

        action = self.heuristic_agent.select_action(game_state)

        # Update statistics from heuristic agent
        self.nodes_evaluated += self.heuristic_agent.nodes_evaluated
        self.total_time_taken += time.time() - start_time
        self.total_moves += 1

        return action

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = super().get_stats()
        heuristic_stats = self.heuristic_agent.get_stats()

        # Merge relevant statistics
        stats.update(
            {"heuristic_nodes_evaluated": heuristic_stats.get("nodesEvaluated", 0)}
        )

        return stats

    def reset_stats(self):
        """Reset statistics for both agents."""
        super().reset_stats()
        self.heuristic_agent.reset_stats()

    def get_info(self) -> Dict[str, Any]:
        """Get agent information for evaluation metadata."""
        info = super().get_info()
        info["description"] = (
            "Full sophisticated heuristic agent with strategic evaluation"
        )
        return info


class CheckpointBaselineAgent(BaselineAgent):
    """
    Wrapper for CheckpointAgent to use as baseline with consistent interface.
    """

    def __init__(self, checkpoint_path: str, player_id: int = 0):
        super().__init__(player_id, "CheckpointAgent")
        self.checkpoint_agent = CheckpointAgent(checkpoint_path, player_id)
        self.name = self.checkpoint_agent.name  # Use the checkpoint's name

    def select_action(self, game_state: GameState) -> Action:
        """Select action using the checkpoint agent."""
        action = self.checkpoint_agent.select_action(game_state)

        # Copy statistics from checkpoint agent
        self.nodes_evaluated = self.checkpoint_agent.nodes_evaluated
        self.total_time_taken = self.checkpoint_agent.total_time_taken
        self.total_moves = self.checkpoint_agent.total_moves

        return action

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return self.checkpoint_agent.get_stats()

    def reset_stats(self):
        """Reset statistics for both agents."""
        super().reset_stats()
        self.checkpoint_agent.reset_stats()

    def get_info(self) -> Dict[str, Any]:
        """Get agent information for evaluation metadata."""
        return self.checkpoint_agent.get_info()


def create_baseline_agent(agent_type: str, **kwargs) -> BaselineAgent:
    """
    Factory function to create baseline agents.

    Args:
        agent_type: Type of baseline agent ('random', 'heuristic', 'checkpoint')
        **kwargs: Additional arguments for specific agent types

    Returns:
        Configured baseline agent
    """
    agent_type = agent_type.lower()

    if agent_type == "random":
        return RandomBaselineAgent(**kwargs)
    elif agent_type == "heuristic":
        return HeuristicBaselineAgent(**kwargs)
    elif agent_type == "checkpoint":
        if "checkpoint_path" not in kwargs:
            raise ValueError("checkpoint_path is required for checkpoint agents")
        return CheckpointBaselineAgent(**kwargs)
    else:
        raise ValueError(
            f"Unknown baseline agent type: {agent_type}. Available types: 'random', 'heuristic', 'checkpoint'"
        )
