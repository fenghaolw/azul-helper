"""
Base agent class for Azul AI agents.

This module provides a common interface that all Azul agents should implement,
eliminating the need for wrappers in the evaluation system.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from game.game_state import Action, GameState


class AzulAgent(ABC):
    """
    Abstract base class for all Azul AI agents.

    This class defines the standard interface that all agents should implement,
    providing both the core gameplay methods and the evaluation/statistics
    methods needed for comprehensive agent analysis.
    """

    def __init__(self, player_id: int = 0, name: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            player_id: The player ID this agent controls
            name: Optional name for the agent (defaults to class name)
        """
        self.player_id = player_id
        self.name = name or self.__class__.__name__

        # Statistics tracking
        self.nodes_evaluated = 0
        self.total_time_taken = 0.0
        self.total_moves = 0
        self.last_move_time = 0.0

    @abstractmethod
    def select_action(
        self, game_state: GameState, deterministic: bool = False
    ) -> Action:
        """
        Select an action given the current game state.

        Args:
            game_state: Current game state
            deterministic: Whether to select deterministically (if supported)

        Returns:
            Selected action
        """
        pass

    def select_action_timed(
        self, game_state: GameState, deterministic: bool = False
    ) -> Action:
        """
        Select an action with automatic timing statistics.

        This method wraps select_action to automatically track timing statistics.
        Subclasses should generally override select_action, not this method.

        Args:
            game_state: Current game state
            deterministic: Whether to select deterministically (if supported)

        Returns:
            Selected action
        """
        start_time = time.time()
        try:
            action = self.select_action(game_state, deterministic)
            return action
        finally:
            # Update timing statistics
            move_time = time.time() - start_time
            self.last_move_time = move_time
            self.total_time_taken += move_time
            self.total_moves += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics for reporting.

        This method returns runtime performance data that changes during execution,
        such as timing information, nodes evaluated, search depths reached, etc.

        Returns:
            Dictionary containing agent performance statistics
        """
        avg_time = self.total_time_taken / max(self.total_moves, 1)
        avg_nodes = self.nodes_evaluated / max(self.total_moves, 1)

        return {
            "name": self.name,
            "player_id": self.player_id,
            "total_moves": self.total_moves,
            "total_time_taken": self.total_time_taken,
            "average_time_per_move": avg_time,
            "last_move_time": self.last_move_time,
            "nodes_evaluated": self.nodes_evaluated,
            "average_nodes_per_move": avg_nodes,
            "algorithm": self._get_algorithm_name(),
        }

    def get_info(self) -> Dict[str, Any]:
        """
        Get static agent metadata for evaluation and identification.

        This method returns configuration and identification information that
        doesn't change during execution, such as agent type, algorithm parameters,
        model paths, etc. This is used for evaluation metadata and reporting.

        Returns:
            Dictionary containing agent type and configuration information
        """
        return {
            "agent_type": self.__class__.__name__,
            "name": self.name,
            "player_id": self.player_id,
        }

    def reset_stats(self) -> None:
        """Reset all statistics to initial values."""
        self.nodes_evaluated = 0
        self.total_time_taken = 0.0
        self.total_moves = 0
        self.last_move_time = 0.0

    def _get_algorithm_name(self) -> str:
        """
        Get the algorithm name for this agent.

        Subclasses can override this to provide more specific algorithm names.

        Returns:
            Algorithm name string
        """
        return self.__class__.__name__

    # Optional methods that some agents may want to implement

    def get_action_probabilities(self, game_state: GameState) -> Optional[Any]:
        """
        Get action probabilities (if supported by the agent).

        Args:
            game_state: Current game state

        Returns:
            Array of action probabilities, or None if not supported
        """
        return None

    def can_provide_probabilities(self) -> bool:
        """
        Check if this agent can provide action probabilities.

        Returns:
            True if get_action_probabilities is implemented meaningfully
        """
        return (
            self.get_action_probabilities.__func__
            is not AzulAgent.get_action_probabilities
        )

    def supports_deterministic_play(self) -> bool:
        """
        Check if this agent supports deterministic action selection.

        Returns:
            True if the agent respects the deterministic parameter
        """
        # Default implementation: assume agents support deterministic play
        # unless they override this method
        return True

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name} (Player {self.player_id})"

    def __repr__(self) -> str:
        """Developer representation of the agent."""
        return (
            f"{self.__class__.__name__}(player_id={self.player_id}, name='{self.name}')"
        )
