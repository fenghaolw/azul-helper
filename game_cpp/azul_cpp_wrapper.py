"""
Python wrapper for the Azul C++ implementation.

This module provides a drop-in replacement for the Python Azul implementation,
using the high-performance C++ backend when available.
"""

import os
import sys

# Add the build directory to the path to find azul_cpp_bindings
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "build")
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

# Use C++ implementation
from azul_cpp_bindings import (
    Action,
    AzulMCTSAgent,
    CenterArea,
    Factory,
    FactoryArea,
    GameState,
    PatternLine,
    PlayerBoard,
    Tile,
    TileColor,
    Wall,
    create_game,
    create_mcts_agent,
)


class AzulAgent:
    """
    Unified interface for Azul agents, supporting both Python and C++ implementations.
    """

    def __init__(self, agent_type="random", **kwargs):
        """
        Create an Azul agent.

        Args:
            agent_type: Type of agent ("random", "mcts", "heuristic")
            **kwargs: Agent-specific parameters
        """
        self.agent_type = agent_type
        self.kwargs = kwargs
        self._agent = None

        if agent_type == "mcts":
            self._agent = create_mcts_agent(**kwargs)
        elif agent_type == "random":
            # Random agent doesn't need special initialization
            pass
        elif agent_type == "heuristic":
            # Could implement heuristic agent here
            pass
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def get_action(self, game_state):
        """Get the best action for the current game state."""
        if self.agent_type == "random":
            import random

            actions = game_state.get_legal_actions()
            return random.choice(actions) if actions else None
        elif self.agent_type == "mcts":
            if hasattr(game_state, "to_openspiel_state"):
                # Convert to OpenSpiel state if available
                openspiel_state = game_state.to_openspiel_state()
                return self._agent.get_action(openspiel_state)
            else:
                # Fallback for non-OpenSpiel states
                actions = game_state.get_legal_actions()
                return actions[0] if actions else None
        else:
            raise NotImplementedError(f"Agent type {self.agent_type} not implemented")

    def reset(self):
        """Reset the agent state."""
        if self._agent and hasattr(self._agent, "reset"):
            self._agent.reset()


def benchmark_performance(num_games=100, num_players=2):
    """
    Benchmark the performance of the current implementation.

    Args:
        num_games: Number of games to simulate
        num_players: Number of players per game

    Returns:
        dict: Performance statistics
    """
    import time

    start_time = time.time()
    games_completed = 0
    total_actions = 0

    for _ in range(num_games):
        game = create_game(num_players=num_players)
        actions_in_game = 0

        while not game.is_game_over():
            actions = game.get_legal_actions()
            if not actions:
                break

            # Apply random action
            import random

            action = random.choice(actions)
            game.apply_action(action)
            actions_in_game += 1
            total_actions += 1

        games_completed += 1

    end_time = time.time()
    elapsed = end_time - start_time

    return {
        "implementation": "C++",
        "games_completed": games_completed,
        "total_actions": total_actions,
        "elapsed_time": elapsed,
        "games_per_second": games_completed / elapsed,
        "actions_per_second": total_actions / elapsed,
        "avg_actions_per_game": (
            total_actions / games_completed if games_completed > 0 else 0
        ),
    }


# Export the main classes and functions
__all__ = [
    "Tile",
    "TileColor",
    "Action",
    "PatternLine",
    "Wall",
    "PlayerBoard",
    "Factory",
    "CenterArea",
    "FactoryArea",
    "GameState",
    "create_game",
    "AzulAgent",
    "benchmark_performance",
    "AzulMCTSAgent",
    "create_mcts_agent",
]
