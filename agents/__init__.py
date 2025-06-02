"""Agents module for Azul RL - Now using OpenSpiel implementations."""

from agents.heuristic_agent import HeuristicAgent, create_heuristic_agent
from agents.improved_heuristic_agent import (
    ImprovedHeuristicAgent,
    create_improved_heuristic_agent,
)
from agents.minimax_agent import MinimaxAgent, create_minimax_agent

# OpenSpiel-based agents (the only MCTS implementations we support)
from agents.openspiel_agents import RandomAgent  # Use OpenSpiel's RandomAgent directly
from agents.openspiel_agents import (
    OpenSpielAlphaZeroAgent,
    OpenSpielMCTSAgent,
)

# Import game types
from game.game_state import GameState

# For convenience, provide direct aliases to the OpenSpiel implementations
MCTSAgent = OpenSpielMCTSAgent
AlphaZeroAgent = OpenSpielAlphaZeroAgent

__all__ = [
    # Core game types
    "GameState",
    # MCTS implementations (OpenSpiel only)
    "MCTSAgent",  # Alias to OpenSpielMCTSAgent
    "AlphaZeroAgent",  # Alias to OpenSpielAlphaZeroAgent
    # OpenSpiel agents (full names)
    "OpenSpielMCTSAgent",
    "OpenSpielAlphaZeroAgent",
    "RandomAgent",  # OpenSpiel RandomAgent
    # Other agents
    "HeuristicAgent",
    "create_heuristic_agent",
    "ImprovedHeuristicAgent",
    "create_improved_heuristic_agent",
    "MinimaxAgent",
    "create_minimax_agent",
]
