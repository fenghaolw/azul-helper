"""Agents module for Azul RL."""

from agents.checkpoint_agent import CheckpointAgent, create_checkpoint_agent
from agents.heuristic_agent import HeuristicAgent, create_heuristic_agent
from agents.improved_heuristic_agent import (
    ImprovedHeuristicAgent,
    create_improved_heuristic_agent,
)
from agents.mcts import MCTS, GameState, MCTSAgent, MCTSNode, NeuralNetwork
from agents.minimax_agent import MinimaxAgent, create_minimax_agent
from agents.random_agent import RandomAgent, create_random_agent

# OpenSpiel-based agents
try:
    from agents.openspiel_agents import (
        OpenSpielAlphaZeroAgent,
        OpenSpielMCTSAgent,
    )
    from agents.openspiel_agents import RandomAgent as OpenSpielRandomAgent

    OPENSPIEL_AVAILABLE = True
except ImportError:
    OPENSPIEL_AVAILABLE = False
    OpenSpielMCTSAgent = None
    OpenSpielAlphaZeroAgent = None
    OpenSpielRandomAgent = None

__all__ = [
    "MCTS",
    "MCTSNode",
    "MCTSAgent",
    "GameState",
    "NeuralNetwork",
    "RandomAgent",
    "create_random_agent",
    "HeuristicAgent",
    "create_heuristic_agent",
    "ImprovedHeuristicAgent",
    "create_improved_heuristic_agent",
    "CheckpointAgent",
    "create_checkpoint_agent",
    "MinimaxAgent",
    "create_minimax_agent",
]

# Add OpenSpiel agents if available
if OPENSPIEL_AVAILABLE:
    __all__.extend(
        [
            "OpenSpielMCTSAgent",
            "OpenSpielAlphaZeroAgent",
            "OpenSpielRandomAgent",
            "OPENSPIEL_AVAILABLE",
        ]
    )
else:
    __all__.append("OPENSPIEL_AVAILABLE")
