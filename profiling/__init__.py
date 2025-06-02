"""
Profiling module for Azul RL project.

This module provides comprehensive profiling tools for different types of agents:
- AzulProfiler: For custom MCTS agents and neural networks
- OpenSpielProfiler: For OpenSpiel-based agents (MCTS, AlphaZero, Random)
"""

from .openspiel_profiler import (
    OpenSpielProfiler,
    OpenSpielProfilerStats,
    create_profiled_agent,
    create_profiled_openspiel_alphazero_agent,
    create_profiled_openspiel_mcts_agent,
    create_profiled_random_agent,
    profile_openspiel_agents_comparison,
)
from .performance_profiler import AzulProfiler, ProfilerStats

__all__ = [
    # Original profiler
    "AzulProfiler",
    "ProfilerStats",
    # OpenSpiel profiler
    "OpenSpielProfiler",
    "OpenSpielProfilerStats",
    "create_profiled_agent",
    "create_profiled_openspiel_mcts_agent",
    "create_profiled_openspiel_alphazero_agent",
    "create_profiled_random_agent",
    "profile_openspiel_agents_comparison",
]
