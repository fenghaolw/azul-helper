"""
Tournament system for evaluating multiple agents against each other.

This module provides a comprehensive tournament framework for running
round-robin evaluations between multiple agents, generating rankings
and detailed comparative statistics.
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple

from evaluation.agent_evaluator import AgentEvaluator
from evaluation.baseline_agents import BaselineAgent
from evaluation.evaluation_config import EvaluationConfig, EvaluationResult
from evaluation.utils import get_evaluation_timestamp, save_evaluation_results


class TournamentResult:
    """Results of a multi-agent tournament."""

    def __init__(self):
        self.timestamp = get_evaluation_timestamp()
        self.agents = []  # List of agent names
        self.matchups = {}  # Dict[Tuple[str, str], EvaluationResult]
        self.win_matrix = {}  # Dict[Tuple[str, str], float] - win rates
        self.rankings = []  # List of (agent_name, total_score, win_rate)
        self.config = None

    def add_matchup(self, agent_a: str, agent_b: str, result: EvaluationResult):
        """Add a matchup result to the tournament."""
        self.matchups[(agent_a, agent_b)] = result
        self.win_matrix[(agent_a, agent_b)] = result.test_agent_win_rate

        if agent_a not in self.agents:
            self.agents.append(agent_a)
        if agent_b not in self.agents:
            self.agents.append(agent_b)

    def calculate_rankings(self):
        """Calculate final rankings based on head-to-head results."""
        agent_stats = {}

        for agent in self.agents:
            total_wins = 0
            total_games = 0
            total_score_diff = 0
            opponents_faced = 0

            # Check results where this agent was the test agent
            for (agent_a, agent_b), result in self.matchups.items():
                if agent_a == agent:
                    total_wins += result.test_agent_wins
                    total_games += result.games_played
                    total_score_diff += (
                        result.average_score_difference * result.games_played
                    )
                    opponents_faced += 1
                elif agent_b == agent:
                    total_wins += result.baseline_agent_wins
                    total_games += result.games_played
                    total_score_diff -= (
                        result.average_score_difference * result.games_played
                    )
                    opponents_faced += 1

            win_rate = total_wins / total_games if total_games > 0 else 0
            avg_score_diff = total_score_diff / total_games if total_games > 0 else 0

            agent_stats[agent] = {
                "total_wins": total_wins,
                "total_games": total_games,
                "win_rate": win_rate,
                "average_score_difference": avg_score_diff,
                "opponents_faced": opponents_faced,
            }

        # Sort agents by win rate (descending), then by average score difference
        self.rankings = sorted(
            [
                (agent, stats["win_rate"], stats["average_score_difference"], stats)
                for agent, stats in agent_stats.items()
            ],
            key=lambda x: (x[1], x[2]),  # Sort by win_rate, then score_diff
            reverse=True,
        )

    def summary(self) -> str:
        """Generate a human-readable tournament summary."""
        lines = [
            "TOURNAMENT RESULTS",
            "=" * 50,
            f"Tournament Date: {self.timestamp}",
            f"Participants: {len(self.agents)} agents",
            f"Total Matchups: {len(self.matchups)}",
            "",
            "FINAL RANKINGS:",
        ]

        for i, (agent, win_rate, score_diff, stats) in enumerate(self.rankings):
            lines.append(
                f"  {i+1}. {agent}: {win_rate:.1%} win rate, "
                f"{score_diff:+.1f} avg score diff "
                f"({stats['total_wins']}/{stats['total_games']} games)"
            )

        lines.extend(
            [
                "",
                "HEAD-TO-HEAD RESULTS:",
            ]
        )

        # Show head-to-head matrix
        lines.append("    " + " ".join(f"{agent:>8}" for agent in self.agents))
        for agent_a in self.agents:
            row = f"{agent_a:>3} "
            for agent_b in self.agents:
                if agent_a == agent_b:
                    row += "    -   "
                elif (agent_a, agent_b) in self.win_matrix:
                    win_rate = self.win_matrix[(agent_a, agent_b)]
                    row += f" {win_rate:>6.1%} "
                elif (agent_b, agent_a) in self.win_matrix:
                    win_rate = 1.0 - self.win_matrix[(agent_b, agent_a)]
                    row += f" {win_rate:>6.1%} "
                else:
                    row += "   N/A  "
            lines.append(row)

        return "\n".join(lines)


class Tournament:
    """
    Tournament system for evaluating multiple agents.

    Supports round-robin tournaments where every agent plays against
    every other agent, with comprehensive result tracking and analysis.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize tournament.

        Args:
            config: Evaluation configuration for all matchups
        """
        self.config = config or EvaluationConfig()
        self.evaluator = AgentEvaluator(self.config)
        self.result = TournamentResult()
        self.result.config = self.config

    def add_agent(self, agent: Any, name: str) -> None:
        """
        Add an agent to the tournament.

        Args:
            agent: Agent object with select_action method
            name: Unique name for the agent
        """
        if not hasattr(self, "_agents"):
            self._agents: Dict[str, Any] = {}

        if name in self._agents:
            raise ValueError(f"Agent name '{name}' already exists in tournament")

        self._agents[name] = agent

    def run_tournament(self, verbose: bool = True) -> TournamentResult:
        """
        Run a complete round-robin tournament.

        Args:
            verbose: Whether to print progress updates

        Returns:
            Tournament results with rankings and detailed matchup data
        """
        if not hasattr(self, "_agents") or len(self._agents) < 2:
            raise ValueError("Need at least 2 agents to run a tournament")

        agent_names = list(self._agents.keys())
        total_matchups = len(agent_names) * (len(agent_names) - 1) // 2

        if verbose:
            print(f"Starting tournament with {len(agent_names)} agents")
            print(f"Total matchups to play: {total_matchups}")
            print("")

        completed_matchups = 0

        # Run all pairwise matchups
        for agent_a_name, agent_b_name in itertools.combinations(agent_names, 2):
            agent_a = self._agents[agent_a_name]
            agent_b = self._agents[agent_b_name]

            if verbose:
                print(
                    f"Matchup {completed_matchups + 1}/{total_matchups}: "
                    f"{agent_a_name} vs {agent_b_name}"
                )

            # Convert agent_b to BaselineAgent if it's not already
            if not isinstance(agent_b, BaselineAgent):
                agent_b_baseline = self._wrap_as_baseline(agent_b, agent_b_name)
            else:
                agent_b_baseline = agent_b

            # Run evaluation
            result = self.evaluator.evaluate_agent(
                test_agent=agent_a,
                baseline_agent=agent_b_baseline,
                test_agent_name=agent_a_name,
                baseline_agent_name=agent_b_name,
            )

            # Add result to tournament
            self.result.add_matchup(agent_a_name, agent_b_name, result)

            completed_matchups += 1

            if verbose:
                print(
                    f"  Result: {agent_a_name} {result.test_agent_win_rate:.1%} vs "
                    f"{agent_b_name} {result.baseline_agent_win_rate:.1%}"
                )
                print("")

        # Calculate final rankings
        self.result.calculate_rankings()

        if verbose:
            print("Tournament complete!")
            print(self.result.summary())

        return self.result

    def _wrap_as_baseline(self, agent: Any, name: str) -> BaselineAgent:
        """Wrap a regular agent as a BaselineAgent for evaluation."""
        from evaluation.baseline_agents import BaselineAgent

        class WrappedAgent(BaselineAgent):
            def __init__(self, wrapped_agent, agent_name):
                super().__init__(name=agent_name)
                self.wrapped_agent = wrapped_agent

            def select_action(self, game_state):
                return self.wrapped_agent.select_action(game_state)

            def get_info(self):
                info = super().get_info()
                if hasattr(self.wrapped_agent, "get_info"):
                    info.update(self.wrapped_agent.get_info())
                return info

        return WrappedAgent(agent, name)

    def save_results(self, filepath: str) -> None:
        """Save tournament results to file."""
        # Save individual matchup results
        import os
        from pathlib import Path

        base_path = Path(filepath).parent
        base_name = Path(filepath).stem

        # Save tournament summary
        summary_path = base_path / f"{base_name}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(self.result.summary())

        # Save detailed matchup results
        matchups_dir = base_path / f"{base_name}_matchups"
        matchups_dir.mkdir(exist_ok=True)

        for (agent_a, agent_b), result in self.result.matchups.items():
            matchup_file = matchups_dir / f"{agent_a}_vs_{agent_b}.json"
            save_evaluation_results(result, str(matchup_file))

        print(f"Tournament results saved to {base_path}")


def run_baseline_comparison_tournament(
    test_agents: Dict[str, Any],
    baseline_types: List[str] = None,
    config: Optional[EvaluationConfig] = None,
    verbose: bool = True,
) -> TournamentResult:
    """
    Run a tournament comparing test agents against standard baselines.

    Args:
        test_agents: Dictionary of {name: agent} for test agents
        baseline_types: List of baseline types to include
        config: Evaluation configuration
        verbose: Whether to print progress

    Returns:
        Tournament results
    """
    if baseline_types is None:
        baseline_types = ["random", "simple_heuristic", "heuristic"]

    tournament = Tournament(config)

    # Add test agents
    for name, agent in test_agents.items():
        tournament.add_agent(agent, name)

    # Add baseline agents
    from evaluation.baseline_agents import create_baseline_agent

    for baseline_type in baseline_types:
        baseline_agent = create_baseline_agent(baseline_type)
        tournament.add_agent(baseline_agent, baseline_type.title() + "Baseline")

    return tournament.run_tournament(verbose=verbose)


def run_checkpoint_progression_tournament(
    checkpoint_paths: List[str],
    checkpoint_names: Optional[List[str]] = None,
    include_baselines: bool = True,
    config: Optional[EvaluationConfig] = None,
    verbose: bool = True,
) -> TournamentResult:
    """
    Run a tournament showing training progression across checkpoints.

    Args:
        checkpoint_paths: List of paths to model checkpoints
        checkpoint_names: Optional custom names for checkpoints
        include_baselines: Whether to include standard baseline agents
        config: Evaluation configuration
        verbose: Whether to print progress

    Returns:
        Tournament results showing training progression
    """
    tournament = Tournament(config)

    # Add checkpoint agents
    from evaluation.baseline_agents import CheckpointAgent

    for i, checkpoint_path in enumerate(checkpoint_paths):
        try:
            checkpoint_agent = CheckpointAgent(checkpoint_path)

            if checkpoint_names and i < len(checkpoint_names):
                name = checkpoint_names[i]
            else:
                name = f"Checkpoint_{i+1}"

            tournament.add_agent(checkpoint_agent, name)

        except Exception as e:
            if verbose:
                print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")

    # Add baseline agents if requested
    if include_baselines:
        from evaluation.baseline_agents import create_baseline_agent

        for baseline_type in ["random", "heuristic"]:
            baseline_agent = create_baseline_agent(baseline_type)
            tournament.add_agent(baseline_agent, baseline_type.title() + "Baseline")

    return tournament.run_tournament(verbose=verbose)
