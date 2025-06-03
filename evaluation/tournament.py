"""
Tournament system for evaluating multiple agents against each other.

This module provides a comprehensive tournament framework for running
round-robin evaluations between multiple agents, generating rankings
and detailed comparative statistics.
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple

from evaluation.agent_evaluator import AgentEvaluator
from evaluation.evaluation_config import EvaluationConfig, EvaluationResult
from evaluation.utils import get_evaluation_timestamp, save_evaluation_results
from training.eta_tracker import ETATracker


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

        # Initialize ETA tracker for tournament progress
        eta_tracker = None
        if verbose and total_matchups > 1:
            eta_tracker = ETATracker(
                total_iterations=total_matchups,
                moving_average_window=min(5, total_matchups),
                enable_phase_tracking=False,
            )
            print(f"Starting tournament with {len(agent_names)} agents")
            print(f"Total matchups to play: {total_matchups}")
            print("")

        completed_matchups = 0

        # Run all pairwise matchups
        for agent_a_name, agent_b_name in itertools.combinations(agent_names, 2):
            agent_a = self._agents[agent_a_name]
            agent_b = self._agents[agent_b_name]

            # Start matchup tracking
            if eta_tracker:
                eta_tracker.start_iteration(completed_matchups + 1)

            if verbose:
                base_msg = f"Matchup {completed_matchups + 1}/{total_matchups}: {agent_a_name} vs {agent_b_name}"

                # Add ETA info for tournament progress
                if eta_tracker and completed_matchups > 0:
                    eta_estimates = eta_tracker.get_eta_estimates()
                    if eta_estimates["best_estimate"]:
                        eta_str = eta_tracker.format_time_display(
                            eta_estimates["best_estimate"]
                        )
                        progress_pct = (completed_matchups / total_matchups) * 100
                        base_msg += f" | Tournament Progress: {progress_pct:.1f}% | ETA: {eta_str}"

                print(base_msg)

            # Import AzulAgent here to avoid circular imports
            from agents.base_agent import AzulAgent

            # Verify both agents are AzulAgent instances
            if not isinstance(agent_a, AzulAgent):
                raise TypeError(
                    f"Agent {agent_a_name} must be an AzulAgent instance, got {type(agent_a)}"
                )
            if not isinstance(agent_b, AzulAgent):
                raise TypeError(
                    f"Agent {agent_b_name} must be an AzulAgent instance, got {type(agent_b)}"
                )

            # Run evaluation directly with both agents as AzulAgent instances
            result = self.evaluator.evaluate_agent(
                test_agent=agent_a,
                baseline_agent=agent_b,  # No wrapper needed - agent_b is already AzulAgent
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

            # End matchup tracking
            if eta_tracker:
                eta_tracker.end_iteration()

        # Calculate final rankings
        self.result.calculate_rankings()

        if verbose:
            if eta_tracker:
                summary = eta_tracker.get_progress_summary()
                elapsed_str = eta_tracker.format_time_display(summary["elapsed_time"])
                avg_matchup_str = eta_tracker.format_time_display(
                    summary["avg_iteration_time"]
                )
                print(
                    f"Tournament complete! Total time: {elapsed_str} (avg: {avg_matchup_str} per matchup)"
                )
            else:
                print("Tournament complete!")
            print(self.result.summary())

        return self.result

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
