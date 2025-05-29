"""
Configuration classes for agent evaluation.

This module defines the configuration and result classes used throughout
the evaluation framework.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class EvaluationConfig:
    """Configuration for agent evaluation experiments."""

    # Basic evaluation settings
    num_games: int = 100
    num_players: int = 2
    timeout_per_move: float = 5.0  # seconds

    # Agent evaluation settings
    deterministic_evaluation: bool = True  # Use deterministic agent actions if possible
    swap_player_positions: bool = (
        True  # Evaluate with agents in different starting positions
    )

    # Randomization settings
    use_fixed_seeds: bool = True  # Use fixed seeds for reproducibility
    random_seed: Optional[int] = 42

    # Parallel execution
    num_workers: int = 1  # Number of parallel workers for evaluation

    # Logging and output
    verbose: bool = True
    save_detailed_logs: bool = False
    save_game_replays: bool = False

    # Result aggregation
    confidence_interval: float = 0.95  # For statistical significance testing

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "num_games": self.num_games,
            "num_players": self.num_players,
            "timeout_per_move": self.timeout_per_move,
            "deterministic_evaluation": self.deterministic_evaluation,
            "swap_player_positions": self.swap_player_positions,
            "use_fixed_seeds": self.use_fixed_seeds,
            "random_seed": self.random_seed,
            "num_workers": self.num_workers,
            "verbose": self.verbose,
            "save_detailed_logs": self.save_detailed_logs,
            "save_game_replays": self.save_game_replays,
            "confidence_interval": self.confidence_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        """Create config from dictionary."""
        return cls(**data)


@dataclass
class GameResult:
    """Result of a single game."""

    game_id: int
    winner: int  # Player index of winner
    final_scores: List[int]  # Final scores for all players
    num_rounds: int
    game_duration: float  # seconds
    agent_stats: Dict[str, Any]  # Per-agent statistics

    # Optional detailed information
    move_history: Optional[List[Any]] = None
    error_log: Optional[str] = None
    timeout_occurred: bool = False


@dataclass
class EvaluationResult:
    """Comprehensive results of an agent evaluation."""

    # Evaluation metadata
    timestamp: str
    config: EvaluationConfig

    # Agent information
    test_agent_name: str
    baseline_agent_name: str
    test_agent_info: Dict[str, Any]
    baseline_agent_info: Dict[str, Any]

    # Game results
    games_played: int
    game_results: List[GameResult]

    # Aggregate statistics
    test_agent_wins: int
    baseline_agent_wins: int
    draws: int

    # Performance metrics
    test_agent_win_rate: float
    baseline_agent_win_rate: float
    average_score_difference: float
    average_game_duration: float

    # Statistical analysis
    confidence_interval: Optional[tuple] = None
    p_value: Optional[float] = None
    is_statistically_significant: Optional[bool] = None

    # Additional metrics
    timeouts: int = 0
    errors: int = 0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.games_played > 0:
            self.test_agent_win_rate = self.test_agent_wins / self.games_played
            self.baseline_agent_win_rate = self.baseline_agent_wins / self.games_played

            # Calculate average score difference (test_agent - baseline_agent)
            total_score_diff = 0
            for game_result in self.game_results:
                if len(game_result.final_scores) >= 2:
                    total_score_diff += (
                        game_result.final_scores[0] - game_result.final_scores[1]
                    )
            self.average_score_difference = total_score_diff / self.games_played

            # Calculate average game duration
            total_duration = sum(gr.game_duration for gr in self.game_results)
            self.average_game_duration = total_duration / self.games_played

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "test_agent_name": self.test_agent_name,
            "baseline_agent_name": self.baseline_agent_name,
            "test_agent_info": self.test_agent_info,
            "baseline_agent_info": self.baseline_agent_info,
            "games_played": self.games_played,
            "test_agent_wins": self.test_agent_wins,
            "baseline_agent_wins": self.baseline_agent_wins,
            "draws": self.draws,
            "test_agent_win_rate": self.test_agent_win_rate,
            "baseline_agent_win_rate": self.baseline_agent_win_rate,
            "average_score_difference": self.average_score_difference,
            "average_game_duration": self.average_game_duration,
            "confidence_interval": self.confidence_interval,
            "p_value": self.p_value,
            "is_statistically_significant": self.is_statistically_significant,
            "timeouts": self.timeouts,
            "errors": self.errors,
            "game_results": [
                {
                    "game_id": gr.game_id,
                    "winner": gr.winner,
                    "final_scores": gr.final_scores,
                    "num_rounds": gr.num_rounds,
                    "game_duration": gr.game_duration,
                    "agent_stats": gr.agent_stats,
                    "timeout_occurred": gr.timeout_occurred,
                    "error_log": gr.error_log,
                }
                for gr in self.game_results
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create result from dictionary."""
        # Reconstruct config
        config = EvaluationConfig.from_dict(data["config"])

        # Reconstruct game results
        game_results = []
        for gr_data in data["game_results"]:
            game_results.append(
                GameResult(
                    game_id=gr_data["game_id"],
                    winner=gr_data["winner"],
                    final_scores=gr_data["final_scores"],
                    num_rounds=gr_data["num_rounds"],
                    game_duration=gr_data["game_duration"],
                    agent_stats=gr_data["agent_stats"],
                    timeout_occurred=gr_data.get("timeout_occurred", False),
                    error_log=gr_data.get("error_log"),
                )
            )

        # Create result object
        result = cls(
            timestamp=data["timestamp"],
            config=config,
            test_agent_name=data["test_agent_name"],
            baseline_agent_name=data["baseline_agent_name"],
            test_agent_info=data["test_agent_info"],
            baseline_agent_info=data["baseline_agent_info"],
            games_played=data["games_played"],
            game_results=game_results,
            test_agent_wins=data["test_agent_wins"],
            baseline_agent_wins=data["baseline_agent_wins"],
            draws=data["draws"],
            test_agent_win_rate=data["test_agent_win_rate"],
            baseline_agent_win_rate=data["baseline_agent_win_rate"],
            average_score_difference=data["average_score_difference"],
            average_game_duration=data["average_game_duration"],
            confidence_interval=data.get("confidence_interval"),
            p_value=data.get("p_value"),
            is_statistically_significant=data.get("is_statistically_significant"),
            timeouts=data.get("timeouts", 0),
            errors=data.get("errors", 0),
        )

        return result

    def summary(self) -> str:
        """Generate a human-readable summary of the evaluation."""
        lines = [
            f"Evaluation Results: {self.test_agent_name} vs {self.baseline_agent_name}",
            f"Timestamp: {self.timestamp}",
            f"Games Played: {self.games_played}",
            "",
            f"Win Rates:",
            f"  {self.test_agent_name}: {self.test_agent_win_rate:.1%} ({self.test_agent_wins} wins)",
            f"  {self.baseline_agent_name}: {self.baseline_agent_win_rate:.1%} ({self.baseline_agent_wins} wins)",
            f"  Draws: {self.draws}",
            "",
            f"Performance Metrics:",
            f"  Average Score Difference: {self.average_score_difference:+.1f}",
            f"  Average Game Duration: {self.average_game_duration:.1f}s",
        ]

        if self.is_statistically_significant is not None:
            lines.extend(
                [
                    "",
                    f"Statistical Analysis:",
                    (
                        f"  P-value: {self.p_value:.4f}"
                        if self.p_value
                        else "  P-value: N/A"
                    ),
                    f"  Statistically Significant: {self.is_statistically_significant}",
                ]
            )

        if self.timeouts > 0 or self.errors > 0:
            lines.extend(
                [
                    "",
                    f"Issues:",
                    f"  Timeouts: {self.timeouts}",
                    f"  Errors: {self.errors}",
                ]
            )

        return "\n".join(lines)
