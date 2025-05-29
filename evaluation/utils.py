"""
Utility functions for agent evaluation.

This module provides statistical analysis, result formatting, and I/O utilities
for the evaluation framework.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def calculate_win_rate(wins: int, total_games: int) -> float:
    """
    Calculate win rate as a percentage.

    Args:
        wins: Number of wins
        total_games: Total number of games played

    Returns:
        Win rate as a float between 0 and 1
    """
    if total_games == 0:
        return 0.0
    return wins / total_games


def calculate_confidence_interval(
    successes: int, total: int, confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate binomial confidence interval for win rate.

    Args:
        successes: Number of successes (wins)
        total: Total number of trials (games)
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for confidence interval
    """
    if total == 0:
        return (0.0, 0.0)

    # Use Wilson score interval for better performance with small samples
    z = stats.norm.ppf((1 + confidence_level) / 2)
    p = successes / total
    n = total

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def calculate_statistical_significance(
    wins_a: int, total_a: int, wins_b: int, total_b: int, alpha: float = 0.05
) -> Tuple[float, bool]:
    """
    Calculate statistical significance between two win rates using Fisher's exact test.

    Args:
        wins_a: Wins for agent A
        total_a: Total games for agent A
        wins_b: Wins for agent B
        total_b: Total games for agent B
        alpha: Significance level (default 0.05)

    Returns:
        Tuple of (p_value, is_significant)
    """
    # Create contingency table
    # Rows: Agent A, Agent B
    # Columns: Wins, Losses
    losses_a = total_a - wins_a
    losses_b = total_b - wins_b

    contingency_table = [[wins_a, losses_a], [wins_b, losses_b]]

    try:
        # Use Fisher's exact test for small samples, chi-square for large
        if total_a < 30 or total_b < 30:
            odds_ratio, p_value = stats.fisher_exact(contingency_table)
        else:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Convert numpy types to native Python types for JSON serialization
        p_value = float(p_value)
        is_significant = bool(p_value < alpha)
        return p_value, is_significant

    except Exception:
        # Fallback to simple z-test for proportions
        p1 = wins_a / total_a if total_a > 0 else 0
        p2 = wins_b / total_b if total_b > 0 else 0

        if total_a == 0 or total_b == 0:
            return 1.0, False

        # Pooled proportion
        p_pool = (wins_a + wins_b) / (total_a + total_b)

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / total_a + 1 / total_b))

        if se == 0:
            return 1.0, False

        # Z-statistic
        z = (p1 - p2) / se

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Convert to native Python types for JSON serialization
        p_value = float(p_value)
        is_significant = bool(p_value < alpha)

        return p_value, is_significant


def format_evaluation_results(result, detailed: bool = False) -> str:
    """
    Format evaluation results as a human-readable string.

    Args:
        result: EvaluationResult object
        detailed: Whether to include detailed statistics

    Returns:
        Formatted string representation
    """
    lines = [
        "=" * 60,
        f"AGENT EVALUATION RESULTS",
        "=" * 60,
        "",
        f"Test Agent: {result.test_agent_name}",
        f"Baseline Agent: {result.baseline_agent_name}",
        f"Evaluation Time: {result.timestamp}",
        f"Games Played: {result.games_played}",
        "",
        "WIN RATES:",
        f"  {result.test_agent_name}: {result.test_agent_win_rate:.1%} ({result.test_agent_wins} wins)",
        f"  {result.baseline_agent_name}: {result.baseline_agent_win_rate:.1%} ({result.baseline_agent_wins} wins)",
        f"  Draws: {result.draws}",
        "",
        "PERFORMANCE METRICS:",
        f"  Average Score Difference: {result.average_score_difference:+.1f}",
        f"  Average Game Duration: {result.average_game_duration:.1f}s",
    ]

    # Add confidence interval if available
    if result.confidence_interval:
        lower, upper = result.confidence_interval
        lines.extend(
            [
                "",
                "STATISTICAL ANALYSIS:",
                f"  95% Confidence Interval: [{lower:.1%}, {upper:.1%}]",
            ]
        )

    # Add significance testing if available
    if result.p_value is not None:
        significance_str = "Yes" if result.is_statistically_significant else "No"
        lines.extend(
            [
                f"  P-value: {result.p_value:.4f}",
                f"  Statistically Significant: {significance_str}",
            ]
        )

    # Add issues if any
    if result.timeouts > 0 or result.errors > 0:
        lines.extend(
            [
                "",
                "ISSUES:",
                f"  Timeouts: {result.timeouts}",
                f"  Errors: {result.errors}",
            ]
        )

    # Add detailed statistics if requested
    if detailed:
        lines.extend(
            [
                "",
                "DETAILED STATISTICS:",
                f"  Configuration: {result.config.num_games} games, {result.config.timeout_per_move}s timeout",
                f"  Test Agent Info: {result.test_agent_info}",
                f"  Baseline Agent Info: {result.baseline_agent_info}",
            ]
        )

        # Game-by-game results
        lines.extend(
            [
                "",
                "GAME RESULTS:",
            ]
        )

        for i, game_result in enumerate(
            result.game_results[:10]
        ):  # Show first 10 games
            winner_name = (
                result.test_agent_name
                if game_result.winner == 0
                else result.baseline_agent_name
            )
            lines.append(
                f"  Game {game_result.game_id + 1}: {winner_name} wins "
                f"({game_result.final_scores[0]}-{game_result.final_scores[1]}) "
                f"in {game_result.num_rounds} rounds"
            )

        if len(result.game_results) > 10:
            lines.append(f"  ... and {len(result.game_results) - 10} more games")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def save_evaluation_results(result, filepath: str) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        result: EvaluationResult object
        filepath: Path to save the results
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Convert result to dictionary and save
    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def load_evaluation_results(filepath: str):
    """
    Load evaluation results from a JSON file.

    Args:
        filepath: Path to the results file

    Returns:
        EvaluationResult object
    """
    from evaluation.evaluation_config import EvaluationResult

    with open(filepath, "r") as f:
        data = json.load(f)

    return EvaluationResult.from_dict(data)


def create_evaluation_summary(
    results_dir: str, output_file: Optional[str] = None
) -> str:
    """
    Create a summary of multiple evaluation results.

    Args:
        results_dir: Directory containing evaluation result files
        output_file: Optional file to save the summary

    Returns:
        Summary string
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # Load all result files
    results = []
    for json_file in results_path.glob("*.json"):
        try:
            result = load_evaluation_results(str(json_file))
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    if not results:
        return "No valid evaluation results found."

    # Sort by timestamp
    results.sort(key=lambda r: r.timestamp)

    # Create summary
    lines = [
        "EVALUATION SUMMARY",
        "=" * 50,
        f"Total Evaluations: {len(results)}",
        f"Date Range: {results[0].timestamp} to {results[-1].timestamp}",
        "",
        "RESULTS BY AGENT:",
    ]

    # Group by test agent
    agent_results: Dict[str, List] = {}
    for result in results:
        agent_name = result.test_agent_name
        if agent_name not in agent_results:
            agent_results[agent_name] = []
        agent_results[agent_name].append(result)

    for agent_name, agent_evals in agent_results.items():
        lines.extend(
            [
                f"\n{agent_name}:",
                f"  Evaluations: {len(agent_evals)}",
            ]
        )

        # Calculate aggregate statistics
        total_wins = sum(r.test_agent_wins for r in agent_evals)
        total_games = sum(r.games_played for r in agent_evals)
        avg_win_rate = total_wins / total_games if total_games > 0 else 0

        lines.extend(
            [
                f"  Overall Win Rate: {avg_win_rate:.1%} ({total_wins}/{total_games})",
                f"  Average Score Difference: {np.mean([r.average_score_difference for r in agent_evals]):+.1f}",
            ]
        )

        # List individual evaluations
        for result in agent_evals:
            lines.append(
                f"    vs {result.baseline_agent_name}: {result.test_agent_win_rate:.1%} "
                f"({result.test_agent_wins}/{result.games_played})"
            )

    summary = "\n".join(lines)

    # Save to file if requested
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(summary)

    return summary


def compare_agents(
    results_a: List,
    results_b: List,
    agent_a_name: str = "Agent A",
    agent_b_name: str = "Agent B",
) -> str:
    """
    Compare performance between two sets of evaluation results.

    Args:
        results_a: List of EvaluationResult objects for agent A
        results_b: List of EvaluationResult objects for agent B
        agent_a_name: Name for agent A
        agent_b_name: Name for agent B

    Returns:
        Comparison summary string
    """
    if not results_a or not results_b:
        return "Cannot compare: insufficient results"

    # Calculate aggregate statistics
    total_wins_a = sum(r.test_agent_wins for r in results_a)
    total_games_a = sum(r.games_played for r in results_a)
    win_rate_a = total_wins_a / total_games_a if total_games_a > 0 else 0

    total_wins_b = sum(r.test_agent_wins for r in results_b)
    total_games_b = sum(r.games_played for r in results_b)
    win_rate_b = total_wins_b / total_games_b if total_games_b > 0 else 0

    # Statistical significance
    p_value, is_significant = calculate_statistical_significance(
        total_wins_a, total_games_a, total_wins_b, total_games_b
    )

    lines = [
        f"AGENT COMPARISON: {agent_a_name} vs {agent_b_name}",
        "=" * 50,
        "",
        f"{agent_a_name}:",
        f"  Win Rate: {win_rate_a:.1%} ({total_wins_a}/{total_games_a})",
        f"  Evaluations: {len(results_a)}",
        f"  Avg Score Diff: {np.mean([r.average_score_difference for r in results_a]):+.1f}",
        "",
        f"{agent_b_name}:",
        f"  Win Rate: {win_rate_b:.1%} ({total_wins_b}/{total_games_b})",
        f"  Evaluations: {len(results_b)}",
        f"  Avg Score Diff: {np.mean([r.average_score_difference for r in results_b]):+.1f}",
        "",
        "COMPARISON:",
        f"  Win Rate Difference: {win_rate_a - win_rate_b:+.1%}",
        f"  P-value: {p_value:.4f}",
        f"  Statistically Significant: {'Yes' if is_significant else 'No'}",
    ]

    return "\n".join(lines)


def get_evaluation_timestamp() -> str:
    """Get a timestamp string for evaluation results."""
    return datetime.now().isoformat()


def ensure_evaluation_dir(base_dir: str = "evaluation_results") -> str:
    """
    Ensure evaluation results directory exists.

    Args:
        base_dir: Base directory for evaluation results

    Returns:
        Path to the evaluation directory
    """
    eval_dir = Path(base_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    return str(eval_dir)
