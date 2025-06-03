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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from evaluation.evaluation_config import ThinkingTimeAnalysis


def _analyze_thinking_times_from_result(result) -> Optional["ThinkingTimeAnalysis"]:
    """
    Analyze thinking time statistics from evaluation results.

    Args:
        result: EvaluationResult object

    Returns:
        ThinkingTimeAnalysis object, or None if no thinking time data
    """
    from evaluation.evaluation_config import ThinkingTimeAnalysis

    analysis = ThinkingTimeAnalysis()
    has_thinking_data = False

    # Aggregate stats across all games
    for game_result in result.game_results:
        if hasattr(game_result, "agent_stats") and game_result.agent_stats:
            if "test_agent" in game_result.agent_stats:
                test_stats = game_result.agent_stats["test_agent"]
                if "thinking_times" in test_stats and isinstance(
                    test_stats["thinking_times"], list
                ):
                    thinking_times = test_stats["thinking_times"]
                    if thinking_times:
                        has_thinking_data = True
                        total_thinking_time = test_stats.get("total_thinking_time", 0.0)
                        if isinstance(total_thinking_time, (int, float)):
                            analysis.test_agent_total_thinking_time += float(
                                total_thinking_time
                            )
                        analysis.test_agent_total_decisions += len(thinking_times)
                        if isinstance(total_thinking_time, (int, float)):
                            analysis.test_agent_thinking_time_per_game.append(
                                float(total_thinking_time)
                            )

                        # Only process if thinking_times contains numbers
                        numeric_times = [
                            t for t in thinking_times if isinstance(t, (int, float))
                        ]
                        if numeric_times:
                            if analysis.test_agent_min_thinking_time == 0.0:
                                analysis.test_agent_min_thinking_time = min(
                                    numeric_times
                                )
                            else:
                                analysis.test_agent_min_thinking_time = min(
                                    analysis.test_agent_min_thinking_time,
                                    min(numeric_times),
                                )
                            analysis.test_agent_max_thinking_time = max(
                                analysis.test_agent_max_thinking_time,
                                max(numeric_times),
                            )

            if "baseline_agent" in game_result.agent_stats:
                baseline_stats = game_result.agent_stats["baseline_agent"]
                if "thinking_times" in baseline_stats and isinstance(
                    baseline_stats["thinking_times"], list
                ):
                    thinking_times = baseline_stats["thinking_times"]
                    if thinking_times:
                        has_thinking_data = True
                        total_thinking_time = baseline_stats.get(
                            "total_thinking_time", 0.0
                        )
                        if isinstance(total_thinking_time, (int, float)):
                            analysis.baseline_agent_total_thinking_time += float(
                                total_thinking_time
                            )
                        analysis.baseline_agent_total_decisions += len(thinking_times)
                        if isinstance(total_thinking_time, (int, float)):
                            analysis.baseline_agent_thinking_time_per_game.append(
                                float(total_thinking_time)
                            )

                        # Only process if thinking_times contains numbers
                        numeric_times = [
                            t for t in thinking_times if isinstance(t, (int, float))
                        ]
                        if numeric_times:
                            if analysis.baseline_agent_min_thinking_time == 0.0:
                                analysis.baseline_agent_min_thinking_time = min(
                                    numeric_times
                                )
                            else:
                                analysis.baseline_agent_min_thinking_time = min(
                                    analysis.baseline_agent_min_thinking_time,
                                    min(numeric_times),
                                )
                            analysis.baseline_agent_max_thinking_time = max(
                                analysis.baseline_agent_max_thinking_time,
                                max(numeric_times),
                            )

    if not has_thinking_data:
        return None

    # Calculate averages
    if analysis.test_agent_total_decisions > 0:
        analysis.test_agent_average_thinking_time = (
            analysis.test_agent_total_thinking_time
            / analysis.test_agent_total_decisions
        )

    if analysis.test_agent_thinking_time_per_game:
        analysis.test_agent_average_thinking_time_per_game = sum(
            analysis.test_agent_thinking_time_per_game
        ) / len(analysis.test_agent_thinking_time_per_game)

    if analysis.baseline_agent_total_decisions > 0:
        analysis.baseline_agent_average_thinking_time = (
            analysis.baseline_agent_total_thinking_time
            / analysis.baseline_agent_total_decisions
        )

    if analysis.baseline_agent_thinking_time_per_game:
        analysis.baseline_agent_average_thinking_time_per_game = sum(
            analysis.baseline_agent_thinking_time_per_game
        ) / len(analysis.baseline_agent_thinking_time_per_game)

    # Add comparison metrics
    analysis.test_agent_thinks_longer = (
        analysis.test_agent_average_thinking_time
        > analysis.baseline_agent_average_thinking_time
    )
    analysis.thinking_time_ratio = analysis.test_agent_average_thinking_time / max(
        analysis.baseline_agent_average_thinking_time, 1e-6
    )
    analysis.total_thinking_time_difference = (
        analysis.test_agent_total_thinking_time
        - analysis.baseline_agent_total_thinking_time
    )

    return analysis


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
        f"Agent A: {result.test_agent_name}",
        f"Agent B: {result.baseline_agent_name}",
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

    # Add thinking time analysis if available
    thinking_analysis = _analyze_thinking_times_from_result(result)
    if thinking_analysis and (
        thinking_analysis.test_agent_total_decisions > 0
        or thinking_analysis.baseline_agent_total_decisions > 0
    ):
        lines.extend(
            [
                "",
                "THINKING TIME ANALYSIS:",
                f"  {result.test_agent_name}:",
                f"    Average per decision: {thinking_analysis.test_agent_average_thinking_time:.3f}s",
                f"    Total decisions: {thinking_analysis.test_agent_total_decisions}",
                f"    Total thinking time: {thinking_analysis.test_agent_total_thinking_time:.2f}s",
                f"    Min/Max decision time: {thinking_analysis.test_agent_min_thinking_time:.3f}s / {thinking_analysis.test_agent_max_thinking_time:.3f}s",
                f"  {result.baseline_agent_name}:",
                f"    Average per decision: {thinking_analysis.baseline_agent_average_thinking_time:.3f}s",
                f"    Total decisions: {thinking_analysis.baseline_agent_total_decisions}",
                f"    Total thinking time: {thinking_analysis.baseline_agent_total_thinking_time:.2f}s",
                f"    Min/Max decision time: {thinking_analysis.baseline_agent_min_thinking_time:.3f}s / {thinking_analysis.baseline_agent_max_thinking_time:.3f}s",
                "",
                "THINKING TIME COMPARISON:",
                f"  Ratio ({result.test_agent_name}/{result.baseline_agent_name}): {thinking_analysis.thinking_time_ratio:.2f}x",
                f"  {result.test_agent_name + ' thinks longer' if thinking_analysis.test_agent_thinks_longer else result.baseline_agent_name + ' thinks longer'}",
                f"  Time difference: {thinking_analysis.total_thinking_time_difference:+.2f}s total",
            ]
        )

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
                f"  {result.test_agent_name} Info: {result.test_agent_info}",
                f"  {result.baseline_agent_name} Info: {result.baseline_agent_info}",
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
            if game_result.winner == 0:
                winner_text = f"{result.test_agent_name} wins"
            elif game_result.winner == 1:
                winner_text = f"{result.baseline_agent_name} wins"
            else:
                winner_text = "Draw"

            lines.append(
                f"  Game {game_result.game_id + 1}: {winner_text} "
                f"({game_result.final_scores[0]}-{game_result.final_scores[1]}) "
                f"in {game_result.num_rounds} rounds"
            )

        if len(result.game_results) > 10:
            lines.append(f"  ... and {len(result.game_results) - 10} more games")

        # Add detailed thinking time statistics if available
        if thinking_analysis and detailed:
            lines.extend(
                [
                    "",
                    "DETAILED THINKING TIME ANALYSIS:",
                    f"  {result.test_agent_name} thinking time per game:",
                ]
            )

            test_times = thinking_analysis.test_agent_thinking_time_per_game[:10]
            for i, time_per_game in enumerate(test_times):
                lines.append(f"    Game {i+1}: {time_per_game:.3f}s")

            if len(thinking_analysis.test_agent_thinking_time_per_game) > 10:
                lines.append(
                    f"    ... and {len(thinking_analysis.test_agent_thinking_time_per_game) - 10} more games"
                )

            lines.extend(
                [
                    f"  {result.baseline_agent_name} thinking time per game:",
                ]
            )

            baseline_times = thinking_analysis.baseline_agent_thinking_time_per_game[
                :10
            ]
            for i, time_per_game in enumerate(baseline_times):
                lines.append(f"    Game {i+1}: {time_per_game:.3f}s")

            if len(thinking_analysis.baseline_agent_thinking_time_per_game) > 10:
                lines.append(
                    f"    ... and {len(thinking_analysis.baseline_agent_thinking_time_per_game) - 10} more games"
                )

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
