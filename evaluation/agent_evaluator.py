"""
Main agent evaluator for conducting comprehensive agent performance assessments.

This module contains the core AgentEvaluator class that orchestrates the evaluation
process, running multiple games between test and baseline agents while collecting
detailed performance metrics and statistics.
"""

import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from evaluation.evaluation_config import EvaluationConfig, EvaluationResult, GameResult
from evaluation.utils import (
    calculate_confidence_interval,
    calculate_statistical_significance,
    get_evaluation_timestamp,
)
from game.game_state import GameState, create_game

if TYPE_CHECKING:
    from evaluation.evaluation_config import ThinkingTimeAnalysis


class AgentEvaluator:
    """
    Comprehensive agent evaluator for pitting agents against baselines.

    This class handles the complete evaluation process including:
    - Running multiple games between test and baseline agents
    - Collecting detailed performance statistics
    - Handling timeouts and errors gracefully
    - Supporting both deterministic and stochastic evaluation
    - Providing statistical analysis of results
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize the agent evaluator.

        Args:
            config: Evaluation configuration (uses defaults if None)
        """
        self.config = config or EvaluationConfig()

        # Set random seed for reproducibility
        if self.config.use_fixed_seeds and self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    def evaluate_agent(
        self,
        test_agent: Any,
        baseline_agent: Any,  # Changed from BaselineAgent to Any to accept AzulAgent
        test_agent_name: Optional[str] = None,
        baseline_agent_name: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a test agent against a baseline agent.

        Args:
            test_agent: The agent to be evaluated (must have select_action method and AzulAgent interface)
            baseline_agent: The baseline agent for comparison (must have AzulAgent interface)
            test_agent_name: Name for the test agent (for reporting)
            baseline_agent_name: Name for the baseline agent (for reporting)

        Returns:
            Comprehensive evaluation results
        """
        # Import AzulAgent here to avoid circular imports
        from agents.base_agent import AzulAgent

        # Check if agents implement the AzulAgent interface
        if not isinstance(test_agent, AzulAgent):
            raise TypeError(
                f"test_agent must be an instance of AzulAgent, got {type(test_agent)}"
            )
        if not isinstance(baseline_agent, AzulAgent):
            raise TypeError(
                f"baseline_agent must be an instance of AzulAgent, got {type(baseline_agent)}"
            )

        # Set default names - now we can use the agent's name property
        if test_agent_name is None:
            test_agent_name = test_agent.name
        if baseline_agent_name is None:
            baseline_agent_name = baseline_agent.name

        if self.config.verbose:
            print(f"Starting evaluation: {test_agent_name} vs {baseline_agent_name}")
            print(
                f"Configuration: {self.config.num_games} games, "
                f"{self.config.timeout_per_move}s timeout"
            )

        # Reset agent statistics
        baseline_agent.reset_stats()
        test_agent.reset_stats()

        # Determine games to play
        games_to_play = self._plan_games()

        # Run games
        game_results = []
        if self.config.num_workers > 1:
            game_results = self._run_games_parallel(
                test_agent, baseline_agent, games_to_play
            )
        else:
            game_results = self._run_games_sequential(
                test_agent, baseline_agent, games_to_play
            )

        # Calculate aggregate statistics
        test_agent_wins = sum(1 for gr in game_results if gr.winner == 0)
        baseline_agent_wins = sum(1 for gr in game_results if gr.winner == 1)
        draws = sum(1 for gr in game_results if gr.winner == -1)

        # Calculate confidence interval
        confidence_interval = calculate_confidence_interval(
            test_agent_wins, len(game_results), self.config.confidence_interval
        )

        # Calculate statistical significance
        p_value, is_significant = calculate_statistical_significance(
            test_agent_wins, len(game_results), baseline_agent_wins, len(game_results)
        )

        # Count timeouts and errors
        timeouts = sum(1 for gr in game_results if gr.timeout_occurred)
        errors = sum(1 for gr in game_results if gr.error_log is not None)

        # Get agent information
        test_agent_info = self._get_agent_info(test_agent)
        baseline_agent_info = baseline_agent.get_info()

        # Create result object
        result = EvaluationResult(
            timestamp=get_evaluation_timestamp(),
            config=self.config,
            test_agent_name=test_agent_name,
            baseline_agent_name=baseline_agent_name,
            test_agent_info=test_agent_info,
            baseline_agent_info=baseline_agent_info,
            games_played=len(game_results),
            game_results=game_results,
            test_agent_wins=test_agent_wins,
            baseline_agent_wins=baseline_agent_wins,
            draws=draws,
            test_agent_win_rate=0.0,  # Will be calculated in __post_init__
            baseline_agent_win_rate=0.0,  # Will be calculated in __post_init__
            average_score_difference=0.0,  # Will be calculated in __post_init__
            average_game_duration=0.0,  # Will be calculated in __post_init__
            confidence_interval=confidence_interval,
            p_value=p_value,
            is_statistically_significant=is_significant,
            timeouts=timeouts,
            errors=errors,
        )

        if self.config.verbose:
            print(f"\nEvaluation complete!")
            print(
                f"Results: {test_agent_name} {result.test_agent_win_rate:.1%} vs "
                f"{baseline_agent_name} {result.baseline_agent_win_rate:.1%}"
            )
            if result.is_statistically_significant:
                print(f"Result is statistically significant (p={result.p_value:.4f})")

        return result

    def _plan_games(self) -> List[Dict[str, Any]]:
        """
        Plan which games to play, including player position swapping.

        Returns:
            List of game configurations
        """
        games = []
        base_games = self.config.num_games

        if self.config.swap_player_positions:
            # Split games between normal and swapped positions
            normal_games = base_games // 2
            swapped_games = base_games - normal_games

            # Normal position games (test agent as player 0)
            for i in range(normal_games):
                seed = (
                    self.config.random_seed + i
                    if self.config.use_fixed_seeds and self.config.random_seed
                    else None
                )
                games.append(
                    {
                        "game_id": i,
                        "test_agent_player": 0,
                        "baseline_agent_player": 1,
                        "seed": seed,
                    }
                )

            # Swapped position games (test agent as player 1)
            for i in range(swapped_games):
                seed = (
                    self.config.random_seed + normal_games + i
                    if self.config.use_fixed_seeds and self.config.random_seed
                    else None
                )
                games.append(
                    {
                        "game_id": normal_games + i,
                        "test_agent_player": 1,
                        "baseline_agent_player": 0,
                        "seed": seed,
                    }
                )
        else:
            # All games with test agent as player 0
            for i in range(base_games):
                seed = (
                    self.config.random_seed + i
                    if self.config.use_fixed_seeds and self.config.random_seed
                    else None
                )
                games.append(
                    {
                        "game_id": i,
                        "test_agent_player": 0,
                        "baseline_agent_player": 1,
                        "seed": seed,
                    }
                )

        return games

    def _run_games_sequential(
        self,
        test_agent: Any,
        baseline_agent: Any,
        games_to_play: List[Dict[str, Any]],
    ) -> List[GameResult]:
        """Run games sequentially (single-threaded)."""
        results = []

        # Show initial progress message
        if len(games_to_play) > 1:
            print(f"Playing {len(games_to_play)} games...")

        for game_config in games_to_play:
            try:
                result = self._run_single_game(test_agent, baseline_agent, game_config)
                results.append(result)

                # Show per-game status in verbose mode
                if self.config.verbose:
                    if result.winner == 0:
                        winner_name = test_agent.name
                        result_str = "wins"
                    elif result.winner == 1:
                        winner_name = baseline_agent.name
                        result_str = "wins"
                    else:
                        winner_name = "Draw"
                        result_str = ""

                    score_str = f"({result.final_scores[0]}-{result.final_scores[1]})"
                    if result_str:
                        print(
                            f"  Game {result.game_id + 1}: {winner_name} {result_str} {score_str} in {result.num_rounds} rounds"
                        )
                    else:
                        print(
                            f"  Game {result.game_id + 1}: {winner_name} {score_str} in {result.num_rounds} rounds"
                        )

            except Exception as e:
                # Create error result
                error_result = GameResult(
                    game_id=game_config["game_id"],
                    winner=-1,
                    final_scores=[0, 0],
                    num_rounds=0,
                    game_duration=0.0,
                    agent_stats={},
                    error_log=str(e),
                    timeout_occurred=False,
                )
                results.append(error_result)

                if self.config.verbose:
                    print(f"Error in game {game_config['game_id']}: {e}")

        # Show completion message
        if len(games_to_play) > 1:
            print(f"Completed all {len(games_to_play)} games!")

        return results

    def _run_games_parallel(
        self,
        test_agent: Any,
        baseline_agent: Any,
        games_to_play: List[Dict[str, Any]],
    ) -> List[GameResult]:
        """Run games in parallel using multiple workers."""
        results = []

        # Show initial progress message
        if len(games_to_play) > 1:
            print(
                f"Playing {len(games_to_play)} games using {self.config.num_workers} workers..."
            )

        # Use ThreadPoolExecutor for I/O bound tasks or ProcessPoolExecutor for CPU bound
        # For now, using ThreadPoolExecutor as it's simpler for shared objects
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all games
            future_to_game = {
                executor.submit(
                    self._run_single_game, test_agent, baseline_agent, game_config
                ): game_config
                for game_config in games_to_play
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_game):
                game_config = future_to_game[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Show per-game status in verbose mode
                    if self.config.verbose:
                        if result.winner == 0:
                            winner_name = test_agent.name
                            result_str = "wins"
                        elif result.winner == 1:
                            winner_name = baseline_agent.name
                            result_str = "wins"
                        else:
                            winner_name = "Draw"
                            result_str = ""

                        score_str = (
                            f"({result.final_scores[0]}-{result.final_scores[1]})"
                        )
                        if result_str:
                            print(
                                f"  Game {result.game_id + 1}: {winner_name} {result_str} {score_str} in {result.num_rounds} rounds"
                            )
                        else:
                            print(
                                f"  Game {result.game_id + 1}: {winner_name} {score_str} in {result.num_rounds} rounds"
                            )

                except Exception as e:
                    # Create error result
                    error_result = GameResult(
                        game_id=game_config["game_id"],
                        winner=-1,
                        final_scores=[0, 0],
                        num_rounds=0,
                        game_duration=0.0,
                        agent_stats={},
                        error_log=str(e),
                        timeout_occurred=False,
                    )
                    results.append(error_result)

                    if self.config.verbose:
                        print(f"Error in game {game_config['game_id']}: {e}")

                completed += 1
                # Always show progress every 10 games (regardless of verbose setting)
                if completed % 10 == 0:
                    print(f"Completed {completed}/{len(games_to_play)} games")

        # Show completion message
        if len(games_to_play) > 1:
            print(f"Completed all {len(games_to_play)} games!")

        # Sort results by game_id to maintain order
        results.sort(key=lambda r: r.game_id)
        return results

    def _run_single_game(
        self,
        test_agent: Any,
        baseline_agent: Any,
        game_config: Dict[str, Any],
    ) -> GameResult:
        """
        Run a single game between test and baseline agents.

        Args:
            test_agent: The test agent
            baseline_agent: The baseline agent
            game_config: Game configuration dictionary

        Returns:
            Game result
        """
        start_time = time.time()

        # Create game
        game = create_game(
            num_players=self.config.num_players, seed=game_config["seed"]
        )

        # Set up agents
        agents: List[Any] = [None, None]
        agents[game_config["test_agent_player"]] = test_agent
        agents[game_config["baseline_agent_player"]] = baseline_agent

        # Track thinking times for each agent
        thinking_times: Dict[str, List[float]] = {
            "test_agent": [],
            "baseline_agent": [],
        }
        total_thinking_time: Dict[str, float] = {
            "test_agent": 0.0,
            "baseline_agent": 0.0,
        }

        # Track move history if requested
        move_history: Optional[List[Dict[str, Any]]] = (
            [] if self.config.save_game_replays else None
        )

        # Play the game
        timeout_occurred = False
        error_log = None
        move_count = 0
        max_moves = 500  # Safety limit to prevent infinite games
        max_game_time = self.config.timeout_per_move * 100  # Total game timeout

        try:
            while not game.game_over and move_count < max_moves:
                current_player = game.current_player
                agent = agents[current_player]

                # Check total game timeout
                total_game_time = time.time() - start_time
                if total_game_time > max_game_time:
                    timeout_occurred = True
                    error_log = f"Total game timeout: {total_game_time:.2f}s > {max_game_time:.2f}s"
                    break

                # Determine which agent type this is for thinking time tracking
                agent_type = (
                    "test_agent"
                    if current_player == game_config["test_agent_player"]
                    else "baseline_agent"
                )

                # Get action with timeout and thinking time tracking
                move_start_time = time.time()
                thinking_start_time = time.time()

                try:
                    if self.config.deterministic_evaluation and hasattr(
                        agent, "select_action"
                    ):
                        # Try to use deterministic mode if available
                        if "deterministic" in agent.select_action.__code__.co_varnames:
                            action = agent.select_action(game, deterministic=True)
                        else:
                            action = agent.select_action(game)
                    else:
                        action = agent.select_action(game)

                    # Record thinking time
                    thinking_end_time = time.time()
                    thinking_duration = thinking_end_time - thinking_start_time
                    thinking_times[agent_type].append(thinking_duration)
                    total_thinking_time[agent_type] += thinking_duration

                    # Check for move timeout
                    move_duration = time.time() - move_start_time
                    if move_duration > self.config.timeout_per_move:
                        timeout_occurred = True
                        error_log = f"Move timeout: {move_duration:.2f}s > {self.config.timeout_per_move}s"
                        break

                    # Apply action
                    success = game.apply_action(action)
                    if not success:
                        raise ValueError(f"Invalid action: {action}")

                    # Record move if tracking history
                    if move_history is not None:
                        move_history.append(
                            {
                                "player": current_player,
                                "action": str(action),
                                "game_state_summary": f"Round {game.round_number}",
                                "thinking_time": thinking_duration,
                                "agent_type": agent_type,
                            }
                        )

                    move_count += 1

                except Exception as e:
                    error_log = f"Move error: {str(e)}\n{traceback.format_exc()}"
                    break

            # Check if game exceeded move limit
            if move_count >= max_moves and not game.game_over:
                error_log = f"Game exceeded maximum moves ({max_moves})"
                timeout_occurred = True

        except Exception as e:
            error_log = f"Game error: {str(e)}\n{traceback.format_exc()}"

        # Calculate game duration
        game_duration = time.time() - start_time

        # Get final scores and winner
        final_scores = game.get_scores()

        # Determine winner (from test agent's perspective)
        if game.winner is not None:
            if game.winner == game_config["test_agent_player"]:
                winner = 0  # Test agent wins
            elif game.winner == game_config["baseline_agent_player"]:
                winner = 1  # Baseline agent wins
            else:
                winner = -1  # Draw (shouldn't happen in Azul)
        else:
            # Game didn't finish properly
            if error_log:
                winner = -1  # Draw due to error
            else:
                # Determine winner by score
                test_score = final_scores[game_config["test_agent_player"]]
                baseline_score = final_scores[game_config["baseline_agent_player"]]
                if test_score > baseline_score:
                    winner = 0
                elif baseline_score > test_score:
                    winner = 1
                else:
                    winner = -1

        # Adjust final scores to test agent perspective (test agent first)
        adjusted_scores = [
            final_scores[game_config["test_agent_player"]],
            final_scores[game_config["baseline_agent_player"]],
        ]

        # Collect agent statistics
        agent_stats = {}
        if hasattr(test_agent, "get_stats"):
            agent_stats["test_agent"] = test_agent.get_stats()
        else:
            agent_stats["test_agent"] = {}

        agent_stats["baseline_agent"] = baseline_agent.get_stats()

        # Add thinking time statistics to agent stats
        agent_stats["test_agent"]["thinking_times"] = thinking_times["test_agent"]
        agent_stats["test_agent"]["total_thinking_time"] = total_thinking_time[
            "test_agent"
        ]
        agent_stats["test_agent"]["average_thinking_time"] = total_thinking_time[
            "test_agent"
        ] / max(1, len(thinking_times["test_agent"]))
        agent_stats["test_agent"]["num_decisions"] = len(thinking_times["test_agent"])

        agent_stats["baseline_agent"]["thinking_times"] = thinking_times[
            "baseline_agent"
        ]
        agent_stats["baseline_agent"]["total_thinking_time"] = total_thinking_time[
            "baseline_agent"
        ]
        agent_stats["baseline_agent"]["average_thinking_time"] = total_thinking_time[
            "baseline_agent"
        ] / max(1, len(thinking_times["baseline_agent"]))
        agent_stats["baseline_agent"]["num_decisions"] = len(
            thinking_times["baseline_agent"]
        )

        # Create result
        return GameResult(
            game_id=game_config["game_id"],
            winner=winner,
            final_scores=adjusted_scores,
            num_rounds=game.round_number,
            game_duration=game_duration,
            agent_stats=agent_stats,
            move_history=move_history,
            error_log=error_log,
            timeout_occurred=timeout_occurred,
        )

    def _get_agent_info(self, agent: Any) -> Dict[str, Any]:
        """Get information about an agent for metadata."""
        info = {
            "agent_type": agent.__class__.__name__,
        }

        # Try to get additional info if available
        if hasattr(agent, "get_info"):
            info.update(agent.get_info())
        elif hasattr(agent, "name"):
            info["name"] = agent.name

        return info

    def analyze_thinking_times(
        self, evaluation_result: EvaluationResult
    ) -> "ThinkingTimeAnalysis":
        """
        Analyze thinking time statistics from evaluation results.

        Args:
            evaluation_result: The evaluation result to analyze

        Returns:
            ThinkingTimeAnalysis object containing thinking time analysis
        """
        from evaluation.evaluation_config import ThinkingTimeAnalysis

        analysis = ThinkingTimeAnalysis()

        # Aggregate stats across all games
        for game_result in evaluation_result.game_results:
            if "test_agent" in game_result.agent_stats:
                test_stats = game_result.agent_stats["test_agent"]
                if "thinking_times" in test_stats and isinstance(
                    test_stats["thinking_times"], list
                ):
                    thinking_times = test_stats["thinking_times"]
                    if thinking_times:
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

    def quick_evaluation(
        self,
        test_agent: Any,
        baseline_agent: Any,
        num_games: int = 20,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Run a quick evaluation with minimal configuration.

        Args:
            test_agent: Agent to evaluate
            baseline_agent: Baseline for comparison
            num_games: Number of games to play
            verbose: Whether to print progress

        Returns:
            Evaluation results
        """
        # Create quick config
        quick_config = EvaluationConfig(
            num_games=num_games,
            timeout_per_move=2.0,
            verbose=verbose,
            num_workers=1,
            save_detailed_logs=False,
            save_game_replays=False,
        )

        # Temporarily replace config
        original_config = self.config
        self.config = quick_config

        try:
            result = self.evaluate_agent(test_agent, baseline_agent)
            return result
        finally:
            # Restore original config
            self.config = original_config
