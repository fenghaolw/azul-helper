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
from typing import Any, Dict, List, Optional, Tuple, Union

from evaluation.baseline_agents import BaselineAgent
from evaluation.evaluation_config import EvaluationConfig, EvaluationResult, GameResult
from evaluation.utils import (
    calculate_confidence_interval,
    calculate_statistical_significance,
    get_evaluation_timestamp,
)
from game.game_state import GameState, create_game


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
        baseline_agent: BaselineAgent,
        test_agent_name: Optional[str] = None,
        baseline_agent_name: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a test agent against a baseline agent.

        Args:
            test_agent: The agent to be evaluated (must have select_action method)
            baseline_agent: The baseline agent for comparison
            test_agent_name: Name for the test agent (for reporting)
            baseline_agent_name: Name for the baseline agent (for reporting)

        Returns:
            Comprehensive evaluation results
        """
        # Set default names
        if test_agent_name is None:
            test_agent_name = getattr(test_agent, "name", test_agent.__class__.__name__)
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
        if hasattr(test_agent, "reset_stats"):
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
        baseline_agent: BaselineAgent,
        games_to_play: List[Dict[str, Any]],
    ) -> List[GameResult]:
        """Run games sequentially in the main thread."""
        results = []

        for i, game_config in enumerate(games_to_play):
            if self.config.verbose and (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{len(games_to_play)} games")

            try:
                result = self._run_single_game(test_agent, baseline_agent, game_config)
                results.append(result)
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

        return results

    def _run_games_parallel(
        self,
        test_agent: Any,
        baseline_agent: BaselineAgent,
        games_to_play: List[Dict[str, Any]],
    ) -> List[GameResult]:
        """Run games in parallel using multiple workers."""
        results = []

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
                if self.config.verbose and completed % 10 == 0:
                    print(f"Completed {completed}/{len(games_to_play)} games")

        # Sort results by game_id to maintain order
        results.sort(key=lambda r: r.game_id)
        return results

    def _run_single_game(
        self,
        test_agent: Any,
        baseline_agent: BaselineAgent,
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

                # Get action with timeout
                move_start_time = time.time()
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
        agent_stats["baseline_agent"] = baseline_agent.get_stats()

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

    def quick_evaluation(
        self,
        test_agent: Any,
        baseline_agent: BaselineAgent,
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
