#!/usr/bin/env python3
"""
Main entry point for the Agent Evaluation Framework.

This script provides easy access to common evaluation tasks including
quick tests, comprehensive evaluations, tournaments, and improved agent comparisons.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Union

import torch

# Add evaluation module to path
sys.path.append(str(Path(__file__).parent))

from evaluation import (
    EvaluationConfig,
    format_evaluation_results,
    save_evaluation_results,
)
from evaluation.agent_evaluator import AgentEvaluator
from evaluation.baseline_agents import HeuristicBaselineAgent, RandomBaselineAgent
from evaluation.utils import ensure_evaluation_dir


# --- Generalized ProgressTrackingEvaluator ---
class ProgressTrackingEvaluator(AgentEvaluator):
    """
    AgentEvaluator subclass that tracks and prints progress, win rate, and agent stats (nodes, depth) during evaluation.
    Can be used for any agent that exposes get_stats().
    """

    def __init__(
        self,
        config,
        tracked_agent,
        tracked_agent_role: str = "test",  # 'test' or 'baseline'
        tracked_agent_name: str = "Agent",
        baseline_name: str = "Baseline",
    ):
        super().__init__(config)
        self.tracked_agent = tracked_agent
        self.tracked_agent_role = tracked_agent_role
        self.tracked_agent_name = tracked_agent_name
        self.baseline_name = baseline_name
        self.game_count = 0
        self.total_games = config.num_games
        self.wins = 0
        self.start_time = time.time()
        self.last_nodes = 0

    def _run_single_game(self, test_agent, baseline_agent, game_config):
        """Override to add progress tracking."""
        game_start = time.time()

        # Pick which agent to track
        agent = test_agent if self.tracked_agent_role == "test" else baseline_agent

        # Get nodes before game
        nodes_before = 0
        if hasattr(agent, "get_stats"):
            stats = agent.get_stats()
            nodes_before = stats.get("nodes_evaluated", 0)

        # Run the actual game
        result = super()._run_single_game(test_agent, baseline_agent, game_config)

        game_time = time.time() - game_start
        self.game_count += 1

        # Track win/loss based on the normalized result
        if result.winner == 0:  # Test agent wins
            self.wins += 1
            outcome = "🏆"
        elif result.winner == 1:  # Baseline wins
            outcome = "❌"
        else:  # Draw
            outcome = "🤝"

        # Get tracked agent stats for this game
        agent_stats_str = ""
        if hasattr(agent, "get_stats"):
            stats = agent.get_stats()
            nodes_after = stats.get("nodes_evaluated", 0)
            nodes_this_game = nodes_after - nodes_before
            depth = stats.get("max_depth_reached", 0)
            if nodes_this_game > 0:
                agent_stats_str = f" (🔢{nodes_this_game:,} nodes, 🔍depth {depth})"

        # Calculate progress and ETA
        progress_pct = (self.game_count / self.total_games) * 100
        elapsed = time.time() - self.start_time

        if self.game_count > 3:  # Wait a few games for stable ETA
            avg_game_time = elapsed / self.game_count
            eta_seconds = avg_game_time * (self.total_games - self.game_count)
            eta_str = (
                f", ETA: {eta_seconds/60:.1f}m"
                if eta_seconds > 60
                else f", ETA: {eta_seconds:.0f}s"
            )
        else:
            eta_str = ""

        # Show progress every 10% or every 5 games, whichever is less frequent
        show_progress = (
            self.game_count % max(1, self.total_games // 10) == 0
            or self.game_count % 5 == 0
            or self.game_count <= 3
            or self.game_count == self.total_games
        )

        if show_progress:
            # Calculate win rate based on total wins so far
            win_rate = (self.wins / self.game_count) * 100
            print(
                f"   Game {self.game_count:3d}/{self.total_games}: {outcome} "
                f"({game_time:.1f}s{agent_stats_str}) | "
                f"Progress: {progress_pct:5.1f}% | "
                f"Win rate: {win_rate:5.1f}%{eta_str}"
            )

        return result


def run_heuristic_vs_random(num_games: int = 100):
    """Run evaluation: Heuristic Agent vs Random Agent."""
    print(f"Running Heuristic vs Random evaluation ({num_games} games)...")

    config = EvaluationConfig(
        num_games=num_games,
        timeout_per_move=3.0,
        verbose=True,
        use_fixed_seeds=True,
        random_seed=42,
    )

    # Create agents
    from agents.heuristic_agent import HeuristicAgent

    test_agent = HeuristicAgent()
    baseline_agent = RandomBaselineAgent(seed=42)

    # Run evaluation
    evaluator = AgentEvaluator(config)
    result = evaluator.evaluate_agent(
        test_agent=test_agent,
        baseline_agent=baseline_agent,
        test_agent_name="HeuristicAgent",
        baseline_agent_name="RandomAgent",
    )

    # Print and save results
    print(format_evaluation_results(result))

    ensure_evaluation_dir()
    save_evaluation_results(result, "evaluation_results/heuristic_vs_random.json")
    print(f"\nResults saved to evaluation_results/heuristic_vs_random.json")

    return result


def run_minimax_evaluation(
    num_games: int = 100,
    difficulty: str = "medium",
    custom_config: Optional[dict] = None,
):
    """Run evaluation: Minimax Agent vs Heuristic Agent."""
    print(f"Running Minimax Agent evaluation ({num_games} games)...")
    print("=" * 60)

    # Create minimax agent with configuration
    from agents.minimax_agent import MinimaxAgent, MinimaxConfig

    if custom_config:
        config = MinimaxConfig(**custom_config)
        print(f"📋 Minimax custom config: {custom_config}")
    else:
        config = MinimaxConfig.create_difficulty_preset(difficulty)
        print(f"🎯 Minimax difficulty: {difficulty}")
        print(f"   ⏱️  Time limit: {config.time_limit}s per move")
        print(f"   🔍 Max depth: {config.max_depth}")
        print(
            f"   🔧 Features: Iterative deepening={config.enable_iterative_deepening}, "
            f"Alpha-beta={config.enable_alpha_beta_pruning}, Move ordering={config.enable_move_ordering}"
        )

    minimax_agent = MinimaxAgent(config=config)

    # Create evaluation config with appropriate timeout based on agent configuration
    eval_config = EvaluationConfig(
        num_games=num_games,
        timeout_per_move=max(
            5.0, config.time_limit + 2.0
        ),  # Give extra time for overhead
        verbose=True,
        use_fixed_seeds=True,
        random_seed=42,
        save_detailed_logs=True,
        confidence_interval=0.95,
        swap_player_positions=True,
    )

    print(f"⚙️  Evaluation timeout per move: {eval_config.timeout_per_move}s")
    print(f"🎲 Baseline: HeuristicAgent")
    print()

    # Run against heuristic baseline only
    baseline_agent = HeuristicBaselineAgent()
    baseline_name = "Heuristic"

    start_time = time.time()

    print(f"🥊 Minimax vs {baseline_name}")
    print("-" * 50)

    # Reset minimax stats
    minimax_agent.reset_stats()

    # Custom evaluation with detailed logging
    result = run_detailed_minimax_evaluation(
        minimax_agent, baseline_agent, baseline_name, config, eval_config
    )

    duration = time.time() - start_time

    # Show results
    print(f"\n📊 Final Results:")
    print(
        f"   🏆 Win rate: {result.test_agent_win_rate:.1%} ({result.test_agent_wins}/{result.games_played})"
    )
    print(f"   📈 Avg score diff: {result.average_score_difference:+.1f}")
    print(f"   ⏱️  Total duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    print(f"   🔢 Avg game duration: {result.average_game_duration:.2f}s")

    if result.is_statistically_significant:
        print(f"   📊 Statistically significant (p={result.p_value:.4f})")
    else:
        print(f"   📊 Not statistically significant (p={result.p_value:.4f})")

    # Save result
    ensure_evaluation_dir()
    filename = f"evaluation_results/minimax_{difficulty}_vs_heuristic.json"
    save_evaluation_results(result, filename)
    print(f"   💾 Saved to {filename}")

    return result


def run_detailed_minimax_evaluation(
    minimax_agent, baseline_agent, baseline_name, config, eval_config
):
    """Run detailed evaluation with game-by-game progress tracking."""
    start_time = time.time()

    # Use generalized progress evaluator, tracking the test agent (minimax)
    progress_evaluator = ProgressTrackingEvaluator(
        eval_config,
        tracked_agent=minimax_agent,
        tracked_agent_role="test",
        tracked_agent_name="MinimaxAgent",
        baseline_name=baseline_name,
    )

    try:
        result = progress_evaluator.evaluate_agent(
            test_agent=minimax_agent,
            baseline_agent=baseline_agent,
            test_agent_name=f"MinimaxAgent(t={config.time_limit}s, d={config.max_depth})",
            baseline_agent_name=baseline_name,
        )

        # Add our custom analysis
        elapsed_time = time.time() - start_time

        print(f"✅ Completed {eval_config.num_games} games in {elapsed_time:.1f}s")
        print(f"⚡ Average time per game: {elapsed_time/eval_config.num_games:.2f}s")

        # Get final minimax statistics
        final_stats = minimax_agent.get_stats()
        if final_stats.get("nodes_evaluated", 0) > 0:
            print(f"🧠 Minimax performance:")
            print(f"   🔢 Total nodes evaluated: {final_stats['nodes_evaluated']:,}")
            print(f"   🔍 Max depth reached: {final_stats['max_depth_reached']}")
            print(
                f"   ⚡ Nodes per game: {final_stats['nodes_evaluated'] / eval_config.num_games:.0f}"
            )
            print(
                f"   🏃 Nodes per second: {final_stats['nodes_evaluated'] / elapsed_time:.0f}"
            )

        return result

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


def run_baseline_comparison(num_games: int = 50):
    """Compare heuristic agent against random agent."""
    print(f"Running baseline comparison: Heuristic vs Random ({num_games} games)...")
    print("=" * 60)

    config = EvaluationConfig(
        num_games=num_games,
        timeout_per_move=2.0,
        verbose=True,
        use_fixed_seeds=True,
        random_seed=42,
    )

    evaluator = AgentEvaluator(config)

    print(f"🧠 Test Agent: HeuristicAgent")
    print(f"🎲 Baseline: RandomAgent")
    print()

    start_time = time.time()

    print(f"🥊 Heuristic vs Random")
    print("-" * 50)

    # Run evaluation
    result = evaluator.evaluate_agent(
        test_agent=HeuristicBaselineAgent(),
        baseline_agent=RandomBaselineAgent(seed=42),
        test_agent_name="HeuristicAgent",
        baseline_agent_name="RandomAgent",
    )

    duration = time.time() - start_time

    print(f"\n📊 Final Results:")
    print(
        f"   🏆 Win rate: {result.test_agent_win_rate:.1%} ({result.test_agent_wins}/{result.games_played})"
    )
    print(f"   📈 Avg score diff: {result.average_score_difference:+.1f}")
    print(f"   ⏱️  Total duration: {duration:.1f}s")

    if result.is_statistically_significant:
        print(f"   📊 Statistically significant (p={result.p_value:.4f})")
    else:
        print(f"   📊 Not statistically significant (p={result.p_value:.4f})")

    # Save results
    ensure_evaluation_dir()
    filename = "evaluation_results/heuristic_vs_random.json"
    save_evaluation_results(result, filename)
    print(f"   💾 Saved to {filename}")

    return result


def run_quick_test(agent_type: str = "heuristic", difficulty: str = "easy"):
    """Run a quick test evaluation."""
    print(f"🚀 Running quick test with {agent_type} agent...")
    print("=" * 40)

    # Create test agent and baseline
    test_agent: Union["HeuristicAgent", "MinimaxAgent"]
    baseline_agent: Union["RandomBaselineAgent", "HeuristicBaselineAgent"]

    if agent_type.lower() == "heuristic":
        from agents.heuristic_agent import HeuristicAgent

        test_agent = HeuristicAgent()
        baseline_agent = RandomBaselineAgent(seed=42)
        print(f"🧠 Test Agent: HeuristicAgent")
        print(f"🎲 Baseline: RandomAgent")
    elif agent_type.lower() == "minimax":
        from agents.minimax_agent import MinimaxAgent, MinimaxConfig

        config = MinimaxConfig.create_difficulty_preset(difficulty)
        test_agent = MinimaxAgent(config=config)
        baseline_agent = HeuristicBaselineAgent()
        print(f"🎯 Test Agent: MinimaxAgent ({difficulty} difficulty)")
        print(f"   ⏱️  Time limit: {config.time_limit}s per move")
        print(f"   🔍 Max depth: {config.max_depth}")
        print(f"🧠 Baseline: HeuristicAgent")
    else:
        print(f"Unknown agent type: {agent_type}")
        return None

    print(f"🎮 Games: 10 (quick test)")
    print()

    start_time = time.time()

    # Quick evaluation
    evaluator = AgentEvaluator()
    result = evaluator.quick_evaluation(
        test_agent=test_agent,
        baseline_agent=baseline_agent,
        num_games=10,
        verbose=True,
    )

    duration = time.time() - start_time

    print()
    print("=" * 40)
    print("🏁 QUICK TEST RESULTS")
    print("=" * 40)
    print(f"⏱️  Duration: {duration:.1f}s")
    print(f"🏆 Win rate: {result.test_agent_win_rate:.1%}")
    print(
        f"📊 Games: {result.test_agent_wins} wins, {result.baseline_agent_wins} losses, {result.draws} draws"
    )
    print(f"📈 Avg score diff: {result.average_score_difference:+.1f}")

    if agent_type.lower() == "minimax" and hasattr(test_agent, "get_stats"):
        stats = test_agent.get_stats()
        if stats.get("nodes_evaluated", 0) > 0:
            print(
                f"🧠 Minimax stats: {stats['nodes_evaluated']:,} nodes, depth {stats['max_depth_reached']}"
            )

    print(result.summary())
    return result


def run_mcts_vs_heuristic(
    num_games: int = 100,
    model_path: Optional[str] = None,
    network_config: Optional[str] = None,
):
    """Run evaluation: OpenSpiel MCTS Agent vs Heuristic Agent."""
    print(f"Running OpenSpiel MCTS vs Heuristic evaluation ({num_games} games)...")
    print("=" * 60)

    from agents.openspiel_agents import OpenSpielMCTSAgent

    # Note: OpenSpiel MCTS uses random rollouts by default and doesn't require neural networks
    # This makes it fast and reliable without needing pre-trained models
    print("Using OpenSpiel MCTS with random rollout evaluator")

    # Create OpenSpiel MCTS agent
    mcts_agent = OpenSpielMCTSAgent(
        num_simulations=400,  # Good balance of strength and speed
        uct_c=1.4,  # Standard UCT exploration
        solve=False,  # Don't use MCTS-Solver for speed
        seed=None,  # Random seed for evaluation variety
    )

    print(
        f"✅ OpenSpiel MCTS agent created with {mcts_agent.num_simulations} simulations"
    )

    # Create evaluation config
    config = EvaluationConfig(
        num_games=num_games,
        timeout_per_move=5.0,  # OpenSpiel MCTS is fast
        verbose=True,
        use_fixed_seeds=True,
        random_seed=42,
        swap_player_positions=True,
        confidence_interval=0.95,
    )

    baseline_agent = HeuristicBaselineAgent()
    baseline_name = "HeuristicAgent"

    # Use detailed evaluation with progress tracking
    result = run_detailed_mcts_evaluation(
        mcts_agent, baseline_agent, baseline_name, config, config
    )

    # Save results
    ensure_evaluation_dir()
    filename = "evaluation_results/openspiel_mcts_vs_heuristic.json"
    save_evaluation_results(result, filename)
    print(f"   💾 Saved to {filename}")

    return result


def run_detailed_mcts_evaluation(
    mcts_agent, baseline_agent, baseline_name, config, eval_config
):
    """Run detailed evaluation with game-by-game progress tracking for OpenSpiel MCTS."""
    start_time = time.time()

    # Use generalized progress evaluator, tracking the test agent (OpenSpiel MCTS)
    progress_evaluator = ProgressTrackingEvaluator(
        eval_config,
        tracked_agent=mcts_agent,
        tracked_agent_role="test",
        tracked_agent_name="OpenSpielMCTSAgent",
        baseline_name=baseline_name,
    )

    try:
        result = progress_evaluator.evaluate_agent(
            test_agent=mcts_agent,
            baseline_agent=baseline_agent,
            test_agent_name="OpenSpielMCTSAgent",
            baseline_agent_name=baseline_name,
        )

        # Add our custom analysis
        elapsed_time = time.time() - start_time

        print(f"✅ Completed {eval_config.num_games} games in {elapsed_time:.1f}s")
        print(f"⚡ Average time per game: {elapsed_time/eval_config.num_games:.2f}s")

        # Get final OpenSpiel MCTS statistics
        print(f"🧠 OpenSpiel MCTS performance:")
        print(f"   🔢 Simulations per move: {mcts_agent.num_simulations:,}")
        print(f"   🎯 UCT exploration constant: {mcts_agent.uct_c}")
        print(f"   ⚡ Average simulations per game: {mcts_agent.num_simulations:.0f}")
        total_simulations = mcts_agent.num_simulations * eval_config.num_games
        print(f"   🏃 Total simulations: {total_simulations:,}")
        print(f"   📊 Simulations per second: {total_simulations / elapsed_time:.0f}")

        return result

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agent Evaluation Framework - 1v1 Agent Comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py heuristic                    # Heuristic vs Random (100 games)
  python run_evaluation.py heuristic --games 50         # Heuristic vs Random (50 games)
  python run_evaluation.py minimax                      # Minimax(medium) vs Heuristic (100 games)
  python run_evaluation.py minimax --difficulty hard    # Minimax(hard) vs Heuristic (100 games)
  python run_evaluation.py minimax --games 50 --difficulty easy  # Minimax(easy) vs Heuristic (50 games)
  python run_evaluation.py mcts --model_path models/checkpoint_100.pth  # MCTS with custom model
  python run_evaluation.py mcts --model_path models/small_model.pth --network_config small  # MCTS with small model
  python run_evaluation.py quick heuristic              # Quick test: Heuristic vs Random (10 games)
  python run_evaluation.py quick minimax --difficulty expert     # Quick test: Minimax(expert) vs Heuristic (10 games)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Evaluation commands")

    # Heuristic evaluation (vs Random)
    heuristic_parser = subparsers.add_parser(
        "heuristic", help="Evaluate heuristic agent vs random baseline"
    )
    heuristic_parser.add_argument(
        "--games", type=int, default=100, help="Number of games to play (default: 100)"
    )

    # Minimax evaluation (vs Heuristic)
    minimax_parser = subparsers.add_parser(
        "minimax", help="Evaluate minimax agent vs heuristic baseline"
    )
    minimax_parser.add_argument(
        "--games", type=int, default=100, help="Number of games to play (default: 100)"
    )
    minimax_parser.add_argument(
        "--difficulty",
        default="medium",
        choices=["easy", "medium", "hard", "expert"],
        help="Minimax difficulty (default: medium)",
    )

    # MCTS evaluation (vs Heuristic)
    mcts_parser = subparsers.add_parser(
        "mcts", help="Evaluate MCTS agent vs heuristic baseline"
    )
    mcts_parser.add_argument(
        "--games", type=int, default=100, help="Number of games to play (default: 100)"
    )
    mcts_parser.add_argument(
        "--model_path",
        type=str,
        help="Path to a pre-trained neural network for MCTS",
    )
    mcts_parser.add_argument(
        "--network_config",
        type=str,
        choices=["small", "medium", "large", "deep"],
        help="Neural network configuration to use (default: inferred from model path or 'medium')",
    )

    # Quick test command
    quick_parser = subparsers.add_parser("quick", help="Quick 10-game evaluation test")
    quick_parser.add_argument(
        "agent",
        choices=["heuristic", "minimax"],
        help="Agent type to test quickly",
    )
    quick_parser.add_argument(
        "--difficulty",
        default="easy",
        choices=["easy", "medium", "hard", "expert"],
        help="Minimax difficulty for quick test (default: easy)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "heuristic":
            run_heuristic_vs_random(args.games)

        elif args.command == "minimax":
            run_minimax_evaluation(args.games, difficulty=args.difficulty)

        elif args.command == "mcts":
            run_mcts_vs_heuristic(args.games, args.model_path, args.network_config)

        elif args.command == "quick":
            run_quick_test(args.agent, args.difficulty)

        else:
            print(f"Unknown command: {args.command}")
            return 1

        print("\nEvaluation completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        return 1

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
