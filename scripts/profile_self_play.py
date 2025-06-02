#!/usr/bin/env python3
"""
Comprehensive profiling script for Azul self-play using OpenSpiel agents.

This script profiles different aspects of gameplay to identify performance bottlenecks:
- OpenSpiel MCTS agent performance
- Game state operations
- Action selection timing
- Overall gameplay performance

Usage:
    python scripts/profile_self_play.py --num-games 5 --simulations 200 --agent mcts
    python scripts/profile_self_play.py --num-games 10 --agent random --enable-cprofile
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling.openspiel_profiler import (
    OpenSpielProfiler,
    create_profiled_agent,
)


class OpenSpielProfiledSelfPlayEngine:
    """Self-play engine with integrated profiling for OpenSpiel agents."""

    def __init__(
        self,
        profiler: OpenSpielProfiler,
        agent_type: str = "mcts",
        mcts_simulations: int = 100,
        verbose: bool = True,
        **agent_kwargs,
    ):
        self.profiler = profiler
        self.agent_type = agent_type
        self.verbose = verbose

        # Create profiled OpenSpiel agent
        self.agent = create_profiled_agent(
            agent_type,
            profiler,
            num_simulations=mcts_simulations,
            **agent_kwargs,
        )

    def play_game(self, num_players: int = 2, seed: Optional[int] = None):
        """Play a single game using OpenSpiel agents."""
        with self.profiler.time_operation("openspiel_self_play.full_game"):
            from game.game_state import GameState

            # Create game state
            game_state = GameState(num_players=num_players, seed=seed)

            # Track game data
            game_data = {"states": [], "actions": [], "rewards": [], "policies": []}

            move_count = 0
            max_moves = 200  # Prevent infinite games

            while not game_state.game_over and move_count < max_moves:
                current_player = game_state.current_player

                # Profile action selection
                with self.profiler.time_operation(
                    "openspiel_self_play.action_selection"
                ):
                    action = self.agent.select_action(game_state, deterministic=False)

                # Profile probability computation
                with self.profiler.time_operation(
                    "openspiel_self_play.policy_computation"
                ):
                    policy = self.agent.get_action_probabilities(game_state)

                # Store game data
                game_data["states"].append(game_state.copy())
                game_data["actions"].append(action)
                game_data["policies"].append(policy)

                if self.verbose and move_count % 10 == 0:
                    print(f"  Move {move_count}: Player {current_player} -> {action}")

                # Apply action
                with self.profiler.time_operation("openspiel_self_play.apply_action"):
                    game_state.apply_action(action)

                move_count += 1

            # Get final scores
            if game_state.game_over:
                scores = game_state.get_scores()
                winner = max(range(len(scores)), key=lambda i: scores[i])

                # Assign rewards (1 for winner, 0 for others)
                final_rewards = [
                    1.0 if i == winner else 0.0 for i in range(num_players)
                ]
                game_data["rewards"] = final_rewards

                if self.verbose:
                    print(f"  Game finished after {move_count} moves")
                    print(f"  Final scores: {scores}")
                    print(f"  Winner: Player {winner}")
            else:
                # Game didn't finish within move limit
                final_rewards = [0.0] * num_players
                game_data["rewards"] = final_rewards

                if self.verbose:
                    print(f"  Game terminated after {max_moves} moves (max limit)")

            return game_data

    def play_games(self, num_games: int = 1):
        """Play multiple games and collect statistics."""
        with self.profiler.time_operation("openspiel_self_play.multiple_games"):
            all_game_data = []

            for game_idx in range(num_games):
                if self.verbose:
                    print(f"\nPlaying game {game_idx + 1}/{num_games}...")

                game_data = self.play_game(num_players=2, seed=42 + game_idx)
                all_game_data.append(game_data)

            if self.verbose:
                print(f"\nCompleted {num_games} games")

            return all_game_data


def run_profiled_self_play(
    num_games: int = 1,
    mcts_simulations: int = 100,
    agent_type: str = "mcts",  # "mcts", "random", "alphazero"
    enable_cprofile: bool = True,
    verbose: bool = True,
):
    """
    Run profiled self-play games using OpenSpiel agents.

    Args:
        num_games: Number of games to play
        mcts_simulations: Number of MCTS simulations (if using MCTS agent)
        agent_type: Type of agent ("mcts", "random", "alphazero")
        enable_cprofile: Whether to enable detailed Python profiling
        verbose: Whether to print detailed progress
    """
    print("=" * 80)
    print("AZUL SELF-PLAY PROFILING WITH OPENSPIEL AGENTS")
    print("=" * 80)
    print(f"Agent type: {agent_type}")
    print(f"Number of games: {num_games}")
    if agent_type == "mcts":
        print(f"MCTS simulations: {mcts_simulations}")
    print(f"cProfile enabled: {enable_cprofile}")
    print(f"Verbose output: {verbose}")
    print("-" * 80)

    # Create profiler
    profiler = OpenSpielProfiler(
        enable_memory_profiling=True, enable_gpu_profiling=True
    )

    # Start detailed profiling if requested
    if enable_cprofile:
        profiler.start_cprofile()

    # Create self-play engine
    engine = OpenSpielProfiledSelfPlayEngine(
        profiler=profiler,
        agent_type=agent_type,
        mcts_simulations=mcts_simulations,
        verbose=verbose,
    )

    # Run games
    start_time = time.time()

    try:
        game_data = engine.play_games(num_games=num_games)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n{'='*80}")
        print("PROFILING RESULTS")
        print(f"{'='*80}")
        print(f"Total runtime: {total_time:.2f} seconds")
        print(f"Average time per game: {total_time/num_games:.2f} seconds")

        # Print profiling summary
        profiler.print_summary()

        # Print detailed cProfile results if enabled
        if enable_cprofile:
            print(f"\n{'='*80}")
            print("DETAILED PYTHON PROFILING (cProfile)")
            print(f"{'='*80}")
            cprofile_results = profiler.stop_cprofile()
            print(cprofile_results)

        # Generate recommendations
        print(f"\n{'='*80}")
        print("PERFORMANCE RECOMMENDATIONS")
        print(f"{'='*80}")
        generate_performance_recommendations(profiler, agent_type)

        return game_data, profiler

    except KeyboardInterrupt:
        print("\nProfiling interrupted by user")
        if enable_cprofile:
            print("\nPartial cProfile results:")
            print(profiler.stop_cprofile())
        return None, profiler

    except Exception as e:
        print(f"\nError during profiling: {e}")
        if enable_cprofile:
            print("\nPartial cProfile results:")
            print(profiler.stop_cprofile())
        raise


def generate_performance_recommendations(profiler, agent_type: str):
    """Generate performance recommendations based on profiling results."""
    summary = profiler.get_summary()

    print("\nBased on profiling results, here are performance recommendations:")
    print("-" * 60)

    # Check if we have timing data
    timing_ops = [
        name
        for name in summary.keys()
        if isinstance(summary.get(name), dict) and "total_time" in summary.get(name, {})
    ]

    if not timing_ops:
        print("⚠️  No timing data available - profiling may not have run correctly")
        return

    # Sort operations by total time
    timing_data = [(name, summary[name]) for name in timing_ops]
    timing_data.sort(key=lambda x: x[1]["total_time"], reverse=True)

    # Top bottlenecks
    print(f"🔍 TOP PERFORMANCE BOTTLENECKS:")
    for i, (name, data) in enumerate(timing_data[:3]):
        percentage = (
            data["total_time"] / sum(d[1]["total_time"] for d in timing_data)
        ) * 100
        print(
            f"   {i+1}. {name}: {data['total_time']:.3f}s ({percentage:.1f}% of total)"
        )

    print(f"\n💡 RECOMMENDATIONS FOR {agent_type.upper()} AGENT:")

    if agent_type == "mcts":
        # MCTS-specific recommendations
        mcts_ops = [name for name, _ in timing_data if "mcts" in name.lower()]
        if mcts_ops:
            print("   • Consider reducing MCTS simulations if action selection is slow")
            print("   • Profile individual MCTS components to identify bottlenecks")

        action_selection_time = next(
            (
                data["avg_time"]
                for name, data in timing_data
                if "action_selection" in name
            ),
            None,
        )
        if action_selection_time and action_selection_time > 0.1:
            print(
                f"   • Action selection is slow ({action_selection_time*1000:.1f}ms avg) - consider optimizing MCTS"
            )

    elif agent_type == "random":
        # Random agent should be very fast
        action_selection_time = next(
            (
                data["avg_time"]
                for name, data in timing_data
                if "action_selection" in name
            ),
            None,
        )
        if action_selection_time and action_selection_time > 0.001:
            print(
                f"   • Random agent action selection seems slow ({action_selection_time*1000:.1f}ms) - investigate"
            )

    # General recommendations
    game_ops = [
        name for name, _ in timing_data if "game." in name or "apply_action" in name
    ]
    if game_ops:
        total_game_time = sum(
            data["total_time"] for name, data in timing_data if name in game_ops
        )
        total_time = sum(data["total_time"] for _, data in timing_data)
        if total_game_time / total_time > 0.3:
            print(
                "   • Game operations take significant time - consider optimizing GameState"
            )

    # Memory recommendations
    if "memory_usage" in summary:
        memory_data = summary["memory_usage"]
        high_memory_ops = [name for name, mb in memory_data.items() if mb > 100]
        if high_memory_ops:
            print("   • High memory usage detected - consider memory optimization")

    # GPU recommendations
    if "gpu_stats" in summary:
        gpu_data = summary["gpu_stats"]
        if gpu_data and any("gpu_used" in data for data in gpu_data.values()):
            print(
                "   • GPU usage detected - ensure TensorFlow is using GPU efficiently"
            )

    print(f"\n📊 OVERALL PERFORMANCE:")
    total_time = sum(data["total_time"] for _, data in timing_data)
    total_calls = sum(data["call_count"] for _, data in timing_data)
    print(f"   • Total profiled time: {total_time:.3f}s")
    print(f"   • Total function calls: {total_calls:,}")
    print(f"   • Average call time: {total_time/total_calls*1000:.2f}ms")


def main():
    """Main entry point for the profiling script."""
    parser = argparse.ArgumentParser(
        description="Profile Azul self-play performance using OpenSpiel agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile 5 games with MCTS agent (200 simulations)
  python scripts/profile_self_play.py --num-games 5 --simulations 200 --agent mcts

  # Profile random agent with detailed Python profiling
  python scripts/profile_self_play.py --num-games 10 --agent random --enable-cprofile

  # Quick performance test
  python scripts/profile_self_play.py --num-games 1 --agent mcts --simulations 50
        """,
    )

    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to play for profiling (default: 1)",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="Number of MCTS simulations per move (default: 100)",
    )

    parser.add_argument(
        "--agent",
        type=str,
        choices=["mcts", "random", "alphazero"],
        default="mcts",
        help="Type of agent to profile (default: mcts)",
    )

    parser.add_argument(
        "--enable-cprofile",
        action="store_true",
        help="Enable detailed Python profiling with cProfile",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_games < 1:
        print("Error: Number of games must be at least 1")
        sys.exit(1)

    if args.simulations < 1:
        print("Error: Number of simulations must be at least 1")
        sys.exit(1)

    # Run profiling
    try:
        result = run_profiled_self_play(
            num_games=args.num_games,
            mcts_simulations=args.simulations,
            agent_type=args.agent,
            enable_cprofile=args.enable_cprofile,
            verbose=not args.quiet,
        )

        if result[0] is not None:
            print(f"\n✅ Profiling completed successfully!")
            print(f"   Games played: {args.num_games}")
            print(f"   Agent type: {args.agent}")
            if args.agent == "mcts":
                print(f"   MCTS simulations: {args.simulations}")
        else:
            print(f"\n❌ Profiling was interrupted or failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during profiling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
