#!/usr/bin/env python3
"""
Detailed MCTS profiler to identify simulation bottlenecks.

This script digs deeper into OpenSpiel MCTS to find where the 6ms+ per simulation is spent.
"""

import cProfile
import io
import os
import pstats
import sys
import time

# Add parent directory to path so we can import from game
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.openspiel_agents import OpenSpielMCTSAgent
from game.azul_openspiel import AzulGame, AzulState
from game.game_state import GameState
from open_spiel.python.algorithms import mcts


def profile_mcts_components():
    """Profile individual MCTS components to identify bottlenecks."""
    print("=" * 60)
    print("DETAILED MCTS COMPONENT PROFILING")
    print("=" * 60)

    # Setup
    game_state = GameState(num_players=2, seed=42)
    azul_game = AzulGame({"deterministic_mode": False})
    openspiel_state = AzulState(azul_game, game_state.num_players)
    openspiel_state._game_state = game_state.copy()

    # Create components
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)

    # Time evaluator operations
    print("\n1. Testing RandomRolloutEvaluator performance:")
    times = []
    for i in range(100):
        test_state = openspiel_state.clone()
        start_time = time.perf_counter()
        evaluator.evaluate(test_state)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)

    avg_time = sum(times) / len(times)
    print(f"   Average evaluator time: {avg_time:.3f}ms")
    print(f"   Min: {min(times):.3f}ms, Max: {max(times):.3f}ms")

    # Time state cloning (used heavily in MCTS)
    print("\n2. Testing state cloning performance:")
    times = []
    for i in range(100):
        start_time = time.perf_counter()
        openspiel_state.clone()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)

    avg_time = sum(times) / len(times)
    print(f"   Average clone time: {avg_time:.3f}ms")
    print(f"   Min: {min(times):.3f}ms, Max: {max(times):.3f}ms")

    # Time legal actions retrieval
    print("\n3. Testing legal actions retrieval:")
    times = []
    for i in range(100):
        start_time = time.perf_counter()
        legal_actions = openspiel_state.legal_actions()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)

    avg_time = sum(times) / len(times)
    print(f"   Average legal_actions time: {avg_time:.3f}ms")
    print(f"   Number of legal actions: {len(legal_actions)}")

    # Time action application
    print("\n4. Testing action application:")
    legal_actions = openspiel_state.legal_actions()
    if legal_actions:
        times = []
        for i in range(50):
            test_state = openspiel_state.clone()
            start_time = time.perf_counter()
            test_state.apply_action(legal_actions[0])
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        avg_time = sum(times) / len(times)
        print(f"   Average apply_action time: {avg_time:.3f}ms")


def profile_rollout_performance():
    """Profile random rollout performance."""
    print("\n" + "=" * 60)
    print("RANDOM ROLLOUT PERFORMANCE ANALYSIS")
    print("=" * 60)

    game_state = GameState(num_players=2, seed=42)
    azul_game = AzulGame({"deterministic_mode": False})
    openspiel_state = AzulState(azul_game, game_state.num_players)
    openspiel_state._game_state = game_state.copy()

    mcts.RandomRolloutEvaluator(n_rollouts=1)

    # Time complete rollouts
    print("\nTesting complete rollout performance:")
    times = []
    rollout_lengths = []

    for i in range(20):  # Fewer iterations since rollouts can be long
        test_state = openspiel_state.clone()

        start_time = time.perf_counter()

        # Simulate a complete random rollout manually to track steps
        moves = 0
        while not test_state.is_terminal() and moves < 200:  # Safety limit
            legal_actions = test_state.legal_actions()
            if not legal_actions:
                break

            # Choose random action
            import random

            action = random.choice(legal_actions)
            test_state.apply_action(action)
            moves += 1

        end_time = time.perf_counter()

        times.append((end_time - start_time) * 1000)
        rollout_lengths.append(moves)

    if times:
        avg_time = sum(times) / len(times)
        avg_length = sum(rollout_lengths) / len(rollout_lengths)
        print(f"   Average rollout time: {avg_time:.3f}ms")
        print(f"   Average rollout length: {avg_length:.1f} moves")
        print(f"   Time per move in rollout: {avg_time/avg_length:.3f}ms")


def profile_with_reduced_simulations():
    """Profile MCTS with very few simulations to isolate overhead."""
    print("\n" + "=" * 60)
    print("MCTS OVERHEAD ANALYSIS (Low Simulation Counts)")
    print("=" * 60)

    game_state = GameState(num_players=2, seed=42)

    simulation_counts = [1, 2, 5, 10, 20]

    for sim_count in simulation_counts:
        print(f"\n--- Testing {sim_count} simulation(s) ---")

        agent = OpenSpielMCTSAgent(num_simulations=sim_count, uct_c=1.4, solve=False)

        # Warm up
        try:
            agent.select_action(game_state, deterministic=True)
        except ValueError:
            # Skip if MCTS fails with very low simulation counts
            print(f"   Skipped - MCTS failed with {sim_count} simulations")
            continue

        # Time multiple selections
        times = []
        for i in range(10):
            try:
                start_time = time.perf_counter()
                agent.select_action(game_state, deterministic=True)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            except ValueError:
                break

        if times:
            avg_time = sum(times) / len(times)
            time_per_sim = avg_time / sim_count
            overhead = avg_time - (sim_count * 0.1)  # Assume 0.1ms per ideal simulation

            print(f"   Average total time: {avg_time:.3f}ms")
            print(f"   Time per simulation: {time_per_sim:.3f}ms")
            print(f"   Estimated overhead: {overhead:.3f}ms")


def profile_mcts_tree_operations():
    """Profile MCTS tree building and traversal."""
    print("\n" + "=" * 60)
    print("MCTS TREE OPERATIONS PROFILING")
    print("=" * 60)

    game_state = GameState(num_players=2, seed=42)
    azul_game = AzulGame({"deterministic_mode": False})

    # Create MCTS bot with verbose mode if available
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)
    mcts_bot = mcts.MCTSBot(
        game=azul_game,
        uct_c=1.4,
        max_simulations=10,  # Small number for detailed analysis
        evaluator=evaluator,
        solve=False,
        verbose=False,
    )

    openspiel_state = AzulState(azul_game, game_state.num_players)
    openspiel_state._game_state = game_state.copy()

    print("\nAnalyzing MCTS tree building with 10 simulations:")

    # Time the search process
    start_time = time.perf_counter()
    try:
        result = mcts_bot.step_with_policy(openspiel_state)
        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000
        print(f"   Total search time: {total_time:.3f}ms")
        print(f"   Time per simulation: {total_time/10:.3f}ms")

        if isinstance(result, tuple) and len(result) == 2:
            policy, action = result
            print(f"   Policy length: {len(policy)}")
            print(f"   Selected action: {action}")

    except Exception as e:
        print(f"   MCTS search failed: {e}")


def profile_with_cprofile_focused():
    """Focused cProfile analysis on MCTS operations."""
    print("\n" + "=" * 60)
    print("FOCUSED CPROFILE ANALYSIS")
    print("=" * 60)

    game_state = GameState(num_players=2, seed=42)
    agent = OpenSpielMCTSAgent(
        num_simulations=50
    )  # Moderate number for clear profiling

    # Profile just a few MCTS selections
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        for i in range(3):  # Just a few iterations
            agent.select_action(game_state, deterministic=True)
    except Exception as e:
        print(f"Profiling failed: {e}")
        return

    profiler.disable()

    # Analyze results - focus on most time-consuming functions
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats("cumulative")
    ps.print_stats(15)  # Top 15 functions

    print("\nTop functions by cumulative time:")
    output = s.getvalue()
    print(output)

    # Also print top functions by internal time
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.sort_stats("tottime")
    ps2.print_stats(10)  # Top 10 by internal time

    print("\nTop functions by internal time:")
    print(s2.getvalue())


def main():
    """Main profiling routine."""
    print("DETAILED MCTS SIMULATION PROFILER")
    print("=================================")

    try:
        # Test basic MCTS performance first
        profile_with_reduced_simulations()

        # Profile individual components
        profile_mcts_components()

        # Profile rollout performance
        profile_rollout_performance()

        # Profile tree operations
        profile_mcts_tree_operations()

        # Detailed cProfile analysis
        profile_with_cprofile_focused()

    except Exception as e:
        print(f"Profiling error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    print("\nKey metrics to analyze:")
    print("1. Time per simulation vs simulation count")
    print("2. Rollout time and length")
    print("3. State operation overhead")
    print("4. cProfile hotspots")


if __name__ == "__main__":
    main()
