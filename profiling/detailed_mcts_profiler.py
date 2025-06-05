#!/usr/bin/env python3
"""
Detailed MCTS profiler to identify simulation bottlenecks.

This script digs deeper into OpenSpiel MCTS to find where the 6ms+ per simulation is spent.
Now updated to use a real neural network evaluator instead of random rollouts.
FIXED: Reuses evaluator instead of recreating it for each test.
"""

import cProfile
import io
import os
import pstats
import sys
import time

# Add parent directory to path so we can import from game
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model

from agents.openspiel_agents import OpenSpielMCTSAgent
from game.azul_openspiel import AzulGame, AzulState
from game.game_state import GameState

# Global variables to store the shared evaluator (PERFORMANCE FIX!)
_shared_nn_evaluator = None
_shared_azul_game = None


def create_neural_network_evaluator():
    """Create a neural network evaluator for more realistic profiling."""
    global _shared_nn_evaluator, _shared_azul_game

    # Return cached evaluator if already created (MAJOR PERFORMANCE FIX!)
    if _shared_nn_evaluator is not None and _shared_azul_game is not None:
        print("♻️  Reusing existing neural network evaluator (performance optimization)")
        return _shared_nn_evaluator, _shared_azul_game

    print("🔧 Creating neural network evaluator (first time only)...")

    # Create game instance
    azul_game = AzulGame({"deterministic_mode": False})

    try:
        # Create a neural network model matching the user's exact configuration
        # ResNet with 256 width, 6 depth - this should be much slower
        model = az_model.Model.build_model(
            "resnet",  # ResNet instead of MLP - much more expensive!
            azul_game.observation_tensor_shape(),  # [16, 16, 4]
            azul_game.num_distinct_actions(),  # 180 actions
            nn_width=256,  # 256 width (vs 128 before)
            nn_depth=6,  # 6 depth (vs 4 before)
            weight_decay=1e-4,
            learning_rate=1e-3,
            path=None,  # No checkpoint path for default model
        )

        if model is not None:
            # Create AlphaZero evaluator with the neural network
            evaluator = az_evaluator.AlphaZeroEvaluator(
                game=azul_game,
                model=model,
            )
            print("✅ Created ResNet neural network evaluator successfully")
            print(f"   Model: ResNet, Width: 256, Depth: 6")
            print(f"   Observation shape: {azul_game.observation_tensor_shape()}")
            print(f"   Action space size: {azul_game.num_distinct_actions()}")

            # Cache for future use (PERFORMANCE FIX!)
            _shared_nn_evaluator = evaluator
            _shared_azul_game = azul_game

            return evaluator, azul_game
        else:
            raise Exception("Model creation returned None")

    except Exception as e:
        print(f"⚠️  Failed to create ResNet neural network evaluator: {e}")
        print("Falling back to random rollout evaluator for comparison")

        # Cache the fallback too
        fallback_evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)
        _shared_nn_evaluator = fallback_evaluator
        _shared_azul_game = azul_game

        return fallback_evaluator, azul_game


def profile_mcts_components():
    """Profile individual MCTS components to identify bottlenecks."""
    print("=" * 60)
    print("DETAILED MCTS COMPONENT PROFILING (WITH NEURAL NETWORK)")
    print("=" * 60)

    # Setup
    game_state = GameState(num_players=2, seed=42)

    # Create neural network evaluator ONCE (PERFORMANCE FIX!)
    evaluator, azul_game = create_neural_network_evaluator()

    openspiel_state = AzulState(azul_game, game_state.num_players)
    openspiel_state._game_state = game_state.copy()

    # Time evaluator operations (this should be much slower with NN)
    print("\n1. Testing Neural Network Evaluator performance:")
    times = []
    for i in range(20):  # Fewer iterations since NN evaluation is slow
        test_state = openspiel_state.clone()
        start_time = time.perf_counter()
        evaluator.evaluate(test_state)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)

    avg_time = sum(times) / len(times)
    print(f"   Average evaluator time: {avg_time:.3f}ms")
    print(f"   Min: {min(times):.3f}ms, Max: {max(times):.3f}ms")

    # Compare with random rollout for reference
    print("\n1b. Comparison with RandomRolloutEvaluator:")
    random_evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)
    times = []
    for i in range(100):
        test_state = openspiel_state.clone()
        start_time = time.perf_counter()
        random_evaluator.evaluate(test_state)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)

    avg_time_random = sum(times) / len(times)
    print(f"   Average random evaluator time: {avg_time_random:.3f}ms")
    print(
        f"   Neural network is {avg_time/avg_time_random:.1f}x slower than random rollouts"
    )

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


def profile_neural_network_mcts():
    """Profile MCTS with neural network evaluator to identify the real bottleneck."""
    print("\n" + "=" * 60)
    print("NEURAL NETWORK MCTS PERFORMANCE ANALYSIS (HIGH SIMULATION COUNTS)")
    print("=" * 60)

    game_state = GameState(num_players=2, seed=42)

    # Create neural network evaluator ONCE (PERFORMANCE FIX!)
    evaluator, azul_game = create_neural_network_evaluator()

    # Test different simulation counts with emphasis on higher counts
    simulation_counts = [1, 2, 5, 10, 20, 50, 100, 200, 400, 800]

    for sim_count in simulation_counts:
        print(f"\n--- Testing {sim_count} simulation(s) with Neural Network ---")

        # Create MCTS bot with SHARED neural network evaluator (PERFORMANCE FIX!)
        mcts_bot = mcts.MCTSBot(
            game=azul_game,
            uct_c=1.4,
            max_simulations=sim_count,
            evaluator=evaluator,  # Reuse the same evaluator!
            solve=False,
            verbose=False,
        )

        openspiel_state = AzulState(azul_game, game_state.num_players)
        openspiel_state._game_state = game_state.copy()

        # Fewer iterations for very high simulation counts to save time
        num_iterations = 5 if sim_count <= 50 else 3 if sim_count <= 200 else 2

        # Time multiple selections
        times = []
        for i in range(num_iterations):
            test_state = openspiel_state.clone()
            try:
                start_time = time.perf_counter()
                mcts_bot.step_with_policy(test_state)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            except Exception as e:
                print(f"   Error with {sim_count} simulations: {e}")
                break

        if times:
            avg_time = sum(times) / len(times)
            time_per_sim = avg_time / sim_count
            print(f"   Average total time: {avg_time:.3f}ms")
            print(f"   Time per simulation: {time_per_sim:.3f}ms")

            # Calculate efficiency metrics
            if sim_count >= 10:
                efficiency = (
                    10 / sim_count
                ) * 100  # Efficiency compared to 10 simulations
                print(f"   Scaling efficiency: {efficiency:.1f}% (vs 10 simulations)")


def profile_rollout_performance():
    """Profile neural network evaluation vs random rollout performance."""
    print("\n" + "=" * 60)
    print("NN EVALUATION VS RANDOM ROLLOUT COMPARISON")
    print("=" * 60)

    game_state = GameState(num_players=2, seed=42)

    # Create both types of evaluators (reuse NN evaluator)
    nn_evaluator, azul_game = create_neural_network_evaluator()
    random_evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)

    openspiel_state = AzulState(azul_game, game_state.num_players)
    openspiel_state._game_state = game_state.copy()

    # Time neural network evaluation
    print("\nTesting Neural Network evaluation performance:")
    nn_times = []
    for i in range(10):  # Fewer iterations for NN
        test_state = openspiel_state.clone()
        start_time = time.perf_counter()
        nn_evaluator.evaluate(test_state)
        end_time = time.perf_counter()
        nn_times.append((end_time - start_time) * 1000)

    avg_nn_time = sum(nn_times) / len(nn_times)
    print(f"   Average NN evaluation time: {avg_nn_time:.3f}ms")

    # Time random rollout evaluation
    print("\nTesting Random Rollout evaluation performance:")
    random_times = []
    for i in range(100):  # More iterations for random rollouts
        test_state = openspiel_state.clone()
        start_time = time.perf_counter()
        random_evaluator.evaluate(test_state)
        end_time = time.perf_counter()
        random_times.append((end_time - start_time) * 1000)

    avg_random_time = sum(random_times) / len(random_times)
    print(f"   Average random evaluation time: {avg_random_time:.3f}ms")

    print(f"\n📊 Performance Comparison:")
    print(f"   Neural Network: {avg_nn_time:.3f}ms per evaluation")
    print(f"   Random Rollout: {avg_random_time:.3f}ms per evaluation")
    print(f"   NN is {avg_nn_time/avg_random_time:.1f}x slower than random rollouts")


def profile_with_reduced_simulations():
    """Profile MCTS with different simulation counts to understand scaling."""
    print("\n" + "=" * 60)
    print("MCTS SCALING ANALYSIS (Neural Network vs Random Rollouts)")
    print("=" * 60)

    game_state = GameState(num_players=2, seed=42)

    # Test wider range of simulation counts
    simulation_counts = [1, 2, 5, 10, 20, 50, 100, 200, 400]

    # Create evaluator ONCE and reuse (MAJOR PERFORMANCE FIX!)
    nn_evaluator, azul_game = create_neural_network_evaluator()

    for sim_count in simulation_counts:
        print(f"\n--- Testing {sim_count} simulation(s) ---")

        # Test with neural network evaluator
        print("  Neural Network Evaluator:")
        try:
            # PERFORMANCE FIX: Reuse the same evaluator instead of creating new one!
            agent = OpenSpielMCTSAgent(
                num_simulations=sim_count,
                uct_c=1.4,
                solve=False,
                evaluator=nn_evaluator,  # Reuse shared evaluator!
            )

            # Adjust number of test iterations based on simulation count
            num_iterations = 3 if sim_count <= 50 else 2 if sim_count <= 200 else 1

            times = []
            for i in range(num_iterations):
                try:
                    start_time = time.perf_counter()
                    agent.select_action(game_state, deterministic=True)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)
                except Exception as e:
                    print(f"    Error: {e}")
                    break

            if times:
                avg_time = sum(times) / len(times)
                time_per_sim = avg_time / sim_count
                print(f"    Average total time: {avg_time:.3f}ms")
                print(f"    Time per simulation: {time_per_sim:.3f}ms")

                # Show scaling compared to base case
                if sim_count == 10:
                    base_time_per_sim = time_per_sim
                elif sim_count > 10 and "base_time_per_sim" in locals():
                    scaling_factor = base_time_per_sim / time_per_sim
                    print(
                        f"    Scaling factor vs 10 sims: {scaling_factor:.1f}x faster per sim"
                    )

        except Exception as e:
            print(f"    NN evaluator failed: {e}")

        # Test with random rollout evaluator for comparison (but only up to 100 sims to save time)
        if sim_count <= 100:
            print("  Random Rollout Evaluator:")
            try:
                agent = OpenSpielMCTSAgent(
                    num_simulations=sim_count, uct_c=1.4, solve=False
                )

                times = []
                for i in range(
                    5 if sim_count <= 20 else 3
                ):  # Fewer iterations for higher counts
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
                    print(f"    Average total time: {avg_time:.3f}ms")
                    print(f"    Time per simulation: {time_per_sim:.3f}ms")
            except Exception as e:
                print(f"    Random evaluator failed: {e}")
        else:
            print("  Random Rollout Evaluator: Skipped (saving time)")


def profile_mcts_tree_operations():
    """Profile MCTS tree building and traversal with neural network evaluator."""
    print("\n" + "=" * 60)
    print("MCTS TREE OPERATIONS PROFILING (NEURAL NETWORK - HIGHER COUNTS)")
    print("=" * 60)

    game_state = GameState(num_players=2, seed=42)

    # Create neural network evaluator ONCE (PERFORMANCE FIX!)
    evaluator, azul_game = create_neural_network_evaluator()

    # Test with multiple simulation counts to see scaling
    simulation_counts = [10, 50, 100, 200]

    for sim_count in simulation_counts:
        print(f"\n--- Analyzing MCTS tree building with {sim_count} simulations ---")

        # Create MCTS bot with SHARED neural network evaluator (PERFORMANCE FIX!)
        mcts_bot = mcts.MCTSBot(
            game=azul_game,
            uct_c=1.4,
            max_simulations=sim_count,
            evaluator=evaluator,  # Reuse the same evaluator!
            solve=False,
            verbose=False,
        )

        openspiel_state = AzulState(azul_game, game_state.num_players)
        openspiel_state._game_state = game_state.copy()

        # Time the search process
        start_time = time.perf_counter()
        try:
            result = mcts_bot.step_with_policy(openspiel_state)
            end_time = time.perf_counter()

            total_time = (end_time - start_time) * 1000
            time_per_sim = total_time / sim_count
            print(f"   Total search time: {total_time:.3f}ms")
            print(f"   Time per simulation: {time_per_sim:.3f}ms")

            if isinstance(result, tuple) and len(result) == 2:
                policy, action = result
                print(f"   Policy length: {len(policy)}")
                print(f"   Selected action: {action}")

        except Exception as e:
            print(f"   MCTS search failed: {e}")


def profile_with_cprofile_focused():
    """Focused cProfile analysis on neural network MCTS operations with higher simulation counts."""
    print("\n" + "=" * 60)
    print("FOCUSED CPROFILE ANALYSIS (NEURAL NETWORK - 100 SIMULATIONS)")
    print("=" * 60)

    game_state = GameState(num_players=2, seed=42)

    # Create neural network evaluator ONCE (PERFORMANCE FIX!)
    evaluator, azul_game = create_neural_network_evaluator()

    # Use higher simulation count to see amortization effects
    agent = OpenSpielMCTSAgent(
        num_simulations=100,  # Higher count to see scaling effects
        evaluator=evaluator,  # Reuse shared evaluator!
    )

    # Profile single MCTS selection with higher simulation count
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        # Just one iteration to see where time is spent at higher simulation counts
        agent.select_action(game_state, deterministic=True)
    except Exception as e:
        print(f"Profiling failed: {e}")
        return

    profiler.disable()

    # Analyze results - focus on most time-consuming functions
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats("cumulative")
    ps.print_stats(20)  # Top 20 functions

    print("\nTop functions by cumulative time (100 simulations):")
    output = s.getvalue()
    print(output)

    # Also print top functions by internal time
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.sort_stats("tottime")
    ps2.print_stats(15)  # Top 15 by internal time

    print("\nTop functions by internal time (100 simulations):")
    print(s2.getvalue())


def main():
    """Main profiling routine."""
    print("DETAILED MCTS SIMULATION PROFILER (NEURAL NETWORK EDITION)")
    print("=========================================================")
    print("🔧 PERFORMANCE FIX: Reusing neural network evaluator across tests")
    print("=========================================================")

    try:
        # Test neural network MCTS performance
        profile_with_reduced_simulations()

        # Profile individual components with neural network
        profile_mcts_components()

        # Profile neural network vs random rollout
        profile_rollout_performance()

        # Profile NN MCTS scaling
        profile_neural_network_mcts()

        # Profile tree operations with neural network
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
    print("1. Neural network evaluation time vs random rollouts")
    print("2. Time per simulation scaling with NN evaluator")
    print("3. MCTS overhead with neural network")
    print("4. cProfile hotspots in NN evaluation")
    print("5. Memory usage patterns with neural networks")
    print("\n🔧 PERFORMANCE FIX APPLIED:")
    print("   - Reused neural network evaluator across all tests")
    print("   - Eliminated expensive model recreation overhead")


if __name__ == "__main__":
    main()
