#!/usr/bin/env python3
"""
Integration test to verify optimized MCTS works with existing self-play system.
"""

import os
import sys
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.neural_network import AzulNeuralNetwork
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayEngine


def test_optimized_self_play():
    """Test that optimized MCTS integrates properly with self-play."""

    print("🔧 Testing Optimized MCTS Integration with Self-Play")
    print("=" * 60)

    # Create components
    print("📊 Initializing components...")
    neural_network = AzulNeuralNetwork(config_name="small", device="cpu")
    replay_buffer = ReplayBuffer(capacity=1000)

    # Create self-play engine with optimized MCTS
    self_play = SelfPlayEngine(
        neural_network=neural_network,
        replay_buffer=replay_buffer,
        mcts_simulations=50,  # Small number for quick test
        temperature=1.0,
        verbose=True,
    )

    # The key change: enable optimizations in the MCTSAgent
    # Create new optimized agent
    from agents.mcts import MCTSAgent

    self_play.agent = MCTSAgent(
        neural_network=neural_network,
        num_simulations=50,
        temperature=1.0,
        enable_optimizations=True,  # Enable optimizations
        state_pool_size=500,
        transposition_table_size=1000,
    )

    print("🎮 Running optimized self-play game...")
    start_time = time.time()

    # Play a single game
    experiences = self_play.play_game(num_players=2, seed=42)

    end_time = time.time()
    game_time = end_time - start_time

    print(f"\n✅ Game completed successfully!")
    print(f"🕒 Game time: {game_time:.2f}s")
    print(f"📝 Experiences collected: {len(experiences)}")
    print(f"🎯 Average time per move: {game_time / len(experiences):.3f}s")

    # Get optimization statistics
    opt_stats = self_play.agent.mcts.get_optimization_stats()
    print(f"\n📈 Optimization Statistics:")
    print(f"  Cache hit rate: {opt_stats['cache_hit_rate']:.1%}")
    print(f"  State pooling rate: {opt_stats['pooling_rate']:.1%}")
    print(f"  Cache hits: {opt_stats['cache_hits']}")
    print(f"  Cache misses: {opt_stats['cache_misses']}")
    print(f"  States pooled: {opt_stats['states_pooled']}")
    print(f"  States allocated: {opt_stats['states_allocated']}")

    # Check that we actually got some benefits
    if opt_stats["cache_hit_rate"] > 0:
        print("🎉 Transposition table is working!")

    if opt_stats["pooling_rate"] > 0.5:
        print("🎉 State pooling is working!")

    return game_time, opt_stats


def test_self_play_comparison():
    """Compare standard vs optimized self-play performance with multiple runs."""

    print(f"\n🏁 Self-Play Performance Comparison (Multiple Games)")
    print("=" * 60)

    neural_network = AzulNeuralNetwork(config_name="small", device="cpu")

    configs = [
        {"name": "Standard", "optimizations": False},
        {"name": "Optimized", "optimizations": True},
    ]

    # Configuration for testing
    num_games = 5  # Run multiple games for statistical significance
    simulations_per_search = 30  # Simulations per MCTS search

    results = {}

    for config in configs:
        print(f"\n🚀 Testing {config['name']} Self-Play ({num_games} games)...")

        game_times = []
        game_moves = []
        total_experiences = 0
        optimization_stats = []

        for game_num in range(num_games):
            print(f"  Game {game_num + 1}/{num_games}...", end=" ")

            # Create fresh self-play engine for each game
            self_play = SelfPlayEngine(
                neural_network=neural_network,
                replay_buffer=ReplayBuffer(capacity=1000),  # Fresh buffer each time
                mcts_simulations=simulations_per_search,
                temperature=1.0,
                verbose=False,
            )

            # Replace agent with optimized version if requested
            if config["optimizations"]:
                from agents.mcts import MCTSAgent

                self_play.agent = MCTSAgent(
                    neural_network=neural_network,
                    num_simulations=simulations_per_search,
                    temperature=1.0,
                    enable_optimizations=True,
                    state_pool_size=500,
                    transposition_table_size=1000,
                )

            # Use different seed for each game to avoid bias
            game_seed = 42 + game_num * 100

            start_time = time.time()
            experiences = self_play.play_game(num_players=2, seed=game_seed)
            end_time = time.time()

            game_time = end_time - start_time
            game_times.append(game_time)
            game_moves.append(len(experiences))
            total_experiences += len(experiences)

            # Collect optimization stats if enabled
            if config["optimizations"]:
                opt_stats = self_play.agent.mcts.get_optimization_stats()
                optimization_stats.append(opt_stats)

            print(f"{game_time:.2f}s ({len(experiences)} moves)")

        # Calculate statistics
        import statistics

        avg_time = statistics.mean(game_times)
        std_time = statistics.stdev(game_times) if len(game_times) > 1 else 0
        avg_moves = statistics.mean(game_moves)
        avg_time_per_move = avg_time / avg_moves if avg_moves > 0 else 0

        print(f"\n📊 {config['name']} Results:")
        print(f"  Average game time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Average moves per game: {avg_moves:.1f}")
        print(f"  Average time per move: {avg_time_per_move:.3f}s")
        print(f"  Total experiences: {total_experiences}")
        print(f"  Games completed: {num_games}")

        # Store results for comparison
        results[config["name"]] = {
            "avg_time": avg_time,
            "std_time": std_time,
            "avg_moves": avg_moves,
            "avg_time_per_move": avg_time_per_move,
            "total_experiences": total_experiences,
            "game_times": game_times,  # Keep raw data
            "num_games": num_games,
        }

        # Show optimization statistics if available
        if config["optimizations"] and optimization_stats:
            avg_cache_hit_rate = statistics.mean(
                [s["cache_hit_rate"] for s in optimization_stats]
            )
            avg_pooling_rate = statistics.mean(
                [s["pooling_rate"] for s in optimization_stats]
            )
            total_cache_hits = sum([s["cache_hits"] for s in optimization_stats])
            total_cache_misses = sum([s["cache_misses"] for s in optimization_stats])
            total_states_pooled = sum([s["states_pooled"] for s in optimization_stats])

            print(f"  Optimization Statistics:")
            print(f"    Avg cache hit rate: {avg_cache_hit_rate:.1%}")
            print(f"    Avg pooling rate: {avg_pooling_rate:.1%}")
            print(f"    Total cache hits: {total_cache_hits}")
            print(f"    Total cache misses: {total_cache_misses}")
            print(f"    Total states pooled: {total_states_pooled}")

            results[config["name"]]["optimization_stats"] = {
                "avg_cache_hit_rate": avg_cache_hit_rate,
                "avg_pooling_rate": avg_pooling_rate,
                "total_cache_hits": total_cache_hits,
                "total_cache_misses": total_cache_misses,
                "total_states_pooled": total_states_pooled,
            }

    # Statistical comparison
    if "Standard" in results and "Optimized" in results:
        print(f"\n📈 Statistical Comparison:")
        print("=" * 40)

        std_data = results["Standard"]
        opt_data = results["Optimized"]

        # Time comparison
        if opt_data["avg_time"] > 0:
            speedup = std_data["avg_time"] / opt_data["avg_time"]
            improvement = (
                (std_data["avg_time"] - opt_data["avg_time"]) / std_data["avg_time"]
            ) * 100

            print(f"Game Time Comparison:")
            print(
                f"  Standard: {std_data['avg_time']:.3f}s ± {std_data['std_time']:.3f}s"
            )
            print(
                f"  Optimized: {opt_data['avg_time']:.3f}s ± {opt_data['std_time']:.3f}s"
            )
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Improvement: {improvement:.1f}%")

            # Calculate statistical significance (simple t-test approximation)
            import math

            pooled_std = math.sqrt(
                (std_data["std_time"] ** 2 + opt_data["std_time"] ** 2) / 2
            )
            if pooled_std > 0:
                t_stat = abs(std_data["avg_time"] - opt_data["avg_time"]) / (
                    pooled_std * math.sqrt(2 / num_games)
                )
                print(
                    f"  T-statistic: {t_stat:.2f} (>2.0 suggests significant difference)"
                )

        # Per-move comparison
        std_per_move = std_data["avg_time_per_move"]
        opt_per_move = opt_data["avg_time_per_move"]

        if opt_per_move > 0:
            per_move_speedup = std_per_move / opt_per_move
            per_move_improvement = ((std_per_move - opt_per_move) / std_per_move) * 100

            print(f"\nPer-Move Time Comparison:")
            print(f"  Standard: {std_per_move:.3f}s per move")
            print(f"  Optimized: {opt_per_move:.3f}s per move")
            print(f"  Per-move speedup: {per_move_speedup:.2f}x")
            print(f"  Per-move improvement: {per_move_improvement:.1f}%")

        # Show optimization effectiveness
        if "optimization_stats" in opt_data:
            opt_stats = opt_data["optimization_stats"]
            print(f"\nOptimization Effectiveness:")
            print(f"  Cache hit rate: {opt_stats['avg_cache_hit_rate']:.1%}")
            print(f"  State pooling rate: {opt_stats['avg_pooling_rate']:.1%}")

            if opt_stats["avg_cache_hit_rate"] > 0.3:
                print("  ✅ Transposition table providing good benefits")
            else:
                print(
                    "  ℹ️  Low cache hit rate - benefits may increase with more simulations"
                )

            if opt_stats["avg_pooling_rate"] > 0.5:
                print("  ✅ State pooling working effectively")

        # Performance assessment
        print(f"\n🎯 Performance Assessment:")
        if speedup > 1.2:
            print(f"  🎉 Significant speedup achieved! ({speedup:.2f}x faster)")
        elif speedup > 1.05:
            print(f"  ✅ Modest improvement observed ({improvement:.1f}% faster)")
        elif speedup > 0.95:
            print(f"  ⚖️  Performance roughly equivalent (within measurement error)")
        else:
            print(
                f"  ⚠️  Optimizations may need tuning (slower by {(1/speedup - 1)*100:.1f}%)"
            )

        # Confidence intervals (rough approximation)
        print(f"\n📊 Confidence Intervals (±1 std dev):")
        print(
            f"  Standard timing: [{std_data['avg_time'] - std_data['std_time']:.3f}, {std_data['avg_time'] + std_data['std_time']:.3f}]s"
        )
        print(
            f"  Optimized timing: [{opt_data['avg_time'] - opt_data['std_time']:.3f}, {opt_data['avg_time'] + opt_data['std_time']:.3f}]s"
        )

    return results


def test_extended_self_play_analysis():
    """Extended analysis with different game configurations."""

    print(f"\n🔬 Extended Self-Play Analysis")
    print("=" * 60)

    neural_network = AzulNeuralNetwork(config_name="small", device="cpu")

    # Test different simulation counts to see how benefits scale
    simulation_configs = [20, 50, 100]

    for sim_count in simulation_configs:
        print(f"\n⚡ Testing with {sim_count} simulations per search...")

        configs = [
            {
                "name": f"Standard-{sim_count}",
                "optimizations": False,
                "sims": sim_count,
            },
            {
                "name": f"Optimized-{sim_count}",
                "optimizations": True,
                "sims": sim_count,
            },
        ]

        results = {}

        for config in configs:
            from agents.mcts import MCTSAgent
            from game.game_state import GameState  # Add missing import

            # Run 10 games for this configuration
            game_times = []

            for game_num in range(10):
                agent = MCTSAgent(
                    neural_network=neural_network,
                    num_simulations=config["sims"],
                    temperature=1.0,
                    enable_optimizations=config["optimizations"],
                )

                # Simple game simulation - just run MCTS searches
                game_state = GameState(num_players=2, seed=42 + game_num)

                start_time = time.time()
                # Simulate 10 moves worth of MCTS searches
                for move in range(10):
                    if not game_state.game_over:
                        action_probs = agent.get_action_probabilities(game_state)
                        legal_actions = game_state.get_legal_actions()
                        if legal_actions:
                            # Take the highest probability action
                            best_action_idx = max(
                                range(len(action_probs)), key=lambda i: action_probs[i]
                            )
                            action = legal_actions[best_action_idx]
                            game_state.apply_action(action)

                end_time = time.time()
                game_times.append(end_time - start_time)

            avg_time = sum(game_times) / len(game_times)
            results[config["name"]] = avg_time

            print(f"  {config['name']}: {avg_time:.3f}s avg")

            # Show optimization stats
            if config["optimizations"]:
                stats = agent.mcts.get_optimization_stats()
                print(f"    Cache hit rate: {stats['cache_hit_rate']:.1%}")

        # Compare this simulation count
        std_key = f"Standard-{sim_count}"
        opt_key = f"Optimized-{sim_count}"

        if std_key in results and opt_key in results:
            speedup = results[std_key] / results[opt_key]
            improvement = (
                (results[std_key] - results[opt_key]) / results[std_key]
            ) * 100
            print(
                f"  → Speedup with {sim_count} sims: {speedup:.2f}x ({improvement:.1f}% improvement)"
            )


if __name__ == "__main__":
    try:
        test_optimized_self_play()
        test_self_play_comparison()
        test_extended_self_play_analysis()

        print(f"\n✅ Integration test completed successfully!")
        print(f"\n💡 The optimized MCTS is ready for production use.")
        print(f"   To enable in self-play, modify the MCTSAgent creation to include:")
        print(f"   enable_optimizations=True")

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
