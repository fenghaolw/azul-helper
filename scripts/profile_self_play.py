#!/usr/bin/env python3
"""
Profile Azul self-play performance.

This script runs profiled self-play to identify performance bottlenecks
in neural network evaluation, game operations, and MCTS simulation.
"""

import argparse
import os
import sys
from typing import Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.mcts import MCTSAgent
from profiling.performance_profiler import (
    AzulProfiler,
    create_profiled_game_state,
    create_profiled_mcts,
    create_profiled_neural_network,
)
from training.neural_network import AzulNeuralNetwork
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayEngine


class ProfiledSelfPlayEngine(SelfPlayEngine):
    """Self-play engine with integrated profiling."""

    def __init__(self, profiler: AzulProfiler, *args, **kwargs):
        self.profiler = profiler
        super().__init__(*args, **kwargs)

        # Replace the agent with profiled version
        self._setup_profiled_agent()

    def _setup_profiled_agent(self):
        """Replace MCTS agent with profiled version."""
        # Create profiled neural network
        ProfiledNeuralNetwork = create_profiled_neural_network(self.profiler)
        profiled_nn = ProfiledNeuralNetwork(self.neural_network)

        # Create profiled MCTS
        ProfiledMCTS = create_profiled_mcts(self.profiler)

        # Replace the agent's MCTS with profiled version
        original_mcts = self.agent.mcts
        self.agent.mcts = ProfiledMCTS(
            neural_network=profiled_nn,
            c_puct=original_mcts.c_puct,
            num_simulations=original_mcts.num_simulations,
            temperature=original_mcts.temperature,
            dirichlet_alpha=getattr(original_mcts, "dirichlet_alpha", 0.3),
            dirichlet_epsilon=getattr(original_mcts, "dirichlet_epsilon", 0.25),
        )

        # Update the neural network reference
        self.neural_network = profiled_nn

    def play_game(self, num_players: int = 2, seed: Optional[int] = None):
        """Play a game with profiling."""
        with self.profiler.time_operation("self_play.full_game"):
            # Use profiled game state
            ProfiledGameState = create_profiled_game_state(self.profiler)

            # Temporarily replace GameState in the method
            import game.game_state

            original_game_state = game.game_state.GameState
            game.game_state.GameState = ProfiledGameState  # type: ignore[misc]

            try:
                return super().play_game(num_players, seed)
            finally:
                # Restore original GameState
                game.game_state.GameState = original_game_state  # type: ignore[misc]


def run_profiled_self_play(
    num_games: int = 1,
    mcts_simulations: int = 800,
    nn_config: str = "medium",
    enable_cprofile: bool = True,
    enable_line_profiler: bool = False,
    verbose: bool = True,
):
    """Run profiled self-play analysis."""
    print("Initializing profiled self-play...")

    # Create profiler
    profiler = AzulProfiler(enable_memory_profiling=True, enable_gpu_profiling=True)

    # Create neural network
    print(f"Loading neural network with {nn_config} configuration...")

    # Try to load existing checkpoint first
    checkpoint_path = f"models/azul_model_{nn_config}.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        neural_network = AzulNeuralNetwork(
            config_name=nn_config, model_path=checkpoint_path
        )
    else:
        print("No checkpoint found, creating randomly initialized network")
        neural_network = AzulNeuralNetwork(config_name=nn_config)

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=10000)

    # Create profiled self-play engine
    engine = ProfiledSelfPlayEngine(
        profiler=profiler,
        neural_network=neural_network,
        replay_buffer=replay_buffer,
        mcts_simulations=mcts_simulations,
        verbose=verbose,
    )

    print(
        f"Starting profiled self-play: {num_games} games with {mcts_simulations} MCTS simulations"
    )

    # Start cProfile if enabled
    if enable_cprofile:
        print("Starting cProfile...")
        profiler.start_cprofile()

    try:
        # Run self-play games with overall timing
        with profiler.time_operation("total_self_play"):
            engine.play_games(num_games=num_games)

    except KeyboardInterrupt:
        print("\nProfiling interrupted by user")
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback

        traceback.print_exc()

    # Stop cProfile and get results
    cprofile_results = ""
    if enable_cprofile:
        print("Stopping cProfile...")
        cprofile_results = profiler.stop_cprofile()

    # Print comprehensive results
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)

    profiler.print_summary()

    if cprofile_results:
        print("\n" + "=" * 80)
        print("CPROFILE HOTSPOTS (Top 30 functions)")
        print("=" * 80)
        print(cprofile_results)

    # Generate recommendations
    generate_performance_recommendations(profiler)

    return profiler


def generate_performance_recommendations(profiler: AzulProfiler):
    """Generate performance optimization recommendations."""
    summary = profiler.get_summary()

    print("\n" + "=" * 80)
    print("PERFORMANCE OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)

    # Analyze timing data for recommendations
    timing_data = {
        name: data
        for name, data in summary.items()
        if isinstance(data, dict) and "total_time" in data
    }

    if not timing_data:
        print("No timing data available for recommendations")
        return

    # Sort by total time to identify bottlenecks
    sorted_timings = sorted(
        timing_data.items(), key=lambda x: x[1]["total_time"], reverse=True
    )

    total_time = sum(data["total_time"] for _, data in sorted_timings)

    print(f"\nTotal execution time: {total_time:.2f}s")
    print("\nTop bottlenecks and recommendations:")
    print("-" * 60)

    for i, (operation, data) in enumerate(sorted_timings[:5]):
        percentage = (data["total_time"] / total_time) * 100
        avg_ms = data["avg_time"] * 1000

        print(f"\n{i+1}. {operation}")
        print(f"   Time: {data['total_time']:.2f}s ({percentage:.1f}% of total)")
        print(f"   Avg per call: {avg_ms:.2f}ms")
        print(f"   Calls: {data['call_count']}")

        # Specific recommendations based on operation type
        if "nn.forward_pass" in operation:
            print("   RECOMMENDATION: Neural network optimization")
            print("   - Consider model quantization or pruning")
            print("   - Batch inference if possible")
            print("   - Verify GPU utilization")
            print("   - Consider smaller model architecture")

        elif "game.get_legal_actions" in operation:
            print("   RECOMMENDATION: Game logic optimization")
            print("   - Cache legal actions when possible")
            print("   - Optimize action generation algorithms")
            print("   - Consider pre-computing common patterns")

        elif "game.apply_action" in operation:
            print("   RECOMMENDATION: Game state mutation optimization")
            print("   - Minimize object creation in apply_action")
            print("   - Use in-place operations where safe")
            print("   - Consider copy-on-write strategies")

        elif "game.copy" in operation:
            print("   RECOMMENDATION: State copying optimization")
            print("   - Implement shallow copying where possible")
            print("   - Use copy-on-write data structures")
            print("   - Consider immutable state representations")

        elif "mcts" in operation:
            print("   RECOMMENDATION: MCTS optimization")
            print("   - Reduce simulation count if acceptable")
            print("   - Optimize tree traversal algorithms")
            print("   - Consider parallel MCTS")
            print("   - Implement progressive bias/widening")

    # GPU recommendations
    gpu_stats = summary.get("gpu_stats", {})
    if gpu_stats:
        print(f"\n\nGPU Analysis:")
        print("-" * 20)

        total_gpu_memory = 0
        for op_name, stats in gpu_stats.items():
            if "gpu_memory_delta_mb" in stats:
                total_gpu_memory += abs(stats["gpu_memory_delta_mb"])

        if total_gpu_memory > 0:
            print(f"Total GPU memory usage: {total_gpu_memory:.2f} MB")
            print("RECOMMENDATION: Monitor GPU memory efficiency")
        else:
            print("GPU memory usage appears minimal")
            print("RECOMMENDATION: Verify GPU is being utilized effectively")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Profile Azul self-play performance")
    parser.add_argument(
        "--games", type=int, default=1, help="Number of games to play (default: 1)"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="MCTS simulations per move (default: 800)",
    )
    parser.add_argument(
        "--nn-config",
        choices=["small", "medium", "large", "deep"],
        default="medium",
        help="Neural network configuration",
    )
    parser.add_argument(
        "--no-cprofile",
        action="store_true",
        help="Disable cProfile (faster but less detailed)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Run profiling
    profiler = run_profiled_self_play(
        num_games=args.games,
        mcts_simulations=args.simulations,
        nn_config=args.nn_config,
        enable_cprofile=not args.no_cprofile,
        verbose=not args.quiet,
    )

    # Optionally save results
    output_file = f"profiling_results_{args.nn_config}_{args.games}games.txt"
    try:
        with open(output_file, "w") as f:
            f.write("Azul Self-Play Profiling Results\n")
            f.write("=" * 50 + "\n\n")

            summary = profiler.get_summary()
            for name, data in summary.items():
                if isinstance(data, dict) and "total_time" in data:
                    f.write(f"{name}:\n")
                    f.write(f"  Total time: {data['total_time']:.3f}s\n")
                    f.write(f"  Average time: {data['avg_time']*1000:.2f}ms\n")
                    f.write(f"  Call count: {data['call_count']}\n")
                    f.write(f"  Calls/second: {data['times_per_second']:.1f}\n\n")

        print(f"\nResults saved to {output_file}")

    except Exception as e:
        print(f"Could not save results to file: {e}")


if __name__ == "__main__":
    main()
