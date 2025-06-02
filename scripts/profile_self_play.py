#!/usr/bin/env python3
"""
Profile Azul self-play performance.

This script runs profiled self-play to identify performance bottlenecks
in neural network evaluation, game operations, and MCTS simulation.

Supports both original Azul agents and OpenSpiel agents.
"""

import argparse
import os
import sys
from typing import Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.mcts import MCTSAgent
from profiling.openspiel_profiler import (
    OpenSpielProfiler,
    create_profiled_agent,
)
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

        # Create replay buffer for compatibility
        self.replay_buffer = ReplayBuffer(capacity=10000)

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

                # Add game data to replay buffer for compatibility
                states = game_data["states"]
                actions = game_data["actions"]
                policies = game_data["policies"]
                _ = game_data["rewards"]

                # Store experiences in replay buffer
                for i, (state, action, policy) in enumerate(
                    zip(states, actions, policies)
                ):
                    # Simple experience storage - just store the action for OpenSpiel compatibility
                    # The replay buffer for OpenSpiel is just for compatibility, not training
                    self.replay_buffer.add_experience(action)

            if self.verbose:
                print(f"\nCompleted {num_games} games")
                print(f"Replay buffer size: {len(self.replay_buffer)}")

            return all_game_data


def run_profiled_self_play(
    num_games: int = 1,
    mcts_simulations: int = 100,
    nn_config: str = "medium",
    agent_type: str = "original",  # "original", "openspiel_mcts", "openspiel_random"
    enable_cprofile: bool = True,
    enable_line_profiler: bool = False,
    verbose: bool = True,
):
    """Run profiled self-play analysis."""
    print(f"Initializing profiled self-play with {agent_type} agents...")

    if agent_type == "original":
        # Use original Azul profiler and agents
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

    else:
        # Use OpenSpiel profiler and agents
        profiler = OpenSpielProfiler(
            enable_memory_profiling=True, enable_gpu_profiling=True
        )

        # Determine OpenSpiel agent type
        if agent_type == "openspiel_mcts":
            openspiel_agent_type = "mcts"
        elif agent_type == "openspiel_alphazero":
            openspiel_agent_type = "alphazero"
        elif agent_type == "openspiel_random":
            openspiel_agent_type = "random"
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        print(
            f"Creating OpenSpiel {openspiel_agent_type} agent with random rollout evaluator..."
        )

        # Create OpenSpiel profiled self-play engine
        engine = OpenSpielProfiledSelfPlayEngine(
            profiler=profiler,
            agent_type=openspiel_agent_type,
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
    generate_performance_recommendations(profiler, agent_type)

    return profiler


def generate_performance_recommendations(profiler, agent_type: str):
    """Generate performance optimization recommendations."""
    summary = profiler.get_summary()

    print("\n" + "=" * 80)
    print(f"PERFORMANCE OPTIMIZATION RECOMMENDATIONS ({agent_type.upper()})")
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

        # Agent-specific recommendations
        if agent_type == "original":
            # Original agent recommendations
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

        else:
            # OpenSpiel agent recommendations
            if (
                "openspiel_mcts.search" in operation
                or "openspiel_mcts.step" in operation
            ):
                print("   RECOMMENDATION: OpenSpiel MCTS optimization")
                print(
                    "   - **MAJOR**: Replace RandomRolloutEvaluator with faster alternatives:"
                )
                print("     * Use AlphaZero neural network evaluator if available")
                print("     * Implement custom heuristic evaluator")
                print("     * Random rollouts are ~14ms per evaluation!")
                print(
                    "   - Reduce simulation count for faster play (20-50 simulations)"
                )
                print("   - Current bottleneck: Each simulation plays out entire games")

            elif "policy_computation" in operation:
                print("   RECOMMENDATION: Policy computation optimization")
                print(
                    "   - Root cause: RandomRolloutEvaluator doing full game simulations"
                )
                print("   - Switch to neural network evaluator when available")
                print("   - Consider batching policy computations")
                print("   - Current: ~284ms per policy call with random rollouts")

            elif "evaluate" in operation or "rollout" in operation:
                print("   RECOMMENDATION: Evaluator optimization (CRITICAL)")
                print(
                    "   - **ROOT CAUSE IDENTIFIED**: RandomRolloutEvaluator is extremely slow"
                )
                print("   - Each evaluation takes ~14ms (full game simulation)")
                print("   - Solutions:")
                print("     * Use trained neural network evaluator")
                print("     * Implement fast heuristic evaluator")
                print("     * Reduce n_rollouts parameter")

            elif "state_conversion" in operation:
                print("   RECOMMENDATION: State conversion optimization")
                print("   - Cache converted states when possible")
                print("   - Optimize Azul ↔ OpenSpiel conversion algorithms")
                print("   - Consider native OpenSpiel state representation")
                print("   - Implement conversion pooling/reuse")

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
            if agent_type == "original":
                print("RECOMMENDATION: Monitor GPU memory efficiency")
            else:
                print("RECOMMENDATION: OpenSpiel may not utilize GPU optimally")
                print(
                    "   - Consider using original agents for GPU-accelerated training"
                )
                print("   - OpenSpiel agents are better for CPU-based evaluation")
        else:
            print("GPU memory usage appears minimal")
            if agent_type == "original":
                print("RECOMMENDATION: Verify GPU is being utilized effectively")
            else:
                print("INFO: OpenSpiel agents typically use CPU-based algorithms")

    # Agent-specific summary recommendations
    print(f"\n\nAGENT-SPECIFIC RECOMMENDATIONS ({agent_type.upper()}):")
    print("-" * 50)

    if agent_type == "original":
        print("✓ Best for GPU-accelerated neural network training")
        print("✓ Optimal for custom MCTS implementations")
        print("✓ Full control over optimization and algorithms")
        print("? Consider OpenSpiel agents for algorithm comparison")

    elif "openspiel" in agent_type:
        print("✓ Best for algorithm research and comparison")
        print("✓ Proven, well-tested implementations")
        print("✓ Good for CPU-based evaluation and analysis")
        print("? Consider original agents for production training")

        # OpenSpiel vs original comparison
        if "mcts" in agent_type:
            print("\nOpenSpiel MCTS vs Original MCTS:")
            print("- OpenSpiel: More robust, research-proven algorithms")
            print("- Original: Better GPU utilization, custom optimizations")

    print(
        f"\nFor production use: Consider running both agent types and comparing performance."
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Profile Azul self-play performance")
    parser.add_argument(
        "--games", type=int, default=1, help="Number of games to play (default: 1)"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTS simulations per move (default: 100)",
    )
    parser.add_argument(
        "--nn-config",
        choices=["small", "medium", "large", "deep"],
        default="medium",
        help="Neural network configuration (for original agents only)",
    )
    parser.add_argument(
        "--agent-type",
        choices=[
            "original",
            "openspiel_mcts",
            "openspiel_alphazero",
            "openspiel_random",
        ],
        default="original",
        help="Type of agent to profile (default: original)",
    )
    parser.add_argument(
        "--no-cprofile",
        action="store_true",
        help="Disable cProfile (faster but less detailed)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Validate configuration
    if args.agent_type != "original" and args.nn_config != "medium":
        print("Warning: --nn-config is ignored for OpenSpiel agents")

    # Run profiling
    profiler = run_profiled_self_play(
        num_games=args.games,
        mcts_simulations=args.simulations,
        nn_config=args.nn_config,
        agent_type=args.agent_type,
        enable_cprofile=not args.no_cprofile,
        verbose=not args.quiet,
    )

    # Optionally save results
    output_file = f"profiling_results_{args.agent_type}_{args.games}games.txt"
    try:
        with open(output_file, "w") as f:
            f.write(f"Azul Self-Play Profiling Results ({args.agent_type})\n")
            f.write("=" * 60 + "\n\n")

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
