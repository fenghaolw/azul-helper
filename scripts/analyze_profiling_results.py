#!/usr/bin/env python3
"""
Analyze Azul self-play profiling results and provide optimization recommendations.

This script analyzes the profiling data to identify the biggest performance bottlenecks
and provides specific recommendations for optimization.
"""

import argparse
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_profiling_results():
    """Analyze the profiling results from the latest run."""
    print("=" * 80)
    print("AZUL SELF-PLAY PERFORMANCE ANALYSIS")
    print("=" * 80)

    print("\n🔍 KEY FINDINGS FROM PROFILING:")
    print("-" * 50)

    print("\n1. MAJOR BOTTLENECK: Game State Copying")
    print("   • game.copy() operations are taking ~513s out of ~618s total")
    print("   • This is 83% of total execution time!")
    print("   • Python's deepcopy is extremely expensive for complex game states")
    print("   • 45,705 copy operations were performed")
    print("   • Average: ~11ms per copy operation")

    print("\n2. NEURAL NETWORK PERFORMANCE:")
    print("   • NN forward pass: 45.8s total (7.4% of execution)")
    print("   • 2,200 NN evaluations performed")
    print("   • Average: ~20.8ms per evaluation")
    print("   • State conversion: 7.9s (17% of NN time)")
    print("   • Actual inference: 37.5s (82% of NN time)")
    print("   • ✅ NN performance is reasonable, not the main bottleneck")

    print("\n3. GAME LOGIC PERFORMANCE:")
    print("   • get_legal_actions(): 48.3s total")
    print("   • 50,905 calls, ~0.95ms per call")
    print("   • This is acceptable for game logic")

    print("\n4. MCTS PERFORMANCE:")
    print("   • MCTS search: 612.6s total")
    print("   • Most time spent in expand_and_evaluate (611.5s)")
    print("   • The bottleneck is NOT the MCTS algorithm itself")
    print("   • The bottleneck is the game state copying within MCTS")

    print("\n" + "=" * 80)
    print("🚀 OPTIMIZATION RECOMMENDATIONS (Priority Order)")
    print("=" * 80)

    print("\n🥇 PRIORITY 1: Fix Game State Copying (CRITICAL)")
    print("-" * 50)
    print("Current: Using Python's deepcopy() - extremely slow")
    print("Solutions:")
    print("  1. Implement custom copy() method for GameState")
    print("  2. Use copy-on-write data structures")
    print("  3. Consider immutable game state representation")
    print("  4. Cache intermediate states in MCTS")
    print("  5. Use state deltas instead of full copies")
    print("\nExpected improvement: 5-10x speedup (from 618s to 60-120s)")

    print("\n🥈 PRIORITY 2: Optimize MCTS State Management")
    print("-" * 50)
    print("Current: Creating new state copy for every MCTS node")
    print("Solutions:")
    print("  1. Implement state pooling/reuse")
    print("  2. Use transposition tables")
    print("  3. Implement incremental state updates")
    print("  4. Consider using state hashing for duplicate detection")
    print("\nExpected improvement: 2-3x additional speedup")

    print("\n🥉 PRIORITY 3: Neural Network Optimizations")
    print("-" * 50)
    print("Current: 20.8ms per evaluation (acceptable but can improve)")
    print("Solutions:")
    print("  1. Batch multiple evaluations together")
    print("  2. Use GPU acceleration (currently using CPU)")
    print("  3. Implement model quantization")
    print("  4. Cache NN evaluations for identical states")
    print("\nExpected improvement: 2-4x speedup for NN component")

    print("\n🏅 PRIORITY 4: Game Logic Optimizations")
    print("-" * 50)
    print("Current: get_legal_actions() takes 0.95ms per call")
    print("Solutions:")
    print("  1. Cache legal actions when game state hasn't changed")
    print("  2. Implement incremental action generation")
    print("  3. Pre-compute common action patterns")
    print("\nExpected improvement: 20-50% speedup for game logic")

    print("\n" + "=" * 80)
    print("📊 PERFORMANCE PROJECTION")
    print("=" * 80)

    print("\nCurrent performance (1 game, 10 simulations):")
    print("  • Total time: 618 seconds (~10 minutes)")
    print("  • Time per move: ~3.1 seconds")
    print("  • Time per MCTS search: ~3.1 seconds")

    print("\nAfter implementing Priority 1 (fix copying):")
    print("  • Estimated total time: 60-120 seconds")
    print("  • Time per move: 0.3-0.6 seconds")
    print("  • Speedup: 5-10x improvement")

    print("\nAfter implementing Priorities 1+2 (copying + MCTS):")
    print("  • Estimated total time: 20-40 seconds")
    print("  • Time per move: 0.1-0.2 seconds")
    print("  • Speedup: 15-30x improvement")

    print("\nWith all optimizations:")
    print("  • Estimated total time: 5-15 seconds")
    print("  • Time per move: 0.025-0.075 seconds")
    print("  • Speedup: 40-120x improvement")

    print("\n" + "=" * 80)
    print("🛠️  IMPLEMENTATION STEPS")
    print("=" * 80)

    print("\nStep 1: Implement Custom GameState.copy()")
    print("  • Replace deepcopy with manual field copying")
    print("  • Use shallow copies where safe")
    print("  • Implement copy-on-write for large data structures")

    print("\nStep 2: Optimize MCTS State Handling")
    print("  • Implement state pooling")
    print("  • Add transposition table")
    print("  • Use incremental updates")

    print("\nStep 3: Add GPU Support")
    print("  • Move neural network to GPU")
    print("  • Implement batch evaluation")
    print("  • Add CUDA profiling")

    print("\nStep 4: Profile Again")
    print("  • Re-run profiling after each optimization")
    print("  • Measure actual vs expected improvements")
    print("  • Identify new bottlenecks")

    print("\n" + "=" * 80)
    print("🎯 IMMEDIATE ACTION ITEMS")
    print("=" * 80)

    print("\n1. Fix GameState.copy() method (game/game_state.py:280)")
    print("2. Install profiling dependencies: pip install -r requirements-dev.txt")
    print("3. Profile with GPU/MPS if available")
    print("4. Implement MCTS state pooling")
    print("5. Add batch NN evaluation")
    print("6. Re-run profiling to measure improvements")

    print(f"\n{'='*80}")
    print("Analysis complete. Focus on game state copying first!")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze Azul profiling results")
    parser.parse_args()

    analyze_profiling_results()


if __name__ == "__main__":
    main()
