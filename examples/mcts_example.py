"""
OpenSpiel MCTS Example for Azul

This example demonstrates how to use OpenSpiel's robust MCTS implementation
with the Azul game. OpenSpiel MCTS is faster, more reliable, and doesn't
require neural networks for strong gameplay.
"""

import os
import sys
import time
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.openspiel_agents import OpenSpielMCTSAgent
from game.game_state import GameState


def main():
    """Demonstrate OpenSpiel MCTS usage with Azul."""
    print("=" * 60)
    print("OpenSpiel MCTS Example for Azul")
    print("=" * 60)

    # Create a game
    game = GameState(num_players=2, seed=42)
    print(f"Created Azul game with {game.num_players} players")

    # Create OpenSpiel MCTS agents with different configurations
    print("\n1. Creating OpenSpiel MCTS Agents")
    print("-" * 40)

    # Fast agent for quick games
    fast_agent = OpenSpielMCTSAgent(num_simulations=100, uct_c=1.4, solve=False)
    print(f"✅ Fast agent: {fast_agent.num_simulations} simulations")

    # Strong agent for competitive play
    strong_agent = OpenSpielMCTSAgent(
        num_simulations=800,
        uct_c=1.0,
        solve=False,  # Can enable solver for exact solutions (slower)
    )
    print(f"✅ Strong agent: {strong_agent.num_simulations} simulations")

    # Test action selection
    print("\n2. Action Selection Demo")
    print("-" * 40)

    legal_actions = game.get_legal_actions()
    print(f"Game has {len(legal_actions)} legal actions")

    # Fast selection
    start_time = time.time()
    action = fast_agent.select_action(game, deterministic=True)
    fast_time = time.time() - start_time
    print(f"Fast agent selected action in {fast_time:.3f}s: {action}")

    # Strong selection
    start_time = time.time()
    action = strong_agent.select_action(game, deterministic=True)
    strong_time = time.time() - start_time
    print(f"Strong agent selected action in {strong_time:.3f}s: {action}")

    # Get action probabilities
    print("\n3. Action Probabilities")
    print("-" * 40)

    probs = strong_agent.get_action_probabilities(game)
    print(f"Got probabilities for {len(probs)} actions")
    print(f"Max probability: {max(probs):.3f}")
    print(f"Min probability: {min(probs):.3f}")

    # Demonstration game
    print("\n4. Playing a Short Game")
    print("-" * 40)

    # Use fast agents for quick demo
    agents = [fast_agent, fast_agent]
    current_game = GameState(num_players=2, seed=123)

    moves = 0
    max_moves = 10  # Limit for demo

    while not current_game.game_over and moves < max_moves:
        current_player = current_game.current_player
        agent = agents[current_player]

        start_time = time.time()
        action = agent.select_action(current_game, deterministic=True)
        move_time = time.time() - start_time

        print(
            f"Move {moves + 1}: Player {current_player} played {action} ({move_time:.3f}s)"
        )

        # Apply the action
        current_game.apply_action(action)
        moves += 1

    if current_game.game_over:
        print(f"\n🏁 Game finished after {moves} moves!")
        scores = [player.score for player in current_game.players]
        winner = scores.index(max(scores))
        print(f"Winner: Player {winner} with score {max(scores)}")
        print(f"Final scores: {scores}")
    else:
        print(f"\n⏸️  Demo stopped after {moves} moves")

    # Performance comparison
    print("\n5. Performance Comparison")
    print("-" * 40)

    test_game = GameState(num_players=2, seed=456)

    # Test different simulation counts
    simulation_counts = [50, 100, 200, 400]

    for sim_count in simulation_counts:
        agent = OpenSpielMCTSAgent(num_simulations=sim_count)

        start_time = time.time()
        action = agent.select_action(test_game, deterministic=True)
        elapsed = time.time() - start_time

        print(
            f"Simulations: {sim_count:3d} | Time: {elapsed:.3f}s | Sims/sec: {sim_count/elapsed:.1f}"
        )

    print("\n6. OpenSpiel MCTS Advantages")
    print("-" * 40)
    print("✅ No neural network required")
    print("✅ Fast C++ implementation")
    print("✅ Robust and well-tested")
    print("✅ Supports MCTS-Solver for exact solutions")
    print("✅ Efficient memory management")
    print("✅ Easy to configure and use")

    print("\n" + "=" * 60)
    print("OpenSpiel MCTS Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
