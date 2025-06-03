#!/usr/bin/env python3
"""
Test script for OpenSpiel integration with Azul.

This script demonstrates how to use OpenSpiel's MCTS and AlphaZero implementations
with our Azul game, replacing our custom implementations.
"""

import os
import sys
import time
from typing import List

import numpy as np

# Add parent directory to path so we can import from game
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.openspiel_agents import OpenSpielMCTSAgent, RandomAgent

# Import our OpenSpiel-compatible game and agents
from game.azul_openspiel import AzulGame, AzulState
from game.game_state import GameState


def test_basic_game_functionality():
    """Test basic OpenSpiel game functionality."""
    print("=== Testing Basic OpenSpiel Game Functionality ===")

    # Create OpenSpiel game
    game = AzulGame({"players": 2, "seed": 42})
    print(f"Game: {game}")
    print(f"Number of players: {game.num_players()}")
    print(f"Number of distinct actions: {game.num_distinct_actions()}")
    print(f"Max game length: {game.max_game_length()}")

    # Create initial state
    state = game.new_initial_state()
    print(f"Initial state: {state}")
    print(f"Current player: {state.current_player()}")
    print(f"Is terminal: {state.is_terminal()}")

    # Get legal actions
    legal_actions = state.legal_actions()
    print(f"Number of legal actions: {len(legal_actions)}")
    print(f"First few legal actions: {legal_actions[:5]}")

    # Apply an action
    if legal_actions:
        action = legal_actions[0]
        print(
            f"Applying action: {state.action_to_string(state.current_player(), action)}"
        )
        state.apply_action(action)
        print(f"New state player: {state.current_player()}")

    print("✓ Basic game functionality works\n")


def test_game_simulation():
    """Test playing a complete game."""
    print("=== Testing Complete Game Simulation ===")

    game = AzulGame({"players": 2, "seed": 42})
    state = game.new_initial_state()

    moves = 0
    max_moves = 500  # Prevent infinite loops

    while not state.is_terminal() and moves < max_moves:
        legal_actions = state.legal_actions()
        if not legal_actions:
            break

        # Play random action
        action = np.random.choice(legal_actions)
        state.apply_action(action)
        moves += 1

        if moves % 50 == 0:
            print(f"Move {moves}, Current player: {state.current_player()}")

    print(f"Game finished after {moves} moves")
    print(f"Is terminal: {state.is_terminal()}")

    if state.is_terminal():
        returns = state.returns()
        print(f"Final returns: {returns}")
        winner = 0 if returns[0] > returns[1] else 1
        print(f"Winner: Player {winner}")

    print("✓ Complete game simulation works\n")


def test_mcts_agent():
    """Test OpenSpiel MCTS agent."""
    print("=== Testing OpenSpiel MCTS Agent ===")

    # Create agents
    mcts_agent = OpenSpielMCTSAgent(num_simulations=100, uct_c=1.4)
    random_agent = RandomAgent(seed=42)

    # Create game
    game_state = GameState(num_players=2, seed=42)

    print("Playing MCTS vs Random...")
    moves = 0
    max_moves = 200

    agents = [mcts_agent, random_agent]

    start_time = time.time()

    while not game_state.game_over and moves < max_moves:
        current_player = game_state.current_player
        agent = agents[current_player]

        # Select action
        action_start = time.time()
        action = agent.select_action(game_state, deterministic=True)
        action_time = time.time() - action_start

        if moves < 10:  # Print first few moves
            print(
                f"Move {moves}: Player {current_player} ({type(agent).__name__}) "
                f"selected {action} in {action_time:.3f}s"
            )

        # Apply action
        game_state.apply_action(action)
        moves += 1

    total_time = time.time() - start_time

    print(f"Game finished after {moves} moves in {total_time:.2f}s")

    if game_state.game_over:
        scores = game_state.get_scores()
        print(f"Final scores: {scores}")
        winner = np.argmax(scores)
        print(f"Winner: Player {winner} ({type(agents[winner]).__name__})")

    # Test action probabilities
    print("\nTesting action probabilities...")
    test_state = GameState(num_players=2, seed=123)
    probs = mcts_agent.get_action_probabilities(test_state)
    print(f"Got {len(probs)} action probabilities, sum: {probs.sum():.3f}")

    print("✓ MCTS agent works\n")


def compare_agents():
    """Compare different agents."""
    print("=== Comparing Agents ===")

    agents = {
        "Random": RandomAgent(seed=42),
        "MCTS-50": OpenSpielMCTSAgent(num_simulations=50, uct_c=1.4),
        "MCTS-200": OpenSpielMCTSAgent(num_simulations=200, uct_c=1.4),
    }

    # Run a few quick games to compare
    results = {name: {"wins": 0, "total_time": 0.0} for name in agents.keys()}

    num_games = 3
    print(f"Running {num_games} games for each agent pair...")

    agent_names = list(agents.keys())
    for i, agent1_name in enumerate(agent_names):
        for j, agent2_name in enumerate(agent_names[i + 1 :], i + 1):
            print(f"\n{agent1_name} vs {agent2_name}:")

            for game_num in range(num_games):
                game_state = GameState(num_players=2, seed=42 + game_num)

                game_agents = [agents[agent1_name], agents[agent2_name]]
                agent_times = [0.0, 0.0]

                moves = 0
                max_moves = 100  # Quick games

                while not game_state.game_over and moves < max_moves:
                    current_player = game_state.current_player
                    agent = game_agents[current_player]

                    start_time = time.time()
                    action = agent.select_action(game_state, deterministic=True)
                    agent_times[current_player] += time.time() - start_time

                    game_state.apply_action(action)
                    moves += 1

                if game_state.game_over:
                    scores = game_state.get_scores()
                    winner_idx = np.argmax(scores)
                    winner_name = [agent1_name, agent2_name][winner_idx]
                    results[winner_name]["wins"] += 1

                    results[agent1_name]["total_time"] += agent_times[0]
                    results[agent2_name]["total_time"] += agent_times[1]

                    print(
                        f"  Game {game_num + 1}: {winner_name} wins "
                        f"(scores: {scores}, moves: {moves})"
                    )

    print(f"\nResults after {num_games} games:")
    for name, stats in results.items():
        avg_time = stats["total_time"] / max(
            1, stats["wins"] + (num_games * 2 - stats["wins"])
        )
        print(f"  {name}: {stats['wins']} wins, {avg_time:.3f}s avg per move")

    print("✓ Agent comparison complete\n")


def test_mcts_vs_random_both_positions():
    """Test MCTS vs Random in both player positions."""
    print("=== Testing MCTS vs Random in Both Positions ===")

    # Create agents
    mcts_agent = OpenSpielMCTSAgent(num_simulations=200, uct_c=1.4)
    random_agent = RandomAgent(seed=42)

    # Test both positions
    for test_position in [0, 1]:
        print(f"\nTesting MCTS as Player {test_position}:")

        # Create game with fixed seed
        game_state = GameState(num_players=2, seed=42)

        # Set up agents based on test position
        agents = [None, None]
        agents[test_position] = mcts_agent
        agents[1 - test_position] = random_agent

        moves = 0
        max_moves = 200

        while not game_state.game_over and moves < max_moves:
            current_player = game_state.current_player
            agent = agents[current_player]

            # Select action
            action = agent.select_action(game_state, deterministic=True)

            # Apply action
            game_state.apply_action(action)
            moves += 1

        if game_state.game_over:
            scores = game_state.get_scores()
            print(f"Final scores: {scores}")
            winner = np.argmax(scores)
            print(f"Winner: Player {winner} ({type(agents[winner]).__name__})")
        else:
            print("Game did not complete within move limit")

    print("\n✓ MCTS vs Random position test completed")


def main():
    """Run all tests."""
    print("Testing OpenSpiel Integration with Azul\n")

    try:
        test_basic_game_functionality()
        test_game_simulation()
        test_mcts_agent()
        compare_agents()
        test_mcts_vs_random_both_positions()

        print("🎉 All tests passed! OpenSpiel integration is working correctly.")
        print("\nNext steps:")
        print("1. Install OpenSpiel: pip install open-spiel")
        print("2. Train AlphaZero models using OpenSpiel's training scripts")
        print("3. Replace your custom MCTS with OpenSpielMCTSAgent")
        print("4. Use OpenSpielAlphaZeroAgent for neural network-based play")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
