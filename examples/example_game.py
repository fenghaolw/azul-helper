#!/usr/bin/env python3
"""
Example script demonstrating basic Azul game functionality.
"""

import os
import random
import sys

# Add parent directory to path so we can import from game
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game_state import create_game


def random_agent(game_state):
    """Simple random agent that picks a random legal action."""
    legal_actions = game_state.get_legal_actions()
    if not legal_actions:
        return None
    return random.choice(legal_actions)


def print_game_state(game_state):
    """Print a simple representation of the game state."""
    print(f"\n=== Round {game_state.round_number} ===")
    print(f"Current Player: {game_state.current_player}")

    # Print scores
    scores = game_state.get_scores()
    print("Scores:", " | ".join(f"P{i}: {score}" for i, score in enumerate(scores)))

    # Print factories
    print("\nFactories:")
    for i, factory in enumerate(game_state.factory_area.factories):
        if factory.tiles:
            tiles_str = ", ".join(str(tile) for tile in factory.tiles)
            print(f"  Factory {i}: [{tiles_str}]")

    # Print center
    if (
        game_state.factory_area.center.tiles
        or game_state.factory_area.center.has_first_player_marker
    ):
        center_tiles = [str(tile) for tile in game_state.factory_area.center.tiles]
        if game_state.factory_area.center.has_first_player_marker:
            center_tiles.append("FIRST")
        print(f"  Center: [{', '.join(center_tiles)}]")

    # Print current player's board
    current_player = game_state.players[game_state.current_player]
    print(f"\nPlayer {game_state.current_player} Board:")

    # Pattern lines
    print("  Pattern Lines:")
    for i, line in enumerate(current_player.pattern_lines):
        if line.tiles:
            tiles_str = ", ".join(str(tile) for tile in line.tiles)
            print(f"    Line {i} ({line.capacity}): [{tiles_str}]")
        else:
            print(f"    Line {i} ({line.capacity}): [empty]")

    # Wall (simplified)
    completed_rows = current_player.wall.get_completed_rows()
    if completed_rows:
        print(f"  Completed wall rows: {completed_rows}")

    # Floor line
    if current_player.floor_line:
        floor_str = ", ".join(str(tile) for tile in current_player.floor_line)
        print(f"  Floor line: [{floor_str}]")


def play_example_game():
    """Play a complete game with random agents."""
    print("Starting Azul game with 2 random agents...")

    # Create game
    game = create_game(num_players=2, seed=42)

    turn_count = 0
    max_turns = 1000  # Safety limit

    while not game.game_over and turn_count < max_turns:
        print_game_state(game)

        # Get action from random agent
        action = random_agent(game)
        if action is None:
            print("No legal actions available!")
            break

        print(f"\nPlayer {game.current_player} action: {action}")

        # Apply action
        success = game.apply_action(action)
        if not success:
            print("Failed to apply action!")
            break

        turn_count += 1

    # Print final results
    print("\n" + "=" * 50)
    print("GAME OVER!")
    print(f"Winner: Player {game.winner}")

    final_scores = game.get_scores()
    print("Final Scores:")
    for i, score in enumerate(final_scores):
        print(f"  Player {i}: {score}")

    print(f"Total turns: {turn_count}")
    print(f"Total rounds: {game.round_number}")


def test_state_vector():
    """Test the state vector representation."""
    print("\nTesting state vector representation...")

    game = create_game(num_players=2, seed=123)
    state_vector = game.get_state_vector()

    print(f"State vector length: {len(state_vector)}")
    print(f"First 10 elements: {state_vector[:10]}")
    print(f"All elements in [0,1] range: {all(0 <= x <= 1 for x in state_vector)}")


if __name__ == "__main__":
    # Run example game
    play_example_game()

    # Test state vector
    test_state_vector()

    print("\nExample completed successfully!")
