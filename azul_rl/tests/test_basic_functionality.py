#!/usr/bin/env python3
"""
Basic tests for Azul game functionality.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game_state import create_game
from game.player_board import PlayerBoard
from game.tile import Tile, TileColor


def test_tile_creation():
    """Test basic tile functionality."""
    print("Testing tile creation...")

    # Test standard tiles
    tiles = Tile.create_standard_tiles()
    assert len(tiles) == 100, f"Expected 100 tiles, got {len(tiles)}"

    # Count each color
    color_counts = {}
    for tile in tiles:
        color = tile.color
        color_counts[color] = color_counts.get(color, 0) + 1

    # Should have 20 of each color
    expected_colors = [
        TileColor.BLUE,
        TileColor.YELLOW,
        TileColor.RED,
        TileColor.BLACK,
        TileColor.WHITE,
    ]
    for color in expected_colors:
        assert (
            color_counts[color] == 20
        ), f"Expected 20 {color} tiles, got {color_counts[color]}"

    # Test first player marker
    first_player = Tile.create_first_player_marker()
    assert first_player.is_first_player_marker

    print("✓ Tile creation tests passed")


def test_player_board():
    """Test player board functionality."""
    print("Testing player board...")

    board = PlayerBoard()

    # Test pattern line capacity
    for i, line in enumerate(board.pattern_lines):
        assert line.capacity == i + 1, f"Line {i} should have capacity {i+1}"

    # Test placing tiles
    blue_tiles = [Tile(TileColor.BLUE), Tile(TileColor.BLUE)]
    board.place_tiles_on_pattern_line(1, blue_tiles)

    assert len(board.pattern_lines[1].tiles) == 2
    assert board.pattern_lines[1].color == TileColor.BLUE

    # Test wall pattern
    wall = board.wall
    assert len(wall.WALL_PATTERN) == 5
    assert len(wall.WALL_PATTERN[0]) == 5

    print("✓ Player board tests passed")


def test_game_creation():
    """Test game creation and basic state."""
    print("Testing game creation...")

    # Test 2-player game
    game = create_game(num_players=2, seed=42)
    assert game.num_players == 2
    assert len(game.players) == 2
    assert game.factory_area.num_factories == 5  # 2*2 + 1

    # Test 4-player game
    game4 = create_game(num_players=4, seed=42)
    assert game4.num_players == 4
    assert len(game4.players) == 4
    assert game4.factory_area.num_factories == 9  # 2*4 + 1

    print("✓ Game creation tests passed")


def test_legal_actions():
    """Test legal action generation."""
    print("Testing legal actions...")

    game = create_game(num_players=2, seed=42)

    # Should have legal actions at start
    actions = game.get_legal_actions()
    assert len(actions) > 0, "Should have legal actions at game start"

    # All actions should be valid
    for action in actions[:5]:  # Test first 5 actions
        assert game.is_action_legal(action), f"Action {action} should be legal"

    print("✓ Legal actions tests passed")


def test_action_application():
    """Test applying actions."""
    print("Testing action application...")

    game = create_game(num_players=2, seed=42)
    initial_player = game.current_player

    # Get and apply a legal action
    actions = game.get_legal_actions()
    action = actions[0]

    success = game.apply_action(action)
    assert success, "Legal action should apply successfully"

    # Player should change (unless round ended)
    if not game.factory_area.is_round_over():
        assert game.current_player != initial_player, "Current player should change"

    print("✓ Action application tests passed")


def test_state_vector():
    """Test state vector generation."""
    print("Testing state vector...")

    game = create_game(num_players=2, seed=42)
    state = game.get_state_vector()

    # Should be a list of numbers
    assert isinstance(state, list), "State should be a list"
    assert len(state) > 0, "State should not be empty"

    # All values should be normalized (between 0 and 1)
    for i, value in enumerate(state):
        assert isinstance(value, (int, float)), f"State[{i}] should be numeric"
        assert 0 <= value <= 1, f"State[{i}] = {value} should be in [0,1]"

    print("✓ State vector tests passed")


def test_game_copy():
    """Test game state copying."""
    print("Testing game copying...")

    game = create_game(num_players=2, seed=42)

    # Apply some actions
    for _ in range(3):
        actions = game.get_legal_actions()
        if actions:
            game.apply_action(actions[0])

    # Copy the game
    game_copy = game.copy()

    # Should be different objects
    assert game is not game_copy, "Copy should be different object"
    assert game.players is not game_copy.players, "Players should be different objects"

    # But should have same state
    assert game.current_player == game_copy.current_player
    assert game.round_number == game_copy.round_number
    assert game.get_scores() == game_copy.get_scores()

    print("✓ Game copying tests passed")


def run_all_tests():
    """Run all tests."""
    print("Running Azul game tests...\n")

    try:
        test_tile_creation()
        test_player_board()
        test_game_creation()
        test_legal_actions()
        test_action_application()
        test_state_vector()
        test_game_copy()

        print("\n🎉 All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
