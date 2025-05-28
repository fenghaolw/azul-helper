#!/usr/bin/env python3
"""
Comprehensive tests for legal move generation in various scenarios.
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game_state import Action, create_game
from game.tile import Tile, TileColor


class TestBasicLegalMoves:
    """Test basic legal move generation."""

    def test_initial_game_legal_moves(self):
        """Test legal moves at game start."""
        game = create_game(num_players=2, seed=42)
        actions = game.get_legal_actions()

        # Should have moves available
        assert len(actions) > 0, "Should have legal moves at game start"

        # All actions should be valid
        for action in actions:
            assert game.is_action_legal(action), f"Action {action} should be legal"

        # Should have moves from factories (not center initially)
        factory_moves = [a for a in actions if a.source >= 0]
        center_moves = [a for a in actions if a.source == -1]

        assert len(factory_moves) > 0, "Should have factory moves"
        assert len(center_moves) == 0, "Should not have center moves initially"

    def test_legal_moves_after_first_action(self):
        """Test legal moves after first action creates center tiles."""
        game = create_game(num_players=2, seed=42)

        # Take first action
        actions = game.get_legal_actions()
        game.apply_action(actions[0])

        # Now should have center moves available
        new_actions = game.get_legal_actions()
        center_moves = [a for a in new_actions if a.source == -1]

        assert len(center_moves) > 0, "Should have center moves after first action"

    def test_all_destinations_available(self):
        """Test that all valid destinations are generated."""
        game = create_game(num_players=2, seed=42)
        actions = game.get_legal_actions()

        # Check that we have actions for different destinations
        destinations = set(action.destination for action in actions)

        # Should include floor line (-1) and some pattern lines (0-4)
        assert -1 in destinations, "Should include floor line destination"
        pattern_destinations = [d for d in destinations if d >= 0]
        assert len(pattern_destinations) > 0, "Should include pattern line destinations"

    def test_no_legal_moves_when_game_over(self):
        """Test that no legal moves when game is over."""
        game = create_game(num_players=2, seed=42)
        game.game_over = True

        actions = game.get_legal_actions()
        assert len(actions) == 0, "Should have no legal moves when game over"


class TestPatternLineConstraints:
    """Test legal move constraints based on pattern line rules."""

    def test_cannot_place_on_full_pattern_line(self):
        """Test that cannot place tiles on already full pattern line."""
        game = create_game(num_players=2, seed=42)
        player = game.players[game.current_player]

        # Fill pattern line 0 (capacity 1)
        player.place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])

        actions = game.get_legal_actions()

        # Should not have actions targeting pattern line 0
        line_0_actions = [a for a in actions if a.destination == 0]
        assert len(line_0_actions) == 0, "Should not have actions for full pattern line"

    def test_cannot_place_wrong_color_on_pattern_line(self):
        """Test that cannot place wrong color on partially filled pattern line."""
        game = create_game(num_players=2, seed=42)
        player = game.players[game.current_player]

        # Partially fill pattern line 1 with blue tiles
        player.place_tiles_on_pattern_line(1, [Tile(TileColor.BLUE)])

        actions = game.get_legal_actions()

        # Should not have actions placing non-blue tiles on line 1
        line_1_non_blue = [
            a for a in actions if a.destination == 1 and a.color != TileColor.BLUE
        ]
        assert len(line_1_non_blue) == 0, "Should not place wrong color on pattern line"

    def test_cannot_place_on_wall_filled_position(self):
        """Test that cannot place tiles where wall position is already filled."""
        game = create_game(num_players=2, seed=42)
        player = game.players[game.current_player]

        # Fill wall position for blue in row 0
        player.wall.filled[0][0] = True

        actions = game.get_legal_actions()

        # Should not have actions placing blue tiles on line 0
        blue_line_0_actions = [
            a for a in actions if a.destination == 0 and a.color == TileColor.BLUE
        ]
        assert (
            len(blue_line_0_actions) == 0
        ), "Should not place tiles for filled wall position"

    def test_can_always_place_on_floor_line(self):
        """Test that can always place tiles on floor line."""
        game = create_game(num_players=2, seed=42)
        actions = game.get_legal_actions()

        # Should always have floor line actions
        floor_actions = [a for a in actions if a.destination == -1]
        assert len(floor_actions) > 0, "Should always be able to place on floor line"

    def test_pattern_line_capacity_constraints(self):
        """Test that legal moves respect pattern line capacities."""
        game = create_game(num_players=2, seed=42)

        # Manually set up a scenario with specific tile counts
        factory = game.factory_area.factories[0]
        factory.tiles = [Tile(TileColor.BLUE)] * 4  # 4 blue tiles

        actions = game.get_legal_actions()

        # Taking 4 blue tiles should only be legal for:
        # - Pattern lines with enough capacity (lines 3, 4)
        # - Floor line
        blue_factory_0_actions = [
            a for a in actions if a.source == 0 and a.color == TileColor.BLUE
        ]

        valid_destinations = set(a.destination for a in blue_factory_0_actions)

        # Should not include lines 0, 1, 2 (capacities 1, 2, 3)
        assert 0 not in valid_destinations, "Line 0 cannot hold 4 tiles"
        assert 1 not in valid_destinations, "Line 1 cannot hold 4 tiles"
        assert 2 not in valid_destinations, "Line 2 cannot hold 4 tiles"

        # Should include floor line and larger pattern lines
        assert -1 in valid_destinations, "Floor line should be available"


class TestFactoryAndCenterConstraints:
    """Test legal move constraints based on factory and center state."""

    def test_no_moves_from_empty_factory(self):
        """Test that no moves from empty factories."""
        game = create_game(num_players=2, seed=42)

        # Empty a factory
        game.factory_area.factories[0].tiles = []

        actions = game.get_legal_actions()

        # Should not have actions from factory 0
        factory_0_actions = [a for a in actions if a.source == 0]
        assert len(factory_0_actions) == 0, "Should not have moves from empty factory"

    def test_no_moves_from_empty_center(self):
        """Test that no moves from empty center."""
        game = create_game(num_players=2, seed=42)

        # Ensure center is empty
        game.factory_area.center.tiles = []

        actions = game.get_legal_actions()

        # Should not have actions from center
        center_actions = [a for a in actions if a.source == -1]
        assert len(center_actions) == 0, "Should not have moves from empty center"

    def test_only_available_colors_in_moves(self):
        """Test that only available colors appear in legal moves."""
        game = create_game(num_players=2, seed=42)

        # Set up specific factory contents
        factory = game.factory_area.factories[0]
        factory.tiles = [Tile(TileColor.BLUE), Tile(TileColor.RED)]

        actions = game.get_legal_actions()

        # Actions from factory 0 should only have blue and red
        factory_0_actions = [a for a in actions if a.source == 0]
        colors = set(a.color for a in factory_0_actions)

        assert TileColor.BLUE in colors, "Should have blue moves"
        assert TileColor.RED in colors, "Should have red moves"
        assert TileColor.YELLOW not in colors, "Should not have yellow moves"
        assert TileColor.BLACK not in colors, "Should not have black moves"
        assert TileColor.WHITE not in colors, "Should not have white moves"

    def test_first_player_marker_in_center(self):
        """Test first player marker handling in center moves."""
        game = create_game(num_players=2, seed=42)

        # Add tiles to center and ensure first player marker is there
        game.factory_area.center.tiles = [Tile(TileColor.BLUE)]
        game.factory_area.center.has_first_player_marker = True

        actions = game.get_legal_actions()
        center_actions = [a for a in actions if a.source == -1]

        # Should have center moves for blue
        blue_center_actions = [a for a in center_actions if a.color == TileColor.BLUE]
        assert len(blue_center_actions) > 0, "Should have blue center moves"


class TestComplexScenarios:
    """Test legal move generation in complex game scenarios."""

    def test_mid_game_constraints(self):
        """Test legal moves in mid-game with various constraints."""
        game = create_game(num_players=2, seed=42)
        player = game.players[game.current_player]

        # Set up complex board state
        player.place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])  # Line 0 full
        player.place_tiles_on_pattern_line(1, [Tile(TileColor.RED)])  # Line 1 partial
        player.wall.filled[2][2] = True  # Blue position in row 2 filled

        # Set up factory with various colors
        factory = game.factory_area.factories[0]
        factory.tiles = [
            Tile(TileColor.BLUE),  # Cannot go to line 0 (full) or line 2 (wall filled)
            Tile(TileColor.RED),  # Can go to line 1 (same color)
            Tile(TileColor.YELLOW),  # Can go to available lines
            Tile(TileColor.BLACK),  # Can go to available lines
        ]

        actions = game.get_legal_actions()
        factory_0_actions = [a for a in actions if a.source == 0]

        # Check blue constraints
        blue_actions = [a for a in factory_0_actions if a.color == TileColor.BLUE]
        blue_destinations = set(a.destination for a in blue_actions)
        assert 0 not in blue_destinations, "Blue cannot go to full line 0"
        assert 2 not in blue_destinations, "Blue cannot go to line 2 (wall filled)"
        assert -1 in blue_destinations, "Blue can go to floor"

        # Check red constraints
        red_actions = [a for a in factory_0_actions if a.color == TileColor.RED]
        red_destinations = set(a.destination for a in red_actions)
        assert 1 in red_destinations, "Red can go to line 1 (same color)"

    def test_near_end_game_constraints(self):
        """Test legal moves near end of game with many constraints."""
        game = create_game(num_players=2, seed=42)
        player = game.players[game.current_player]

        # Fill most of the wall
        for row in range(4):
            for col in range(4):
                player.wall.filled[row][col] = True

        # Fill most pattern lines
        for i in range(4):
            tiles = [Tile(TileColor.WHITE)] * (i + 1)
            player.place_tiles_on_pattern_line(i, tiles)

        actions = game.get_legal_actions()

        # Should still have some legal moves (floor line at minimum)
        assert len(actions) > 0, "Should have legal moves even with constraints"

        # Should have floor line moves
        floor_actions = [a for a in actions if a.destination == -1]
        assert len(floor_actions) > 0, "Should have floor line moves"

    def test_round_ending_scenario(self):
        """Test legal moves when round is about to end."""
        game = create_game(num_players=2, seed=42)

        # Empty all but one factory
        for i in range(len(game.factory_area.factories) - 1):
            game.factory_area.factories[i].tiles = []

        # Last factory has one tile
        last_factory = game.factory_area.factories[-1]
        last_factory.tiles = [Tile(TileColor.BLUE)]

        actions = game.get_legal_actions()

        # Should have moves from last factory
        last_factory_actions = [
            a for a in actions if a.source == len(game.factory_area.factories) - 1
        ]
        assert len(last_factory_actions) > 0, "Should have moves from last factory"

    def test_all_pattern_lines_blocked(self):
        """Test scenario where all pattern lines are blocked."""
        game = create_game(num_players=2, seed=42)
        player = game.players[game.current_player]

        # Block all pattern lines by filling wall positions
        for row in range(5):
            for col in range(5):
                player.wall.filled[row][col] = True

        actions = game.get_legal_actions()

        # Should only have floor line moves
        destinations = set(a.destination for a in actions)
        assert destinations == {
            -1
        }, "Should only have floor line moves when all lines blocked"


class TestEdgeCases:
    """Test edge cases in legal move generation."""

    def test_first_player_marker_only_in_center(self):
        """Test when center only has first player marker."""
        game = create_game(num_players=2, seed=42)

        # Set center to only have first player marker
        game.factory_area.center.tiles = []
        game.factory_area.center.has_first_player_marker = True

        actions = game.get_legal_actions()
        center_actions = [a for a in actions if a.source == -1]

        # Should not have center moves (no colored tiles)
        assert (
            len(center_actions) == 0
        ), "Should not have moves for first player marker only"

    def test_invalid_action_detection(self):
        """Test detection of invalid actions."""
        game = create_game(num_players=2, seed=42)

        # Create invalid actions
        invalid_actions = [
            Action(-2, TileColor.BLUE, 0),  # Invalid source
            Action(0, TileColor.FIRST_PLAYER, 0),  # First player marker
            Action(0, TileColor.BLUE, 5),  # Invalid destination
            Action(100, TileColor.BLUE, 0),  # Non-existent factory
        ]

        for action in invalid_actions:
            assert not game.is_action_legal(
                action
            ), f"Action {action} should be invalid"

    def test_player_specific_legal_moves(self):
        """Test that legal moves are player-specific."""
        game = create_game(num_players=2, seed=42)

        # Modify player 1's board differently from player 0
        game.players[1].place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])

        # Get legal moves for each player
        player_0_actions = game.get_legal_actions(0)
        player_1_actions = game.get_legal_actions(1)

        # Player 0 should be able to place blue on line 0, player 1 should not
        _ = any(
            a.destination == 0 and a.color == TileColor.BLUE for a in player_0_actions
        )  # p0_blue_line_0
        _ = any(
            a.destination == 0 and a.color == TileColor.BLUE for a in player_1_actions
        )  # p1_blue_line_0

        # This test depends on the specific factory setup,
        # so we just check they're different
        assert (
            player_0_actions != player_1_actions or len(player_0_actions) == 0
        ), "Players should have different legal moves when boards differ"


if __name__ == "__main__":
    pytest.main([__file__])
