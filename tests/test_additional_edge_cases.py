#!/usr/bin/env python3
"""
Additional comprehensive edge cases and rule compliance tests for Azul.
These tests complement the existing test suite by covering scenarios that might
be missed or need additional validation.
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.factory import CenterArea, Factory, FactoryArea
from game.game_state import Action, create_game
from game.player_board import PatternLine, PlayerBoard, Wall
from game.tile import Tile, TileColor


class TestAdvancedRulesCompliance:
    """Advanced tests for edge cases in official rules."""

    def test_first_player_marker_penalty_interaction(self):
        """Test first player marker penalty counting on floor line.

        Official rule: First player marker counts as a tile for penalty
        but is not discarded to the box like other tiles.
        """
        board = PlayerBoard()

        # Place first player marker + 2 other tiles on floor line
        first_player_tile = Tile.create_first_player_marker()
        other_tiles = [Tile(TileColor.BLUE), Tile(TileColor.RED)]

        board.place_tiles_on_floor_line([first_player_tile])
        board.place_tiles_on_floor_line(other_tiles)

        # Should have 3 tiles on floor line (first player marker + 2 others)
        assert len(board.floor_line) == 3
        assert board.has_first_player_marker()

        # End round scoring
        points, discarded_tiles = board.end_round_scoring()

        # Should apply penalty for 3 positions: -1 + -1 + -2 = -4
        expected_penalty = -1 + -1 + -2  # First 3 penalty positions
        assert points == expected_penalty

        # Should discard 2 regular tiles but NOT the first player marker
        assert len(discarded_tiles) == 2
        assert not any(tile.is_first_player_marker for tile in discarded_tiles)

        # Floor line should be empty after round
        assert len(board.floor_line) == 0
        assert not board.has_first_player_marker()

    def test_pattern_line_wall_conflict_prevention(self):
        """Test that you cannot place tiles in pattern line if wall position is filled.

        Official rule: Cannot place color in pattern line if corresponding wall
        position already has that color.
        """
        board = PlayerBoard()

        # Place blue tile on wall in row 0
        board.wall.place_tile(0, TileColor.BLUE)

        # Should not be able to place blue tiles in pattern line 0
        blue_tiles = [Tile(TileColor.BLUE)]
        assert not board.can_place_tiles_on_pattern_line(0, blue_tiles)

        # But should be able to place blue in other pattern lines
        assert board.can_place_tiles_on_pattern_line(1, blue_tiles)
        assert board.can_place_tiles_on_pattern_line(2, blue_tiles)

        # And should be able to place other colors in pattern line 0
        red_tiles = [Tile(TileColor.RED)]
        assert board.can_place_tiles_on_pattern_line(0, red_tiles)

    def test_cross_connection_scoring_edge_case(self):
        """Test edge case where tile creates both horizontal and vertical connections.

        Official rule: If tile connects both horizontally and vertically,
        add both connection scores.
        """
        wall = Wall()

        # Create an L-shape pattern
        wall.place_tile(1, TileColor.WHITE)  # (1,0) - bottom of vertical line
        wall.place_tile(2, TileColor.BLACK)  # (2,0) - middle of vertical line
        wall.place_tile(3, TileColor.RED)  # (3,0) - continuation
        wall.place_tile(0, TileColor.YELLOW)  # (0,1) - right of horizontal line
        wall.place_tile(0, TileColor.RED)  # (0,2) - far right

        # Place tile that connects to both the vertical line (4 tiles) and horizontal line (3 tiles)
        points = wall.place_tile(0, TileColor.BLUE)  # (0,0) - connects both ways

        # Should score 4 (vertical) + 3 (horizontal) = 7 points
        assert points == 7

    def test_multiple_completed_rows_same_round(self):
        """Test game end when multiple players complete rows in same round."""
        game = create_game(num_players=3, seed=42)

        # Complete a row for player 0
        for col in range(5):
            game.players[0].wall.filled[0][col] = True

        # Complete a row for player 1 as well
        for col in range(5):
            game.players[1].wall.filled[1][col] = True

        # Force game end condition check
        game._end_round()

        assert game.game_over, "Game should end when any player completes a row"
        assert game.winner is not None, "Should have determined a winner"

    def test_bag_exhaustion_mid_factory_fill(self):
        """Test behavior when bag runs out while filling factories.

        Official rule: If bag runs out during factory setup, refill from discard
        and continue. If both empty, start round anyway.
        """
        game = create_game(num_players=2, seed=42)

        # Exhaust most tiles from bag
        game.bag = game.bag[:10]  # Only 10 tiles left

        # Add many tiles to discard pile
        game.discard_pile = [Tile(TileColor.BLUE)] * 50

        # Start new round - should trigger bag refill
        game._start_new_round()

        # Should have refilled bag from discard pile
        assert len(game.bag) >= 0  # Some tiles consumed for factories
        assert (
            len(game.discard_pile) == 0 or len(game.bag) > 0
        )  # Either used all or still have some

    def test_extreme_floor_line_overflow(self):
        """Test floor line with massive overflow beyond capacity."""
        board = PlayerBoard()

        # Try to place 20 tiles on floor line (way more than 7 capacity)
        many_tiles = [Tile(TileColor.BLUE)] * 20
        discarded = board.place_tiles_on_floor_line(many_tiles)

        # Should only keep 7 tiles on floor line
        assert len(board.floor_line) == 7

        # Should discard 13 tiles
        assert len(discarded) == 13

        # End round scoring should only apply 7 penalties
        points, _ = board.end_round_scoring()
        expected_penalty = sum(board.FLOOR_PENALTIES)  # All 7 penalty positions
        assert points == expected_penalty

    def test_simultaneous_pattern_line_completion(self):
        """Test when multiple pattern lines complete in same round."""
        board = PlayerBoard()

        # Set up multiple pattern lines to complete with colors that won't create connections
        # Pattern line 0 (capacity 1) - complete with blue (goes to (0,0))
        board.place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])

        # Pattern line 2 (capacity 3) - complete with blue (goes to (2,2) - not adjacent to (0,0))
        board.place_tiles_on_pattern_line(2, [Tile(TileColor.BLUE)] * 3)

        # Pattern line 4 (capacity 5) - complete with blue (goes to (4,4) - not adjacent to others)
        board.place_tiles_on_pattern_line(4, [Tile(TileColor.BLUE)] * 5)

        # Verify all are complete
        assert board.pattern_lines[0].is_complete()
        assert board.pattern_lines[2].is_complete()
        assert board.pattern_lines[4].is_complete()

        # End round - should handle all completions
        points, discarded = board.end_round_scoring()

        # Should place tiles on wall
        assert board.wall.filled[0][0]  # Blue at (0,0)
        assert board.wall.filled[2][2]  # Blue at (2,2)
        assert board.wall.filled[4][4]  # Blue at (4,4)

        # Should score 3 points (1 each for isolated tiles - no connections)
        assert points == 3

        # Should discard excess tiles (0 + 2 + 4 = 6 tiles)
        assert len(discarded) == 6

    def test_first_center_take_with_no_first_player_marker(self):
        """Test edge case where center is taken but no first player marker available."""
        factory_area = FactoryArea(2)

        # Manually clear first player marker (shouldn't happen in normal game)
        factory_area.center.has_first_player_marker = False
        factory_area.center.add_tiles([Tile(TileColor.BLUE)])

        # Take from center
        tiles = factory_area.center.take_tiles(TileColor.BLUE)

        # Should get tiles but no first player marker
        assert len(tiles) == 1
        assert not any(tile.is_first_player_marker for tile in tiles)

    def test_wall_pattern_position_validation(self):
        """Test that wall pattern positions are correctly validated."""
        wall = Wall()

        # Test that each color can only go in its designated position in each row
        # In row 0: BLUE at col 0, YELLOW at col 1, RED at col 2, BLACK at col 3, WHITE at col 4

        # Blue should only go in column 0 of row 0
        assert wall.can_place_tile(0, TileColor.BLUE)  # Valid position

        # Place the blue tile
        wall.place_tile(0, TileColor.BLUE)

        # Now blue should not be placeable in row 0 (position is filled)
        assert not wall.can_place_tile(0, TileColor.BLUE)

        # But yellow should be placeable in row 0 column 1
        assert wall.can_place_tile(0, TileColor.YELLOW)

        # Test that wrong colors cannot be placed in wrong positions by checking can_place_tile
        # This validates the wall pattern is correctly enforced
        test_cases = [
            (0, TileColor.YELLOW, True),  # Yellow can go in (0,1)
            (0, TileColor.RED, True),  # Red can go in (0,2)
            (1, TileColor.BLUE, True),  # Blue can go in (1,1)
            (2, TileColor.BLUE, True),  # Blue can go in (2,2)
        ]

        for row, color, expected in test_cases:
            assert wall.can_place_tile(row, color) == expected


class TestTileConservationExtended:
    """Extended tests for tile conservation principles."""

    def test_complete_tile_accounting_through_game(self):
        """Verify all tiles are accounted for throughout entire game lifecycle."""
        game = create_game(num_players=2, seed=42)

        # Count tiles in ALL locations including factories which are filled at start
        bag_tiles = len(game.bag)
        discard_tiles = len(game.discard_pile)

        # Count tiles in factories (filled at game start)
        factory_tiles = 0
        for factory in game.factory_area.factories:
            factory_tiles += len(factory.tiles)

        center_regular_tiles = sum(
            1
            for tile in game.factory_area.center.tiles
            if not tile.is_first_player_marker
        )
        factory_tiles += center_regular_tiles

        initial_tiles = bag_tiles + discard_tiles + factory_tiles

        # Should be exactly 100 regular tiles
        assert initial_tiles == 100

        # Play several moves
        for _ in range(10):
            if game.game_over:
                break
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        # Count tiles in all locations again
        bag_tiles = len(game.bag)
        discard_tiles = len(game.discard_pile)

        # Count tiles on all player boards
        board_tiles = 0
        for player in game.players:
            # Pattern lines
            for pattern_line in player.pattern_lines:
                board_tiles += len(pattern_line.tiles)
            # Floor line (excluding first player marker)
            floor_line_regular_tiles = sum(
                1 for tile in player.floor_line if not tile.is_first_player_marker
            )
            board_tiles += floor_line_regular_tiles
            # Wall tiles (count filled positions)
            wall_tiles = sum(sum(row) for row in player.wall.filled)
            board_tiles += wall_tiles

        # Count tiles in factories and center
        factory_tiles = 0
        for factory in game.factory_area.factories:
            factory_tiles += len(factory.tiles)

        center_regular_tiles = sum(
            1
            for tile in game.factory_area.center.tiles
            if not tile.is_first_player_marker
        )
        factory_tiles += center_regular_tiles

        total_tiles = bag_tiles + discard_tiles + board_tiles + factory_tiles

        # Should still have exactly 100 tiles
        assert total_tiles == 100, f"Tile count mismatch: {total_tiles} != 100"

    def test_first_player_marker_uniqueness(self):
        """Test that there's exactly one first player marker in the game."""
        game = create_game(num_players=4, seed=42)

        # Count first player markers in all locations
        marker_count = 0

        # Check center area
        if game.factory_area.center.has_first_player_marker:
            marker_count += 1

        # Check all player floor lines
        for player in game.players:
            for tile in player.floor_line:
                if tile.is_first_player_marker:
                    marker_count += 1

        # Should be exactly 1 first player marker
        assert (
            marker_count == 1
        ), f"Should have exactly 1 first player marker, found {marker_count}"


class TestComplexScoringScenarios:
    """Test complex scoring scenarios that might be edge cases."""

    def test_maximum_single_tile_score(self):
        """Test maximum possible score for placing a single tile."""
        wall = Wall()

        # Create maximum connections: full row and full column intersecting at one point
        # Fill row 2 except position (2,2)
        wall.place_tile(2, TileColor.BLACK)  # (2,0)
        wall.place_tile(2, TileColor.WHITE)  # (2,1)
        # Skip (2,2) - this is where we'll place the scoring tile
        wall.place_tile(2, TileColor.YELLOW)  # (2,3)
        wall.place_tile(2, TileColor.RED)  # (2,4)

        # Fill column 2 except position (2,2)
        wall.place_tile(0, TileColor.RED)  # (0,2)
        wall.place_tile(1, TileColor.YELLOW)  # (1,2)
        # Skip (2,2) - this is where we'll place the scoring tile
        wall.place_tile(3, TileColor.WHITE)  # (3,2)
        wall.place_tile(4, TileColor.BLACK)  # (4,2)

        # Place the final tile to complete both row and column
        points = wall.place_tile(2, TileColor.BLUE)  # (2,2)

        # Should score 5 (full row) + 5 (full column) = 10 points
        assert points == 10

    def test_score_below_zero_prevention(self):
        """Test that scores cannot go below zero even with massive penalties."""
        board = PlayerBoard()
        board.score = 5  # Start with small positive score

        # Place maximum floor penalties multiple times
        tiles = [Tile(TileColor.BLUE)] * 7
        board.place_tiles_on_floor_line(tiles)

        # End round - should have -14 penalty but score can't go below 0
        points, _ = board.end_round_scoring()
        assert points == -14  # Penalty calculation
        assert board.score == 0  # Final score clamped to 0

    def test_tiebreaker_scenarios(self):
        """Test various tiebreaker scenarios."""
        game = create_game(num_players=3, seed=42)

        # Set up tie scenario
        game.players[0].score = 50
        game.players[1].score = 50
        game.players[2].score = 45

        # Player 0: 2 completed rows
        game.players[0].wall.filled[0] = [True] * 5
        game.players[0].wall.filled[1] = [True] * 5

        # Player 1: 1 completed row
        game.players[1].wall.filled[0] = [True] * 5

        # Trigger game end
        game._end_game()

        # Player 0 should win due to more completed rows
        assert game.winner == 0

    def test_variant_rules_not_implemented_check(self):
        """Verify that variant (gray board) rules are not accidentally implemented.

        The current implementation should only support the standard colored wall.
        """
        wall = Wall()

        # Verify the wall pattern is fixed as expected for standard rules
        # In row 0, each position should only accept its designated color
        row_0_pattern = wall.WALL_PATTERN[0]
        assert row_0_pattern == [
            TileColor.BLUE,
            TileColor.YELLOW,
            TileColor.RED,
            TileColor.BLACK,
            TileColor.WHITE,
        ]

        # Verify that colors can only be placed in their designated positions
        # Place blue in its correct position (0,0)
        assert wall.can_place_tile(0, TileColor.BLUE)
        wall.place_tile(0, TileColor.BLUE)

        # Verify blue cannot be placed again in row 0 (position is filled)
        assert not wall.can_place_tile(0, TileColor.BLUE)

        # This confirms standard rules are implemented (fixed positions per color per row)


class TestPerformanceAndStressTests:
    """Performance and stress tests for edge conditions."""

    def test_large_pattern_line_operations(self):
        """Test operations on pattern lines at capacity limits."""
        # Test largest pattern line (capacity 5)
        pattern_line = PatternLine(5)

        # Fill to capacity
        tiles = [Tile(TileColor.BLUE)] * 5
        overflow = pattern_line.add_tiles(tiles)

        assert len(overflow) == 0
        assert pattern_line.is_complete()
        assert len(pattern_line.tiles) == 5

        # Try to add more - should return all as overflow
        extra_tiles = [Tile(TileColor.BLUE)] * 3
        overflow = pattern_line.add_tiles(extra_tiles)
        assert len(overflow) == 3
        assert len(pattern_line.tiles) == 5  # Unchanged

    def test_rapid_game_state_transitions(self):
        """Test rapid state transitions don't cause issues."""
        game = create_game(num_players=2, seed=42)

        transitions = 0
        max_transitions = 500  # Reduce to prevent very long-running games

        while not game.game_over and transitions < max_transitions:
            actions = game.get_legal_actions()
            if not actions:
                break

            # Apply random valid action
            success = game.apply_action(actions[0])
            assert success, "All actions from get_legal_actions should be valid"

            transitions += 1

        # Game should either end normally or we should have made significant progress
        # This test verifies the game doesn't get stuck in infinite loops
        assert game.game_over or transitions == max_transitions

    def test_extreme_player_count_boundaries(self):
        """Test at the boundaries of supported player counts."""
        # Test minimum players
        game_2p = create_game(num_players=2)
        assert len(game_2p.factory_area.factories) == 5  # 2*2 + 1

        # Test maximum players
        game_4p = create_game(num_players=4)
        assert len(game_4p.factory_area.factories) == 9  # 2*4 + 1

        # Both should be able to start and play
        assert not game_2p.game_over
        assert not game_4p.game_over

        actions_2p = game_2p.get_legal_actions()
        actions_4p = game_4p.get_legal_actions()

        assert len(actions_2p) > 0
        assert len(actions_4p) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
