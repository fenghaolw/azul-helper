#!/usr/bin/env python3
"""
Comprehensive tests for Azul scoring mechanics.
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game_state import create_game
from game.player_board import PlayerBoard, Wall
from game.tile import Tile, TileColor


class TestWallScoring:
    """Test wall scoring mechanics."""

    def test_single_tile_placement(self):
        """Test scoring for placing a single isolated tile."""
        wall = Wall()

        # Place a single tile - should score 1 point
        points = wall.place_tile(0, TileColor.BLUE)
        assert points == 1, "Single tile should score 1 point"

    def test_horizontal_connections(self):
        """Test scoring for horizontal tile connections."""
        wall = Wall()

        # Place first tile
        wall.place_tile(0, TileColor.BLUE)  # Position (0,0)

        # Place adjacent tile horizontally - should score 2 points
        points = wall.place_tile(0, TileColor.YELLOW)  # Position (0,1)
        assert points == 2, "Two connected tiles should score 2 points"

        # Place third tile in same row - should score 3 points
        points = wall.place_tile(0, TileColor.RED)  # Position (0,2)
        assert points == 3, "Three connected tiles should score 3 points"

    def test_vertical_connections(self):
        """Test scoring for vertical tile connections."""
        wall = Wall()

        # Place first tile
        wall.place_tile(0, TileColor.BLUE)  # Position (0,0)

        # Place adjacent tile vertically - should score 2 points
        points = wall.place_tile(1, TileColor.WHITE)  # Position (1,0)
        assert points == 2, "Two vertically connected tiles should score 2 points"

    def test_cross_connections(self):
        """Test scoring when tile connects both horizontally and vertically."""
        wall = Wall()

        # Create a cross pattern
        wall.place_tile(0, TileColor.BLUE)  # (0,0)
        wall.place_tile(0, TileColor.YELLOW)  # (0,1)
        wall.place_tile(1, TileColor.WHITE)  # (1,0)

        # Place tile that connects both ways
        points = wall.place_tile(1, TileColor.BLUE)  # (1,1)
        assert points == 4, "Cross connection should score 4 points (2+2)"

    def test_complex_scoring_pattern(self):
        """Test complex scoring with multiple connections."""
        wall = Wall()

        # Build a more complex pattern
        wall.place_tile(0, TileColor.BLUE)  # (0,0)
        wall.place_tile(0, TileColor.YELLOW)  # (0,1)
        wall.place_tile(0, TileColor.RED)  # (0,2)
        wall.place_tile(1, TileColor.WHITE)  # (1,0)
        wall.place_tile(2, TileColor.BLACK)  # (2,0)

        # Place tile connecting to both horizontal and vertical lines
        points = wall.place_tile(1, TileColor.BLUE)  # (1,1)
        assert points == 4, "Should score 3 (horizontal) + 1 (vertical) = 4"

    def test_completed_rows_columns_colors(self):
        """Test detection of completed rows, columns, and colors."""
        wall = Wall()

        # Complete first row
        wall.place_tile(0, TileColor.BLUE)
        wall.place_tile(0, TileColor.YELLOW)
        wall.place_tile(0, TileColor.RED)
        wall.place_tile(0, TileColor.BLACK)
        wall.place_tile(0, TileColor.WHITE)

        assert wall.is_row_complete(0), "First row should be complete"
        assert len(wall.get_completed_rows()) == 1

        # Complete first column
        wall.place_tile(1, TileColor.WHITE)
        wall.place_tile(2, TileColor.BLACK)
        wall.place_tile(3, TileColor.RED)
        wall.place_tile(4, TileColor.YELLOW)

        assert wall.is_column_complete(0), "First column should be complete"
        assert len(wall.get_completed_columns()) == 1

        # Check if blue color is complete (need all 5 blue positions)
        wall.place_tile(1, TileColor.BLUE)
        wall.place_tile(2, TileColor.BLUE)
        wall.place_tile(3, TileColor.BLUE)
        wall.place_tile(4, TileColor.BLUE)

        assert wall.is_color_complete(TileColor.BLUE), "Blue color should be complete"
        assert len(wall.get_completed_colors()) == 1


class TestPlayerBoardScoring:
    """Test player board scoring mechanics."""

    def test_pattern_line_completion(self):
        """Test scoring when pattern lines are completed."""
        board = PlayerBoard()

        # Fill pattern line 2 (capacity 3) with blue tiles
        blue_tiles = [Tile(TileColor.BLUE)] * 3
        board.place_tiles_on_pattern_line(2, blue_tiles)

        assert board.pattern_lines[2].is_complete()

        # End round scoring should move tile to wall and score points
        points, discard = board.end_round_scoring()
        assert points == 1, "Single tile placement should score 1 point"
        assert len(discard) == 2, "Should discard 2 excess tiles"
        assert board.wall.filled[2][2], "Blue tile should be on wall at (2,2)"

    def test_floor_line_penalties(self):
        """Test floor line penalty calculations."""
        board = PlayerBoard()

        # Place tiles on floor line
        tiles = [Tile(TileColor.BLUE)] * 7  # Fill all penalty positions
        board.place_tiles_on_floor_line(tiles)

        points, discard = board.end_round_scoring()
        expected_penalty = sum(board.FLOOR_PENALTIES)  # -1-1-2-2-2-3-3 = -14
        assert points == expected_penalty, f"Should have penalty of {expected_penalty}"

    def test_score_cannot_go_negative(self):
        """Test that player score cannot go below 0."""
        board = PlayerBoard()
        board.score = 5

        # Add enough floor penalties to go negative
        tiles = [Tile(TileColor.BLUE)] * 7
        board.place_tiles_on_floor_line(tiles)

        points, _ = board.end_round_scoring()
        assert board.score == 0, "Score should not go below 0"

    def test_first_player_marker_handling(self):
        """Test first player marker on floor line."""
        board = PlayerBoard()

        # Add first player marker to floor line
        first_player_tile = Tile.create_first_player_marker()
        board.place_tiles_on_floor_line([first_player_tile])

        assert board.has_first_player_marker()

        # End round - first player marker should not be discarded
        points, discard = board.end_round_scoring()
        assert not any(tile.is_first_player_marker for tile in discard)

    def test_final_scoring_bonuses(self):
        """Test final scoring bonuses for completed rows/columns/colors."""
        board = PlayerBoard()

        # Manually set up completed patterns for testing
        # Complete first row
        for col in range(5):
            board.wall.filled[0][col] = True

        # Complete first column
        for row in range(5):
            board.wall.filled[row][0] = True

        # Complete blue color (all blue positions)
        blue_positions = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        for row, col in blue_positions:
            board.wall.filled[row][col] = True

        initial_score = board.score
        bonus = board.final_scoring()

        expected_bonus = 2 + 7 + 10  # 2 for row, 7 for column, 10 for color
        assert bonus == expected_bonus
        assert board.score == initial_score + expected_bonus


class TestEdgeCaseScoring:
    """Test edge cases in scoring."""

    def test_overflow_tiles_to_floor(self):
        """Test that overflow tiles go to floor line."""
        board = PlayerBoard()

        # Try to place 4 tiles on pattern line with capacity 2
        tiles = [Tile(TileColor.BLUE)] * 4
        board.place_tiles_on_pattern_line(1, tiles)  # Line 1 has capacity 2

        # Since can_add_tiles returns False for 4 tiles on capacity 2 line,
        # all tiles go to floor line
        assert (
            len(board.pattern_lines[1].tiles) == 0
        ), "Pattern line should reject all tiles"
        assert len(board.floor_line) == 4, "All tiles should go to floor"

    def test_invalid_color_placement(self):
        """Test placing wrong color on pattern line."""
        board = PlayerBoard()

        # Place blue tiles on line 0
        blue_tiles = [Tile(TileColor.BLUE)]
        board.place_tiles_on_pattern_line(0, blue_tiles)

        # Try to place red tiles on same line
        red_tiles = [Tile(TileColor.RED)]
        board.place_tiles_on_pattern_line(0, red_tiles)

        # Red tiles should go to floor line
        assert len(board.floor_line) == 1
        assert board.floor_line[0].color == TileColor.RED

    def test_wall_position_already_filled(self):
        """Test placing tile where wall position is already filled."""
        board = PlayerBoard()

        # Manually fill wall position for blue in row 0
        board.wall.filled[0][0] = True

        # Try to place blue tiles on pattern line 0
        blue_tiles = [Tile(TileColor.BLUE)]
        can_place = board.can_place_tiles_on_pattern_line(0, blue_tiles)

        assert (
            not can_place
        ), "Should not be able to place tiles for filled wall position"

    def test_multiple_pattern_lines_complete_same_round(self):
        """Test scoring when multiple pattern lines complete in same round."""
        board = PlayerBoard()

        # Complete multiple pattern lines
        board.place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])
        board.place_tiles_on_pattern_line(1, [Tile(TileColor.YELLOW)] * 2)

        points, discard = board.end_round_scoring()

        # Should score for both tiles placed on wall
        assert points == 2, "Should score 1 point for each tile placed"
        assert len(discard) == 1, "Should discard 1 excess tile from line 1"

    def test_empty_pattern_lines_no_scoring(self):
        """Test that empty pattern lines don't affect scoring."""
        board = PlayerBoard()

        # Don't place any tiles
        points, discard = board.end_round_scoring()

        assert points == 0, "No tiles should mean no points"
        assert len(discard) == 0, "No tiles to discard"

    def test_partial_pattern_lines_no_wall_placement(self):
        """Test that incomplete pattern lines don't move to wall."""
        board = PlayerBoard()

        # Partially fill pattern line
        board.place_tiles_on_pattern_line(2, [Tile(TileColor.BLUE)] * 2)  # Capacity 3

        points, discard = board.end_round_scoring()

        assert points == 0, "Incomplete lines should not score"
        assert len(discard) == 0, "No tiles should be discarded"
        assert not board.wall.filled[2][2], "No tile should be placed on wall"


if __name__ == "__main__":
    pytest.main([__file__])
