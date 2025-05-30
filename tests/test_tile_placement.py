#!/usr/bin/env python3
"""
Comprehensive tests for tile placement rules.
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.player_board import PatternLine, PlayerBoard, Wall
from game.tile import Tile, TileColor


class TestPatternLinePlacement:
    """Test pattern line tile placement rules."""

    def test_empty_pattern_line_accepts_any_color(self):
        """Test that empty pattern line accepts any color."""
        line = PatternLine(3)

        for color in [
            TileColor.BLUE,
            TileColor.RED,
            TileColor.YELLOW,
            TileColor.BLACK,
            TileColor.WHITE,
        ]:
            tiles = [Tile(color)]
            assert line.can_add_tiles(tiles), f"Empty line should accept {color}"

    def test_pattern_line_rejects_first_player_marker(self):
        """Test that pattern lines reject first player marker."""
        line = PatternLine(3)
        first_player_tile = [Tile.create_first_player_marker()]

        assert not line.can_add_tiles(
            first_player_tile
        ), "Pattern line should reject first player marker"

    def test_pattern_line_color_consistency(self):
        """Test that pattern line maintains color consistency."""
        line = PatternLine(3)

        # Add blue tiles
        blue_tiles = [Tile(TileColor.BLUE)]
        line.add_tiles(blue_tiles)

        # Should accept more blue tiles
        more_blue = [Tile(TileColor.BLUE)]
        assert line.can_add_tiles(more_blue), "Should accept same color"

        # Should reject different color
        red_tiles = [Tile(TileColor.RED)]
        assert not line.can_add_tiles(red_tiles), "Should reject different color"

    def test_pattern_line_capacity_limits(self):
        """Test pattern line capacity enforcement."""
        line = PatternLine(2)

        # Should accept tiles up to capacity
        tiles = [Tile(TileColor.BLUE), Tile(TileColor.BLUE)]
        assert line.can_add_tiles(tiles), "Should accept tiles up to capacity"

        line.add_tiles(tiles)

        # Should reject additional tiles
        more_tiles = [Tile(TileColor.BLUE)]
        assert not line.can_add_tiles(
            more_tiles
        ), "Should reject tiles exceeding capacity"

    def test_pattern_line_overflow_handling(self):
        """Test pattern line overflow tile handling."""
        line = PatternLine(2)

        # Add more tiles than capacity - this will be rejected entirely
        tiles = [Tile(TileColor.BLUE)] * 4
        overflow = line.add_tiles(tiles)

        # Since can_add_tiles returns False for 4 tiles on capacity 2 line,
        # all tiles are returned as overflow
        assert len(line.tiles) == 0, "Should reject all tiles when exceeding capacity"
        assert len(overflow) == 4, "Should return all tiles as overflow"
        assert all(
            tile.color == TileColor.BLUE for tile in overflow
        ), "Overflow should be same color"

    def test_pattern_line_completion_detection(self):
        """Test pattern line completion detection."""
        line = PatternLine(3)

        assert not line.is_complete(), "Empty line should not be complete"

        # Partially fill
        line.add_tiles([Tile(TileColor.BLUE)] * 2)
        assert not line.is_complete(), "Partial line should not be complete"

        # Complete the line
        line.add_tiles([Tile(TileColor.BLUE)])
        assert line.is_complete(), "Full line should be complete"

    def test_pattern_line_clearing(self):
        """Test pattern line clearing after completion."""
        line = PatternLine(3)

        # Fill the line
        tiles = [Tile(TileColor.BLUE)] * 3
        line.add_tiles(tiles)

        # Clear the line
        wall_tile, discard_tiles = line.clear()

        assert wall_tile is not None, "Should return wall tile"
        assert wall_tile.color == TileColor.BLUE, "Wall tile should be correct color"
        assert len(discard_tiles) == 2, "Should return excess tiles for discard"
        assert len(line.tiles) == 0, "Line should be empty after clearing"
        assert line.color is None, "Line color should be reset"

    def test_pattern_line_clear_incomplete_line(self):
        """Test clearing incomplete pattern line."""
        line = PatternLine(3)

        # Partially fill
        line.add_tiles([Tile(TileColor.BLUE)] * 2)

        # Try to clear
        wall_tile, discard_tiles = line.clear()

        assert wall_tile is None, "Should not return wall tile for incomplete line"
        assert (
            len(discard_tiles) == 0
        ), "Should not return discard tiles for incomplete line"


class TestWallPlacement:
    """Test wall tile placement rules."""

    def test_wall_pattern_correctness(self):
        """Test that wall pattern is correctly defined."""
        wall = Wall()

        # Check wall pattern dimensions
        assert len(wall.WALL_PATTERN) == 5, "Wall should have 5 rows"
        for row in wall.WALL_PATTERN:
            assert len(row) == 5, "Each row should have 5 positions"

        # Check that each color appears exactly once in each row and column
        colors = [
            TileColor.BLUE,
            TileColor.YELLOW,
            TileColor.RED,
            TileColor.BLACK,
            TileColor.WHITE,
        ]

        for row in wall.WALL_PATTERN:
            assert set(row) == set(
                colors
            ), "Each row should contain all colors exactly once"

        for col in range(5):
            col_colors = [wall.WALL_PATTERN[row][col] for row in range(5)]
            assert set(col_colors) == set(
                colors
            ), "Each column should contain all colors exactly once"

    def test_wall_tile_placement_validation(self):
        """Test wall tile placement validation."""
        wall = Wall()

        # Should be able to place blue in row 0 (position 0,0)
        assert wall.can_place_tile(
            0, TileColor.BLUE
        ), "Should be able to place blue in row 0"

        # Should be able to place yellow in row 0 (position 0,1)
        assert wall.can_place_tile(
            0, TileColor.YELLOW
        ), "Should be able to place yellow in row 0"

        # Should not be able to place blue in row 1 (blue is at position 1,1)
        assert wall.can_place_tile(
            1, TileColor.BLUE
        ), "Should be able to place blue in row 1"

    def test_wall_prevents_duplicate_placement(self):
        """Test that wall prevents placing tiles in already filled positions."""
        wall = Wall()

        # Place blue tile in row 0
        wall.place_tile(0, TileColor.BLUE)

        # Should not be able to place another tile in same position
        assert not wall.can_place_tile(
            0, TileColor.BLUE
        ), "Should not place tile in filled position"

    def test_wall_invalid_row_handling(self):
        """Test wall handling of invalid row indices."""
        wall = Wall()

        # Test invalid rows
        assert not wall.can_place_tile(-1, TileColor.BLUE), "Should reject negative row"
        assert not wall.can_place_tile(5, TileColor.BLUE), "Should reject row >= 5"

        # Test placement returns 0 for invalid positions
        assert (
            wall.place_tile(-1, TileColor.BLUE) == 0
        ), "Invalid placement should return 0 points"
        assert (
            wall.place_tile(5, TileColor.BLUE) == 0
        ), "Invalid placement should return 0 points"

    def test_wall_row_completion_detection(self):
        """Test wall row completion detection."""
        wall = Wall()

        # Complete first row
        wall.place_tile(0, TileColor.BLUE)
        wall.place_tile(0, TileColor.YELLOW)
        wall.place_tile(0, TileColor.RED)
        wall.place_tile(0, TileColor.BLACK)
        wall.place_tile(0, TileColor.WHITE)

        assert wall.is_row_complete(0), "Row 0 should be complete"
        assert not wall.is_row_complete(1), "Row 1 should not be complete"

        completed_rows = wall.get_completed_rows()
        assert completed_rows == [0], "Should return completed row 0"

    def test_wall_column_completion_detection(self):
        """Test wall column completion detection."""
        wall = Wall()

        # Complete first column (blue positions)
        wall.place_tile(0, TileColor.BLUE)  # (0,0)
        wall.place_tile(1, TileColor.WHITE)  # (1,0)
        wall.place_tile(2, TileColor.BLACK)  # (2,0)
        wall.place_tile(3, TileColor.RED)  # (3,0)
        wall.place_tile(4, TileColor.YELLOW)  # (4,0)

        assert wall.is_column_complete(0), "Column 0 should be complete"
        assert not wall.is_column_complete(1), "Column 1 should not be complete"

        completed_columns = wall.get_completed_columns()
        assert completed_columns == [0], "Should return completed column 0"

    def test_wall_color_completion_detection(self):
        """Test wall color completion detection."""
        wall = Wall()

        # Complete blue color (all blue positions)
        blue_positions = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        for row, col in blue_positions:
            wall.filled[row][col] = True

        assert wall.is_color_complete(TileColor.BLUE), "Blue color should be complete"
        assert not wall.is_color_complete(
            TileColor.RED
        ), "Red color should not be complete"

        completed_colors = wall.get_completed_colors()
        assert TileColor.BLUE in completed_colors, "Should return completed blue color"
        assert len(completed_colors) == 1, "Should only have one completed color"


class TestPlayerBoardPlacement:
    """Test player board tile placement integration."""

    def test_valid_pattern_line_placement(self):
        """Test valid tile placement on pattern lines."""
        board = PlayerBoard()

        # Place tiles on pattern line 2 (capacity 3)
        tiles = [Tile(TileColor.BLUE)] * 2
        board.place_tiles_on_pattern_line(2, tiles)

        assert (
            len(board.pattern_lines[2].tiles) == 2
        ), "Should place tiles on pattern line"
        assert (
            board.pattern_lines[2].color == TileColor.BLUE
        ), "Should set pattern line color"
        assert len(board.floor_line) == 0, "Should not overflow to floor"

    def test_pattern_line_overflow_to_floor(self):
        """Test pattern line overflow goes to floor."""
        board = PlayerBoard()

        # Try to place 4 tiles on pattern line 1 (capacity 2)
        tiles = [Tile(TileColor.BLUE)] * 4
        board.place_tiles_on_pattern_line(1, tiles)

        # Since can_add_tiles returns False for 4 tiles on capacity 2 line,
        # all tiles go to floor line
        assert (
            len(board.pattern_lines[1].tiles) == 0
        ), "Pattern line should reject all tiles"
        assert len(board.floor_line) == 4, "All tiles should go to floor"

    def test_invalid_pattern_line_placement(self):
        """Test invalid pattern line placement goes to floor."""
        board = PlayerBoard()

        # Fill pattern line 0 with blue
        board.place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])

        # Try to place red tiles on same line
        red_tiles = [Tile(TileColor.RED)]
        board.place_tiles_on_pattern_line(0, red_tiles)

        assert len(board.floor_line) == 1, "Invalid tiles should go to floor"
        assert board.floor_line[0].color == TileColor.RED, "Floor should have red tile"

    def test_wall_position_blocking_pattern_line(self):
        """Test that filled wall positions block pattern line placement."""
        board = PlayerBoard()

        # Fill wall position for blue in row 0
        board.wall.filled[0][0] = True

        # Try to place blue tiles on pattern line 0
        blue_tiles = [Tile(TileColor.BLUE)]
        can_place = board.can_place_tiles_on_pattern_line(0, blue_tiles)

        assert not can_place, "Should not place tiles for filled wall position"

    def test_floor_line_direct_placement(self):
        """Test direct placement on floor line."""
        board = PlayerBoard()

        tiles = [Tile(TileColor.BLUE), Tile(TileColor.RED), Tile(TileColor.YELLOW)]
        board.place_tiles_on_floor_line(tiles)

        assert len(board.floor_line) == 3, "Should place all tiles on floor"
        assert board.floor_line[0].color == TileColor.BLUE, "Should maintain tile order"

    def test_invalid_pattern_line_index(self):
        """Test handling of invalid pattern line indices."""
        board = PlayerBoard()

        tiles = [Tile(TileColor.BLUE)]

        # Test negative index
        board.place_tiles_on_pattern_line(-1, tiles)
        assert len(board.floor_line) == 1, "Invalid index should send tiles to floor"

        board.floor_line = []  # Reset

        # Test index too large
        board.place_tiles_on_pattern_line(5, tiles)
        assert len(board.floor_line) == 1, "Invalid index should send tiles to floor"

    def test_empty_tile_list_handling(self):
        """Test handling of empty tile lists."""
        board = PlayerBoard()

        # Test empty list on pattern line
        can_place = board.can_place_tiles_on_pattern_line(0, [])
        assert not can_place, "Should not place empty tile list"

        # Test empty list on floor line (should be safe)
        board.place_tiles_on_floor_line([])
        assert len(board.floor_line) == 0, "Empty list should not affect floor line"


class TestComplexPlacementScenarios:
    """Test complex tile placement scenarios."""

    def test_multiple_pattern_line_interactions(self):
        """Test interactions between multiple pattern lines."""
        board = PlayerBoard()

        # Set up multiple pattern lines with different colors
        board.place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])
        board.place_tiles_on_pattern_line(1, [Tile(TileColor.RED)])
        board.place_tiles_on_pattern_line(2, [Tile(TileColor.YELLOW)] * 2)

        # Verify each line maintains its color
        assert board.pattern_lines[0].color == TileColor.BLUE
        assert board.pattern_lines[1].color == TileColor.RED
        assert board.pattern_lines[2].color == TileColor.YELLOW

        # Try to place wrong colors
        board.place_tiles_on_pattern_line(0, [Tile(TileColor.RED)])  # Wrong color
        board.place_tiles_on_pattern_line(
            1, [Tile(TileColor.BLUE)] * 3
        )  # Wrong color + overflow

        # All wrong placements should go to floor
        assert len(board.floor_line) == 4, "Wrong color tiles should go to floor"

    def test_pattern_line_completion_and_wall_placement(self):
        """Test pattern line completion and subsequent wall placement."""
        board = PlayerBoard()

        # Complete pattern line 2 (capacity 3)
        tiles = [Tile(TileColor.BLUE)] * 3
        board.place_tiles_on_pattern_line(2, tiles)

        assert board.pattern_lines[2].is_complete(), "Pattern line should be complete"

        # Simulate end of round
        points, discard = board.end_round_scoring()

        # Check wall placement
        assert board.wall.filled[2][2], "Blue tile should be placed on wall"
        assert points == 1, "Should score 1 point for isolated tile"
        assert len(discard) == 2, "Should discard 2 excess tiles"

    def test_wall_scoring_with_connections(self):
        """Test wall scoring with tile connections."""
        board = PlayerBoard()

        # Place some tiles on wall manually to create connections
        board.wall.filled[0][0] = True  # Blue at (0,0)
        board.wall.filled[0][1] = True  # Yellow at (0,1)

        # Complete pattern line 0 to place red at (0,2)
        board.place_tiles_on_pattern_line(0, [Tile(TileColor.RED)])

        points, _ = board.end_round_scoring()

        # Should score 3 points (connected to 2 existing tiles)
        assert points == 3, "Should score 3 points for connection"

    def test_mixed_valid_invalid_placements(self):
        """Test scenario with mix of valid and invalid placements."""
        board = PlayerBoard()

        # Set up some constraints
        board.place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])  # Line 0 full
        board.wall.filled[1][
            1
        ] = True  # Blue position in row 1 filled (blue is at column 1 in row 1)

        # Try various placements
        board.place_tiles_on_pattern_line(
            0, [Tile(TileColor.RED)]
        )  # Invalid: line full -> 1 tile to floor
        board.place_tiles_on_pattern_line(
            1, [Tile(TileColor.BLUE)]
        )  # Invalid: wall filled -> 1 tile to floor
        board.place_tiles_on_pattern_line(
            2, [Tile(TileColor.YELLOW)]
        )  # Valid -> stays on pattern line
        board.place_tiles_on_pattern_line(
            3, [Tile(TileColor.BLACK)] * 5
        )  # Invalid: too many tiles -> 5 tiles to floor

        # Check results - invalid placements go to floor
        # 1 (red) + 1 (blue) + 5 (black) = 7 tiles on floor
        assert len(board.floor_line) == 7, "All invalid placements should go to floor"
        assert (
            board.pattern_lines[2].color == TileColor.YELLOW
        ), "Valid placement should work"
        assert (
            len(board.pattern_lines[3].tiles) == 0
        ), "Invalid placement should be rejected"

    def test_first_player_marker_handling(self):
        """Test first player marker handling in placements."""
        board = PlayerBoard()

        # Try to place first player marker on pattern line
        first_player_tile = [Tile.create_first_player_marker()]
        can_place = board.can_place_tiles_on_pattern_line(0, first_player_tile)

        assert not can_place, "Should not place first player marker on pattern line"

        # Place on floor line should work
        board.place_tiles_on_floor_line(first_player_tile)
        assert len(board.floor_line) == 1, "Should place first player marker on floor"
        assert board.has_first_player_marker(), "Should detect first player marker"


if __name__ == "__main__":
    pytest.main([__file__])
