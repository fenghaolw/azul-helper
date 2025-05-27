#!/usr/bin/env python3
"""
Tests to verify compliance with official Azul rules.
Based on the official rulebook from Plan B Games.
"""

import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game_state import Action, create_game
from game.player_board import PlayerBoard, Wall
from game.tile import Tile, TileColor
from game.factory import Factory, CenterArea, FactoryArea


class TestOfficialRulesCompliance:
    """Test compliance with official Azul rules."""

    def test_floor_line_capacity_and_overflow(self):
        """Test floor line capacity according to official rules.
        
        Official rule: Floor line has 7 penalty positions. If all spaces are occupied,
        return any further fallen tiles to the lid of the game box.
        """
        board = PlayerBoard()
        
        # Place exactly 7 tiles on floor line
        seven_tiles = [Tile(TileColor.BLUE)] * 7
        discarded = board.place_tiles_on_floor_line(seven_tiles)
        
        assert len(board.floor_line) == 7, "Floor line should hold exactly 7 tiles"
        assert len(discarded) == 0, "No tiles should be discarded when floor line has space"
        
        # Try to place more tiles - they should be discarded
        excess_tiles = [Tile(TileColor.RED)] * 5
        discarded = board.place_tiles_on_floor_line(excess_tiles)
        
        assert len(board.floor_line) == 7, "Floor line should still hold exactly 7 tiles"
        assert len(discarded) == 5, "Excess tiles should be discarded to box"

    def test_floor_line_penalties(self):
        """Test floor line penalties according to official rules.
        
        Official rule: Floor line penalties are -1, -1, -2, -2, -2, -3, -3
        """
        board = PlayerBoard()
        expected_penalties = [-1, -1, -2, -2, -2, -3, -3]
        
        assert board.FLOOR_PENALTIES == expected_penalties, "Floor penalties should match official rules"
        
        # Test cumulative penalty calculation
        tiles = [Tile(TileColor.BLUE)] * 7
        board.place_tiles_on_floor_line(tiles)
        
        points, _ = board.end_round_scoring()
        expected_total_penalty = sum(expected_penalties)  # -14
        assert points == expected_total_penalty, f"Total penalty should be {expected_total_penalty}"

    def test_factory_count_by_player_count(self):
        """Test factory count according to official rules.
        
        Official rule: Number of factories = 2 * num_players + 1
        - 2 players: 5 factories
        - 3 players: 7 factories  
        - 4 players: 9 factories
        """
        for num_players in [2, 3, 4]:
            game = create_game(num_players=num_players)
            expected_factories = 2 * num_players + 1
            assert len(game.factory_area.factories) == expected_factories, \
                f"Should have {expected_factories} factories for {num_players} players"

    def test_tile_distribution(self):
        """Test tile distribution according to official rules.
        
        Official rule: 100 tiles total (20 of each of 5 colors)
        """
        tiles = Tile.create_standard_tiles()
        assert len(tiles) == 100, "Should have 100 tiles total"
        
        # Count tiles by color
        color_counts = {}
        for tile in tiles:
            color_counts[tile.color] = color_counts.get(tile.color, 0) + 1
        
        expected_colors = [TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE]
        for color in expected_colors:
            assert color_counts[color] == 20, f"Should have 20 {color.value} tiles"

    def test_wall_pattern_correctness(self):
        """Test wall pattern according to official rules.
        
        Official rule: Wall has specific pattern where each row and column
        contains each color exactly once (Latin square property).
        """
        wall = Wall()
        colors = [TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE]
        
        # Check each row has all colors exactly once
        for row in range(5):
            row_colors = set(wall.WALL_PATTERN[row])
            assert row_colors == set(colors), f"Row {row} should contain all colors exactly once"
        
        # Check each column has all colors exactly once
        for col in range(5):
            col_colors = set(wall.WALL_PATTERN[row][col] for row in range(5))
            assert col_colors == set(colors), f"Column {col} should contain all colors exactly once"

    def test_pattern_line_capacities(self):
        """Test pattern line capacities according to official rules.
        
        Official rule: Pattern lines have capacities 1, 2, 3, 4, 5 from top to bottom.
        """
        board = PlayerBoard()
        
        for i in range(5):
            expected_capacity = i + 1
            assert board.pattern_lines[i].capacity == expected_capacity, \
                f"Pattern line {i} should have capacity {expected_capacity}"

    def test_scoring_rules(self):
        """Test scoring rules according to official rules.
        
        Official rules:
        - Single tile: 1 point
        - Connected tiles: count all connected tiles (horizontal OR vertical)
        - If connected both ways: add horizontal + vertical counts
        """
        wall = Wall()
        
        # Single tile placement
        points = wall.place_tile(0, TileColor.BLUE)
        assert points == 1, "Single tile should score 1 point"
        
        # Horizontal connection
        points = wall.place_tile(0, TileColor.YELLOW)
        assert points == 2, "Two horizontally connected tiles should score 2 points"
        
        # Add vertical connection to create cross
        wall.place_tile(1, TileColor.WHITE)  # Below blue tile
        points = wall.place_tile(1, TileColor.BLUE)  # Creates cross connection
        assert points == 4, "Cross connection should score 4 points (2 horizontal + 2 vertical)"

    def test_game_end_condition(self):
        """Test game end condition according to official rules.
        
        Official rule: Game ends after the round in which at least one player
        has completed a horizontal line of 5 consecutive tiles.
        """
        game = create_game(num_players=2)
        
        # Complete a row for player 0
        for col in range(5):
            game.players[0].wall.filled[0][col] = True
        
        # Force round end
        game._end_round()
        
        assert game.game_over, "Game should end when player completes a row"

    def test_final_scoring_bonuses(self):
        """Test final scoring bonuses according to official rules.
        
        Official rules:
        - 2 points for each complete horizontal line
        - 7 points for each complete vertical line  
        - 10 points for each color with all 5 tiles placed
        """
        board = PlayerBoard()
        initial_score = board.score
        
        # Complete first row
        for col in range(5):
            board.wall.filled[0][col] = True
        
        # Complete first column
        for row in range(5):
            board.wall.filled[row][0] = True
        
        # Complete blue color (diagonal positions)
        blue_positions = [(0,0), (1,1), (2,2), (3,3), (4,4)]
        for row, col in blue_positions:
            board.wall.filled[row][col] = True
        
        bonus = board.final_scoring()
        expected_bonus = 2 + 7 + 10  # row + column + color
        assert bonus == expected_bonus, f"Should award {expected_bonus} bonus points"
        assert board.score == initial_score + expected_bonus, "Score should include bonuses"

    def test_first_player_marker_rules(self):
        """Test first player marker rules according to official rules.
        
        Official rules:
        - First player to take from center gets first player marker
        - First player marker goes on floor line and counts as penalty
        - Player with first player marker goes first next round
        """
        game = create_game(num_players=2)
        
        # Add tiles to center
        game.factory_area.center.add_tiles([Tile(TileColor.BLUE)])
        
        # Take from center - should get first player marker
        tiles = game.factory_area.center.take_tiles(TileColor.BLUE)
        
        # Should include first player marker
        has_first_player = any(tile.is_first_player_marker for tile in tiles)
        assert has_first_player, "Taking from center should include first player marker"
        
        # Place on floor line
        game.players[0].place_tiles_on_floor_line(tiles)
        assert game.players[0].has_first_player_marker(), "Player should have first player marker"

    def test_pattern_line_color_restrictions(self):
        """Test pattern line color restrictions according to official rules.
        
        Official rules:
        - Pattern line can only hold tiles of same color
        - Cannot place tiles if corresponding wall position is already filled
        """
        board = PlayerBoard()
        
        # Place blue tile in pattern line 0
        board.place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])
        
        # Try to place different color - should fail
        assert not board.can_place_tiles_on_pattern_line(0, [Tile(TileColor.RED)]), \
            "Cannot place different color in same pattern line"
        
        # Fill wall position for blue in row 0
        board.wall.place_tile(0, TileColor.BLUE)
        
        # Try to place blue in pattern line 0 again - should fail
        assert not board.can_place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)]), \
            "Cannot place tile if wall position already filled"

    def test_factory_tile_distribution(self):
        """Test factory tile distribution according to official rules.
        
        Official rule: Each factory display holds exactly 4 tiles.
        """
        game = create_game(num_players=2)
        
        # Check initial setup
        for i, factory in enumerate(game.factory_area.factories):
            assert len(factory.tiles) == 4, f"Factory {i} should have exactly 4 tiles"

    def test_round_end_tile_movement(self):
        """Test tile movement at round end according to official rules.
        
        Official rules:
        - Move rightmost tile from complete pattern lines to wall
        - Remove remaining tiles from complete pattern lines to discard
        - Incomplete pattern lines keep their tiles
        """
        board = PlayerBoard()
        
        # Set up complete and incomplete pattern lines
        board.place_tiles_on_pattern_line(1, [Tile(TileColor.BLUE)] * 2)  # Complete
        board.place_tiles_on_pattern_line(2, [Tile(TileColor.RED)] * 2)   # Incomplete (needs 3)
        
        # End round scoring
        points, discarded = board.end_round_scoring()
        
        # Complete line should move to wall and discard excess
        assert board.wall.filled[1][1], "Blue tile should be on wall"  # Blue goes to position (1,1)
        assert len(discarded) == 1, "Should discard 1 excess blue tile"
        
        # Incomplete line should keep tiles
        assert len(board.pattern_lines[2].tiles) == 2, "Incomplete pattern line should keep tiles"
        assert board.pattern_lines[2].color == TileColor.RED, "Pattern line should keep color"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 