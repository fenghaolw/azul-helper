#!/usr/bin/env python3
"""
Comprehensive tests for edge cases and boundary conditions.
"""

import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game_state import Action, create_game
from game.factory import Factory, CenterArea, FactoryArea
from game.player_board import PatternLine, PlayerBoard, Wall
from game.tile import Tile, TileColor


class TestBoundaryConditions:
    """Test boundary conditions and limits."""

    def test_minimum_maximum_players(self):
        """Test minimum and maximum player counts."""
        # Test minimum players
        game = create_game(num_players=2)
        assert game.num_players == 2, "Should support 2 players"
        
        # Test maximum players
        game = create_game(num_players=4)
        assert game.num_players == 4, "Should support 4 players"
        
        # Test invalid player counts
        with pytest.raises(ValueError):
            create_game(num_players=1)
        
        with pytest.raises(ValueError):
            create_game(num_players=5)

    def test_pattern_line_boundary_indices(self):
        """Test pattern line boundary indices."""
        board = PlayerBoard()
        
        # Test valid indices
        for i in range(5):
            assert board.can_place_tiles_on_pattern_line(i, [Tile(TileColor.BLUE)]), \
                f"Should accept valid index {i}"
        
        # Test invalid indices
        invalid_indices = [-1, 5, 10, -10]
        for i in invalid_indices:
            # Should not crash, but should return False
            result = board.can_place_tiles_on_pattern_line(i, [Tile(TileColor.BLUE)])
            assert not result, f"Should reject invalid index {i}"

    def test_wall_boundary_positions(self):
        """Test wall boundary positions."""
        wall = Wall()
        
        # Test valid positions
        for row in range(5):
            for color in [TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE]:
                # Should be able to place each color in each row (at correct column)
                can_place = wall.can_place_tile(row, color)
                assert isinstance(can_place, bool), f"Should return boolean for row {row}, color {color}"
        
        # Test invalid rows
        invalid_rows = [-1, 5, 10, -10]
        for row in invalid_rows:
            assert not wall.can_place_tile(row, TileColor.BLUE), f"Should reject invalid row {row}"
            assert wall.place_tile(row, TileColor.BLUE) == 0, f"Should return 0 points for invalid row {row}"

    def test_factory_area_player_count_scaling(self):
        """Test factory area scaling with player count."""
        # Test different player counts
        for num_players in [2, 3, 4]:
            factory_area = FactoryArea(num_players)
            expected_factories = 2 * num_players + 1
            assert factory_area.num_factories == expected_factories, \
                f"Should have {expected_factories} factories for {num_players} players"
            assert len(factory_area.factories) == expected_factories, \
                f"Should create {expected_factories} factory objects"

    def test_floor_line_penalty_boundaries(self):
        """Test floor line penalty boundaries."""
        board = PlayerBoard()
        
        # Test maximum penalties
        max_tiles = len(board.FLOOR_PENALTIES) + 5  # More than penalty positions
        tiles = [Tile(TileColor.BLUE)] * max_tiles
        board.place_tiles_on_floor_line(tiles)
        
        points, _ = board.end_round_scoring()
        
        # Should only apply penalties for defined positions
        expected_penalty = sum(board.FLOOR_PENALTIES)
        assert points == expected_penalty, f"Should apply maximum penalty of {expected_penalty}"

    def test_tile_count_boundaries(self):
        """Test tile count boundaries."""
        # Test standard tile creation
        tiles = Tile.create_standard_tiles()
        assert len(tiles) == 100, "Should create exactly 100 standard tiles"
        
        # Count each color
        color_counts = {}
        for tile in tiles:
            color_counts[tile.color] = color_counts.get(tile.color, 0) + 1
        
        # Should have exactly 20 of each color
        expected_colors = [TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE]
        for color in expected_colors:
            assert color_counts[color] == 20, f"Should have exactly 20 {color} tiles"


class TestEmptyStateHandling:
    """Test handling of empty states and collections."""

    def test_empty_factory_handling(self):
        """Test handling of empty factories."""
        factory = Factory()
        
        # Empty factory should have no tiles
        assert factory.is_empty(), "New factory should be empty"
        assert len(factory.get_available_colors()) == 0, "Empty factory should have no colors"
        assert not factory.has_color(TileColor.BLUE), "Empty factory should not have any color"
        
        # Taking from empty factory should return empty lists
        taken, remaining = factory.take_tiles(TileColor.BLUE)
        assert len(taken) == 0, "Should take no tiles from empty factory"
        assert len(remaining) == 0, "Should have no remaining tiles in empty factory"

    def test_empty_center_handling(self):
        """Test handling of empty center area."""
        center = CenterArea()
        
        # Empty center should behave correctly
        assert center.is_empty(), "New center should be empty"
        assert len(center.get_available_colors()) == 0, "Empty center should have no colors"
        assert not center.has_color(TileColor.BLUE), "Empty center should not have any color"
        
        # Taking from empty center should return empty list
        taken = center.take_tiles(TileColor.BLUE)
        assert len(taken) == 0, "Should take no tiles from empty center"

    def test_empty_pattern_line_handling(self):
        """Test handling of empty pattern lines."""
        line = PatternLine(3)
        
        # Empty line should behave correctly
        assert not line.is_complete(), "Empty line should not be complete"
        assert line.color is None, "Empty line should have no color"
        assert len(line.tiles) == 0, "Empty line should have no tiles"
        
        # Clearing empty line should return nothing
        wall_tile, discard_tiles = line.clear()
        assert wall_tile is None, "Empty line should not provide wall tile"
        assert len(discard_tiles) == 0, "Empty line should not provide discard tiles"

    def test_empty_tile_list_operations(self):
        """Test operations with empty tile lists."""
        board = PlayerBoard()
        
        # Empty tile list should be handled gracefully
        assert not board.can_place_tiles_on_pattern_line(0, []), "Should not place empty tile list"
        
        # Placing empty list should not change state
        board.place_tiles_on_floor_line([])
        assert len(board.floor_line) == 0, "Empty list should not affect floor line"

    def test_empty_bag_and_discard_handling(self):
        """Test handling when both bag and discard pile are empty."""
        game = create_game(num_players=2, seed=42)
        
        # Empty both bag and discard
        game.bag = []
        game.discard_pile = []
        
        # Should handle gracefully
        game._start_new_round()
        
        # All factories should be empty
        for factory in game.factory_area.factories:
            assert factory.is_empty(), "Factory should be empty when no tiles available"


class TestInvalidInputHandling:
    """Test handling of invalid inputs and malformed data."""

    def test_invalid_action_parameters(self):
        """Test handling of invalid action parameters."""
        game = create_game(num_players=2, seed=42)
        
        # Test invalid actions
        invalid_actions = [
            Action(-2, TileColor.BLUE, 0),  # Invalid source
            Action(100, TileColor.BLUE, 0),  # Non-existent factory
            Action(0, TileColor.BLUE, 10),  # Invalid destination
            Action(0, TileColor.FIRST_PLAYER, 0),  # First player marker as color
        ]
        
        for action in invalid_actions:
            assert not game.is_action_legal(action), f"Action {action} should be invalid"
            assert not game.apply_action(action), f"Action {action} should not apply"

    def test_invalid_tile_colors_in_pattern_lines(self):
        """Test handling of invalid tile colors."""
        line = PatternLine(3)
        
        # First player marker should be rejected
        first_player_tiles = [Tile.create_first_player_marker()]
        assert not line.can_add_tiles(first_player_tiles), "Should reject first player marker"
        
        # Adding first player marker should return all tiles as overflow
        overflow = line.add_tiles(first_player_tiles)
        assert len(overflow) == 1, "Should return first player marker as overflow"
        assert overflow[0].is_first_player_marker, "Overflow should be first player marker"

    def test_malformed_wall_operations(self):
        """Test wall operations with malformed inputs."""
        wall = Wall()
        
        # Test with None color (should handle gracefully)
        try:
            result = wall.can_place_tile(0, None)
            # Should either return False or raise appropriate exception
            assert isinstance(result, bool), "Should return boolean or raise exception"
        except (TypeError, AttributeError):
            # Acceptable to raise exception for None input
            pass

    def test_negative_tile_counts(self):
        """Test handling of negative or invalid tile counts."""
        board = PlayerBoard()
        
        # Test with negative score
        board.score = -10
        points, _ = board.end_round_scoring()
        assert board.score == 0, "Score should not go below 0"

    def test_invalid_player_indices(self):
        """Test handling of invalid player indices."""
        game = create_game(num_players=2, seed=42)
        
        # Test invalid player indices
        invalid_indices = [-1, 2, 10]
        for player_id in invalid_indices:
            actions = game.get_legal_actions(player_id)
            # Should handle gracefully (return empty list or current player's actions)
            assert isinstance(actions, list), f"Should return list for invalid player {player_id}"


class TestConcurrencyAndStateConsistency:
    """Test state consistency and potential concurrency issues."""

    def test_game_state_copying_deep_copy(self):
        """Test that game state copying creates true deep copies."""
        game = create_game(num_players=2, seed=42)
        
        # Modify original game
        game.players[0].score = 50
        game.players[0].place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])
        
        # Create copy
        game_copy = game.copy()
        
        # Modify original after copying
        game.players[0].score = 100
        game.players[0].place_tiles_on_pattern_line(1, [Tile(TileColor.RED)])
        
        # Copy should not be affected
        assert game_copy.players[0].score == 50, "Copy should not be affected by original changes"
        assert len(game_copy.players[0].pattern_lines[1].tiles) == 0, "Copy should not be affected"

    def test_action_application_atomicity(self):
        """Test that action application is atomic (all or nothing)."""
        game = create_game(num_players=2, seed=42)
        
        # Get initial state
        initial_state = {
            'current_player': game.current_player,
            'round_number': game.round_number,
            'factory_tiles': [len(f.tiles) for f in game.factory_area.factories],
            'center_tiles': len(game.factory_area.center.tiles)
        }
        
        # Try to apply invalid action
        invalid_action = Action(-2, TileColor.BLUE, 0)
        success = game.apply_action(invalid_action)
        
        # State should be unchanged
        assert not success, "Invalid action should not succeed"
        assert game.current_player == initial_state['current_player'], "Current player should not change"
        assert game.round_number == initial_state['round_number'], "Round should not change"

    def test_simultaneous_pattern_line_modifications(self):
        """Test consistency when modifying multiple pattern lines."""
        board = PlayerBoard()
        
        # Modify multiple pattern lines
        board.place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])
        board.place_tiles_on_pattern_line(1, [Tile(TileColor.RED)] * 2)
        board.place_tiles_on_pattern_line(2, [Tile(TileColor.YELLOW)] * 3)
        
        # Each line should maintain its own state
        assert board.pattern_lines[0].color == TileColor.BLUE
        assert board.pattern_lines[1].color == TileColor.RED
        assert board.pattern_lines[2].color == TileColor.YELLOW
        
        assert len(board.pattern_lines[0].tiles) == 1
        assert len(board.pattern_lines[1].tiles) == 2
        assert len(board.pattern_lines[2].tiles) == 3


class TestResourceExhaustion:
    """Test behavior under resource exhaustion scenarios."""

    def test_bag_exhaustion_handling(self):
        """Test handling when bag is exhausted."""
        game = create_game(num_players=2, seed=42)
        
        # Exhaust the bag
        game.bag = []
        game.discard_pile = []
        
        # Try to start new round
        game._start_new_round()
        
        # Should handle gracefully
        assert all(factory.is_empty() for factory in game.factory_area.factories), \
            "All factories should be empty when bag exhausted"

    def test_maximum_floor_line_tiles(self):
        """Test behavior with maximum floor line tiles."""
        board = PlayerBoard()
        
        # Add many tiles to floor line (more than capacity)
        many_tiles = [Tile(TileColor.BLUE)] * 20
        discarded = board.place_tiles_on_floor_line(many_tiles)
        
        # Should handle gracefully according to official rules
        assert len(board.floor_line) == 7, "Floor line should hold maximum 7 tiles"
        assert len(discarded) == 13, "Excess tiles should be discarded to box"
        
        # Scoring should only apply defined penalties
        points, _ = board.end_round_scoring()
        expected_penalty = sum(board.FLOOR_PENALTIES)
        assert points == expected_penalty, "Should only apply defined penalties"

    def test_maximum_wall_completion(self):
        """Test behavior when wall is completely filled."""
        wall = Wall()
        
        # Fill entire wall
        for row in range(5):
            for col in range(5):
                wall.filled[row][col] = True
        
        # All completion checks should return True
        assert len(wall.get_completed_rows()) == 5, "All rows should be complete"
        assert len(wall.get_completed_columns()) == 5, "All columns should be complete"
        assert len(wall.get_completed_colors()) == 5, "All colors should be complete"
        
        # Should not be able to place any more tiles
        for color in [TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE]:
            for row in range(5):
                assert not wall.can_place_tile(row, color), f"Should not place {color} in row {row}"

    def test_extreme_game_length(self):
        """Test behavior in extremely long games."""
        game = create_game(num_players=2, seed=42)
        
        # Simulate many rounds without game end
        max_rounds = 50
        for round_num in range(max_rounds):
            if game.game_over:
                break
                
            # Prevent game end by not completing any rows
            for player in game.players:
                # Clear any completed rows to prevent game end
                for row in range(5):
                    if any(player.wall.filled[row]):
                        for col in range(5):
                            player.wall.filled[row][col] = False
            
            # Force round end
            game._end_round()
        
        # Game should handle many rounds gracefully
        assert game.round_number <= max_rounds + 1, "Should handle many rounds"


class TestDataIntegrity:
    """Test data integrity and consistency checks."""

    def test_tile_conservation(self):
        """Test that tiles are conserved throughout the game."""
        game = create_game(num_players=2, seed=42)
        
        # Count initial tiles (excluding first player marker)
        initial_total = len(game.bag) + len(game.discard_pile)
        
        # Add tiles in factories and center (excluding first player marker)
        for factory in game.factory_area.factories:
            initial_total += len(factory.tiles)
        # Center tiles (excluding first player marker)
        initial_total += len([tile for tile in game.factory_area.center.tiles if not tile.is_first_player_marker])
        
        # Add tiles on player boards (excluding first player marker)
        for player in game.players:
            for pattern_line in player.pattern_lines:
                initial_total += len(pattern_line.tiles)
            # Floor line tiles (excluding first player marker)
            initial_total += len([tile for tile in player.floor_line if not tile.is_first_player_marker])
            # Count tiles on wall
            for row in range(5):
                for col in range(5):
                    if player.wall.filled[row][col]:
                        initial_total += 1
        
        # Play some actions
        for _ in range(10):
            actions = game.get_legal_actions()
            if actions and not game.game_over:
                # Prefer pattern line actions for better game progression
                pattern_actions = [a for a in actions if a.destination >= 0]
                if pattern_actions:
                    game.apply_action(pattern_actions[0])
                else:
                    game.apply_action(actions[0])
            else:
                break
        
        # Count tiles after actions (excluding first player marker)
        final_total = len(game.bag) + len(game.discard_pile)
        
        for factory in game.factory_area.factories:
            final_total += len(factory.tiles)
        # Center tiles (excluding first player marker)
        final_total += len([tile for tile in game.factory_area.center.tiles if not tile.is_first_player_marker])
        
        for player in game.players:
            for pattern_line in player.pattern_lines:
                final_total += len(pattern_line.tiles)
            # Floor line tiles (excluding first player marker)
            final_total += len([tile for tile in player.floor_line if not tile.is_first_player_marker])
            for row in range(5):
                for col in range(5):
                    if player.wall.filled[row][col]:
                        final_total += 1
        
        # Tiles should be conserved (excluding first player marker)
        assert final_total == initial_total, f"Tiles should be conserved throughout game. Initial: {initial_total}, Final: {final_total}"

    def test_score_consistency(self):
        """Test that scores are consistent and non-negative."""
        game = create_game(num_players=2, seed=42)
        
        # Play some actions
        for _ in range(20):
            actions = game.get_legal_actions()
            if actions and not game.game_over:
                game.apply_action(actions[0])
            else:
                break
        
        # All scores should be non-negative
        for i, player in enumerate(game.players):
            assert player.score >= 0, f"Player {i} score should be non-negative"

    def test_wall_pattern_integrity(self):
        """Test that wall pattern maintains integrity."""
        wall = Wall()
        
        # Verify wall pattern is a valid Latin square
        colors = [TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE]
        
        # Each row should have each color exactly once
        for row in range(5):
            row_colors = set(wall.WALL_PATTERN[row])
            assert row_colors == set(colors), f"Row {row} should have all colors exactly once"
        
        # Each column should have each color exactly once
        for col in range(5):
            col_colors = set(wall.WALL_PATTERN[row][col] for row in range(5))
            assert col_colors == set(colors), f"Column {col} should have all colors exactly once"


if __name__ == "__main__":
    pytest.main([__file__]) 