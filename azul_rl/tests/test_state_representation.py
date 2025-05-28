"""
Tests for the Azul game state numerical representation.

This module tests the complete numerical representation of the Azul game state
and validates that all components are correctly encoded.
"""

import numpy as np
import pytest
from typing import Dict

from azul_rl.game import (
    GameState,
    AzulStateRepresentation,
    ColorIndex,
    StateConfig,
    create_state_representation,
    get_state_documentation,
    TileColor,
)


class TestStateRepresentation:
    """Test the numerical state representation."""
    
    def test_state_creation(self):
        """Test basic state representation creation."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)
        
        assert state_repr.num_players == 2
        assert state_repr.num_factories == 5
        assert state_repr.current_player == game.current_player
        assert state_repr.round_number == game.round_number
    
    def test_state_shapes(self):
        """Test that all state arrays have correct shapes."""
        game = GameState(num_players=3, seed=42)
        state_repr = AzulStateRepresentation(game)
        
        expected_shapes = {
            'current_player': (1,),
            'round_number': (1,),
            'game_over': (1,),
            'winner': (1,),
            'player_scores': (StateConfig.MAX_PLAYERS,),
            'pattern_lines': (StateConfig.MAX_PLAYERS, StateConfig.PATTERN_LINES, 8),
            'walls': (StateConfig.MAX_PLAYERS, StateConfig.WALL_SIZE, StateConfig.WALL_SIZE),
            'floor_lines': (StateConfig.MAX_PLAYERS, StateConfig.FLOOR_LINE_SIZE, 7),
            'first_player_markers': (StateConfig.MAX_PLAYERS,),
            'factories': (StateConfig.MAX_FACTORIES, StateConfig.TILES_PER_FACTORY, 6),
            'center_tiles': (StateConfig.MAX_CENTER_TILES, 6),
            'center_first_player_marker': (1,),
            'tile_supply': (2, StateConfig.NUM_COLORS),
        }
        
        actual_shapes = state_repr.state_shape
        
        for key, expected_shape in expected_shapes.items():
            assert key in actual_shapes
            assert actual_shapes[key] == expected_shape
    
    def test_flat_state_vector(self):
        """Test flattened state vector generation."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)
        
        # Test normalized vector
        flat_vector = state_repr.get_flat_state_vector(normalize=True)
        assert isinstance(flat_vector, np.ndarray)
        assert flat_vector.dtype == np.float32
        assert len(flat_vector) == state_repr.flat_state_size
        
        # Check normalization (most values should be in [0, 1])
        assert np.all(flat_vector >= -0.1)  # Allow small negative values for some components
        assert np.all(flat_vector <= 1.1)   # Allow small overflow for some components
        
        # Test unnormalized vector
        flat_vector_unnorm = state_repr.get_flat_state_vector(normalize=False)
        assert isinstance(flat_vector_unnorm, np.ndarray)
        assert len(flat_vector_unnorm) == state_repr.flat_state_size
    
    def test_state_dict(self):
        """Test state dictionary generation."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)
        
        state_dict = state_repr.get_state_dict()
        
        # Check all expected keys are present
        expected_keys = {
            'current_player', 'round_number', 'game_over', 'winner',
            'player_scores', 'pattern_lines', 'walls', 'floor_lines',
            'first_player_markers', 'factories', 'center_tiles',
            'center_first_player_marker', 'tile_supply'
        }
        
        assert set(state_dict.keys()) == expected_keys
        
        # Check types
        for key, array in state_dict.items():
            assert isinstance(array, np.ndarray)
    
    def test_player_view(self):
        """Test player-specific state views."""
        game = GameState(num_players=3, seed=42)
        state_repr = AzulStateRepresentation(game)
        
        # Test player view without hidden information
        player_view = state_repr.get_player_view(player_id=1, include_hidden=False)
        
        # Check that bag contents are hidden (only total shown)
        bag_row = player_view['tile_supply'][0]
        assert bag_row[0] > 0  # Total count
        assert np.sum(bag_row[1:]) == 0  # Individual colors hidden
        
        # Test player view with hidden information
        player_view_full = state_repr.get_player_view(player_id=1, include_hidden=True)
        
        # Check that bag contents are visible
        bag_row_full = player_view_full['tile_supply'][0]
        assert np.sum(bag_row_full) > 0  # Should have color distribution
    
    def test_color_encoding(self):
        """Test color to index conversion."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)
        
        # Test color mappings
        assert state_repr._color_to_index(TileColor.BLUE) == ColorIndex.BLUE
        assert state_repr._color_to_index(TileColor.YELLOW) == ColorIndex.YELLOW
        assert state_repr._color_to_index(TileColor.RED) == ColorIndex.RED
        assert state_repr._color_to_index(TileColor.BLACK) == ColorIndex.BLACK
        assert state_repr._color_to_index(TileColor.WHITE) == ColorIndex.WHITE
        assert state_repr._color_to_index(TileColor.FIRST_PLAYER) == ColorIndex.FIRST_PLAYER
    
    def test_pattern_line_encoding(self):
        """Test pattern line encoding."""
        game = GameState(num_players=2, seed=42)
        
        # Make a move to put tiles in pattern lines
        actions = game.get_legal_actions()
        if actions:
            game.apply_action(actions[0])
        
        state_repr = AzulStateRepresentation(game)
        
        # Check pattern line encoding structure
        pattern_lines = state_repr.pattern_lines
        
        for player_idx in range(game.num_players):
            for line_idx in range(5):
                # Capacity should be normalized
                capacity = pattern_lines[player_idx, line_idx, 0]
                assert 0.2 <= capacity <= 1.0  # (1/5) to (5/5)
                
                # Count should be normalized by capacity
                count = pattern_lines[player_idx, line_idx, 1]
                assert 0.0 <= count <= 1.0
                
                # Color encoding should be one-hot or empty
                color_encoding = pattern_lines[player_idx, line_idx, 2:7]
                assert np.sum(color_encoding) <= 1.0  # At most one color
    
    def test_wall_encoding(self):
        """Test wall encoding."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)
        
        # Wall should be binary
        walls = state_repr.walls
        assert walls.dtype == np.int8
        assert np.all((walls == 0) | (walls == 1))
        
        # Initially should be empty
        assert np.all(walls == 0)
    
    def test_factory_encoding(self):
        """Test factory encoding."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)
        
        factories = state_repr.factories
        
        # Check that factories have tiles initially
        for factory_idx in range(state_repr.num_factories):
            factory_tiles = factories[factory_idx]
            
            # Count tiles in this factory
            tile_count = np.sum(factory_tiles[:, 0])
            assert tile_count == 4  # Each factory should have 4 tiles initially
            
            # Check color encoding
            for tile_idx in range(4):
                if factory_tiles[tile_idx, 0] == 1:  # Has tile
                    color_encoding = factory_tiles[tile_idx, 1:6]
                    assert np.sum(color_encoding) == 1  # Exactly one color
    
    def test_tile_supply_encoding(self):
        """Test tile supply encoding."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)
        
        # Use the helper method to get tile distribution
        distribution = state_repr.get_tile_distribution()
        
        # Validate total tile count
        assert distribution['total'] == distribution['expected_total'], \
            f"Expected {distribution['expected_total']}, got {distribution['total']}"
        
        # Initially, most tiles should be in bag, some on factories, none elsewhere
        assert distribution['bag'] > 0  # Should have tiles in bag
        assert distribution['discard'] == 0  # No discarded tiles initially
        assert distribution['factories'] == state_repr.num_factories * 4  # Each factory has 4 tiles initially
        assert distribution['center'] == 0  # Center should be empty initially
        assert distribution['player_boards'] == 0  # No tiles on player boards initially
        assert distribution['walls'] == 0  # No tiles on walls initially
        
        print(f"Tile distribution: {distribution}")
        
        # Test after making some moves
        for _ in range(3):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])
        
        # Get new distribution after moves
        new_state_repr = AzulStateRepresentation(game)
        new_distribution = new_state_repr.get_tile_distribution()
        
        # Total should still be conserved
        assert new_distribution['total'] == new_distribution['expected_total'], \
            f"After moves: Expected {new_distribution['expected_total']}, got {new_distribution['total']}"
        
        print(f"After moves: {new_distribution}")
    
    def test_integration_with_game_state(self):
        """Test integration with GameState class."""
        game = GameState(num_players=2, seed=42)
        
        # Test get_numerical_state method
        state_repr = game.get_numerical_state()
        assert isinstance(state_repr, AzulStateRepresentation)
        
        # Test get_state_vector method (should return list)
        state_vector = game.get_state_vector()
        assert isinstance(state_vector, list)
        assert len(state_vector) == state_repr.flat_state_size
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        game = GameState(num_players=2, seed=42)
        
        # Test create_state_representation function
        state_repr = create_state_representation(game)
        assert isinstance(state_repr, AzulStateRepresentation)
        
        # Test documentation function
        doc = get_state_documentation()
        assert isinstance(doc, str)
        assert "AZUL GAME STATE NUMERICAL REPRESENTATION" in doc
        assert "COLOR ENCODING" in doc
        assert "USAGE EXAMPLES" in doc
    
    def test_state_consistency_after_moves(self):
        """Test that state representation remains consistent after game moves."""
        game = GameState(num_players=2, seed=42)
        
        # Make several moves
        for _ in range(5):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])
            else:
                break
        
        state_repr = AzulStateRepresentation(game)
        
        # Basic consistency checks
        assert state_repr.current_player == game.current_player
        assert state_repr.round_number == game.round_number
        assert state_repr.game_over == int(game.game_over)
        
        # Check that scores match
        for i, player in enumerate(game.players):
            if i < len(state_repr.player_scores):
                assert state_repr.player_scores[i] == player.score
    
    def test_state_size_calculation(self):
        """Test state size calculation."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)
        
        # Calculate expected size manually
        expected_size = (
            4 +  # Global state
            StateConfig.MAX_PLAYERS +  # Player scores
            StateConfig.MAX_PLAYERS * StateConfig.PATTERN_LINES * 8 +  # Pattern lines
            StateConfig.MAX_PLAYERS * StateConfig.WALL_SIZE * StateConfig.WALL_SIZE +  # Walls
            StateConfig.MAX_PLAYERS * StateConfig.FLOOR_LINE_SIZE * 7 +  # Floor lines (updated)
            StateConfig.MAX_PLAYERS +  # First player markers
            StateConfig.MAX_FACTORIES * StateConfig.TILES_PER_FACTORY * 6 +  # Factories
            StateConfig.MAX_CENTER_TILES * 6 +  # Center tiles
            1 +  # Center first player marker
            2 * StateConfig.NUM_COLORS  # Tile supply
        )
        
        assert state_repr.flat_state_size == expected_size
        
        # Verify with actual flattened vector
        flat_vector = state_repr.get_flat_state_vector()
        assert len(flat_vector) == expected_size
    
    def test_color_representation_accuracy(self):
        """Test that color encoding accurately represents the actual game state colors."""
        game = GameState(num_players=2, seed=42)
        
        # Make strategic moves to get tiles in pattern lines, not just floor
        moves_made = 0
        max_moves = 10
        
        while moves_made < max_moves and not game.game_over:
            actions = game.get_legal_actions()
            if not actions:
                break
            
            # Try to find an action that places tiles on a pattern line
            pattern_line_action = None
            for action in actions:
                if action.destination >= 0:  # Pattern line (not floor)
                    pattern_line_action = action
                    break
            
            # Use pattern line action if available, otherwise use first action
            chosen_action = pattern_line_action if pattern_line_action else actions[0]
            game.apply_action(chosen_action)
            moves_made += 1
            
            # Stop if we have some tiles in pattern lines
            if moves_made >= 3:
                has_pattern_tiles = False
                for player in game.players:
                    for pattern_line in player.pattern_lines:
                        if len(pattern_line.tiles) > 0:
                            has_pattern_tiles = True
                            break
                    if has_pattern_tiles:
                        break
                if has_pattern_tiles:
                    break
        
        state_repr = AzulStateRepresentation(game)
        
        # Test factory color encoding
        print("\nValidating factory colors...")
        for factory_idx, factory in enumerate(game.factory_area.factories):
            if factory_idx < state_repr.num_factories:
                encoded_factory = state_repr.factories[factory_idx]
                
                for tile_idx, tile in enumerate(factory.tiles):
                    if tile_idx < 4:  # Max 4 tiles per factory
                        # Check if tile presence is correctly encoded
                        has_tile_encoded = encoded_factory[tile_idx, 0]
                        assert has_tile_encoded == 1, f"Factory {factory_idx}, tile {tile_idx} should be marked as present"
                        
                        # Check if color is correctly encoded
                        expected_color_idx = state_repr._color_to_index(tile.color)
                        if expected_color_idx < 5:  # Regular colors only
                            color_encoding = encoded_factory[tile_idx, 1:6]
                            assert color_encoding[expected_color_idx] == 1, \
                                f"Factory {factory_idx}, tile {tile_idx}: expected color {tile.color} (idx {expected_color_idx}) to be encoded"
                            assert color_encoding.sum() == 1, \
                                f"Factory {factory_idx}, tile {tile_idx}: exactly one color should be encoded"
        
        # Test center area color encoding
        print("Validating center colors...")
        center_tiles = game.factory_area.center.tiles
        for tile_idx, tile in enumerate(center_tiles):
            if tile_idx < len(state_repr.center_tiles) and tile.color != TileColor.FIRST_PLAYER:
                encoded_tile = state_repr.center_tiles[tile_idx]
                
                # Check tile presence
                assert encoded_tile[0] == 1, f"Center tile {tile_idx} should be marked as present"
                
                # Check color encoding
                expected_color_idx = state_repr._color_to_index(tile.color)
                if expected_color_idx < 5:
                    color_encoding = encoded_tile[1:6]
                    assert color_encoding[expected_color_idx] == 1, \
                        f"Center tile {tile_idx}: expected color {tile.color} (idx {expected_color_idx}) to be encoded"
                    assert color_encoding.sum() == 1, \
                        f"Center tile {tile_idx}: exactly one color should be encoded"
        
        # Test player board color encoding
        print("Validating player board colors...")
        pattern_lines_with_tiles = 0
        
        for player_idx, player in enumerate(game.players):
            if player_idx < state_repr.num_players:
                
                # Test pattern lines
                for line_idx, pattern_line in enumerate(player.pattern_lines):
                    encoded_line = state_repr.pattern_lines[player_idx, line_idx]
                    
                    if pattern_line.color is not None:
                        pattern_lines_with_tiles += 1
                        expected_color_idx = state_repr._color_to_index(pattern_line.color)
                        if expected_color_idx < 5:
                            color_encoding = encoded_line[2:7]  # Colors are at indices 2-6
                            assert color_encoding[expected_color_idx] == 1, \
                                f"Player {player_idx}, pattern line {line_idx}: expected color {pattern_line.color} to be encoded"
                            assert encoded_line[7] == 0, \
                                f"Player {player_idx}, pattern line {line_idx}: empty indicator should be 0 when color is present"
                            
                            # Verify tile count encoding
                            expected_count = len(pattern_line.tiles) / (line_idx + 1)  # Normalized
                            actual_count = encoded_line[1]
                            assert abs(expected_count - actual_count) < 0.01, \
                                f"Player {player_idx}, pattern line {line_idx}: tile count mismatch"
                            
                            print(f"  ✓ Player {player_idx}, line {line_idx}: {pattern_line.color.value} with {len(pattern_line.tiles)} tiles")
                    else:
                        # Empty line should have empty indicator set
                        assert encoded_line[7] == 1, \
                            f"Player {player_idx}, pattern line {line_idx}: empty indicator should be 1 when no color"
                
                # Test floor line colors
                for pos, tile in enumerate(player.floor_line):
                    if pos < len(state_repr.floor_lines[player_idx]):
                        encoded_pos = state_repr.floor_lines[player_idx, pos]
                        
                        # Check tile presence
                        assert encoded_pos[0] == 1, f"Player {player_idx}, floor position {pos} should be marked as occupied"
                        
                        # Check color encoding
                        if tile.color == TileColor.FIRST_PLAYER:
                            assert encoded_pos[6] == 1, \
                                f"Player {player_idx}, floor position {pos}: first player marker should be encoded"
                            assert encoded_pos[1:6].sum() == 0, \
                                f"Player {player_idx}, floor position {pos}: no regular color should be encoded for first player marker"
                        else:
                            expected_color_idx = state_repr._color_to_index(tile.color)
                            if expected_color_idx < 5:
                                color_encoding = encoded_pos[1:6]
                                assert color_encoding[expected_color_idx] == 1, \
                                    f"Player {player_idx}, floor position {pos}: expected color {tile.color} to be encoded"
                                assert color_encoding.sum() == 1, \
                                    f"Player {player_idx}, floor position {pos}: exactly one color should be encoded"
                                assert encoded_pos[6] == 0, \
                                    f"Player {player_idx}, floor position {pos}: first player marker should not be set for regular tile"
        
        print(f"Pattern lines with tiles found: {pattern_lines_with_tiles}")
        assert pattern_lines_with_tiles > 0, "Test should have at least some tiles in pattern lines for comprehensive validation"
        print("All color representations validated successfully!")
    
    def test_tile_supply_color_accuracy(self):
        """Test that tile supply counts match actual tile colors in bag and discard."""
        game = GameState(num_players=2, seed=42)
        
        # Make some moves to get tiles in discard pile
        for _ in range(10):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])
            else:
                break
        
        state_repr = AzulStateRepresentation(game)
        
        # Count actual tiles by color in bag
        actual_bag_counts = {color: 0 for color in [TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE]}
        for tile in game.bag:
            if tile.color in actual_bag_counts:
                actual_bag_counts[tile.color] += 1
        
        # Count actual tiles by color in discard pile
        actual_discard_counts = {color: 0 for color in [TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE]}
        for tile in game.discard_pile:
            if tile.color in actual_discard_counts:
                actual_discard_counts[tile.color] += 1
        
        # Compare with encoded counts
        color_names = [TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE]
        
        print(f"\nBag tile validation:")
        for color_idx, color in enumerate(color_names):
            encoded_count = state_repr.tile_supply[0, color_idx]
            actual_count = actual_bag_counts[color]
            print(f"  {color.value}: encoded={encoded_count}, actual={actual_count}")
            assert encoded_count == actual_count, \
                f"Bag count mismatch for {color.value}: encoded={encoded_count}, actual={actual_count}"
        
        print(f"\nDiscard pile validation:")
        for color_idx, color in enumerate(color_names):
            encoded_count = state_repr.tile_supply[1, color_idx]
            actual_count = actual_discard_counts[color]
            print(f"  {color.value}: encoded={encoded_count}, actual={actual_count}")
            assert encoded_count == actual_count, \
                f"Discard count mismatch for {color.value}: encoded={encoded_count}, actual={actual_count}"
        
        print("Tile supply color counts validated successfully!")
    
    def test_wall_pattern_accuracy(self):
        """Test that wall encoding matches the actual wall pattern and placements."""
        game = GameState(num_players=2, seed=42)
        
        # Play a longer game to get some wall placements
        for _ in range(20):
            actions = game.get_legal_actions()
            if actions and not game.game_over:
                game.apply_action(actions[0])
            else:
                break
        
        state_repr = AzulStateRepresentation(game)
        
        print(f"\nWall validation:")
        for player_idx, player in enumerate(game.players):
            if player_idx < state_repr.num_players:
                print(f"  Player {player_idx}:")
                
                for row in range(5):
                    for col in range(5):
                        actual_filled = player.wall.filled[row][col]
                        encoded_filled = state_repr.walls[player_idx, row, col]
                        
                        assert actual_filled == bool(encoded_filled), \
                            f"Player {player_idx}, wall position ({row},{col}): actual={actual_filled}, encoded={bool(encoded_filled)}"
                
                # Count total wall tiles
                actual_wall_tiles = sum(sum(row) for row in player.wall.filled)
                encoded_wall_tiles = int(state_repr.walls[player_idx].sum())
                print(f"    Wall tiles: actual={actual_wall_tiles}, encoded={encoded_wall_tiles}")
                
                assert actual_wall_tiles == encoded_wall_tiles, \
                    f"Player {player_idx}: wall tile count mismatch"
        
        print("Wall representations validated successfully!")


def test_state_config():
    """Test StateConfig constants."""
    assert StateConfig.MAX_PLAYERS == 4
    assert StateConfig.NUM_COLORS == 5
    assert StateConfig.WALL_SIZE == 5
    assert StateConfig.PATTERN_LINES == 5
    assert StateConfig.FLOOR_LINE_SIZE == 7
    assert StateConfig.TILES_PER_COLOR == 20
    assert StateConfig.TOTAL_TILES == 100


def test_color_index_enum():
    """Test ColorIndex enum."""
    assert ColorIndex.BLUE == 0
    assert ColorIndex.YELLOW == 1
    assert ColorIndex.RED == 2
    assert ColorIndex.BLACK == 3
    assert ColorIndex.WHITE == 4
    assert ColorIndex.FIRST_PLAYER == 5
    assert ColorIndex.EMPTY == 6


if __name__ == "__main__":
    # Run a simple demonstration
    print("=== Azul State Representation Demo ===")
    
    # Create a game
    game = GameState(num_players=2, seed=42)
    print(f"Created game with {game.num_players} players")
    
    # Create state representation
    state_repr = AzulStateRepresentation(game)
    print(f"State representation created")
    print(f"Flat state size: {state_repr.flat_state_size}")
    
    # Show state shapes
    print("\nState component shapes:")
    for name, shape in state_repr.state_shape.items():
        print(f"  {name}: {shape}")
    
    # Get flattened vector
    flat_vector = state_repr.get_flat_state_vector(normalize=True)
    print(f"\nFlattened vector shape: {flat_vector.shape}")
    print(f"Value range: [{flat_vector.min():.3f}, {flat_vector.max():.3f}]")
    
    # Make some moves and show state changes
    print("\nMaking moves...")
    for i in range(3):
        actions = game.get_legal_actions()
        if actions:
            action = actions[0]
            game.apply_action(action)
            print(f"  Move {i+1}: {action}")
            
            new_state_repr = AzulStateRepresentation(game)
            new_flat_vector = new_state_repr.get_flat_state_vector(normalize=True)
            
            # Calculate difference
            diff = np.sum(np.abs(flat_vector - new_flat_vector))
            print(f"    State difference: {diff:.3f}")
            flat_vector = new_flat_vector
    
    print("\nDemo completed successfully!") 