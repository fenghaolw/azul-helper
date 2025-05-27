"""
Azul Game State Numerical Representation

This module defines the complete numerical representation of the Azul game state
using NumPy arrays for efficient computation and machine learning applications.

All game components are represented numerically with clear indexing schemes
and normalization strategies for neural network compatibility.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from enum import IntEnum

from .tile import TileColor
from .game_state import GameState


class ColorIndex(IntEnum):
    """Numerical indices for tile colors."""
    BLUE = 0
    YELLOW = 1
    RED = 2
    BLACK = 3
    WHITE = 4
    FIRST_PLAYER = 5
    EMPTY = 6  # For empty positions


class StateConfig:
    """Configuration constants for state representation."""
    
    # Game constants
    MAX_PLAYERS = 4
    NUM_COLORS = 5  # Excluding first player marker
    NUM_FACTORIES_2P = 5
    NUM_FACTORIES_3P = 7
    NUM_FACTORIES_4P = 9
    MAX_FACTORIES = NUM_FACTORIES_4P
    
    # Board dimensions
    WALL_SIZE = 5
    PATTERN_LINES = 5
    FLOOR_LINE_SIZE = 7
    
    # Tile counts
    TILES_PER_COLOR = 20
    TOTAL_TILES = NUM_COLORS * TILES_PER_COLOR
    TILES_PER_FACTORY = 4
    
    # Normalization constants
    MAX_SCORE = 200  # Conservative estimate for score normalization
    MAX_ROUND = 15   # Conservative estimate for round normalization
    MAX_CENTER_TILES = 40  # Conservative estimate for center area


class AzulStateRepresentation:
    """
    Complete numerical representation of Azul game state.
    
    The state is represented as a collection of NumPy arrays that capture
    all relevant game information in a format suitable for machine learning.
    """
    
    def __init__(self, game_state: GameState):
        """Initialize state representation from a GameState object."""
        self.num_players = game_state.num_players
        self.num_factories = 2 * self.num_players + 1
        
        # Convert game state to numerical representation
        self._convert_from_game_state(game_state)
    
    def _convert_from_game_state(self, game_state: GameState) -> None:
        """Convert GameState object to numerical arrays."""
        
        # === GLOBAL GAME STATE ===
        self.current_player = game_state.current_player
        self.round_number = game_state.round_number
        self.game_over = int(game_state.game_over)
        self.winner = game_state.winner if game_state.winner is not None else -1
        
        # === PLAYER BOARDS ===
        self.player_scores = np.array([p.score for p in game_state.players], dtype=np.int32)
        
        # Pattern lines: shape (num_players, 5, 7)
        # For each player, each pattern line: [capacity, count, color_one_hot(5)]
        self.pattern_lines = np.zeros((StateConfig.MAX_PLAYERS, StateConfig.PATTERN_LINES, 7), dtype=np.float32)
        
        # Wall: shape (num_players, 5, 5) - binary filled/empty
        self.walls = np.zeros((StateConfig.MAX_PLAYERS, StateConfig.WALL_SIZE, StateConfig.WALL_SIZE), dtype=np.int8)
        
        # Floor lines: shape (num_players, 7, 7)
        # For each position: [has_tile, color_one_hot(5), first_player_marker]
        self.floor_lines = np.zeros((StateConfig.MAX_PLAYERS, StateConfig.FLOOR_LINE_SIZE, 7), dtype=np.int8)
        
        # First player markers: shape (num_players,)
        self.first_player_markers = np.zeros(StateConfig.MAX_PLAYERS, dtype=np.int8)
        
        for i, player in enumerate(game_state.players):
            if i < self.num_players:
                self._encode_player_board(player, i)
        
        # === FACTORIES ===
        # Factory displays: shape (max_factories, 4, 6)
        # For each tile position: [has_tile, color_one_hot(5)]
        self.factories = np.zeros((StateConfig.MAX_FACTORIES, StateConfig.TILES_PER_FACTORY, 6), dtype=np.int8)
        
        for i, factory in enumerate(game_state.factory_area.factories):
            if i < self.num_factories:
                self._encode_factory(factory, i)
        
        # === CENTER AREA ===
        # Center tiles: shape (max_center_tiles, 6)
        # For each tile: [has_tile, color_one_hot(5)]
        self.center_tiles = np.zeros((StateConfig.MAX_CENTER_TILES, 6), dtype=np.int8)
        self.center_first_player_marker = int(game_state.factory_area.center.has_first_player_marker)
        
        self._encode_center_area(game_state.factory_area.center)
        
        # === TILE SUPPLY ===
        # Bag and discard pile counts by color: shape (2, 5)
        # Row 0: bag, Row 1: discard pile
        self.tile_supply = np.zeros((2, StateConfig.NUM_COLORS), dtype=np.int32)
        
        self._encode_tile_supply(game_state.bag, game_state.discard_pile)
    
    def _encode_player_board(self, player, player_idx: int) -> None:
        """Encode a single player's board state."""
        
        # Pattern lines
        for line_idx, pattern_line in enumerate(player.pattern_lines):
            # Capacity (normalized)
            self.pattern_lines[player_idx, line_idx, 0] = (line_idx + 1) / 5.0
            
            # Current count (normalized)
            self.pattern_lines[player_idx, line_idx, 1] = len(pattern_line.tiles) / (line_idx + 1)
            
            # Color one-hot encoding
            if pattern_line.color is not None:
                color_idx = self._color_to_index(pattern_line.color)
                if color_idx < 5:  # Valid color
                    self.pattern_lines[player_idx, line_idx, 2 + color_idx] = 1.0
            else:
                self.pattern_lines[player_idx, line_idx, 6] = 1.0  # Empty
        
        # Wall
        for row in range(StateConfig.WALL_SIZE):
            for col in range(StateConfig.WALL_SIZE):
                self.walls[player_idx, row, col] = int(player.wall.filled[row][col])
        
        # Floor line
        for pos, tile in enumerate(player.floor_line):
            if pos < StateConfig.FLOOR_LINE_SIZE:
                self.floor_lines[player_idx, pos, 0] = 1  # Has tile
                color_idx = self._color_to_index(tile.color)
                if color_idx == ColorIndex.FIRST_PLAYER:
                    self.floor_lines[player_idx, pos, 6] = 1  # First player marker
                elif color_idx < 5:  # Valid color index
                    self.floor_lines[player_idx, pos, 1 + color_idx] = 1
        
        # First player marker
        self.first_player_markers[player_idx] = int(player.has_first_player_marker())
    
    def _encode_factory(self, factory, factory_idx: int) -> None:
        """Encode a single factory's state."""
        for tile_idx, tile in enumerate(factory.tiles):
            if tile_idx < StateConfig.TILES_PER_FACTORY:
                self.factories[factory_idx, tile_idx, 0] = 1  # Has tile
                color_idx = self._color_to_index(tile.color)
                if color_idx < 5:  # Valid color (excluding first player)
                    self.factories[factory_idx, tile_idx, 1 + color_idx] = 1
    
    def _encode_center_area(self, center) -> None:
        """Encode the center area state."""
        for tile_idx, tile in enumerate(center.tiles):
            if tile_idx < StateConfig.MAX_CENTER_TILES:
                self.center_tiles[tile_idx, 0] = 1  # Has tile
                color_idx = self._color_to_index(tile.color)
                if color_idx < 5:  # Valid color
                    self.center_tiles[tile_idx, 1 + color_idx] = 1
    
    def _encode_tile_supply(self, bag: List, discard_pile: List) -> None:
        """Encode tile supply (bag and discard pile)."""
        # Count tiles by color in bag
        for tile in bag:
            color_idx = self._color_to_index(tile.color)
            if color_idx < 5:  # Valid color
                self.tile_supply[0, color_idx] += 1
        
        # Count tiles by color in discard pile
        for tile in discard_pile:
            color_idx = self._color_to_index(tile.color)
            if color_idx < 5:  # Valid color
                self.tile_supply[1, color_idx] += 1
    
    def _color_to_index(self, color: TileColor) -> int:
        """Convert TileColor to numerical index."""
        color_map = {
            TileColor.BLUE: ColorIndex.BLUE,
            TileColor.YELLOW: ColorIndex.YELLOW,
            TileColor.RED: ColorIndex.RED,
            TileColor.BLACK: ColorIndex.BLACK,
            TileColor.WHITE: ColorIndex.WHITE,
            TileColor.FIRST_PLAYER: ColorIndex.FIRST_PLAYER,
        }
        return color_map.get(color, ColorIndex.EMPTY)
    
    def get_flat_state_vector(self, normalize: bool = True) -> np.ndarray:
        """
        Get a flattened state vector for neural networks.
        
        Args:
            normalize: Whether to normalize values to [0, 1] range
            
        Returns:
            1D numpy array containing the complete game state
        """
        components = []
        
        # Global state (4 values)
        global_state = np.array([
            self.current_player / (self.num_players - 1) if normalize else self.current_player,
            self.round_number / StateConfig.MAX_ROUND if normalize else self.round_number,
            self.game_over,  # Already 0/1
            (self.winner + 1) / self.num_players if normalize and self.winner >= 0 else 0
        ], dtype=np.float32)
        components.append(global_state)
        
        # Player scores (max_players values)
        if normalize:
            scores = self.player_scores.astype(np.float32) / StateConfig.MAX_SCORE
        else:
            scores = self.player_scores.astype(np.float32)
        # Pad to max players
        padded_scores = np.zeros(StateConfig.MAX_PLAYERS, dtype=np.float32)
        padded_scores[:len(scores)] = scores
        components.append(padded_scores)
        
        # Player boards
        components.append(self.pattern_lines.flatten())  # Already normalized
        components.append(self.walls.flatten().astype(np.float32))  # Already 0/1
        components.append(self.floor_lines.flatten().astype(np.float32))  # Already 0/1
        components.append(self.first_player_markers.astype(np.float32))  # Already 0/1
        
        # Factories
        components.append(self.factories.flatten().astype(np.float32))  # Already 0/1
        
        # Center area
        components.append(self.center_tiles.flatten().astype(np.float32))  # Already 0/1
        components.append(np.array([self.center_first_player_marker], dtype=np.float32))  # Already 0/1
        
        # Tile supply
        if normalize:
            supply = self.tile_supply.astype(np.float32) / StateConfig.TILES_PER_COLOR
        else:
            supply = self.tile_supply.astype(np.float32)
        components.append(supply.flatten())
        
        return np.concatenate(components)
    
    def get_state_dict(self) -> Dict[str, np.ndarray]:
        """
        Get state as a dictionary of named arrays.
        
        Returns:
            Dictionary mapping component names to numpy arrays
        """
        return {
            # Global state
            'current_player': np.array([self.current_player], dtype=np.int32),
            'round_number': np.array([self.round_number], dtype=np.int32),
            'game_over': np.array([self.game_over], dtype=np.int8),
            'winner': np.array([self.winner], dtype=np.int32),
            
            # Player state
            'player_scores': self.player_scores,
            'pattern_lines': self.pattern_lines,
            'walls': self.walls,
            'floor_lines': self.floor_lines,
            'first_player_markers': self.first_player_markers,
            
            # Factory state
            'factories': self.factories,
            'center_tiles': self.center_tiles,
            'center_first_player_marker': np.array([self.center_first_player_marker], dtype=np.int8),
            
            # Tile supply
            'tile_supply': self.tile_supply,
        }
    
    def get_player_view(self, player_id: int, include_hidden: bool = False) -> Dict[str, np.ndarray]:
        """
        Get state from a specific player's perspective.
        
        Args:
            player_id: The player whose perspective to use
            include_hidden: Whether to include hidden information (bag contents)
            
        Returns:
            Dictionary of state arrays from player's perspective
        """
        state = self.get_state_dict()
        
        if not include_hidden:
            # Hide exact bag contents, only show total counts
            bag_totals = np.sum(state['tile_supply'][0])
            discard_totals = np.sum(state['tile_supply'][1])
            state['tile_supply'] = np.array([[bag_totals, 0, 0, 0, 0], 
                                           state['tile_supply'][1]], dtype=np.int32)
        
        # Reorder players so current player is first
        if player_id != 0:
            for key in ['player_scores', 'pattern_lines', 'walls', 'floor_lines', 'first_player_markers']:
                original = state[key]
                reordered = np.roll(original, -player_id, axis=0)
                state[key] = reordered
            
            # Adjust current_player index
            state['current_player'] = np.array([(self.current_player - player_id) % self.num_players], dtype=np.int32)
        
        return state
    
    @property
    def state_shape(self) -> Dict[str, Tuple[int, ...]]:
        """Get the shape of each state component."""
        return {
            'current_player': (1,),
            'round_number': (1,),
            'game_over': (1,),
            'winner': (1,),
            'player_scores': (StateConfig.MAX_PLAYERS,),
            'pattern_lines': (StateConfig.MAX_PLAYERS, StateConfig.PATTERN_LINES, 7),
            'walls': (StateConfig.MAX_PLAYERS, StateConfig.WALL_SIZE, StateConfig.WALL_SIZE),
            'floor_lines': (StateConfig.MAX_PLAYERS, StateConfig.FLOOR_LINE_SIZE, 7),
            'first_player_markers': (StateConfig.MAX_PLAYERS,),
            'factories': (StateConfig.MAX_FACTORIES, StateConfig.TILES_PER_FACTORY, 6),
            'center_tiles': (StateConfig.MAX_CENTER_TILES, 6),
            'center_first_player_marker': (1,),
            'tile_supply': (2, StateConfig.NUM_COLORS),
        }
    
    @property
    def flat_state_size(self) -> int:
        """Get the size of the flattened state vector."""
        # Calculate expected size manually
        expected_size = (
            4 +  # Global state
            StateConfig.MAX_PLAYERS +  # Player scores
            StateConfig.MAX_PLAYERS * StateConfig.PATTERN_LINES * 7 +  # Pattern lines
            StateConfig.MAX_PLAYERS * StateConfig.WALL_SIZE * StateConfig.WALL_SIZE +  # Walls
            StateConfig.MAX_PLAYERS * StateConfig.FLOOR_LINE_SIZE * 7 +  # Floor lines (updated)
            StateConfig.MAX_PLAYERS +  # First player markers
            StateConfig.MAX_FACTORIES * StateConfig.TILES_PER_FACTORY * 6 +  # Factories
            StateConfig.MAX_CENTER_TILES * 6 +  # Center tiles
            1 +  # Center first player marker
            2 * StateConfig.NUM_COLORS  # Tile supply
        )
        return expected_size

    def get_tile_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of tiles across all game locations.
        
        Returns:
            Dictionary with tile counts for each location
        """
        # Count tiles in supply
        bag_tiles = int(np.sum(self.tile_supply[0]))
        discard_tiles = int(np.sum(self.tile_supply[1]))
        
        # Count tiles on factories
        factory_tiles = 0
        for factory_idx in range(self.num_factories):
            factory_tiles += int(np.sum(self.factories[factory_idx, :, 0]))
        
        # Count tiles in center
        center_tiles = int(np.sum(self.center_tiles[:, 0]))
        
        # Count tiles on player boards (pattern lines + floor lines)
        player_tiles = 0
        first_player_markers = 0
        
        for player_idx in range(self.num_players):
            # Pattern lines
            for line_idx in range(5):
                capacity = line_idx + 1
                fill_ratio = self.pattern_lines[player_idx, line_idx, 1]
                tiles_in_line = int(fill_ratio * capacity)
                player_tiles += tiles_in_line
            
            # Floor lines - count tiles but exclude first player markers
            for pos in range(7):
                if self.floor_lines[player_idx, pos, 0] == 1:  # Has tile
                    if self.floor_lines[player_idx, pos, 6] == 1:  # First player marker
                        first_player_markers += 1
                    else:  # Regular tile
                        player_tiles += 1
        
        # Count tiles on walls
        wall_tiles = 0
        for player_idx in range(self.num_players):
            wall_tiles += int(np.sum(self.walls[player_idx]))
        
        # Count first player markers separately (they're not regular tiles)
        if self.center_first_player_marker:
            first_player_markers += 1
        
        # Add first player markers held by players
        first_player_markers += int(np.sum(self.first_player_markers))
        
        total_tiles = bag_tiles + discard_tiles + factory_tiles + center_tiles + player_tiles + wall_tiles
        
        return {
            'bag': bag_tiles,
            'discard': discard_tiles,
            'factories': factory_tiles,
            'center': center_tiles,
            'player_boards': player_tiles,
            'walls': wall_tiles,
            'first_player_markers': first_player_markers,
            'total': total_tiles,
            'expected_total': StateConfig.TOTAL_TILES
        }


def create_state_representation(game_state: GameState) -> AzulStateRepresentation:
    """
    Convenience function to create a state representation from a GameState.
    
    Args:
        game_state: The GameState object to convert
        
    Returns:
        AzulStateRepresentation object
    """
    return AzulStateRepresentation(game_state)


def get_state_documentation() -> str:
    """
    Get comprehensive documentation of the state representation format.
    
    Returns:
        String containing detailed documentation
    """
    doc = """
    AZUL GAME STATE NUMERICAL REPRESENTATION DOCUMENTATION
    =====================================================
    
    This document describes the complete numerical representation of the Azul game state
    using NumPy arrays. All components are designed for efficient computation and 
    machine learning applications.
    
    ## GLOBAL GAME STATE
    
    ### current_player: int
    - Index of the current player (0 to num_players-1)
    
    ### round_number: int  
    - Current round number (starts at 1)
    
    ### game_over: int
    - 0 = game in progress, 1 = game finished
    
    ### winner: int
    - Index of winning player (-1 if game not finished or tied)
    
    ## PLAYER STATE
    
    ### player_scores: array(MAX_PLAYERS,)
    - Current score for each player
    - Padded with zeros for unused player slots
    
    ### pattern_lines: array(MAX_PLAYERS, 5, 7)
    - For each player and each pattern line (0-4):
      - [0]: Capacity (normalized to [0,1])
      - [1]: Current tile count (normalized by capacity)
      - [2-6]: Color one-hot encoding (blue, yellow, red, black, white)
      - [6]: Empty indicator (1 if no color assigned)
    
    ### walls: array(MAX_PLAYERS, 5, 5)
    - Binary array indicating filled positions on each player's wall
    - 1 = tile placed, 0 = empty
    
    ### floor_lines: array(MAX_PLAYERS, 7, 7)
    - For each player and each floor position (0-6):
      - [0]: Has tile indicator (1 if occupied, 0 if empty)
      - [1-5]: Color one-hot encoding (blue, yellow, red, black, white)
      - [6]: First player marker indicator
    
    ### first_player_markers: array(MAX_PLAYERS,)
    - Binary indicator of which player has the first player marker
    
    ## FACTORY STATE
    
    ### factories: array(MAX_FACTORIES, 4, 6)
    - For each factory and each tile position (0-3):
      - [0]: Has tile indicator (1 if occupied, 0 if empty)
      - [1-5]: Color one-hot encoding (blue, yellow, red, black, white)
    
    ### center_tiles: array(MAX_CENTER_TILES, 6)
    - For each tile in center area:
      - [0]: Has tile indicator (1 if occupied, 0 if empty)
      - [1-5]: Color one-hot encoding (blue, yellow, red, black, white)
    
    ### center_first_player_marker: int
    - Binary indicator if first player marker is in center (1) or not (0)
    
    ## TILE SUPPLY
    
    ### tile_supply: array(2, 5)
    - Row 0: Count of each color in bag
    - Row 1: Count of each color in discard pile
    - Columns: [blue, yellow, red, black, white]
    
    ## COLOR ENCODING
    
    Throughout the representation, colors are encoded as:
    - 0: Blue
    - 1: Yellow  
    - 2: Red
    - 3: Black
    - 4: White
    - 5: First Player Marker (where applicable)
    - 6: Empty/None (where applicable)
    
    ## NORMALIZATION
    
    Values are normalized for neural network compatibility:
    - Scores: Divided by MAX_SCORE (200)
    - Round numbers: Divided by MAX_ROUND (15)
    - Pattern line counts: Divided by line capacity
    - All other values are already in [0,1] range
    
    ## USAGE EXAMPLES
    
    ```python
    # Create representation from game state
    state_repr = AzulStateRepresentation(game_state)
    
    # Get flattened vector for neural networks
    flat_vector = state_repr.get_flat_state_vector(normalize=True)
    
    # Get structured state dictionary
    state_dict = state_repr.get_state_dict()
    
    # Get player-specific view
    player_view = state_repr.get_player_view(player_id=0, include_hidden=False)
    ```
    
    ## TOTAL STATE SIZE
    
    The flattened state vector contains approximately 1,500+ values, providing
    a complete numerical representation of the game state suitable for 
    machine learning algorithms.
    """
    return doc 