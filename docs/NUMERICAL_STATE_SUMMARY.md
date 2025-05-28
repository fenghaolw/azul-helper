# Azul Game State Numerical Representation - Final Summary

## Overview

This document summarizes the complete, finalized numerical representation of the Azul board game state using NumPy arrays. The implementation is designed for machine learning and reinforcement learning applications.

## ✅ Acceptance Criteria Met

All components of the game state are now represented numerically:

- ✅ **Factories**: 9×4×6 array encoding all factory displays
- ✅ **Center Area**: 40×6 array for center tiles + first player marker flag
- ✅ **Player Boards**: Complete encoding for all 4 players
  - ✅ **Pattern Lines**: 4×5×7 array with capacity, count, and color encoding
  - ✅ **Wall**: 4×5×5 binary array for tile placement
  - ✅ **Floor Line**: 4×7×7 array with tile and color encoding
- ✅ **Scores**: 4-element array for all player scores
- ✅ **First Player Token**: Binary indicators per player
- ✅ **Tiles in Bag/Discard**: 2×5 array counting tiles by color and location
- ✅ **Current Player**: Scalar index
- ✅ **Current Round**: Scalar round number
- ✅ **Game State**: Game over flag and winner information

## Data Structures

### Primary Data Structure: NumPy Arrays

All game state components use NumPy arrays with appropriate data types:

- **int8**: Binary flags and small integers (memory efficient)
- **int32**: Scores and tile counts
- **float32**: Normalized values for neural networks

### Fixed-Size Arrays

All arrays use fixed dimensions to ensure consistency:
- **4 players maximum** (padded with zeros for fewer players)
- **9 factories maximum** (supports 2-4 player games)
- **40 center tiles maximum** (conservative estimate)

## Complete State Specification

### Global Game State (4 values)
```python
current_player: int32     # Index of current player (0-3)
round_number: int32       # Current round number (starts at 1)
game_over: int8          # 0=in progress, 1=finished
winner: int32            # Winning player index (-1 if not finished)
```

### Player State (per player)
```python
player_scores: (4,) int32                    # Current scores
pattern_lines: (4, 5, 7) float32            # Pattern line state
walls: (4, 5, 5) int8                       # Wall tile placement
floor_lines: (4, 7, 7) int8                 # Floor line tiles
first_player_markers: (4,) int8             # First player marker location
```

### Factory State
```python
factories: (9, 4, 6) int8                   # Factory displays
center_tiles: (40, 6) int8                  # Center area tiles
center_first_player_marker: (1,) int8       # First player marker in center
```

### Tile Supply
```python
tile_supply: (2, 5) int32                   # Bag and discard pile counts
```

## Encoding Schemes

### Color Encoding (Consistent Throughout)
- **0**: Blue
- **1**: Yellow
- **2**: Red
- **3**: Black
- **4**: White
- **5**: First Player Marker
- **6**: Empty/None

### Pattern Lines Encoding (7 values per line)
- `[0]`: Capacity (normalized: 0.2, 0.4, 0.6, 0.8, 1.0)
- `[1]`: Current count (normalized by capacity)
- `[2-6]`: Color one-hot encoding
- `[6]`: Empty indicator

### Floor Lines Encoding (7 values per position)
- `[0]`: Has tile indicator
- `[1-5]`: Color one-hot encoding
- `[6]`: First player marker indicator

### Factory/Center Encoding (6 values per tile)
- `[0]`: Has tile indicator
- `[1-5]`: Color one-hot encoding

## Normalization Strategy

Values are normalized for neural network compatibility:

- **Scores**: Divided by MAX_SCORE (200)
- **Round numbers**: Divided by MAX_ROUND (15)
- **Pattern line counts**: Divided by line capacity
- **Current player**: Divided by (num_players - 1)
- **Tile counts**: Divided by TILES_PER_COLOR (20)
- **Binary indicators**: Already in {0, 1}

## State Vector Size

**Total flattened state size: 915 values**

Breakdown:
- Global state: 4 values
- Player scores: 4 values
- Pattern lines: 4 × 5 × 7 = 140 values
- Walls: 4 × 5 × 5 = 100 values
- Floor lines: 4 × 7 × 7 = 196 values
- First player markers: 4 values
- Factories: 9 × 4 × 6 = 216 values
- Center tiles: 40 × 6 = 240 values
- Center first player marker: 1 value
- Tile supply: 2 × 5 = 10 values

## API Usage

### Basic Usage
```python
from azul_rl.game import GameState, AzulStateRepresentation

# Create game
game = GameState(num_players=2, seed=42)

# Get numerical representation
state_repr = game.get_numerical_state()

# Get flattened vector for ML
flat_vector = state_repr.get_flat_state_vector(normalize=True)

# Get structured dictionary
state_dict = state_repr.get_state_dict()
```

### Player-Specific Views
```python
# Hide bag contents from player perspective
player_view = state_repr.get_player_view(player_id=1, include_hidden=False)

# Full information view
full_view = state_repr.get_player_view(player_id=1, include_hidden=True)
```

## Implementation Features

### ✅ Completeness
- All game state information captured
- Lossless representation (can reconstruct exact game state)
- Supports all player counts (2-4)

### ✅ Efficiency
- Memory-efficient data types
- Fixed-size arrays for consistent performance
- Suitable for batch processing

### ✅ ML Compatibility
- Normalized values in [0,1] range
- Consistent array shapes
- One-hot encodings for categorical data
- Compatible with PyTorch/TensorFlow

### ✅ Flexibility
- Player-specific views (hidden information)
- Both normalized and raw value access
- Structured dictionary and flat vector formats

## Validation

The implementation includes comprehensive tests covering:

- ✅ Shape consistency across all components
- ✅ Value range validation (normalization)
- ✅ Encoding correctness (one-hot, binary)
- ✅ State consistency after game moves
- ✅ Integration with existing GameState class
- ✅ Player view functionality
- ✅ Hidden information handling
- ✅ **Color representation accuracy** - validates that encoded colors match actual game state
- ✅ **Tile supply color counts** - verifies bag/discard color distributions
- ✅ **Wall pattern accuracy** - ensures wall placements are correctly encoded
- ✅ **Factory tile accessibility** - demonstrates easy access to factory tile counts and colors

## Documentation

Complete documentation available in:
- `azul_rl/docs/STATE_REPRESENTATION.md` - Detailed specification
- `azul_rl/game/state_representation.py` - Inline code documentation
- `azul_rl/tests/test_state_representation.py` - Usage examples and validation

## Conclusion

The numerical state representation is **complete, documented, and validated**. It provides a comprehensive, efficient, and ML-friendly encoding of the Azul game state that meets all acceptance criteria and is ready for use in reinforcement learning and other machine learning applications.
