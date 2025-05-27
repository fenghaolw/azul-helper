# Azul RL Examples and Demonstrations

This directory contains example scripts and demonstrations for the Azul reinforcement learning implementation.

## Available Examples

### 🎮 Game Examples

#### `example_game.py`
**Basic game demonstration**
- Shows how to create and play a complete Azul game
- Demonstrates basic game mechanics and API usage
- Good starting point for understanding the game implementation

```bash
python3 examples/example_game.py
```

### 🔢 State Representation Examples

#### `demo_tile_counting.py`
**Tile counting and accessibility demonstration**
- Shows how to access and count tiles in different game locations
- Demonstrates factory tile counting (answers "Why can we access factory tiles?")
- Validates tile conservation across all game components
- Uses both helper methods and direct array access

```bash
python3 examples/demo_tile_counting.py
```

**Sample Output:**
```
=== Azul Tile Counting Demonstration ===

1. INITIAL GAME STATE
========================================
Complete tile distribution:
  bag: 80
  discard: 0
  factories: 20
  center: 0
  player_boards: 0
  walls: 0
  first_player_markers: 1
  total: 100
  expected_total: 100

2. MANUAL FACTORY TILE COUNTING
========================================
  Factory 0: 4 tiles
  Factory 1: 4 tiles
  Factory 2: 4 tiles
  Factory 3: 4 tiles
  Factory 4: 4 tiles
  Total factory tiles: 20

CONCLUSION: Factory tiles are easily accessible!
```

#### `demo_color_validation.py`
**Color representation validation demonstration**
- Validates that color encoding accurately represents actual game state
- Shows side-by-side comparison of actual vs encoded colors
- Demonstrates pattern line color tracking during gameplay
- Validates color accuracy across all game components

```bash
python3 examples/demo_color_validation.py
```

**Sample Output:**
```
=== Azul Color Representation Validation Demo ===

1. INITIAL FACTORY COLORS
==================================================
Factory 0: ['white', 'blue', 'blue', 'white']
Factory 1: ['yellow', 'yellow', 'yellow', 'blue']

2. ENCODED FACTORY COLORS
==================================================
Factory 0: ['white', 'blue', 'blue', 'white']
Factory 1: ['yellow', 'yellow', 'yellow', 'blue']

3. MAKING MOVES AND TRACKING COLOR CHANGES
==================================================
Move 1: Action(factory_0, blue, line_1)
  → Placing on pattern line 1
  Player 0 pattern lines: ['line_1:blue(2)']

5. PATTERN LINE COLOR VALIDATION
==================================================
Player 0 pattern lines:
  Line 0: actual=blue(1), encoded=blue ✓
  Line 1: actual=blue(2), encoded=blue ✓

CONCLUSION: Color representations are accurate and validated!
```

## Running Examples

### Prerequisites
Make sure you're in the azul_rl directory and have the package installed:

```bash
cd azul_rl
pip install -e .
```

### Running Individual Examples
```bash
# Basic game example
python3 examples/example_game.py

# Tile counting demonstration
python3 examples/demo_tile_counting.py

# Color validation demonstration
python3 examples/demo_color_validation.py
```

### Running All Examples
```bash
# Run all examples in sequence
for example in examples/*.py; do
    echo "=== Running $example ==="
    python3 "$example"
    echo
done
```

## What These Examples Demonstrate

### 🎯 **Core Functionality**
- **Game Creation**: How to create and configure Azul games
- **Action Handling**: Legal action generation and application
- **Game Flow**: Round progression and game termination

### 🔢 **Numerical State Representation**
- **Complete Coverage**: All game components represented numerically
- **Tile Accessibility**: Easy access to tiles in all locations
- **Color Accuracy**: Validated color encoding across all components
- **Tile Conservation**: All 100 tiles tracked throughout gameplay

### 🧪 **Validation and Testing**
- **Color Validation**: Encoded colors match actual game state
- **Tile Counting**: Accurate tile distribution tracking
- **State Consistency**: Numerical representation matches game logic

### 🚀 **Machine Learning Readiness**
- **NumPy Arrays**: Efficient numerical representation
- **Normalized Values**: [0,1] range for neural networks
- **Fixed Shapes**: Consistent dimensions for batch processing
- **Complete Information**: All game state captured numerically

## Integration with Tests

These examples complement the comprehensive test suite in `tests/test_state_representation.py`:

- **Examples**: Interactive demonstrations showing functionality
- **Tests**: Automated validation ensuring correctness
- **Documentation**: Clear specifications and usage patterns

## Next Steps

After exploring these examples, you can:

1. **Implement RL Agents**: Use the numerical state representation for training
2. **Create Custom Games**: Modify game parameters and rules
3. **Analyze Gameplay**: Use tile counting and color tracking for analysis
4. **Extend Functionality**: Add new features building on the validated foundation

The numerical state representation is **complete, validated, and ready** for machine learning applications! 