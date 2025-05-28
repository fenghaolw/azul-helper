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

### 🤖 AI Agent Examples

#### `mcts_example.py`
**Comprehensive MCTS demonstration with PyTorch neural networks**
- Complete implementation of MCTS algorithm with neural network guidance
- Shows how to use MCTS for Azul game playing with real PyTorch models
- Demonstrates all MCTS phases: Selection, Expansion, Simulation, Backpropagation
- Includes performance benchmarking, network comparisons, and advanced features
- AlphaZero-style implementation with UCT selection

```bash
python3 examples/mcts_example.py
```

**Features:**
- **Basic MCTS Usage**: Simple example with Azul game states
- **Gameplay Demonstration**: MCTS agent playing actual Azul games
- **Network Configurations**: Comparison of small, medium, and large networks
- **Performance Benchmarking**: Speed tests across different configurations
- **MCTS Features**: Temperature effects, deterministic vs stochastic selection
- **Numerical State Integration**: Shows how neural networks use game state representation
- **Model Persistence**: Saving and loading trained PyTorch models
- **Custom Networks**: Creating networks with custom architectures

**Sample Output:**
```
MCTS with Azul Game - Basic Example
============================================================
Initial state: GameState(round=1, player=0, game_over=False)
Legal actions: 78 available

Neural network evaluation:
Action probabilities shape: (78,)
State value: 0.123
Probabilities sum: 1.000

Running MCTS with 50 simulations...
Root node statistics: MCTSNode(N=50, Q=0.206, P=0.000, children=78)

MCTS Performance Benchmark
============================================================
Testing small network:
  Simulations:  5 | Time: 0.201s | Sims/sec: 24.9 | Root visits: 5 | Children: 78
  Simulations: 10 | Time: 0.402s | Sims/sec: 24.9 | Root visits: 10 | Children: 78
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

# Comprehensive MCTS example
python3 examples/mcts_example.py

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

### 🤖 **AI and Machine Learning**
- **MCTS Algorithm**: Complete Monte Carlo Tree Search implementation
- **Neural Network Integration**: PyTorch models for game evaluation
- **Action Selection**: Deterministic and stochastic strategies
- **Performance Optimization**: Benchmarking and configuration tuning
- **Model Persistence**: Saving and loading trained models

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

## MCTS Example Details

The comprehensive `mcts_example.py` includes eight different demonstrations:

1. **Basic MCTS Usage** (`run_basic_mcts_example`)
   - Simple MCTS search with neural network evaluation
   - Shows action probabilities before and after MCTS

2. **Gameplay Demonstration** (`demonstrate_gameplay`)
   - MCTS agent playing actual Azul game moves
   - Move-by-move analysis with scores

3. **Network Configuration Comparison** (`compare_network_configurations`)
   - Performance comparison of small, medium, and large networks
   - Parameter counts and evaluation times

4. **Performance Benchmarking** (`benchmark_mcts_performance`)
   - Speed tests across different simulation counts
   - Network size impact on performance

5. **MCTS Features** (`demonstrate_mcts_features`)
   - Temperature effects on action selection
   - Deterministic vs stochastic behavior

6. **Numerical State Integration** (`demonstrate_numerical_state`)
   - How neural networks use game state representation
   - Consistency validation

7. **Model Persistence** (`demonstrate_model_saving`)
   - Saving and loading PyTorch models
   - Verification of loaded model accuracy

8. **Custom Network Creation** (`demonstrate_custom_network`)
   - Creating networks with custom architectures
   - Advanced configuration options

## Integration with Tests

These examples complement the comprehensive test suite in `tests/test_mcts.py`:

- **Examples**: Interactive demonstrations showing functionality
- **Tests**: Automated validation ensuring correctness
- **Documentation**: Clear specifications and usage patterns

## Next Steps

After exploring these examples, you can:

1. **Implement RL Agents**: Use the numerical state representation for training
2. **Create Custom Games**: Modify game parameters and rules
3. **Analyze Gameplay**: Use tile counting and color tracking for analysis
4. **Train Neural Networks**: Use MCTS for self-play training
5. **Extend Functionality**: Add new features building on the validated foundation

The numerical state representation is **complete, validated, and ready** for machine learning applications, and the MCTS implementation provides a solid foundation for advanced AI game playing!
