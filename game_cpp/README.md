# Azul C++ Implementation

This directory contains a high-performance C++ implementation of the Azul board game, with OpenSpiel integration and MCTS agent support.

## Features

- **Core Game Logic**: Complete implementation of Azul game rules in C++
- **OpenSpiel Integration**: Compatible with OpenSpiel for advanced AI algorithms
- **MCTS Agent**: Monte Carlo Tree Search agent for strong gameplay
- **Python Bindings**: Expose C++ functionality to Python for easy integration
- **Performance Optimized**: Efficient data structures and algorithms for fast gameplay

## File Structure

```
game_cpp/
├── tile.h/cpp              # Tile and TileColor definitions
├── action.h/cpp             # Action representation
├── player_board.h/cpp       # PatternLine, Wall, and PlayerBoard classes
├── factory.h/cpp            # Factory, CenterArea, and FactoryArea classes
├── game_state.h/cpp         # Main GameState class
├── azul_openspiel.h/cpp     # OpenSpiel integration
├── mcts_agent.h/cpp         # MCTS agent implementation
├── python_bindings.cpp      # Python bindings using pybind11
├── CMakeLists.txt           # Build configuration
└── README.md               # This file
```

## Dependencies

### Required
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.12+
- pybind11 (for Python bindings)

### Optional
- OpenSpiel (for advanced AI algorithms and MCTS)

## Building

The primary way to build the C++ implementation is by using the provided `build.sh` script. This script handles CMake configuration and compilation.

### Standard Build

This will build the core game logic and Python bindings (if pybind11 is found).

```bash
cd game_cpp
./build.sh
```
The script creates a `build` directory and compiles the project. By default, it attempts a `Release` build.

### Build with OpenSpiel Support

To build with OpenSpiel integration (required for the MCTS agent and other OpenSpiel-based AI):

1.  **Install OpenSpiel**: Follow the instructions on the [official OpenSpiel repository](https://github.com/deepmind/open_spiel) to clone and build it. Make a note of the path to your OpenSpiel installation.
    Example OpenSpiel build steps (refer to their documentation for the latest):
    ```bash
    git clone https://github.com/deepmind/open_spiel.git
    cd open_spiel
    mkdir build
    cd build
    # Adjust CMake options as needed, e.g., for Python version or C++ compiler
    cmake ..
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    ```

2.  **Build Azul with OpenSpiel**: Set the `OPENSPIEL_ROOT` environment variable to the root directory of your OpenSpiel installation before running the `build.sh` script.

    ```bash
    cd game_cpp
    export OPENSPIEL_ROOT=/path/to/your/open_spiel_installation
    ./build.sh
    ```

### Python Integration and Dependencies

Python bindings are built by default if `pybind11` is detected in your Python environment.

1.  **Install pybind11**:
    ```bash
    pip install pybind11
    ```
    Ensure `pybind11` is installed for the Python environment you intend to use with Azul. CMake will typically find Python and pybind11 automatically. If you have multiple Python versions, ensure the one with pybind11 is active or correctly configured in your PATH.

2.  After building (e.g., via `./build.sh`), the Python module `azul_cpp_bindings` will be created in the `game_cpp/build` directory. To use it, ensure this directory is in your `PYTHONPATH` or install the package appropriately.

### Manual Build (Advanced)

For more control over the build process (e.g., choosing a different build type like `Debug`), you can run CMake commands manually:

```bash
cd game_cpp
mkdir -p build
cd build
# For a debug build:
cmake -DCMAKE_BUILD_TYPE=Debug ..
# For a release build with OpenSpiel:
# cmake -DCMAKE_BUILD_TYPE=Release -DOPENSPIEL_ROOT=/path/to/open_spiel ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
```
This is essentially what `build.sh` automates.

## Testing

To run all C++ unit tests, use the `run_all_tests.sh` script from the `game_cpp` directory:

```bash
cd game_cpp
./run_all_tests.sh
```
This script will configure the project (in `Debug` mode by default, which is suitable for testing), build all targets including the test executables, and then run them using CTest. It provides a summary of passed and failed tests.

### Running Tests Manually with CTest

Alternatively, after building the project (e.g., by running `./build.sh` or manually with CMake), you can execute tests directly using CTest from within the build directory. This gives you more control over CTest options (e.g., running specific tests or using different output formats).

```bash
cd game_cpp/build
# Ensure the project has been built first
ctest --output-on-failure
# Or for more verbose output:
# ctest -V
```
Using `ctest` directly is useful if you want to test a specific build configuration (e.g., `Release`) that you've already compiled, or if you need more detailed test output.

## Usage

### C++ Usage

```cpp
#include "game_state.h"

// Create a new game
azul::GameState game(2, 42); // 2 players, seed 42

// Get legal actions
auto actions = game.get_legal_actions();

// Apply an action
if (!actions.empty()) {
    game.apply_action(actions[0]);
}

// Check game state
if (game.is_game_over()) {
    int winner = game.get_winner();
    auto scores = game.get_scores();
}
```

### Python Usage

```python
import azul_cpp_bindings as azul

# Create a new game
game = azul.create_game(num_players=2, seed=42)

# Get legal actions
actions = game.get_legal_actions()

# Apply an action
if actions:
    game.apply_action(actions[0])

# Check game state
if game.is_game_over():
    winner = game.get_winner()
    scores = game.get_scores()
```

### MCTS Agent Usage (with OpenSpiel)

```python
import azul_cpp_bindings as azul

# Create MCTS agent
agent = azul.create_mcts_agent(
    player_id=0,
    num_simulations=1000,
    uct_c=1.4,
    seed=42
)

# Use in game loop
game = azul.AzulGame({"players": 2})
state = game.NewInitialState()

while not state.IsTerminal():
    if state.CurrentPlayer() == 0:
        action = agent.get_action(state)
        state.DoApplyAction(action)
    else:
        # Other player logic
        legal_actions = state.LegalActions()
        action = legal_actions[0]  # Random choice
        state.DoApplyAction(action)
```

## Performance Benefits

The C++ implementation provides significant performance improvements over the Python version:

- **Memory Efficiency**: Tile pooling and optimized data structures
- **Computation Speed**: 10-100x faster game simulation
- **Cache Optimization**: Precomputed lookup tables for wall patterns
- **MCTS Performance**: Faster tree search for stronger AI

## Integration with Existing Python Code

The C++ implementation is designed to be a drop-in replacement for the Python version:

```python
# Replace Python imports
# from game.game_state import GameState, Action
# with C++ imports
from azul_cpp_bindings import GameState, Action

# The API is compatible
game = GameState(num_players=2, seed=42)
actions = game.get_legal_actions()
# ... rest of code works the same
```

## OpenSpiel Compatibility

The implementation is fully compatible with OpenSpiel's algorithms:

- **MCTS**: Monte Carlo Tree Search
- **AlphaZero**: Deep reinforcement learning
- **CFR**: Counterfactual Regret Minimization
- **Minimax**: Perfect information game solving

## Contributing

When modifying the C++ code:

1. Maintain API compatibility with the Python version
2. Add appropriate Python bindings for new functionality
3. Update tests to cover new features
4. Follow C++17 best practices
5. Ensure OpenSpiel integration remains functional

## Troubleshooting

### Common Build Issues

1. **Missing pybind11**: Install with `pip install pybind11`
2. **OpenSpiel not found**: Set `OPENSPIEL_ROOT` correctly
3. **C++17 support**: Ensure compiler supports C++17
4. **CMake version**: Upgrade to CMake 3.12+

### Runtime Issues

1. **Import errors**: Check Python path and module compilation. Ensure `game_cpp/build` is in `PYTHONPATH` or the module is installed.
2. **Segmentation faults**: Verify object lifetime management, especially with Python bindings.
3. **Performance issues**: Ensure you are using a `Release` build type for performance-critical tasks. The `build.sh` script defaults to a `Release` build. If building manually, use `cmake -DCMAKE_BUILD_TYPE=Release ..`. This enables optimizations like `-O3` (on GCC/Clang). For debugging, use the `Debug` build type (`cmake -DCMAKE_BUILD_TYPE=Debug ..`), which is what `run_all_tests.sh` uses by default.
