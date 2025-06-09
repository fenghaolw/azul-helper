# Azul Helper - OpenSpiel Integration (Local Forked Approach)

A comprehensive C++ library for Azul game AI development using a **local forked OpenSpiel Azul game** approach. This setup gives you full control over the Azul game logic while leveraging OpenSpiel's powerful algorithms for MCTS, minimax, and other AI techniques.

## 🏗️ Architecture Overview

### Local Integration Approach

This project uses a **local forked Azul game** approach rather than depending on the upstream OpenSpiel Azul implementation:

1. **Local Azul Game**: `azul.cc` and `azul.h` are forked copies that you can modify
2. **Local OpenSpiel Library**: `libopen_spiel.dylib` provides core OpenSpiel functionality 
3. **Custom Agents**: Your agents work with both the local Azul game and OpenSpiel algorithms
4. **Full Control**: Modify game rules, scoring, or mechanics as needed for research

### Benefits

- 🎯 **Full Game Control**: Modify Azul game logic without rebuilding entire OpenSpiel
- 🚀 **Fast Iteration**: Changes to game rules only require rebuilding this project
- 🔧 **Stable Integration**: Uses clean OpenSpiel library without dependency conflicts
- 📊 **Zero-Sum Support**: Properly configured for minimax and other zero-sum algorithms
- 🤖 **Advanced AI**: Leverage OpenSpiel's MCTS, minimax, and other algorithms

## 🚀 Quick Start

### Prerequisites

- **C++17** or later
- **CMake 3.17** or later
- **OpenSpiel** built as shared library (see setup below)

### Setup Instructions

To use LibTorch AlphaZero training, complete these additional steps **before building**:

#### Prerequisites for LibTorch Support

1. **Install required system dependencies**:
   ```bash
   # Install wget (required for OpenSpiel's install script)
   brew install wget
   
   # Install OpenMP (required for LibTorch runtime)
   brew install libomp
   ```

2. **Configure OpenSpiel for LibTorch**:
   ```bash
   cd /path/to/open_spiel
   
   # Edit global_variables.sh to enable LibTorch support
   # Set these variables:
   export OPEN_SPIEL_BUILD_WITH_LIBNOP="ON"
   export OPEN_SPIEL_BUILD_WITH_LIBTORCH="ON"
   export OPEN_SPIEL_BUILD_WITH_LIBTORCH_DOWNLOAD_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.3.0.zip"
   ```

3. **Download LibTorch and dependencies**:
   ```bash
   # Run OpenSpiel install script to download LibTorch and libnop
   ./install.sh
   ```

4. **Fix OpenMP runtime linking** (macOS specific):
   ```bash
   # Create symbolic link for OpenMP library
   ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib ./open_spiel/libtorch/libtorch/lib/libomp.dylib
   ```

5. **Build OpenSpiel with LibTorch support**:
   ```bash
   rm -rf build && mkdir build && cd build
   BUILD_SHARED_LIB=ON CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=${CXX} ../open_spiel
   make -j$(nproc) open_spiel
   ```

7. **Copy the shared library**:
   ```bash
   cp libopen_spiel.dylib /path/to/azul-helper/game_cpp/
   ```

8. **Build the project**:
   ```bash
   cd game_cpp
   mkdir -p build && cd build
   cmake ..
   make
   ```

9. **Test the integration**:
   ```bash
   ./azul_evaluation_demo    # Run agent evaluation and tournament
   ./azul_profiling_demo     # Profile agent performance (use --help for options)
   ```

#### Using LibTorch AlphaZero

Once setup is complete, you can use LibTorch AlphaZero training:

```bash
# Build with LibTorch support (from game_cpp directory)
mkdir -p build && cd build
cmake ..
make

# Run LibTorch AlphaZero training
./train_neural_mcts.sh --libtorch --device=cpu    # CPU training
./train_neural_mcts.sh --libtorch --device=mps    # Apple Silicon GPU training
```

**Benefits of LibTorch Integration**:
- 🚀 **Pure C++ Performance**: No Python overhead
- 🧠 **Direct Tensor Operations**: Same operations as PyTorch but in C++
- 💾 **Better Memory Efficiency**: Optimized for large-scale training
- 📦 **Production Ready**: Deploy without Python dependencies

## 🎮 Available Demos & Tools

### Agent Evaluation Demo (`azul_evaluation_demo`)

Comprehensive evaluation system for testing agent performance with head-to-head matchups and tournaments:

```bash
./azul_evaluation_demo
```

**Features**:
- **Head-to-Head Evaluations**: Minimax vs MCTS, both vs Random baseline
- **3-Agent Tournament**: Round-robin between Minimax_D4, MCTS_1000, and MCTS_1000_UCT2
- **Statistical Analysis**: Win rates, confidence intervals, p-values
- **Performance Metrics**: Average game duration, score differences
- **Results Export**: Detailed results saved to `evaluation_results.txt`

**Sample Output**:
```
=== TOURNAMENT RESULTS ===
1. MCTS_1000: 61.7% win rate
2. MCTS_1000_UCT2: 46.7% win rate  
3. Minimax_D4: 41.7% win rate
```

### Agent Profiling Demo (`azul_profiling_demo`)

Performance profiling tool with configurable CLI options for analyzing agent computation:

```bash
# Profile MCTS agent with custom settings
./azul_profiling_demo --agent mcts --simulations 1000 --games 5

# Profile Minimax agent with specific depth
./azul_profiling_demo --agent minimax --depth 4 --games 3

# Quick profiling with minimal output
./azul_profiling_demo --agent mcts --simulations 500 --games 2 --quiet

# Show all available options
./azul_profiling_demo --help
```

**CLI Options**:
- `--agent <type>`: Agent type (`mcts` or `minimax`)
- `--depth <n>`: Minimax search depth (1-10, default: 4)
- `--simulations <n>`: MCTS simulation count (10-10000, default: 1000)
- `--uct <c>`: MCTS UCT exploration constant (0.1-5.0, default: 1.4)
- `--games <n>`: Number of profiling games (1-100, default: 5)
- `--seed <n>`: Random seed for reproducibility (default: 42)
- `--quiet`: Reduce output verbosity
- `--help`: Show usage information

**Features**:
- **Fast Execution**: Always competes against random agent for minimum runtime
- **Detailed Timing**: Function-level profiling with call counts and averages
- **Memory Tracking**: Memory allocation and deallocation monitoring
- **Performance Hotspots**: Top 5 most expensive operations
- **Custom Reports**: Saves reports with descriptive filenames like `profiling_mcts_s1000_uct14_report.txt`

**Sample Profiling Output**:
```
=== PROFILING RESULTS ===
Function                      Calls     Total(ms)   Avg(ms)     % Time
get_action                    54        105.31      1.95        99.9%
mcts_search_core             54        105.29      1.95        99.9%
```

## 📋 Usage Examples

### Quick Evaluation
```bash
# Run complete agent evaluation (head-to-head + tournament)
./azul_evaluation_demo

# Quick MCTS profiling
./azul_profiling_demo --agent mcts --games 3 --quiet

# Deep minimax analysis
./azul_profiling_demo --agent minimax --depth 5 --games 5
```

### Research Workflows
```bash
# Compare MCTS configurations
./azul_profiling_demo --agent mcts --simulations 500 --uct 1.0 --games 10
./azul_profiling_demo --agent mcts --simulations 500 --uct 2.0 --games 10

# Minimax depth analysis
for depth in 2 3 4 5; do
  ./azul_profiling_demo --agent minimax --depth $depth --games 5 --quiet
done

# Tournament evaluation
./azul_evaluation_demo > tournament_results.txt
```

### Performance Benchmarking
```bash
# Profile different agent configurations
./azul_profiling_demo --agent mcts --simulations 100  --games 10
./azul_profiling_demo --agent mcts --simulations 500  --games 10  
./azul_profiling_demo --agent mcts --simulations 1000 --games 10

# Compare with minimax baseline
./azul_profiling_demo --agent minimax --depth 3 --games 10
```

## 🤖 Agent Implementations

### RandomAgent

**File**: `random_agent.h`, `random_agent.cpp`

Simple baseline agent that selects actions uniformly at random.

**Features**:
- Configurable random seed
- Cross-compatible with OpenSpiel and custom game states
- Useful for baseline comparisons

**Usage**:
```cpp
azul::RandomAgent agent(player_id, seed);
auto action = agent.get_action(state);
```

### MinimaxAgent

**File**: `minimax_agent.h`, `minimax_agent.cpp`

Advanced agent using OpenSpiel's optimized minimax algorithms with custom evaluation.

**Features**:
- Leverages OpenSpiel's `ExpectiminimaxSearch` for stochastic games
- Custom Azul-specific evaluation function
- Configurable search depth
- Node exploration tracking

**Usage**:
```cpp
azul::MinimaxAgent agent(player_id, depth);
auto action = agent.get_action(state);
std::cout << "Explored " << agent.nodes_explored() << " nodes" << std::endl;
```

## 🛠️ Development Workflow

### Modifying Game Logic

1. **Edit Local Files**: Modify `azul.cc` and `azul.h` as needed
2. **Maintain Zero-Sum**: Keep `GameType::Utility::kZeroSum` for minimax benefits
3. **Rebuild**: `make` in the build directory
4. **Test**: Run demos to verify changes work

### Adding New Agents

1. **Create Agent Files**: Follow the pattern of `random_agent.h/cpp`
2. **OpenSpiel Integration**: Always uses OpenSpiel for game state management
3. **Type Aliases**: Use `ActionType` and `GameStateType` for compatibility
4. **Update CMakeLists.txt**: Add new source files to build targets

### Performance Optimization

- **MinimaxAgent**: Leverages OpenSpiel's optimized C++ algorithms
- **Custom Evaluation**: Implement domain-specific heuristics in `evaluate_state()`
- **Depth Tuning**: Balance search depth vs. computation time
- **Node Tracking**: Monitor `nodes_explored()` for performance insights

## 📊 Integration Features

### OpenSpiel Algorithm Access

- **MCTS**: `open_spiel::algorithms::MCTSBot`
- **Minimax**: `open_spiel::algorithms::ExpectiminimaxSearch`
- **Alpha-Beta**: `open_spiel::algorithms::AlphaBetaSearch` (for deterministic variants)
- **Random Rollouts**: `open_spiel::algorithms::RandomRolloutEvaluator`

### Game State Compatibility

- **OpenSpiel States**: Full `open_spiel::State` interface
- **Custom States**: Fallback support for non-OpenSpiel implementations
- **Type Safety**: Conditional compilation ensures compatibility

### Zero-Sum Benefits

With proper zero-sum configuration:
- **Minimax**: Optimal play computation
- **Alpha-Beta Pruning**: Efficient search space reduction
- **Game Theory**: Nash equilibrium analysis
- **Performance**: Optimized algorithms for competitive scenarios

## 🔧 Build Configuration

### CMakeLists.txt Overview

The build system automatically:
1. **Detects OpenSpiel**: Finds source headers and local shared library
2. **Links Dependencies**: Handles Abseil and other OpenSpiel dependencies
3. **Compiles Local Game**: Builds your forked Azul game into `azul_local`
4. **Force Registration**: Ensures local game is registered with OpenSpiel
5. **Creates Targets**: Builds both demo and test executables

### Key Build Targets

- **`azul_local`**: Static library containing your local Azul game
- **`azul_evaluation_demo`**: Agent evaluation and tournament system
- **`azul_profiling_demo`**: Performance profiling tool with CLI options
- **`azul_mcts_demo`**: MCTS demonstration executable
- **`azul_agents_test`**: Agent integration test executable

## 🐛 Troubleshooting

### Common Issues

1. **"Unknown game 'azul'"**
   - **Cause**: Local game registration not working
   - **Fix**: Ensure force linking code is present and executes

2. **OpenSpiel Library Not Found**
   - **Cause**: `libopen_spiel.dylib` missing or wrong path
   - **Fix**: Copy shared library to `game_cpp/` directory

3. **Minimax Algorithm Failures**
   - **Cause**: Game not properly classified as zero-sum
   - **Fix**: Verify `GameType::Utility::kZeroSum` in `azul.cc`

4. **Build Errors**
   - **Cause**: Missing OpenSpiel headers or dependencies
   - **Fix**: Ensure OpenSpiel source is available and CMake can find it

### Debug Tips

- **Verbose Build**: `make VERBOSE=1` to see detailed compilation
- **Game Properties**: Check game type classification in demo output
- **Agent Fallbacks**: MinimaxAgent has graceful degradation to random selection

## 📈 Performance Characteristics

### MCTS Performance
- **Speed**: ~50,000-70,000 simulations/second
- **Memory**: <1 MB for typical game trees
- **Scalability**: Handles deep game trees efficiently

### Minimax Performance
- **Node Exploration**: 50-3,500 nodes per move (depth dependent)
- **Zero-Sum Optimization**: Significant speedup over general-sum
- **Custom Evaluation**: Domain-specific heuristics for better play

## 🔮 Future Extensions

### Potential Enhancements

1. **Neural Network Integration**: Connect TensorFlow/PyTorch models
2. **Multi-Threading**: Parallel MCTS or minimax search
3. **Game Variants**: Implement Azul: Stained Glass of Sintra, etc.
4. **Tournament System**: Automated agent evaluation framework
5. **Visualization**: Game state rendering and move analysis

### Research Applications

- **AI Algorithm Comparison**: Systematic evaluation of different approaches
- **Game Balance Analysis**: Modify rules and measure impact
- **Human vs AI Studies**: Behavioral analysis with custom game variants
- **Optimization Research**: Fine-tune evaluation functions and search parameters

## 📝 License

This project follows the same Apache 2.0 license as OpenSpiel. The forked Azul game files retain their original OpenSpiel copyright and license terms.

---

**Happy Coding!** 🎯 This setup gives you a powerful, flexible foundation for Azul AI research and development.