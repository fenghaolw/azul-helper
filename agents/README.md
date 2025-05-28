# Monte Carlo Tree Search (MCTS) Implementation

This directory contains a complete implementation of the Monte Carlo Tree Search algorithm with neural network guidance, following the AlphaZero approach.

## Overview

The MCTS implementation consists of several key components:

1. **MCTSNode**: Represents nodes in the search tree
2. **MCTS**: The main search algorithm
3. **MCTSAgent**: High-level agent interface
4. **AzulNeuralNetwork**: Neural network implementation for Azul game states
5. **Protocols**: GameState and NeuralNetwork interfaces

The implementation is designed to work directly with Azul GameState objects, leveraging the numerical state representation for neural network evaluation. **No adapters are needed** - the MCTS protocol has been aligned with the Azul GameState interface.

## Components

### MCTSNode Class

The `MCTSNode` class stores the essential information for each node in the search tree:

- `N`: Visit count
- `Q`: Total action value (sum of backpropagated values)
- `P`: Prior probability from neural network
- `state`: The game state this node represents
- `children`: Dictionary mapping actions to child nodes

### MCTS Algorithm

The `MCTS` class implements the four phases of Monte Carlo Tree Search:

#### 1. Selection Phase
Uses the UCT (Upper Confidence Bound for Trees) formula to select actions:
```
UCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

#### 2. Expansion Phase
Creates child nodes for all legal actions and gets prior probabilities from the neural network.

#### 3. Simulation Phase
Uses the neural network's value estimation instead of random rollouts (AlphaZero approach).

#### 4. Backpropagation Phase
Updates visit counts (N) and action values (Q) up the tree, alternating signs for different players.

### AzulNeuralNetwork Class

The `AzulNeuralNetwork` class provides a **PyTorch neural network** implementation specifically designed for Azul game states:

- **Real Neural Network**: Uses actual PyTorch models with learnable parameters
- **Direct Integration**: Works directly with Azul GameState objects
- **Numerical State Leverage**: Uses `get_numerical_state()` for consistent evaluation
- **Proper Action Encoding**: Uses PettingZoo environment for correct action mapping
- **Multiple Configurations**: Supports 'small', 'medium', 'large', and 'deep' architectures
- **GPU Support**: Can run on CPU or CUDA devices with automatic detection
- **Model Persistence**: Save and load trained models
- **Training Integration**: Seamlessly integrates with the training pipeline

### Key Features

- **Direct Azul Integration**: No adapters needed - works directly with Azul GameState
- **Numerical State Representation**: Leverages the complete numerical state representation
- **Neural Network Guidance**: Uses NN for both action priors and state evaluation
- **Proper Action Encoding**: Uses PettingZoo environment for correct action space mapping
- **Dirichlet Noise**: Adds exploration noise to the root node
- **Temperature Control**: Adjustable randomness in action selection
- **UCT Selection**: Balances exploration and exploitation
- **Protocol-Based**: Uses Python protocols for flexible interfaces
- **Dual State Support**: Handles both in-place modification (Azul) and immutable game states

## Usage

### Basic Usage with Azul GameState

```python
from azul_rl.agents.mcts import MCTS, MCTSAgent, AzulNeuralNetwork
from azul_rl.game.game_state import GameState as AzulGameState

# Create Azul game state
azul_state = AzulGameState(num_players=2, seed=42)

# Create PyTorch neural network for Azul
neural_network = AzulNeuralNetwork(config_name="medium", device="auto")

# Create MCTS instance
mcts = MCTS(
    neural_network=neural_network,
    c_puct=1.0,              # Exploration constant
    num_simulations=800,      # Number of simulations
    temperature=1.0,          # Temperature for action selection
    dirichlet_alpha=0.3,      # Dirichlet noise parameter
    dirichlet_epsilon=0.25    # Dirichlet noise mixing
)

# Run search - returns probabilities for each legal action
action_probabilities, root_node = mcts.search(azul_state)

# Or use the agent interface
agent = MCTSAgent(neural_network, num_simulations=800)
action = agent.select_action(azul_state)  # Returns an Action object
```

### Advanced Neural Network Configuration

```python
from azul_rl.agents.mcts import MCTS, MCTSAgent, AzulNeuralNetwork
from azul_rl.game.game_state import GameState as AzulGameState

# Create Azul game state
azul_state = AzulGameState(num_players=2, seed=42)

# Option 1: Create network with specific configuration
neural_network = AzulNeuralNetwork(config_name="large", device="cuda")

# Option 2: Load a pre-trained model
# neural_network = AzulNeuralNetwork(
#     config_name="medium",
#     model_path="path/to/trained_model.pth",
#     device="auto"  # Automatically selects GPU if available
# )

# Option 3: Use custom model
# from azul_rl.training.neural_network import create_azul_network
# custom_model = create_azul_network("large", hidden_sizes=(1024, 512, 256))
# neural_network = AzulNeuralNetwork(model=custom_model, device="cpu")

print(f"Model info: {neural_network.get_model_info()}")

# Use with MCTS
mcts = MCTS(neural_network, num_simulations=800)
action_probabilities, root_node = mcts.search(azul_state)

# Or use with agent
agent = MCTSAgent(neural_network, num_simulations=800)
action = agent.select_action(azul_state)

# Save trained model
neural_network.save_model("my_azul_model.pth", epoch=100, training_step=5000)
```

### Example Implementation

See `../examples/mcts_example.py` for a comprehensive example with:
- **Basic MCTS Usage**: Simple MCTS search with neural network evaluation
- **Gameplay Demonstration**: MCTS agent playing actual Azul games
- **Network Configurations**: Comparison of small, medium, and large networks
- **Performance Benchmarking**: Speed tests across different configurations
- **MCTS Features**: Temperature effects, deterministic vs stochastic selection
- **Numerical State Integration**: Shows how neural networks use game state representation
- **Model Persistence**: Saving and loading trained PyTorch models
- **Custom Networks**: Creating networks with custom architectures

### Running the Example

```python
# Run the comprehensive MCTS example
python3 examples/mcts_example.py
```

This will demonstrate all eight key features:
1. Basic MCTS usage with Azul game states
2. MCTS agent gameplay demonstration
3. Neural network configuration comparison
4. Performance benchmarking
5. MCTS features (temperature, selection modes)
6. Numerical state representation integration
7. Model saving and loading
8. Custom network creation

## Interfaces

### GameState Protocol (Aligned with Azul GameState)

Your game state must implement:

```python
def get_legal_actions(self) -> List[Any]:
    """Return list of legal actions (Action objects for Azul)."""

def apply_action(self, action: Any) -> Union['GameState', bool]:
    """Apply action and return new state OR modify in-place and return success boolean."""

@property
def game_over(self) -> bool:
    """Return True if this is a terminal state."""

@property
def current_player(self) -> int:
    """Return the current player index."""

def get_numerical_state(self) -> Any:
    """Get numerical representation of the state for neural network evaluation."""

def copy(self) -> 'GameState':
    """Create a deep copy of the game state."""
```

**Note**: The protocol supports both patterns:
- **Immutable states**: `apply_action` returns a new state
- **Mutable states** (like Azul `GameState`): `apply_action` modifies in-place and returns boolean

### NeuralNetwork Protocol

Your neural network must implement:

```python
def evaluate(self, state: GameState) -> Tuple[np.ndarray, float]:
    """
    Evaluate a game state.

    Args:
        state: Game state to evaluate

    Returns:
        Tuple of (action_probabilities, state_value)
        - action_probabilities: probabilities for each legal action
        - state_value: estimated value of the state for current player
    """
```

## Parameters

### MCTS Parameters

- `c_puct` (float): Exploration constant for UCT formula (default: 1.0)
- `num_simulations` (int): Number of MCTS simulations (default: 800)
- `temperature` (float): Temperature for action selection (default: 1.0)
  - 0.0 = deterministic (select best action)
  - 1.0 = proportional to visit counts
  - >1.0 = more random
- `dirichlet_alpha` (float): Alpha parameter for Dirichlet noise (default: 0.3)
- `dirichlet_epsilon` (float): Mixing parameter for Dirichlet noise (default: 0.25)

### Performance Considerations

- **Simulation Count**: More simulations = better play but slower
- **Neural Network Speed**: Faster NN evaluation = more simulations possible
- **Game Complexity**: Azul is complex but provides rich strategic gameplay
- **Tree Reuse**: Consider reusing subtrees between moves
- **Parallelization**: The algorithm can be parallelized across simulations

## Testing

Run the comprehensive test suite:

```bash
python -m pytest azul_rl/tests/test_mcts.py -v
```

The tests cover:
- Node operations and statistics
- Azul GameState functionality and MCTS protocol compatibility
- Neural network evaluation (AzulNeuralNetwork)
- MCTS search behavior
- Agent action selection
- Full Azul game simulations
- Direct Azul GameState integration (no adapters needed)

## Algorithm Details

### UCT Formula

The Upper Confidence Bound for Trees balances exploitation (Q values) and exploration (visit counts and priors):

```
UCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Where:
- `Q(s,a)`: Average value of action a from state s
- `P(s,a)`: Prior probability from neural network
- `N(s)`: Visit count of state s
- `N(s,a)`: Visit count of action a from state s
- `c_puct`: Exploration constant

### Backpropagation

Values are backpropagated up the tree with alternating signs to account for the adversarial nature of two-player games:

```python
# Update leaf node
leaf.N += 1
leaf.Q += value

# Update path with alternating signs
current_value = -value  # Flip for parent's perspective
for node, action in reversed(path):
    node.N += 1
    node.Q += current_value
    current_value = -current_value  # Flip for next level
```

### Dirichlet Noise

Exploration noise is added to the root node to encourage exploration:

```python
noise = np.random.dirichlet([alpha] * num_actions)
child.P = (1 - epsilon) * child.P + epsilon * noise[i]
```

### Dual State Pattern Support

The implementation handles both state modification patterns:

```python
# Make a copy to avoid modifying the original state
child_state = node.state.copy()

result = child_state.apply_action(action)

# If apply_action returns a boolean, the state was modified in-place
if isinstance(result, bool):
    if not result:
        continue  # Skip if action failed
    # child_state is already the modified state
else:
    # If apply_action returns a new state, use that
    child_state = result
```

## Performance Results

```
MCTS Performance Benchmark
============================================================
Testing small network:
  Simulations:  5 | Time: 0.193s | Sims/sec: 25.9 | Root visits: 5 | Children: 78
  Simulations: 10 | Time: 0.365s | Sims/sec: 27.4 | Root visits: 10 | Children: 78
  Simulations: 20 | Time: 0.766s | Sims/sec: 26.1 | Root visits: 20 | Children: 78
  Simulations: 50 | Time: 1.743s | Sims/sec: 28.7 | Root visits: 50 | Children: 78

Testing medium network:
  Simulations:  5 | Time: 0.170s | Sims/sec: 29.4 | Root visits: 5 | Children: 78
  Simulations: 10 | Time: 0.638s | Sims/sec: 15.7 | Root visits: 10 | Children: 78
  Simulations: 20 | Time: 0.736s | Sims/sec: 27.2 | Root visits: 20 | Children: 78
  Simulations: 50 | Time: 1.740s | Sims/sec: 28.7 | Root visits: 50 | Children: 78
```

**Key Insights**:
- Azul games: ~25-30 simulations per second
- Performance is consistent across simulation counts
- Numerical state representation provides efficient evaluation
- Complex game state with ~78 legal actions at start
- Medium networks show similar performance to small networks

## Key Improvements

### ✅ Direct Integration
- **No Adapters Needed**: MCTS protocol aligned with Azul GameState
- **Native Action Objects**: Works directly with Azul Action objects
- **Simplified Architecture**: Removed unnecessary abstraction layers

### ✅ Numerical State Leverage
- **Complete Integration**: Uses `get_numerical_state()` for NN evaluation
- **Consistent Hashing**: Deterministic evaluation based on numerical state
- **Efficient Representation**: 935-dimensional normalized state vector

### ✅ Robust State Handling
- **Dual Pattern Support**: Handles both immutable and mutable state patterns
- **Error Resilience**: Graceful handling of illegal actions
- **Memory Efficiency**: Proper state copying to avoid side effects

### ✅ PyTorch-Only Implementation
- **Real Neural Networks**: Uses actual PyTorch models instead of mock implementations
- **Multiple Configurations**: Small, medium, large, and deep architectures
- **GPU Support**: Automatic device selection and CUDA support
- **Model Persistence**: Save and load trained models
- **Training Integration**: Seamless integration with training pipeline

## References

- [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://www.nature.com/articles/nature16961)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- [A Survey of Monte Carlo Tree Search Methods](https://ieeexplore.ieee.org/document/6145622)
