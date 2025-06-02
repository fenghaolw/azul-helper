# Azul RL Agents

This directory contains agent implementations for playing Azul. All MCTS implementations now use **OpenSpiel** for superior performance and reliability.

## Available Agents

### MCTS Agents (OpenSpiel-based)

- **OpenSpielMCTSAgent**: Robust MCTS implementation using OpenSpiel's optimized C++ backend
- **OpenSpielAlphaZeroAgent**: AlphaZero implementation using OpenSpiel (supports neural networks)

### Rule-based Agents

- **HeuristicAgent**: Basic rule-based agent with Azul strategy heuristics
- **ImprovedHeuristicAgent**: Advanced rule-based agent with strategic guidelines
- **MinimaxAgent**: Minimax with alpha-beta pruning and iterative deepening

### Utility Agents

- **RandomAgent**: OpenSpiel's random action selection (baseline for comparisons)

## Quick Start

### Using OpenSpiel MCTS (Recommended)

```python
from agents import MCTSAgent

# Create a strong MCTS agent
agent = MCTSAgent(
    num_simulations=800,  # Higher = stronger but slower
    uct_c=1.4,           # UCT exploration constant
    solve=False,         # Use MCTS-Solver for exact solutions (slower)
)

# Use the agent
action = agent.select_action(game_state, deterministic=True)
```

### Using Minimax Agent

```python
from agents import MinimaxAgent
from agents.minimax_agent import MinimaxConfig

# Create a minimax agent
config = MinimaxConfig.create_difficulty_preset("hard")
agent = MinimaxAgent(config=config)

action = agent.select_action(game_state)
```

### Using Heuristic Agents

```python
from agents import ImprovedHeuristicAgent

# Create an improved heuristic agent
agent = ImprovedHeuristicAgent()
action = agent.select_action(game_state)
```

### Using Random Agent (Baseline)

```python
from agents import RandomAgent

# Create a random agent for baseline comparison
agent = RandomAgent(seed=42)
action = agent.select_action(game_state)
```

## OpenSpiel Advantages

The OpenSpiel implementations provide several benefits over custom implementations:

- **Performance**: Optimized C++ backend with Python bindings
- **Reliability**: Extensively tested and maintained by Google Research
- **Features**: Includes MCTS-Solver for exact solutions when feasible
- **Simplicity**: No neural network required for strong MCTS play
- **Scalability**: Efficient memory management and parallelization

## Agent Comparison

| Agent | Strength | Speed | Features |
|-------|----------|-------|----------|
| OpenSpielMCTSAgent | High | Fast | Tree search, random rollouts |
| MinimaxAgent | High | Medium | Perfect information, alpha-beta pruning |
| ImprovedHeuristicAgent | Medium | Very Fast | Rule-based, strategic patterns |
| HeuristicAgent | Low-Medium | Very Fast | Basic rules |
| RandomAgent | Baseline | Very Fast | Random selection |

## Usage Examples

### Agent vs Agent Evaluation

```bash
# Compare MCTS vs Minimax
python run_evaluation.py mcts --games 10
python run_evaluation.py quick minimax --difficulty hard

# Quick tests
python run_evaluation.py quick heuristic
```

### Direct Agent Comparison

```python
from agents import MCTSAgent, MinimaxAgent, ImprovedHeuristicAgent
from agents.minimax_agent import MinimaxConfig

# Create different agents
mcts_agent = MCTSAgent(num_simulations=400)
minimax_agent = MinimaxAgent(config=MinimaxConfig.create_difficulty_preset("medium"))
heuristic_agent = ImprovedHeuristicAgent()

# Test them on the same game state
game_state = GameState(num_players=2, seed=42)
mcts_action = mcts_agent.select_action(game_state)
minimax_action = minimax_agent.select_action(game_state)
heuristic_action = heuristic_agent.select_action(game_state)
```

### Custom Configuration

```python
# High-strength MCTS with solver
strong_agent = MCTSAgent(
    num_simulations=1600,
    uct_c=1.0,
    solve=True,  # Use exact solver when possible
    max_memory=2000000  # Larger tree cache
)

# Fast MCTS for quick games
fast_agent = MCTSAgent(
    num_simulations=100,
    uct_c=2.0,  # More exploration for quick games
)
```

## Performance Notes

- **OpenSpiel MCTS**: ~1000+ simulations/second depending on hardware
- **Minimax**: Explores ~10K+ nodes/second with alpha-beta pruning  
- **Heuristic agents**: ~100K+ evaluations/second
- **Random agent**: Nearly instant action selection

For best performance:
1. Use OpenSpiel MCTS for strong gameplay without neural networks
2. Use Minimax for perfect information analysis
3. Use Improved Heuristic for very fast reasonable play
4. Use Random for baseline comparisons

## Migration Notes

**Removed agents** (replaced by better alternatives):
- ❌ **Custom RandomAgent** → Use `RandomAgent` (OpenSpiel version)
- ❌ **CheckpointAgent** → Use OpenSpiel AlphaZero training + MCTS evaluation

**All imports still work** via aliases in `agents/__init__.py`:
```python
from agents import MCTSAgent  # → OpenSpielMCTSAgent
from agents import RandomAgent  # → OpenSpiel RandomAgent
```

This ensures backward compatibility while using superior implementations.
