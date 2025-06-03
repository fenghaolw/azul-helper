# Agent Evaluation Framework

A comprehensive evaluation system for Azul AI agents that enables systematic performance assessment through head-to-head gameplay, statistical analysis, and tournament-style comparisons.

## 🚀 Features

### Core Capabilities
- **Head-to-Head Evaluation**: Compare any two agents directly through gameplay
- **Statistical Analysis**: Calculate win rates, confidence intervals, and statistical significance
- **Tournament System**: Run round-robin tournaments between multiple agents
- **Parallel Processing**: Support for multi-threaded evaluation for faster results
- **Comprehensive Logging**: Detailed game logs, replay saving, and performance metrics
- **Unified Architecture**: All agents inherit from a common base class for consistent behavior

### Available Agent Types
- **Random Agent**: Selects actions uniformly at random
- **Heuristic Agent**: Uses rule-based strategies with basic game knowledge
- **Improved Heuristic Agent**: Advanced rule-based agent with sophisticated evaluation
- **Minimax Agent**: Custom minimax with alpha-beta pruning and iterative deepening
- **OpenSpiel Minimax Agent**: OpenSpiel's optimized minimax implementation
- **MCTS Agent**: Monte Carlo Tree Search using OpenSpiel's implementation

### Statistical Features
- **Confidence Intervals**: Wilson score intervals for win rate estimation
- **Significance Testing**: Fisher's exact test and chi-square tests
- **Position Swapping**: Evaluate agents from different starting positions
- **Reproducible Results**: Fixed seeds for consistent evaluation

## 📦 Installation

The evaluation framework is included with the Azul Helper project. Ensure you have the required dependencies:

```bash
pip install scipy numpy torch
```

## 🎯 Quick Start

### Basic Evaluation

```python
from evaluation import AgentEvaluator, EvaluationConfig
from agents import HeuristicAgent, RandomAgent

# Create configuration
config = EvaluationConfig(
    num_games=100,
    timeout_per_move=3.0,
    verbose=True
)

# Create agents
test_agent = HeuristicAgent()
opponent_agent = RandomAgent()

# Run evaluation
evaluator = AgentEvaluator(config)
result = evaluator.evaluate_agent(test_agent, opponent_agent)

# Print results
print(result.summary())
```

### Command Line Usage

```bash
# Quick evaluation
python -m evaluation.cli quick --test-agent heuristic

# Full evaluation
python -m evaluation.cli evaluate \
  --agent-a minimax --agent-b random \
  --num-games 100 --output results.json

# Tournament
python -m evaluation.cli tournament \
  --agents "MCTS:mcts" "Minimax:minimax" "Heuristic:heuristic" \
  --output tournament_results
```

## 📊 Configuration Options

### EvaluationConfig Parameters

```python
config = EvaluationConfig(
    # Basic settings
    num_games=100,                    # Number of games to play
    num_players=2,                    # Always 2 for Azul
    timeout_per_move=5.0,             # Seconds per move
    
    # Evaluation settings  
    deterministic_evaluation=True,     # Use deterministic mode if available
    swap_player_positions=True,        # Test both starting positions
    
    # Randomization
    use_fixed_seeds=True,             # Reproducible results
    random_seed=42,                   # Base random seed
    
    # Performance
    num_workers=1,                    # Parallel workers
    
    # Output
    verbose=True,                     # Progress updates
    save_detailed_logs=False,         # Detailed game logs
    save_game_replays=False,          # Full game replays
    
    # Statistics
    confidence_interval=0.95,         # Confidence level
)
```

## 🏆 Tournament System

### Basic Tournament

```python
from evaluation import Tournament, EvaluationConfig
from agents import HeuristicAgent, MinimaxAgent, RandomAgent

# Create tournament
config = EvaluationConfig(num_games=50)
tournament = Tournament(config)

# Add agents
tournament.add_agent(RandomAgent(), 'Random')
tournament.add_agent(HeuristicAgent(), 'Heuristic')
tournament.add_agent(MinimaxAgent(), 'Minimax')

# Run tournament
result = tournament.run_tournament()
print(result.summary())
```

## 📈 Statistical Analysis

### Win Rate Analysis

```python
# The framework automatically calculates:
result.test_agent_win_rate        # Win rate as float (0.0 to 1.0)
result.confidence_interval        # (lower_bound, upper_bound)
result.p_value                    # Statistical significance
result.is_statistically_significant  # Boolean result
```

## 🎮 Agent Types

### Creating Agents

```python
# Available agent types
from agents import (
    HeuristicAgent,
    ImprovedHeuristicAgent,
    MinimaxAgent,
    RandomAgent,
    MCTSAgent,
    OpenSpielMinimaxAgent
)

# Heuristic agents
heuristic_agent = HeuristicAgent(player_id=0)
improved_agent = ImprovedHeuristicAgent(player_id=0)

# Minimax agents
minimax_agent = MinimaxAgent(player_id=0, time_limit=2.0, max_depth=4)
openspiel_minimax = OpenSpielMinimaxAgent(depth=4, enable_alpha_beta=True)

# MCTS agent
mcts_agent = MCTSAgent(num_simulations=400, uct_c=1.4)

# Random agent
random_agent = RandomAgent(seed=42)
```

## 📁 Results and Output

### Result Structure

```python
result = evaluator.evaluate_agent(test_agent, baseline_agent)

# Access results
print(f"Win rate: {result.test_agent_win_rate:.1%}")
print(f"Games played: {result.games_played}")
print(f"Average score difference: {result.average_score_difference:+.1f}")
print(f"P-value: {result.p_value:.4f}")

# Individual game results
for game_result in result.game_results:
    print(f"Game {game_result.game_id}: Winner {game_result.winner}, "
          f"Scores {game_result.final_scores}")
```

### Saving and Loading Results

```python
from evaluation.utils import save_evaluation_results, load_evaluation_results

# Save results
save_evaluation_results(result, 'my_evaluation.json')

# Load results
loaded_result = load_evaluation_results('my_evaluation.json')

# Generate summary report
summary = create_evaluation_summary('evaluation_results/', 'summary.txt')
```

## 🛠️ Command Line Interface

### Available Commands

```bash
# Quick evaluation (10 games)
python -m evaluation.cli quick --test-agent heuristic

# Full evaluation with options
python -m evaluation.cli evaluate \
  --agent-a mcts \
  --agent-b minimax \
  --num-games 200 \
  --workers 4 \
  --output detailed_results.json

# Tournament between agents
python -m evaluation.cli tournament \
  --agents "MyMCTS:mcts:num_simulations=800" \
  --agents "Minimax:minimax:time_limit=2.0,max_depth=6" \
  --agents "Heuristic:heuristic" \
  --num-games 100 \
  --output tournament_2024

# Generate summary from results directory
python -m evaluation.cli summary evaluation_results/ --output summary.txt
```

### Supported Agent Types

The following agent types are available in the CLI:

- `heuristic` - Basic rule-based agent
- `improved_heuristic` - Advanced rule-based agent
- `minimax` - Custom minimax with alpha-beta pruning
- `openspiel_minimax` - OpenSpiel's optimized minimax
- `mcts` - Monte Carlo Tree Search
- `random` - Random action selection

### Agent Parameter Format

For tournaments, specify agents using the format: `name:type[:arg1=val1,arg2=val2,...]`

```bash
# Examples
--agents "MCTS_Fast:mcts:num_simulations=100,uct_c=1.0"
--agents "Minimax_Deep:minimax:max_depth=6,time_limit=3.0"
--agents "Random_Seed42:random:seed=42"
```

### Common Agent Parameters

#### Minimax Agent Parameters
- `time_limit` - Maximum time per move (default: 1.0)
- `max_depth` - Maximum search depth (default: 4)

#### MCTS Agent Parameters
- `num_simulations` - Number of MCTS simulations (default: 400)
- `uct_c` - UCT exploration constant (default: 1.4)
- `solve` - Use exact solver when possible (default: False)

#### OpenSpiel Minimax Parameters
- `depth` - Search depth (default: 4)
- `enable_alpha_beta` - Use alpha-beta pruning (default: True)
- `time_limit` - Optional time limit in seconds

## 📊 Example Use Cases

### 1. Development Testing
```python
# Quick feedback during development
evaluator = AgentEvaluator()
result = evaluator.quick_evaluation(my_agent, RandomAgent(), num_games=10)
print(f"Quick test: {result.test_agent_win_rate:.1%} win rate")
```

### 2. Algorithm Comparison
```python
# Compare different algorithms
config = EvaluationConfig(num_games=200, num_workers=4)
evaluator = AgentEvaluator(config)

# Compare minimax vs MCTS
minimax_agent = MinimaxAgent(time_limit=2.0, max_depth=5)
mcts_agent = MCTSAgent(num_simulations=500)

result = evaluator.evaluate_agent(minimax_agent, mcts_agent)
```

### 3. Parameter Tuning
```python
# Compare different configurations
agents = {
    'MCTS_100': MCTSAgent(num_simulations=100),
    'MCTS_500': MCTSAgent(num_simulations=500),
    'MCTS_1000': MCTSAgent(num_simulations=1000),
}

tournament = Tournament()
for name, agent in agents.items():
    tournament.add_agent(agent, name)

result = tournament.run_tournament()
```

## 🔧 Advanced Features

### Parallel Evaluation

```python
# Use multiple workers for faster evaluation
config = EvaluationConfig(
    num_games=1000,
    num_workers=8,  # Use 8 parallel workers
    timeout_per_move=10.0
)

evaluator = AgentEvaluator(config)
result = evaluator.evaluate_agent(test_agent, baseline_agent)
```

### Statistical Significance Testing

```python
from evaluation.utils import calculate_statistical_significance

# Compare two sets of results
p_value, is_significant = calculate_statistical_significance(
    wins_a=80, total_a=100,
    wins_b=60, total_b=100,
    alpha=0.05
)

print(f"P-value: {p_value:.4f}")
print(f"Significant difference: {is_significant}")
```

## 📝 Best Practices

### 1. Evaluation Design
- Use at least 100 games for meaningful statistics
- Enable position swapping for fair comparison
- Use fixed seeds for reproducible results
- Test multiple agent configurations

### 2. Statistical Considerations
- Check confidence intervals, not just win rates
- Test statistical significance for important comparisons
- Use larger sample sizes for close comparisons
- Consider multiple evaluation runs

### 3. Performance Optimization
- Use parallel workers for large evaluations
- Adjust timeout based on agent complexity
- Save detailed logs only when needed
- Use quick evaluation during development

### 4. Result Analysis
- Save all evaluation results for later analysis
- Generate summary reports for multiple evaluations
- Track performance over time
- Use tournaments to compare multiple approaches

## 🐛 Troubleshooting

### Common Issues

**Agent Compatibility**: All agents inherit from `AzulAgent` base class with standardized interface.

**Memory Issues**: Reduce `num_workers` or `num_games` for memory-constrained environments.

**Timeout Errors**: Increase `timeout_per_move` for slower agents (especially minimax with deep search).

**Import Errors**: Ensure all dependencies are installed: `pip install scipy numpy torch`

### Error Handling

The framework includes robust error handling:
- Invalid actions are caught and logged
- Timeouts are tracked in results
- Failed games are recorded with error messages
- Evaluation continues even if individual games fail

## 🤝 Integration

The evaluation framework integrates seamlessly with:
- **Training Pipeline**: Evaluate models during training
- **Algorithm Development**: Compare different AI approaches
- **Parameter Tuning**: Systematic parameter evaluation
- **Research**: Reproducible experimental results
- **Benchmarking**: Standard performance comparisons

## 📄 API Reference

### Core Classes
- `AgentEvaluator`: Main evaluation orchestrator
- `EvaluationConfig`: Configuration management
- `EvaluationResult`: Result container with statistics
- `Tournament`: Multi-agent tournament system
- `AzulAgent`: Base class for all agents

### Utility Functions
- `format_evaluation_results()`: Human-readable result formatting
- `save_evaluation_results()` / `load_evaluation_results()`: Persistence
- `calculate_confidence_interval()`: Statistical confidence intervals
- `calculate_statistical_significance()`: Significance testing

For detailed API documentation, see the docstrings in each module. 