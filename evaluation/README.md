# Agent Evaluation Framework

A comprehensive evaluation system for Azul AI agents that enables systematic performance assessment through head-to-head gameplay, statistical analysis, and tournament-style comparisons.

## 🚀 Features

### Core Capabilities
- **Head-to-Head Evaluation**: Pit any agent against baseline agents (random, heuristic, or previous checkpoints)
- **Statistical Analysis**: Calculate win rates, confidence intervals, and statistical significance
- **Tournament System**: Run round-robin tournaments between multiple agents
- **Parallel Processing**: Support for multi-threaded evaluation for faster results
- **Comprehensive Logging**: Detailed game logs, replay saving, and performance metrics
- **Checkpoint Comparison**: Evaluate current models against previous training iterations

### Baseline Agents
- **Random Agent**: Selects actions uniformly at random
- **Simple Heuristic Agent**: Uses basic game knowledge and simple rules
- **Full Heuristic Agent**: Sophisticated rule-based agent with strategic evaluation
- **Checkpoint Agent**: Loads and uses previous model checkpoints

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
from evaluation import AgentEvaluator, EvaluationConfig, RandomAgent
from agents.heuristic_agent import HeuristicAgent

# Create configuration
config = EvaluationConfig(
    num_games=100,
    timeout_per_move=3.0,
    verbose=True
)

# Create agents
test_agent = HeuristicAgent()
baseline_agent = RandomAgent()

# Run evaluation
evaluator = AgentEvaluator(config)
result = evaluator.evaluate_agent(test_agent, baseline_agent)

# Print results
print(result.summary())
```

### Command Line Usage

```bash
# Quick evaluation
python -m evaluation.cli quick --test-agent heuristic

# Full evaluation
python -m evaluation.cli evaluate \
  --test-agent mcts --baseline-agent random \
  --num-games 100 --output results.json

# Tournament
python -m evaluation.cli tournament \
  --agents "MCTS:mcts" "Heuristic:heuristic" \
  --include-baselines --output tournament_results
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
from evaluation.baseline_agents import create_baseline_agent

# Create tournament
config = EvaluationConfig(num_games=50)
tournament = Tournament(config)

# Add agents
tournament.add_agent(create_baseline_agent('random'), 'Random')
tournament.add_agent(create_baseline_agent('heuristic'), 'Heuristic')
tournament.add_agent(your_agent, 'YourAgent')

# Run tournament
result = tournament.run_tournament()
print(result.summary())
```

### Checkpoint Progression Tournament

```python
from evaluation.tournament import run_checkpoint_progression_tournament

# Compare multiple training checkpoints
checkpoint_paths = [
    'models/checkpoint_100.pth',
    'models/checkpoint_200.pth', 
    'models/checkpoint_300.pth',
]

result = run_checkpoint_progression_tournament(
    checkpoint_paths=checkpoint_paths,
    checkpoint_names=['Early', 'Mid', 'Late'],
    include_baselines=True
)
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

### Comparing Multiple Results

```python
from evaluation.utils import compare_agents

# Load multiple evaluation results
results_a = [load_evaluation_results('agent_a_1.json'), ...]
results_b = [load_evaluation_results('agent_b_1.json'), ...]

comparison = compare_agents(results_a, results_b, 'Agent A', 'Agent B')
print(comparison)
```

## 🎮 Agent Types

### Creating Test Agents

```python
# Heuristic Agent
from agents.heuristic_agent import HeuristicAgent
agent = HeuristicAgent(player_id=0)

# MCTS Agent
from agents.mcts import MCTSAgent
from training.neural_network import AzulNetwork
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = AzulNetwork(input_size=471, hidden_size=256, num_actions=1430, device=device)
agent = MCTSAgent(neural_network=network)
```

### Creating Baseline Agents

```python
from evaluation.baseline_agents import create_baseline_agent

# Built-in baseline types
random_agent = create_baseline_agent('random', seed=42)
simple_agent = create_baseline_agent('simple_heuristic')
heuristic_agent = create_baseline_agent('heuristic')

# Checkpoint agent
checkpoint_agent = create_baseline_agent('checkpoint', 
                                       checkpoint_path='models/best_model.pth')
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
  --test-agent mcts \
  --baseline-agent heuristic \
  --num-games 200 \
  --workers 4 \
  --output detailed_results.json

# Tournament between agents
python -m evaluation.cli tournament \
  --agents "MyAgent:mcts:num_simulations=800" \
  --agents "Baseline:heuristic" \
  --include-baselines \
  --num-games 100 \
  --output tournament_2024

# Generate summary from results directory
python -m evaluation.cli summary evaluation_results/ --output summary.txt
```

### Agent Specification Format

For tournaments, specify agents using the format: `name:type[:arg1=val1,arg2=val2,...]`

```bash
# Examples
--agents "MCTS_Fast:mcts:num_simulations=100,temperature=0.0"
--agents "Checkpoint_v2:checkpoint:checkpoint_path=models/v2.pth"
--agents "Random_Seed42:random:seed=42"
```

## 📊 Example Use Cases

### 1. Development Testing
```python
# Quick feedback during development
evaluator = AgentEvaluator()
result = evaluator.quick_evaluation(my_agent, RandomAgent(), num_games=10)
print(f"Quick test: {result.test_agent_win_rate:.1%} win rate")
```

### 2. Model Validation
```python
# Thorough evaluation before deployment
config = EvaluationConfig(
    num_games=500,
    num_workers=8,
    confidence_interval=0.99
)
evaluator = AgentEvaluator(config)
result = evaluator.evaluate_agent(my_model, heuristic_baseline)
```

### 3. Training Progress Tracking
```python
# Compare against previous checkpoint
from evaluation.baseline_agents import CheckpointAgent

previous_model = CheckpointAgent('models/checkpoint_100.pth')
current_model = my_current_model

result = evaluator.evaluate_agent(current_model, previous_model)
improvement = result.test_agent_win_rate > 0.5
print(f"Model improved: {improvement}")
```

### 4. Hyperparameter Comparison
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

### Custom Baseline Agents

```python
from evaluation.baseline_agents import BaselineAgent

class MyCustomBaseline(BaselineAgent):
    def __init__(self):
        super().__init__(name="CustomBaseline")
    
    def select_action(self, game_state):
        # Your custom logic here
        actions = game_state.get_legal_actions()
        return actions[0]  # Simple example
```

### Parallel Evaluation

```python
# Use multiple workers for faster evaluation
config = EvaluationConfig(
    num_games=1000,
    num_workers=8,  # Use 8 parallel workers
    timeout_per_move=10.0
)

# Each worker runs games independently
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
- Include multiple baseline types

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
- Track performance over time with checkpoint evaluations
- Use tournaments to compare multiple approaches

## 🐛 Troubleshooting

### Common Issues

**Agent Compatibility**: Ensure your agent has a `select_action(game_state)` method.

**Memory Issues**: Reduce `num_workers` or `num_games` for memory-constrained environments.

**Timeout Errors**: Increase `timeout_per_move` for slower agents.

**Checkpoint Loading**: Verify checkpoint file format and model architecture compatibility.

### Error Handling

The framework includes robust error handling:
- Invalid actions are caught and logged
- Timeouts are tracked in results
- Failed games are recorded with error messages
- Evaluation continues even if individual games fail

## 📚 Examples

See `examples/evaluation_examples.py` for comprehensive usage examples, including:
- Basic evaluation setup
- Agent comparison workflows
- Tournament organization
- Statistical analysis
- Configuration options
- Checkpoint evaluation

Run the examples:
```bash
python examples/evaluation_examples.py
```

## 🤝 Integration

The evaluation framework integrates seamlessly with:
- **Training Pipeline**: Evaluate models during training
- **Model Selection**: Compare different architectures
- **Hyperparameter Tuning**: Systematic parameter evaluation
- **Research**: Reproducible experimental results
- **Production**: Validate models before deployment

## 📄 API Reference

### Core Classes
- `AgentEvaluator`: Main evaluation orchestrator
- `EvaluationConfig`: Configuration management
- `EvaluationResult`: Result container with statistics
- `Tournament`: Multi-agent tournament system
- `BaselineAgent`: Base class for baseline implementations

### Utility Functions
- `format_evaluation_results()`: Human-readable result formatting
- `save_evaluation_results()` / `load_evaluation_results()`: Persistence
- `calculate_confidence_interval()`: Statistical confidence intervals
- `calculate_statistical_significance()`: Significance testing

For detailed API documentation, see the docstrings in each module. 