# Azul Neural Network Architecture

This document provides a comprehensive overview of the neural network architecture implemented for the Azul board game AI system.

## Overview

The `AzulNetwork` is a PyTorch-based neural network designed specifically for the Azul board game. It implements a dual-head architecture with:

- **Input Layer**: Processes the flattened game state representation
- **Shared Body**: Fully connected layers with optional residual connections
- **Policy Head**: Outputs action probabilities using softmax activation
- **Value Head**: Outputs state value estimation using tanh activation

## Architecture Details

### Input Layer

The input layer processes the complete game state representation:

```python
Input Size: ~915 values (auto-detected from AzulStateRepresentation)
├── Global state (4 values)
├── Player scores (4 values)
├── Pattern lines (4×5×8 = 160 values)
├── Walls (4×5×5 = 100 values)
├── Floor lines (4×7×7 = 196 values)
├── First player markers (4 values)
├── Factories (9×4×6 = 216 values)
├── Center tiles (40×6 = 240 values)
├── Center first player marker (1 value)
└── Tile supply (2×5 = 10 values)
```

### Shared Body Layers

The shared body consists of fully connected layers with:

- **Batch Normalization**: Optional, enabled by default
- **ReLU Activation**: For non-linearity
- **Dropout**: Configurable rate (default: 0.1)
- **Residual Connections**: Optional, enabled by default

```python
# Example medium configuration
Input → Linear(915, 512) → BatchNorm → ReLU → Dropout
     → Linear(512, 512) → BatchNorm → ReLU → Dropout (+ residual)
     → Linear(512, 256) → BatchNorm → ReLU → Dropout (+ residual)
```

### Policy Head

Outputs action probabilities for the 500-dimensional action space:

```python
Features → Linear(256, 128) → BatchNorm → ReLU → Dropout
        → Linear(128, 500) → Softmax(temperature)
```

- **Temperature Scaling**: Configurable temperature for exploration control
- **Legal Action Masking**: Supports masking illegal actions
- **Probability Normalization**: Ensures probabilities sum to 1

### Value Head

Outputs state value estimation in the range [-1, 1]:

```python
Features → Linear(256, 128) → BatchNorm → ReLU → Dropout
        → Linear(128, 64) → ReLU
        → Linear(64, 1) → Tanh
```

## Predefined Configurations

### Small Configuration
```python
{
    'hidden_sizes': (256, 256),
    'dropout_rate': 0.1,
    'use_batch_norm': True,
    'use_residual': False,
}
```
- **Parameters**: ~400K
- **Use Case**: Fast training/inference, resource-constrained environments

### Medium Configuration (Default)
```python
{
    'hidden_sizes': (512, 512, 256),
    'dropout_rate': 0.1,
    'use_batch_norm': True,
    'use_residual': True,
}
```
- **Parameters**: ~1.2M
- **Use Case**: Balanced performance and efficiency

### Large Configuration
```python
{
    'hidden_sizes': (1024, 512, 512, 256),
    'dropout_rate': 0.15,
    'use_batch_norm': True,
    'use_residual': True,
}
```
- **Parameters**: ~2.5M
- **Use Case**: Maximum performance, sufficient computational resources

### Deep Configuration
```python
{
    'hidden_sizes': (512, 512, 512, 256, 256),
    'dropout_rate': 0.2,
    'use_batch_norm': True,
    'use_residual': True,
}
```
- **Parameters**: ~1.8M
- **Use Case**: Complex pattern learning, experimental deep architectures

## Usage Examples

### Basic Usage

```python
from azul_rl.training import create_azul_network
from azul_rl.game import GameState, AzulStateRepresentation

# Create network
network = create_azul_network('medium')

# Create game state
game = GameState(num_players=2, seed=42)
state_repr = AzulStateRepresentation(game)
state_vector = state_repr.get_flat_state_vector(normalize=True)

# Make prediction
action_probs, state_value = network.predict(state_vector)
```

### With Legal Action Masking

```python
from azul_rl.game.pettingzoo_env import AzulAECEnv
import numpy as np

# Get legal actions
legal_actions = game.get_legal_actions()

# Create action mask
env = AzulAECEnv(num_players=2)
action_mask = np.zeros(500)
for action in legal_actions:
    action_idx = env._encode_action(action)
    if 0 <= action_idx < 500:
        action_mask[action_idx] = 1.0

# Predict with masking
action_probs, state_value = network.predict(state_vector, action_mask)
```

### Batch Processing

```python
import torch

# Prepare batch of states
states_batch = torch.FloatTensor(multiple_state_vectors)
masks_batch = torch.FloatTensor(multiple_action_masks)

# Forward pass
network.eval()
with torch.no_grad():
    policy_probs, values = network(states_batch)
    masked_probs = network.get_action_probabilities(states_batch, masks_batch)
```

### Custom Configuration

```python
# Create custom network
custom_network = AzulNetwork(
    hidden_sizes=(1024, 256, 128),
    dropout_rate=0.05,
    use_batch_norm=False,
    use_residual=True
)

# Or override predefined config
network = create_azul_network('large', dropout_rate=0.05)
```

## Training Integration

### Loss Functions

The network is designed to work with standard RL loss functions:

```python
# Policy loss (cross-entropy)
policy_loss = -torch.sum(target_policy * torch.log(predicted_policy + 1e-8))

# Value loss (MSE)
value_loss = torch.nn.functional.mse_loss(predicted_value, target_value)

# Combined loss
total_loss = policy_loss + value_loss_weight * value_loss
```

### Optimization

Recommended optimizer settings:

```python
optimizer = torch.optim.Adam(
    network.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1000,
    gamma=0.95
)
```

## Performance Characteristics

### Inference Speed

| Configuration | Forward Pass (ms) | Parameters | Memory (MB) |
|---------------|-------------------|------------|-------------|
| Small         | ~2                | 400K       | ~15         |
| Medium        | ~3                | 1.2M       | ~25         |
| Large         | ~5                | 2.5M       | ~45         |
| Deep          | ~4                | 1.8M       | ~35         |

*Benchmarks on CPU (Intel i7) with batch size 1*

### Memory Usage

- **Training**: ~2-3x inference memory due to gradients
- **Batch Processing**: Linear scaling with batch size
- **GPU Acceleration**: Significant speedup for batch sizes > 32

## Integration with RL Algorithms

### Deep Q-Networks (DQN)

```python
# Use policy head as Q-value estimator
q_values = network.policy_head(features)
action = torch.argmax(q_values, dim=-1)
```

### Policy Gradient Methods

```python
# Use both heads
action_probs, baseline = network(state)
action = torch.multinomial(action_probs, 1)
```

### Actor-Critic Methods

```python
# Policy head as actor, value head as critic
actor_probs, critic_value = network(state)
```

### Monte Carlo Tree Search (MCTS)

```python
# Use for node evaluation and expansion
def evaluate_node(state):
    probs, value = network.predict(state)
    return probs, value.item()
```

## Best Practices

### Training

1. **Batch Normalization**: Keep enabled for stable training
2. **Residual Connections**: Use for deeper networks (3+ layers)
3. **Dropout**: Start with 0.1, increase for overfitting
4. **Temperature**: Use 1.0 for training, adjust for inference
5. **Legal Action Masking**: Always mask illegal actions

### Inference

1. **Evaluation Mode**: Always call `network.eval()` for inference
2. **No Gradients**: Use `torch.no_grad()` context
3. **Batch Processing**: Use batches for multiple predictions
4. **Temperature Tuning**: Lower for exploitation, higher for exploration

### Memory Management

1. **Gradient Accumulation**: For large effective batch sizes
2. **Mixed Precision**: Use `torch.cuda.amp` for GPU training
3. **Checkpoint Saving**: Save model state regularly
4. **Memory Profiling**: Monitor GPU memory usage

## Validation and Testing

The network includes comprehensive tests covering:

- **Architecture Validation**: Layer creation and connectivity
- **Forward Pass**: Output shapes and value ranges
- **Legal Action Masking**: Correct probability masking
- **Batch Processing**: Multiple state handling
- **Game Integration**: Real game state processing
- **Configuration Testing**: All predefined configurations

Run tests with:
```bash
python -m pytest azul_rl/tests/test_neural_network.py -v
```

## Future Extensions

### Potential Improvements

1. **Attention Mechanisms**: For better factory/center processing
2. **Convolutional Layers**: For spatial pattern recognition
3. **Transformer Architecture**: For sequence modeling
4. **Multi-Task Learning**: Joint training on multiple objectives
5. **Ensemble Methods**: Multiple network combination

### Research Directions

1. **Architecture Search**: Automated architecture optimization
2. **Regularization**: Advanced techniques for generalization
3. **Transfer Learning**: Pre-training on related games
4. **Interpretability**: Understanding learned representations
5. **Efficiency**: Model compression and quantization

## Troubleshooting

### Common Issues

1. **NaN Values**: Check learning rate, gradient clipping
2. **Poor Convergence**: Adjust architecture, learning rate
3. **Overfitting**: Increase dropout, add regularization
4. **Memory Issues**: Reduce batch size, use gradient accumulation
5. **Slow Training**: Use GPU, optimize data loading

### Debugging Tools

```python
# Check model info
info = network.get_model_info()
print(f"Parameters: {info['total_parameters']:,}")

# Visualize gradients
for name, param in network.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item():.6f}")

# Monitor activations
_, _, features = network(state, return_features=True)
print(f"Feature statistics: {features.mean():.3f} ± {features.std():.3f}")
```

The AzulNetwork provides a robust foundation for reinforcement learning in the Azul board game, with flexibility for various training scenarios and research applications.
