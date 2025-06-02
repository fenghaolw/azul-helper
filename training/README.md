# Azul RL Training

This directory contains training infrastructure for Azul neural networks. We have transitioned to **OpenSpiel AlphaZero** as our primary training method.

## 🚀 Recommended Training Method

### OpenSpiel AlphaZero (Primary)
Use `openspiel_alphazero_training.py` for all production training:

```bash
# Basic training
python training/openspiel_alphazero_training.py

# Custom configuration
python training/openspiel_alphazero_training.py \
  --num_iterations=200 \
  --num_self_play_games=100 \
  --num_mcts_simulations=800 \
  --learning_rate=0.001 \
  --checkpoint_dir=models/my_training

# Available parameters:
python training/openspiel_alphazero_training.py --help
```

**Advantages:**
- ✅ **Mature**: Google Research's battle-tested implementation
- ✅ **Optimized**: Highly efficient C++ backend
- ✅ **Reliable**: Extensively tested and validated
- ✅ **Complete**: Includes self-play, training, evaluation
- ✅ **Research-grade**: Used in academic papers

## 📁 File Status

### ✅ Active Files
- `openspiel_alphazero_training.py` - **Primary training script**
- `eta_tracker.py` - Time estimation utilities
- `README.md` - This documentation

### ⚠️ Deprecated Files (Legacy)
The following files are **deprecated** and will be removed in future versions:

- `training_loop.py` - Custom AlphaZero implementation (use OpenSpiel instead)
- `full_training_run.py` - Wrapper for custom training
- `self_play.py` - Custom self-play engine  
- `training_utils.py` - Utilities for custom training
- `neural_network.py` - Custom neural network (OpenSpiel has better models)
- `replay_buffer.py` - Custom replay buffer
- `run_initial_training.py` - Simple training runner
- `monitor_dashboard.py` - Custom monitoring

**Migration Path:** Switch to `openspiel_alphazero_training.py` for all new training.

## 🔧 Quick Start Examples

### Train a New Model
```bash
# Start fresh training
python training/openspiel_alphazero_training.py \
  --checkpoint_dir=models/my_azul_model \
  --num_iterations=100
```

### Continue Training
```bash
# Resume from checkpoint (if supported)
python training/openspiel_alphazero_training.py \
  --checkpoint_dir=models/my_azul_model \
  --num_iterations=200
```

### Evaluate Trained Model
```bash
# Use the trained model with OpenSpiel MCTS
python run_evaluation.py mcts --games=50
```

## 🏗️ Architecture

OpenSpiel AlphaZero provides:

1. **Self-Play**: Generate training games using MCTS + neural network
2. **Training**: Update neural network on self-play data  
3. **Evaluation**: Test model strength against baselines
4. **Checkpointing**: Save/load model checkpoints

All integrated into a single, reliable pipeline.

## 🔄 Migration Guide

### Old Custom Training
```python
# ❌ Deprecated approach
from training.training_loop import AzulTrainer, TrainingConfig

config = TrainingConfig(max_iterations=100)
trainer = AzulTrainer(config)
trainer.train()
```

### New OpenSpiel Training  
```bash
# ✅ Recommended approach
python training/openspiel_alphazero_training.py \
  --num_iterations=100 \
  --checkpoint_dir=models/azul_alphazero
```

## 📊 Performance Comparison

| Method | Speed | Reliability | Maintenance |
|--------|-------|-------------|-------------|
| **OpenSpiel AlphaZero** | ⚡ Fast | ✅ High | 🔧 Low |
| Custom Training | 🐌 Slower | ❓ Variable | 🛠️ High |

## 🎯 Future Plans

1. **Phase 1**: Deprecate custom training files
2. **Phase 2**: Remove deprecated files in next major version  
3. **Phase 3**: Extend OpenSpiel integration for custom research

Use `openspiel_alphazero_training.py` for all new projects! 