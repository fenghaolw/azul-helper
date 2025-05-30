# Azul Neural Network Training System

This directory contains the complete training infrastructure for the Azul neural network using AlphaZero-style self-play reinforcement learning.

## Quick Start

### 1. Run Initial Training (Demo)

For a quick demonstration of the full training pipeline:

```bash
python -m training.run_initial_training
```

This runs a short training session (10 iterations) with reduced parameters to demonstrate the complete self-play → training → evaluation loop.

### 2. Run Full Training

For production training with default hyperparameters:

```bash
python -m training.full_training_run
```

With custom settings:

```bash
python -m training.full_training_run \
    --iterations 1000 \
    --mcts-sims 800 \
    --batch-size 512 \
    --learning-rate 0.001 \
    --network medium
```

### 3. Monitor Training Progress

#### ETA Tracking
The training system includes sophisticated progress tracking with estimated completion times:
- Real-time progress bars and completion estimates
- Phase-based timing (self-play, training, evaluation)
- Multiple estimation methods for accurate predictions

Enable detailed ETA tracking:
```bash
python -m training.demo_eta_tracking  # Quick demonstration
```

See [ETA_TRACKING.md](ETA_TRACKING.md) for detailed documentation.

#### Real-time Dashboard (requires matplotlib)
```bash
python -m training.monitor_dashboard logs/full_training_*/training_log.json
```

#### TensorBoard
```bash
tensorboard --logdir=logs/full_training_*/tensorboard
```

#### Command-line Summary
```bash
python -m training.monitor_dashboard logs/full_training_*/training_log.json --summary
```

## Training Pipeline

The training system implements the full AlphaZero-style loop:

1. **Self-Play Generation**: MCTS-guided games generate training data
2. **Experience Storage**: Game states, MCTS policies, and outcomes stored in replay buffer
3. **Neural Network Training**: Sample batches and train with cross-entropy (policy) and MSE (value) losses
4. **Model Evaluation**: Periodic evaluation games to track improvement
5. **Checkpoint Management**: Save best models and recent checkpoints

## Key Components

### Core Modules

- **`training_loop.py`**: Main training orchestrator (`AzulTrainer`)
- **`self_play.py`**: Self-play engine for data generation
- **`neural_network.py`**: Neural network architecture and interface
- **`replay_buffer.py`**: Experience storage and sampling
- **`training_utils.py`**: Training utilities (loss functions, batch sampling, checkpointing)

### Training Scripts

- **`full_training_run.py`**: Production training with monitoring
- **`run_initial_training.py`**: Quick demonstration script
- **`monitor_dashboard.py`**: Real-time training visualization

## Hyperparameters

### Default Settings

```python
{
    # Self-play
    "self_play_games_per_iteration": 100,
    "mcts_simulations": 800,
    "temperature": 1.0,
    "temperature_threshold": 30,
    
    # Replay buffer
    "buffer_capacity": 100000,
    "min_buffer_size": 5000,
    
    # Training
    "batch_size": 512,
    "learning_rate": 0.001,
    "training_steps_per_iteration": 1000,
    
    # Model
    "network_config": "medium",  # small/medium/large
    
    # Evaluation
    "eval_games": 50,
    "eval_frequency": 5,
    
    # General
    "max_iterations": 1000,
    "device": "auto"  # auto/cpu/cuda/mps
}
```

### Network Configurations

- **Small**: 4 residual blocks, 128 channels (fast training)
- **Medium**: 8 residual blocks, 256 channels (balanced)
- **Large**: 16 residual blocks, 512 channels (highest capacity)

## Monitoring & Logging

### Metrics Tracked

- **Loss Values**: Total, policy (cross-entropy), value (MSE)
- **Self-Play Stats**: Games played, buffer size, avg moves/game
- **Evaluation Scores**: Model performance, best score tracking
- **Training Efficiency**: Iteration time, total elapsed time

### Output Files

```
logs/full_training_YYYYMMDD_HHMMSS/
├── hyperparameters.json      # Training configuration
├── training_log.json         # Detailed iteration logs
├── final_results.json        # Final training summary
└── tensorboard/              # TensorBoard event files

models/full_training_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── checkpoint_iter_0010_score_0.1234.pt
│   ├── checkpoint_iter_0020_score_0.2345.pt
│   └── best_checkpoint.pt -> checkpoint_iter_0020_score_0.2345.pt
└── training_history.json
```

## Advanced Usage

### Resume Training

```bash
python -m training.full_training_run \
    --resume models/full_training_*/checkpoints/checkpoint_iter_0100.pt
```

### Custom Training Loop

```python
from training.training_loop import AzulTrainer, TrainingConfig

# Create custom configuration
config = TrainingConfig(
    self_play_games_per_iteration=200,
    mcts_simulations=1600,
    batch_size=1024,
    learning_rate=0.0001,
    network_config="large",
    max_iterations=5000
)

# Initialize trainer
trainer = AzulTrainer(config, save_dir="models/custom_training")

# Run training
results = trainer.train()
```

### Distributed Training (Future)

The system is designed to support distributed self-play and training:
- Multiple self-play workers can generate games in parallel
- Training can use multiple GPUs via PyTorch DDP
- Shared replay buffer for experience collection

## Tips for Effective Training

1. **Start Small**: Use `network_config="small"` for initial experiments
2. **Monitor Progress**: Watch loss curves and evaluation scores
3. **Adjust MCTS Simulations**: More simulations = better data but slower
4. **Buffer Management**: Ensure buffer fills before increasing batch size
5. **Learning Rate Schedule**: Consider decay after initial convergence
6. **Temperature Annealing**: Reduce exploration as training progresses

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller network
2. **Slow Training**: Reduce MCTS simulations or self-play games
3. **No Improvement**: Check if buffer is diverse, adjust learning rate
4. **Unstable Loss**: Reduce learning rate or clip gradients

### Performance Optimization

- **GPU Usage**: Ensure CUDA/MPS is available with `--device auto`
- **CPU Parallelism**: Set `OMP_NUM_THREADS` for torch operations
- **Batch Processing**: Larger batches generally train faster
- **Checkpoint Frequency**: Balance between safety and I/O overhead

## Evaluation

After training, evaluate your model:

```bash
# Test against heuristic agent
python run_evaluation.py \
    --agent1 checkpoint \
    --checkpoint models/full_training_*/checkpoints/best_checkpoint.pt \
    --agent2 heuristic \
    --num_games 100
```

## Contributing

When adding new features:
1. Update hyperparameter defaults if needed
2. Add new metrics to monitoring
3. Ensure checkpoint compatibility
4. Document changes in this README 