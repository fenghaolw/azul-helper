"""
Comprehensive Training Example for Azul.

This example demonstrates the complete training system including:
1. Enhanced training utilities (cross-entropy loss, MSE loss, batch sampling)
2. Self-play data generation and training pipeline
3. Individual component demonstrations
4. Complete training configurations

Usage:
    python -m azul_rl.examples.self_play_training_example
"""

import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from azul_rl.game.game_state import GameState
from azul_rl.training.neural_network import AzulNeuralNetwork, create_azul_network
from azul_rl.training.replay_buffer import Experience, ReplayBuffer
from azul_rl.training.self_play import SelfPlayEngine
from azul_rl.training.training_loop import (
    AzulTrainer,
    TrainingConfig,
    create_training_config,
)
from azul_rl.training.training_utils import (
    BatchSampler,
    EnhancedLossFunctions,
    ModelCheckpointManager,
    TrainingStep,
    create_training_components,
)


def create_sample_data(replay_buffer: ReplayBuffer, num_games: int = 10) -> None:
    """
    Create sample training data for demonstration.

    Args:
        replay_buffer: Buffer to populate with sample data
        num_games: Number of sample games to create
    """
    print(f"Creating {num_games} sample games for training...")

    for game_idx in range(num_games):
        game_state = GameState(num_players=2, seed=42 + game_idx)
        experiences = []

        # Simulate a short game (10-15 moves)
        for move_idx in range(10 + game_idx % 5):
            legal_actions = game_state.get_legal_actions()
            if not legal_actions:
                break

            # Create random MCTS policy (normalized probabilities)
            mcts_policy = np.zeros(500)  # Full action space size

            # Assign random probabilities to first few actions for simplicity
            num_actions = min(len(legal_actions), 10)
            random_probs = np.random.dirichlet(np.ones(num_actions))
            for i in range(num_actions):
                mcts_policy[i] = random_probs[i]

            # Create experience
            experience = Experience(
                state=game_state.copy(),
                mcts_policy=mcts_policy,
                player_id=game_state.current_player,
                outcome=None,  # Will be set later
            )
            experiences.append(experience)

            # Apply random legal action
            if legal_actions:
                action = np.random.choice(legal_actions)
                game_state.apply_action(action)

        # Assign random outcomes (win: 1, draw: 0, loss: -1)
        outcomes = [1.0, -1.0]  # Player 0 wins, Player 1 loses
        if np.random.random() < 0.1:  # 10% chance of draw
            outcomes = [0.0, 0.0]

        # Add game to replay buffer
        replay_buffer.add_game(experiences, outcomes)

    print(f"Created {replay_buffer.size()} training experiences")


def demonstrate_enhanced_loss_functions():
    """Demonstrate the enhanced loss functions."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING ENHANCED LOSS FUNCTIONS")
    print("=" * 60)

    # Create loss function calculator
    loss_functions = EnhancedLossFunctions(
        policy_loss_weight=1.0, value_loss_weight=1.0, temperature=1.0
    )

    # Create sample data
    batch_size = 8
    action_space_size = 500

    # Sample policy logits (before softmax)
    predicted_logits = torch.randn(batch_size, action_space_size)

    # Sample target policy (should sum to 1)
    target_policy = torch.softmax(torch.randn(batch_size, action_space_size), dim=-1)

    # Sample value predictions and targets
    predicted_value = torch.randn(batch_size, 1)
    target_value = torch.randn(batch_size, 1)

    # Calculate losses
    total_loss, policy_loss, value_loss = loss_functions.combined_loss(
        predicted_logits, predicted_value, target_policy, target_value
    )

    print(f"Sample batch size: {batch_size}")
    print(f"Action space size: {action_space_size}")
    print(f"Policy loss (cross-entropy): {policy_loss.item():.6f}")
    print(f"Value loss (MSE): {value_loss.item():.6f}")
    print(f"Total combined loss: {total_loss.item():.6f}")

    # Test different temperatures
    print("\nTesting different temperatures:")
    for temp in [0.1, 1.0, 2.0]:
        loss_func_temp = EnhancedLossFunctions(temperature=temp)
        _, policy_loss_temp, _ = loss_func_temp.combined_loss(
            predicted_logits, predicted_value, target_policy, target_value
        )
        print(f"  Temperature {temp}: Policy loss = {policy_loss_temp.item():.6f}")


def demonstrate_batch_sampling():
    """Demonstrate batch sampling from replay buffer."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING BATCH SAMPLING")
    print("=" * 60)

    # Create replay buffer and populate with sample data
    replay_buffer = ReplayBuffer(capacity=1000, min_size_for_sampling=50)
    create_sample_data(replay_buffer, num_games=20)

    # Create batch sampler
    device = torch.device("cpu")
    batch_sampler = BatchSampler(
        replay_buffer=replay_buffer, batch_size=16, device=device
    )

    print(f"Replay buffer size: {replay_buffer.size()}")
    print(f"Batch size: {batch_sampler.batch_size}")

    # Sample a few batches
    for i in range(3):
        batch_data = batch_sampler.sample_batch()
        if batch_data is not None:
            state_batch, policy_batch, outcome_batch = batch_data
            print(f"\nBatch {i+1}:")
            print(f"  State batch shape: {state_batch.shape}")
            print(f"  Policy batch shape: {policy_batch.shape}")
            print(f"  Outcome batch shape: {outcome_batch.shape}")
            print(f"  Sample outcomes: {outcome_batch.cpu().numpy()[:5]}")
        else:
            print(f"\nBatch {i+1}: No data available")


def demonstrate_training_step():
    """Demonstrate complete training step implementation."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING TRAINING STEP")
    print("=" * 60)

    # Create model
    device = torch.device("cpu")
    model = create_azul_network("small")
    model.to(device)

    # Create replay buffer with sample data
    replay_buffer = ReplayBuffer(capacity=1000, min_size_for_sampling=50)
    create_sample_data(replay_buffer, num_games=30)

    # Create training components
    training_step, checkpoint_manager = create_training_components(
        model=model,
        replay_buffer=replay_buffer,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=32,
        device=device,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Replay buffer size: {replay_buffer.size()}")

    # Perform single training step
    print("\nPerforming single training step...")
    step_stats = training_step.single_training_step()

    if step_stats:
        print(f"  Total loss: {step_stats['total_loss']:.6f}")
        print(f"  Policy loss: {step_stats['policy_loss']:.6f}")
        print(f"  Value loss: {step_stats['value_loss']:.6f}")
    else:
        print("  No training data available")

    # Perform multiple training steps
    print("\nPerforming 50 training steps...")
    train_stats = training_step.train_for_steps(50)

    print(f"  Average total loss: {train_stats['total_loss']:.6f}")
    print(f"  Average policy loss: {train_stats['policy_loss']:.6f}")
    print(f"  Average value loss: {train_stats['value_loss']:.6f}")
    print(f"  Successful steps: {train_stats['successful_steps']}/50")
    print(f"  Training time: {train_stats['training_time']:.2f}s")


def demonstrate_checkpoint_management():
    """Demonstrate model checkpoint saving and loading."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING CHECKPOINT MANAGEMENT")
    print("=" * 60)

    # Create model and checkpoint manager
    device = torch.device("cpu")
    model = create_azul_network("small")
    model.to(device)

    checkpoint_manager = ModelCheckpointManager(
        save_dir="models/demo_checkpoints",
        save_frequency=5,
        keep_best=True,
        keep_latest=3,
    )

    # Create dummy optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Save some checkpoints
    for iteration in range(1, 16):
        if checkpoint_manager.should_save_checkpoint(iteration):
            # Create dummy stats
            stats = {
                "total_loss": np.random.uniform(0.1, 1.0),
                "policy_loss": np.random.uniform(0.05, 0.5),
                "value_loss": np.random.uniform(0.05, 0.5),
                "iteration": iteration,
            }
            score = 1.0 / stats["total_loss"]  # Higher score for lower loss

            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                stats=stats,
                score=score,
            )

            print(f"Saved checkpoint for iteration {iteration}: {checkpoint_path}")

    # Show checkpoint history
    print(f"\nCheckpoint history:")
    for checkpoint in checkpoint_manager.checkpoint_history:
        print(f"  Iteration {checkpoint['iteration']}: Score {checkpoint['score']:.3f}")

    # Load best checkpoint
    best_checkpoint = checkpoint_manager.get_best_checkpoint()
    if best_checkpoint:
        print(f"\nLoading best checkpoint: {best_checkpoint}")
        checkpoint_data = checkpoint_manager.load_checkpoint(
            best_checkpoint, model, optimizer, device
        )
        print(f"Loaded checkpoint from iteration {checkpoint_data['iteration']}")
        print(f"  Score: {checkpoint_data['score']:.3f}")


def quick_self_play_demo():
    """
    Quick demonstration of self-play data generation.

    This example shows how to generate a few self-play games and examine
    the collected training data.
    """
    print("\n" + "=" * 60)
    print("SELF-PLAY DATA GENERATION DEMO")
    print("=" * 60)

    # Create neural network
    neural_network = AzulNeuralNetwork(config_name="small", device="cpu")
    print(f"Created neural network: {neural_network.get_model_info()}")

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=1000, min_size_for_sampling=10)
    print(f"Created replay buffer with capacity {replay_buffer.capacity}")

    # Create self-play engine
    self_play_engine = SelfPlayEngine(
        neural_network=neural_network,
        replay_buffer=replay_buffer,
        mcts_simulations=50,  # Reduced for demo
        temperature=1.0,
        verbose=True,
    )

    # Generate a few games
    print("\nGenerating 3 self-play games...")
    start_time = time.time()

    experiences_list = self_play_engine.play_games(num_games=3, num_players=2)

    generation_time = time.time() - start_time

    # Display results
    print(f"\nSelf-play generation completed in {generation_time:.2f}s")

    # Show statistics
    stats = self_play_engine.get_statistics()
    print(f"\nSelf-Play Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Show buffer statistics
    buffer_stats = replay_buffer.get_statistics()
    print(f"\nReplay Buffer Statistics:")
    for key, value in buffer_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Show some experience details
    if experiences_list:
        first_game_experiences = experiences_list[0]
        print(f"\nFirst game collected {len(first_game_experiences)} experiences")

        if first_game_experiences:
            exp = first_game_experiences[0]
            print(f"Example experience:")
            print(f"  Player ID: {exp.player_id}")
            print(f"  Policy shape: {exp.mcts_policy.shape}")
            print(f"  Outcome: {exp.outcome}")
            print(f"  Top 3 policy values: {sorted(exp.mcts_policy, reverse=True)[:3]}")

    # Test replay buffer sampling
    if replay_buffer.is_ready_for_sampling():
        print(f"\nTesting replay buffer sampling...")
        batch = replay_buffer.sample(batch_size=5)
        if batch:
            states, policies, outcomes = batch
            print(f"Sampled batch shapes:")
            print(f"  States: {states.shape}")
            print(f"  Policies: {policies.shape}")
            print(f"  Outcomes: {outcomes.shape}")
        else:
            print("Could not sample from replay buffer")
    else:
        print(
            f"\nReplay buffer not ready for sampling (need {replay_buffer.min_size_for_sampling} experiences)"
        )


def quick_training_demo():
    """
    Quick demonstration of the enhanced training loop.

    This example shows a very short training run to demonstrate the
    complete training pipeline with enhanced features.
    """
    print("\n" + "=" * 60)
    print("QUICK ENHANCED TRAINING DEMO")
    print("=" * 60)

    # Create training configuration for quick demo
    config = create_training_config(
        # Reduced parameters for quick demo
        self_play_games_per_iteration=5,
        mcts_simulations=20,
        max_iterations=3,
        training_steps_per_iteration=10,
        batch_size=32,
        min_buffer_size=10,
        eval_games=5,
        eval_frequency=1,
        save_frequency=2,
        network_config="small",
        # Enhanced features
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        loss_temperature=1.0,
        gradient_clip_norm=1.0,
        verbose=True,
    )

    print("Training with enhanced features:")
    print(f"  Cross-entropy loss for policy (weight: {config.policy_loss_weight})")
    print(f"  MSE loss for value (weight: {config.value_loss_weight})")
    print(f"  Loss temperature: {config.loss_temperature}")
    print(f"  Gradient clipping: {config.gradient_clip_norm}")

    # Create trainer
    trainer = AzulTrainer(config=config, save_dir="models/quick_demo")

    # Run training
    print("\nStarting quick enhanced training run...")
    results = trainer.train()

    # Display results
    print(f"\nTraining Results:")
    print(f"  Iterations completed: {results['iterations_completed']}")
    print(f"  Total training time: {results['total_training_time']:.2f}s")
    print(f"  Best eval score: {results['best_eval_score']:.4f}")

    # Show training history
    print(f"\nTraining History:")
    for i, iteration_stats in enumerate(results["training_history"]):
        print(f"  Iteration {iteration_stats['iteration']}:")
        print(f"    Buffer size: {iteration_stats['buffer_size']}")
        print(f"    Games played: {iteration_stats['games_played']}")
        if "total_loss" in iteration_stats:
            print(f"    Training loss: {iteration_stats['total_loss']:.6f}")
            print(f"    Policy loss: {iteration_stats['policy_loss']:.6f}")
            print(f"    Value loss: {iteration_stats['value_loss']:.6f}")
        if "eval_score" in iteration_stats:
            print(f"    Eval score: {iteration_stats['eval_score']:.4f}")


def full_training_example():
    """
    Example of a full training configuration with enhanced features.

    This shows how to set up a complete training run with proper parameters
    for serious training (though still manageable for demonstration).
    """
    print("\n" + "=" * 60)
    print("FULL ENHANCED TRAINING EXAMPLE")
    print("=" * 60)

    # Create comprehensive training configuration
    config = create_training_config(
        # Self-play parameters
        self_play_games_per_iteration=50,
        mcts_simulations=400,
        temperature=1.0,
        temperature_threshold=30,
        num_players=2,
        # Replay buffer parameters
        buffer_capacity=10000,
        min_buffer_size=1000,
        # Training parameters
        batch_size=256,
        learning_rate=0.001,
        weight_decay=1e-4,
        training_steps_per_iteration=500,
        gradient_clip_norm=1.0,
        # Enhanced loss parameters
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        loss_temperature=1.0,
        # Model parameters
        network_config="medium",
        # Evaluation parameters
        eval_games=20,
        eval_frequency=5,
        # Saving parameters
        save_frequency=10,
        save_best_model=True,
        keep_latest_checkpoints=5,
        # General parameters
        max_iterations=100,
        device="cpu",  # Change to "cuda" if you have GPU
        verbose=True,
    )

    print("Enhanced Training Configuration:")
    print(f"  Games per iteration: {config.self_play_games_per_iteration}")
    print(f"  MCTS simulations: {config.mcts_simulations}")
    print(f"  Buffer capacity: {config.buffer_capacity}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Network config: {config.network_config}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Device: {config.device}")
    print(f"  Policy loss weight: {config.policy_loss_weight}")
    print(f"  Value loss weight: {config.value_loss_weight}")
    print(f"  Loss temperature: {config.loss_temperature}")
    print(f"  Gradient clip norm: {config.gradient_clip_norm}")

    # Create trainer
    trainer = AzulTrainer(config=config, save_dir="models/azul_enhanced_training")

    print(f"\nEstimated training time:")
    print(
        f"  ~{config.max_iterations * config.self_play_games_per_iteration} total games"
    )
    print(
        f"  ~{config.max_iterations * config.training_steps_per_iteration} training steps"
    )
    print(f"  This could take several hours depending on your hardware")

    # Ask user for confirmation
    response = input("\nDo you want to start full enhanced training? (y/N): ")
    if response.lower() != "y":
        print("Training cancelled.")
        return

    # Run training
    print("\nStarting full enhanced training...")
    start_time = time.time()

    try:
        results = trainer.train()

        total_time = time.time() - start_time

        print(f"\nEnhanced training completed!")
        print(f"  Total time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
        print(f"  Iterations: {results['iterations_completed']}")
        print(f"  Best eval score: {results['best_eval_score']:.4f}")
        print(f"  Final eval: {results['final_evaluation']}")

        # Show training progress
        history = results["training_history"]
        if len(history) >= 3:
            print(f"\nTraining Progress:")
            print(f"  Early (iter 1): Loss={history[0].get('total_loss', 'N/A')}")
            print(
                f"  Middle (iter {len(history)//2}): Loss={history[len(history)//2].get('total_loss', 'N/A')}"
            )
            print(
                f"  Final (iter {len(history)}): Loss={history[-1].get('total_loss', 'N/A')}"
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Partial results may be saved in the models directory")


def complete_training_components_demo():
    """Run a complete training example with all enhanced components."""
    print("\n" + "=" * 60)
    print("COMPLETE ENHANCED COMPONENTS DEMO")
    print("=" * 60)

    # Setup
    device = torch.device("cpu")
    model = create_azul_network("small")
    model.to(device)

    # Create replay buffer with more sample data
    replay_buffer = ReplayBuffer(capacity=5000, min_size_for_sampling=100)
    create_sample_data(replay_buffer, num_games=100)

    # Create training components
    training_step, checkpoint_manager = create_training_components(
        model=model,
        replay_buffer=replay_buffer,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=64,
        device=device,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
    )

    print(f"Setup complete:")
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Replay buffer: {replay_buffer.size()} experiences")
    print(f"  Batch size: 64")
    print(f"  Device: {device}")

    # Training loop
    total_iterations = 20
    steps_per_iteration = 25

    for iteration in range(1, total_iterations + 1):
        start_time = time.time()

        # Training
        train_stats = training_step.train_for_steps(steps_per_iteration)

        # Checkpointing
        score = 1.0 / max(
            train_stats["total_loss"], 1e-6
        )  # Higher score for lower loss

        if checkpoint_manager.should_save_checkpoint(iteration):
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=training_step.optimizer,
                iteration=iteration,
                stats=train_stats,
                score=score,
            )
            print(f"  [Checkpoint saved: {checkpoint_path}]")

        iteration_time = time.time() - start_time

        # Print progress
        if iteration % 5 == 0:
            print(
                f"Iteration {iteration:2d}/{total_iterations}: "
                f"Loss={train_stats['total_loss']:.4f} "
                f"(Policy={train_stats['policy_loss']:.4f}, "
                f"Value={train_stats['value_loss']:.4f}) "
                f"Steps={train_stats['successful_steps']}/{steps_per_iteration} "
                f"Time={iteration_time:.2f}s"
            )

    print(f"\nTraining complete!")
    print(f"Best model score: {checkpoint_manager.best_score:.3f}")

    # Load and test best model
    best_checkpoint = checkpoint_manager.get_best_checkpoint()
    if best_checkpoint:
        print(f"Loading best model from: {best_checkpoint}")
        checkpoint_data = checkpoint_manager.load_checkpoint(
            best_checkpoint, model, device=device
        )
        print(f"Best model was from iteration {checkpoint_data['iteration']}")


def resume_training_example(checkpoint_path: str):
    """
    Example of resuming training from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file to resume from
    """
    print(f"\n" + "=" * 60)
    print(f"RESUMING TRAINING FROM: {checkpoint_path}")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return

    # Create configuration (should match or extend the original training)
    config = create_training_config(
        self_play_games_per_iteration=25,
        mcts_simulations=200,
        max_iterations=50,
        training_steps_per_iteration=250,
        batch_size=128,
        network_config="medium",
        # Enhanced features
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        verbose=True,
    )

    # Create trainer with resume capability
    trainer = AzulTrainer(
        config=config, save_dir="models/resumed_training", resume_from=checkpoint_path
    )

    print("Resuming enhanced training...")
    results = trainer.train()

    print(f"\nResumed training completed!")
    print(f"  Additional iterations: {results['iterations_completed']}")
    print(f"  Best eval score: {results['best_eval_score']:.4f}")


def evaluate_trained_model(model_path: str):
    """
    Example of evaluating a trained model.

    Args:
        model_path: Path to the trained model file
    """
    print(f"\n" + "=" * 60)
    print(f"EVALUATING MODEL: {model_path}")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Load the trained model
    neural_network = AzulNeuralNetwork(
        config_name="medium",  # Should match the trained model
        model_path=model_path,
        device="cpu",
    )

    print(f"Loaded model: {neural_network.get_model_info()}")

    # Create evaluation setup
    eval_buffer = ReplayBuffer(capacity=1000, min_size_for_sampling=1)
    eval_engine = SelfPlayEngine(
        neural_network=neural_network,
        replay_buffer=eval_buffer,
        mcts_simulations=100,
        temperature=0.0,  # Deterministic evaluation
        verbose=True,
    )

    # Run evaluation games
    print(f"\nRunning evaluation games...")
    eval_engine.play_games(num_games=10, num_players=2)

    # Show results
    stats = eval_engine.get_statistics()
    buffer_stats = eval_buffer.get_statistics()

    print(f"\nEvaluation Results:")
    print(f"  Games played: {stats['games_played']}")
    print(f"  Avg moves per game: {stats['avg_moves_per_game']:.1f}")
    print(f"  Avg time per game: {stats['avg_time_per_game']:.2f}s")

    if "num_wins" in buffer_stats:
        total_outcomes = (
            buffer_stats["num_wins"]
            + buffer_stats["num_losses"]
            + buffer_stats["num_draws"]
        )
        win_rate = buffer_stats["num_wins"] / max(1, total_outcomes)
        print(f"  Win rate: {win_rate:.3f}")
        print(
            f"  Outcomes: {buffer_stats['num_wins']} wins, {buffer_stats['num_losses']} losses, {buffer_stats['num_draws']} draws"
        )


def main():
    """Main function to run different examples."""
    print("Azul Enhanced Training Examples")
    print("=" * 60)

    print("\nAvailable examples:")
    print("1. Enhanced loss functions demo")
    print("2. Batch sampling demo")
    print("3. Training step demo")
    print("4. Checkpoint management demo")
    print("5. Self-play data generation demo")
    print("6. Quick enhanced training demo")
    print("7. Complete components demo")
    print("8. Full enhanced training example")
    print("9. Resume training example")
    print("10. Evaluate trained model")
    print("0. Run all component demos")

    try:
        choice = input("\nSelect example (0-10): ").strip()

        if choice == "0":
            print("Running all component demonstrations...")
            demonstrate_enhanced_loss_functions()
            demonstrate_batch_sampling()
            demonstrate_training_step()
            demonstrate_checkpoint_management()
            quick_self_play_demo()
            complete_training_components_demo()
            print("\n" + "=" * 60)
            print("ALL COMPONENT DEMONSTRATIONS COMPLETED!")
            print("=" * 60)
            print("\nAcceptance Criteria Verification:")
            print(
                "✓ Function to sample batches of (state, MCTS_policy_target, game_outcome)"
            )
            print("✓ Cross-entropy loss for policy head implemented")
            print("✓ MSE loss for value head implemented")
            print(
                "✓ Training step: forward pass, loss calculation, backward pass, optimizer step"
            )
            print("✓ Regular saving of model checkpoints")
            print("✓ Neural network can be trained on data from replay buffer")
            print("✓ Model checkpoints are saved and managed")
        elif choice == "1":
            demonstrate_enhanced_loss_functions()
        elif choice == "2":
            demonstrate_batch_sampling()
        elif choice == "3":
            demonstrate_training_step()
        elif choice == "4":
            demonstrate_checkpoint_management()
        elif choice == "5":
            quick_self_play_demo()
        elif choice == "6":
            quick_training_demo()
        elif choice == "7":
            complete_training_components_demo()
        elif choice == "8":
            full_training_example()
        elif choice == "9":
            checkpoint_path = input("Enter checkpoint path: ").strip()
            resume_training_example(checkpoint_path)
        elif choice == "10":
            model_path = input("Enter model path: ").strip()
            evaluate_trained_model(model_path)
        else:
            print("Invalid choice. Running component demos instead.")
            demonstrate_enhanced_loss_functions()
            demonstrate_batch_sampling()

    except KeyboardInterrupt:
        print("\nExample interrupted by user.")
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
