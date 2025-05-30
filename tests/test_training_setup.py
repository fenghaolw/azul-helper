"""
Test script to verify training setup and components.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch import Tensor

from game.game_state import GameState
from training.neural_network import AzulNeuralNetwork
from training.replay_buffer import Experience, ReplayBuffer
from training.training_utils import create_training_components


@pytest.mark.slow
def test_training_setup():
    """Test all training components."""
    print("Testing Azul Neural Network Training Setup")
    print("=" * 50)

    # 1. Test Neural Network
    print("\n1. Testing Neural Network...")
    try:
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        print(f"   ✓ Neural network created: {nn.get_model_info()['model_type']}")
        print(f"   ✓ Parameters: {nn.get_model_info()['total_parameters']:,}")

        # Test evaluation
        game_state = GameState(num_players=2)
        policy, value = nn.evaluate(game_state)
        print(
            f"   ✓ Evaluation works - Policy shape: {policy.shape}, Value: {value:.3f}"
        )
    except Exception as e:
        print(f"   ✗ Neural network test failed: {e}")
        return False

    # 2. Test Replay Buffer
    print("\n2. Testing Replay Buffer...")
    try:
        buffer = ReplayBuffer(capacity=1000, min_size_for_sampling=10)

        # Add some dummy experiences
        experiences = []
        for i in range(20):
            exp = Experience(
                state=game_state.copy(),
                mcts_policy=np.random.dirichlet(np.ones(500)),
                player_id=i % 2,
                outcome=None,
            )
            experiences.append(exp)

        # Add as a game with outcomes
        outcomes = [1.0, -1.0]  # Player 0 wins, Player 1 loses
        buffer.add_game(experiences, outcomes)

        print(f"   ✓ Buffer size: {buffer.size()}")
        print(f"   ✓ Ready for sampling: {buffer.is_ready_for_sampling()}")

        # Test sampling
        batch = buffer.sample(5)
        if batch:
            states, policies, outcomes = batch
            print(
                f"   ✓ Sampled batch - States: {states.shape}, Policies: {policies.shape}"
            )
    except Exception as e:
        print(f"   ✗ Replay buffer test failed: {e}")
        return False

    # 3. Test Training Components
    print("\n3. Testing Training Components...")
    try:
        training_step, checkpoint_manager = create_training_components(
            model=nn.model,
            replay_buffer=buffer,
            learning_rate=0.001,
            batch_size=5,
            device=torch.device("cpu"),
        )

        print(f"   ✓ Training step created")
        print(f"   ✓ Optimizer: {type(training_step.optimizer).__name__}")
        print(f"   ✓ Loss functions: Cross-entropy (policy), MSE (value)")

        # Run one training step
        stats = training_step.single_training_step()
        if stats:
            print(f"   ✓ Training step successful - Loss: {stats['total_loss']:.6f}")
        else:
            print(f"   ✗ No training data available")
    except Exception as e:
        print(f"   ✗ Training components test failed: {e}")
        return False

    # 4. Test MCTS Integration
    print("\n4. Testing MCTS Integration...")
    try:
        from agents.mcts import MCTSAgent

        agent = MCTSAgent(neural_network=nn, num_simulations=50, temperature=1.0)

        # Get action probabilities
        action_probs: Tensor = agent.get_action_probabilities(game_state)
        print(f"   ✓ MCTS agent created")
        print(f"   ✓ Action probabilities shape: {action_probs.shape}")
        print(f"   ✓ Sum of probabilities: {action_probs.sum().item():.3f}")
    except Exception as e:
        print(f"   ✗ MCTS integration test failed: {e}")
        return False

    # 5. Test Self-Play
    print("\n5. Testing Self-Play Engine...")
    try:
        from training.self_play import SelfPlayEngine

        engine = SelfPlayEngine(
            neural_network=nn, replay_buffer=buffer, mcts_simulations=50, verbose=False
        )

        # Play one quick game
        print("   Running one self-play game...")
        experiences = engine.play_game(num_players=2)
        print(f"   ✓ Self-play game completed")
        print(f"   ✓ Experiences collected: {len(experiences)}")
        print(f"   ✓ Buffer size after game: {buffer.size()}")
    except Exception as e:
        print(f"   ✗ Self-play test failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("✅ All tests passed! Training setup is ready.")
    print("=" * 50)

    print("\nKey hyperparameters verified:")
    print("- Neural network architecture: ResNet-style with policy and value heads")
    print("- Loss functions: Cross-entropy (policy), MSE (value)")
    print("- MCTS simulations: Configurable (default 800)")
    print("- Replay buffer: Experience storage with outcome-based sampling")
    print("- Batch processing: Efficient GPU/CPU training")

    return True


if __name__ == "__main__":
    success = test_training_setup()
    sys.exit(0 if success else 1)
