#!/usr/bin/env python3
"""
Example script demonstrating the Azul PettingZoo AEC Environment.

This script shows how to use the Azul game as a PettingZoo environment
for multi-agent reinforcement learning.
"""

import random
from typing import Callable, Dict, List

import numpy as np

from azul_rl.game import AzulAECEnv, env, raw_env, wrapped_env


def random_policy(observation: np.ndarray, legal_actions: List[int]) -> int:
    """
    Simple random policy that selects a random legal action.

    Args:
        observation: Current observation (unused in random policy)
        legal_actions: List of legal action integers

    Returns:
        Selected action integer
    """
    if not legal_actions:
        return 0  # Fallback (should not happen)
    return random.choice(legal_actions)


def greedy_policy(observation: np.ndarray, legal_actions: List[int]) -> int:
    """
    Simple greedy policy that always selects the first legal action.

    Args:
        observation: Current observation (unused in greedy policy)
        legal_actions: List of legal action integers

    Returns:
        Selected action integer
    """
    if not legal_actions:
        return 0  # Fallback (should not happen)
    return legal_actions[0]


def play_game_with_policies(
    policies: Dict[str, Callable],
    num_players: int = 2,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Play a complete game using the provided policies.

    Args:
        policies: Dictionary mapping agent names to policy functions
        num_players: Number of players
        seed: Random seed for reproducibility
        verbose: Whether to print game progress

    Returns:
        Dictionary with game results
    """
    # Create environment
    env_instance = AzulAECEnv(
        num_players=num_players, seed=seed, render_mode="human" if verbose else None
    )
    env_instance.reset()

    if verbose:
        print(f"=== Starting Azul Game with {num_players} players ===")
        print(f"Agents: {env_instance.agents}")
        print(f"Policies: {list(policies.keys())}")
        print()

    game_history = []
    step_count = 0
    max_steps = 1000  # Safety limit

    while not all(env_instance.terminations.values()) and step_count < max_steps:
        if env_instance.agent_selection is None:
            break

        current_agent = env_instance.agent_selection

        # Get observation and legal actions
        observation = env_instance.observe(current_agent)
        legal_actions = env_instance.get_legal_actions()

        if not legal_actions:
            if verbose:
                print(f"No legal actions for {current_agent}, ending game")
            break

        # Select action using policy
        policy = policies.get(current_agent, random_policy)
        action = policy(observation, legal_actions)

        if verbose and step_count % 10 == 0:  # Print every 10th step
            print(
                f"Step {step_count}: {current_agent} taking action {action} (from {len(legal_actions)} legal actions)"
            )

        # Record step
        game_history.append(
            {
                "step": step_count,
                "agent": current_agent,
                "action": action,
                "num_legal_actions": len(legal_actions),
                "observation_shape": observation.shape,
            }
        )

        # Take step
        env_instance.step(action)
        step_count += 1

        # Check for game end
        if all(env_instance.terminations.values()):
            break

    # Get final results
    final_scores = (
        env_instance.game_state.get_scores()
        if env_instance.game_state
        else [0] * num_players
    )
    final_rewards = {
        agent: env_instance._cumulative_rewards[agent] for agent in env_instance.agents
    }

    if verbose:
        print(f"\n=== Game Completed in {step_count} steps ===")
        print("Final Scores:")
        for i, score in enumerate(final_scores):
            agent = f"player_{i}"
            reward = final_rewards.get(agent, 0)
            print(f"  {agent}: {score} points (total reward: {reward:.1f})")

        if env_instance.game_state and env_instance.game_state.winner is not None:
            winner = f"player_{env_instance.game_state.winner}"
            print(f"\nWinner: {winner}")

        print("\nGame Statistics:")
        print(f"  Total steps: {step_count}")
        print(
            f"  Average actions per step: {np.mean([h['num_legal_actions'] for h in game_history]):.1f}"
        )
        print(
            f"  Observation shape: {game_history[0]['observation_shape'] if game_history else 'N/A'}"
        )

    return {
        "final_scores": final_scores,
        "final_rewards": final_rewards,
        "winner": env_instance.game_state.winner if env_instance.game_state else None,
        "steps": step_count,
        "history": game_history,
    }


def demonstrate_basic_usage():
    """Demonstrate basic usage of the Azul PettingZoo environment."""
    print("=== Basic Usage Demonstration ===\n")

    # Create environment using different factory functions
    print("1. Creating environments using factory functions:")

    # Raw environment
    raw_env_instance = raw_env(num_players=2, seed=42)
    print(f"   Raw environment: {type(raw_env_instance).__name__}")

    # Standard environment
    env_instance = env(num_players=3, seed=123)
    print(f"   Standard environment: {type(env_instance).__name__}")

    # Wrapped environment
    wrapped_env_instance = wrapped_env(num_players=4, seed=456)
    print(f"   Wrapped environment: {type(wrapped_env_instance).__name__}")

    print()

    # Demonstrate basic API
    print("2. Basic API demonstration:")
    env_instance = AzulAECEnv(num_players=2, seed=42)
    env_instance.reset()

    print(f"   Agents: {env_instance.agents}")
    print(f"   Current agent: {env_instance.agent_selection}")

    # Get spaces
    agent = env_instance.agents[0]
    action_space = env_instance.action_space(agent)
    observation_space = env_instance.observation_space(agent)

    print(f"   Action space: {action_space}")
    print(f"   Observation space shape: {observation_space.shape}")

    # Get observation and legal actions
    observation = env_instance.observe(agent)
    legal_actions = env_instance.get_legal_actions()

    print(f"   Observation shape: {observation.shape}")
    print(f"   Number of legal actions: {len(legal_actions)}")
    print(f"   Legal actions (first 5): {legal_actions[:5]}")

    print()


def demonstrate_game_play():
    """Demonstrate playing games with different policies."""
    print("=== Game Play Demonstration ===\n")

    # Game 1: Random vs Random
    print("Game 1: Random vs Random")
    policies_1 = {"player_0": random_policy, "player_1": random_policy}
    result_1 = play_game_with_policies(
        policies_1, num_players=2, seed=42, verbose=False
    )
    print(f"Winner: player_{result_1['winner']} with scores {result_1['final_scores']}")
    print()

    # Game 2: Greedy vs Random
    print("Game 2: Greedy vs Random")
    policies_2 = {"player_0": greedy_policy, "player_1": random_policy}
    result_2 = play_game_with_policies(
        policies_2, num_players=2, seed=42, verbose=False
    )
    print(f"Winner: player_{result_2['winner']} with scores {result_2['final_scores']}")
    print()

    # Game 3: 3-player game with mixed policies
    print("Game 3: 3-player mixed policies")
    policies_3 = {
        "player_0": greedy_policy,
        "player_1": random_policy,
        "player_2": greedy_policy,
    }
    result_3 = play_game_with_policies(
        policies_3, num_players=3, seed=123, verbose=False
    )
    print(f"Winner: player_{result_3['winner']} with scores {result_3['final_scores']}")
    print()


def demonstrate_detailed_game():
    """Demonstrate a detailed game with verbose output."""
    print("=== Detailed Game Demonstration ===\n")

    policies = {"player_0": random_policy, "player_1": greedy_policy}

    result = play_game_with_policies(policies, num_players=2, seed=42, verbose=True)

    print("\nDetailed Results:")
    print(f"  Game completed in {result['steps']} steps")
    print(f"  Final scores: {result['final_scores']}")
    print(f"  Final rewards: {result['final_rewards']}")


def demonstrate_environment_features():
    """Demonstrate various environment features."""
    print("=== Environment Features Demonstration ===\n")

    env_instance = AzulAECEnv(num_players=2, seed=42, render_mode="human")
    env_instance.reset()

    print("1. Rendering:")
    rendered = env_instance.render()
    print(rendered[:300] + "..." if len(rendered) > 300 else rendered)
    print()

    print("2. Action encoding/decoding:")
    legal_actions = env_instance.get_legal_actions()
    if legal_actions:
        action_int = legal_actions[0]
        action_obj = env_instance._decode_action(action_int)
        encoded_back = env_instance._encode_action(action_obj)

        print(f"   Action integer: {action_int}")
        print(f"   Decoded action: {action_obj}")
        print(f"   Encoded back: {encoded_back}")
        print(f"   Round-trip successful: {action_int == encoded_back}")
    print()

    print("3. Observation details:")
    agent = env_instance.agent_selection
    observation = env_instance.observe(agent)

    print(f"   Agent: {agent}")
    print(f"   Observation shape: {observation.shape}")
    print(f"   Observation range: [{observation.min():.3f}, {observation.max():.3f}]")
    print(f"   Observation mean: {observation.mean():.3f}")
    print()

    print("4. Last() method:")
    obs, reward, termination, truncation, info = env_instance.last()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Reward: {reward}")
    print(f"   Termination: {termination}")
    print(f"   Truncation: {truncation}")
    print(f"   Info: {info}")


def run_performance_test():
    """Run a simple performance test."""
    print("=== Performance Test ===\n")

    import time

    num_games = 10
    total_steps = 0
    start_time = time.time()

    for i in range(num_games):
        policies = {"player_0": random_policy, "player_1": random_policy}
        result = play_game_with_policies(
            policies, num_players=2, seed=42 + i, verbose=False
        )
        total_steps += result["steps"]

    end_time = time.time()
    total_time = end_time - start_time

    print("Performance Results:")
    print(f"  Games played: {num_games}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Time per game: {total_time / num_games:.3f} seconds")
    print(f"  Total steps: {total_steps}")
    print(f"  Steps per game: {total_steps / num_games:.1f}")
    print(f"  Steps per second: {total_steps / total_time:.1f}")


def main():
    """Main demonstration function."""
    print("🎲 Azul PettingZoo Environment Demonstration 🎲\n")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    try:
        # Run demonstrations
        demonstrate_basic_usage()
        demonstrate_game_play()
        demonstrate_detailed_game()
        demonstrate_environment_features()
        run_performance_test()

        print("\n✅ All demonstrations completed successfully!")
        print("\nThe Azul PettingZoo environment is ready for use in multi-agent RL!")
        print("\nNext steps:")
        print(
            "  1. Train RL agents using your favorite framework (e.g., Stable-Baselines3)"
        )
        print("  2. Experiment with different reward functions")
        print("  3. Try different observation representations")
        print("  4. Implement more sophisticated policies")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
