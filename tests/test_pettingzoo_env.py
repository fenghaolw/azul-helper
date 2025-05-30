"""
Tests for the PettingZoo AEC Environment implementation.
"""

import random

import numpy as np
import pytest
from pettingzoo.test import api_test, parallel_api_test

from game import Action, AzulAECEnv, TileColor, env, raw_env, wrapped_env


class TestAzulAECEnv:
    """Test the Azul AEC Environment implementation."""

    def test_environment_creation(self):
        """Test basic environment creation."""
        # Test default creation
        env_instance = AzulAECEnv()
        assert env_instance.num_players == 2
        assert len(env_instance.possible_agents) == 2
        assert env_instance.possible_agents == ["player_0", "player_1"]

        # Test with different number of players
        env_3p = AzulAECEnv(num_players=3)
        assert env_3p.num_players == 3
        assert len(env_3p.possible_agents) == 3

        env_4p = AzulAECEnv(num_players=4)
        assert env_4p.num_players == 4
        assert len(env_4p.possible_agents) == 4

        # Test invalid number of players
        with pytest.raises(ValueError):
            AzulAECEnv(num_players=1)

        with pytest.raises(ValueError):
            AzulAECEnv(num_players=5)

    def test_reset_functionality(self):
        """Test environment reset functionality."""
        env_instance = AzulAECEnv(num_players=2, seed=42)

        # Reset environment
        env_instance.reset()

        # Check initial state
        assert env_instance.game_state is not None
        assert env_instance.state_representation is not None
        assert env_instance.agent_selection == "player_0"
        assert len(env_instance.agents) == 2

        # Check that all agents are not terminated
        for agent in env_instance.agents:
            assert not env_instance.terminations[agent]
            assert not env_instance.truncations[agent]
            assert env_instance.rewards[agent] == 0.0

    def test_action_encoding_decoding(self):
        """Test action encoding and decoding."""
        env_instance = AzulAECEnv()

        # Test various actions
        test_actions = [
            Action(source=-1, color=TileColor.BLUE, destination=-1),
            Action(source=0, color=TileColor.RED, destination=2),
            Action(source=3, color=TileColor.WHITE, destination=4),
            Action(source=8, color=TileColor.BLACK, destination=0),
        ]

        for action in test_actions:
            # Encode and decode
            encoded = env_instance._encode_action(action)
            decoded = env_instance._decode_action(encoded)

            # Check that decoding gives back the original action
            assert decoded.source == action.source
            assert decoded.color == action.color
            assert decoded.destination == action.destination

    def test_observation_spaces(self):
        """Test observation space setup."""
        env_instance = AzulAECEnv(num_players=3)

        # Check that observation spaces are set up correctly
        for agent in env_instance.possible_agents:
            obs_space = env_instance.observation_space(agent)
            assert obs_space.shape[0] > 0  # Should have some dimensions
            assert obs_space.dtype == np.float32
            assert obs_space.low.min() == 0.0
            assert obs_space.high.max() == 1.0

    def test_action_spaces(self):
        """Test action space setup."""
        env_instance = AzulAECEnv(num_players=2)

        # Check that action spaces are set up correctly
        for agent in env_instance.possible_agents:
            action_space = env_instance.action_space(agent)
            assert action_space.n == 500  # Should have 500 possible actions

    def test_observe_functionality(self):
        """Test observation functionality."""
        env_instance = AzulAECEnv(num_players=2, seed=42)
        env_instance.reset()

        # Test observations for all agents
        for agent in env_instance.agents:
            obs = env_instance.observe(agent)
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert obs.shape == env_instance.observation_space(agent).shape
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

    def test_last_functionality(self):
        """Test last() method functionality."""
        env_instance = AzulAECEnv(num_players=2, seed=42)
        env_instance.reset()

        # Get last observation
        obs, reward, termination, truncation, info = env_instance.last()

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(termination, bool)
        assert isinstance(truncation, bool)
        assert isinstance(info, dict)

    def test_legal_actions(self):
        """Test legal action generation."""
        env_instance = AzulAECEnv(num_players=2, seed=42)
        env_instance.reset()

        # Get legal actions for current agent
        legal_actions = env_instance.get_legal_actions()
        assert isinstance(legal_actions, list)
        assert len(legal_actions) > 0  # Should have some legal actions initially

        # All legal actions should be valid integers
        for action in legal_actions:
            assert isinstance(action, int)
            assert 0 <= action < 500

    def test_step_functionality(self):
        """Test step functionality."""
        env_instance = AzulAECEnv(num_players=2, seed=42)
        env_instance.reset()

        initial_agent = env_instance.agent_selection

        # Get a legal action and take a step
        legal_actions = env_instance.get_legal_actions()
        assert len(legal_actions) > 0

        action = legal_actions[0]
        env_instance.step(action)

        # Agent should have changed (unless game ended)
        if not env_instance.game_state.game_over:
            assert env_instance.agent_selection != initial_agent

    def test_invalid_action_handling(self):
        """Test handling of invalid actions."""
        env_instance = AzulAECEnv(num_players=2, seed=42)
        env_instance.reset()

        initial_agent = env_instance.agent_selection

        # Try an invalid action (out of range)
        env_instance.step(999)  # Invalid action

        # Should get negative reward
        assert env_instance.rewards[initial_agent] == -1.0

    def test_game_completion(self):
        """Test that games can complete successfully."""
        env_instance = AzulAECEnv(num_players=2, seed=42)
        env_instance.reset()

        max_steps = 1000  # Safety limit
        steps = 0

        while not all(env_instance.terminations.values()) and steps < max_steps:
            if env_instance.agent_selection is None:
                break

            legal_actions = env_instance.get_legal_actions()
            if legal_actions:
                # Use random action instead of always first action
                action = random.choice(legal_actions)
                env_instance.step(action)
            else:
                # No legal actions available, game should end
                break
            steps += 1

            # Additional check: if game state says game is over, break
            if env_instance.game_state and env_instance.game_state.game_over:
                break

        # Game should eventually end
        if steps >= max_steps:
            print(
                f"Game state: game_over={env_instance.game_state.game_over if env_instance.game_state else 'None'}"
            )
            print(f"Agent selection: {env_instance.agent_selection}")
            print(f"Terminations: {env_instance.terminations}")
            print(f"Steps taken: {steps}")

        assert steps < max_steps, f"Game took too long to complete ({steps} steps)"

        # If game ended, all agents should be terminated
        if env_instance.game_state and env_instance.game_state.game_over:
            for agent in env_instance.agents:
                assert env_instance.terminations[agent]

    def test_render_functionality(self):
        """Test rendering functionality."""
        env_instance = AzulAECEnv(num_players=2, seed=42, render_mode="human")
        env_instance.reset()

        # Test human rendering
        rendered = env_instance.render()
        assert isinstance(rendered, str)
        assert "Azul Game" in rendered

        # Test rgb_array rendering
        env_instance.render_mode = "rgb_array"
        rendered_array = env_instance.render()
        assert isinstance(rendered_array, np.ndarray)
        assert rendered_array.shape == (400, 600, 3)

    def test_deterministic_behavior(self):
        """Test that environment behavior is deterministic with same seed."""
        # Create two environments with same seed
        env1 = AzulAECEnv(num_players=2, seed=42)
        env2 = AzulAECEnv(num_players=2, seed=42)

        env1.reset()
        env2.reset()

        # Take same actions in both environments
        for _ in range(10):
            if env1.agent_selection is None or env2.agent_selection is None:
                break

            legal_actions1 = env1.get_legal_actions()
            legal_actions2 = env2.get_legal_actions()

            # Should have same legal actions
            assert legal_actions1 == legal_actions2

            if legal_actions1:
                action = legal_actions1[0]
                env1.step(action)
                env2.step(action)

                # Should have same observations
                obs1 = (
                    env1.observe(env1.agent_selection) if env1.agent_selection else None
                )
                obs2 = (
                    env2.observe(env2.agent_selection) if env2.agent_selection else None
                )

                if obs1 is not None and obs2 is not None:
                    np.testing.assert_array_equal(obs1, obs2)

    def test_factory_functions(self):
        """Test the factory functions for creating environments."""
        # Test env() function
        env_instance = env(num_players=3, seed=42)
        assert isinstance(env_instance, AzulAECEnv)
        assert env_instance.num_players == 3

        # Test raw_env() function
        raw_env_instance = raw_env(num_players=2, seed=123)
        assert isinstance(raw_env_instance, AzulAECEnv)
        assert raw_env_instance.num_players == 2

        # Test wrapped_env() function
        wrapped_env_instance = wrapped_env(num_players=4, seed=456)
        # Should be wrapped, so not directly AzulAECEnv
        assert hasattr(wrapped_env_instance, "env")  # Should have wrapped environment

    def test_pettingzoo_api_compliance(self):
        """Test compliance with PettingZoo API."""
        # Test the raw environment
        env_instance = raw_env(num_players=2, seed=42)

        # Run the full API test
        api_test(env_instance, num_cycles=1000, verbose_progress=True)

    def test_multiple_games(self):
        """Test running multiple games in sequence."""
        env_instance = AzulAECEnv(num_players=2, seed=42)

        for game_num in range(3):
            env_instance.reset(seed=42 + game_num)

            # Play a few moves
            for _ in range(20):
                if env_instance.agent_selection is None:
                    break

                legal_actions = env_instance.get_legal_actions()
                if legal_actions:
                    action = legal_actions[0]
                    env_instance.step(action)
                else:
                    break

            # Environment should be in valid state
            assert env_instance.game_state is not None
            assert env_instance.state_representation is not None


class TestAzulAECEnvIntegration:
    """Integration tests for the Azul AEC Environment."""

    def test_full_game_simulation(self):
        """Test a complete game simulation."""
        env_instance = AzulAECEnv(num_players=2, seed=42)
        env_instance.reset()

        game_history = []
        max_steps = 1000

        for step in range(max_steps):
            if env_instance.agent_selection is None:
                break

            # Record game state
            current_agent = env_instance.agent_selection
            obs = env_instance.observe(current_agent)
            legal_actions = env_instance.get_legal_actions()

            game_history.append(
                {
                    "step": step,
                    "agent": current_agent,
                    "observation_shape": obs.shape,
                    "num_legal_actions": len(legal_actions),
                    "game_over": (
                        env_instance.game_state.game_over
                        if env_instance.game_state
                        else True
                    ),
                }
            )

            if not legal_actions:
                break

            # Take action
            action = legal_actions[0]
            env_instance.step(action)

            # Check if game ended
            if all(env_instance.terminations.values()):
                break

        # Verify game completed properly
        assert len(game_history) > 0
        assert step < max_steps, "Game took too long"

        # Check that we had reasonable number of moves
        assert len(game_history) > 10, "Game ended too quickly"

    def test_reward_calculation(self):
        """Test that rewards are calculated correctly."""
        env_instance = AzulAECEnv(num_players=2, seed=42)
        env_instance.reset()

        total_rewards = {agent: 0.0 for agent in env_instance.agents}

        # Play game and track rewards
        for _ in range(200):  # Increase steps to ensure game completion
            if env_instance.agent_selection is None:
                break

            current_agent = env_instance.agent_selection
            legal_actions = env_instance.get_legal_actions()

            if not legal_actions:
                break

            # Take action
            action = random.choice(
                legal_actions
            )  # Use random action for better gameplay
            env_instance.step(action)

            # Track total rewards
            total_rewards[current_agent] += env_instance.rewards[current_agent]

            if all(env_instance.terminations.values()):
                break

        # At least one player should have gained some points
        print(f"Total rewards: {total_rewards}")
        assert any(
            reward != 0 for reward in total_rewards.values()
        ), f"No rewards earned: {total_rewards}"


if __name__ == "__main__":
    # Run basic tests
    print("=== Testing Azul PettingZoo Environment ===")

    # Test basic functionality
    test_env = TestAzulAECEnv()
    test_env.test_environment_creation()
    print("✓ Environment creation test passed")

    test_env.test_reset_functionality()
    print("✓ Reset functionality test passed")

    test_env.test_action_encoding_decoding()
    print("✓ Action encoding/decoding test passed")

    test_env.test_step_functionality()
    print("✓ Step functionality test passed")

    # Test integration
    integration_test = TestAzulAECEnvIntegration()
    integration_test.test_full_game_simulation()
    print("✓ Full game simulation test passed")

    print("\nAll tests completed successfully!")
    print("The Azul environment is ready for use with PettingZoo!")
