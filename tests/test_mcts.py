"""
Tests for the Monte Carlo Tree Search implementation.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from agents.mcts import (
    MCTS,
    GameState,
    MCTSAgent,
    MCTSNode,
    NeuralNetwork,
)
from game.game_state import Action
from game.game_state import GameState as AzulGameState
from training.neural_network import AzulNeuralNetwork


class TestMCTSNode:
    """Test the MCTSNode class."""

    def test_node_initialization(self):
        """Test node initialization."""
        azul_state = AzulGameState(num_players=2, seed=42)
        node = MCTSNode(azul_state)

        assert node.state == azul_state
        assert node.parent is None
        assert node.action is None
        assert node.N == 0
        assert node.Q == 0.0
        assert node.P == 0.0
        assert not node.is_expanded
        assert len(node.children) == 0

    def test_node_with_parent(self):
        """Test node with parent."""
        parent_state = AzulGameState(num_players=2, seed=42)
        child_state = parent_state.copy()
        action = parent_state.get_legal_actions()[0]
        child_state.apply_action(action)

        parent = MCTSNode(parent_state)
        child = MCTSNode(child_state, parent=parent, action=action, prior_prob=0.5)

        assert child.parent == parent
        assert child.action == action
        assert child.P == 0.5

    def test_is_leaf(self):
        """Test is_leaf method."""
        azul_state = AzulGameState(num_players=2, seed=42)
        node = MCTSNode(azul_state)

        assert node.is_leaf()

        # Add a child
        child_state = azul_state.copy()
        action = azul_state.get_legal_actions()[0]
        child_state.apply_action(action)
        node.add_child(action, child_state, 0.5)

        assert not node.is_leaf()

    def test_is_root(self):
        """Test is_root method."""
        azul_state = AzulGameState(num_players=2, seed=42)
        root = MCTSNode(azul_state)

        assert root.is_root()

        child_state = azul_state.copy()
        action = azul_state.get_legal_actions()[0]
        child_state.apply_action(action)
        child = MCTSNode(child_state, parent=root, action=action)

        assert not child.is_root()

    def test_get_value(self):
        """Test get_value method."""
        azul_state = AzulGameState(num_players=2, seed=42)
        node = MCTSNode(azul_state)

        # Initially should return 0
        assert node.get_value() == 0.0

        # After updates
        node.N = 10
        node.Q = 5.0
        assert node.get_value() == 0.5

    def test_add_child(self):
        """Test add_child method."""
        azul_state = AzulGameState(num_players=2, seed=42)
        parent = MCTSNode(azul_state)

        child_state = azul_state.copy()
        action = azul_state.get_legal_actions()[0]
        child_state.apply_action(action)
        child = parent.add_child(action, child_state, 0.3)

        assert action in parent.children
        assert parent.children[action] == child
        assert child.parent == parent
        assert child.action == action
        assert child.P == 0.3


class TestAzulGameState:
    """Test the Azul GameState class with MCTS protocol."""

    def test_initialization(self):
        """Test Azul game state initialization."""
        azul_state = AzulGameState(num_players=2, seed=42)

        assert azul_state.current_player == 0
        assert not azul_state.game_over
        assert azul_state.num_players == 2

    def test_legal_actions(self):
        """Test get_legal_actions method."""
        azul_state = AzulGameState(num_players=2, seed=42)

        legal_actions = azul_state.get_legal_actions()

        # Should have some legal actions at the start
        assert len(legal_actions) > 0
        assert all(isinstance(action, Action) for action in legal_actions)

    def test_apply_action(self):
        """Test apply_action method."""
        azul_state = AzulGameState(num_players=2, seed=42)

        legal_actions = azul_state.get_legal_actions()
        assert len(legal_actions) > 0

        # Apply first legal action
        action = legal_actions[0]
        new_state = azul_state.apply_action(action)

        # Should return True for successful application
        assert new_state is True

    def test_numerical_state(self):
        """Test numerical state representation."""
        azul_state = AzulGameState(num_players=2, seed=42)

        numerical_state = azul_state.get_numerical_state()
        assert numerical_state is not None

        # Test that we can get a flat vector
        state_vector = numerical_state.get_flat_state_vector(normalize=True)
        assert isinstance(state_vector, np.ndarray)
        assert len(state_vector) > 0


class TestAzulNeuralNetwork:
    """Test the AzulNeuralNetwork class."""

    def test_initialization(self):
        """Test neural network initialization."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")

        assert nn.device.type == "cpu"
        assert nn.model is not None

    def test_evaluate(self):
        """Test evaluate method."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        azul_state = AzulGameState(num_players=2, seed=42)

        action_probs, value = nn.evaluate(azul_state)

        legal_actions = azul_state.get_legal_actions()

        assert len(action_probs) == len(legal_actions)
        assert np.isclose(action_probs.sum(), 1.0)
        assert -1 <= value <= 1
        assert all(prob >= 0 for prob in action_probs)

    def test_evaluate_consistency(self):
        """Test that evaluation is consistent for same state."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        azul_state = AzulGameState(num_players=2, seed=42)

        # Set to eval mode for consistency
        nn.model.eval()

        probs1, value1 = nn.evaluate(azul_state)
        probs2, value2 = nn.evaluate(azul_state)

        assert np.allclose(probs1, probs2)
        assert np.isclose(value1, value2)

    def test_evaluate_uses_numerical_state(self):
        """Test that evaluation uses numerical state representation."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        azul_state = AzulGameState(num_players=2, seed=42)

        # This should not raise an exception
        action_probs, value = nn.evaluate(azul_state)

        # Should return proper types
        assert isinstance(action_probs, np.ndarray)
        assert isinstance(value, float)


class MockGameState(GameState):
    """Mock implementation of GameState for testing."""

    def __init__(
        self,
        game_over: bool = False,
        current_player: int = 0,
        winner: Optional[int] = None,
        legal_actions: List[Any] = None,
    ):
        self._game_over = game_over
        self._current_player = current_player
        self._winner = winner
        self._legal_actions = legal_actions or []

    def get_legal_actions(self) -> List[Any]:
        return self._legal_actions

    def apply_action(self, action: Any) -> "MockGameState":
        return MockGameState(
            game_over=self._game_over,
            current_player=self._current_player,
            winner=self._winner,
            legal_actions=self._legal_actions,
        )

    @property
    def game_over(self) -> bool:
        return self._game_over

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def winner(self) -> Optional[int]:
        return self._winner

    def get_numerical_state(self) -> Any:
        return np.zeros(1)  # Mock numerical state

    def copy(self) -> "MockGameState":
        return MockGameState(
            game_over=self._game_over,
            current_player=self._current_player,
            winner=self._winner,
            legal_actions=self._legal_actions.copy(),
        )


class TestMCTS:
    """Test the MCTS class."""

    def test_initialization(self):
        """Test MCTS initialization."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        mcts = MCTS(nn, num_simulations=10)

        assert mcts.neural_network == nn
        assert mcts.num_simulations == 10

    def test_terminal_value_calculation(self):
        """Test terminal value calculation for different game outcomes."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        mcts = MCTS(nn)

        # Test win scenario
        win_state = MockGameState(game_over=True, current_player=0, winner=0)
        assert mcts._get_terminal_value(win_state) == 1.0

        # Test loss scenario
        loss_state = MockGameState(game_over=True, current_player=0, winner=1)
        assert mcts._get_terminal_value(loss_state) == -1.0

        # Test draw scenario
        draw_state = MockGameState(game_over=True, current_player=0, winner=None)
        assert mcts._get_terminal_value(draw_state) == 0.0

    def test_terminal_value_perspective(self):
        """Test terminal value calculation from different player perspectives."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        mcts = MCTS(nn)

        # Create a state where player 0 wins
        state = MockGameState(game_over=True, winner=0)

        # Test from player 0's perspective
        state._current_player = 0
        assert mcts._get_terminal_value(state) == 1.0

        # Test from player 1's perspective
        state._current_player = 1
        assert mcts._get_terminal_value(state) == -1.0

    def test_terminal_value_edge_cases(self):
        """Test terminal value calculation for edge cases."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        mcts = MCTS(nn)

        # Test non-terminal state
        non_terminal = MockGameState(game_over=False, current_player=0, winner=None)
        assert mcts._get_terminal_value(non_terminal) == 0.0

        # Test terminal state with no winner but game over
        no_winner = MockGameState(game_over=True, current_player=0, winner=None)
        assert mcts._get_terminal_value(no_winner) == 0.0

        # Test terminal state with winner but different current player
        winner_diff_player = MockGameState(game_over=True, current_player=1, winner=0)
        assert mcts._get_terminal_value(winner_diff_player) == -1.0

    def test_terminal_value_consistency(self):
        """Test consistency of terminal value calculation across multiple calls."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        mcts = MCTS(nn)

        # Create a state where player 0 wins
        state = MockGameState(game_over=True, current_player=0, winner=0)

        # Should get same value multiple times
        value1 = mcts._get_terminal_value(state)
        value2 = mcts._get_terminal_value(state)
        value3 = mcts._get_terminal_value(state)
        assert value1 == value2 == value3 == 1.0

    def test_terminal_value_multiplayer(self):
        """Test terminal value calculation in multiplayer scenarios."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        mcts = MCTS(nn)

        # Create a 3-player game state
        state = MockGameState(game_over=True, winner=1)

        # Test from each player's perspective
        state._current_player = 0
        assert mcts._get_terminal_value(state) == -1.0

        state._current_player = 1
        assert mcts._get_terminal_value(state) == 1.0

        state._current_player = 2
        assert mcts._get_terminal_value(state) == -1.0

    def test_search_with_azul_game(self):
        """Test MCTS search with Azul game."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        mcts = MCTS(nn, num_simulations=5)
        azul_state = AzulGameState(num_players=2, seed=42)

        action_probs, root_node = mcts.search(azul_state)

        legal_actions = azul_state.get_legal_actions()

        assert len(action_probs) == len(legal_actions)
        assert np.isclose(sum(action_probs), 1.0)
        assert root_node.N > 0  # Should have been visited

    def test_search_visits_nodes(self):
        """Test that MCTS search visits nodes."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        mcts = MCTS(nn, num_simulations=10)
        azul_state = AzulGameState(num_players=2, seed=42)

        action_probs, root_node = mcts.search(azul_state)

        # Root should have been visited multiple times
        assert root_node.N >= 10

    def test_temperature_effect(self):
        """Test temperature effect on action selection."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        azul_state = AzulGameState(num_players=2, seed=42)

        # High temperature (more random)
        mcts_high = MCTS(nn, num_simulations=20, temperature=2.0)
        probs_high, _ = mcts_high.search(azul_state)

        # Low temperature (more deterministic)
        mcts_low = MCTS(nn, num_simulations=20, temperature=0.1)
        probs_low, _ = mcts_low.search(azul_state)

        # Low temperature should be more concentrated
        entropy_high = -sum(p * np.log(p + 1e-8) for p in probs_high if p > 0)
        entropy_low = -sum(p * np.log(p + 1e-8) for p in probs_low if p > 0)

        assert entropy_low <= entropy_high


class TestMCTSAgent:
    """Test the MCTSAgent class."""

    def test_initialization(self):
        """Test agent initialization."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        agent = MCTSAgent(nn, num_simulations=10)

        assert agent.mcts.neural_network == nn

    def test_select_action_with_azul_game(self):
        """Test action selection with Azul game."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        agent = MCTSAgent(nn, num_simulations=5)
        azul_state = AzulGameState(num_players=2, seed=42)

        action = agent.select_action(azul_state)

        legal_actions = azul_state.get_legal_actions()
        assert action in legal_actions

    def test_deterministic_selection(self):
        """Test deterministic action selection."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        agent = MCTSAgent(nn, num_simulations=10)
        azul_state = AzulGameState(num_players=2, seed=42)

        action1 = agent.select_action(azul_state, deterministic=True)
        action2 = agent.select_action(azul_state, deterministic=True)

        # Should select the same action deterministically
        assert action1 == action2

    def test_get_action_probabilities(self):
        """Test getting action probabilities."""
        nn = AzulNeuralNetwork(config_name="small", device="cpu")
        agent = MCTSAgent(nn, num_simulations=5)
        azul_state = AzulGameState(num_players=2, seed=42)

        probs = agent.get_action_probabilities(azul_state)

        legal_actions = azul_state.get_legal_actions()
        assert len(probs) == len(legal_actions)
        assert np.isclose(sum(probs), 1.0)


def test_azul_game_simulation():
    """Test MCTS with a real Azul game simulation."""
    nn = AzulNeuralNetwork(config_name="small", device="cpu")
    agent = MCTSAgent(nn, num_simulations=5)  # Small number for testing

    azul_state = AzulGameState(num_players=2, seed=42)

    moves = 0
    while not azul_state.game_over and moves < 10:  # Limit moves for testing
        legal_actions = azul_state.get_legal_actions()

        if not legal_actions:
            break

        # Select action using MCTS
        action = agent.select_action(azul_state)
        assert action in legal_actions

        # Apply action
        success = azul_state.apply_action(action)
        assert success

        moves += 1

    # Should have made some moves
    assert moves > 0


def test_full_azul_game_with_mcts():
    """Test a complete game with MCTS agents."""
    nn = AzulNeuralNetwork(config_name="small", device="cpu")
    agent = MCTSAgent(nn, num_simulations=3)  # Very small for testing

    azul_state = AzulGameState(num_players=2, seed=123)

    moves = 0
    max_moves = 50  # Prevent infinite loops

    while not azul_state.game_over and moves < max_moves:
        legal_actions = azul_state.get_legal_actions()

        if not legal_actions:
            break

        # Get action probabilities
        probs = agent.get_action_probabilities(azul_state)
        assert len(probs) == len(legal_actions)
        assert np.isclose(sum(probs), 1.0)

        # Select and apply action
        action = agent.select_action(azul_state)
        success = azul_state.apply_action(action)
        assert success

        moves += 1

    # Should have made reasonable number of moves
    assert moves > 5


if __name__ == "__main__":
    pytest.main([__file__])
