"""
Monte Carlo Tree Search (MCTS) implementation for game playing.

This module implements the MCTS algorithm with PyTorch neural network guidance,
following the AlphaZero approach with UCT-based selection.

The implementation works directly with Azul GameState objects, leveraging
the numerical state representation for neural network evaluation and proper
action encoding from the PettingZoo environment.

Key Features:
- AlphaZero-style MCTS with neural network priors
- Direct Azul GameState compatibility
- Proper action encoding using PettingZoo environment
- PyTorch neural network integration
- Multiple network configurations (small, medium, large, deep)
- GPU support with automatic device selection
- Model persistence and loading capabilities
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

# PyTorch imports for neural network support
import torch
import torch.nn as nn


class GameState(Protocol):
    """Protocol defining the interface for game states (aligned with Azul GameState)."""

    def get_legal_actions(self) -> List[Any]:
        """Return list of legal actions (can be Action objects or integers)."""
        ...

    def apply_action(self, action: Any) -> "GameState":
        """Return new state after applying action."""
        ...

    @property
    def game_over(self) -> bool:
        """Return True if this is a terminal state."""
        ...

    @property
    def current_player(self) -> int:
        """Return the current player index."""
        ...

    def get_numerical_state(self) -> Any:
        """Get numerical representation of the state for neural network evaluation."""
        ...

    def copy(self) -> "GameState":
        """Create a deep copy of the game state."""
        ...


class NeuralNetwork(Protocol):
    """Protocol defining the interface for neural network evaluation."""

    def evaluate(self, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Evaluate a game state.

        Args:
            state: Game state to evaluate

        Returns:
            Tuple of (action_probabilities, state_value)
            - action_probabilities: numpy array of probabilities for each legal action
            - state_value: estimated value of the state for current player
        """
        ...


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.

    Attributes:
        state: The game state this node represents
        parent: Parent node (None for root)
        action: Action taken from parent to reach this node
        children: Dictionary mapping actions to child nodes
        N: Visit count
        Q: Total action value (sum of all backpropagated values)
        P: Prior probability from neural network
        is_expanded: Whether this node has been expanded
    """

    def __init__(
        self,
        state: GameState,
        parent: Optional["MCTSNode"] = None,
        action: Optional[Any] = None,
        prior_prob: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[Any, "MCTSNode"] = {}

        # MCTS statistics
        self.N = 0  # Visit count
        self.Q = 0.0  # Total action value
        self.P = prior_prob  # Prior probability from NN
        self.is_expanded = False

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if this node is the root (no parent)."""
        return self.parent is None

    def get_value(self) -> float:
        """Get average value (Q/N) for this node."""
        if self.N == 0:
            return 0.0
        return self.Q / self.N

    def add_child(
        self, action: Any, child_state: GameState, prior_prob: float
    ) -> "MCTSNode":
        """Add a child node for the given action."""
        child = MCTSNode(child_state, parent=self, action=action, prior_prob=prior_prob)
        self.children[action] = child
        return child

    def __repr__(self) -> str:
        return f"MCTSNode(N={self.N}, Q={self.Q:.3f}, P={self.P:.3f}, children={len(self.children)})"


class MCTS:
    """
    Monte Carlo Tree Search implementation with neural network guidance.

    This implementation follows the AlphaZero approach:
    - Selection uses UCT with neural network priors
    - Expansion creates all legal child nodes
    - Simulation uses neural network evaluation
    - Backpropagation updates statistics up the tree

    The implementation works directly with Azul GameState objects and leverages
    the numerical state representation for neural network evaluation.
    """

    def __init__(
        self,
        neural_network: NeuralNetwork,
        c_puct: float = 1.0,
        num_simulations: int = 800,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        """
        Initialize MCTS.

        Args:
            neural_network: Neural network for state evaluation
            c_puct: Exploration constant for UCT formula
            num_simulations: Number of MCTS simulations to run
            temperature: Temperature for action selection
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Mixing parameter for Dirichlet noise
        """
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.nodes_evaluated = 0
        self.max_depth_reached = 0

    def search(self, root_state: GameState) -> Tuple[np.ndarray, MCTSNode]:
        """
        Run MCTS search from the given root state.

        Args:
            root_state: Initial game state to search from

        Returns:
            Tuple of (action_probabilities, root_node)
            - action_probabilities: Improved action probabilities for each legal action
            - root_node: Root node of the search tree (for analysis)
        """
        # Create root node
        root = MCTSNode(root_state)

        # Add Dirichlet noise to root node for exploration (skip if deterministic)
        if self.temperature > 0:
            self._add_dirichlet_noise(root)
        else:
            # For deterministic mode, just expand without noise
            if not root.is_expanded:
                self._expand_and_evaluate(root)

        # Track stats for this search
        max_depth_this_search = 0

        # Run simulations
        for _ in range(self.num_simulations):
            depth = self._simulate_with_depth(root)
            if depth > max_depth_this_search:
                max_depth_this_search = depth

        # Update stats
        self.nodes_evaluated += self.num_simulations
        if max_depth_this_search > self.max_depth_reached:
            self.max_depth_reached = max_depth_this_search

        # Calculate action probabilities based on visit counts
        action_probs = self._get_action_probabilities(root)

        return action_probs, root

    def _simulate_with_depth(self, root: MCTSNode) -> int:
        """
        Run a single MCTS simulation and return the depth reached.
        """
        path = []
        current = root
        depth = 0
        while not current.is_leaf() and not current.state.game_over:
            action = self._select_action(current)
            path.append((current, action))
            current = current.children[action]
            depth += 1
        # Expansion and Simulation
        if not current.state.game_over:
            value = self._expand_and_evaluate(current)
        else:
            value = self._get_terminal_value(current.state)
        self._backpropagate(path, current, value)
        return depth

    def _select_action(self, node: MCTSNode) -> Any:
        """
        Select the best action using UCT formula.

        UCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            node: Node to select action from

        Returns:
            Selected action
        """
        if not node.is_expanded:
            # If node not expanded, expand it first
            self._expand_and_evaluate(node)

        best_action = None
        best_value = float("-inf")

        for action, child in node.children.items():
            # UCT formula
            if child.N == 0:
                # Unvisited nodes get infinite priority
                uct_value = float("inf")
            else:
                q_value = child.Q / child.N
                exploration = self.c_puct * child.P * math.sqrt(node.N) / (1 + child.N)
                uct_value = q_value + exploration

            if uct_value > best_value:
                best_value = uct_value
                best_action = action

        return best_action

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand a node by creating children for all legal actions and evaluate with NN.

        Args:
            node: Node to expand

        Returns:
            Value from neural network evaluation
        """
        # Get legal actions
        legal_actions = node.state.get_legal_actions()

        # Get neural network evaluation
        action_probs, value = self.neural_network.evaluate(node.state)

        # Create child nodes for all legal actions
        for i, action in enumerate(legal_actions):
            # Apply action to get child state
            # Handle different apply_action patterns:
            # - Some states return new state (like SimpleGameState)
            # - Some states modify in-place and return boolean (like AzulGameState)

            # First, make a copy to avoid modifying the original state
            child_state = node.state.copy()

            try:
                result = child_state.apply_action(action)

                # If apply_action returns a boolean, the state was modified in-place
                if isinstance(result, bool):
                    if not result:
                        continue  # Skip if action failed
                    # child_state is already the modified state
                else:
                    # If apply_action returns a new state, use that
                    child_state = result

            except Exception:
                # If apply_action fails, skip this action
                continue

            # Get prior probability for this action
            prior_prob = action_probs[i] if i < len(action_probs) else 0.0

            # Add child node
            node.add_child(action, child_state, prior_prob)

        node.is_expanded = True
        return value

    def _backpropagate(
        self, path: List[Tuple[MCTSNode, Any]], leaf: MCTSNode, value: float
    ) -> None:
        """
        Backpropagate value up the tree, updating N and Q values.

        Args:
            path: List of (node, action) pairs from root to leaf
            leaf: Leaf node where simulation ended
            value: Value to backpropagate
        """
        # Update leaf node
        leaf.N += 1
        leaf.Q += value

        # Update path with alternating signs (for two-player games)
        current_value = -value  # Flip value for parent's perspective
        for node, action in reversed(path):
            node.N += 1
            node.Q += current_value
            current_value = -current_value  # Flip for next level up

    def _get_action_probabilities(self, root: MCTSNode) -> np.ndarray:
        """
        Get action probabilities based on visit counts.

        Note: Temperature effects are only visible when there are meaningful differences
        in visit counts between actions. If all actions have similar visit counts
        (e.g., when num_simulations is low relative to the number of legal actions),
        temperature will have minimal effect since visit_count^(1/T) will be similar
        for all actions.

        Args:
            root: Root node of search tree

        Returns:
            numpy array of probabilities for each legal action
        """
        legal_actions = root.state.get_legal_actions()

        if not legal_actions:
            return np.array([])

        # Get visit counts for each action
        visit_counts = []
        for action in legal_actions:
            if action in root.children:
                visit_counts.append(root.children[action].N)
            else:
                visit_counts.append(0)

        # Apply temperature
        if self.temperature == 0:
            # Deterministic: select action with highest visit count
            probs = np.zeros(len(visit_counts))
            best_action_idx = np.argmax(visit_counts)
            probs[best_action_idx] = 1.0
        else:
            # Stochastic: probabilities proportional to visit counts^(1/temperature)
            # Lower temperature -> more concentrated on high visit count actions
            # Higher temperature -> more uniform distribution
            visit_counts_array = np.array(visit_counts, dtype=np.float64)
            if np.sum(visit_counts_array) == 0:
                # No visits yet, uniform distribution
                probs = np.ones(len(visit_counts)) / len(visit_counts)
            else:
                # Apply temperature
                visit_counts_array = visit_counts_array ** (1.0 / self.temperature)
                probs_array = visit_counts_array / np.sum(visit_counts_array)
                probs = probs_array

        return probs

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """
        Add Dirichlet noise to root node for exploration.

        Args:
            root: Root node to add noise to
        """
        # First expand the root to get children
        if not root.is_expanded:
            self._expand_and_evaluate(root)

        if not root.children:
            return

        # Generate Dirichlet noise
        num_actions = len(root.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)

        # Apply noise to children
        for i, child in enumerate(root.children.values()):
            child.P = (
                1 - self.dirichlet_epsilon
            ) * child.P + self.dirichlet_epsilon * noise[i]

    def _get_terminal_value(self, state: GameState) -> float:
        """
        Get the value of a terminal state.

        Args:
            state: Terminal game state

        Returns:
            Value from current player's perspective (1 for win, -1 for loss, 0 for draw)
        """
        # This is game-specific logic that should be implemented based on the game
        # For now, return 0 (draw) as a safe default
        # In a real implementation, you would check the game result
        return 0.0


class MCTSAgent:
    """High-level agent interface for MCTS."""

    def __init__(self, neural_network: NeuralNetwork, **mcts_kwargs):
        """
        Initialize MCTS agent.

        Args:
            neural_network: Neural network for state evaluation
            **mcts_kwargs: Additional arguments for MCTS
        """
        self.mcts = MCTS(neural_network, **mcts_kwargs)

    def select_action(self, state: GameState, deterministic: bool = False) -> Any:
        """
        Select an action using MCTS.

        Args:
            state: Current game state
            deterministic: If True, select best action deterministically

        Returns:
            Selected action
        """
        # Set temperature based on deterministic flag
        original_temp = self.mcts.temperature
        if deterministic:
            self.mcts.temperature = 0.0

        try:
            # Run MCTS search
            action_probs, root_node = self.mcts.search(state)

            # Select action based on probabilities
            legal_actions = state.get_legal_actions()

            if not legal_actions:
                raise ValueError("No legal actions available")

            if deterministic or self.mcts.temperature == 0:
                # Select action with highest probability
                best_idx = np.argmax(action_probs)
                return legal_actions[best_idx]
            else:
                # Sample action according to probabilities
                action_idx = np.random.choice(len(action_probs), p=action_probs)
                return legal_actions[action_idx]

        finally:
            # Restore original temperature
            self.mcts.temperature = original_temp

    def get_action_probabilities(self, state: GameState) -> np.ndarray:
        """
        Get action probabilities for the given state.

        Args:
            state: Game state to evaluate

        Returns:
            numpy array of probabilities for each legal action
        """
        action_probs, _ = self.mcts.search(state)
        return action_probs

    def get_stats(self) -> dict:
        """Return statistics for reporting (nodes evaluated, max depth)."""
        return {
            "nodes_evaluated": getattr(self.mcts, "nodes_evaluated", 0),
            "max_depth_reached": getattr(self.mcts, "max_depth_reached", 0),
        }
