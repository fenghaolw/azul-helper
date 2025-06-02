"""
OpenSpiel-based agents for Azul.

This module provides agent implementations that use OpenSpiel's optimized
MCTS and AlphaZero algorithms instead of our custom implementations.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pyspiel

from game.azul_openspiel import AzulGame, AzulState
from game.game_state import Action, GameState
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model


class OpenSpielMCTSAgent:
    """MCTS agent using OpenSpiel's implementation."""

    def __init__(
        self,
        num_simulations: int = 100,
        uct_c: float = 1.4,
        max_memory: int = 1000000,
        solve: bool = False,
        seed: Optional[int] = None,
        evaluator: Optional[Any] = None,
    ):
        """
        Initialize OpenSpiel MCTS agent.

        Args:
            num_simulations: Number of MCTS simulations per move
            uct_c: UCT exploration constant
            max_memory: Maximum memory for search tree
            solve: Whether to use MCTS-Solver (slower but more accurate)
            seed: Random seed
            evaluator: Custom evaluator (if None, uses RandomRolloutEvaluator)
        """
        self.num_simulations = num_simulations
        self.uct_c = uct_c
        self.max_memory = max_memory
        self.solve = solve
        self.seed = seed

        # Create game instance
        self._azul_game = AzulGame()

        # Create evaluator if not provided
        if evaluator is None:
            # Use random rollout evaluator (reliable and proven)
            evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)

        # Create MCTS bot
        self._searcher = mcts.MCTSBot(
            game=self._azul_game,
            uct_c=uct_c,
            max_simulations=num_simulations,
            evaluator=evaluator,
            solve=solve,
            verbose=False,
        )

    def select_action(self, state: GameState, deterministic: bool = False) -> Action:
        """
        Select an action using MCTS.

        Args:
            state: Current game state
            deterministic: Whether to select deterministically (best action) or stochastically

        Returns:
            Selected action
        """
        # Convert to OpenSpiel state
        openspiel_state = self._convert_to_openspiel_state(state)

        # Get action from MCTS
        if deterministic:
            # Use step_with_policy to get the best action
            result = self._searcher.step_with_policy(openspiel_state)
            if isinstance(result, tuple) and len(result) == 2:
                # step_with_policy returns (policy_list, best_action)
                policy_list, action_int = result
            else:
                action_int = result
        else:
            # Use step for stochastic selection
            action_int = self._searcher.step(openspiel_state)

        # Handle case where action_int might be a list, array, or tuple
        if isinstance(action_int, (list, np.ndarray)):
            action_int = action_int[0] if len(action_int) > 0 else 0
        elif isinstance(action_int, tuple):
            action_int = action_int[0]  # Extract action from (action, value) tuple

        # Ensure it's an integer
        if isinstance(action_int, tuple):
            # Handle nested tuples
            while isinstance(action_int, tuple):
                action_int = action_int[0]

        action_int = int(action_int)

        # Convert back to Azul action
        return self._convert_to_azul_action(action_int, openspiel_state)

    def get_action_probabilities(self, state: GameState) -> np.ndarray:
        """
        Get action probabilities from MCTS search.

        Args:
            state: Current game state

        Returns:
            Array of action probabilities for legal actions
        """
        openspiel_state = self._convert_to_openspiel_state(state)

        # Run MCTS to get policy - use step_with_policy to get the actual policy
        policy_list, _ = self._searcher.step_with_policy(openspiel_state)

        # Convert policy list to dictionary for easy lookup
        policy_dict = {action: prob for action, prob in policy_list}

        # Convert to probabilities over legal actions
        legal_actions = state.get_legal_actions()
        probs = np.zeros(len(legal_actions))

        for i, action in enumerate(legal_actions):
            action_int = self._action_to_int(action, openspiel_state)
            if action_int in policy_dict:
                probs[i] = policy_dict[action_int]

        # Normalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(legal_actions)) / len(legal_actions)

        return probs

    def _convert_to_openspiel_state(self, azul_state: GameState) -> AzulState:
        """Convert Azul GameState to OpenSpiel AzulState."""
        openspiel_state = AzulState(self._azul_game, azul_state.num_players)
        openspiel_state._game_state = azul_state.copy()
        return openspiel_state

    def _convert_to_azul_action(
        self, action_int: int, openspiel_state: AzulState
    ) -> Action:
        """Convert OpenSpiel action integer to Azul Action."""
        return openspiel_state._int_to_azul_action(action_int)

    def _action_to_int(self, action: Action, openspiel_state: AzulState) -> int:
        """Convert Azul Action to OpenSpiel action integer."""
        return openspiel_state._azul_action_to_int(action)

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        return {
            "num_simulations": self.num_simulations,
            "uct_c": self.uct_c,
            "algorithm": "OpenSpiel MCTS",
        }


class OpenSpielAlphaZeroAgent:
    """AlphaZero agent using OpenSpiel's implementation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_simulations: int = 800,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        dirichlet_noise: float = 0.25,
        seed: Optional[int] = None,
    ):
        """
        Initialize OpenSpiel AlphaZero agent.

        Args:
            model_path: Path to trained neural network model
            num_simulations: Number of MCTS simulations per move
            c_puct: PUCT exploration constant
            temperature: Temperature for action selection
            dirichlet_noise: Dirichlet noise parameter
            seed: Random seed
        """
        self.model_path = model_path
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_noise = dirichlet_noise
        self.seed = seed

        # Create game instance
        self._azul_game = AzulGame()

        # Load or create neural network model
        if model_path:
            self._model = self._load_model(model_path)
        else:
            # Create default model for random play
            self._model = self._create_default_model()

        # Create evaluator and MCTS bot
        if self._model is not None:
            # Create AlphaZero evaluator
            self._evaluator = az_evaluator.AlphaZeroEvaluator(
                game=self._azul_game,
                model=self._model,
            )
        else:
            # Fall back to random rollout evaluator
            print("Warning: Using random rollout evaluator instead of AlphaZero")
            self._evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)

        # Create MCTS bot with evaluator
        self._bot = mcts.MCTSBot(
            game=self._azul_game,
            uct_c=c_puct,
            max_simulations=num_simulations,
            evaluator=self._evaluator,
            solve=False,
            dirichlet_noise=dirichlet_noise,
            verbose=False,
        )

    def _load_model(self, model_path: str) -> Any:
        """Load trained neural network model."""
        # This would load a trained AlphaZero model
        # For now, return a placeholder
        return self._create_default_model()

    def _create_default_model(self) -> Any:
        """Create a default neural network model."""
        # Create a simple neural network for Azul
        # This would be replaced with a proper model loading mechanism
        try:
            model = az_model.Model.build_model(
                "mlp",  # Model type
                self._azul_game.observation_tensor_shape(),
                self._azul_game.num_distinct_actions(),
                nn_width=128,
                nn_depth=4,
                weight_decay=1e-4,
                learning_rate=1e-3,
                path=None,  # No checkpoint path for default model
            )
            return model
        except Exception as e:
            print(f"Warning: Could not create AlphaZero model: {e}")
            # Fall back to a simple dummy model that just returns random priors
            return None

    def select_action(self, state: GameState, deterministic: bool = False) -> Action:
        """
        Select an action using AlphaZero.

        Args:
            state: Current game state
            deterministic: Whether to select deterministically

        Returns:
            Selected action
        """
        # Convert to OpenSpiel state
        openspiel_state = self._convert_to_openspiel_state(state)

        # Set temperature based on deterministic flag
        old_temp = self.temperature
        if deterministic:
            self.temperature = 0.0

        try:
            # Get action from AlphaZero
            action_int = self._bot.step(openspiel_state)

            # Convert back to Azul action
            return self._convert_to_azul_action(action_int, openspiel_state)
        finally:
            self.temperature = old_temp

    def get_action_probabilities(self, state: GameState) -> np.ndarray:
        """
        Get action probabilities from AlphaZero.

        Args:
            state: Current game state

        Returns:
            Array of action probabilities for legal actions
        """
        openspiel_state = self._convert_to_openspiel_state(state)

        # Get policy from neural network + MCTS - use step_with_policy to get the actual policy
        policy_list, _ = self._bot.step_with_policy(openspiel_state)

        # Convert policy list to dictionary for easy lookup
        policy_dict = {action: prob for action, prob in policy_list}

        # Convert to probabilities over legal actions
        legal_actions = state.get_legal_actions()
        probs = np.zeros(len(legal_actions))

        for i, action in enumerate(legal_actions):
            action_int = self._action_to_int(action, openspiel_state)
            if action_int in policy_dict:
                probs[i] = policy_dict[action_int]

        # Normalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(legal_actions)) / len(legal_actions)

        return probs

    def _convert_to_openspiel_state(self, azul_state: GameState) -> AzulState:
        """Convert Azul GameState to OpenSpiel AzulState."""
        openspiel_state = AzulState(self._azul_game, azul_state.num_players)
        openspiel_state._game_state = azul_state.copy()
        return openspiel_state

    def _convert_to_azul_action(
        self, action_int: int, openspiel_state: AzulState
    ) -> Action:
        """Convert OpenSpiel action integer to Azul Action."""
        return openspiel_state._int_to_azul_action(action_int)

    def _action_to_int(self, action: Action, openspiel_state: AzulState) -> int:
        """Convert Azul Action to OpenSpiel action integer."""
        return openspiel_state._azul_action_to_int(action)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "num_simulations": self.num_simulations,
            "c_puct": self.c_puct,
            "temperature": self.temperature,
            "algorithm": "OpenSpiel AlphaZero",
            "model_path": self.model_path,
        }


class RandomAgent:
    """Random agent for baseline comparison."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize random agent."""
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def select_action(self, state: GameState, deterministic: bool = False) -> Action:
        """Select a random legal action."""
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available")

        # Use random.choice instead of np.random.choice for Action objects
        import random

        return random.choice(legal_actions)

    def get_action_probabilities(self, state: GameState) -> np.ndarray:
        """Get uniform probabilities over legal actions."""
        legal_actions = state.get_legal_actions()
        return np.ones(len(legal_actions)) / len(legal_actions)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "algorithm": "Random",
            "seed": self.seed,
        }
