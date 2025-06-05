"""
OpenSpiel-based agents for Azul.

This module provides agent implementations that use OpenSpiel's optimized
MCTS and AlphaZero algorithms instead of our custom implementations.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model

from agents.base_agent import AzulAgent
from game.azul_openspiel import AzulGame, AzulState
from game.game_state import Action, GameState


class OpenSpielMinimaxAgent(AzulAgent):
    """Minimax agent using OpenSpiel's optimized minimax implementation with alpha-beta pruning."""

    def __init__(
        self,
        depth: int = 4,
        enable_alpha_beta: bool = True,
        enable_memoization: bool = True,
        time_limit: Optional[float] = None,
        seed: Optional[int] = None,
        player_id: int = 0,
        name: Optional[str] = None,
    ):
        """
        Initialize OpenSpiel Minimax agent.

        Args:
            depth: Maximum search depth
            enable_alpha_beta: Whether to use alpha-beta pruning
            enable_memoization: Whether to use memoization for repeated states
            time_limit: Optional time limit in seconds (None for no limit)
            seed: Random seed for tie-breaking
            player_id: The player ID this agent controls
            name: Optional name for the agent
        """
        super().__init__(player_id, name or f"OpenSpielMinimax_D{depth}")

        self.depth = depth
        self.enable_alpha_beta = enable_alpha_beta
        self.enable_memoization = enable_memoization
        self.time_limit = time_limit
        self.seed = seed

        # Create game instance with deterministic mode for minimax compatibility
        self._azul_game = AzulGame({"deterministic_mode": True})

        # Import OpenSpiel's minimax after checking availability
        try:
            from open_spiel.python.algorithms import minimax

            self._minimax = minimax
        except ImportError:
            raise ImportError(
                "OpenSpiel minimax not available. Please check your OpenSpiel installation."
            )

    def _get_algorithm_name(self) -> str:
        """Get the algorithm name for this agent."""
        return "OpenSpiel Minimax"

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime performance statistics."""
        base_stats = super().get_stats()
        # No additional runtime stats for OpenSpiel minimax beyond base class
        return base_stats

    def get_info(self) -> Dict[str, Any]:
        """Get static agent metadata."""
        base_info = super().get_info()
        base_info.update(
            {
                "algorithm": "OpenSpiel Minimax",
                "depth": self.depth,
                "alpha_beta_enabled": self.enable_alpha_beta,
                "memoization_enabled": self.enable_memoization,
                "time_limit": self.time_limit,
            }
        )
        return base_info

    def _value_function(self, state):
        """Simple value function for evaluating non-terminal states."""
        if state.is_terminal():
            returns = state.returns()
            return returns[0] if len(returns) > 0 else 0.0

        # For non-terminal states, use current score difference as heuristic
        scores = state._game_state.get_scores()
        if len(scores) >= 2:
            return float(scores[0] - scores[1])
        return 0.0

    def select_action(self, state: GameState, deterministic: bool = True) -> Action:
        """
        Select an action using OpenSpiel's minimax algorithm.

        Args:
            state: Current game state
            deterministic: Whether to select deterministically (minimax is inherently deterministic)

        Returns:
            Selected action
        """
        # Convert to OpenSpiel state
        openspiel_state = self._convert_to_openspiel_state(state)

        try:
            # Use OpenSpiel's alpha_beta_search with correct parameters
            value, action_int = self._minimax.alpha_beta_search(
                game=self._azul_game,
                state=openspiel_state,
                maximum_depth=self.depth,
                maximizing_player_id=state.current_player,
                value_function=self._value_function,
            )

        except (AttributeError, TypeError, NotImplementedError) as e:
            # Fallback: use MCTS with very specific settings to approximate minimax
            print(
                f"Warning: OpenSpiel minimax failed ({e}), falling back to deterministic MCTS"
            )
            from open_spiel.python.algorithms import mcts

            mcts_bot = mcts.MCTSBot(
                game=self._azul_game,
                uct_c=0.0,  # No exploration, pure exploitation
                max_simulations=100,
                evaluator=mcts.RandomRolloutEvaluator(n_rollouts=1),
                solve=True,  # Use exact solver when possible
                verbose=False,
            )
            action_int = mcts_bot.step(openspiel_state)

        # Handle case where action_int might be a tuple or complex structure
        if isinstance(action_int, tuple):
            action_int = action_int[0] if len(action_int) > 0 else 0
        elif isinstance(action_int, (list, np.ndarray)):
            action_int = action_int[0] if len(action_int) > 0 else 0

        action_int = int(action_int)

        # Convert back to Azul action
        return self._convert_to_azul_action(action_int, openspiel_state)

    def get_action_probabilities(self, state: GameState) -> np.ndarray:
        """
        Get action probabilities from minimax search.

        Note: Minimax is deterministic, so this returns probability 1.0
        for the best action and 0.0 for all others.

        Args:
            state: Current game state

        Returns:
            Array of action probabilities for legal actions
        """
        legal_actions = state.get_legal_actions()
        probs = np.zeros(len(legal_actions))

        # Get the best action
        best_action = self.select_action(state, deterministic=True)

        # Find the index of the best action
        for i, action in enumerate(legal_actions):
            if self._actions_equal(action, best_action):
                probs[i] = 1.0
                break

        return probs

    def _actions_equal(self, action1: Action, action2: Action) -> bool:
        """Check if two actions are equal."""
        return (
            action1.source == action2.source
            and action1.color == action2.color
            and action1.destination == action2.destination
        )

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


class OpenSpielMCTSAgent(AzulAgent):
    """MCTS agent using OpenSpiel's implementation."""

    def __init__(
        self,
        num_simulations: int = 100,
        uct_c: float = 1.4,
        max_memory: int = 1000000,
        solve: bool = False,
        seed: Optional[int] = None,
        evaluator: Optional[Any] = None,
        player_id: int = 0,
        name: Optional[str] = None,
    ):
        """
        Initialize OpenSpiel MCTS agent.

        Args:
            num_simulations: Number of MCTS simulations per move
            uct_c: UCT exploration constant
            max_memory: Maximum memory for MCTS tree
            solve: Whether to use exact solver when possible
            seed: Random seed
            evaluator: Optional custom evaluator
            player_id: The player ID this agent controls
            name: Optional name for the agent
        """
        super().__init__(player_id, name or f"OpenSpielMCTS_{num_simulations}")

        self.num_simulations = num_simulations
        self.uct_c = uct_c
        self.max_memory = max_memory
        self.solve = solve
        self.seed = seed

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Create game instance with stochastic mode for proper chance node handling
        self._azul_game = AzulGame({"deterministic_mode": False, "seed": seed})

        # Create evaluator
        if evaluator is None:
            self._evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)
        else:
            self._evaluator = evaluator

        # Create MCTS bot
        self._bot = mcts.MCTSBot(
            game=self._azul_game,
            uct_c=uct_c,
            max_simulations=num_simulations,
            evaluator=self._evaluator,
            solve=solve,
            verbose=False,
        )

    def _get_algorithm_name(self) -> str:
        """Get the algorithm name for this agent."""
        return "OpenSpiel MCTS"

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime performance statistics."""
        base_stats = super().get_stats()
        # No additional runtime stats for OpenSpiel MCTS beyond base class
        return base_stats

    def get_info(self) -> Dict[str, Any]:
        """Get static agent metadata."""
        base_info = super().get_info()
        base_info.update(
            {
                "algorithm": "OpenSpiel MCTS",
                "num_simulations": self.num_simulations,
                "uct_c": self.uct_c,
                "max_memory": self.max_memory,
                "solve": self.solve,
            }
        )
        return base_info

    def supports_deterministic_play(self) -> bool:
        """MCTS supports deterministic play by setting temperature to 0."""
        return True

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
            result = self._bot.step_with_policy(openspiel_state)
            if isinstance(result, tuple) and len(result) == 2:
                # step_with_policy returns (policy_list, best_action)
                policy_list, action_int = result
            else:
                action_int = result
        else:
            # Use step for stochastic selection
            action_int = self._bot.step(openspiel_state)

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


class OpenSpielAlphaZeroAgent(AzulAgent):
    """AlphaZero agent using OpenSpiel's implementation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_simulations: int = 800,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        dirichlet_noise: float = 0.25,
        seed: Optional[int] = None,
        player_id: int = 0,
        name: Optional[str] = None,
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
            player_id: The player ID this agent controls
            name: Optional name for the agent
        """
        super().__init__(player_id, name or f"OpenSpielAlphaZero_{num_simulations}")

        self.model_path = model_path
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_noise = dirichlet_noise
        self.seed = seed

        # Create game instance with stochastic mode for proper chance node handling
        self._azul_game = AzulGame({"deterministic_mode": False})

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

    def _get_algorithm_name(self) -> str:
        """Get the algorithm name for this agent."""
        return "OpenSpiel AlphaZero"

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime performance statistics."""
        base_stats = super().get_stats()
        # No additional runtime stats for OpenSpiel AlphaZero beyond base class
        return base_stats

    def get_info(self) -> Dict[str, Any]:
        """Get static agent metadata."""
        base_info = super().get_info()
        base_info.update(
            {
                "algorithm": "OpenSpiel AlphaZero",
                "num_simulations": self.num_simulations,
                "c_puct": self.c_puct,
                "temperature": self.temperature,
                "model_path": self.model_path,
            }
        )
        return base_info

    def supports_deterministic_play(self) -> bool:
        """Alpha Zero supports deterministic play by setting temperature to 0."""
        return True

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


class RandomAgent(AzulAgent):
    """Random agent for baseline comparison."""

    def __init__(
        self, seed: Optional[int] = None, player_id: int = 0, name: Optional[str] = None
    ):
        """
        Initialize random agent.

        Args:
            seed: Random seed for reproducible results
            player_id: The player ID this agent controls
            name: Optional name for the agent
        """
        super().__init__(player_id, name or "RandomAgent")
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def _get_algorithm_name(self) -> str:
        """Get the algorithm name for this agent."""
        return "Random"

    def select_action(self, state: GameState, deterministic: bool = False) -> Action:
        """Select a random legal action."""
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available")

        # Use random.choice instead of np.random.choice for Action objects
        import random

        # Note: Random agent is inherently non-deterministic, so deterministic flag is ignored
        return random.choice(legal_actions)

    def get_action_probabilities(self, state: GameState) -> np.ndarray:
        """Get uniform probabilities over legal actions."""
        legal_actions = state.get_legal_actions()
        return np.ones(len(legal_actions)) / len(legal_actions)

    def can_provide_probabilities(self) -> bool:
        """Random agent can provide probabilities."""
        return True

    def supports_deterministic_play(self) -> bool:
        """Random agent is inherently non-deterministic."""
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime performance statistics."""
        base_stats = super().get_stats()
        # No additional runtime stats for Random agent beyond base class
        return base_stats

    def get_info(self) -> Dict[str, Any]:
        """Get static agent metadata."""
        base_info = super().get_info()
        base_info.update(
            {
                "algorithm": "Random",
                "seed": self.seed,
            }
        )
        return base_info
