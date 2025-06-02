"""
OpenSpiel-compatible Azul game implementation.

This module adapts the existing Azul GameState implementation to work with OpenSpiel's
APIs, allowing us to use OpenSpiel's MCTS and AlphaZero implementations.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pyspiel

from game.game_state import Action, GameState, TileColor


class AzulGame(pyspiel.Game):
    """OpenSpiel-compatible Azul game implementation."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = {}

        self._num_players = params.get("players", 2)
        if self._num_players < 2 or self._num_players > 4:
            raise ValueError("Number of players must be between 2 and 4")

        self._seed = params.get("seed", None)

        super().__init__(
            pyspiel.GameType(
                short_name="azul",
                long_name="Azul",
                dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
                chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
                information=pyspiel.GameType.Information.PERFECT_INFORMATION,
                utility=(
                    pyspiel.GameType.Utility.ZERO_SUM
                    if self._num_players == 2
                    else pyspiel.GameType.Utility.GENERAL_SUM
                ),
                reward_model=pyspiel.GameType.RewardModel.TERMINAL,
                max_num_players=4,
                min_num_players=2,
                provides_information_state_string=True,
                provides_information_state_tensor=True,
                provides_observation_string=True,
                provides_observation_tensor=True,
                provides_factored_observation_string=False,
            ),
            pyspiel.GameInfo(
                num_distinct_actions=self._calculate_max_actions(),
                max_chance_outcomes=100,  # Max tiles that can be drawn
                num_players=self._num_players,
                min_utility=-100.0,  # Approximate minimum score
                max_utility=100.0,  # Approximate maximum score
                utility_sum=0.0 if self._num_players == 2 else None,
                max_game_length=200,  # Conservative estimate
            ),
            params or {},
        )

    def _calculate_max_actions(self) -> int:
        """Calculate the maximum number of distinct actions possible."""
        # 5 factories + 1 center = 6 sources
        # 5 colors
        # 5 pattern lines + 1 floor = 6 destinations
        return 6 * 5 * 6

    def new_initial_state(self) -> "AzulState":
        """Create a new initial state."""
        return AzulState(self, self._num_players, self._seed)

    def num_distinct_actions(self) -> int:
        """Return the number of distinct actions."""
        return self._calculate_max_actions()

    def num_players(self) -> int:
        """Return the number of players."""
        return self._num_players

    def min_utility(self) -> float:
        """Return the minimum utility."""
        return -100.0

    def max_utility(self) -> float:
        """Return the maximum utility."""
        return 100.0

    def utility_sum(self) -> float:
        """Return the sum of utilities (0 for zero-sum games)."""
        return 0.0

    def observation_tensor_shape(self) -> List[int]:
        """Return the shape of the observation tensor."""
        # This will be calculated based on state representation
        return [500]  # Placeholder - will be refined

    def information_state_tensor_shape(self) -> List[int]:
        """Return the shape of the information state tensor."""
        return self.observation_tensor_shape()

    def max_game_length(self) -> int:
        """Return the maximum game length."""
        return 200


class AzulState(pyspiel.State):
    """OpenSpiel-compatible Azul state implementation."""

    def __init__(self, game: AzulGame, num_players: int, seed: Optional[int] = None):
        super().__init__(game)
        self._game_state = GameState(num_players, seed)
        self._action_mapping = self._build_action_mapping()
        self._reverse_action_mapping = {v: k for k, v in self._action_mapping.items()}

    def _build_action_mapping(self) -> Dict[Action, int]:
        """Build mapping from Azul Actions to OpenSpiel action integers."""
        mapping = {}
        action_id = 0

        # Map all possible actions to integers
        # Source: -1 (center), 0-4 (factories)
        # Color: all TileColor enum values
        # Destination: -1 (floor), 0-4 (pattern lines)
        for source in range(-1, 5):  # -1 for center, 0-4 for factories
            for color in TileColor:
                for dest in range(-1, 5):  # -1 for floor, 0-4 for pattern lines
                    action = Action(source, color, dest)
                    mapping[action] = action_id
                    action_id += 1

        return mapping

    def _azul_action_to_int(self, action: Action) -> int:
        """Convert Azul Action to OpenSpiel integer action."""
        # Create a comparable action key
        action_key = Action(action.source, action.color, action.destination)
        if action_key in self._action_mapping:
            return self._action_mapping[action_key]
        else:
            # If exact action not found, try to find a match by attributes
            for mapped_action, action_id in self._action_mapping.items():
                if (
                    mapped_action.source == action.source
                    and mapped_action.color == action.color
                    and mapped_action.destination == action.destination
                ):
                    return action_id
            raise ValueError(f"Action {action} not found in mapping")

    def _int_to_azul_action(self, action_int: int) -> Action:
        """Convert OpenSpiel integer action to Azul Action."""
        if action_int in self._reverse_action_mapping:
            return self._reverse_action_mapping[action_int]
        else:
            raise ValueError(
                f"Action integer {action_int} not found in reverse mapping"
            )

    def current_player(self) -> int:
        """Return the current player."""
        if self._game_state.game_over:
            return pyspiel.PlayerId.TERMINAL
        return self._game_state.current_player

    def legal_actions(self, player: Optional[int] = None) -> List[int]:
        """Return legal actions as OpenSpiel integers."""
        if player is None:
            player = self.current_player()

        if self._game_state.game_over or player == pyspiel.PlayerId.TERMINAL:
            return []

        azul_actions = self._game_state.get_legal_actions(player)
        return [self._azul_action_to_int(action) for action in azul_actions]

    def legal_actions_mask(self, player: Optional[int] = None) -> np.ndarray:
        """Return a boolean mask indicating which actions are legal."""
        if player is None:
            player = self.current_player()

        # Create a mask for all possible actions
        mask = np.zeros(self.get_game().num_distinct_actions(), dtype=bool)

        if not self._game_state.game_over and player != pyspiel.PlayerId.TERMINAL:
            legal_action_ints = self.legal_actions(player)
            mask[legal_action_ints] = True

        return mask

    def apply_action(self, action) -> None:
        """Apply an action to the state."""
        # Convert numpy types to regular int
        if hasattr(action, "item"):  # numpy types have .item() method
            action = action.item()

        if isinstance(action, (int, np.integer)):
            azul_action = self._int_to_azul_action(int(action))
        else:
            azul_action = action

        self._game_state.apply_action(azul_action)

    def is_terminal(self) -> bool:
        """Check if the state is terminal."""
        return self._game_state.game_over

    def returns(self) -> List[float]:
        """Return the returns for each player."""
        if not self.is_terminal():
            return [0.0] * self._game_state.num_players

        scores = self._game_state.get_scores()
        if self._game_state.num_players == 2:
            # Zero-sum: normalize so that utilities sum to 0
            return [scores[0] - scores[1], scores[1] - scores[0]]
        else:
            # Multi-player: return raw scores
            return [float(score) for score in scores]

    def rewards(self) -> List[float]:
        """Return the immediate rewards."""
        # Azul gives rewards only at the end
        if self.is_terminal():
            return self.returns()
        return [0.0] * self._game_state.num_players

    def player_reward(self, player: int) -> float:
        """Return the reward for a specific player."""
        rewards = self.rewards()
        return rewards[player] if player < len(rewards) else 0.0

    def is_chance_node(self) -> bool:
        """Check if this is a chance node."""
        # In our Azul implementation, chance is handled internally
        # (tile drawing from bag), so from OpenSpiel's perspective,
        # there are no explicit chance nodes
        return False

    def chance_outcomes(self) -> List[tuple]:
        """Return chance outcomes (not used in our implementation)."""
        return []

    def observation_string(self, player: Optional[int] = None) -> str:
        """Return observation as string."""
        if player is None:
            player = self.current_player()
        # Use the existing game state representation
        return str(self._game_state)

    def information_state_string(self, player: Optional[int] = None) -> str:
        """Return information state as string."""
        return self.observation_string(player)

    def observation_tensor(self, player: Optional[int] = None) -> np.ndarray:
        """Return observation as tensor."""
        if player is None:
            player = self.current_player()

        # Use the existing state vector representation
        state_vector = self._game_state.get_state_vector()
        return np.array(state_vector, dtype=np.float32)

    def information_state_tensor(self, player: Optional[int] = None) -> np.ndarray:
        """Return information state as tensor."""
        return self.observation_tensor(player)

    def action_to_string(self, player: int, action: int) -> str:
        """Convert action to string representation."""
        azul_action = self._int_to_azul_action(action)
        return str(azul_action)

    def __str__(self) -> str:
        """String representation of the state."""
        return str(self._game_state)

    def clone(self) -> "AzulState":
        """Create a copy of this state."""
        new_state = AzulState(self.get_game(), self._game_state.num_players)
        new_state._game_state = self._game_state.copy()
        return new_state


# Register the game with OpenSpiel
pyspiel.register_game(
    pyspiel.GameType(
        short_name="azul",
        long_name="Azul",
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
        information=pyspiel.GameType.Information.PERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.ZERO_SUM,
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        max_num_players=4,
        min_num_players=2,
        provides_information_state_string=True,
        provides_information_state_tensor=True,
        provides_observation_string=True,
        provides_observation_tensor=True,
    ),
    AzulGame,
)
