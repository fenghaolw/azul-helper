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

        # Chance mode configuration for different algorithms
        # - DETERMINISTIC: Required for OpenSpiel minimax (treats current state as deterministic)
        # - EXPLICIT_STOCHASTIC: More accurate for MCTS/AlphaZero (models tile drawing randomness)
        # The choice depends on the algorithm being used
        use_deterministic = params.get("deterministic_mode", False)
        self.chance_mode = (
            pyspiel.GameType.ChanceMode.DETERMINISTIC
            if use_deterministic
            else pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC
        )

        super().__init__(
            pyspiel.GameType(
                short_name="azul",
                long_name="Azul",
                dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
                chance_mode=self.chance_mode,
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
                num_distinct_actions=180,  # Optimized encoding with exact action space
                max_chance_outcomes=100,  # Max tiles that can be drawn
                num_players=self._num_players,
                min_utility=-100.0,  # Approximate minimum score
                max_utility=100.0,  # Approximate maximum score
                utility_sum=0.0 if self._num_players == 2 else None,
                max_game_length=200,  # Conservative estimate
            ),
            params or {},
        )

    def new_initial_state(self) -> "AzulState":
        """Create a new initial state."""
        return AzulState(self, self._num_players, self._seed)

    def num_distinct_actions(self) -> int:
        """The maximum number of distinct actions in the game."""
        # 6 sources (center + 5 factories)
        # 5 colors (first_player token is not explicitly chosen)
        # 6 destinations (floor + 5 pattern lines)
        # Optimized encoding: 6 * 5 * 6 = 180 actions exactly
        return 180

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
        # This matches the actual size returned by GameState.get_state_vector()
        return [935]

    def information_state_tensor_shape(self) -> List[int]:
        """Return the shape of the information state tensor."""
        return self.observation_tensor_shape()

    def max_game_length(self) -> int:
        """Return the maximum game length."""
        return 200


class AzulState(pyspiel.State):
    """OpenSpiel state wrapper for Azul game."""

    # Color to index mapping for efficient conversion (FIRST_PLAYER excluded from encoding)
    _COLOR_TO_IDX = {
        TileColor.BLUE: 0,
        TileColor.YELLOW: 1,
        TileColor.RED: 2,
        TileColor.BLACK: 3,
        TileColor.WHITE: 4,
        # TileColor.FIRST_PLAYER: not included in encoding (never explicitly chosen)
    }

    # Index to color mapping for reverse conversion
    _IDX_TO_COLOR = {v: k for k, v in _COLOR_TO_IDX.items()}

    def __init__(self, game: AzulGame, num_players: int, seed: Optional[int] = None):
        pyspiel.State.__init__(self, game)
        self._game_state = GameState(num_players, seed)

    def _int_to_azul_action(self, action_int: int) -> Action:
        """Convert OpenSpiel integer action to Azul Action using optimized encoding."""
        # Input validation
        if not (0 <= action_int < 180):
            raise ValueError(
                f"Action integer {action_int} is out of valid range [0, 179]"
            )
        # Reverse the optimized formula: source_idx * 30 + color_idx * 6 + dest_idx
        source_idx = action_int // 30
        remainder = action_int % 30
        color_idx = remainder // 6
        dest_idx = remainder % 6

        source = source_idx - 1  # 0 becomes -1, 1-5 becomes 0-4
        color = self._IDX_TO_COLOR[color_idx]
        destination = dest_idx - 1  # 0 becomes -1, 1-5 becomes 0-4

        return Action(source, color, destination)

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

        # Batch convert actions using optimized encoding
        action_ints = []
        for action in azul_actions:
            source_idx = action.source + 1
            dest_idx = action.destination + 1
            color_idx = self._COLOR_TO_IDX[action.color]
            # Optimized encoding: source_idx * 30 + color_idx * 6 + dest_idx
            action_int = source_idx * 30 + color_idx * 6 + dest_idx
            action_ints.append(action_int)

        return action_ints

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
        max_score = max(scores)
        winners = [i for i, score in enumerate(scores) if score == max_score]

        if self._game_state.num_players == 2:
            # Zero-sum: winner gets 1.0, loser gets -1.0
            if len(winners) == 1:
                return [1.0 if i == winners[0] else -1.0 for i in range(2)]
            else:
                # Draw
                return [0.0, 0.0]
        else:
            # Multi-player: winners get 1.0, others get -1.0/(n-1)
            n = len(scores)
            return [1.0 if i in winners else -1.0 / (n - 1) for i in range(n)]

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
        # Create new state efficiently
        new_state = AzulState.__new__(AzulState)
        pyspiel.State.__init__(new_state, self.get_game())

        # Copy the game state
        new_state._game_state = self._game_state.copy()

        return new_state


# Registration is handled by game/register_azul.py
