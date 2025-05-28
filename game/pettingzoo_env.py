"""
PettingZoo AEC Environment for Azul Game.

This module provides a PettingZoo-compatible environment wrapper for the Azul game,
implementing the Agent Environment Cycle (AEC) API.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from game.game_state import Action, GameState, TileColor
from game.state_representation import AzulStateRepresentation


class AzulAECEnv(AECEnv):
    """
    PettingZoo AEC Environment for Azul.

    This environment implements the Agent Environment Cycle (AEC) API for the Azul board game.
    It supports 2-4 players and provides both observation and action spaces compatible with
    reinforcement learning frameworks.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "azul_v1",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        num_players: int = 2,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Azul AEC environment.

        Args:
            num_players: Number of players (2-4)
            seed: Random seed for reproducibility
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()

        if num_players < 2 or num_players > 4:
            raise ValueError("Number of players must be between 2 and 4")

        self.num_players = num_players
        self.render_mode = render_mode
        self._seed = seed

        # Initialize agents
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = self.possible_agents[:]

        # Initialize game state
        self.game_state: Optional[GameState] = None
        self.state_representation: Optional[AzulStateRepresentation] = None

        # Action and observation spaces
        self._setup_spaces()

        # Agent selector for turn management
        self._agent_selector = agent_selector(self.agents)

        # Environment state
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}  # Instantaneous rewards
        self.infos: Dict[str, Dict[str, Any]] = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {
            agent: 0.0 for agent in self.agents
        }  # Cumulative rewards for last()

        # Current agent
        self.agent_selection: Optional[str] = None

    def _setup_spaces(self) -> None:
        """Setup action and observation spaces."""
        # Action space: We'll encode actions as integers
        # Maximum possible actions: 9 factories * 5 colors * 6 destinations = 270
        # Plus center area: 5 colors * 6 destinations = 30
        # Total: 300 (we'll use a larger space for safety)
        self.action_spaces = {
            agent: spaces.Discrete(500) for agent in self.possible_agents
        }

        # Observation space: Use the flattened state vector
        # We'll determine the exact size after creating a dummy game
        dummy_game = GameState(num_players=self.num_players, seed=42)
        dummy_repr = AzulStateRepresentation(dummy_game)
        obs_size = dummy_repr.flat_state_size

        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> None:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
        """
        if seed is not None:
            self._seed = seed

        # Reset agents
        self.agents = self.possible_agents[:]

        # Initialize new game
        self.game_state = GameState(num_players=self.num_players, seed=self._seed)
        self.state_representation = AzulStateRepresentation(self.game_state)

        # Reset environment state
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {
            agent: 0.0 for agent in self.agents
        }  # Reset instantaneous rewards
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {
            agent: 0.0 for agent in self.agents
        }  # Reset cumulative rewards

        # Reset agent selector
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action: Union[int, Action]) -> None:
        """
        Execute one step in the environment.

        Args:
            action: Action to execute (either integer or Action object)
        """
        if self.agent_selection is None:
            return

        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # Agent is done, just move to next agent
            self._was_dead_step(action)
            return

        acting_agent = self.agent_selection

        # Clear instantaneous rewards for all agents
        self.rewards = {agent: 0.0 for agent in self.agents}

        # Convert action if necessary
        if isinstance(action, int):
            action_obj = self._decode_action(action)
        else:
            action_obj = action

        # Handle invalid action
        if action_obj is None or self.game_state is None:
            # Give penalty to acting agent
            self.rewards[acting_agent] = -1.0
            self._cumulative_rewards[acting_agent] += self.rewards[acting_agent]

            self.agent_selection = self._agent_selector.next()
            return

        # Try to apply the action
        prev_scores = self.game_state.get_scores()
        success = self.game_state.apply_action(action_obj)

        if not success:
            # Give penalty to acting agent
            self.rewards[acting_agent] = -1.0
            self._cumulative_rewards[acting_agent] += self.rewards[acting_agent]

            self.agent_selection = self._agent_selector.next()
            return

        # Action was successful, update state representation
        self.state_representation = AzulStateRepresentation(self.game_state)

        # Calculate score change for acting agent
        current_scores = self.game_state.get_scores()
        current_player_idx = int(acting_agent.split("_")[1])

        score_diff = (
            current_scores[current_player_idx] - prev_scores[current_player_idx]
        )

        # Give reward to acting agent
        self.rewards[acting_agent] = float(score_diff)
        self._cumulative_rewards[acting_agent] += self.rewards[acting_agent]

        # Check if game is over
        if self.game_state.game_over:
            final_scores = self.game_state.get_scores()
            winner_score = max(final_scores)

            # Give winner bonus to all winning agents
            for i, agent_id in enumerate(self.agents):
                if final_scores[i] == winner_score:
                    winner_bonus = 10.0
                    self.rewards[agent_id] += winner_bonus
                    self._cumulative_rewards[agent_id] += winner_bonus

                # Mark all agents as terminated
                self.terminations[agent_id] = True

            self.agent_selection = None
        else:
            self.agent_selection = self._agent_selector.next()

    def observe(self, agent: str) -> np.ndarray:
        """
        Get observation for a specific agent.

        Args:
            agent: Agent identifier

        Returns:
            Observation array for the agent
        """
        if self.state_representation is None:
            # Return zero observation if game not started
            obs_size = self.observation_spaces[agent].shape[0]
            return np.zeros(obs_size, dtype=np.float32)

        # Get player-specific view
        # Note: We could use player_view for player-specific observations in the future
        # player_id = int(agent.split("_")[1])
        # player_view = self.state_representation.get_player_view(
        #     player_id=player_id,
        #     include_hidden=False,  # Hide bag contents from other players
        # )

        # Convert to flat vector
        # We'll use the same flattening as the state representation
        # but from the player's perspective
        return self.state_representation.get_flat_state_vector(normalize=True)

    def last(self, observe: bool = True) -> tuple:
        """
        Get the last observation, reward, termination, truncation, and info.

        Args:
            observe: Whether to return the observation (PettingZoo API requirement)

        Returns:
            Tuple of (observation, reward, termination, truncation, info)
        """
        agent = self.agent_selection

        if agent is None:
            # Game is over, use the last agent that acted
            agent = self._agent_selector.last_agent
            if agent is None:
                agent = self.agents[0] if self.agents else self.possible_agents[0]

        obs = self.observe(agent) if observe else None

        # Return cumulative reward and reset it
        reward = self._cumulative_rewards.get(agent, 0.0)
        self._cumulative_rewards[agent] = 0.0

        return (
            obs,
            reward,
            self.terminations.get(agent, True),
            self.truncations.get(agent, False),
            self.infos.get(agent, {}),
        )

    def _was_dead_step(self, action):
        """Handle step for an agent that is already terminated/truncated."""
        # Just move to the next agent
        self.agent_selection = self._agent_selector.next()

    def _accumulate_rewards(self):
        """Accumulate rewards (for compatibility with PettingZoo wrappers)."""
        # This method is called by some PettingZoo wrappers
        # Our implementation already handles cumulative rewards properly
        pass

    def render(self) -> Optional[Union[str, np.ndarray]]:
        """
        Render the environment.

        Returns:
            Rendered output (string for human mode, array for rgb_array mode)
        """
        if self.game_state is None:
            return "Game not initialized"

        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        else:
            return None

    def close(self) -> None:
        """
        Close the environment and clean up resources.

        Required by PettingZoo API when render method is defined.
        """
        # No resources to clean up for this environment
        pass

    def _render_human(self) -> str:
        """Render in human-readable text format."""
        if self.game_state is None:
            return "Game not initialized"

        output = []
        output.append(f"=== Azul Game - Round {self.game_state.round_number} ===")
        output.append(f"Current Player: {self.agent_selection}")
        output.append(f"Game Over: {self.game_state.game_over}")

        if self.game_state.game_over and self.game_state.winner is not None:
            output.append(f"Winner: player_{self.game_state.winner}")

        # Player scores
        scores = self.game_state.get_scores()
        output.append("\nScores:")
        for i, score in enumerate(scores):
            output.append(f"  player_{i}: {score}")

        # Factory displays
        output.append("\nFactories:")
        for i, factory in enumerate(self.game_state.factory_area.factories):
            colors = [tile.color.value for tile in factory.tiles]
            output.append(f"  Factory {i}: {colors}")

        # Center area
        center_colors = [
            tile.color.value for tile in self.game_state.factory_area.center.tiles
        ]
        output.append(f"  Center: {center_colors}")
        if self.game_state.factory_area.center.has_first_player_marker:
            output.append("  (First player marker in center)")

        return "\n".join(output)

    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array (placeholder implementation)."""
        # This would require a more complex rendering system
        # For now, return a simple placeholder
        return np.zeros((400, 600, 3), dtype=np.uint8)

    def _encode_action(self, action: Action) -> int:
        """
        Encode an Action object to an integer.

        Args:
            action: Action object to encode

        Returns:
            Integer encoding of the action
        """
        # Encoding scheme:
        # source: -1 (center) maps to 0, factories 0-8 map to 1-9
        # color: 0-4 (blue, yellow, red, black, white)
        # destination: -1 (floor) maps to 0, pattern lines 0-4 map to 1-5

        source_encoded = action.source + 1  # -1 -> 0, 0-8 -> 1-9

        # Handle color encoding - convert to integer
        color_value: Union[str, int, TileColor]
        if hasattr(action.color, "value"):
            # TileColor enum case
            color_value = action.color.value
        else:
            # Direct value case (could be TileColor, str, or int)
            color_value = action.color

        # Convert color string to index if needed
        color_encoded: int
        if isinstance(color_value, str):
            color_map: Dict[str, int] = {
                "blue": 0,
                "yellow": 1,
                "red": 2,
                "black": 3,
                "white": 4,
            }
            if color_value in color_map:
                color_encoded = color_map[color_value]
            else:
                color_encoded = 0
        elif isinstance(color_value, TileColor):
            # Handle TileColor enum directly - convert string value to index
            color_string = color_value.value
            color_map_for_enum: Dict[str, int] = {
                "blue": 0,
                "yellow": 1,
                "red": 2,
                "black": 3,
                "white": 4,
            }
            if color_string in color_map_for_enum:
                color_encoded = color_map_for_enum[color_string]
            else:
                color_encoded = 0
        else:
            # Already an integer
            color_encoded = int(color_value)

        dest_encoded = action.destination + 1  # -1 -> 0, 0-4 -> 1-5

        # Combine: source (10 values) * color (5 values) * destination (6 values) = 300 combinations
        return source_encoded * 30 + color_encoded * 6 + dest_encoded

    def _decode_action(self, action_int: int) -> Optional[Action]:
        """
        Decode an integer to an Action object.

        Args:
            action_int: Integer encoding of the action

        Returns:
            Action object or None if invalid
        """
        if action_int < 0 or action_int >= 500:
            return None

        # Reverse the encoding
        dest_encoded = action_int % 6
        color_encoded = (action_int // 6) % 5
        source_encoded = action_int // 30

        # Convert back to original values
        source = source_encoded - 1  # 0 -> -1, 1-9 -> 0-8
        destination = dest_encoded - 1  # 0 -> -1, 1-5 -> 0-4

        # Convert color index to TileColor
        color_map = [
            TileColor.BLUE,
            TileColor.YELLOW,
            TileColor.RED,
            TileColor.BLACK,
            TileColor.WHITE,
        ]
        if color_encoded >= len(color_map):
            return None
        color = color_map[color_encoded]

        # Validate ranges
        if source < -1 or source >= 9:  # -1 for center, 0-8 for factories
            return None
        if destination < -1 or destination >= 5:  # -1 for floor, 0-4 for pattern lines
            return None

        return Action(source=source, color=color, destination=destination)

    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for an agent."""
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for an agent."""
        return self.observation_spaces[agent]

    def get_legal_actions(self, agent: Optional[str] = None) -> List[int]:
        """
        Get legal actions for an agent as integer encodings.

        Args:
            agent: Agent identifier (uses current agent if None)

        Returns:
            List of legal action integers
        """
        if agent is None:
            agent = self.agent_selection

        if agent is None or self.game_state is None:
            return []

        player_id = int(agent.split("_")[1])
        legal_actions = self.game_state.get_legal_actions(player_id)

        return [self._encode_action(action) for action in legal_actions]


def env(**kwargs) -> AzulAECEnv:
    """
    Create an Azul AEC environment.

    Args:
        **kwargs: Arguments passed to AzulAECEnv constructor

    Returns:
        AzulAECEnv instance
    """
    return AzulAECEnv(**kwargs)


def raw_env(**kwargs) -> AzulAECEnv:
    """
    Create a raw Azul AEC environment (without wrappers).

    Args:
        **kwargs: Arguments passed to AzulAECEnv constructor

    Returns:
        AzulAECEnv instance
    """
    return AzulAECEnv(**kwargs)


# Wrapped environment with common PettingZoo wrappers
def wrapped_env(**kwargs) -> wrappers.OrderEnforcingWrapper:
    """
    Create a wrapped Azul AEC environment with standard PettingZoo wrappers.

    Args:
        **kwargs: Arguments passed to AzulAECEnv constructor

    Returns:
        Wrapped AzulAECEnv instance
    """
    # Set render_mode to "human" if not specified for CaptureStdoutWrapper compatibility
    if "render_mode" not in kwargs:
        kwargs["render_mode"] = "human"

    env_instance = AzulAECEnv(**kwargs)

    # Only add CaptureStdoutWrapper if render_mode is "human"
    if kwargs.get("render_mode") == "human":
        env_instance = wrappers.CaptureStdoutWrapper(env_instance)

    env_instance = wrappers.TerminateIllegalWrapper(env_instance, illegal_reward=-1)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance
