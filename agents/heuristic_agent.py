"""
Heuristic-based Azul AI Agent

This agent makes decisions based on hand-crafted heuristics and game knowledge
rather than random exploration. It provides intelligent gameplay while neural
networks are being trained.
"""

import random
from typing import List, Optional, Tuple

from agents.base_agent import AzulAgent
from game.game_state import Action, GameState, PlayerBoard, TileColor


class HeuristicAgent(AzulAgent):
    """
    A heuristic-based agent for Azul that makes strategic decisions
    based on game knowledge and scoring opportunities.
    """

    def __init__(self, player_id: int = 0, name: Optional[str] = None):
        """
        Initialize the heuristic agent.

        Args:
            player_id: The player ID this agent controls
            name: Optional name for the agent
        """
        super().__init__(player_id, name or "HeuristicAgent")

    def select_action(
        self, game_state: GameState, deterministic: bool = False
    ) -> Action:
        """
        Select the best action based on heuristic evaluation.

        Args:
            game_state: Current game state
            deterministic: Whether to select deterministically (always True for heuristic)

        Returns:
            Best action according to heuristics
        """
        self.nodes_evaluated = 0

        # Dynamically determine our player ID from the current game state
        # This allows the agent to work correctly even when position swapping is enabled
        current_player_id = game_state.current_player

        available_actions = game_state.get_legal_actions()

        if not available_actions:
            # Check if game is over before raising error
            if game_state.game_over:
                # Game has ended, this is expected
                raise ValueError("No legal actions available - game is over")
            else:
                # Game is not over but no actions available - this indicates a bug
                raise ValueError(
                    "No legal actions available - possible game state inconsistency"
                )

        if len(available_actions) == 1:
            self.nodes_evaluated = 1
            return available_actions[0]

        # Evaluate all possible actions
        best_action = available_actions[
            0
        ]  # Initialize with first action instead of None
        best_score = float("-inf")

        for action in available_actions:
            score = self._evaluate_action(game_state, action, current_player_id)
            self.nodes_evaluated += 1

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _evaluate_action(
        self, game_state: GameState, action: Action, player_id: int
    ) -> float:
        """
        Evaluate the quality of an action using multiple heuristics.

        Args:
            game_state: Current game state
            action: Action to evaluate
            player_id: The ID of the player we're evaluating for

        Returns:
            Heuristic score for the action (higher is better)
        """
        player_board = game_state.players[player_id]
        score = 0.0

        # Get the number of tiles we would take
        tiles_taken = self._count_tiles_taken(game_state, action)

        if action.destination == -1:  # Floor line
            score += self._evaluate_floor_move(tiles_taken)
        else:  # Pattern line
            score += self._evaluate_pattern_line_move(
                game_state, action, player_board, tiles_taken
            )

        # Add strategic considerations
        score += self._evaluate_strategic_factors(game_state, action, player_board)

        # Add small random factor to break ties
        score += random.uniform(-0.1, 0.1)

        return score

    def _count_tiles_taken(self, game_state: GameState, action: Action) -> int:
        """Count how many tiles would be taken by this action."""
        if action.source == -1:  # Center
            return sum(
                1
                for tile in game_state.factory_area.center.tiles
                if tile.color == action.color
            )
        else:  # Factory
            factory = game_state.factory_area.factories[action.source]
            return sum(1 for tile in factory.tiles if tile.color == action.color)

    def _evaluate_floor_move(self, tiles_taken: int) -> float:
        """
        Evaluate a move to the floor line.
        Generally negative, but sometimes necessary.
        """
        # Heavy penalty for floor moves, but less penalty if taking fewer tiles
        base_penalty = -50.0
        tile_penalty = -10.0 * tiles_taken
        return base_penalty + tile_penalty

    def _evaluate_pattern_line_move(
        self,
        game_state: GameState,
        action: Action,
        player_board: PlayerBoard,
        tiles_taken: int,
    ) -> float:
        """
        Evaluate a move to a pattern line.
        """
        line_index = action.destination
        current_line = player_board.pattern_lines[line_index]
        line_capacity = line_index + 1

        # Count current tiles in the line
        current_tiles = len(current_line.tiles)

        # Check if line already has different colored tiles
        if current_tiles > 0:
            existing_color = current_line.color
            if existing_color != action.color:
                return -1000.0  # Invalid move

        # Check if we already have this color on the wall
        if not player_board.wall.can_place_tile(line_index, action.color):
            return -1000.0  # Can't place on wall

        score = 0.0

        # Calculate how many tiles can fit
        tiles_that_fit = min(tiles_taken, line_capacity - current_tiles)
        overflow_tiles = tiles_taken - tiles_that_fit

        # Bonus for filling lines completely
        if current_tiles + tiles_that_fit == line_capacity:
            score += 100.0  # Completion bonus

            # Add expected scoring bonus for wall placement
            wall_score = self._calculate_wall_scoring(
                player_board, line_index, action.color
            )
            score += wall_score * 2.0  # Weight wall scoring highly

        # Bonus for progress toward completion
        progress_bonus = (tiles_that_fit / line_capacity) * 20.0
        score += progress_bonus

        # Penalty for overflow to floor
        if overflow_tiles > 0:
            score -= overflow_tiles * 15.0

        # Prefer taking more tiles when it's efficient
        if tiles_that_fit == tiles_taken:  # No overflow
            score += tiles_taken * 5.0

        # Prefer shorter lines early in the game (easier to complete)
        if game_state.round_number <= 3:
            score += (5 - line_index) * 3.0

        return score

    def _calculate_wall_scoring(
        self, player_board: PlayerBoard, line_index: int, color: TileColor
    ) -> int:
        """
        Calculate the expected scoring from placing a tile on the wall.
        """
        wall = player_board.wall

        # Find the column for this color in this row using wall pattern
        col = None
        for c, wall_color in enumerate(wall.WALL_PATTERN[line_index]):
            if wall_color == color:
                col = c
                break

        if col is None:
            return 0

        # Count horizontal connections
        horizontal_score = 1  # The tile itself

        # Check left
        for c in range(col - 1, -1, -1):
            if wall.filled[line_index][c]:
                horizontal_score += 1
            else:
                break

        # Check right
        for c in range(col + 1, 5):
            if wall.filled[line_index][c]:
                horizontal_score += 1
            else:
                break

        # Count vertical connections
        vertical_score = 1  # The tile itself

        # Check up
        for r in range(line_index - 1, -1, -1):
            if wall.filled[r][col]:
                vertical_score += 1
            else:
                break

        # Check down
        for r in range(line_index + 1, 5):
            if wall.filled[r][col]:
                vertical_score += 1
            else:
                break

        # Return the scoring logic from Azul rules
        if horizontal_score > 1 and vertical_score > 1:
            return horizontal_score + vertical_score
        else:
            return max(horizontal_score, vertical_score)

    def _evaluate_strategic_factors(
        self, game_state: GameState, action: Action, player_board: PlayerBoard
    ) -> float:
        """
        Evaluate strategic factors like opponent blocking and tile availability.
        """
        score = 0.0

        # Prefer taking from factories over center (gives first player token)
        if action.source >= 0:  # Factory
            score += 5.0
        else:  # Center
            # But taking from center gives first player token
            if game_state.factory_area.center.has_first_player_marker:
                score += 10.0  # First player advantage

        # Consider tile scarcity
        color_count = self._count_color_availability(game_state, action.color)
        if color_count <= 3:  # Scarce color
            score += 15.0

        # Look ahead: prefer colors that might be useful later
        if self._is_color_useful_later(player_board, action.color):
            score += 8.0

        # Defensive play: if taking fewer tiles, deny them to opponents
        tiles_taken = self._count_tiles_taken(game_state, action)
        if tiles_taken >= 3:
            score += 5.0  # Denying tiles to opponents

        return score

    def _count_color_availability(self, game_state: GameState, color: TileColor) -> int:
        """Count how many tiles of this color are still available."""
        count = 0

        # Count in factories
        for factory in game_state.factory_area.factories:
            count += sum(1 for tile in factory.tiles if tile.color == color)

        # Count in center
        count += sum(
            1 for tile in game_state.factory_area.center.tiles if tile.color == color
        )

        return count

    def _is_color_useful_later(
        self, player_board: PlayerBoard, color: TileColor
    ) -> bool:
        """
        Check if this color could be useful in future rounds.
        """
        # Check if we can still place this color anywhere on the wall
        for row in range(5):
            if player_board.wall.can_place_tile(row, color):
                return True

        return False

    def _get_algorithm_name(self) -> str:
        """Get the algorithm name for this agent."""
        return "Heuristic-based"

    def get_stats(self) -> dict:
        """Get runtime performance statistics."""
        base_stats = super().get_stats()
        # No additional runtime stats for heuristic agent beyond base class
        return base_stats

    def get_info(self) -> dict:
        """Get static agent metadata."""
        base_info = super().get_info()
        base_info.update(
            {
                "algorithm": "Heuristic-based",
                "features": "Pattern completion, Wall scoring, Strategic play",
                "description": "Rule-based agent using strategic heuristics",
            }
        )
        return base_info

    def reset_stats(self):
        """Reset statistics."""
        self.nodes_evaluated = 0


def create_heuristic_agent(player_id: int = 0) -> HeuristicAgent:
    """
    Factory function to create a heuristic agent.

    Args:
        player_id: The player ID for this agent

    Returns:
        Configured heuristic agent
    """
    return HeuristicAgent(player_id)
