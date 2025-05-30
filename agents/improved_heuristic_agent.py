"""
Improved Heuristic-based Azul AI Agent

This agent implements advanced strategic guidelines for better gameplay:
1. Prioritize top rows (first 3 rows) for quick completion
2. Maximize adjacency bonuses for wall placement
3. Focus on column completion for bonus points
4. Value the first player token strategically
5. Monitor and block opponents when beneficial
6. Be cautious with color bonuses (full color sets)
7. Avoid unfinished rows late in game (especially row 5)
8. Strategic discarding to force opponent penalties
"""

import random
from typing import List, Optional, Tuple

from game.game_state import Action, GameState, PlayerBoard, TileColor


class ImprovedHeuristicAgent:
    """
    An improved heuristic-based agent for Azul that implements
    advanced strategic principles for competitive gameplay.
    """

    def __init__(self, player_id: int = 0):
        """
        Initialize the improved heuristic agent.

        Args:
            player_id: The player ID this agent controls
        """
        self.player_id = player_id
        self.nodes_evaluated = 0

    def select_action(self, game_state: GameState) -> Action:
        """
        Select the best action based on improved heuristic evaluation.

        Args:
            game_state: Current game state

        Returns:
            Best action according to improved heuristics
        """
        self.nodes_evaluated = 0

        # Dynamically determine our player ID from the current game state
        # This allows the agent to work correctly even when position swapping is enabled
        current_player_id = game_state.current_player

        available_actions = game_state.get_legal_actions()

        if not available_actions:
            if game_state.game_over:
                raise ValueError("No legal actions available - game is over")
            else:
                raise ValueError(
                    "No legal actions available - possible game state inconsistency"
                )

        if len(available_actions) == 1:
            self.nodes_evaluated = 1
            return available_actions[0]

        # Evaluate all possible actions with improved heuristics
        best_action = available_actions[0]
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
        Evaluate the quality of an action using improved heuristics.

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
            score += self._evaluate_floor_move(game_state, action, tiles_taken)
        else:  # Pattern line
            score += self._evaluate_pattern_line_move(
                game_state, action, player_board, tiles_taken
            )

        # Add strategic considerations based on improved guidelines
        score += self._evaluate_strategic_factors(game_state, action, player_board)

        # Add opponent blocking considerations
        score += self._evaluate_opponent_blocking(game_state, action, player_id)

        # Add first player token evaluation
        score += self._evaluate_first_player_token(game_state, action)

        # Add small random factor to break ties
        score += random.uniform(-0.05, 0.05)

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

    def _evaluate_floor_move(
        self, game_state: GameState, action: Action, tiles_taken: int
    ) -> float:
        """
        Evaluate a move to the floor line with strategic discarding in mind.
        """
        # Base penalty for floor moves
        base_penalty = -40.0
        tile_penalty = -8.0 * tiles_taken

        # Strategic discarding: sometimes beneficial to force opponents to take penalties
        strategic_bonus = 0.0

        # If this forces opponents into bad positions (denying good options)
        remaining_options = self._count_remaining_good_options(game_state, action)
        if remaining_options < 2:  # Forces opponents into limited choices
            strategic_bonus += 15.0

        # Late game: prefer floor moves less as penalties are more costly
        if game_state.round_number >= 4:
            base_penalty -= 10.0

        return base_penalty + tile_penalty + strategic_bonus

    def _evaluate_pattern_line_move(
        self,
        game_state: GameState,
        action: Action,
        player_board: PlayerBoard,
        tiles_taken: int,
    ) -> float:
        """
        Evaluate a move to a pattern line with improved strategic considerations.
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

        # Guideline 1: Prioritize top rows (0, 1, 2) for quick completion
        top_row_bonus = 0.0
        if line_index <= 2:
            top_row_bonus = (3 - line_index) * 15.0  # 45, 30, 15 for rows 0, 1, 2
        score += top_row_bonus

        # Bonus for filling lines completely
        if current_tiles + tiles_that_fit == line_capacity:
            completion_bonus = 80.0

            # Extra bonus for completing top rows
            if line_index <= 2:
                completion_bonus += 20.0

            score += completion_bonus

            # Guideline 2: Maximize adjacency bonuses for wall placement
            wall_score = self._calculate_wall_scoring_with_adjacency(
                player_board, line_index, action.color
            )
            score += wall_score * 3.0  # Higher weight for adjacency

            # Guideline 3: Focus on column completion
            column_completion_bonus = self._evaluate_column_completion(
                player_board, line_index, action.color
            )
            score += column_completion_bonus

        # Bonus for progress toward completion (but prioritize actual completion)
        progress_bonus = (tiles_that_fit / line_capacity) * 10.0
        score += progress_bonus

        # Heavy penalty for overflow to floor
        if overflow_tiles > 0:
            overflow_penalty = overflow_tiles * 20.0
            score -= overflow_penalty

        # Prefer taking more tiles when it's efficient
        if tiles_that_fit == tiles_taken:  # No overflow
            efficiency_bonus = min(tiles_taken * 3.0, 15.0)  # Cap the bonus
            score += efficiency_bonus

        # Guideline 7: Avoid unfinished rows late in the game, especially row 5
        if game_state.round_number >= 4:
            if line_index == 4:  # Row 5 (index 4)
                # Heavy penalty for starting row 5 late in game
                if current_tiles == 0:
                    score -= 50.0
                # Also penalty for partial row 5 that won't complete
                elif current_tiles + tiles_that_fit < line_capacity:
                    score -= 25.0

            # General penalty for starting new rows late in game
            elif current_tiles == 0 and line_index >= 3:
                score -= 15.0

        # Guideline 6: Be cautious with color bonuses - don't overcommit
        color_commitment_penalty = self._evaluate_color_commitment_risk(
            game_state, player_board, action.color
        )
        score += color_commitment_penalty

        return score

    def _calculate_wall_scoring_with_adjacency(
        self, player_board: PlayerBoard, line_index: int, color: TileColor
    ) -> int:
        """
        Calculate wall scoring with emphasis on adjacency bonuses.
        """
        wall = player_board.wall

        # Find the column for this color in this row
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

        # Enhanced scoring for adjacency
        base_score = 0
        if horizontal_score > 1 and vertical_score > 1:
            base_score = horizontal_score + vertical_score
        else:
            base_score = max(horizontal_score, vertical_score)

        # Bonus for creating multiple connections
        if horizontal_score > 2 or vertical_score > 2:
            base_score += 5  # Bonus for longer chains

        return base_score

    def _evaluate_column_completion(
        self, player_board: PlayerBoard, line_index: int, color: TileColor
    ) -> float:
        """
        Evaluate potential for column completion (Guideline 3).
        """
        wall = player_board.wall

        # Find the column for this color
        col = None
        for c, wall_color in enumerate(wall.WALL_PATTERN[line_index]):
            if wall_color == color:
                col = c
                break

        if col is None:
            return 0.0

        # Count how many tiles are already in this column
        filled_in_column = sum(1 for r in range(5) if wall.filled[r][col])

        # Bonus based on column completion potential
        column_bonus = 0.0

        # Strong bonus for completing columns (especially central ones)
        if filled_in_column == 4:  # Would complete the column
            column_bonus = 60.0
            # Extra bonus for central columns (more flexible)
            if 1 <= col <= 3:
                column_bonus += 15.0

        elif filled_in_column >= 2:  # Good progress toward completion
            column_bonus = filled_in_column * 8.0
            # Bonus for central columns
            if 1 <= col <= 3:
                column_bonus += 5.0

        return column_bonus

    def _evaluate_color_commitment_risk(
        self, game_state: GameState, player_board: PlayerBoard, color: TileColor
    ) -> float:
        """
        Evaluate risk of over-committing to color completion (Guideline 6).
        """
        # Count how many of this color we can still place
        placeable_positions = 0
        for row in range(5):
            if player_board.wall.can_place_tile(row, color):
                placeable_positions += 1

        # Count how many of this color are available in the game
        available_tiles = self._count_color_availability(game_state, color)

        # Risk penalty if we're chasing a color that's hard to complete
        risk_penalty = 0.0

        # High risk: few positions left but committing heavily
        if placeable_positions <= 2 and available_tiles >= 4:
            risk_penalty = -10.0

        # Very high risk: only one position left
        if placeable_positions == 1:
            risk_penalty = -15.0

        return risk_penalty

    def _evaluate_strategic_factors(
        self, game_state: GameState, action: Action, player_board: PlayerBoard
    ) -> float:
        """
        Evaluate strategic factors with improved considerations.
        Simplified to avoid performance issues.
        """
        score = 0.0

        # Tile scarcity considerations (simplified)
        color_count = self._count_color_availability(game_state, action.color)
        if color_count <= 2:  # Very scarce
            score += 15.0
        elif color_count <= 4:  # Somewhat scarce
            score += 8.0

        # Look ahead: prefer colors that are useful for multiple rows (simplified)
        if action.destination != -1:
            # Quick check if color can be placed on wall
            if player_board.wall.can_place_tile(action.destination, action.color):
                score += 5.0

                # Bonus for central columns (columns 1-3)
                for c, wall_color in enumerate(
                    player_board.wall.WALL_PATTERN[action.destination]
                ):
                    if wall_color == action.color and 1 <= c <= 3:
                        score += 6.0
                        break

        return score

    def _evaluate_opponent_blocking(
        self, game_state: GameState, action: Action, player_id: int
    ) -> float:
        """
        Evaluate opponent blocking opportunities (Guideline 5).
        Simplified to avoid performance issues.
        """
        # Early return if no opponents to block
        if len(game_state.players) <= 1:
            return 0.0

        blocking_bonus = 0.0

        # Check what this action denies to opponents
        tiles_denied = self._count_tiles_taken(game_state, action)

        # Simple heuristic: if denying 3+ tiles, give bonus
        if tiles_denied >= 3:
            blocking_bonus = tiles_denied * 3.0
        elif tiles_denied >= 2:
            blocking_bonus = tiles_denied * 1.5

        return blocking_bonus

    def _evaluate_first_player_token(
        self, game_state: GameState, action: Action
    ) -> float:
        """
        Evaluate the value of taking the first player token (Guideline 4).
        """
        if action.source != -1:  # Not taking from center
            return 0.0

        if not game_state.factory_area.center.has_first_player_marker:
            return 0.0

        # Value first player token more in early rounds
        token_value = 0.0
        if game_state.round_number <= 2:
            token_value = 25.0  # Very valuable early
        elif game_state.round_number <= 4:
            token_value = 15.0  # Still valuable mid-game
        else:
            token_value = 8.0  # Less valuable late game

        return token_value

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

    def _count_useful_rows(self, player_board: PlayerBoard, color: TileColor) -> int:
        """Count how many rows this color could be useful for."""
        useful_count = 0
        for row in range(5):
            if player_board.wall.can_place_tile(row, color):
                # Check if pattern line is empty or has same color
                pattern_line = player_board.pattern_lines[row]
                if not pattern_line.tiles or pattern_line.color == color:
                    useful_count += 1
        return useful_count

    def _count_remaining_good_options(
        self, game_state: GameState, action: Action
    ) -> int:
        """Count how many good options remain after this action."""
        # Simplified to avoid complex simulations
        remaining_moves = 0

        # Count other factory options
        for source_id in range(len(game_state.factory_area.factories)):
            if source_id != action.source:
                factory = game_state.factory_area.factories[source_id]
                if factory.tiles:  # Has tiles
                    remaining_moves += 1

        # Count center if not the source of this action
        if action.source != -1 and game_state.factory_area.center.tiles:
            remaining_moves += 1

        return remaining_moves

    def _assess_opponent_need(
        self, opponent_board: PlayerBoard, color: TileColor
    ) -> float:
        """Assess how much an opponent needs a specific color (0.0 to 1.0)."""
        # Simplified version to avoid performance issues
        need_score = 0.0

        # Quick check: if opponent has partial lines with this color
        for pattern_line in opponent_board.pattern_lines:
            if pattern_line.color == color and pattern_line.tiles:
                need_score += 0.3  # Simple bonus

        return min(need_score, 1.0)

    def _would_improve_turn_order(self, game_state: GameState) -> bool:
        """Check if getting first player token would improve our position."""
        # Simple heuristic: beneficial if we're not currently first
        return self.player_id != 0  # Assuming player 0 is usually first

    def get_stats(self) -> dict:
        """Get statistics about the agent's last decision."""
        return {
            "nodesEvaluated": self.nodes_evaluated,
            "algorithm": "Improved Heuristic-based",
            "features": "Top row priority, Adjacency maximization, Column focus, Strategic blocking",
        }

    def get_info(self) -> dict:
        """Get agent information for evaluation metadata."""
        return {
            "agent_type": self.__class__.__name__,
            "algorithm": "Improved Heuristic-based",
            "features": "Advanced strategic guidelines implementation",
            "player_id": self.player_id,
        }

    def reset_stats(self):
        """Reset statistics."""
        self.nodes_evaluated = 0


def create_improved_heuristic_agent(player_id: int = 0) -> ImprovedHeuristicAgent:
    """
    Factory function to create an improved heuristic agent.

    Args:
        player_id: The player ID for this agent

    Returns:
        Configured improved heuristic agent
    """
    return ImprovedHeuristicAgent(player_id)
