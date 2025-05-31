"""
Minimax AI Agent with Alpha-Beta Pruning for Azul

This agent uses the minimax algorithm with alpha-beta pruning to look ahead
and choose optimal moves. Based on the implementation described at:
https://domwil.co.uk/posts/azul-ai/

Key features:
- Iterative deepening with time limits
- Alpha-beta pruning for performance
- Move ordering using previous search results
- Heuristic evaluation function for intermediate game states
- Configurable difficulty settings
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from game.game_state import Action, GameState


@dataclass
class MinimaxConfig:
    """Configuration class for MinimaxAgent difficulty settings."""

    # Time management
    time_limit: float = 1.0  # Maximum time per move in seconds

    # Search depth settings
    max_depth: Optional[int] = None  # Maximum search depth (None for adaptive)
    min_depth: int = 1  # Minimum search depth for iterative deepening
    adaptive_depth: bool = True  # Use adaptive depth based on game state

    # Node limits
    max_nodes: Optional[int] = None  # Maximum nodes to evaluate (None for adaptive)
    adaptive_node_limit: bool = True  # Adapt node limit based on time limit

    # Algorithm features
    enable_iterative_deepening: bool = True  # Use iterative deepening
    enable_alpha_beta_pruning: bool = True  # Use alpha-beta pruning
    enable_move_ordering: bool = True  # Use move ordering for better pruning

    # Performance tuning
    time_buffer: float = 0.9  # Use 90% of time limit to leave buffer
    depth_time_threshold: float = (
        0.1  # If depth takes less than this ratio of time, continue deeper
    )

    @classmethod
    def create_difficulty_preset(cls, difficulty: str) -> "MinimaxConfig":
        """Create a configuration preset for different difficulty levels."""
        difficulty = difficulty.lower()

        if difficulty == "easy":
            return cls(
                time_limit=0.3,
                max_depth=2,
                adaptive_depth=False,
                max_nodes=500,
                adaptive_node_limit=False,
                enable_move_ordering=False,
            )
        elif difficulty == "medium":
            return cls(
                time_limit=0.7,
                max_depth=4,
                adaptive_depth=True,
                max_nodes=2000,
                adaptive_node_limit=True,
                enable_move_ordering=True,
            )
        elif difficulty == "hard":
            return cls(
                time_limit=1.5,
                max_depth=6,
                adaptive_depth=True,
                max_nodes=5000,
                adaptive_node_limit=True,
                enable_move_ordering=True,
            )
        elif difficulty == "expert":
            return cls(
                time_limit=3.0,
                max_depth=8,
                adaptive_depth=True,
                max_nodes=10000,
                adaptive_node_limit=True,
                enable_move_ordering=True,
            )
        elif difficulty == "custom":
            # Return default config for custom tuning
            return cls()
        else:
            raise ValueError(
                f"Unknown difficulty preset: {difficulty}. Available: easy, medium, hard, expert, custom"
            )


class MinimaxAgent:
    """
    A minimax agent with alpha-beta pruning for Azul.
    """

    def __init__(
        self, player_id: int = 0, config: Optional[MinimaxConfig] = None, **kwargs
    ):
        """
        Initialize the minimax agent.

        Args:
            player_id: The player ID this agent controls
            config: MinimaxConfig object with all settings
            **kwargs: Legacy support - individual config parameters
        """
        self.player_id = player_id

        # Handle legacy initialization and config merging
        if config is None:
            # Create config from kwargs for backward compatibility
            config = MinimaxConfig()
            if "time_limit" in kwargs:
                config.time_limit = kwargs["time_limit"]
            if "max_depth" in kwargs:
                config.max_depth = kwargs["max_depth"]
                config.adaptive_depth = False
            if "max_nodes" in kwargs:
                config.max_nodes = kwargs["max_nodes"]
                config.adaptive_node_limit = False

        self.config = config

        # Legacy property for backward compatibility
        self.time_limit = self.config.time_limit

        # Performance tracking
        self.nodes_evaluated: int = 0
        self.max_depth_reached: int = 0
        self.move_scores: Dict[str, float] = {}  # For move ordering

    def select_action(self, game_state: GameState) -> Action:
        """
        Select the best action using minimax with iterative deepening.

        Args:
            game_state: Current game state

        Returns:
            Best action according to minimax search
        """
        self.nodes_evaluated = 0
        self.max_depth_reached = 0
        self.move_scores.clear()

        start_time = time.time()
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

        best_action = available_actions[0]

        # Determine search depth limits
        if self.config.adaptive_depth and self.config.max_depth is None:
            # Adaptive max depth based on game complexity
            max_depth = min(8, 15 - len(available_actions) // 10)
        elif self.config.max_depth is not None:
            max_depth = self.config.max_depth
        else:
            max_depth = 6  # Default fallback

        # Use iterative deepening or fixed depth
        if self.config.enable_iterative_deepening:
            start_depth = self.config.min_depth
            end_depth = max_depth + 1
        else:
            # Fixed depth search
            start_depth = max_depth
            end_depth = max_depth + 1

        # Iterative deepening loop
        for depth in range(start_depth, end_depth):
            if (
                time.time() - start_time
                >= self.config.time_limit * self.config.time_buffer
            ):
                break

            # Check node limit
            if self._should_stop_search(start_time):
                break

            try:
                # Sort moves based on previous iteration results for better alpha-beta pruning
                if self.config.enable_move_ordering:
                    sorted_actions = self._sort_moves(
                        available_actions, depth > self.config.min_depth
                    )
                else:
                    sorted_actions = available_actions

                current_best_action = None
                current_best_score = float("-inf")
                depth_start_time = time.time()

                for action in sorted_actions:
                    # Time check for this depth level
                    if (
                        time.time() - start_time
                        >= self.config.time_limit * self.config.time_buffer
                    ):
                        break

                    # Create a copy of the game state and apply the action
                    new_state = game_state.copy()
                    if not new_state.apply_action(action, skip_validation=True):
                        continue

                    # Evaluate this move with minimax
                    if self.config.enable_alpha_beta_pruning:
                        score = self._minimax(
                            new_state,
                            depth - 1,
                            float("-inf"),
                            float("inf"),
                            current_player_id,
                            start_time,
                        )
                    else:
                        score = self._minimax_no_pruning(
                            new_state,
                            depth - 1,
                            current_player_id,
                            start_time,
                        )

                    # Store score for move ordering in next iteration
                    if self.config.enable_move_ordering:
                        self.move_scores[self._action_key(action)] = score

                    if score > current_best_score:
                        current_best_score = score
                        current_best_action = action

                # Update best result if this depth completed successfully
                if current_best_action is not None:
                    best_action = current_best_action
                    self.max_depth_reached = depth

                    # Check if we should continue to deeper search
                    if self.config.enable_iterative_deepening:
                        depth_time = time.time() - depth_start_time
                        if (
                            depth_time
                            < self.config.time_limit * self.config.depth_time_threshold
                            and depth < 3
                        ):
                            continue
                else:
                    # If we couldn't complete this depth, stop
                    break

            except TimeoutError:
                # Time expired during this depth, use previous result
                break

        return best_action

    def _should_stop_search(self, start_time: float) -> bool:
        """Check if search should be stopped due to time or node limits."""
        # Time limit check
        if time.time() - start_time >= self.config.time_limit * self.config.time_buffer:
            return True

        # Node limit check
        max_nodes = self._get_effective_node_limit()
        if max_nodes is not None and self.nodes_evaluated > max_nodes:
            return True

        return False

    def _get_effective_node_limit(self) -> Optional[int]:
        """Get the effective node limit based on configuration."""
        if self.config.max_nodes is not None:
            return self.config.max_nodes
        elif self.config.adaptive_node_limit:
            # Adaptive node limit based on time limit
            return min(10000, int(self.config.time_limit * 5000))
        else:
            return None

    def _minimax(
        self,
        game_state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        original_player_id: int,
        start_time: float,
    ) -> float:
        """
        Minimax algorithm with alpha-beta pruning.

        Args:
            game_state: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            original_player_id: The player we're evaluating for
            start_time: Start time for timeout checking

        Returns:
            Evaluated score for this position
        """
        # Check timeout
        if self._should_stop_search(start_time):
            raise TimeoutError("Search limits exceeded")

        self.nodes_evaluated += 1

        # Terminal conditions
        if depth == 0 or game_state.game_over:
            return self._evaluate_position(game_state, original_player_id)

        legal_actions = game_state.get_legal_actions()

        # Check if round is over (no legal actions but game continues)
        if not legal_actions:
            return self._evaluate_position(game_state, original_player_id)

        # Determine if current player is maximizing (our player) or minimizing (opponent)
        maximizing_player = game_state.current_player == original_player_id

        if maximizing_player:
            max_eval = float("-inf")
            for action in legal_actions:
                new_state = game_state.copy()
                if not new_state.apply_action(action):
                    continue

                eval_score = self._minimax(
                    new_state, depth - 1, alpha, beta, original_player_id, start_time
                )
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    break  # Alpha-beta pruning
            return max_eval
        else:
            min_eval = float("inf")
            for action in legal_actions:
                new_state = game_state.copy()
                if not new_state.apply_action(action):
                    continue

                eval_score = self._minimax(
                    new_state, depth - 1, alpha, beta, original_player_id, start_time
                )
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    break  # Alpha-beta pruning
            return min_eval

    def _minimax_no_pruning(
        self,
        game_state: GameState,
        depth: int,
        original_player_id: int,
        start_time: float,
    ) -> float:
        """
        Minimax algorithm without alpha-beta pruning (for educational/testing purposes).

        Args:
            game_state: Current game state
            depth: Remaining search depth
            original_player_id: The player we're evaluating for
            start_time: Start time for timeout checking

        Returns:
            Evaluated score for this position
        """
        # Check timeout
        if self._should_stop_search(start_time):
            raise TimeoutError("Search limits exceeded")

        self.nodes_evaluated += 1

        # Terminal conditions
        if depth == 0 or game_state.game_over:
            return self._evaluate_position(game_state, original_player_id)

        legal_actions = game_state.get_legal_actions()

        # Check if round is over (no legal actions but game continues)
        if not legal_actions:
            return self._evaluate_position(game_state, original_player_id)

        # Determine if current player is maximizing (our player) or minimizing (opponent)
        maximizing_player = game_state.current_player == original_player_id

        if maximizing_player:
            max_eval = float("-inf")
            for action in legal_actions:
                new_state = game_state.copy()
                if not new_state.apply_action(action):
                    continue

                eval_score = self._minimax_no_pruning(
                    new_state, depth - 1, original_player_id, start_time
                )
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float("inf")
            for action in legal_actions:
                new_state = game_state.copy()
                if not new_state.apply_action(action):
                    continue

                eval_score = self._minimax_no_pruning(
                    new_state, depth - 1, original_player_id, start_time
                )
                min_eval = min(min_eval, eval_score)
            return min_eval

    def _evaluate_position(self, game_state: GameState, player_id: int) -> float:
        """
        Evaluate the current position from the perspective of player_id.
        Based on the heuristic described in the blog post - simulate round ending immediately.

        Args:
            game_state: Game state to evaluate
            player_id: Player to evaluate for

        Returns:
            Score (positive = good for player, negative = bad)
        """
        if game_state.game_over:
            scores = game_state.get_scores()
            our_score = scores[player_id]
            best_opponent_score: float = max(
                scores[i] for i in range(len(scores)) if i != player_id
            )
            return our_score - best_opponent_score

        # Calculate the value if the round ended immediately (as in blog post)
        our_player = game_state.players[player_id]
        our_projected_score = self._calculate_round_end_score(our_player)

        # Calculate best opponent projected score
        best_opponent_projected_score: float = float("-inf")
        for i, player in enumerate(game_state.players):
            if i != player_id:
                opponent_score = self._calculate_round_end_score(player)
                best_opponent_projected_score = max(
                    best_opponent_projected_score, opponent_score
                )

        return our_projected_score - best_opponent_projected_score

    def _calculate_round_end_score(self, player_board) -> float:
        """
        Calculate the score a player would have if the round ended immediately.
        This simulates moving all completed pattern lines to the wall.
        Optimized version for performance.

        Args:
            player_board: Player board to evaluate

        Returns:
            Projected total score
        """
        current_score = player_board.score
        projected_additional_score = 0.0

        # Quick check each pattern line for completion
        for line_idx, pattern_line in enumerate(player_board.pattern_lines):
            line_capacity = line_idx + 1

            if len(pattern_line.tiles) == line_capacity and pattern_line.tiles:
                # This line is complete, simulate placing on wall
                color = pattern_line.color

                # Check if we can place this tile (should be true if line is valid)
                if player_board.wall.can_place_tile(line_idx, color):
                    # Estimate wall scoring without full simulation
                    # This is a simplified version that's much faster
                    base_score = 1

                    # Quick bonus for likely connections
                    # Check if adjacent positions are filled for horizontal connections
                    wall_row = player_board.wall.filled[line_idx]
                    filled_count = sum(1 for filled in wall_row if filled)
                    if filled_count > 0:
                        base_score += min(
                            filled_count, 3
                        )  # Approximate connection bonus

                    # Small bonus for completing lower lines (easier)
                    if line_idx < 2:
                        base_score += 1

                    projected_additional_score += base_score

        # Subtract floor line penalties
        floor_penalty = len(player_board.floor_line)
        if floor_penalty > 0:
            # Use actual Azul penalty structure
            penalties = [1, 1, 2, 2, 2, 3, 3]
            total_penalty = sum(
                penalties[i] for i in range(min(floor_penalty, len(penalties)))
            )
            projected_additional_score -= total_penalty

        return current_score + projected_additional_score

    def _simulate_wall_scoring(
        self, wall_filled: List[List[bool]], row: int, col: int
    ) -> int:
        """
        Simulate the scoring for placing a tile on the wall.

        Args:
            wall_filled: 2D array representing wall filled state
            row: Row to place tile
            col: Column to place tile

        Returns:
            Points earned from this placement
        """
        score = 1  # Base score for the tile

        # Check horizontal connections
        horizontal_length = 1
        # Check left
        for c in range(col - 1, -1, -1):
            if wall_filled[row][c]:
                horizontal_length += 1
            else:
                break
        # Check right
        for c in range(col + 1, 5):
            if wall_filled[row][c]:
                horizontal_length += 1
            else:
                break

        # Check vertical connections
        vertical_length = 1
        # Check up
        for r in range(row - 1, -1, -1):
            if wall_filled[r][col]:
                vertical_length += 1
            else:
                break
        # Check down
        for r in range(row + 1, 5):
            if wall_filled[r][col]:
                vertical_length += 1
            else:
                break

        # If connected to other tiles, use the larger connection
        if horizontal_length > 1 or vertical_length > 1:
            score = max(horizontal_length, vertical_length)
            if horizontal_length > 1 and vertical_length > 1:
                score = horizontal_length + vertical_length

        return score

    def _sort_moves(
        self, actions: List[Action], use_previous_scores: bool
    ) -> List[Action]:
        """
        Sort moves for better alpha-beta pruning performance.

        Args:
            actions: Available actions to sort
            use_previous_scores: Whether to use scores from previous iteration

        Returns:
            Sorted list of actions (best first)
        """
        if use_previous_scores and self.move_scores:
            # Sort by previous scores (highest first)
            def get_score(action):
                key = self._action_key(action)
                return self.move_scores.get(key, 0.0)

            return sorted(actions, key=get_score, reverse=True)
        else:
            # Use heuristic ordering when no previous scores available
            def heuristic_score(action):
                score = 0.0

                # Prefer completing pattern lines
                if action.destination >= 0:  # Not floor line
                    score += 10.0

                    # Prefer shorter lines (easier to complete)
                    score += (5 - action.destination) * 2.0

                    # Prefer taking more tiles efficiently
                    if action.source == -1:  # Center
                        score += 3.0  # Center often has more tiles
                else:
                    # Floor line moves are generally bad, but sometimes necessary
                    score -= 5.0

                return score

            return sorted(actions, key=heuristic_score, reverse=True)

    def _action_key(self, action: Action) -> str:
        """Create a string key for an action for move ordering."""
        return f"{action.source}_{action.color.value}_{action.destination}"

    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            "nodes_evaluated": self.nodes_evaluated,
            "max_depth_reached": self.max_depth_reached,
            "time_limit": self.config.time_limit,
            "max_depth_configured": self.config.max_depth,
            "adaptive_depth": self.config.adaptive_depth,
            "iterative_deepening": self.config.enable_iterative_deepening,
            "alpha_beta_pruning": self.config.enable_alpha_beta_pruning,
            "move_ordering": self.config.enable_move_ordering,
            "max_nodes_configured": self.config.max_nodes,
        }

    def get_info(self) -> dict:
        """Get agent information."""
        return {
            "name": "MinimaxAgent",
            "type": "minimax_alpha_beta",
            "description": "Minimax agent with alpha-beta pruning and iterative deepening",
            "config": {
                "time_limit": self.config.time_limit,
                "max_depth": self.config.max_depth,
                "adaptive_depth": self.config.adaptive_depth,
                "iterative_deepening": self.config.enable_iterative_deepening,
                "alpha_beta_pruning": self.config.enable_alpha_beta_pruning,
                "move_ordering": self.config.enable_move_ordering,
                "max_nodes": self.config.max_nodes,
            },
        }

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Update legacy property
        self.time_limit = self.config.time_limit

    def set_difficulty_preset(self, difficulty: str) -> None:
        """Set configuration to a difficulty preset."""
        self.config = MinimaxConfig.create_difficulty_preset(difficulty)
        self.time_limit = self.config.time_limit

    def reset_stats(self):
        """Reset performance statistics."""
        self.nodes_evaluated = 0
        self.max_depth_reached = 0
        self.move_scores.clear()


def create_minimax_agent(
    player_id: int = 0,
    difficulty: Optional[str] = None,
    config: Optional[MinimaxConfig] = None,
    **kwargs,
) -> MinimaxAgent:
    """
    Create a MinimaxAgent with specified parameters.

    Args:
        player_id: Player ID for the agent
        difficulty: Difficulty preset ("easy", "medium", "hard", "expert", "custom")
        config: MinimaxConfig object with all settings
        **kwargs: Individual config parameters (for legacy support)

    Returns:
        Configured MinimaxAgent instance
    """
    if difficulty is not None:
        config = MinimaxConfig.create_difficulty_preset(difficulty)
    elif config is None:
        config = MinimaxConfig(**kwargs)

    return MinimaxAgent(player_id=player_id, config=config)
