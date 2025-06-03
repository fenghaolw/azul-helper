import random
from typing import TYPE_CHECKING, Dict, List, Optional, cast

if TYPE_CHECKING:
    from game.state_representation import AzulStateRepresentation

from game.factory import FactoryArea
from game.player_board import PlayerBoard
from game.tile import Tile, TileColor


class Action:
    """Represents a player action in the game."""

    def __init__(self, source: int, color: TileColor, destination: int):
        """
        Args:
            source: Factory index (-1 for center)
            color: Color of tiles to take
            destination: Pattern line index (0-4) or -1 for floor line
        """
        self.source = source
        self.color = color
        self.destination = destination
        # Precompute hash for better performance in MCTS tree operations
        self._hash = hash((source, color.value, destination))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Action):
            return False
        # Fast path: compare hash first (most likely to differ)
        if hasattr(other, "_hash") and self._hash != other._hash:
            return False
        return (
            self.source == other.source
            and self.color == other.color
            and self.destination == other.destination
        )

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        source_str = "center" if self.source == -1 else f"factory_{self.source}"
        dest_str = "floor" if self.destination == -1 else f"line_{self.destination}"
        return f"Action({source_str}, {self.color.value}, {dest_str})"


class GameState:
    """Represents the complete state of an Azul game."""

    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        if num_players < 2 or num_players > 4:
            raise ValueError("Number of players must be between 2 and 4")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        self.num_players = num_players
        self.current_player = 0
        self.game_over = False
        self.winner: Optional[int] = None
        self.round_number = 1

        # Initialize game components
        self.players = [PlayerBoard() for _ in range(num_players)]
        self.factory_area = FactoryArea(num_players)
        self.discard_pile: List[Tile] = []

        # Set up initial game state
        self._initialize_tiles()
        self._start_new_round()

        # Default first player to 0 for the first round
        # For the first round, it defaults to 0 from __init__

    def _initialize_tiles(self) -> None:
        """Initialize the bag with all tiles."""
        self.bag = Tile.create_standard_tiles()
        random.shuffle(self.bag)
        self.discard_pile = []

    def _start_new_round(self) -> None:
        """Start a new round by filling factories."""
        # If bag doesn't have enough tiles, refill from discard pile
        tiles_needed = self.factory_area.num_factories * 4
        if len(self.bag) < tiles_needed:
            self.bag.extend(self.discard_pile)
            self.discard_pile = []
            random.shuffle(self.bag)

        # Setup factory area
        self.factory_area.setup_round(self.bag)

        # Note: current_player is already set in _end_round for subsequent rounds
        # For the first round, it defaults to 0 from __init__

    def _get_tiles_from_source(
        self, source: int, color: TileColor
    ) -> Optional[List[Tile]]:
        """Helper method to get tiles from a source. Returns None if invalid."""
        if source == -1:
            # Taking from center - optimized tile filtering
            center_tiles = self.factory_area.center.tiles
            if not center_tiles:
                return None
            # Use list comprehension for better performance than generator + list()
            test_tiles = [tile for tile in center_tiles if tile.color == color]
            return test_tiles if test_tiles else None
        else:
            # Taking from factory - validate bounds first with early exit
            factories = self.factory_area.factories
            if source < 0 or source >= len(factories):
                return None

            factory_tiles = factories[source].tiles
            if not factory_tiles:
                return None

            # Use list comprehension for better performance
            test_tiles = [tile for tile in factory_tiles if tile.color == color]
            return test_tiles if test_tiles else None

    def get_legal_actions(self, player_id: Optional[int] = None) -> List[Action]:
        """Get all legal actions for the current player."""
        if player_id is None:
            player_id = self.current_player

        if self.game_over:
            return []

        # Validate player_id
        if player_id < 0 or player_id >= self.num_players:
            return []

        player = self.players[player_id]
        actions = []

        # Get available tile sources
        available_moves = self.factory_area.get_available_moves()

        # If no moves are available (all sources empty), this indicates
        # an unusual game state that should trigger game end
        if not available_moves:
            # Check if we're in a valid round-over state
            if self.factory_area.is_round_over():
                # This should trigger _end_round() in the next apply_action call
                # For now, return empty list which will cause the calling code to handle appropriately
                return []
            else:
                # This is an inconsistent state - we have no moves but round isn't over
                # This could happen if bag is exhausted mid-round
                # Force game to end gracefully
                self._end_game()
                return []

        # Pre-cache wall validation for all colors to avoid repeated lookups
        wall_cache: Dict[int, Dict[TileColor, bool]] = {}
        for line_idx in range(5):
            wall_cache[line_idx] = {}
            for color in [
                TileColor.BLUE,
                TileColor.YELLOW,
                TileColor.RED,
                TileColor.BLACK,
                TileColor.WHITE,
            ]:
                wall_cache[line_idx][color] = player.wall.can_place_tile(
                    line_idx, color
                )

        # Cache pattern line data to avoid repeated attribute access
        pattern_line_cache = []
        for i in range(5):
            pl = player.pattern_lines[i]
            pattern_line_cache.append(
                {
                    "capacity": pl.capacity,
                    "tiles_count": len(pl.tiles),
                    "color": pl.color,  # This can be None
                    "is_empty": not pl.tiles,
                }
            )

        # Cache tile source lookups
        tile_cache = {}
        for source, color in available_moves:
            if (source, color) not in tile_cache:
                tile_cache[(source, color)] = self._get_tiles_from_source(source, color)

        for source, color in available_moves:
            # Use cached tiles
            test_tiles = tile_cache[(source, color)]
            if not test_tiles:
                continue  # Skip if no tiles of this color available

            # Floor line is always valid
            actions.append(Action(source, color, -1))

            # Check pattern lines (0-4) with optimized validation using cached data
            tiles_count = len(test_tiles)

            for dest in range(5):
                # Fast wall check first (most likely to eliminate invalid moves)
                if not wall_cache[dest][color]:
                    continue

                # Use cached pattern line data
                pl_data = pattern_line_cache[dest]

                # Pattern line validation (fully inlined and optimized)
                if color == TileColor.FIRST_PLAYER:
                    continue

                # If line is empty, check capacity only
                if pl_data["is_empty"]:
                    # capacity is guaranteed to be int, tiles_count is guaranteed to be int
                    capacity = cast(int, pl_data["capacity"])
                    if tiles_count <= capacity:
                        actions.append(Action(source, color, dest))
                else:
                    # If line has tiles, check color and capacity
                    # Pattern line color is guaranteed to be non-None when not empty
                    pl_color = pl_data["color"]
                    tiles_count_cached = cast(int, pl_data["tiles_count"])
                    capacity_cached = cast(int, pl_data["capacity"])
                    if (
                        pl_color == color
                        and tiles_count_cached + tiles_count <= capacity_cached
                    ):
                        actions.append(Action(source, color, dest))

        return actions

    def is_action_legal(self, action: Action, player_id: Optional[int] = None) -> bool:
        """Check if an action is legal - optimized version."""
        if player_id is None:
            player_id = self.current_player

        if self.game_over:
            return False

        # Validate player_id
        if player_id < 0 or player_id >= self.num_players:
            return False

        # Fast validation for floor line (always valid if source/color valid)
        if action.destination == -1:
            # Check if source has the color directly without creating color sets
            if action.source == -1:
                return any(
                    tile.color == action.color
                    for tile in self.factory_area.center.tiles
                )
            else:
                if action.source < 0 or action.source >= len(
                    self.factory_area.factories
                ):
                    return False
                return any(
                    tile.color == action.color
                    for tile in self.factory_area.factories[action.source].tiles
                )

        # For pattern lines, we need the actual tiles to check capacity
        if 0 <= action.destination <= 4:
            test_tiles = self._get_tiles_from_source(action.source, action.color)
            if not test_tiles:
                return False
            return self.players[player_id].can_place_tiles_on_pattern_line(
                action.destination, test_tiles
            )

        # Invalid destination
        return False

    def apply_action(self, action: Action, skip_validation: bool = False) -> bool:
        """Apply an action and return True if successful.

        Args:
            action: The action to apply
            skip_validation: If True, skip is_action_legal check (use when action is already validated)
        """
        if not skip_validation and not self.is_action_legal(action):
            return False

        player = self.players[self.current_player]

        # Take tiles from source
        if action.source == -1:
            tiles = self.factory_area.take_from_center(action.color)
        else:
            tiles = self.factory_area.take_from_factory(action.source, action.color)

        # Place tiles on destination
        if action.destination == -1:
            discarded = player.place_tiles_on_floor_line(tiles)
        else:
            discarded = player.place_tiles_on_pattern_line(action.destination, tiles)

        # Track discarded tiles (all go to discard pile per official rules)
        self.discard_pile.extend(discarded)

        # Check if round is over
        if self.factory_area.is_round_over():
            self._end_round()
        else:
            self._next_player()

        return True

    def _next_player(self) -> None:
        """Move to next player."""
        self.current_player = (self.current_player + 1) % self.num_players

    def _end_round(self) -> None:
        """End the current round and perform scoring."""
        # Determine who will be first player next round (before clearing floor lines)
        next_first_player = self.current_player  # Default to current player
        for i, player in enumerate(self.players):
            if player.has_first_player_marker():
                next_first_player = i
                break

        # Score all players
        for player in self.players:
            points, discard_tiles = player.end_round_scoring()
            self.discard_pile.extend(discard_tiles)

        # Check for game end condition (any player has completed a row)
        game_should_end = False
        for player in self.players:
            if player.wall.get_completed_rows():
                game_should_end = True
                break

        if game_should_end:
            self._end_game()
        else:
            self.round_number += 1
            self.current_player = (
                next_first_player  # Set the first player for next round
            )
            self._start_new_round()

    def _end_game(self) -> None:
        """End the game and determine winner."""
        # Final scoring
        for player in self.players:
            player.final_scoring()

        # Determine winner (highest score, ties broken by completed rows)
        max_score = max(player.score for player in self.players)
        winners = [
            i for i, player in enumerate(self.players) if player.score == max_score
        ]

        if len(winners) == 1:
            self.winner = winners[0]
        else:
            # Tiebreaker: most completed rows
            max_rows = max(
                len(self.players[i].wall.get_completed_rows()) for i in winners
            )
            final_winners = [
                i
                for i in winners
                if len(self.players[i].wall.get_completed_rows()) == max_rows
            ]
            self.winner = final_winners[0]  # If still tied, first player wins

        self.game_over = True

    def get_scores(self) -> List[int]:
        """Get current scores for all players."""
        return [player.score for player in self.players]

    def get_state_vector(self) -> List[float]:
        """Get a numerical representation of the game state for ML models."""
        # Import here to avoid circular imports
        from game.state_representation import AzulStateRepresentation

        # Create numerical representation and return flattened vector
        state_repr = AzulStateRepresentation(self)
        return state_repr.get_flat_state_vector(normalize=True).tolist()

    def get_numerical_state(self) -> "AzulStateRepresentation":
        """
        Get the complete numerical state representation.

        Returns:
            AzulStateRepresentation object containing all game state as NumPy arrays
        """
        from game.state_representation import AzulStateRepresentation

        return AzulStateRepresentation(self)

    def copy(self) -> "GameState":
        """Create an optimized copy of the game state.

        This method is optimized to minimize object allocation and copying overhead.
        Key optimizations:
        1. Uses __new__ to avoid __init__ overhead completely
        2. Direct attribute assignment without validation
        3. Optimized shallow copying for immutable data
        """
        # Create new instance without calling __init__ to avoid re-initialization
        new_state = GameState.__new__(GameState)

        # Copy simple immutable/atomic fields - these are fast direct assignments
        new_state.num_players = self.num_players
        new_state.current_player = self.current_player
        new_state.round_number = self.round_number
        new_state.game_over = self.game_over
        new_state.winner = self.winner

        # Optimized player copying - avoid any loops or list comprehensions
        if self.num_players == 2:
            new_state.players = [self.players[0].copy(), self.players[1].copy()]
        elif self.num_players == 3:
            new_state.players = [
                self.players[0].copy(),
                self.players[1].copy(),
                self.players[2].copy(),
            ]
        elif self.num_players == 4:
            new_state.players = [
                self.players[0].copy(),
                self.players[1].copy(),
                self.players[2].copy(),
                self.players[3].copy(),
            ]
        else:
            # Fallback for unusual cases
            new_state.players = [
                self.players[i].copy() for i in range(self.num_players)
            ]

        # Copy factory area
        new_state.factory_area = self.factory_area.copy()

        # Optimize tile list copying - tiles are immutable, so shallow copy is safe
        # Use slice copy which is faster than list() constructor for small-medium lists
        new_state.bag = self.bag[:]
        new_state.discard_pile = self.discard_pile[:]

        return new_state

    def __repr__(self) -> str:
        return (
            f"GameState(players={self.num_players}, round={self.round_number}, "
            f"current_player={self.current_player}, game_over={self.game_over})"
        )


def create_game(num_players: int = 2, seed: Optional[int] = None) -> GameState:
    """Convenience function to create a new game."""
    return GameState(num_players, seed)
