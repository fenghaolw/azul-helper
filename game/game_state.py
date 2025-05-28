import copy
import random
from typing import TYPE_CHECKING, List, Optional

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

    def __eq__(self, other) -> bool:
        if not isinstance(other, Action):
            return False
        return (
            self.source == other.source
            and self.color == other.color
            and self.destination == other.destination
        )

    def __hash__(self) -> int:
        return hash((self.source, self.color, self.destination))

    def __repr__(self) -> str:
        source_str = "center" if self.source == -1 else f"factory_{self.source}"
        dest_str = "floor" if self.destination == -1 else f"line_{self.destination}"
        return f"Action({source_str}, {self.color.value}, {dest_str})"


class GameState:
    """Represents the complete state of an Azul game."""

    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        if num_players < 2 or num_players > 4:
            raise ValueError("Number of players must be between 2 and 4")

        self.num_players = num_players
        self.players: List[PlayerBoard] = [PlayerBoard() for _ in range(num_players)]
        self.factory_area = FactoryArea(num_players)

        # Game state
        self.current_player = 0
        self.round_number = 1
        self.game_over = False
        self.winner: Optional[int] = None

        # Tile management
        self.bag: List[Tile] = []
        self.discard_pile: List[Tile] = (
            []
        )  # All discarded tiles go here (lid of the game box)

        # Random seed for reproducibility
        if seed is not None:
            random.seed(seed)

        self._initialize_tiles()
        self._start_new_round()

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

        for source, color in available_moves:
            # For each destination (pattern lines + floor line)
            for dest in range(-1, 5):  # -1 = floor, 0-4 = pattern lines
                if dest == -1:
                    # Can always place on floor line
                    actions.append(Action(source, color, dest))
                else:
                    # Check if can place on pattern line
                    # Get tiles that would be taken
                    if source == -1:
                        # Simulate taking from center
                        test_tiles = [
                            tile
                            for tile in self.factory_area.center.tiles
                            if tile.color == color
                        ]
                    else:
                        # Simulate taking from factory
                        test_tiles = [
                            tile
                            for tile in self.factory_area.factories[source].tiles
                            if tile.color == color
                        ]

                    if test_tiles and player.can_place_tiles_on_pattern_line(
                        dest, test_tiles
                    ):
                        actions.append(Action(source, color, dest))

        return actions

    def is_action_legal(self, action: Action, player_id: Optional[int] = None) -> bool:
        """Check if an action is legal."""
        return action in self.get_legal_actions(player_id)

    def apply_action(self, action: Action) -> bool:
        """Apply an action and return True if successful."""
        if not self.is_action_legal(action):
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
        """Create a deep copy of the game state."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"GameState(players={self.num_players}, round={self.round_number}, "
            f"current_player={self.current_player}, game_over={self.game_over})"
        )


def create_game(num_players: int = 2, seed: Optional[int] = None) -> GameState:
    """Convenience function to create a new game."""
    return GameState(num_players, seed)
