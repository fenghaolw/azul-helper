from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from typing import TypeVar

    T = TypeVar("T", bound="Tile")


class TileColor(Enum):
    """Enum representing the different tile colors in Azul."""

    BLUE = "blue"
    YELLOW = "yellow"
    RED = "red"
    BLACK = "black"
    WHITE = "white"
    FIRST_PLAYER = "first_player"  # Special first player marker


class Tile:
    """Represents a single tile in the Azul game."""

    def __init__(self, color: TileColor):
        self.color = color

    def __eq__(self, other) -> bool:
        if not isinstance(other, Tile):
            return False
        return self.color == other.color

    def __hash__(self) -> int:
        return hash(self.color)

    def __repr__(self) -> str:
        return f"Tile({self.color.value})"

    def __str__(self) -> str:
        return self.color.value

    @property
    def is_first_player_marker(self) -> bool:
        """Check if this tile is the first player marker."""
        return self.color == TileColor.FIRST_PLAYER

    @classmethod
    def create_standard_tiles(cls) -> List["Tile"]:
        """Create the standard set of tiles for Azul (20 of each color)."""
        tiles = []
        for color in [
            TileColor.BLUE,
            TileColor.YELLOW,
            TileColor.RED,
            TileColor.BLACK,
            TileColor.WHITE,
        ]:
            tiles.extend([cls(color) for _ in range(20)])
        return tiles

    @classmethod
    def create_first_player_marker(cls) -> "Tile":
        """Create the first player marker tile."""
        return cls(TileColor.FIRST_PLAYER)
