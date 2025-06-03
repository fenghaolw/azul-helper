from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from typing import TypeVar

    T = TypeVar("T", bound="Tile")


class TileColor(IntEnum):
    """Enum representing the different tile colors in Azul.

    Using IntEnum for better performance - faster hashing and comparisons.
    """

    BLUE = 0
    YELLOW = 1
    RED = 2
    BLACK = 3
    WHITE = 4
    FIRST_PLAYER = 5  # Special first player marker

    def __str__(self) -> str:
        """Return string representation for compatibility."""
        names = {
            0: "blue",
            1: "yellow",
            2: "red",
            3: "black",
            4: "white",
            5: "first_player",
        }
        return names[self.value]


class Tile:
    """Represents a single tile in the Azul game."""

    # Tile pool for reusing tile instances (significant memory optimization)
    _tile_pool: Dict[TileColor, "Tile"] = {}

    def __init__(self, color: TileColor):
        self.color = color

    def __eq__(self, other) -> bool:
        if not isinstance(other, Tile):
            return False
        return self.color == other.color

    def __hash__(self) -> int:
        return hash(self.color)

    def __repr__(self) -> str:
        return f"Tile({str(self.color)})"

    def __str__(self) -> str:
        return str(self.color)

    @property
    def is_first_player_marker(self) -> bool:
        """Check if this tile is the first player marker."""
        return self.color == TileColor.FIRST_PLAYER

    @classmethod
    def get_tile(cls, color: TileColor) -> "Tile":
        """Get a tile instance from the pool (optimization for memory usage)."""
        if color not in cls._tile_pool:
            cls._tile_pool[color] = cls(color)
        return cls._tile_pool[color]

    @classmethod
    def create_standard_tiles(cls) -> List["Tile"]:
        """Create the standard set of tiles for Azul (20 of each color).

        Now uses tile pooling for better memory efficiency.
        """
        tiles = []
        for color in [
            TileColor.BLUE,
            TileColor.YELLOW,
            TileColor.RED,
            TileColor.BLACK,
            TileColor.WHITE,
        ]:
            # Reuse the same tile instance 20 times - tiles are immutable
            tile_instance = cls.get_tile(color)
            tiles.extend([tile_instance] * 20)
        return tiles

    @classmethod
    def create_first_player_marker(cls) -> "Tile":
        """Create the first player marker tile."""
        return cls.get_tile(TileColor.FIRST_PLAYER)
