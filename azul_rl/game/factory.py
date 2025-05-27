import random
from typing import Dict, List, Optional

from .tile import Tile, TileColor


class Factory:
    """Represents a factory display that holds 4 tiles."""

    def __init__(self):
        self.tiles: List[Tile] = []

    def fill_from_bag(self, bag: List[Tile]) -> None:
        """Fill factory with 4 tiles from the bag."""
        self.tiles = []
        for _ in range(4):
            if bag:
                tile = bag.pop()
                self.tiles.append(tile)

    def take_tiles(self, color: TileColor) -> tuple[List[Tile], List[Tile]]:
        """Take all tiles of specified color. Returns (taken_tiles, remaining_tiles)."""
        taken = []
        remaining = []

        for tile in self.tiles:
            if tile.color == color:
                taken.append(tile)
            else:
                remaining.append(tile)

        self.tiles = []
        return taken, remaining

    def is_empty(self) -> bool:
        """Check if factory is empty."""
        return len(self.tiles) == 0

    def has_color(self, color: TileColor) -> bool:
        """Check if factory has tiles of specified color."""
        return any(tile.color == color for tile in self.tiles)

    def get_available_colors(self) -> List[TileColor]:
        """Get list of colors available in this factory."""
        colors = set()
        for tile in self.tiles:
            if tile.color != TileColor.FIRST_PLAYER:
                colors.add(tile.color)
        return list(colors)

    def __repr__(self) -> str:
        return f"Factory({[str(tile) for tile in self.tiles]})"


class CenterArea:
    """Represents the center area where leftover tiles accumulate."""

    def __init__(self):
        self.tiles: List[Tile] = []
        self.has_first_player_marker = False

    def add_tiles(self, tiles: List[Tile]) -> None:
        """Add tiles to the center area."""
        self.tiles.extend(tiles)

    def add_first_player_marker(self) -> None:
        """Add the first player marker to the center."""
        self.has_first_player_marker = True

    def take_tiles(self, color: TileColor) -> List[Tile]:
        """Take all tiles of specified color from center."""
        taken = []
        remaining = []

        for tile in self.tiles:
            if tile.color == color:
                taken.append(tile)
            else:
                remaining.append(tile)

        self.tiles = remaining

        # If taking tiles from center for first time this round, also take first player marker
        if taken and self.has_first_player_marker:
            taken.append(Tile.create_first_player_marker())
            self.has_first_player_marker = False

        return taken

    def is_empty(self) -> bool:
        """Check if center area is empty (ignoring first player marker)."""
        return len(self.tiles) == 0

    def has_color(self, color: TileColor) -> bool:
        """Check if center has tiles of specified color."""
        return any(tile.color == color for tile in self.tiles)

    def get_available_colors(self) -> List[TileColor]:
        """Get list of colors available in center."""
        colors = set()
        for tile in self.tiles:
            if tile.color != TileColor.FIRST_PLAYER:
                colors.add(tile.color)
        return list(colors)

    def clear(self) -> None:
        """Clear the center area."""
        self.tiles = []
        self.has_first_player_marker = False

    def __repr__(self) -> str:
        tiles_str = [str(tile) for tile in self.tiles]
        if self.has_first_player_marker:
            tiles_str.append("first_player")
        return f"Center({tiles_str})"


class FactoryArea:
    """Manages all factories and the center area."""

    def __init__(self, num_players: int):
        # Number of factories = 2 * num_players + 1
        self.num_factories = 2 * num_players + 1
        self.factories: List[Factory] = [Factory() for _ in range(self.num_factories)]
        self.center = CenterArea()

    def setup_round(self, bag: List[Tile]) -> None:
        """Setup factories for a new round."""
        # Clear center area
        self.center.clear()

        # Add first player marker to center
        self.center.add_first_player_marker()

        # Fill each factory with 4 tiles
        for factory in self.factories:
            factory.fill_from_bag(bag)

    def take_from_factory(self, factory_index: int, color: TileColor) -> List[Tile]:
        """Take tiles from a specific factory."""
        if factory_index < 0 or factory_index >= len(self.factories):
            return []

        taken, remaining = self.factories[factory_index].take_tiles(color)

        # Add remaining tiles to center
        if remaining:
            self.center.add_tiles(remaining)

        return taken

    def take_from_center(self, color: TileColor) -> List[Tile]:
        """Take tiles from center area."""
        return self.center.take_tiles(color)

    def is_round_over(self) -> bool:
        """Check if all factories and center are empty."""
        all_factories_empty = all(factory.is_empty() for factory in self.factories)
        center_empty = self.center.is_empty()
        return all_factories_empty and center_empty

    def get_available_moves(self) -> List[tuple[int, TileColor]]:
        """Get all available moves. Returns list of (source, color) where source -1 = center."""
        moves = []

        # Check center
        for color in self.center.get_available_colors():
            moves.append((-1, color))

        # Check factories
        for i, factory in enumerate(self.factories):
            for color in factory.get_available_colors():
                moves.append((i, color))

        return moves

    def __repr__(self) -> str:
        return f"FactoryArea(factories={self.factories}, center={self.center})"
