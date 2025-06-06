from typing import List, Optional, Tuple

from game.tile import Tile, TileColor


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

    def take_tiles(self, color: TileColor) -> Tuple[List[Tile], List[Tile]]:
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

    def get_tiles(self) -> Tuple[List[Tile], List[Tile]]:
        """Get all tiles from the factory.

        Returns:
            Tuple[List[Tile], List[Tile]]: A tuple containing (taken_tiles, remaining_tiles)
        """
        taken = self.tiles.copy()
        self.tiles = []
        return taken, []

    def copy(self) -> "Factory":
        """Create an optimized copy of this factory."""
        new_factory = Factory.__new__(Factory)
        new_factory.tiles = list(self.tiles)  # Shallow copy - tiles are immutable
        return new_factory


class CenterArea:
    """Represents the center area where leftover tiles accumulate."""

    def __init__(self):
        self.tiles: List[Tile] = []
        self.has_first_player_marker = False
        # Store a single first player marker instance that gets reused
        self._first_player_marker = Tile.create_first_player_marker()

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

        # If taking tiles from center for first time this round,
        # also take first player marker
        if taken and self.has_first_player_marker:
            # Use the single reusable first player marker instance
            taken.append(self._first_player_marker)
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

    def copy(self) -> "CenterArea":
        """Create an optimized copy of this center area."""
        new_center = CenterArea.__new__(CenterArea)
        new_center.tiles = list(self.tiles)  # Shallow copy - tiles are immutable
        new_center.has_first_player_marker = self.has_first_player_marker
        # Reuse the shared first player marker instance
        new_center._first_player_marker = self._first_player_marker
        return new_center


class FactoryArea:
    """Manages all factories and the center area."""

    def __init__(self, num_players: int) -> None:
        # Number of factories = 2 * num_players + 1
        self.num_factories = 2 * num_players + 1
        self.factories: List[Factory] = [Factory() for _ in range(self.num_factories)]
        self.center = CenterArea()

    def setup_round(self, bag: List[Tile]) -> None:
        """Setup factories for a new round."""
        # Clear center area
        self.center.clear()

        # Clear all factories first
        for factory in self.factories:
            factory.tiles = []

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
        """Check if all factories and center are empty of regular tiles."""
        all_factories_empty = all(factory.is_empty() for factory in self.factories)

        # Center is considered empty if it has no regular tiles (first player marker doesn't count)
        center_has_regular_tiles = any(
            tile.color != TileColor.FIRST_PLAYER for tile in self.center.tiles
        )

        return all_factories_empty and not center_has_regular_tiles

    def get_available_moves(self) -> List[tuple[int, TileColor]]:
        """Get all available moves.

        Returns list of (source, color) where source -1 = center.
        """
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

    def get_tiles(self) -> Tuple[List[Tile], List[Tile]]:
        """Get all tiles from all factories and center.

        Returns:
            Tuple[List[Tile], List[Tile]]: A tuple containing (taken_tiles, remaining_tiles)
        """
        taken: List[Tile] = []
        remaining: List[Tile] = []

        # Get tiles from all factories
        for factory in self.factories:
            factory_taken, factory_remaining = factory.get_tiles()
            taken.extend(factory_taken)
            remaining.extend(factory_remaining)

        # Get tiles from center
        center_tiles = self.center.tiles.copy()
        self.center.clear()
        taken.extend(center_tiles)

        return taken, remaining

    def copy(self) -> "FactoryArea":
        """Create an optimized copy of this factory area."""
        new_area = FactoryArea.__new__(FactoryArea)
        new_area.num_factories = self.num_factories

        # Copy factories - use list comprehension for type safety
        new_area.factories = [
            self.factories[i].copy() for i in range(self.num_factories)
        ]

        new_area.center = self.center.copy()
        return new_area
