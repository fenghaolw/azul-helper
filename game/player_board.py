from typing import List, Optional, Tuple

from game.tile import Tile, TileColor


class PatternLine:
    """Represents a single pattern line on the player board."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tiles: List[Tile] = []
        self.color: Optional[TileColor] = None

    def can_add_tiles(self, tiles: List[Tile]) -> bool:
        """Check if tiles can be added to this pattern line."""
        if not tiles:
            return False

        tile_color = tiles[0].color

        # Can't add first player marker to pattern lines
        if tile_color == TileColor.FIRST_PLAYER:
            return False

        # Fast path: check capacity first (most likely constraint)
        if len(tiles) > self.capacity:
            return False

        # If line is empty, any valid color can be added
        if not self.tiles:
            return True

        # If line has tiles, new tiles must be same color and fit
        return (
            self.color == tile_color and len(self.tiles) + len(tiles) <= self.capacity
        )

    def add_tiles(self, tiles: List[Tile]) -> List[Tile]:
        """Add tiles to pattern line. Returns overflow tiles."""
        if not self.can_add_tiles(tiles):
            return tiles

        if not self.tiles:
            self.color = tiles[0].color

        overflow = []
        for tile in tiles:
            if len(self.tiles) < self.capacity:
                self.tiles.append(tile)
            else:
                overflow.append(tile)

        return overflow

    def is_complete(self) -> bool:
        """Check if pattern line is complete."""
        return len(self.tiles) == self.capacity

    def clear(self) -> Tuple[Optional[Tile], List[Tile]]:
        """Clear the pattern line. Returns (tile_for_wall, tiles_for_discard)."""
        if not self.is_complete():
            return None, []

        wall_tile = self.tiles[0] if self.tiles else None
        discard_tiles = self.tiles[1:] if len(self.tiles) > 1 else []

        self.tiles = []
        self.color = None

        return wall_tile, discard_tiles

    def copy(self) -> "PatternLine":
        """Create an optimized copy of this pattern line."""
        new_line = PatternLine.__new__(PatternLine)
        new_line.capacity = self.capacity
        new_line.tiles = list(self.tiles)  # Shallow copy - tiles are immutable
        new_line.color = self.color
        return new_line


class Wall:
    """Represents the wall (scoring area) of the player board."""

    # Standard Azul wall pattern
    WALL_PATTERN = [
        [
            TileColor.BLUE,
            TileColor.YELLOW,
            TileColor.RED,
            TileColor.BLACK,
            TileColor.WHITE,
        ],
        [
            TileColor.WHITE,
            TileColor.BLUE,
            TileColor.YELLOW,
            TileColor.RED,
            TileColor.BLACK,
        ],
        [
            TileColor.BLACK,
            TileColor.WHITE,
            TileColor.BLUE,
            TileColor.YELLOW,
            TileColor.RED,
        ],
        [
            TileColor.RED,
            TileColor.BLACK,
            TileColor.WHITE,
            TileColor.BLUE,
            TileColor.YELLOW,
        ],
        [
            TileColor.YELLOW,
            TileColor.RED,
            TileColor.BLACK,
            TileColor.WHITE,
            TileColor.BLUE,
        ],
    ]

    # Precomputed lookup table: COLOR_COLUMNS[row][color] = column
    # This avoids the expensive linear search in can_place_tile
    COLOR_COLUMNS = [
        {color: col for col, color in enumerate(row)} for row in WALL_PATTERN
    ]

    def __init__(self):
        # Track which positions are filled
        self.filled = [[False for _ in range(5)] for _ in range(5)]

    def can_place_tile(self, row: int, color: TileColor) -> bool:
        """Check if a tile of given color can be placed in the row."""
        # Early bounds check with single comparison
        if row < 0 or row >= 5:
            return False

        # Use precomputed lookup table - get() with default for safety
        col = self.COLOR_COLUMNS[row].get(color, -1)
        if col == -1:
            return False

        # Direct array access - fastest check
        return not self.filled[row][col]

    def place_tile(self, row: int, color: TileColor) -> int:
        """Place a tile on the wall. Returns points scored."""
        # Check for valid row index
        if row < 0 or row >= 5:
            return 0

        # Use precomputed lookup table instead of linear search
        col = self.COLOR_COLUMNS[row].get(color)
        if col is None:
            return 0

        self.filled[row][col] = True
        return self._calculate_points(row, col)

    def _calculate_points(self, row: int, col: int) -> int:
        """Calculate points for placing a tile at given position."""
        points = 1

        # Check horizontal connections
        horizontal = 1
        # Check left
        for c in range(col - 1, -1, -1):
            if self.filled[row][c]:
                horizontal += 1
            else:
                break
        # Check right
        for c in range(col + 1, 5):
            if self.filled[row][c]:
                horizontal += 1
            else:
                break

        # Check vertical connections
        vertical = 1
        # Check up
        for r in range(row - 1, -1, -1):
            if self.filled[r][col]:
                vertical += 1
            else:
                break
        # Check down
        for r in range(row + 1, 5):
            if self.filled[r][col]:
                vertical += 1
            else:
                break

        # If connected in both directions, add both
        if horizontal > 1 and vertical > 1:
            points = horizontal + vertical
        elif horizontal > 1:
            points = horizontal
        elif vertical > 1:
            points = vertical

        return points

    def is_row_complete(self, row: int) -> bool:
        """Check if a row is completely filled."""
        return all(self.filled[row])

    def is_column_complete(self, col: int) -> bool:
        """Check if a column is completely filled."""
        return all(self.filled[row][col] for row in range(5))

    def is_color_complete(self, color: TileColor) -> bool:
        """Check if all tiles of a color are placed."""
        for row in range(5):
            for col in range(5):
                if self.WALL_PATTERN[row][col] == color and not self.filled[row][col]:
                    return False
        return True

    def get_completed_rows(self) -> List[int]:
        """Get list of completed row indices."""
        return [row for row in range(5) if self.is_row_complete(row)]

    def get_completed_columns(self) -> List[int]:
        """Get list of completed column indices."""
        return [col for col in range(5) if self.is_column_complete(col)]

    def get_completed_colors(self) -> List[TileColor]:
        """Get list of completed colors."""
        colors = [
            TileColor.BLUE,
            TileColor.YELLOW,
            TileColor.RED,
            TileColor.BLACK,
            TileColor.WHITE,
        ]
        return [color for color in colors if self.is_color_complete(color)]

    def copy(self) -> "Wall":
        """Create an optimized copy of this wall."""
        new_wall = Wall.__new__(Wall)
        # Optimize copying of 2D boolean array - use list comprehension for speed
        new_wall.filled = [row[:] for row in self.filled]
        return new_wall


class PlayerBoard:
    """Represents a player's board including pattern lines, wall, and floor line."""

    # Penalty points for floor line positions
    FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3]

    def __init__(self):
        self.pattern_lines = [PatternLine(i + 1) for i in range(5)]
        self.wall = Wall()
        self.floor_line: List[Tile] = []
        self.score = 0

    def can_place_tiles_on_pattern_line(
        self, line_index: int, tiles: List[Tile]
    ) -> bool:
        """Check if tiles can be placed on the specified pattern line."""
        # Early bounds check
        if line_index < 0 or line_index >= 5 or not tiles:
            return False

        color = tiles[0].color
        pattern_line = self.pattern_lines[line_index]

        # Fast inline pattern line validation (avoid method call overhead)
        tile_color = color

        # Can't add first player marker to pattern lines
        if tile_color == TileColor.FIRST_PLAYER:
            return False

        # If line is empty, check capacity and wall
        if not pattern_line.tiles:
            if len(tiles) > pattern_line.capacity:
                return False
        else:
            # If line has tiles, new tiles must be same color and fit
            if (
                pattern_line.color != tile_color
                or len(pattern_line.tiles) + len(tiles) > pattern_line.capacity
            ):
                return False

        # Only check wall constraint after pattern line validation passes
        return self.wall.can_place_tile(line_index, color)

    def place_tiles_on_pattern_line(
        self, line_index: int, tiles: List[Tile]
    ) -> List[Tile]:
        """Place tiles on pattern line. Overflow goes to floor line.

        Returns discarded tiles.
        """
        if line_index < 0 or line_index >= 5:
            return self.place_tiles_on_floor_line(tiles)

        # Check if tiles can be placed on this pattern line
        if not self.can_place_tiles_on_pattern_line(line_index, tiles):
            return self.place_tiles_on_floor_line(tiles)

        overflow = self.pattern_lines[line_index].add_tiles(tiles)
        return self.place_tiles_on_floor_line(overflow)

    def place_tiles_on_floor_line(self, tiles: List[Tile]) -> List[Tile]:
        """Place tiles directly on floor line.

        Returns excess tiles that should be discarded.
        """
        discarded = []
        for tile in tiles:
            if len(self.floor_line) < 7:  # Floor line can only hold 7 tiles
                self.floor_line.append(tile)
            else:
                discarded.append(
                    tile
                )  # Excess tiles are discarded (returned to box) per Azul rules
        return discarded

    def end_round_scoring(self) -> Tuple[int, List[Tile]]:
        """Perform end-of-round scoring. Returns (points_scored, tiles_to_discard)."""
        points = 0
        tiles_to_discard = []

        # Move completed pattern lines to wall
        for i, pattern_line in enumerate(self.pattern_lines):
            if pattern_line.is_complete():
                wall_tile, discard_tiles = pattern_line.clear()
                if wall_tile:
                    points += self.wall.place_tile(i, wall_tile.color)
                tiles_to_discard.extend(discard_tiles)

        # Apply floor line penalties
        floor_penalty = 0
        for i, _tile in enumerate(self.floor_line):
            if i < len(self.FLOOR_PENALTIES):
                floor_penalty += self.FLOOR_PENALTIES[i]

        # Add floor tiles to discard (except first player marker)
        for tile in self.floor_line:
            if not tile.is_first_player_marker:
                tiles_to_discard.append(tile)

        self.floor_line = []

        # Update score (can't go below 0)
        self.score = max(0, self.score + points + floor_penalty)

        return points + floor_penalty, tiles_to_discard

    def final_scoring(self) -> int:
        """Calculate final bonus points."""
        bonus_points = 0

        # Bonus for completed rows (2 points each)
        bonus_points += len(self.wall.get_completed_rows()) * 2

        # Bonus for completed columns (7 points each)
        bonus_points += len(self.wall.get_completed_columns()) * 7

        # Bonus for completed colors (10 points each)
        bonus_points += len(self.wall.get_completed_colors()) * 10

        self.score += bonus_points
        return bonus_points

    def has_first_player_marker(self) -> bool:
        """Check if player has the first player marker."""
        return any(tile.is_first_player_marker for tile in self.floor_line)

    def remove_first_player_marker(self) -> bool:
        """Remove and return whether first player marker was present."""
        for i, tile in enumerate(self.floor_line):
            if tile.is_first_player_marker:
                self.floor_line.pop(i)
                return True
        return False

    def copy(self) -> "PlayerBoard":
        """Create an optimized copy of this player board."""
        # Use __new__ to avoid __init__ overhead
        new_board = PlayerBoard.__new__(PlayerBoard)

        # Copy pattern lines - use list comprehension for type safety
        new_board.pattern_lines = [self.pattern_lines[i].copy() for i in range(5)]

        # Copy wall
        new_board.wall = self.wall.copy()

        # Shallow copy floor line - tiles are immutable
        new_board.floor_line = list(self.floor_line)

        # Copy score
        new_board.score = self.score

        return new_board
