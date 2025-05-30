import { Tile, ScoreDetails, FinalScoreDetails } from './types.js';

function stringToTile(tileString: string): Tile | null {
    const s = tileString.toLowerCase();
    // Direct comparison with enum values (which are strings themselves)
    if (s === Tile.Red) return Tile.Red;
    if (s === Tile.Blue) return Tile.Blue;
    if (s === Tile.Yellow) return Tile.Yellow;
    if (s === Tile.Black) return Tile.Black;
    if (s === Tile.White) return Tile.White; // Tile.White is 'white'
    if (s === Tile.FirstPlayer) return Tile.FirstPlayer; // Tile.FirstPlayer is 'firstPlayer'
    // The BGA data for floor uses "firstplayer" (all lowercase for the token)
    // Tile.FirstPlayer is 'firstPlayer', so the above line handles it due to s.toLowerCase().

    console.warn(`[PlayerBoard] Unknown tile string for enum conversion: "${tileString}"`);
    return null;
}

export class PlayerBoard {
  // Wall pattern for Azul (each row has tiles in specific order)
  public static readonly WALL_PATTERN = [
    [Tile.Blue, Tile.Yellow, Tile.Red, Tile.Black, Tile.White],
    [Tile.White, Tile.Blue, Tile.Yellow, Tile.Red, Tile.Black],
    [Tile.Black, Tile.White, Tile.Blue, Tile.Yellow, Tile.Red],
    [Tile.Red, Tile.Black, Tile.White, Tile.Blue, Tile.Yellow],
    [Tile.Yellow, Tile.Red, Tile.Black, Tile.White, Tile.Blue]
  ];

  wall: Array<Array<Tile | null>> = [[], [], [], [], []]; // Allow null for empty wall spots
  lines: Array<Array<Tile>> = [[], [], [], [], []];
  floor: Array<Tile> = [];
  score: number = 0;

  constructor() {
    // Initialize wall with nulls and lines with empty arrays
    for (let i = 0; i < 5; i++) {
      this.wall[i] = Array(5).fill(null); // Wall spots are initially empty (null)
      this.lines[i] = [];
    }
  }

  // Method to load state from BGA-like data
  loadState(data: { lines: string[][]; wall: string[][]; floor: string[]; score: number }): void {
    this.score = data.score;

    // Clear and load lines
    this.lines = Array(5).fill(null).map(() => []);
    for (let i = 0; i < 5; i++) {
      if (data.lines[i]) {
        this.lines[i] = data.lines[i].map(sTile => stringToTile(sTile)).filter(t => t !== null) as Tile[];
      }
    }

    // Clear and load wall
    this.wall = Array(5).fill(null).map(() => Array(5).fill(null));
    for (let r = 0; r < 5; r++) {
      if (data.wall[r]) {
        for (let c = 0; c < 5; c++) {
          if (data.wall[r][c] && data.wall[r][c] !== '') { // Check for empty string representing no tile
            const tile = stringToTile(data.wall[r][c]);
            if (tile) {
              // We need to place the tile on the wall according to WALL_PATTERN
              // The BGA data wall is already in [row][col] format with actual tiles
              // if it's already placed.
              // So, if data.wall[r][c] is a tile string, it means it IS that tile.
              this.wall[r][c] = tile;
            }
          } else {
            this.wall[r][c] = null; // Explicitly set empty spots to null
          }
        }
      }
    }

    // Load floor
    this.floor = data.floor.map(sTile => stringToTile(sTile)).filter(t => t !== null) as Tile[];

    console.log('PlayerBoard loaded state:', {
        score: this.score,
        lines: this.lines,
        wall: this.wall,
        floor: this.floor
    });
  }

  // Check if a tile can be placed in a specific line
  canPlaceTile(tile: Tile, lineIndex: number): boolean {
    if (lineIndex === -1) return true; // Floor can always accept tiles
    if (lineIndex < 0 || lineIndex > 4) return false;

    const line = this.lines[lineIndex];
    const maxTiles = lineIndex + 1;

    // Check if line is full
    if (line.length >= maxTiles) return false;

    // Check if line is empty or contains same tile type
    if (line.length === 0) {
      // Check if this tile type is already on the wall in this row
      const wallCol = PlayerBoard.WALL_PATTERN[lineIndex].indexOf(tile);
      // Ensure the spot on the wall for this tile type is actually empty (null)
      return wallCol !== -1 && this.wall[lineIndex][wallCol] === null;
    }

    return line[0] === tile;
  }

  // Place tiles in a line (returns excess tiles for floor)
  placeTiles(tile: Tile, count: number, lineIndex: number): Tile[] {
    const excess: Tile[] = [];

    if (lineIndex === -1) {
      // Place on floor
      for (let i = 0; i < count; i++) {
        this.floor.push(tile);
      }
      return excess;
    }

    const line = this.lines[lineIndex];
    const maxTiles = lineIndex + 1;
    const spacesLeft = maxTiles - line.length;
    const tilesToPlace = Math.min(count, spacesLeft);
    const excessCount = count - tilesToPlace;

    // Place tiles in line
    for (let i = 0; i < tilesToPlace; i++) {
      line.push(tile);
    }

    // Put excess tiles on floor
    for (let i = 0; i < excessCount; i++) {
      excess.push(tile);
    }

    return excess;
  }

  // Move completed lines to wall and return score gained
  moveToWall(): { scoreGained: number; details: ScoreDetails } {
    let scoreGained = 0;
    const details: ScoreDetails = {
      tilesPlaced: [],
      floorPenalties: [],
      totalTileScore: 0,
      totalFloorPenalty: 0,
      previousScore: this.score,
      newScore: 0
    };

    // Move completed lines to wall
    for (let i = 0; i < 5; i++) {
      const line = this.lines[i];
      const requiredTiles = i + 1;

      if (line.length === requiredTiles) {
        const tileToPlace = line[0]; // The tile type to be placed
        const wallCol = PlayerBoard.WALL_PATTERN[i].indexOf(tileToPlace);

        if (wallCol !== -1 && this.wall[i][wallCol] === null) { // Check if spot is actually empty
            this.wall[i][wallCol] = tileToPlace; // Place tile on wall

            // Calculate score for this tile
            const adjacentInfo = this.getAdjacentTilesInfo(i, wallCol);
            const tileScore = this.calculateTileScore(i, wallCol, tileToPlace); // Pass tile for logging
            scoreGained += tileScore;
            details.totalTileScore += tileScore;

            console.log(`  Tile ${tileToPlace} placed at (${i + 1}, ${wallCol + 1}): ${tileScore} points`);
            console.log(`    Adjacent tiles: ${adjacentInfo.horizontal} horizontal, ${adjacentInfo.vertical} vertical`);
            console.log(`    Score calculation: 1 base + ${Math.max(0,adjacentInfo.horizontal -1)} horizontal + ${Math.max(0,adjacentInfo.vertical-1)} vertical = ${tileScore}`);


            details.tilesPlaced.push({
              tile: tileToPlace,
              row: i,
              col: wallCol,
              score: tileScore,
              adjacentTiles: adjacentInfo
            });

            // Clear the line (tiles go back to bag except one)
            this.lines[i] = [];
        } else {
            // This case should ideally not happen if canPlaceTile was checked correctly
            // or if line somehow filled with a tile already on the wall for that row.
            // For now, assume the tiles from this line go to floor as penalty if they can't be placed.
            console.warn(`Cannot place tile ${tileToPlace} on wall row ${i}, col ${wallCol}. Spot occupied or invalid. Tiles go to floor.`);
            this.floor.push(...line);
            this.lines[i] = [];
        }
      }
    }

    // Calculate floor penalties
    const floorPenalties = [-1, -1, -2, -2, -2, -3, -3];
    let floorTileCount = 0;
    for (const tile of this.floor) {
      if (floorTileCount < floorPenalties.length) {
        const penalty = floorPenalties[floorTileCount];
        scoreGained += penalty;
        details.totalFloorPenalty += penalty;
        details.floorPenalties.push({
          tile,
          position: floorTileCount,
          penalty
        });
        floorTileCount++;
      }
    }

    // Clear floor (first player token stays for next round if present)
    const hasFirstPlayerToken = this.floor.includes(Tile.FirstPlayer);
    this.floor = hasFirstPlayerToken ? [Tile.FirstPlayer] : [];

    this.score = Math.max(0, this.score + scoreGained);
    details.newScore = this.score;

    return { scoreGained, details };
  }

  // Get adjacent tiles info for debugging
  private getAdjacentTilesInfo(row: number, col: number): { horizontal: number; vertical: number } {
    // Count horizontal connected tiles
    let horizontalCount = 1;
    // Count left
    for (let c = col - 1; c >= 0; c--) {
      if (this.wall[row][c] !== null) { // Check if a tile is present
        horizontalCount++;
      } else {
        break;
      }
    }
    // Count right
    for (let c = col + 1; c < 5; c++) {
      if (this.wall[row][c] !== null) { // Check if a tile is present
        horizontalCount++;
      } else {
        break;
      }
    }

    // Count vertical connected tiles
    let verticalCount = 1;
    // Count up
    for (let r = row - 1; r >= 0; r--) {
      if (this.wall[r][PlayerBoard.WALL_PATTERN[r].indexOf(PlayerBoard.WALL_PATTERN[row][col])] !== null) {
        verticalCount++;
      } else {
        break;
      }
    }
    // Count down
    for (let r = row + 1; r < 5; r++) {
      if (this.wall[r][PlayerBoard.WALL_PATTERN[r].indexOf(PlayerBoard.WALL_PATTERN[row][col])] !== null) {
        verticalCount++;
      } else {
        break;
      }
    }

    return { horizontal: horizontalCount, vertical: verticalCount };
  }

  // Calculate score for placing a tile at specific position
  private calculateTileScore(row: number, col: number, tileBeingPlaced: Tile): number {
    // Ensure the tile being scored is actually on the wall at [row][col] for accurate adjacent check
    if (this.wall[row][col] !== tileBeingPlaced) {
        console.warn(`Scoring mismatch: Tile ${tileBeingPlaced} not found at wall[${row}][${col}] during scoring.`);
        // Fallback to avoid errors, though this indicates a logic issue.
        // If the tile isn't there, it gets 1 point for itself if it were placed.
        // But if it's not there, adjacent checks are for the tile that *is* there, or null.
        // This implies it couldn't be placed, so score should be 0 for this specific tile,
        // or handled by moveToWall logic (e.g. tiles go to floor).
        // For robustness, if we reach here assuming it *was* just placed:
        if (this.wall[row][col] === null) return 1; // It was just placed in an empty spot
    }

    let score = 0; // Tile itself gives 0 unless part of a line
    const adjacent = this.getAdjacentTilesInfo(row, col);

    if (adjacent.horizontal > 0) score += adjacent.horizontal; // if 1 tile, score 1; if 2, score 2, etc.
    if (adjacent.vertical > 0) score += adjacent.vertical;

    // If it forms both a horizontal AND vertical line, the tile itself is counted twice in the above.
    // But it should only be counted once. So if both are > 1 (meaning lines of 2+), subtract 1.
    if (adjacent.horizontal > 1 && adjacent.vertical > 1) {
        score -=1;
    }
    // If it's an isolated tile (no adjacent tiles either way), it scores 1 point.
    // adjacent.horizontal and vertical would be 1 each. Score = 1+1 = 2. This is wrong.
    // If horiz=1 and vert=1, then score = 1.
    // Correct logic:
    // A tile always scores at least 1 point for itself if placed.
    // If it extends a horizontal line, it gets points for all tiles in that line (including itself).
    // If it extends a vertical line, it gets points for all tiles in that line (including itself).
    // If it's part of both, it's counted in both sums, so the tile itself is counted twice.
    // The base point is for the tile itself.
    // score for horizontal line = num tiles in horizontal line (if > 1, else 0 points from line)
    // score for vertical line = num tiles in vertical line (if > 1, else 0 points from line)
    // total score = (points from horizontal) + (points from vertical)
    // if horizontal line length = 1 AND vertical line length = 1, then score is 1.
    // else, score = (length of horiz line) + (length of vert line)
    // Example: H line of 3, V line of 2. Tile is at intersection. H=3, V=2. Score = 3+2 = 5. Correct.
    // Example: H line of 1 (isolated), V line of 1 (isolated). H=1, V=1. Score = 1+1 = 2. Incorrect, should be 1.

    if (adjacent.horizontal === 1 && adjacent.vertical === 1) {
        return 1; // Isolated tile
    }
    score = 0;
    if (adjacent.horizontal > 1) score += adjacent.horizontal;
    if (adjacent.vertical > 1) score += adjacent.vertical;
    if (adjacent.horizontal === 1 && adjacent.vertical > 1) score +=1; // Part of vertical only, count itself
    if (adjacent.vertical === 1 && adjacent.horizontal > 1) score +=1; // Part of horizontal only, count itself

    return Math.max(1, score); // Must score at least 1 if placed.
  }

  // Check if game should end (player has completed a row)
  hasCompletedRow(): boolean {
    return this.wall.some(row => row.filter(tile => tile !== null).length === 5);
  }

  // Calculate final bonus scores
  calculateFinalScore(): { bonus: number; details: FinalScoreDetails } {
    let bonus = 0;
    const details: FinalScoreDetails = {
      completedRows: 0,
      completedColumns: 0,
      completedColors: 0,
      rowBonus: 0,
      columnBonus: 0,
      colorBonus: 0,
      previousScore: this.score
    };

    // Row completion bonus (2 points per complete row)
    for (const row of this.wall) {
      if (row.filter(tile => tile !== null).length === 5) {
        details.completedRows++;
        details.rowBonus += 2;
        bonus += 2;
      }
    }

    // Column completion bonus (7 points per complete column)
    for (let col = 0; col < 5; col++) {
      let columnComplete = true;
      for (let row = 0; row < 5; row++) {
        // Check against the specific tile that SHOULD be in wall[row][col]
        if (this.wall[row][PlayerBoard.WALL_PATTERN[row].indexOf(PlayerBoard.WALL_PATTERN[row][col])] !== PlayerBoard.WALL_PATTERN[row][col]) {
          columnComplete = false;
          break;
        }
      }
      if (columnComplete) {
        details.completedColumns++;
        details.columnBonus += 7;
        bonus += 7;
      }
    }

    // Color completion bonus (10 points per complete color)
    const colorCounts = new Map<Tile, number>();
    for (const row of this.wall) {
      for (const tile of row) {
        if (tile !== null) { // Only count actual tiles
          colorCounts.set(tile, (colorCounts.get(tile) || 0) + 1);
        }
      }
    }

    for (const count of colorCounts.values()) {
      if (count === 5) {
        details.completedColors++;
        details.colorBonus += 10;
        bonus += 10;
      }
    }

    this.score += bonus;
    return { bonus, details };
  }

  // Create a deep copy of the player board
  clone(): PlayerBoard {
    const cloned = new PlayerBoard();
    cloned.score = this.score;

    // Deep copy wall
    for (let i = 0; i < 5; i++) {
      cloned.wall[i] = [...this.wall[i]];
      cloned.lines[i] = [...this.lines[i]];
    }

    cloned.floor = [...this.floor];

    return cloned;
  }

  // Get the tile type that should be placed in a specific wall position
  static getWallTile(row: number, col: number): Tile {
    return PlayerBoard.WALL_PATTERN[row][col];
  }
}
