import { Tile, ScoreDetails, FinalScoreDetails } from "./types";

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

  console.warn(
    `[PlayerBoard] Unknown tile string for enum conversion: "${tileString}"`,
  );
  return null;
}

export class PlayerBoard {
  // Wall pattern for Azul (each row has tiles in specific order)
  public static readonly WALL_PATTERN = [
    [Tile.Blue, Tile.Yellow, Tile.Red, Tile.Black, Tile.White],
    [Tile.White, Tile.Blue, Tile.Yellow, Tile.Red, Tile.Black],
    [Tile.Black, Tile.White, Tile.Blue, Tile.Yellow, Tile.Red],
    [Tile.Red, Tile.Black, Tile.White, Tile.Blue, Tile.Yellow],
    [Tile.Yellow, Tile.Red, Tile.Black, Tile.White, Tile.Blue],
  ];

  wall: Array<Array<Tile | null>> = [[], [], [], [], []]; // Allow null for empty wall spots
  lines: Array<Array<Tile>> = [[], [], [], [], []];
  floor: Array<Tile> = [];
  score: number = 0;
  name: string = "Player"; // Default name
  private finalScoringApplied: boolean = false;

  constructor(name: string = "Player") {
    this.name = name;
    // Initialize wall with nulls and lines with empty arrays
    for (let i = 0; i < 5; i++) {
      this.wall[i] = Array(5).fill(null); // Wall spots are initially empty (null)
      this.lines[i] = [];
    }
  }

  // Reset the board for a new game
  reset(): void {
    this.score = 0;
    this.finalScoringApplied = false;
    this.floor = [];

    for (let i = 0; i < 5; i++) {
      this.wall[i] = Array(5).fill(null);
      this.lines[i] = [];
    }
  }

  // Method to load state from BGA-like data
  loadState(data: {
    lines: string[][];
    wall: string[][];
    floor: string[];
    score: number;
  }): void {
    this.score = data.score;

    // Clear and load lines
    this.lines = Array(5)
      .fill(null)
      .map(() => []);
    for (let i = 0; i < 5; i++) {
      if (data.lines[i]) {
        this.lines[i] = data.lines[i]
          .map((sTile) => stringToTile(sTile))
          .filter((t) => t !== null) as Tile[];
      }
    }

    // Clear and load wall
    this.wall = Array(5)
      .fill(null)
      .map(() => Array(5).fill(null));
    for (let r = 0; r < 5; r++) {
      if (data.wall[r]) {
        for (let c = 0; c < 5; c++) {
          if (data.wall[r][c] && data.wall[r][c] !== "") {
            // Check for empty string representing no tile
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
    this.floor = data.floor
      .map((sTile) => stringToTile(sTile))
      .filter((t) => t !== null) as Tile[];

    console.log("PlayerBoard loaded state:", {
      score: this.score,
      lines: this.lines,
      wall: this.wall,
      floor: this.floor,
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
  moveToWall(discardPile: Array<Tile> = []): {
    scoreGained: number;
    details: ScoreDetails;
  } {
    let scoreGained = 0;
    const details: ScoreDetails = {
      tilesPlaced: [],
      floorPenalties: [],
      totalTileScore: 0,
      totalFloorPenalty: 0,
      previousScore: this.score,
      newScore: 0,
    };

    // Move completed lines to wall
    for (let i = 0; i < 5; i++) {
      const line = this.lines[i];
      const requiredTiles = i + 1;

      if (line.length === requiredTiles) {
        const tileToPlace = line[0]; // The tile type to be placed
        const wallCol = PlayerBoard.WALL_PATTERN[i].indexOf(tileToPlace);

        if (wallCol !== -1 && this.wall[i][wallCol] === null) {
          // Check if spot is actually empty
          this.wall[i][wallCol] = tileToPlace; // Place tile on wall

          // Calculate score for this tile
          const adjacentInfo = this.getAdjacentTilesInfo(i, wallCol);
          const tileScore = this.calculateTileScore(i, wallCol, tileToPlace); // Pass tile for logging
          scoreGained += tileScore;
          details.totalTileScore += tileScore;

          console.log(
            `  Tile ${tileToPlace} placed at (${i + 1}, ${wallCol + 1}): ${tileScore} points`,
          );
          console.log(
            `    Adjacent tiles: ${adjacentInfo.horizontal} horizontal, ${adjacentInfo.vertical} vertical`,
          );
          console.log(
            `    Score calculation: 1 base + ${Math.max(0, adjacentInfo.horizontal - 1)} horizontal + ${Math.max(0, adjacentInfo.vertical - 1)} vertical = ${tileScore}`,
          );

          details.tilesPlaced.push({
            tile: tileToPlace,
            row: i,
            col: wallCol,
            score: tileScore,
            adjacentTiles: adjacentInfo,
          });

          // Discard excess tiles from completed line (all but one go to discard pile)
          const tilesToDiscard = line.slice(1); // All except the one placed on wall
          if (tilesToDiscard.length > 0) {
            console.log(
              `  Discarding ${tilesToDiscard.length} excess ${tileToPlace} tiles to discard pile`,
            );
            discardPile.push(...tilesToDiscard);
          }

          // Clear the line
          this.lines[i] = [];
        } else {
          // This case should ideally not happen if canPlaceTile was checked correctly
          // or if line somehow filled with a tile already on the wall for that row.
          // For now, assume the tiles from this line go to floor as penalty if they can't be placed.
          console.warn(
            `Cannot place tile ${tileToPlace} on wall row ${i}, col ${wallCol}. Spot occupied or invalid. Tiles go to floor.`,
          );
          this.floor.push(...line);
          this.lines[i] = [];
        }
      }
    }

    // Calculate floor penalties and discard floor tiles (except first player token)
    const floorPenalties = [-1, -1, -2, -2, -2, -3, -3];
    let floorTileCount = 0;
    const tilesToDiscardFromFloor = [];

    for (const tile of this.floor) {
      if (floorTileCount < floorPenalties.length) {
        const penalty = floorPenalties[floorTileCount];
        scoreGained += penalty;
        details.totalFloorPenalty += penalty;
        details.floorPenalties.push({
          tile,
          position: floorTileCount,
          penalty,
        });
        floorTileCount++;
      }

      // Collect non-first-player tiles for discarding
      if (tile !== Tile.FirstPlayer) {
        tilesToDiscardFromFloor.push(tile);
      }
    }

    // Discard floor tiles (except first player token)
    if (tilesToDiscardFromFloor.length > 0) {
      console.log(
        `  Discarding ${tilesToDiscardFromFloor.length} tiles from floor to discard pile`,
      );
      discardPile.push(...tilesToDiscardFromFloor);
    }

    // Clear floor (first player token stays for next round if present)
    const hasFirstPlayerToken = this.floor.includes(Tile.FirstPlayer);
    this.floor = hasFirstPlayerToken ? [Tile.FirstPlayer] : [];

    this.score = Math.max(0, this.score + scoreGained);
    details.newScore = this.score;

    return { scoreGained, details };
  }

  // Get adjacent tiles info for debugging
  private getAdjacentTilesInfo(
    row: number,
    col: number,
  ): { horizontal: number; vertical: number } {
    // Count horizontal connected tiles
    let horizontalCount = 1;
    // Count left
    for (let c = col - 1; c >= 0; c--) {
      if (this.wall[row][c] !== null) {
        // Check if a tile is present
        horizontalCount++;
      } else {
        break;
      }
    }
    // Count right
    for (let c = col + 1; c < 5; c++) {
      if (this.wall[row][c] !== null) {
        // Check if a tile is present
        horizontalCount++;
      } else {
        break;
      }
    }

    // Count vertical connected tiles
    let verticalCount = 1;
    // Count up
    for (let r = row - 1; r >= 0; r--) {
      if (
        this.wall[r][
        PlayerBoard.WALL_PATTERN[r].indexOf(
          PlayerBoard.WALL_PATTERN[row][col],
        )
        ] !== null
      ) {
        verticalCount++;
      } else {
        break;
      }
    }
    // Count down
    for (let r = row + 1; r < 5; r++) {
      if (
        this.wall[r][
        PlayerBoard.WALL_PATTERN[r].indexOf(
          PlayerBoard.WALL_PATTERN[row][col],
        )
        ] !== null
      ) {
        verticalCount++;
      } else {
        break;
      }
    }

    return { horizontal: horizontalCount, vertical: verticalCount };
  }

  // Calculate score for placing a tile at specific position
  private calculateTileScore(
    row: number,
    col: number,
    tileBeingPlaced: Tile,
  ): number {
    console.log(`\nCalculating score for tile ${tileBeingPlaced} at (${row}, ${col}) for ${this.name}`);

    // Ensure the tile being scored is actually on the wall at [row][col] for accurate adjacent check
    if (this.wall[row][col] !== tileBeingPlaced) {
      console.warn(
        `Scoring mismatch: Tile ${tileBeingPlaced} not found at wall[${row}][${col}] during scoring for ${this.name}.`,
      );
      if (this.wall[row][col] === null) {
        console.log(`Tile not found in wall for ${this.name}, returning base score of 1`);
        return 1;
      }
    }

    const adjacent = this.getAdjacentTilesInfo(row, col);
    console.log(`Adjacent tiles info: horizontal=${adjacent.horizontal}, vertical=${adjacent.vertical}`);

    let score = 0;

    // Handle isolated tile case
    if (adjacent.horizontal === 1 && adjacent.vertical === 1) {
      console.log('Isolated tile detected, returning base score of 1');
      return 1;
    }

    // Calculate score based on lines
    if (adjacent.horizontal > 1) {
      score += adjacent.horizontal;
      console.log(`Adding ${adjacent.horizontal} points from horizontal line`);
    }
    if (adjacent.vertical > 1) {
      score += adjacent.vertical;
      console.log(`Adding ${adjacent.vertical} points from vertical line`);
    }
    if (adjacent.horizontal === 1 && adjacent.vertical > 1) {
      score += 1;
      console.log('Adding 1 point for tile being part of vertical line only');
    }
    if (adjacent.vertical === 1 && adjacent.horizontal > 1) {
      score += 1;
      console.log('Adding 1 point for tile being part of horizontal line only');
    }

    const finalScore = Math.max(1, score);
    console.log(`Final score calculation: ${score} points (minimum 1)`);
    return finalScore;
  }

  // Check if game should end (player has completed a row)
  hasCompletedRow(): boolean {
    return this.wall.some(
      (row) => row.filter((tile) => tile !== null).length === 5,
    );
  }

  // Calculate final bonus scores
  calculateFinalScore(): { bonus: number; details: FinalScoreDetails } {
    const result = this.getFinalScoreCalculation();

    // Only apply the bonus once
    if (!this.finalScoringApplied) {
      this.score += result.bonus;
      this.finalScoringApplied = true;
    }

    return result;
  }

  // Calculate final bonus scores without modifying the board (for UI display)
  getFinalScoreCalculation(): { bonus: number; details: FinalScoreDetails } {
    console.log(`\nCalculating end-of-game bonuses for ${this.name}:`);
    let bonus = 0;
    const details: FinalScoreDetails = {
      completedRows: 0,
      completedColumns: 0,
      completedColors: 0,
      rowBonus: 0,
      columnBonus: 0,
      colorBonus: 0,
      previousScore: this.score,
    };

    // Row completion bonus (2 points per complete row)
    console.log(`\nChecking completed rows for ${this.name}:`);
    for (let i = 0; i < this.wall.length; i++) {
      const row = this.wall[i];
      const filledTiles = row.filter((tile) => tile !== null).length;
      console.log(`${this.name} - Row ${i + 1}: ${filledTiles}/5 tiles filled`);
      if (filledTiles === 5) {
        details.completedRows++;
        details.rowBonus += 2;
        bonus += 2;
        console.log(`${this.name} - Row ${i + 1} completed! Adding 2 points (total row bonus: ${details.rowBonus})`);
      }
    }

    // Column completion bonus (7 points per complete column)
    console.log(`\nChecking completed columns for ${this.name}:`);
    for (let col = 0; col < 5; col++) {
      let columnComplete = true;
      for (let row = 0; row < 5; row++) {
        // Check against the specific tile that SHOULD be in wall[row][col]
        if (
          this.wall[row][
          PlayerBoard.WALL_PATTERN[row].indexOf(
            PlayerBoard.WALL_PATTERN[row][col],
          )
          ] !== PlayerBoard.WALL_PATTERN[row][col]
        ) {
          columnComplete = false;
          break;
        }
      }
      if (columnComplete) {
        details.completedColumns++;
        details.columnBonus += 7;
        bonus += 7;
        console.log(`${this.name} - Column ${col + 1} completed! Adding 7 points (total column bonus: ${details.columnBonus})`);
      } else {
        console.log(`${this.name} - Column ${col + 1}: incomplete`);
      }
    }

    // Color completion bonus (10 points per complete color)
    console.log(`\nChecking completed colors for ${this.name}:`);
    const colorCounts = new Map<Tile, number>();
    for (const row of this.wall) {
      for (const tile of row) {
        if (tile !== null) {
          colorCounts.set(tile, (colorCounts.get(tile) || 0) + 1);
        }
      }
    }

    for (const [color, count] of colorCounts.entries()) {
      console.log(`${this.name} - Color ${color}: ${count}/5 tiles placed`);
      if (count === 5) {
        details.completedColors++;
        details.colorBonus += 10;
        bonus += 10;
        console.log(`${this.name} - Color ${color} completed! Adding 10 points (total color bonus: ${details.colorBonus})`);
      }
    }

    console.log(`\nFinal bonus summary for ${this.name}:`);
    console.log(`${this.name} - Row bonus: ${details.rowBonus} points (${details.completedRows} completed rows)`);
    console.log(`${this.name} - Column bonus: ${details.columnBonus} points (${details.completedColumns} completed columns)`);
    console.log(`${this.name} - Color bonus: ${details.colorBonus} points (${details.completedColors} completed colors)`);
    console.log(`${this.name} - Total bonus: ${bonus} points`);
    console.log(`${this.name} - Previous score: ${details.previousScore}`);
    console.log(`${this.name} - New total score: ${details.previousScore + bonus}`);

    return { bonus, details };
  }

  // Create a deep copy of the player board
  clone(): PlayerBoard {
    const cloned = new PlayerBoard(this.name);
    cloned.score = this.score;
    cloned.finalScoringApplied = this.finalScoringApplied;

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
