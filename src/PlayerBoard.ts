import { Tile, ScoreDetails, FinalScoreDetails } from './types.js';

export class PlayerBoard {
  // Wall pattern for Azul (each row has tiles in specific order)
  private static readonly WALL_PATTERN = [
    [Tile.Blue, Tile.Yellow, Tile.Red, Tile.Black, Tile.White],
    [Tile.White, Tile.Blue, Tile.Yellow, Tile.Red, Tile.Black],
    [Tile.Black, Tile.White, Tile.Blue, Tile.Yellow, Tile.Red],
    [Tile.Red, Tile.Black, Tile.White, Tile.Blue, Tile.Yellow],
    [Tile.Yellow, Tile.Red, Tile.Black, Tile.White, Tile.Blue]
  ];

  wall: Array<Array<Tile>> = [[], [], [], [], []];
  lines: Array<Array<Tile>> = [[], [], [], [], []];
  floor: Array<Tile> = [];
  score: number = 0;

  constructor() {
    // Initialize empty arrays
    for (let i = 0; i < 5; i++) {
      this.wall[i] = [];
      this.lines[i] = [];
    }
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
      return wallCol !== -1 && !this.wall[lineIndex].includes(tile);
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
        const tile = line[0];
        const wallCol = PlayerBoard.WALL_PATTERN[i].indexOf(tile);
        
        // Place tile on wall
        this.wall[i].push(tile);

        // Calculate score for this tile
        const adjacentInfo = this.getAdjacentTilesInfo(i, wallCol);
        const tileScore = this.calculateTileScore(i, wallCol);
        scoreGained += tileScore;
        details.totalTileScore += tileScore;

        console.log(`  Tile ${tile} placed at (${i + 1}, ${wallCol + 1}): ${tileScore} points`);
        console.log(`    Adjacent tiles: ${adjacentInfo.horizontal} horizontal, ${adjacentInfo.vertical} vertical`);
        console.log(`    Score calculation: 1 base + ${adjacentInfo.horizontal - 1} horizontal + ${adjacentInfo.vertical - 1} vertical = ${tileScore}`);

        details.tilesPlaced.push({
          tile,
          row: i,
          col: wallCol,
          score: tileScore,
          adjacentTiles: adjacentInfo
        });

        // Clear the line (tiles go back to bag except one)
        this.lines[i] = [];
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
      if (this.wall[row].includes(PlayerBoard.WALL_PATTERN[row][c])) {
        horizontalCount++;
      } else {
        break;
      }
    }
    // Count right  
    for (let c = col + 1; c < 5; c++) {
      if (this.wall[row].includes(PlayerBoard.WALL_PATTERN[row][c])) {
        horizontalCount++;
      } else {
        break;
      }
    }

    // Count vertical connected tiles
    let verticalCount = 1;
    // Count up
    for (let r = row - 1; r >= 0; r--) {
      if (this.wall[r].includes(PlayerBoard.WALL_PATTERN[r][col])) {
        verticalCount++;
      } else {
        break;
      }
    }
    // Count down
    for (let r = row + 1; r < 5; r++) {
      if (this.wall[r].includes(PlayerBoard.WALL_PATTERN[r][col])) {
        verticalCount++;
      } else {
        break;
      }
    }

    return { horizontal: horizontalCount, vertical: verticalCount };
  }

  // Calculate score for placing a tile at specific position
  private calculateTileScore(row: number, col: number): number {
    let score = 1;
    const adjacent = this.getAdjacentTilesInfo(row, col);

    if (adjacent.horizontal > 1) score += (adjacent.horizontal - 1);
    if (adjacent.vertical > 1) score += (adjacent.vertical - 1);

    return score;
  }

  // Check if game should end (player has completed a row)
  hasCompletedRow(): boolean {
    return this.wall.some(row => row.length === 5);
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
      if (row.length === 5) {
        details.completedRows++;
        details.rowBonus += 2;
        bonus += 2;
      }
    }

    // Column completion bonus (7 points per complete column)
    for (let col = 0; col < 5; col++) {
      let columnComplete = true;
      for (let row = 0; row < 5; row++) {
        if (!this.wall[row].includes(PlayerBoard.WALL_PATTERN[row][col])) {
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
        if (tile !== Tile.FirstPlayer) {
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