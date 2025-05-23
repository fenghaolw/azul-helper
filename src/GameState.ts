import { Tile, Move, GameResult, GamePhase } from './types.js';
import { PlayerBoard } from './PlayerBoard.js';

export class GameState {
  tilebag: Array<Tile> = [];
  factories: Array<Array<Tile>> = [];
  center: Array<Tile> = [];
  playerBoards: Array<PlayerBoard> = [];
  availableMoves: Array<Move> = [];
  currentPlayer: number = 0;
  round: number = 1;
  phase: GamePhase = GamePhase.TileSelection;
  gameOver: boolean = false;
  firstPlayerIndex: number = 0;
  numPlayers: number = 2;

  constructor(numPlayers: number = 2) {
    this.numPlayers = Math.max(2, Math.min(4, numPlayers));
    this.newGame();
  }

  // Initialize a new game
  newGame(): void {
    this.tilebag = [];
    this.factories = [];
    this.center = [];
    this.playerBoards = [];
    this.currentPlayer = 0;
    this.round = 1;
    this.phase = GamePhase.TileSelection;
    this.gameOver = false;
    this.firstPlayerIndex = 0;

    // Create player boards
    for (let i = 0; i < this.numPlayers; i++) {
      this.playerBoards.push(new PlayerBoard());
    }

    this.newRound();
  }

  // Start a new round
  newRound(): void {
    this.phase = GamePhase.TileSelection;
    this.currentPlayer = this.firstPlayerIndex;
    
    // Remove first player token from previous holder's floor
    for (const board of this.playerBoards) {
      board.floor = board.floor.filter(tile => tile !== Tile.FirstPlayer);
    }
    
    // Create tile bag (20 tiles of each color)
    this.tilebag = [];
    const regularTiles = [Tile.Red, Tile.Blue, Tile.Yellow, Tile.Black, Tile.White];
    for (const tile of regularTiles) {
      for (let i = 0; i < 20; i++) {
        this.tilebag.push(tile);
      }
    }
    
    // Shuffle tilebag
    this.shuffleArray(this.tilebag);
    
    // Create factories (2 * numPlayers + 1)
    const numFactories = 2 * this.numPlayers + 1;
    this.factories = [];
    for (let i = 0; i < numFactories; i++) {
      this.factories.push([]);
    }
    
    this.createFactories();
    
    // Clear center and add first player token
    this.center = [Tile.FirstPlayer];
    
    this.getMoves();
  }

  // Fill factories with tiles from bag
  createFactories(): void {
    for (let f = 0; f < this.factories.length; f++) {
      this.factories[f] = [];
      for (let t = 0; t < 4; t++) {
        if (this.tilebag.length > 0) {
          const tile = this.tilebag.pop()!;
          this.factories[f].push(tile);
        }
      }
    }
  }

  // Generate available moves for current player
  getMoves(): void {
    this.availableMoves = [];
    
    if (this.phase !== GamePhase.TileSelection) {
      return;
    }

    const currentBoard = this.playerBoards[this.currentPlayer];

    // Check factory moves
    for (let factoryIndex = 0; factoryIndex < this.factories.length; factoryIndex++) {
      const factory = this.factories[factoryIndex];
      const uniqueTiles = [...new Set(factory)];
      
      for (const tile of uniqueTiles) {
        // Try each pattern line
        for (let lineIndex = 0; lineIndex <= 4; lineIndex++) {
          if (currentBoard.canPlaceTile(tile, lineIndex)) {
            this.availableMoves.push({
              factoryIndex,
              tile,
              lineIndex
            });
          }
        }
        
        // Floor is always available
        this.availableMoves.push({
          factoryIndex,
          tile,
          lineIndex: -1
        });
      }
    }

    // Check center moves
    if (this.center.length > 0) {
      const uniqueTiles = [...new Set(this.center.filter(t => t !== Tile.FirstPlayer))];
      
      for (const tile of uniqueTiles) {
        // Try each pattern line
        for (let lineIndex = 0; lineIndex <= 4; lineIndex++) {
          if (currentBoard.canPlaceTile(tile, lineIndex)) {
            this.availableMoves.push({
              factoryIndex: -1,
              tile,
              lineIndex
            });
          }
        }
        
        // Floor is always available
        this.availableMoves.push({
          factoryIndex: -1,
          tile,
          lineIndex: -1
        });
      }
    }
  }

  // Play a move
  playMove(move: Move): boolean {
    if (!this.isValidMove(move)) {
      console.log(`Invalid move attempted: ${JSON.stringify(move)}`);
      return false;
    }

    console.log(`Player ${this.currentPlayer + 1} plays move:`, move);

    const currentBoard = this.playerBoards[this.currentPlayer];
    let tilesToPlace: Tile[] = [];

    if (move.factoryIndex === -1) {
      // Take from center
      tilesToPlace = this.center.filter(t => t === move.tile);
      this.center = this.center.filter(t => t !== move.tile);
      console.log(`  Taking ${tilesToPlace.length} ${move.tile} tiles from center`);
      
      // Check for first player token
      if (this.center.includes(Tile.FirstPlayer)) {
        currentBoard.floor.push(Tile.FirstPlayer);
        this.center = this.center.filter(t => t !== Tile.FirstPlayer);
        this.firstPlayerIndex = this.currentPlayer;
        console.log(`  Also took first player token (will go first next round)`);
      }
    } else {
      // Take from factory
      const factory = this.factories[move.factoryIndex];
      tilesToPlace = factory.filter(t => t === move.tile);
      const remainingTiles = factory.filter(t => t !== move.tile);
      console.log(`  Taking ${tilesToPlace.length} ${move.tile} tiles from factory ${move.factoryIndex}`);
      console.log(`  Moving ${remainingTiles.length} remaining tiles to center: ${remainingTiles.join(', ')}`);
      
      // Move remaining tiles to center
      this.center.push(...remainingTiles);
      
      // Clear factory
      this.factories[move.factoryIndex] = [];
    }

    // Place tiles on player board
    const excess = currentBoard.placeTiles(move.tile, tilesToPlace.length, move.lineIndex);
    if (move.lineIndex === -1) {
      console.log(`  Placing all ${tilesToPlace.length} tiles on floor`);
    } else {
      const placed = tilesToPlace.length - excess.length;
      console.log(`  Placing ${placed} tiles on pattern line ${move.lineIndex + 1}`);
      if (excess.length > 0) {
        console.log(`  ${excess.length} excess tiles go to floor`);
      }
    }
    currentBoard.floor.push(...excess);

    return this.nextTurn();
  }

  // Move to next turn
  nextTurn(): boolean {
    // Check if round is over (all factories empty and center only has first player token or is empty)
    const factoriesEmpty = this.factories.every(f => f.length === 0);
    const centerEmpty = this.center.length === 0 || 
      (this.center.length === 1 && this.center[0] === Tile.FirstPlayer);

    if (factoriesEmpty && centerEmpty) {
      return this.endRound();
    }

    // Move to next player
    this.currentPlayer = (this.currentPlayer + 1) % this.numPlayers;
    this.getMoves();
    return false;
  }

  // End current round
  endRound(): boolean {
    this.phase = GamePhase.WallTiling;

    // Move tiles from pattern lines to wall
    const roundScoringDetails = [];
    for (let i = 0; i < this.playerBoards.length; i++) {
      const result = this.playerBoards[i].moveToWall();
      roundScoringDetails.push({
        player: i,
        scoreGained: result.scoreGained,
        details: result.details
      });
    }

    // Log detailed scoring information
    console.log(`=== Round ${this.round} Scoring ===`);
    for (const playerResult of roundScoringDetails) {
      console.log(`Player ${playerResult.player + 1}:`);
      console.log(`  Previous score: ${playerResult.details.previousScore}`);
      console.log(`  Tiles placed: ${playerResult.details.tilesPlaced.length}`);
      
      for (const tile of playerResult.details.tilesPlaced) {
        console.log(`    ${tile.tile} at (${tile.row + 1}, ${tile.col + 1}): ${tile.score} points`);
        console.log(`      Adjacent: ${tile.adjacentTiles.horizontal} horizontal, ${tile.adjacentTiles.vertical} vertical`);
      }
      
      if (playerResult.details.floorPenalties.length > 0) {
        console.log(`  Floor penalties: ${playerResult.details.totalFloorPenalty} points`);
        for (const penalty of playerResult.details.floorPenalties) {
          console.log(`    ${penalty.tile} at position ${penalty.position + 1}: ${penalty.penalty} points`);
        }
      }
      
      console.log(`  Total tile score: ${playerResult.details.totalTileScore}`);
      console.log(`  Total floor penalty: ${playerResult.details.totalFloorPenalty}`);
      console.log(`  Round score change: ${playerResult.scoreGained}`);
      console.log(`  New total score: ${playerResult.details.newScore}`);
      console.log('');
    }

    // Store scoring details for UI display
    (this as any).lastRoundScoringDetails = roundScoringDetails;

    // Check if game should end
    const gameEnded = this.playerBoards.some(board => board.hasCompletedRow());
    
    if (gameEnded) {
      return this.endGame();
    }

    // Start new round
    this.round++;
    this.newRound();
    return false;
  }

  // End the game
  endGame(): boolean {
    this.phase = GamePhase.GameEnd;
    this.gameOver = true;

    // Calculate final scores
    console.log(`=== Final Scoring ===`);
    for (let i = 0; i < this.playerBoards.length; i++) {
      const result = this.playerBoards[i].calculateFinalScore();
      console.log(`Player ${i + 1} Final Bonuses:`);
      console.log(`  Previous score: ${result.details.previousScore}`);
      console.log(`  Completed rows: ${result.details.completedRows} (${result.details.rowBonus} points)`);
      console.log(`  Completed columns: ${result.details.completedColumns} (${result.details.columnBonus} points)`);
      console.log(`  Completed colors: ${result.details.completedColors} (${result.details.colorBonus} points)`);
      console.log(`  Total bonus: ${result.bonus}`);
      console.log(`  Final score: ${this.playerBoards[i].score}`);
      console.log('');
    }

    return true;
  }

  // Check if a move is valid
  isValidMove(move: Move): boolean {
    return this.availableMoves.some(m => 
      m.factoryIndex === move.factoryIndex &&
      m.tile === move.tile &&
      m.lineIndex === move.lineIndex
    );
  }

  // Get game result
  getResult(): GameResult {
    const scores = this.playerBoards.map(board => board.score);
    const maxScore = Math.max(...scores);
    const winners = scores.map((score, index) => ({ score, index }))
      .filter(p => p.score === maxScore);
    
    return {
      winner: winners.length === 1 ? winners[0].index : -1,
      scores,
      gameOver: this.gameOver
    };
  }

  // Create a deep copy of the game state for AI simulation
  clone(): GameState {
    const cloned = new GameState(this.numPlayers);
    
    cloned.tilebag = [...this.tilebag];
    cloned.center = [...this.center];
    cloned.currentPlayer = this.currentPlayer;
    cloned.round = this.round;
    cloned.phase = this.phase;
    cloned.gameOver = this.gameOver;
    cloned.firstPlayerIndex = this.firstPlayerIndex;
    
    // Deep copy factories
    cloned.factories = this.factories.map(factory => [...factory]);
    
    // Deep copy player boards
    cloned.playerBoards = this.playerBoards.map(board => board.clone());
    
    // Regenerate moves for current state
    cloned.getMoves();
    
    return cloned;
  }

  // Optimized clone that only copies what's needed for a specific move
  smartClone(_move: Move): GameState {
    const cloned = this.clone();
    return cloned;
  }

  // Utility function to shuffle array
  private shuffleArray<T>(array: T[]): void {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }

  // Advanced evaluation of the game state for AI
  evaluatePosition(playerIndex: number): number {
    if (this.gameOver) {
      const result = this.getResult();
      if (result.winner === playerIndex) return 1000;
      if (result.winner !== -1 && result.winner !== playerIndex) return -1000;
      return 0;
    }

    const playerBoard = this.playerBoards[playerIndex];
    const opponentBoards = this.playerBoards.filter((_, i) => i !== playerIndex);
    
    const playerEval = this.evaluatePlayerBoard(playerBoard);
    
    // Calculate best opponent evaluation
    let bestOpponentEval = 0;
    for (const opponentBoard of opponentBoards) {
      const opponentEval = this.evaluatePlayerBoard(opponentBoard);
      bestOpponentEval = Math.max(bestOpponentEval, opponentEval);
    }

    // 5. Add defensive considerations (block opponent progress)
    const defensiveValue = this.evaluateDefensiveOpportunities(playerIndex);
    
    const evaluation = playerEval - bestOpponentEval + defensiveValue;
    
    // Debug logging for AI evaluation
    if (Math.random() < 0.05) { // Log 5% of evaluations
      console.log(`AI eval: Player ${playerIndex + 1}: ${playerEval.toFixed(1)} vs Opponent: ${bestOpponentEval.toFixed(1)} + Defense: ${defensiveValue.toFixed(1)} = ${evaluation.toFixed(1)}`);
    }

    return evaluation;
  }

  // Comprehensive evaluation of a single player board
  private evaluatePlayerBoard(board: PlayerBoard): number {
    let evaluation = board.score;
    
    // Clone board to simulate potential moves
    const boardClone = board.clone();
    
    // 1. Immediate scoring potential from completed lines
    for (let i = 0; i < 5; i++) {
      const line = boardClone.lines[i];
      if (line.length === i + 1 && line.length > 0) {
        const tile = line[0];
        const wallCol = PlayerBoard.getWallTile(i, 0) === tile ? 0 :
                         PlayerBoard.getWallTile(i, 1) === tile ? 1 :
                         PlayerBoard.getWallTile(i, 2) === tile ? 2 :
                         PlayerBoard.getWallTile(i, 3) === tile ? 3 :
                         PlayerBoard.getWallTile(i, 4) === tile ? 4 : -1;
        
        if (wallCol !== -1 && !boardClone.wall[i].includes(tile)) {
          boardClone.wall[i].push(tile);
          evaluation += this.calculateTileScore(boardClone, i, wallCol);
        }
      }
    }

    // 2. Floor penalties
    const floorPenalties = [-1, -1, -2, -2, -2, -3, -3];
    for (let i = 0; i < Math.min(boardClone.floor.length, floorPenalties.length); i++) {
      evaluation += floorPenalties[i];
    }

    // 3. Strategic bonuses based on progress toward end-game goals
    const strategicBonus = this.evaluateStrategicProgress(boardClone);
    
    // 4. Apply game phase multiplier to strategic bonuses
    const gamePhaseMultiplier = this.getGamePhaseMultiplier();
    evaluation += strategicBonus * gamePhaseMultiplier;

    return evaluation;
  }

  // Evaluate strategic progress toward end-game bonuses
  private evaluateStrategicProgress(board: PlayerBoard): number {
    let strategicValue = 0;
    
    // 4. Row completion progress (2 points each when complete)
    for (let row = 0; row < 5; row++) {
      const tilesInRow = board.wall[row].length;
      if (tilesInRow === 5) {
        strategicValue += 2; // Full bonus for completed row
      } else if (tilesInRow === 4) {
        strategicValue += 1.5; // High value for near-complete row
      } else if (tilesInRow === 3) {
        strategicValue += 0.8; // Good progress
      } else if (tilesInRow === 2) {
        strategicValue += 0.3; // Some progress
      }
    }

    // 5. Column completion progress (7 points each when complete)
    for (let col = 0; col < 5; col++) {
      let tilesInColumn = 0;
      for (let row = 0; row < 5; row++) {
        if (board.wall[row].includes(PlayerBoard.getWallTile(row, col))) {
          tilesInColumn++;
        }
      }
      
      if (tilesInColumn === 5) {
        strategicValue += 7; // Full bonus for completed column
      } else if (tilesInColumn === 4) {
        strategicValue += 5; // Very high value for near-complete column
      } else if (tilesInColumn === 3) {
        strategicValue += 3; // Good progress
      } else if (tilesInColumn === 2) {
        strategicValue += 1; // Some progress
      }
    }

    // 6. Color completion progress (10 points each when complete)
    const colorCounts = new Map<Tile, number>();
    const regularTiles = [Tile.Red, Tile.Blue, Tile.Yellow, Tile.Black, Tile.White];
    
    for (const row of board.wall) {
      for (const tile of row) {
        if (regularTiles.includes(tile)) {
          colorCounts.set(tile, (colorCounts.get(tile) || 0) + 1);
        }
      }
    }

    for (const [_tile, count] of colorCounts.entries()) {
      if (count === 5) {
        strategicValue += 10; // Full bonus for completed color
      } else if (count === 4) {
        strategicValue += 7; // Very high value for near-complete color
      } else if (count === 3) {
        strategicValue += 4; // Good progress
      } else if (count === 2) {
        strategicValue += 1.5; // Some progress
      }
    }

    // 7. Line completion potential (encourage filling useful lines)
    for (let i = 0; i < 5; i++) {
      const line = board.lines[i];
      const required = i + 1;
      const filled = line.length;
      
      if (filled > 0 && filled < required) {
        // Value based on how close to completion and strategic importance
        const completionRatio = filled / required;
        const lineValue = (i + 1) * 0.5; // Higher lines worth more
        strategicValue += completionRatio * lineValue;
      }
    }

    return strategicValue;
  }

  // Get multiplier based on game phase - strategic bonuses matter more later in game
  private getGamePhaseMultiplier(): number {
    // Count total tiles placed across all players
    let totalTilesPlaced = 0;
    for (const board of this.playerBoards) {
      for (const row of board.wall) {
        totalTilesPlaced += row.length;
      }
    }
    
    // Max tiles per player is 25, so for 2 players max is 50
    const maxTiles = this.numPlayers * 25;
    const gameProgress = totalTilesPlaced / maxTiles;
    
    // Early game (0-30%): Focus more on immediate scoring (0.5x strategic)
    // Mid game (30-70%): Balanced approach (1.0x strategic)  
    // Late game (70%+): Heavily weight strategic bonuses (2.0x strategic)
    if (gameProgress < 0.3) {
      return 0.5;
    } else if (gameProgress < 0.7) {
      return 1.0;
    } else {
      return 2.0;
    }
  }

  // Evaluate defensive opportunities (denying opponent strategic progress)
  private evaluateDefensiveOpportunities(_playerIndex: number): number {
    // This is a placeholder for now - could be enhanced to evaluate
    // moves that prevent opponents from completing rows/columns/colors
    // For example, taking tiles the opponent needs, or blocking factory access
    
    // Simple defensive heuristic: slight bonus for taking from center
    // (reduces opponent's first player token opportunities)
    const centerTiles = this.center.filter(t => t !== Tile.FirstPlayer).length;
    return centerTiles * 0.1;
  }

  private calculateTileScore(board: PlayerBoard, row: number, col: number): number {
    let score = 1;

    // Count horizontal connected tiles
    let horizontalCount = 1;
    const wallPattern = [
      [Tile.Blue, Tile.Yellow, Tile.Red, Tile.Black, Tile.White],
      [Tile.White, Tile.Blue, Tile.Yellow, Tile.Red, Tile.Black],
      [Tile.Black, Tile.White, Tile.Blue, Tile.Yellow, Tile.Red],
      [Tile.Red, Tile.Black, Tile.White, Tile.Blue, Tile.Yellow],
      [Tile.Yellow, Tile.Red, Tile.Black, Tile.White, Tile.Blue]
    ];
    
    for (let c = col - 1; c >= 0; c--) {
      if (board.wall[row].includes(wallPattern[row][c])) {
        horizontalCount++;
      } else break;
    }
    for (let c = col + 1; c < 5; c++) {
      if (board.wall[row].includes(wallPattern[row][c])) {
        horizontalCount++;
      } else break;
    }

    // Count vertical connected tiles  
    let verticalCount = 1;
    for (let r = row - 1; r >= 0; r--) {
      if (board.wall[r].includes(wallPattern[r][col])) {
        verticalCount++;
      } else break;
    }
    for (let r = row + 1; r < 5; r++) {
      if (board.wall[r].includes(wallPattern[r][col])) {
        verticalCount++;
      } else break;
    }

    if (horizontalCount > 1) score += (horizontalCount - 1);
    if (verticalCount > 1) score += (verticalCount - 1);

    return score;
  }
} 