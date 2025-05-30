import { Tile, Move, GameResult, GamePhase } from './types.js';
import { PlayerBoard } from './PlayerBoard.js';

// Helper function to safely convert string to Tile enum
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

    console.warn(`[GameState] Unknown tile string for enum conversion: "${tileString}"`);
    return null;
}

/**
 * Base GameState class containing core game logic shared between web app and extension
 */
export abstract class BaseGameState {
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
  }

  // Load game state from BGA-like data structure
  loadFromBga(bgaData: {
    factories: string[][];
    center: string[];
    playerBoards: { lines: string[][]; wall: string[][]; floor: string[]; score: number }[];
    currentPlayer: number;
    round: number; // BGA might not provide round, default or calculate if necessary
  }): void {
    this.numPlayers = bgaData.playerBoards.length;
    this.round = bgaData.round !== undefined ? bgaData.round : 1; // Default round to 1 if not provided
    this.currentPlayer = bgaData.currentPlayer;
    this.phase = GamePhase.TileSelection; // Assuming BGA state is always during tile selection phase for AI
    this.gameOver = false; // Reset game over status
    this.firstPlayerIndex = 0; // Reset, will be determined by first player token

    // Initialize factories
    this.factories = bgaData.factories.map(factory =>
      factory.map(sTile => stringToTile(sTile)).filter(t => t !== null) as Tile[]
    );

    // Initialize center, carefully handling FirstPlayer token
    this.center = bgaData.center
        .map(sTile => stringToTile(sTile))
        .filter(t => t !== null) as Tile[];

    // Initialize player boards
    if (this.playerBoards.length !== this.numPlayers) {
        this.playerBoards = [];
        for (let i = 0; i < this.numPlayers; i++) {
            this.playerBoards.push(new PlayerBoard());
        }
    }

    let firstPlayerTokenFoundOnBoard = false;
    for (let i = 0; i < this.numPlayers; i++) {
      this.playerBoards[i].loadState(bgaData.playerBoards[i]);
      // Check if this player has the first player token on their floor
      if (this.playerBoards[i].floor.includes(Tile.FirstPlayer)) {
        this.firstPlayerIndex = i;
        firstPlayerTokenFoundOnBoard = true;
        // Remove from center if it was also there (BGA might be inconsistent)
        this.center = this.center.filter(t => t !== Tile.FirstPlayer);
      }
    }

    // If first player token wasn't on a board, check if it's in the center
    if (!firstPlayerTokenFoundOnBoard && !this.center.includes(Tile.FirstPlayer)) {
        // If it's NOWHERE, but it should be SOMEWHERE in TileSelection phase (unless all tiles taken from center already)
        // This logic might need adjustment based on exact BGA state representation when center is emptied.
        // For now, if no token and center is empty of regular tiles, assume previous round's first player keeps it.
        // If center still has tiles but no token, it implies it was taken.
        // This is tricky without knowing BGA's exact first player token rules post-center-clearing.
        // A simple assumption: if not on a board and not in center, it's not in play for *this specific turn's start*.
        // The firstPlayerIndex would then be determined by who *takes* it from the center.
    } else if (this.center.includes(Tile.FirstPlayer) && firstPlayerTokenFoundOnBoard) {
        // If on a board AND in center, prioritize board, remove from center.
        this.center = this.center.filter(t => t !== Tile.FirstPlayer);
    }

    // If no player has the token yet and it's not in the center, this implies it hasn't been picked up.
    // The `firstPlayerIndex` will be set when a player takes it from the center via `playMove`.
    // If it IS in the center, no player is `firstPlayerIndex` yet for *next* round until it's taken.
    // If it IS on a player's board, that player is `firstPlayerIndex` for *next* round.

    // Ensure a valid currentPlayer (e.g. if BGA sends an out-of-bounds index)
    this.currentPlayer = Math.max(0, Math.min(this.numPlayers - 1, this.currentPlayer));

    this.getMoves(); // Crucially, generate moves for the loaded state.
    console.log('GameState loaded from BGA data:', this);
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
    this.onNewRound();
    return false;
  }

  // Abstract method for handling new round - implemented differently by subclasses
  protected abstract onNewRound(): void;

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
  clone(): BaseGameState {
    // This will be overridden by concrete subclasses
    throw new Error('clone() must be implemented by concrete subclasses');
  }

  // Optimized clone that only copies what's needed for a specific move
  smartClone(_move: Move): BaseGameState {
    return this.clone();
  }

  // Utility function to shuffle array
  protected shuffleArray<T>(array: T[]): void {
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

    // Enhanced defensive considerations
    const defensiveValue = this.evaluateDefensiveOpportunities(playerIndex);

    // Tactical evaluation (tile scarcity, forcing moves, etc.)
    const tacticalValue = this.evaluateTacticalOpportunities(playerIndex);

    // Tempo evaluation (first player advantage, timing)
    const tempoValue = this.evaluateTempoAdvantage(playerIndex);

    const evaluation = playerEval - bestOpponentEval + defensiveValue + tacticalValue + tempoValue;

    // Debug logging for AI evaluation
    if (Math.random() < 0.05) { // Log 5% of evaluations
      console.log(`AI eval: Player ${playerIndex + 1}: ${playerEval.toFixed(1)} vs Opponent: ${bestOpponentEval.toFixed(1)} + Defense: ${defensiveValue.toFixed(1)} + Tactical: ${tacticalValue.toFixed(1)} + Tempo: ${tempoValue.toFixed(1)} = ${evaluation.toFixed(1)}`);
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

  // Evaluate strategic progress toward end-game bonuses (Expert Strategy Implementation)
  private evaluateStrategicProgress(board: PlayerBoard): number {
    let strategicValue = 0;
    const tileSupply = this.analyzeTileSupply();
    const gamePhase = this.getGamePhase();

    // EXPERT STRATEGY 1 & 7: Prioritize top 3 rows, avoid 5th row late game
    for (let row = 0; row < 5; row++) {
      const tilesInRow = board.wall[row].length;
      const missingTiles = this.getMissingTilesForRow(board, row);
      const isFeasible = this.isObjectiveFeasible(missingTiles, tileSupply);

      // Expert Strategy: Row priority weighting
      let rowPriorityMultiplier = 1.0;
      if (row <= 2) {
        rowPriorityMultiplier = 1.5; // Higher priority for top 3 rows
      } else if (row === 4 && gamePhase === 'late') {
        rowPriorityMultiplier = 0.3; // Avoid 5th row in late game
      } else if (row === 4 && gamePhase === 'endgame') {
        rowPriorityMultiplier = 0.1; // Strongly avoid 5th row in endgame
      }

      if (tilesInRow === 5) {
        strategicValue += 2 * rowPriorityMultiplier; // Full bonus for completed row
      } else if (tilesInRow === 4 && isFeasible) {
        strategicValue += 1.5 * rowPriorityMultiplier; // High value for near-complete row (if possible)
      } else if (tilesInRow === 4 && !isFeasible) {
        strategicValue -= 1 * rowPriorityMultiplier; // Penalty for impossible near-complete row
      } else if (tilesInRow === 3 && isFeasible) {
        strategicValue += 0.8 * rowPriorityMultiplier; // Good progress (if possible)
      } else if (tilesInRow === 3 && !isFeasible) {
        strategicValue -= 0.5 * rowPriorityMultiplier; // Penalty for impossible progress
      } else if (tilesInRow === 2 && isFeasible) {
        strategicValue += 0.3 * rowPriorityMultiplier; // Some progress (if possible)
      }
    }

    // EXPERT STRATEGY 3: Focus on column completion, prioritize central columns
    for (let col = 0; col < 5; col++) {
      let tilesInColumn = 0;
      const missingTiles: Tile[] = [];

      for (let row = 0; row < 5; row++) {
        const expectedTile = PlayerBoard.getWallTile(row, col);
        if (board.wall[row].includes(expectedTile)) {
          tilesInColumn++;
        } else {
          missingTiles.push(expectedTile);
        }
      }

      const isFeasible = this.isObjectiveFeasible(missingTiles, tileSupply);

      // Expert Strategy: Central column priority (columns 1, 2, 3 are more valuable)
      let columnPriorityMultiplier = 1.0;
      if (col >= 1 && col <= 3) {
        columnPriorityMultiplier = 1.4; // Higher priority for central columns
      }

      if (tilesInColumn === 5) {
        strategicValue += 7 * columnPriorityMultiplier; // Full bonus for completed column
      } else if (tilesInColumn === 4 && isFeasible) {
        strategicValue += 5 * columnPriorityMultiplier; // Very high value for near-complete column (if possible)
      } else if (tilesInColumn === 4 && !isFeasible) {
        strategicValue -= 2 * columnPriorityMultiplier; // Penalty for impossible near-complete column
      } else if (tilesInColumn === 3 && isFeasible) {
        strategicValue += 3 * columnPriorityMultiplier; // Good progress (if possible)
      } else if (tilesInColumn === 3 && !isFeasible) {
        strategicValue -= 1 * columnPriorityMultiplier; // Penalty for impossible progress
      } else if (tilesInColumn === 2 && isFeasible) {
        strategicValue += 1 * columnPriorityMultiplier; // Some progress (if possible)
      }
    }

    // EXPERT STRATEGY 6: Be cautious with color bonuses - reduce priority significantly
    const colorCounts = this.getColorCounts(board);
    const regularTiles = [Tile.Red, Tile.Blue, Tile.Yellow, Tile.Black, Tile.White];

    for (const tile of regularTiles) {
      const count = colorCounts.get(tile) || 0;
      const supply = tileSupply.get(tile);
      const needed = 5 - count;
      const isFeasible = supply ? supply.totalRemaining >= needed : false;

      // Expert Strategy: Significantly reduce color bonus priority (risky and unreliable)
      const colorCautionMultiplier = 0.4; // Much lower priority than expert advice suggests

      if (count === 5) {
        strategicValue += 10 * colorCautionMultiplier; // Full bonus for completed color (reduced)
      } else if (count === 4 && isFeasible) {
        strategicValue += 2; // Reduced value - only pursue if very close and feasible
      } else if (count === 4 && !isFeasible) {
        strategicValue -= 5; // Heavy penalty for impossible near-complete color
      } else if (count === 3 && isFeasible) {
        strategicValue += 1; // Minimal value - don't actively pursue
      } else if (count === 3 && !isFeasible) {
        strategicValue -= 2; // Penalty for impossible progress
      } else if (count === 2) {
        // Don't value early color progress at all - too risky
        strategicValue += 0;
      }
    }

    // 7. Line completion potential (encourage filling useful lines) - with feasibility check
    for (let i = 0; i < 5; i++) {
      const line = board.lines[i];
      const required = i + 1;
      const filled = line.length;

      if (filled > 0 && filled < required) {
        const neededTile = line[0];
        const needed = required - filled;
        const supply = tileSupply.get(neededTile);
        const isFeasible = supply ? supply.totalRemaining >= needed : false;

        if (isFeasible) {
          // Value based on how close to completion and strategic importance
          const completionRatio = filled / required;
          const lineValue = (i + 1) * 0.5; // Higher lines worth more
          strategicValue += completionRatio * lineValue;
        } else {
          // Penalty for impossible lines
          strategicValue -= (filled * 0.5); // Penalty proportional to wasted effort
        }
      }
    }

    return strategicValue;
  }

  // Get multiplier based on game phase - strategic bonuses matter more later in game
  private getGamePhaseMultiplier(): number {
    const gamePhase = this.getGamePhase();

    // Adaptive multiplier based on realistic game progression
    switch (gamePhase) {
      case 'early':
        return 0.3; // Focus on immediate scoring and positioning (0-2 tiles per player)
      case 'mid':
        return 0.8; // Start considering strategic bonuses (3-7 tiles per player)
      case 'late':
        return 1.8; // Heavily weight strategic bonuses (8+ tiles per player)
      case 'endgame':
        return 3.0; // Maximum strategic focus when end game is triggered
      default:
        return 1.0;
    }
  }

  // EXPERT STRATEGY 5: Enhanced defensive opportunities (monitor and block opponents)
  private evaluateDefensiveOpportunities(playerIndex: number): number {
    let defensiveValue = 0;
    const opponentBoards = this.playerBoards.filter((_, i) => i !== playerIndex);
    const tileSupply = this.analyzeTileSupply();
    const gamePhase = this.getGamePhase();

    // Expert Strategy: Increase defensive focus based on game phase
    let defensiveMultiplier = 1.0;
    if (gamePhase === 'late' || gamePhase === 'endgame') {
      defensiveMultiplier = 1.5; // More aggressive blocking late game
    }

    // 1. Enhanced blocking of opponent's near-complete objectives
    for (const opponentBoard of opponentBoards) {
      // PRIORITY 1: Block near-complete rows (especially top 3 rows)
      for (let row = 0; row < 5; row++) {
        if (opponentBoard.wall[row].length === 4) {
          const missingTile = this.findMissingTileInRow(opponentBoard, row);
          if (missingTile && this.isTileAvailableInFactories(missingTile)) {
            const supply = tileSupply.get(missingTile);
            let blockValue = 3; // Base blocking value

            // Expert Strategy: Higher priority for blocking top 3 rows
            if (row <= 2) {
              blockValue *= 1.5;
            }

            if (supply && supply.totalRemaining <= 3) {
              blockValue *= 2.5; // Critical blocking when tile is very scarce
            }

            defensiveValue += blockValue * defensiveMultiplier;
          }
        }
      }

      // PRIORITY 2: Block near-complete columns (especially central columns)
      for (let col = 0; col < 5; col++) {
        let tilesInColumn = 0;
        for (let row = 0; row < 5; row++) {
          if (opponentBoard.wall[row].includes(PlayerBoard.getWallTile(row, col))) {
            tilesInColumn++;
          }
        }
        if (tilesInColumn === 4) {
          const missingTile = this.findMissingTileInColumn(opponentBoard, col);
          if (missingTile && this.isTileAvailableInFactories(missingTile)) {
            const supply = tileSupply.get(missingTile);
            let blockValue = 5; // Base blocking value for columns

            // Expert Strategy: Higher priority for blocking central columns
            if (col >= 1 && col <= 3) {
              blockValue *= 1.4;
            }

            if (supply && supply.totalRemaining <= 3) {
              blockValue *= 2.5; // Critical blocking when tile is very scarce
            }

            defensiveValue += blockValue * defensiveMultiplier;
          }
        }
      }

      // PRIORITY 3: Block near-complete colors (but lower priority per expert advice)
      const colorCounts = this.getColorCounts(opponentBoard);
      for (const [tile, count] of colorCounts.entries()) {
        if (count === 4 && this.isTileAvailableInFactories(tile)) {
          const supply = tileSupply.get(tile);
          // Expert Strategy: Reduced priority for color blocking (colors are risky/rare)
          let blockValue = 4; // Reduced from 7 - colors are less reliable

          if (supply && supply.totalRemaining <= 2) {
            blockValue = 8; // Reduced from 15 - still block if critical but lower priority
          }

          defensiveValue += blockValue * defensiveMultiplier;
        }
      }

      // 4. Check for opponent's impossible objectives and reduce their value
      for (let row = 0; row < 5; row++) {
        const missingTiles = this.getMissingTilesForRow(opponentBoard, row);
        if (!this.isObjectiveFeasible(missingTiles, tileSupply)) {
          // Opponent is pursuing impossible row - less need to block
          defensiveValue -= 1;
        }
      }
    }

    // 2. Evaluate forcing opponents to take floor penalties
    const availableTiles = this.getAvailableTileTypes();
    for (const opponentBoard of opponentBoards) {
      for (const tile of availableTiles) {
        if (this.wouldForceTileToFloor(opponentBoard, tile)) {
          defensiveValue += 2; // Bonus for forcing opponent penalties
        }
      }
    }

    // 3. Tile scarcity considerations (now includes total supply analysis)
    defensiveValue += this.evaluateTileScarcity(playerIndex);

    return defensiveValue;
  }

  // Evaluate tactical opportunities (tile scarcity, forcing moves, etc.)
  private evaluateTacticalOpportunities(playerIndex: number): number {
    let tacticalValue = 0;
    const playerBoard = this.playerBoards[playerIndex];

    // 1. Evaluate tile efficiency (taking exactly what you need)
    for (const move of this.availableMoves) {
      if (move.lineIndex >= 0) {
        const line = playerBoard.lines[move.lineIndex];
        const needed = (move.lineIndex + 1) - line.length;
        const available = this.countTilesInSource(move.factoryIndex, move.tile);

        // Bonus for taking exactly what you need (no waste)
        if (available === needed) {
          tacticalValue += 1.5;
        }
        // Penalty for significant waste
        else if (available > needed + 2) {
          tacticalValue -= 0.5;
        }
      }
    }

    // 2. Evaluate "hate drafting" opportunities
    const opponentBoards = this.playerBoards.filter((_, i) => i !== playerIndex);
    for (const opponentBoard of opponentBoards) {
      for (const move of this.availableMoves) {
        if (this.isHighValueTileForOpponent(opponentBoard, move.tile)) {
          tacticalValue += 1; // Bonus for denying valuable tiles to opponents
        }
      }
    }

    // 3. End-game timing considerations
    if (this.isNearEndGame()) {
      // Prioritize completing your own objectives over blocking
      tacticalValue *= 0.7;
    }

    return tacticalValue;
  }

  // EXPERT STRATEGY 4 & 8: Enhanced tempo advantage with strategic first player token value
  private evaluateTempoAdvantage(playerIndex: number): number {
    let tempoValue = 0;
    const gamePhase = this.getGamePhase();

    // EXPERT STRATEGY 4: Enhanced first player token evaluation
    if (this.center.includes(Tile.FirstPlayer)) {
      const isPlayerBehind = this.isPlayerBehind(playerIndex);
      let tokenValue = 1; // Base value

      // Higher value in early rounds (expert advice)
      if (gamePhase === 'early') {
        tokenValue = 3; // Very valuable early game
      } else if (gamePhase === 'mid') {
        tokenValue = 2; // Good value mid game
      } else {
        tokenValue = 1; // Standard value late game
      }

      // Adjust based on position
      if (isPlayerBehind) {
        tokenValue += 1; // Extra value when behind
      }

      tempoValue += tokenValue;
    }

    // 2. Round timing considerations
    const tilesRemaining = this.getTotalTilesRemaining();
    if (tilesRemaining <= 5) {
      // Near end of round - prioritize completing lines
      tempoValue += this.evaluateLineCompletionUrgency(playerIndex);
    }

    // 3. Factory control (taking from factories vs center)
    const factoryMoves = this.availableMoves.filter(m => m.factoryIndex >= 0).length;
    const centerMoves = this.availableMoves.filter(m => m.factoryIndex === -1).length;

    if (factoryMoves > centerMoves) {
      tempoValue += 0.5; // Slight bonus for having factory options
    }

    // EXPERT STRATEGY 8: Evaluate strategic discarding opportunities
    tempoValue += this.evaluateStrategicDiscarding(playerIndex);

    return tempoValue;
  }

  // EXPERT STRATEGY 2: Enhanced adjacency bonus calculation with central column preference
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

    // Expert Strategy: Bonus for central column placement (better adjacency potential)
    if (col >= 1 && col <= 3) {
      score += 0.5; // Small bonus for central placement
    }

    return score;
  }

  // Helper methods for enhanced AI evaluation

  private findMissingTileInRow(board: PlayerBoard, row: number): Tile | null {
    const wallPattern = PlayerBoard.WALL_PATTERN[row];
    for (let col = 0; col < 5; col++) {
      const expectedTile = wallPattern[col];
      if (!board.wall[row].includes(expectedTile)) {
        return expectedTile;
      }
    }
    return null;
  }

  private findMissingTileInColumn(board: PlayerBoard, col: number): Tile | null {
    for (let row = 0; row < 5; row++) {
      const expectedTile = PlayerBoard.getWallTile(row, col);
      if (!board.wall[row].includes(expectedTile)) {
        return expectedTile;
      }
    }
    return null;
  }

  private isTileAvailableInFactories(tile: Tile): boolean {
    // Check factories
    for (const factory of this.factories) {
      if (factory.includes(tile)) return true;
    }
    // Check center
    return this.center.includes(tile);
  }

  private getColorCounts(board: PlayerBoard): Map<Tile, number> {
    const colorCounts = new Map<Tile, number>();
    const regularTiles = [Tile.Red, Tile.Blue, Tile.Yellow, Tile.Black, Tile.White];

    for (const row of board.wall) {
      for (const tile of row) {
        if (tile !== null && regularTiles.includes(tile)) {
          colorCounts.set(tile, (colorCounts.get(tile) || 0) + 1);
        }
      }
    }
    return colorCounts;
  }

  private getAvailableTileTypes(): Tile[] {
    const tileSet = new Set<Tile>();

    // Add tiles from factories
    for (const factory of this.factories) {
      for (const tile of factory) {
        if (tile !== Tile.FirstPlayer) {
          tileSet.add(tile);
        }
      }
    }

    // Add tiles from center
    for (const tile of this.center) {
      if (tile !== Tile.FirstPlayer) {
        tileSet.add(tile);
      }
    }

    return Array.from(tileSet);
  }

  private wouldForceTileToFloor(board: PlayerBoard, tile: Tile): boolean {
    // Check if the tile can be placed in any pattern line
    for (let i = 0; i < 5; i++) {
      const line = board.lines[i];

      // Can place if line is empty or contains same tile and has space
      if (line.length === 0 || (line[0] === tile && line.length < i + 1)) {
        // Also check if wall position is available
        const wallCol = PlayerBoard.WALL_PATTERN[i].indexOf(tile);
        if (wallCol !== -1 && !board.wall[i].includes(tile)) {
          return false; // Can be placed, won't go to floor
        }
      }
    }
    return true; // Would be forced to floor
  }

  private evaluateTileScarcity(playerIndex: number): number {
    let scarcityValue = 0;
    const playerBoard = this.playerBoards[playerIndex];

    // Get comprehensive tile supply analysis
    const tileSupply = this.analyzeTileSupply();

    // Evaluate scarcity for tiles we need
    for (let i = 0; i < 5; i++) {
      const line = playerBoard.lines[i];
      if (line.length > 0 && line.length < i + 1) {
        const neededTile = line[0];
        const needed = (i + 1) - line.length;
        const supply = tileSupply.get(neededTile);

        if (!supply) continue;

        // Critical: Check if there are enough tiles remaining to complete this line
        if (supply.totalRemaining < needed) {
          scarcityValue -= 5; // Heavy penalty for impossible lines
        } else if (supply.availableThisRound < needed && supply.totalRemaining < needed + 2) {
          scarcityValue += 3; // High value for critically scarce tiles
        } else if (supply.availableThisRound <= 2) {
          scarcityValue += 2; // High value for immediately scarce tiles
        } else if (supply.availableThisRound <= 4) {
          scarcityValue += 1; // Medium value for moderately scarce tiles
        }

        // Bonus for securing tiles when total supply is getting low
        if (supply.totalRemaining <= 5) {
          scarcityValue += 2; // Urgent to secure when few remain in game
        }
      }
    }

    return scarcityValue;
  }

  private countTilesInSource(factoryIndex: number, tile: Tile): number {
    if (factoryIndex === -1) {
      // Count in center
      return this.center.filter(t => t === tile).length;
    } else {
      // Count in specific factory
      return this.factories[factoryIndex].filter(t => t === tile).length;
    }
  }

  private isHighValueTileForOpponent(opponentBoard: PlayerBoard, tile: Tile): boolean {
    // Check if this tile would help opponent complete lines or strategic goals
    for (let i = 0; i < 5; i++) {
      const line = opponentBoard.lines[i];

      // High value if it would complete a line
      if (line.length > 0 && line[0] === tile && line.length === i) {
        return true;
      }

      // High value if it would start a valuable line
      if (line.length === 0) {
        const wallCol = PlayerBoard.WALL_PATTERN[i].indexOf(tile);
        if (wallCol !== -1 && !opponentBoard.wall[i].includes(tile)) {
          // Check if this would help complete row/column/color
          if (this.wouldHelpCompleteObjective(opponentBoard, i, wallCol, tile)) {
            return true;
          }
        }
      }
    }

    return false;
  }

  private wouldHelpCompleteObjective(board: PlayerBoard, row: number, col: number, tile: Tile): boolean {
    // Check if placing this tile would significantly advance row/column/color completion

    // Row completion check
    if (board.wall[row].length >= 3) return true;

    // Column completion check
    let columnCount = 0;
    for (let r = 0; r < 5; r++) {
      if (board.wall[r].includes(PlayerBoard.getWallTile(r, col))) {
        columnCount++;
      }
    }
    if (columnCount >= 3) return true;

    // Color completion check
    const colorCounts = this.getColorCounts(board);
    const currentCount = colorCounts.get(tile) || 0;
    if (currentCount >= 3) return true;

    return false;
  }

  private isNearEndGame(): boolean {
    // Check if any player has completed a row (triggers end game)
    for (const board of this.playerBoards) {
      for (const row of board.wall) {
        if (row.length === 5) return true;
      }
    }

    // Check if tile bag is getting low
    return this.tilebag.length < 20;
  }

  private isPlayerBehind(playerIndex: number): boolean {
    const playerScore = this.playerBoards[playerIndex].score;
    const maxOpponentScore = Math.max(...this.playerBoards
      .filter((_, i) => i !== playerIndex)
      .map(board => board.score));

    return playerScore < maxOpponentScore - 5; // Behind by more than 5 points
  }

  private getTotalTilesRemaining(): number {
    let total = 0;
    for (const factory of this.factories) {
      total += factory.length;
    }
    total += this.center.filter(t => t !== Tile.FirstPlayer).length;
    return total;
  }

  private evaluateLineCompletionUrgency(playerIndex: number): number {
    let urgencyValue = 0;
    const playerBoard = this.playerBoards[playerIndex];

    for (let i = 0; i < 5; i++) {
      const line = playerBoard.lines[i];
      const needed = (i + 1) - line.length;

      if (line.length > 0 && needed <= 2) {
        // Urgent to complete lines that are close to completion
        urgencyValue += (3 - needed); // More urgent as fewer tiles needed
      }
    }

    return urgencyValue;
  }

  // Comprehensive tile supply analysis - tracks total game supply vs usage
  private analyzeTileSupply(): Map<Tile, { totalRemaining: number; availableThisRound: number; usedByOpponents: number; usedByPlayer: number }> {
    const regularTiles = [Tile.Red, Tile.Blue, Tile.Yellow, Tile.Black, Tile.White];
    const supply = new Map();

    for (const tile of regularTiles) {
      // Start with total game supply (20 of each color)
      let totalRemaining = 20;
      let usedByPlayer = 0;
      let usedByOpponents = 0;

      // Count tiles used by all players (on walls and in pattern lines)
      for (let playerIndex = 0; playerIndex < this.playerBoards.length; playerIndex++) {
        const board = this.playerBoards[playerIndex];
        let playerUsage = 0;

        // Count tiles on wall
        for (const row of board.wall) {
          for (const wallTile of row) {
            if (wallTile === tile) {
              playerUsage++;
              totalRemaining--;
            }
          }
        }

        // Count tiles in pattern lines
        for (const line of board.lines) {
          for (const lineTile of line) {
            if (lineTile === tile) {
              playerUsage++;
              totalRemaining--;
            }
          }
        }

        // Count tiles on floor (they're discarded)
        for (const floorTile of board.floor) {
          if (floorTile === tile) {
            totalRemaining--;
          }
        }

        // Track usage by player vs opponents
        if (playerIndex === this.currentPlayer) {
          usedByPlayer = playerUsage;
        } else {
          usedByOpponents += playerUsage;
        }
      }

      // Count tiles in tile bag (not yet drawn)
      for (const bagTile of this.tilebag) {
        if (bagTile === tile) {
          // These are still in the bag, so they're part of totalRemaining
          // (already counted in the 20 starting tiles)
        }
      }

      // Count tiles available this round (factories + center)
      let availableThisRound = 0;
      for (const factory of this.factories) {
        for (const factoryTile of factory) {
          if (factoryTile === tile) {
            availableThisRound++;
          }
        }
      }

      for (const centerTile of this.center) {
        if (centerTile === tile) {
          availableThisRound++;
        }
      }

      supply.set(tile, {
        totalRemaining,
        availableThisRound,
        usedByOpponents,
        usedByPlayer
      });
    }

    return supply;
  }

  // Get missing tiles needed to complete a specific row
  private getMissingTilesForRow(board: PlayerBoard, row: number): Tile[] {
    const missingTiles: Tile[] = [];
    const wallPattern = PlayerBoard.WALL_PATTERN[row];

    for (let col = 0; col < 5; col++) {
      const expectedTile = wallPattern[col];
      if (!board.wall[row].includes(expectedTile)) {
        missingTiles.push(expectedTile);
      }
    }

    return missingTiles;
  }

  // Check if an objective (row/column/color completion) is feasible given tile supply
  private isObjectiveFeasible(neededTiles: Tile[], tileSupply: Map<Tile, any>): boolean {
    // Count how many of each tile type we need
    const tileCounts = new Map<Tile, number>();

    for (const tile of neededTiles) {
      tileCounts.set(tile, (tileCounts.get(tile) || 0) + 1);
    }

    // Check if we have enough of each tile type remaining in the game
    for (const [tile, needed] of tileCounts.entries()) {
      const supply = tileSupply.get(tile);
      if (!supply || supply.totalRemaining < needed) {
        return false; // Not enough tiles remaining
      }
    }

    return true; // All needed tiles are available
  }

  // Get current game phase for expert strategy decisions
  private getGamePhase(): 'early' | 'mid' | 'late' | 'endgame' {
    // Check if end game is triggered (any completed row)
    const endGameTriggered = this.isNearEndGame();
    if (endGameTriggered) return 'endgame';

    // Count total tiles placed across all players
    let totalTilesPlaced = 0;
    for (const board of this.playerBoards) {
      for (const row of board.wall) {
        totalTilesPlaced += row.filter(tile => tile !== null).length;
      }
    }

    // Realistic game progression based on typical Azul games
    // Most games end around 8-15 tiles per player (not 25)
    const avgTilesPerPlayer = totalTilesPlaced / this.numPlayers;

    // More realistic thresholds based on actual Azul gameplay
    if (avgTilesPerPlayer < 3) return 'early';      // 0-2 tiles per player
    if (avgTilesPerPlayer < 8) return 'mid';        // 3-7 tiles per player
    return 'late';                                  // 8+ tiles per player (approaching endgame)
  }

  // EXPERT STRATEGY 8: Evaluate strategic discarding opportunities
  private evaluateStrategicDiscarding(playerIndex: number): number {
    let discardingValue = 0;
    const opponentBoards = this.playerBoards.filter((_, i) => i !== playerIndex);

    // Evaluate moves that force opponents to take more negative points
    for (const move of this.availableMoves) {
      if (move.factoryIndex >= 0) {
        // Taking from factory puts remaining tiles in center
        const factory = this.factories[move.factoryIndex];
        const remainingTiles = factory.filter(t => t !== move.tile);

        // Check if remaining tiles would be problematic for opponents
        for (const opponentBoard of opponentBoards) {
          for (const tile of remainingTiles) {
            if (this.wouldForceTileToFloor(opponentBoard, tile)) {
              discardingValue += 1.5; // Bonus for forcing opponent floor penalties
            }
          }
        }
      }
    }

    return discardingValue;
  }
}

/**
 * GameState for web app - includes full game simulation capabilities
 */
export class WebAppGameState extends BaseGameState {

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

  // Implementation of abstract method
  protected onNewRound(): void {
    this.newRound();
  }

  // Create a deep copy of the game state for AI simulation
  clone(): WebAppGameState {
    const cloned = new WebAppGameState(this.numPlayers);

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
}

/**
 * GameState for BGA extension - loads state from BGA data for AI analysis
 */
export class BGAGameState extends BaseGameState {

  // Load game state from BGA-like data structure
  loadFromBga(bgaData: {
    factories: string[][];
    center: string[];
    playerBoards: { lines: string[][]; wall: string[][]; floor: string[]; score: number }[];
    currentPlayer: number;
    round: number; // BGA might not provide round, default or calculate if necessary
  }): void {
    this.numPlayers = bgaData.playerBoards.length;
    this.round = bgaData.round !== undefined ? bgaData.round : 1; // Default round to 1 if not provided
    this.currentPlayer = bgaData.currentPlayer;
    this.phase = GamePhase.TileSelection; // Assuming BGA state is always during tile selection phase for AI
    this.gameOver = false; // Reset game over status
    this.firstPlayerIndex = 0; // Reset, will be determined by first player token

    // Initialize factories
    this.factories = bgaData.factories.map(factory =>
      factory.map(sTile => stringToTile(sTile)).filter(t => t !== null) as Tile[]
    );

    // Initialize center, carefully handling FirstPlayer token
    this.center = bgaData.center
        .map(sTile => stringToTile(sTile))
        .filter(t => t !== null) as Tile[];

    // Initialize player boards
    if (this.playerBoards.length !== this.numPlayers) {
        this.playerBoards = [];
        for (let i = 0; i < this.numPlayers; i++) {
            this.playerBoards.push(new PlayerBoard());
        }
    }

    let firstPlayerTokenFoundOnBoard = false;
    for (let i = 0; i < this.numPlayers; i++) {
      this.playerBoards[i].loadState(bgaData.playerBoards[i]);
      // Check if this player has the first player token on their floor
      if (this.playerBoards[i].floor.includes(Tile.FirstPlayer)) {
        this.firstPlayerIndex = i;
        firstPlayerTokenFoundOnBoard = true;
        // Remove from center if it was also there (BGA might be inconsistent)
        this.center = this.center.filter(t => t !== Tile.FirstPlayer);
      }
    }

    // If first player token wasn't on a board, check if it's in the center
    if (!firstPlayerTokenFoundOnBoard && !this.center.includes(Tile.FirstPlayer)) {
        // If it's NOWHERE, but it should be SOMEWHERE in TileSelection phase (unless all tiles taken from center already)
        // This logic might need adjustment based on exact BGA state representation when center is emptied.
        // For now, if no token and center is empty of regular tiles, assume previous round's first player keeps it.
        // If center still has tiles but no token, it implies it was taken.
        // This is tricky without knowing BGA's exact first player token rules post-center-clearing.
        // A simple assumption: if not on a board and not in center, it's not in play for *this specific turn's start*.
        // The firstPlayerIndex would then be determined by who *takes* it from the center.
    } else if (this.center.includes(Tile.FirstPlayer) && firstPlayerTokenFoundOnBoard) {
        // If on a board AND in center, prioritize board, remove from center.
        this.center = this.center.filter(t => t !== Tile.FirstPlayer);
    }

    // If no player has the token yet and it's not in the center, this implies it hasn't been picked up.
    // The `firstPlayerIndex` will be set when a player takes it from the center via `playMove`.
    // If it IS in the center, no player is `firstPlayerIndex` yet for *next* round until it's taken.
    // If it IS on a player's board, that player is `firstPlayerIndex` for *next* round.

    // Ensure a valid currentPlayer (e.g. if BGA sends an out-of-bounds index)
    this.currentPlayer = Math.max(0, Math.min(this.numPlayers - 1, this.currentPlayer));

    this.getMoves(); // Crucially, generate moves for the loaded state.
    console.log('GameState loaded from BGA data:', this);
  }

  // Implementation of abstract method - BGA extension doesn't need to handle new rounds
  protected onNewRound(): void {
    // BGA extension only analyzes current position, doesn't simulate full rounds
    // This should not be called in normal BGA extension usage
    console.warn('BGAGameState.onNewRound() called - this should not happen in normal extension usage');
  }

  // Create a deep copy of the game state for AI simulation
  clone(): BGAGameState {
    const cloned = new BGAGameState(this.numPlayers);

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
}

// Export the original GameState class name for backward compatibility
// Web app should use WebAppGameState, extension should use BGAGameState
export const GameState = WebAppGameState;
