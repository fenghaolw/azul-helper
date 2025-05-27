import { GameState } from './GameState.js';
import { Move, SearchResult, Tile } from './types.js';

export class AzulAI {
  private playerIndex: number;
  private maxThinkingTime: number; // in milliseconds
  private nodesEvaluated: number = 0;

  constructor(playerIndex: number, maxThinkingTime: number = 2000) {
    this.playerIndex = playerIndex;
    this.maxThinkingTime = maxThinkingTime;
  }

  // Get the best move using iterative deepening with alpha-beta pruning
  getBestMove(gameState: GameState): SearchResult {
    const startTime = Date.now();
    this.nodesEvaluated = 0;

    if (gameState.availableMoves.length === 0) {
      throw new Error('No available moves');
    }

    if (gameState.availableMoves.length === 1) {
      return {
        move: gameState.availableMoves[0],
        value: 0,
        depth: 1,
        nodesEvaluated: 1
      };
    }

    let bestMove = gameState.availableMoves[0];
    let bestValue = -Infinity;
    let searchDepth = 1;
    let previousMoveOrdering: Move[] = [...gameState.availableMoves];

    // Iterative deepening
    while (Date.now() - startTime < this.maxThinkingTime) {
      try {
        const result = this.alphaBetaSearch(
          gameState, 
          searchDepth, 
          -Infinity, 
          Infinity, 
          true,
          startTime,
          previousMoveOrdering
        );

        if (result) {
          bestMove = result.move;
          bestValue = result.value;
          previousMoveOrdering = result.moveOrdering || [...gameState.availableMoves];
        }

        // If we found a winning move, no need to search deeper
        if (bestValue >= 1000) {
          break;
        }

        searchDepth++;

        // If time is running out, break
        if (Date.now() - startTime > this.maxThinkingTime * 0.8) {
          break;
        }

      } catch (timeoutError) {
        // Time limit exceeded during search
        break;
      }
    }

    console.log(`AI Player ${this.playerIndex + 1} decision:`);
    console.log(`  Selected move: Factory ${bestMove.factoryIndex}, Tile ${bestMove.tile}, Line ${bestMove.lineIndex + 1}`);
    console.log(`  Evaluation: ${bestValue}, Depth: ${searchDepth - 1}, Nodes: ${this.nodesEvaluated}`);

    return {
      move: bestMove,
      value: bestValue,
      depth: searchDepth - 1,
      nodesEvaluated: this.nodesEvaluated
    };
  }

  // Alpha-beta search with move ordering
  private alphaBetaSearch(
    gameState: GameState, 
    depth: number, 
    alpha: number, 
    beta: number, 
    maximizingPlayer: boolean,
    startTime: number,
    moveOrdering?: Move[]
  ): { move: Move; value: number; moveOrdering?: Move[] } | null {

    // Check timeout
    if (Date.now() - startTime > this.maxThinkingTime) {
      throw new Error('Time limit exceeded');
    }

    this.nodesEvaluated++;

    // Terminal conditions
    if (depth === 0 || gameState.gameOver || gameState.availableMoves.length === 0) {
      const value = gameState.evaluatePosition(this.playerIndex);
      return { move: gameState.availableMoves[0] || { factoryIndex: 0, tile: 'red' as any, lineIndex: 0 }, value };
    }

    // Order moves based on heuristics and previous search results
    let orderedMoves = [...gameState.availableMoves];
    
    // Apply move ordering heuristics
    orderedMoves.sort((a, b) => {
      let scoreA = this.getMoveOrderingScore(gameState, a);
      let scoreB = this.getMoveOrderingScore(gameState, b);
      
      // Apply previous iteration ordering if available
      if (moveOrdering && depth === 1) {
        const aIndex = moveOrdering.findIndex(m => this.movesEqual(m, a));
        const bIndex = moveOrdering.findIndex(m => this.movesEqual(m, b));
        
        if (aIndex !== -1) scoreA += 100 - aIndex; // Boost previously good moves
        if (bIndex !== -1) scoreB += 100 - bIndex;
      }
      
      return maximizingPlayer ? scoreB - scoreA : scoreA - scoreB;
    });

    let bestMove = orderedMoves[0];
    let bestValue = maximizingPlayer ? -Infinity : Infinity;
    const newMoveOrdering: Move[] = [];

    for (const move of orderedMoves) {
      // Create new game state
      const newGameState = gameState.clone();
      
      // Play the move
      const roundEnded = newGameState.playMove(move);
      
      if (roundEnded) {
        // If round ended, evaluate final position
        const value = newGameState.evaluatePosition(this.playerIndex);
        if (maximizingPlayer) {
          if (value > bestValue) {
            bestValue = value;
            bestMove = move;
          }
          alpha = Math.max(alpha, value);
        } else {
          if (value < bestValue) {
            bestValue = value;
            bestMove = move;
          }
          beta = Math.min(beta, value);
        }
        newMoveOrdering.push(move);
      } else {
        // Continue search
        const result = this.alphaBetaSearch(
          newGameState, 
          depth - 1, 
          alpha, 
          beta, 
          newGameState.currentPlayer === this.playerIndex,
          startTime
        );

        if (result) {
          const value = result.value;
          
          if (maximizingPlayer) {
            if (value > bestValue) {
              bestValue = value;
              bestMove = move;
            }
            alpha = Math.max(alpha, value);
          } else {
            if (value < bestValue) {
              bestValue = value;
              bestMove = move;
            }
            beta = Math.min(beta, value);
          }
          
          newMoveOrdering.push(move);
        }
      }

      // Alpha-beta pruning
      if (beta <= alpha) {
        break;
      }
    }

    return { 
      move: bestMove, 
      value: bestValue, 
      moveOrdering: depth === 1 ? newMoveOrdering : undefined 
    };
  }

  // Helper method to compare moves
  private movesEqual(move1: Move, move2: Move): boolean {
    return move1.factoryIndex === move2.factoryIndex &&
           move1.tile === move2.tile &&
           move1.lineIndex === move2.lineIndex;
  }

  // Get move with simple heuristic (for testing/comparison)
  getSimpleMove(gameState: GameState): Move {
    if (gameState.availableMoves.length === 0) {
      throw new Error('No available moves');
    }

    // Simple strategy: prefer moves that complete lines, avoid floor
    let bestMove = gameState.availableMoves[0];
    let bestScore = -Infinity;

    for (const move of gameState.availableMoves) {
      let score = 0;

      // Avoid floor moves
      if (move.lineIndex === -1) {
        score -= 10;
      } else {
        const currentBoard = gameState.playerBoards[this.playerIndex];
        const line = currentBoard.lines[move.lineIndex];
        const requiredTiles = move.lineIndex + 1;
        
        // Prefer moves that complete lines
        if (line.length === requiredTiles - 1) {
          score += 20;
        } else if (line.length === requiredTiles - 2) {
          score += 10;
        }

        // Prefer moves that start new lines with more potential
        if (line.length === 0) {
          score += move.lineIndex * 2; // Higher lines are worth more
        }
      }

      // Prefer taking from factories over center (except for first player token)
      if (move.factoryIndex >= 0) {
        score += 5;
      }

      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }
    }

    return bestMove;
  }

  // Utility method to get AI thinking stats
  getStats(): { nodesEvaluated: number } {
    return { nodesEvaluated: this.nodesEvaluated };
  }

  // Reset stats
  resetStats(): void {
    this.nodesEvaluated = 0;
  }

  // Expert Strategy Enhanced: Heuristic scoring for move ordering (better moves searched first)
  private getMoveOrderingScore(gameState: GameState, move: Move): number {
    let score = 0;
    const playerBoard = gameState.playerBoards[this.playerIndex];
    // const numPlayers = gameState.numPlayers; // Available if needed for scaling

    // Standard Azul wall pattern. Ensure these color strings match your game's representation.
    const WALL_PATTERN: any[][] = [
      ['blue', 'yellow', 'red', 'black', 'teal'], // Row 0
      ['teal', 'blue', 'yellow', 'red', 'black'], // Row 1
      ['black', 'teal', 'blue', 'yellow', 'red'], // Row 2
      ['red', 'black', 'teal', 'blue', 'yellow'], // Row 3
      ['yellow', 'red', 'black', 'teal', 'blue']  // Row 4
    ];

    const getWallPatternForRow = (lineIndex: number): any[] | undefined => {
        if (lineIndex >= 0 && lineIndex < WALL_PATTERN.length) {
            return WALL_PATTERN[lineIndex];
        }
        return undefined;
    };

    const getWallColumn = (lineIndex: number, tileColor: any): number => {
      const wallPatternRow = getWallPatternForRow(lineIndex);
      if (wallPatternRow) {
        return wallPatternRow.indexOf(tileColor);
      }
      return -1; 
    };

    // 1. Evaluate moves placing tiles on pattern lines
    if (move.lineIndex >= 0) {
      const line = playerBoard.lines[move.lineIndex];
      const currentLength = line.length;
      const capacity = move.lineIndex + 1;
      const needed = capacity - currentLength;
      const tilesAvailableFromSource = this.countAvailableTilesInSource(gameState, move.factoryIndex, move.tile);

      let canPlaceThisTile = false;
      if (currentLength === 0 || line[0] === move.tile) { // Line is empty or matches color
        const wallCol = getWallColumn(move.lineIndex, move.tile);
        // Check if wall spot for this tile in this row is empty
        if (wallCol !== -1 && playerBoard.wall[move.lineIndex][wallCol] === null) {
          canPlaceThisTile = true;
        }
      }

      if (canPlaceThisTile) {
        // A. Bonus for completing a line
        if (tilesAvailableFromSource >= needed) {
          score += 50; // Strong bonus for completion
          // A1. Bonus for exact completion (efficiency)
          if (tilesAvailableFromSource === needed) {
            score += 25;
          }
        } else {
          // A2. Bonus for partial filling
          score += (tilesAvailableFromSource / needed) * 20;
        }

        // B. Penalty for significant overfill for THIS line (leading to floor tiles)
        if (tilesAvailableFromSource > needed) {
          const excess = tilesAvailableFromSource - needed;
          score -= excess * 5; // Moderate penalty for overfill
        }

        // C. Row Priority (Favor top rows)
        if (move.lineIndex <= 1) { // Rows 1 and 2 (0-indexed)
          score += 15;
        } else if (move.lineIndex <= 2) { // Row 3
          score += 10;
        }
        // Penalize 5th row (lineIndex 4) especially in later game stages
        if (move.lineIndex === 4) {
          const gamePhase = this.getGamePhase(gameState); // Rely on AI's own getGamePhase
          if (gamePhase === 'late' || gamePhase === 'endgame') {
            score -= 20;
          } else {
            score -= 5; // Less penalty early/mid game
          }
        }

        // D. Contribution to Wall Bonuses (Simplified)
        // Check if placing this tile helps set up for a column or color bonus
        const wallColForMove = getWallColumn(move.lineIndex, move.tile);
        if (wallColForMove !== -1) {
            let tilesInWallColumn = 0;
            for(let r=0; r<5; r++) if(playerBoard.wall[r][wallColForMove] !== null) tilesInWallColumn++;
            if(tilesInWallColumn === 3) score += 5; // Getting close to column bonus
            if(tilesInWallColumn === 4) score += 10; // Very close

            let tilesOfColorOnWall = 0;
            for(let r_wall=0; r_wall<5; r_wall++) for(let c_wall=0; c_wall<5; c_wall++) if(playerBoard.wall[r_wall][c_wall] === move.tile) tilesOfColorOnWall++;
            if(tilesOfColorOnWall === 3) score += 5; // Getting close to color bonus
            if(tilesOfColorOnWall === 4) score += 10; // Very close
        }

      } else {
        // Heavy penalty if the move tries to place a tile in an invalid line (color mismatch, wall spot taken)
        score -= 100;
      }
    } else { 
      // 2. Evaluate floor moves (move.lineIndex === -1)
      const tilesAvailableFromSource = this.countAvailableTilesInSource(gameState, move.factoryIndex, move.tile);
      score -= 30 + (tilesAvailableFromSource * 3); // Base penalty + per-tile penalty
    }

    // 3. First Player Token
    if (move.factoryIndex === -1 && gameState.center.includes(Tile.FirstPlayer as any)) {
      // Check if *this move* takes the first player token
      // This occurs if move.tile is NOT FirstPlayer, but FirstPlayer is in center
      // OR if move.tile IS FirstPlayer (though AI usually wouldn't pick FP token as a tile type)
      let takesFirstPlayerToken = false;
      const firstPlayerTokenInCenter = gameState.center.includes(Tile.FirstPlayer as any);
      
      if (firstPlayerTokenInCenter) {
        // If we clear all of move.tile, and FP token is the only thing left (or also taken if move.tile IS FP)
        // This simplified check assumes taking *any* color from center when FP token is present means you *might* take it.
        // A more precise check would be in playMove, but for heuristic, this is an approximation.
        takesFirstPlayerToken = true; 
      }

      if (takesFirstPlayerToken) {
        const gamePhase = this.getGamePhase(gameState);
        if (gamePhase === 'early') score += 25;
        else if (gamePhase === 'mid') score += 15;
        else score += 10;
      }
    }

    // 4. Prefer taking from factories over center (if not for first player token)
    if (move.factoryIndex >= 0) {
      score += 5;
    }

    // 5. Tile Scarcity (Simplified: count tiles available *this round* for a needed color)
    if (move.lineIndex >= 0) {
        const line = playerBoard.lines[move.lineIndex];
        if (line.length > 0 && line.length < move.lineIndex + 1) { // Line started but not full
            const neededTile = line[0];
            if (move.tile === neededTile) { // If the current move is for the tile type we need for this line
                const totalAvailableThisRound = this.countTotalTilesInPlay(gameState, neededTile);
                if (totalAvailableThisRound <= 2) score += 15; // Very scarce this round
                else if (totalAvailableThisRound <= 4) score += 8; // Moderately scarce
            }
        }
    }

    // 6. Basic Defensive Move Check (Denying opponent an immediate obvious completion)
    const opponentIndices = gameState.playerBoards.map((_,i) => i).filter(i => i !== this.playerIndex);
    for (const opponentIndex of opponentIndices) {
        const opponentBoard = gameState.playerBoards[opponentIndex];
        // A. Opponent can complete a line of 4 with this move's tile type
        for (let i = 0; i < 5; i++) {
            const oppLine = opponentBoard.lines[i];
            if (oppLine.length === i && i < 4) { // Line has 4/5, 3/4, 2/3, 1/2 tiles and needs 1 more (i.e. oppLine.length === i)
                 // And the line is for the color of the tile in the current move we are evaluating
                if (oppLine[0] === move.tile) {
                    // And opponent can actually place it (wall spot open)
                    const wallCol = getWallColumn(i, move.tile);
                    if (wallCol !== -1 && opponentBoard.wall[i][wallCol] === null) {
                        const numTilesOfThisTypeAtSource = this.countAvailableTilesInSource(gameState, move.factoryIndex, move.tile);
                        if (numTilesOfThisTypeAtSource > 0) { // If taking this tile would actually remove it
                             score += 20; // Defensive bonus for blocking line completion
                        }
                    }
                }
            }
        }
    }

    return score;
  }

  // getGamePhase, countAvailableTilesInSource, countTotalTilesInPlay remain as previously defined in AI.ts
  // (Assuming they were accepted or are simple enough)
  // If getGamePhase was complex and in GameState.ts, AI.ts would need its own simplified version or GameState's passed.
  // For now, assume AI.ts has a functional getGamePhase.

  // Make sure helper methods from previous iteration are present if used by the new getMoveOrderingScore
  // From previous iteration, AI.ts had its own getGamePhase, countAvailableTilesInSource, countTotalTilesInPlay

  private getGamePhase(gameState: GameState): 'early' | 'mid' | 'late' | 'endgame' {
    // This is a simplified version for AI.ts, GameState.ts has the more detailed one.
    // We use this for the move ordering heuristic.
    const endGameTriggered = gameState.playerBoards.some(board =>
      board.wall.some(row => row.every(tile => tile !== null))
    );
    if (endGameTriggered) return 'endgame';

    let totalTilesPlaced = 0;
    for (const board of gameState.playerBoards) {
      for (const row of board.wall) {
        totalTilesPlaced += row.filter(tile => tile !== null).length;
      }
    }
    const avgTilesPerPlayer = totalTilesPlaced / gameState.numPlayers;
    if (avgTilesPerPlayer < 3) return 'early';
    if (avgTilesPerPlayer < 7) return 'mid'; // Adjusted mid threshold slightly for heuristic
    return 'late';
  }

  private countAvailableTilesInSource(gameState: GameState, factoryIndex: number, tile: any): number {
    if (factoryIndex === -1) { // Center
      return gameState.center.filter(t => t === tile).length;
    } else if (factoryIndex >= 0 && factoryIndex < gameState.factories.length) { // Specific factory
      return gameState.factories[factoryIndex].filter(t => t === tile).length;
    }
    return 0; 
  }

  private countTotalTilesInPlay(gameState: GameState, tile: any): number {
    let total = 0;
    gameState.factories.forEach(factory => total += factory.filter(t => t === tile).length);
    total += gameState.center.filter(t => t === tile).length;
    return total;
  }
} 