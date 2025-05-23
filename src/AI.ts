import { GameState } from './GameState.js';
import { Move, SearchResult } from './types.js';

export class AzulAI {
  private playerIndex: number;
  private maxThinkingTime: number; // in milliseconds
  private nodesEvaluated: number = 0;

  constructor(playerIndex: number, maxThinkingTime: number = 1000) {
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

    // Order moves based on previous search results
    let orderedMoves = [...gameState.availableMoves];
    if (moveOrdering && depth === 1) {
      // Sort based on previous iteration's results
      orderedMoves.sort((a, b) => {
        const aIndex = moveOrdering.findIndex(m => this.movesEqual(m, a));
        const bIndex = moveOrdering.findIndex(m => this.movesEqual(m, b));
        if (aIndex === -1 && bIndex === -1) return 0;
        if (aIndex === -1) return 1;
        if (bIndex === -1) return -1;
        return maximizingPlayer ? aIndex - bIndex : bIndex - aIndex;
      });
    }

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
} 