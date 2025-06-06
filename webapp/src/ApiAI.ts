import { BaseGameState } from './GameState.js';
import { Move, SearchResult } from './types.js';

interface ApiAIResponse {
  move: Move;
  stats: {
    nodesEvaluated: number;
    searchTime: number;
    agent_type: string;
    agent_name: string;
  };
  success: boolean;
  error?: string;
}

export class ApiAI {
  private playerIndex: number;
  private apiBaseUrl: string;
  private lastStats: {
    nodesEvaluated: number;
    agent_type?: string;
    agent_name?: string;
    searchTime?: number;
    lastMoveTime?: Date;
    totalMoves?: number;
    averageSearchTime?: number;
  } = { nodesEvaluated: 0, totalMoves: 0, averageSearchTime: 0 };
  private isConnected: boolean = false;

  constructor(playerIndex: number, apiBaseUrl?: string) {
    this.playerIndex = playerIndex;

    // If no specific URL provided, we'll auto-discover the server
    if (apiBaseUrl) {
      this.apiBaseUrl = apiBaseUrl;
      this.checkConnection();
    } else {
      this.apiBaseUrl = ''; // Will be set by auto-discovery
      this.autoDiscoverServer();
    }
  }

  private async autoDiscoverServer(): Promise<void> {
    console.log('🔍 Auto-discovering C++ AI server...');

    // Try ports 5000-5009
    for (let port = 5000; port < 5010; port++) {
      try {
        const testUrl = `http://localhost:${port}`;

        // Create a timeout promise
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error('Timeout')), 1000);
        });

        // Race between fetch and timeout
        const response = await Promise.race([
          fetch(`${testUrl}/health`, { method: 'GET' }),
          timeoutPromise
        ]);

        if (response.ok) {
          const data = await response.json();
          if (data.status === 'healthy') {
            this.apiBaseUrl = testUrl;
            this.isConnected = true;
            console.log(`✅ Found C++ AI server on port ${port}`);
            console.log(`   Agent: ${data.agent_type} (${data.agent_name})`);
            return;
          }
        }
      } catch (error) {
        // Port not responding, continue to next port
        continue;
      }
    }

    // No server found
    this.isConnected = false;
    this.apiBaseUrl = 'http://localhost:5000'; // Fallback
    console.warn('❌ No C++ AI server found on ports 5000-5009');
    console.warn('💡 Try starting the server with: python start.py');
  }

  private async checkConnection(): Promise<void> {
    try {
      if (!this.apiBaseUrl) {
        this.isConnected = false;
        return;
      }

      const response = await fetch(`${this.apiBaseUrl}/health`);
      if (response.ok) {
        const data = await response.json();
        this.isConnected = data.status === 'healthy';
        console.log('C++ AI connection status:', this.isConnected ? 'connected' : 'disconnected');
      } else {
        this.isConnected = false;
        console.warn('C++ AI server not responding');
      }
    } catch (error) {
      this.isConnected = false;
      console.warn('Failed to connect to C++ AI server:', error);
    }
  }

  async getBestMove(gameState: BaseGameState): Promise<SearchResult> {
    // If auto-discovery hasn't completed yet, wait a bit and retry
    if (!this.apiBaseUrl && !this.isConnected) {
      console.log('⏳ Waiting for server auto-discovery...');
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    if (!this.isConnected) {
      await this.checkConnection();
      if (!this.isConnected) {
        // Try auto-discovery one more time if we have no URL set
        if (!this.apiBaseUrl) {
          await this.autoDiscoverServer();
        }

        if (!this.isConnected) {
          throw new Error('C++ AI server is not available. Please start the API server with: python start.py');
        }
      }
    }

    if (gameState.availableMoves.length === 0) {
      throw new Error('No available moves');
    }

    if (gameState.availableMoves.length === 1) {
      // Update stats even for single move case
      const previousTotalMoves = this.lastStats.totalMoves || 0;
      this.lastStats = {
        ...this.lastStats,
        nodesEvaluated: 1,
        searchTime: 0.001, // Very fast since no search needed
        lastMoveTime: new Date(),
        totalMoves: previousTotalMoves + 1,
        averageSearchTime: this.lastStats.averageSearchTime || 0 // Keep existing average
      };

      return {
        move: gameState.availableMoves[0],
        value: 0,
        depth: 1,
        nodesEvaluated: 1
      };
    }

    try {
      const requestBody = {
        gameState: this.convertGameStateForAPI(gameState)
      };

      console.log(`AI Player ${this.playerIndex + 1} requesting move from C++ agent...`);

      const response = await fetch(`${this.apiBaseUrl}/agent/move`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const data: ApiAIResponse = await response.json();

      if (!data.success || data.error) {
        throw new Error(data.error || 'C++ AI failed to generate move');
      }

      // Update stats
      const previousTotalMoves = this.lastStats.totalMoves || 0;
      const previousAverage = this.lastStats.averageSearchTime || 0;
      const newTotalMoves = previousTotalMoves + 1;
      const newAverage = (previousAverage * previousTotalMoves + data.stats.searchTime) / newTotalMoves;

      this.lastStats = {
        nodesEvaluated: data.stats.nodesEvaluated,
        agent_type: data.stats.agent_type,
        agent_name: data.stats.agent_name,
        searchTime: data.stats.searchTime,
        lastMoveTime: new Date(),
        totalMoves: newTotalMoves,
        averageSearchTime: newAverage
      };

      // Log decision
      console.log(`AI Player ${this.playerIndex + 1} decision (${data.stats.agent_type}):`);
      console.log(`  Selected move: Factory ${data.move.factoryIndex}, Tile ${data.move.tile}, Line ${data.move.lineIndex + 1}`);
      console.log(`  Search time: ${data.stats.searchTime.toFixed(3)}s`);
      console.log(`  Nodes evaluated: ${data.stats.nodesEvaluated}`);

      return {
        move: data.move,
        value: 0, // C++ AI doesn't return evaluation value in same format
        depth: 0, // MCTS doesn't use traditional depth
        nodesEvaluated: data.stats.nodesEvaluated
      };

    } catch (error) {
      console.error('C++ AI error:', error);
      // Fallback to simple move selection and update stats
      const result = this.getSimpleMove(gameState);

      // Update stats for fallback move
      const previousTotalMoves = this.lastStats.totalMoves || 0;
      this.lastStats = {
        ...this.lastStats,
        nodesEvaluated: result.nodesEvaluated,
        searchTime: 0.01, // Assume 10ms for simple heuristic
        lastMoveTime: new Date(),
        totalMoves: previousTotalMoves + 1,
        averageSearchTime: this.lastStats.averageSearchTime || 0 // Keep existing average
      };

      return result;
    }
  }

  private convertGameStateForAPI(gameState: BaseGameState): any {
    // Convert the webapp game state to a format the C++ API can understand
    return {
      currentPlayer: gameState.currentPlayer,
      round: gameState.round,
      gameOver: gameState.gameOver,
      playerBoards: gameState.playerBoards.map((board, index) => ({
        playerId: index,
        score: board.score,
        wall: board.wall,
        lines: board.lines,
        floor: board.floor
      })),
      factories: gameState.factories,
      center: gameState.center,
      firstPlayerIndex: gameState.firstPlayerIndex,
      availableMoves: gameState.availableMoves
    };
  }

  getSimpleMove(gameState: BaseGameState): SearchResult {
    // Fallback to simple move selection when C++ AI fails
    if (gameState.availableMoves.length === 0) {
      throw new Error('No available moves for fallback');
    }

    // Simple heuristic: prefer moves that complete pattern lines
    let bestMove = gameState.availableMoves[0];
    let bestScore = -1;

    for (const move of gameState.availableMoves) {
      let score = 0;

      // Prefer moves to pattern lines that are close to completion
      if (move.lineIndex >= 0 && move.lineIndex < gameState.playerBoards[this.playerIndex].lines.length) {
        const line = gameState.playerBoards[this.playerIndex].lines[move.lineIndex];
        const lineCapacity = move.lineIndex + 1;
        const currentCount = line.filter(tile => tile !== null).length;
        const tilesNeeded = lineCapacity - currentCount;

        // Higher score for lines that will be completed or nearly completed
        if (tilesNeeded <= 2) {
          score += (3 - tilesNeeded) * 10;
        }
      }

      // Slightly prefer taking from factories over center
      if (move.factoryIndex >= 0) {
        score += 1;
      }

      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }
    }

    return {
      move: bestMove,
      value: bestScore,
      depth: 1,
      nodesEvaluated: gameState.availableMoves.length
    };
  }

  getStats(): {
    nodesEvaluated: number;
    agent_type?: string;
    agent_name?: string;
    searchTime?: number;
    lastMoveTime?: Date;
    totalMoves?: number;
    averageSearchTime?: number;
  } {
    return this.lastStats;
  }

  resetStats(): void {
    this.lastStats = { nodesEvaluated: 0, totalMoves: 0, averageSearchTime: 0 };
  }

  isServerConnected(): boolean {
    return this.isConnected;
  }

  getAgentDisplayName(): string {
    if (this.lastStats.agent_name && this.lastStats.agent_type) {
      return `${this.lastStats.agent_name} (${this.lastStats.agent_type.toUpperCase()})`;
    }
    return this.lastStats.agent_type?.toUpperCase() || 'C++ AI';
  }
}
