import { BaseGameState } from './GameState.js';
import { Move, SearchResult } from './types.js';

interface PythonAIResponse {
  move: Move;
  stats: {
    nodesEvaluated: number;
    searchTime: number;
    simulations: number;
    algorithm?: string;
    agent_type?: string;
  };
  success: boolean;
  error?: string;
}

interface AgentInfo {
  current_agent_type: string;
  active_agent: string;
  neural_network_available: boolean;
  algorithm?: string;
  features?: string;
  simulations?: number;
  exploration_constant?: number;
  temperature?: number;
  neural_network_info?: any;
}

interface AgentType {
  id: string;
  name: string;
  description: string;
}

export class PythonAI {
  private playerIndex: number;
  private maxThinkingTime: number; // in milliseconds
  private apiBaseUrl: string;
  private lastStats: { 
    nodesEvaluated: number;
    algorithm?: string;
    agent_type?: string;
  } = { nodesEvaluated: 0 };
  private isConnected: boolean = false;
  private agentInfo: AgentInfo | null = null;

  constructor(playerIndex: number, maxThinkingTime: number = 2000, apiBaseUrl?: string) {
    this.playerIndex = playerIndex;
    this.maxThinkingTime = maxThinkingTime;
    
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
    console.log('🔍 Auto-discovering Python AI server...');
    
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
          if (data.status === 'healthy' && data.agent_initialized) {
            this.apiBaseUrl = testUrl;
            this.isConnected = true;
            console.log(`✅ Found Python AI server on port ${port}`);
            console.log(`   Agent: ${data.active_agent_type}`);
            console.log(`   Type: ${data.current_agent_type}`);
            
            // Get detailed agent info
            await this.refreshAgentInfo();
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
    console.warn('❌ No Python AI server found on ports 5000-5009');
    console.warn('💡 Try starting the server with: python3 api_server.py');
  }

  private async checkConnection(): Promise<void> {
    try {
      if (!this.apiBaseUrl) {
        // Auto-discovery is still in progress
        this.isConnected = false;
        return;
      }
      
      const response = await fetch(`${this.apiBaseUrl}/health`);
      if (response.ok) {
        const data = await response.json();
        this.isConnected = data.status === 'healthy' && data.agent_initialized;
        console.log('Python AI connection status:', data);
        
        // Get detailed agent info
        if (this.isConnected) {
          await this.refreshAgentInfo();
        }
      } else {
        this.isConnected = false;
        console.warn('Python AI server not responding');
      }
    } catch (error) {
      this.isConnected = false;
      console.warn('Failed to connect to Python AI server:', error);
    }
  }

  private async refreshAgentInfo(): Promise<void> {
    try {
      if (!this.apiBaseUrl) return;
      
      const response = await fetch(`${this.apiBaseUrl}/agent/info`);
      if (response.ok) {
        this.agentInfo = await response.json();
        console.log('Python AI agent info:', this.agentInfo);
      }
    } catch (error) {
      console.warn('Failed to get agent info:', error);
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
          throw new Error('Python AI server is not available. Please start the API server with: python3 api_server.py');
        }
      }
    }

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

    try {
      const requestBody = {
        gameState: this.convertGameStateForAPI(gameState),
        thinkingTime: this.maxThinkingTime
      };

      console.log(`AI Player ${this.playerIndex + 1} requesting move from Python agent...`);

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

      const data: PythonAIResponse = await response.json();

      if (!data.success || data.error) {
        throw new Error(data.error || 'Python AI failed to generate move');
      }

      // Update stats
      this.lastStats = { 
        nodesEvaluated: data.stats.nodesEvaluated,
        algorithm: data.stats.algorithm,
        agent_type: data.stats.agent_type
      };

      // Log decision with agent type info
      const agentTypeDisplay = data.stats.agent_type === 'mcts' ? '🤖 MCTS' : '🧠 Heuristic';
      console.log(`AI Player ${this.playerIndex + 1} decision (${agentTypeDisplay}):`);
      console.log(`  Selected move: Factory ${data.move.factoryIndex}, Tile ${data.move.tile}, Line ${data.move.lineIndex + 1}`);
      console.log(`  Search time: ${data.stats.searchTime.toFixed(3)}s`);
      console.log(`  Algorithm: ${data.stats.algorithm || 'Unknown'}`);
      if (data.stats.simulations > 0) {
        console.log(`  Simulations: ${data.stats.simulations}`);
      }
      console.log(`  Nodes evaluated: ${data.stats.nodesEvaluated}`);

      return {
        move: data.move,
        value: 0, // Python AI doesn't return evaluation value in same format
        depth: 0, // MCTS doesn't use traditional depth
        nodesEvaluated: data.stats.nodesEvaluated
      };

    } catch (error) {
      console.error('Python AI error:', error);
      // Fallback to simple move selection
      return this.getSimpleMove(gameState);
    }
  }

  private convertGameStateForAPI(gameState: BaseGameState): any {
    // Convert the webapp game state to a format the Python API can understand
    // This is a simplified conversion - in a full implementation you'd need to
    // convert all the game state details
    
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
    // Fallback to simple move selection when Python AI fails
    if (gameState.availableMoves.length === 0) {
      throw new Error('No available moves for fallback');
    }

    // Simple heuristic: prefer moves that complete pattern lines
    let bestMove = gameState.availableMoves[0];
    let bestScore = -1;

    for (const move of gameState.availableMoves) {
      let score = 0;
      
      // Prefer moves to pattern lines over floor
      if (move.lineIndex >= 0) {
        score += 10;
        
        // Prefer completing lines
        const currentBoard = gameState.playerBoards[this.playerIndex];
        const currentLine = currentBoard.lines[move.lineIndex];
        const tilesNeeded = move.lineIndex + 1;
        const currentTiles = currentLine.filter(tile => tile !== null).length;
        
        if (currentTiles > 0) {
          score += currentTiles; // Prefer lines with existing tiles
        }
        
        // Bonus for completing a line
        if (currentTiles === tilesNeeded - 1) {
          score += 20;
        }
      }
      
      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }
    }

    this.lastStats = { 
      nodesEvaluated: 1,
      algorithm: 'Fallback heuristic',
      agent_type: 'fallback'
    };

    return {
      move: bestMove,
      value: bestScore,
      depth: 1,
      nodesEvaluated: 1
    };
  }

  getStats(): { 
    nodesEvaluated: number;
    algorithm?: string;
    agent_type?: string;
  } {
    return this.lastStats;
  }

  resetStats(): void {
    this.lastStats = { nodesEvaluated: 0 };
  }

  async getAgentInfo(): Promise<AgentInfo | null> {
    try {
      if (!this.apiBaseUrl) return null;
      
      const response = await fetch(`${this.apiBaseUrl}/agent/info`);
      if (response.ok) {
        this.agentInfo = await response.json();
        return this.agentInfo;
      }
    } catch (error) {
      console.warn('Failed to get agent info:', error);
    }
    return null;
  }

  async getAgentTypes(): Promise<AgentType[]> {
    try {
      if (!this.apiBaseUrl) return [];
      
      const response = await fetch(`${this.apiBaseUrl}/agent/types`);
      if (response.ok) {
        const data = await response.json();
        return data.available_types;
      }
    } catch (error) {
      console.warn('Failed to get agent types:', error);
    }
    return [];
  }

  async configureAgent(config: { 
    agentType?: string;
    networkConfig?: string; 
    simulations?: number;
  }): Promise<boolean> {
    try {
      if (!this.apiBaseUrl) return false;
      
      const response = await fetch(`${this.apiBaseUrl}/agent/configure`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config)
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          // Refresh agent info after reconfiguration
          await this.refreshAgentInfo();
          return true;
        }
      }
    } catch (error) {
      console.warn('Failed to configure agent:', error);
    }
    return false;
  }

  isServerConnected(): boolean {
    return this.isConnected;
  }

  getCurrentAgentInfo(): AgentInfo | null {
    return this.agentInfo;
  }

  getAgentDisplayName(): string {
    if (!this.agentInfo) return 'Unknown';
    
    if (this.agentInfo.active_agent === 'mcts') {
      return '🤖 MCTS Agent';
    } else if (this.agentInfo.active_agent === 'heuristic') {
      return '🧠 Heuristic Agent';
    }
    
    return '❓ Unknown Agent';
  }
} 