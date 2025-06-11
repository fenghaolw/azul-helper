/// <reference types="chrome"/>

import { BGAGameState } from '../../webapp/src/GameState.js';
import { Tile } from '../../webapp/src/types.js';


interface BGAState {
  factories: string[][];
  center: string[];
  playerBoards: {
    lines: string[][];
    wall: string[][];
    floor: string[];
    score: number;
  }[];
  currentPlayer: number;
  round: number;
}

interface AnalysisRequest {
  action: 'analyzePosition';
  gameState: BGAState;
}

interface AnalysisStats {
  time: number;
}

interface AnalysisResponse {
  move: { factoryIndex: number; tile: string; lineIndex: number };
  stats: AnalysisStats;
  error?: string;
}

async function analyzePosition(bgaState: BGAState): Promise<AnalysisResponse> {
  try {
    const gameState = new BGAGameState(bgaState.playerBoards.length);
    gameState.loadFromBga(bgaState);

    // Convert game state to API format
    const apiGameState = {
      currentPlayer: gameState.currentPlayer,
      roundNumber: gameState.round,
      gameEnded: gameState.gameOver,
      players: gameState.playerBoards.map((board: any, index: number) => ({
        playerId: index,
        score: board.score,
        wall: board.wall.map((row: any[]) =>
          row.map((cell: any) => cell !== null)
        ),
        patternLines: board.lines.map((line: any) => {
          const count = line.filter((tile: any) => tile !== null).length;
          const color = line.find((tile: any) => tile !== null);
          return {
            count,
            color: color || null
          };
        }),
        floorLine: board.floor.map((tile: string) =>
          tile === Tile.FirstPlayer ? "F" : tile
        ),
      })),
      factories: gameState.factories.map((factory: any) => {
        const counts: { [key: string]: number } = {};
        factory.forEach((tile: string) => {
          counts[tile] = (counts[tile] || 0) + 1;
        });
        return counts;
      }),
      centerPile: gameState.center.reduce((acc: { [key: string]: number }, tile: string) => {
        if (tile !== Tile.FirstPlayer) {
          acc[tile] = (acc[tile] || 0) + 1;
        }
        return acc;
      }, {}),
      firstPlayerNextRound: gameState.firstPlayerIndex,
      firstPlayerTileAvailable: gameState.center.includes(Tile.FirstPlayer),
    };

    const response = await fetch('http://localhost:5001/agent/move', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        gameState: apiGameState,
        playerId: gameState.currentPlayer,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();

    if (!result.success || result.error) {
      throw new Error(result.error || 'Unknown error during AI analysis');
    }

    return {
      move: {
        factoryIndex: result.move.factoryIndex,
        tile: result.move.tile,
        lineIndex: result.move.lineIndex,
      },
      stats: {
        time: result.stats.searchTime * 1000,
      },
    };
  } catch (error) {
    console.error('Error analyzing position in background:', error);
    const errorMove = { factoryIndex: 0, tile: 'error', lineIndex: 0 };
    return {
      move: errorMove,
      stats: { time: 0 },
      error: error instanceof Error ? error.message : 'Unknown error during AI analysis',
    };
  }
}

// Handle action click to open side panel
chrome.action.onClicked.addListener(async tab => {
  if (tab.id) {
    await chrome.sidePanel.open({ tabId: tab.id });
  }
});

chrome.runtime.onMessage.addListener(
  (
    request: AnalysisRequest,
    _sender: chrome.runtime.MessageSender,
    sendResponse: (_: AnalysisResponse) => void
  ) => {
    if (request.action === 'analyzePosition') {
      analyzePosition(request.gameState).then(sendResponse);
      return true;
    }
    return false;
  }
);
