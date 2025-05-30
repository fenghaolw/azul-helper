/// <reference types="chrome"/>

import { BGAGameState } from '../../webapp/src/GameState.js';
import { AzulAI } from '../../webapp/src/AI.js';

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
  timeLimit: number;
}

interface AnalysisStats {
  nodes: number;
  depth: number;
  time: number;
  score: number;
}

interface AnalysisResponse {
  move: { factoryIndex: number; tile: string; lineIndex: number };
  stats: AnalysisStats;
  error?: string;
}

async function analyzePosition(bgaState: BGAState, timeLimit: number): Promise<AnalysisResponse> {
  try {
    const gameState = new BGAGameState(bgaState.playerBoards.length);

    gameState.loadFromBga(bgaState);

    const ai = new AzulAI(gameState.currentPlayer, timeLimit);

    const startTime = performance.now();
    const result = ai.getBestMove(gameState);
    const endTime = performance.now();

    const stats: AnalysisStats = {
      nodes: result.nodesEvaluated,
      depth: result.depth,
      time: Math.round(endTime - startTime),
      score: result.value,
    };

    const responseMove = {
      factoryIndex: result.move.factoryIndex,
      tile: result.move.tile.toString(),
      lineIndex: result.move.lineIndex,
    };

    return { move: responseMove, stats };
  } catch (error) {
    console.error('Error analyzing position in background:', error);
    const errorMove = { factoryIndex: 0, tile: 'error', lineIndex: 0 };
    return {
      move: errorMove,
      stats: { nodes: 0, depth: 0, time: 0, score: 0 },
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
      analyzePosition(request.gameState, request.timeLimit).then(sendResponse);
      return true;
    }
    return false;
  }
);
