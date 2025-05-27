/// <reference types="chrome"/>

import { GameStateData, PlayerBoard } from './types';

interface DifficultySetting {
  time: number;
  label: string;
}

interface DifficultySettings {
  [key: number]: DifficultySetting;
}

interface Move {
  factoryIndex: number;
  tile: string;
  lineIndex: number;
}

interface AnalysisStats {
  nodes: number;
  depth: number;
  time: number;
  score: number;
}

interface AnalysisResponse {
  move: Move;
  stats: AnalysisStats;
  error?: string;
}

// Difficulty settings
const difficultySettings: DifficultySettings = {
  1: { time: 500, label: 'Easy' },
  2: { time: 1000, label: 'Medium' },
  3: { time: 2000, label: 'Hard' },
  4: { time: 5000, label: 'Expert' },
};

// UI Elements
const suggestionEl = document.getElementById('suggestion') as HTMLDivElement;
const detailsEl = document.getElementById('details') as HTMLDivElement;
const tilePreviewEl = document.getElementById('tile-preview') as HTMLDivElement;
const nodesEl = document.getElementById('nodes') as HTMLSpanElement;
const depthEl = document.getElementById('depth') as HTMLSpanElement;
const timeEl = document.getElementById('time') as HTMLSpanElement;
const difficultySlider = document.getElementById('difficulty') as HTMLInputElement;
const difficultyLabel = document.getElementById('difficulty-label') as HTMLSpanElement;
const analyzeButton = document.getElementById('analyze') as HTMLButtonElement;
const errorEl = document.getElementById('error') as HTMLDivElement;
const gameStateEl = document.getElementById('game-state') as HTMLDivElement;
const playerBoardsEl = document.getElementById('player-boards') as HTMLDivElement;

// Map tile color to SVG filename
const tileSVGs: { [key: string]: string } = {
  red: 'tile-red.svg',
  blue: 'tile-blue.svg',
  yellow: 'tile-yellow.svg',
  black: 'tile-black.svg',
  white: 'tile-turquoise.svg', // Assuming 'white' from game state means the turquoise-colored tile
  firstplayer: 'tile-overlay-dark.svg', // For the first player token, if represented as a tile
  // Add other actual tile names here based on console output
};

// Helper to create an <img> for a tile
function createTileIcon(tile: string): HTMLImageElement {
  const img = document.createElement('img');
  const tileKey = tile.toLowerCase();
  const svgFile = tileSVGs[tileKey];

  if (!svgFile) {
    console.error(`No SVG file found for tile key: '${tileKey}' (original: '${tile}')`);
    img.src = chrome.runtime.getURL('icons/tile-overlay-dark.svg'); // Fallback icon
  } else {
    img.src = chrome.runtime.getURL(`icons/${svgFile}`);
  }
  img.alt = tile;
  img.width = 20;
  img.height = 20;
  img.style.marginRight = '2px';
  img.style.verticalAlign = 'middle';
  return img;
}

// Render game state (factories and center)
function renderGameState(gameState: GameStateData | null) {
  if (!gameState) {
    gameStateEl.innerHTML = '';
    return;
  }

  let html = '<div class="game-state-section">';
  html += '<h3>Factories & Center</h3>';

  // Factories
  gameState.factories.forEach((factory: string[], i: number) => {
    html += '<div class="factory-item">';
    html += `<span class="factory-label">Factory ${i + 1}:</span>`;
    html += '<div>';
    if (!factory || factory.length === 0) {
      html += '<span class="empty-indicator">(empty)</span>';
    } else {
      factory.forEach(tile => {
        if (tile && tile.trim() !== '') {
          const tileKey = tile.toLowerCase();
          const svgFile = tileSVGs[tileKey];
          if (!svgFile) {
            console.error(`Rendering: No SVG for tile key: '${tileKey}' (original: '${tile}')`);
          }
          html += `<img src="${chrome.runtime.getURL(`icons/${svgFile || 'tile-overlay-dark.svg'}`)}" alt="${tile}" width="16" height="16" style="margin-right:1px;vertical-align:middle;">`;
        }
      });
    }
    html += '</div></div>';
  });

  // Center
  html += '<div class="center-item">';
  html += '<span class="factory-label">Center:</span>';
  html += '<div>';
  if (!gameState.center || gameState.center.length === 0) {
    html += '<span class="empty-indicator">(empty)</span>';
  } else {
    gameState.center.forEach((tile: string) => {
      if (tile && tile.trim() !== '') {
        const tileKey = tile.toLowerCase();
        const svgFile = tileSVGs[tileKey];
        if (!svgFile) {
          console.error(`Rendering: No SVG for tile key: '${tileKey}' (original: '${tile}')`);
        }
        html += `<img src="${chrome.runtime.getURL(`icons/${svgFile || 'tile-overlay-dark.svg'}`)}" alt="${tile}" width="16" height="16" style="margin-right:1px;vertical-align:middle;">`;
      }
    });
  }
  html += '</div></div>';
  html += '</div>';

  gameStateEl.innerHTML = html;
}

// Render player boards
function renderPlayerBoards(gameState: GameStateData | null) {
  if (!gameState || !gameState.playerBoards) {
    playerBoardsEl.innerHTML = '';
    return;
  }

  let html = '<h3 style="margin:8px 0 6px 0; font-size:12px; color:#2c3e50;">Player Boards</h3>';

  gameState.playerBoards.forEach((board: PlayerBoard, index: number) => {
    const isCurrentPlayer = index === gameState.currentPlayer;
    const playerName = isCurrentPlayer ? 'Current Player' : 'Opponent';

    html += `<div class="player-board ${isCurrentPlayer ? 'current-player' : ''}">`;
    html += '<div class="player-board-header">';
    html += `<span>${playerName}</span>`;
    html += `<span class="score-display">${board.score} pts</span>`;
    html += '</div>';

    // Pattern Lines
    html += '<div class="board-section">';
    html += '<div class="board-section-title">Pattern Lines</div>';
    html += '<div class="pattern-lines">';

    for (let i = 0; i < 5; i++) {
      const line = board.lines[i] || [];
      html += '<div class="pattern-line">';
      html += `<span class="line-number">${i + 1}:</span>`;

      if (line.length === 0) {
        html += '<span class="empty-indicator">(empty)</span>';
      } else {
        line.forEach((tile: string) => {
          if (tile && tile.trim() !== '') {
            const tileKey = tile.toLowerCase();
            const svgFile = tileSVGs[tileKey];
            html += `<img src="${chrome.runtime.getURL(`icons/${svgFile || 'tile-overlay-dark.svg'}`)}" alt="${tile}" width="12" height="12" style="vertical-align:middle;margin-right:1px;">`;
          }
        });
      }
      html += '</div>';
    }
    html += '</div></div>';

    // Wall
    html += '<div class="board-section">';
    html += '<div class="board-section-title">Wall</div>';
    html += '<div class="wall-grid">';

    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 5; col++) {
        const wallTile = board.wall[row] && board.wall[row][col];
        html += '<div class="wall-spot';
        if (wallTile && wallTile.trim() !== '') {
          html += ' filled';
        }
        html += '">';

        if (wallTile && wallTile.trim() !== '') {
          const tileKey = wallTile.toLowerCase();
          const svgFile = tileSVGs[tileKey];
          html += `<img src="${chrome.runtime.getURL(`icons/${svgFile || 'tile-overlay-dark.svg'}`)}" alt="${wallTile}" width="10" height="10">`;
        }
        html += '</div>';
      }
    }
    html += '</div></div>';

    // Floor Line
    html += '<div class="board-section">';
    html += '<div class="board-section-title">Floor Line</div>';
    html += '<div class="floor-line">';

    if (!board.floor || board.floor.length === 0) {
      html += '<span class="empty-indicator">(empty)</span>';
    } else {
      board.floor.forEach((tile: string) => {
        if (tile && tile.trim() !== '') {
          const tileKey = tile.toLowerCase();
          const svgFile = tileSVGs[tileKey];
          html += `<img src="${chrome.runtime.getURL(`icons/${svgFile || 'tile-overlay-dark.svg'}`)}" alt="${tile}" width="12" height="12" style="vertical-align:middle;margin-right:1px;">`;
        }
      });
    }
    html += '</div></div>';

    html += '</div>'; // Close player-board div
  });

  playerBoardsEl.innerHTML = html;
}

// Update difficulty label (no auto re-analysis)
difficultySlider.addEventListener('input', () => {
  const level = parseInt(difficultySlider.value);
  difficultyLabel.textContent = difficultySettings[level].label;
  // Note: User needs to click analyze button to see effect of new difficulty
});

// Analyze button click handler
analyzeButton.addEventListener('click', () => {
  runAIAnalysis();
});

// Set default difficulty and auto-extract game state when side panel loads
document.addEventListener('DOMContentLoaded', () => {
  const highestDifficulty = 4; // Or Math.max(...Object.keys(difficultySettings).map(Number));
  difficultySlider.value = highestDifficulty.toString();
  difficultyLabel.textContent = difficultySettings[highestDifficulty].label;

  // Auto-extract game state on load
  setTimeout(() => {
    extractAndDisplayGameState();
  }, 500); // Small delay to ensure content script is ready
});

// Store current game state
let currentGameState: GameStateData | null = null;

// Extract and display game state (fast operation)
async function extractAndDisplayGameState() {
  try {
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab.id) {
      throw new Error('No active tab found');
    }

    // Get game state from content script
    const response = await chrome.tabs.sendMessage(tab.id, { action: 'getGameState' });

    if (!response || !response.gameState) {
      console.error('Invalid or missing gameState from content script:', response);
      errorEl.textContent = 'Could not extract game state from the page';
      return;
    }

    console.log('Received gameState:', JSON.parse(JSON.stringify(response.gameState))); // Log a deep copy

    // Store current game state
    currentGameState = response.gameState;

    // Clear any previous errors
    errorEl.textContent = '';

    // Render game state
    renderGameState(response.gameState);

    // Render player boards
    renderPlayerBoards(response.gameState);

    // Only set initial message if no suggestion is currently displayed
    if (suggestionEl.textContent === '' || suggestionEl.textContent === 'Analyzing game state...') {
      suggestionEl.textContent = 'Click "Analyze Position" to get AI suggestion';
    }
    // Don't clear existing AI analysis results
  } catch (error) {
    errorEl.textContent = error instanceof Error ? error.message : 'An unknown error occurred';
    currentGameState = null;
  }
}

// Run AI analysis (slow operation)
async function runAIAnalysis() {
  if (!currentGameState) {
    errorEl.textContent = 'No game state available. Please wait for game state to be extracted.';
    return;
  }

  try {
    // Show loading state
    suggestionEl.textContent = 'Analyzing position...';
    detailsEl.textContent = '';
    tilePreviewEl.innerHTML = '';
    errorEl.textContent = '';

    // Get difficulty setting
    const level = parseInt(difficultySlider.value);
    const timeLimit = difficultySettings[level].time;

    // Send game state to background script for AI analysis
    chrome.runtime.sendMessage(
      {
        action: 'analyzePosition',
        gameState: currentGameState,
        timeLimit,
      },
      handleAnalysisResponse
    );
  } catch (error) {
    errorEl.textContent = error instanceof Error ? error.message : 'An unknown error occurred';
    suggestionEl.textContent = 'Error analyzing position';
  }
}

// Handle AI analysis response
function handleAnalysisResponse(response: AnalysisResponse): void {
  if (response.error) {
    errorEl.textContent = response.error;
    suggestionEl.textContent = 'Error analyzing position';
    return;
  }

  const { move, stats } = response;

  // Update suggestion
  suggestionEl.textContent = formatMove(move);

  // Update details
  detailsEl.textContent = `Expected score: ${stats.score}`;

  // Update tile preview
  tilePreviewEl.innerHTML = ''; // Clear previous preview
  if (move && move.tile) {
    const tileIconElement = createTileIcon(move.tile);
    tilePreviewEl.appendChild(tileIconElement);
  } else {
    console.warn('No tile information in the suggested move for preview.');
    // Optionally display a placeholder or nothing
  }

  // Update stats
  nodesEl.textContent = `Nodes evaluated: ${stats.nodes}`;
  depthEl.textContent = `Search depth: ${stats.depth}`;
  timeEl.textContent = `Time: ${stats.time}ms`;
}

// Format move for display
function formatMove(move: Move): string {
  const source = move.factoryIndex === -1 ? 'center' : `factory ${move.factoryIndex + 1}`;
  const destination = move.lineIndex === -1 ? 'floor' : `pattern line ${move.lineIndex + 1}`;
  return `Take ${move.tile} tiles from ${source} and place them in ${destination}`;
}

// Listen for game state updates and auto-extract
chrome.runtime.onMessage.addListener((message: { action: string }) => {
  if (message.action === 'gameStateUpdated') {
    // Auto-extract game state when it changes (but don't run AI analysis)
    extractAndDisplayGameState();
  }
});
