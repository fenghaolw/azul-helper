/// <reference types="chrome"/>

interface PlayerBoard {
  id: string; // BGA Player ID
  lines: string[][]; // Array of 5 arrays of tile strings
  wall: string[][]; // 5x5 array of tile strings (empty string if no tile)
  floor: string[]; // Array of tile strings
  score: number;
}

interface GameState {
  factories: string[][];
  center: string[];
  playerBoards: PlayerBoard[]; // All player boards (for compatibility)
  currentPlayerBoard: PlayerBoard; // Current player's board
  opponentBoard: PlayerBoard; // Opponent's board (for 2-player games)
  currentPlayer: number; // Index in the playerBoards array
  round: number;
}

// Function to extract game state from the DOM
function extractGameState(): any | null {
  try {
    const factoriesData: string[][] = [];
    let centerData: string[] = [];
    let activePlayerBgaId: string | null = null;
    let currentPlayerIndex = 0; // Default to 0, will be updated

    // Helper to map tile class to color string
    const mapTileClassToColor = (tileElement: Element): string => {
      const classList = tileElement.className.split(' ');
      // find a class that starts with tile, is longer than 4 chars, and does not contain tile-
      const tileTypeClass = classList.find(cls => /^tile\d$/.test(cls));

      if (tileTypeClass) {
        switch (tileTypeClass) {
          case 'tile0':
            return 'firstplayer';
          case 'tile1':
            return 'black';
          case 'tile2':
            return 'white'; // Maps to turquoise icon in popup via 'white' key
          case 'tile3':
            return 'blue';
          case 'tile4':
            return 'yellow';
          case 'tile5':
            return 'red';
          default:
            console.warn(
              `Unknown tile class for color mapping: ${tileTypeClass} on element ${tileElement.id}`
            );
            return '';
        }
      }
      return '';
    };

    // Process Center (factory0)
    const centerElement = document.getElementById('factory0');
    if (centerElement && centerElement.classList.contains('factory-center')) {
      centerData = Array.from(centerElement.querySelectorAll('.tile'))
        .map(tileEl => mapTileClassToColor(tileEl))
        .filter(color => color !== '' && color !== 'firstplayer'); // Filter out firstplayer from center for AI
    } else {
      console.warn(
        'Center element (#factory0 with class .factory-center) not found or missing class.'
      );
    }

    // Process regular factories (factory1 to factoryN)
    for (let i = 1; i <= 9; i++) {
      const factoryElement = document.getElementById(`factory${i}`);
      if (factoryElement && factoryElement.classList.contains('factory')) {
        const factoryTiles = Array.from(factoryElement.querySelectorAll('.tile'))
          .map(tileEl => mapTileClassToColor(tileEl))
          .filter(color => color !== ''); // Keep all actual tiles
        factoriesData.push(factoryTiles);
      }
    }

    // Process Player Boards
    const playerBoardsData: GameState['playerBoards'] = [];
    const playerBoardWrappers = document.querySelectorAll('div[id^="player-table-wrapper-"]');
    const bgaPlayerIdsInOrder: string[] = [];

    playerBoardWrappers.forEach(wrapper => {
      const wrapperId = wrapper.id;
      const playerIdMatch = wrapperId.match(/player-table-wrapper-(\d+)/);
      if (!playerIdMatch || !playerIdMatch[1]) return;
      const bgaPlayerId = playerIdMatch[1];
      bgaPlayerIdsInOrder.push(bgaPlayerId);

      const playerTable = document.getElementById(`player-table-${bgaPlayerId}`);
      if (!playerTable) return;

      // Note: Active player detection is now done separately using avatar image

      // Score
      const scoreElement = playerTable.querySelector('.score') || wrapper.querySelector('.score');
      const score = scoreElement ? parseInt(scoreElement.textContent || '0', 10) : 0;

      // Pattern Lines
      const lines: string[][] = [];
      for (let i = 1; i <= 5; i++) {
        const lineElement = document.getElementById(`player-table-${bgaPlayerId}-line${i}`);
        lines.push(
          lineElement
            ? Array.from(lineElement.querySelectorAll('.tile'))
                .map(tileEl => mapTileClassToColor(tileEl))
                .filter(color => color !== '')
            : []
        );
      }

      // Floor Line (line0)
      const floorLineElement = document.getElementById(`player-table-${bgaPlayerId}-line0`);
      let floor: string[] = [];
      if (floorLineElement) {
        floor = Array.from(floorLineElement.querySelectorAll('.tile'))
          .map(tileEl => mapTileClassToColor(tileEl))
          .filter(color => color !== '');
      }

      // Wall (5x5)
      const wall: string[][] = Array(5)
        .fill(null)
        .map(() => Array(5).fill('')); // Initialize 5x5 empty wall
      const wallContainer = document.getElementById(`player-table-${bgaPlayerId}-wall`);
      if (wallContainer) {
        for (let r = 1; r <= 5; r++) {
          for (let c = 1; c <= 5; c++) {
            const wallSpot = document.getElementById(
              `player-table-${bgaPlayerId}-wall-spot-${r}-${c}`
            );
            const tileElement = wallSpot?.querySelector('.tile'); // Check if a tile is INSIDE the spot
            if (tileElement) {
              wall[r - 1][c - 1] = mapTileClassToColor(tileElement);
            }
          }
        }
      }

      playerBoardsData.push({
        id: bgaPlayerId,
        lines,
        wall,
        floor,
        score,
      });
    });

    // Determine active player using avatar image
    const activeAvatarImg = document.querySelector(
      'img.avatar.avatar_active[id^="avatar_active_"]'
    );
    if (activeAvatarImg) {
      const avatarId = activeAvatarImg.id;
      const playerIdMatch = avatarId.match(/avatar_active_(\d+)/);
      if (playerIdMatch && playerIdMatch[1]) {
        activePlayerBgaId = playerIdMatch[1];
        console.log(`Found active player via avatar: ${activePlayerBgaId}`);
      }
    }

    if (activePlayerBgaId) {
      currentPlayerIndex = bgaPlayerIdsInOrder.indexOf(activePlayerBgaId);
      if (currentPlayerIndex === -1) currentPlayerIndex = 0; // Fallback if ID not found in order
    } else {
      console.warn('Could not determine active player from avatar image. Defaulting to player 0.');
    }

    const round = 0; // Placeholder for round

    // Set current player board and opponent board
    const currentPlayerBoard = playerBoardsData[currentPlayerIndex] || playerBoardsData[0];
    const opponentBoard =
      playerBoardsData.find((_, index) => index !== currentPlayerIndex) ||
      playerBoardsData[1] ||
      playerBoardsData[0];

    return {
      factories: factoriesData,
      center: centerData,
      playerBoards: playerBoardsData,
      currentPlayerBoard,
      opponentBoard,
      currentPlayer: currentPlayerIndex,
      round,
    };
  } catch (error) {
    console.error('Error extracting game state:', error);
    return null;
  }
}

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request: { action: string }, _sender, sendResponse) => {
  if (request.action === 'getGameState') {
    const gameState = extractGameState();
    if (gameState) {
      // Deep copy for logging to avoid issues with mutable state if any
      console.log('Sending gameState to popup:', JSON.parse(JSON.stringify(gameState)));
    } else {
      console.log('No gameState to send to popup.');
    }
    sendResponse({ gameState });
    return true; // Indicates that the response is sent asynchronously
  }
});

// Set up a mutation observer to detect game state changes
const observer = new MutationObserver(() => {
  console.log('DOM mutation observed, attempting to re-extract game state.');
  const gameState = extractGameState();
  if (gameState) {
    chrome.runtime
      .sendMessage({
        action: 'gameStateUpdated',
        gameState,
      })
      .catch(error => {
        if (
          error.message &&
          !error.message.includes('Receiving end does not exist') &&
          !error.message.includes('Could not establish connection. Receiving end does not exist.')
        ) {
          console.error('Error sending gameStateUpdated message:', error);
        }
      });
  }
});

const gameArea = document.getElementById('game_play_area') || document.body;

observer.observe(gameArea, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeFilter: ['class', 'style', 'id'], // Added 'id' as tile IDs might change
});

console.log('Azul Helper content script loaded and observer started.');
 