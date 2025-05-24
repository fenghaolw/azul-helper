import { GameState } from './GameState.js';
import { PlayerBoard } from './PlayerBoard.js';
import { Tile, Move } from './types.js';

interface LayoutConfig {
  canvas: { width: number; height: number };
  title: { x: number; y: number };
  factories: { 
    startX: number; 
    startY: number; 
    size: number; 
    spacing: number;
    perRow: number;
  };
  center: { 
    x: number; 
    y: number; 
    width: number; 
    height: number;
  };
  playerBoards: {
    startX: number;
    startY: number;
    width: number;
    height: number;
    spacingX: number;
    spacingY: number;
    patternLines: {
      startY: number;
      height: number;
      spacing: number;
      tileSize: number;
    };
    wall: {
      startX: number;
      tileSize: number;
      spacing: number;
    };
    floor: {
      startY: number;
      height: number;
    };
  };
  gameInfo: { x: number; y: number };
}

export class GameRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private gameState: GameState;
  private selectedFactory: number = -2; // -2 = none, -1 = center, 0+ = factory
  private selectedTile: Tile | null = null;
  private hoveredLine: number = -2; // -2 = none, -1 = floor, 0-4 = pattern lines
  private hoveredFactory: number = -2; // For visual feedback
  private hoveredTile: Tile | null = null; // For tile highlighting

  // SVG tile images
  private tileImages: Map<Tile, HTMLImageElement> = new Map();
  private imagesLoaded: boolean = false;

  // Centralized layout configuration - single source of truth
  private layout: LayoutConfig = {
    canvas: { width: 1200, height: 800 },
    title: { x: 600, y: 25 }, // canvas.width / 2
    factories: {
      startX: 50,
      startY: 80,
      size: 120,
      spacing: 20,
      perRow: 0 // Will be calculated based on number of factories
    },
    center: {
      x: 50,
      y: 420,
      width: 400,
      height: 120
    },
    playerBoards: {
      startX: 500,
      startY: 80,
      width: 300,
      height: 400,
      spacingX: 50,
      spacingY: 50,
      patternLines: {
        startY: 40, // Relative to board
        height: 30,
        spacing: 5,
        tileSize: 25
      },
      wall: {
        startX: 150, // Relative to board
        tileSize: 25,
        spacing: 2
      },
      floor: {
        startY: 340, // Relative to board (height - 60)
        height: 30
      }
    },
    gameInfo: { x: 50, y: 560 }
  };

  // Fallback Material Design inspired tile colors (in case images fail to load)
  private tileColors = {
    [Tile.Red]: '#f44336',     // Material Red 500
    [Tile.Blue]: '#2196f3',    // Material Blue 500
    [Tile.Yellow]: '#ffeb3b',  // Material Yellow 500
    [Tile.Black]: '#424242',   // Material Grey 800
    [Tile.White]: '#fafafa',   // Material Grey 50
    [Tile.FirstPlayer]: '#9c27b0' // Material Purple 500
  };

  constructor(canvas: HTMLCanvasElement, gameState: GameState) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.gameState = gameState;
    
    // Set up high-DPI canvas for crisp rendering
    this.setupHighDPICanvas();
    
    // Initialize layout
    this.layout.factories.perRow = Math.ceil(this.gameState.factories.length / 2);
    
    // Load SVG tile images
    this.loadTileImages();
    
    // Add event listeners
    this.canvas.addEventListener('click', this.handleClick.bind(this));
    this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
  }

  private setupHighDPICanvas(): void {
    const devicePixelRatio = window.devicePixelRatio || 1;
    const logicalWidth = this.layout.canvas.width;
    const logicalHeight = this.layout.canvas.height;
    
    // Set the actual canvas size in memory (scaled up for high-DPI)
    this.canvas.width = logicalWidth * devicePixelRatio;
    this.canvas.height = logicalHeight * devicePixelRatio;
    
    // Set the CSS size to the logical size
    this.canvas.style.width = logicalWidth + 'px';
    this.canvas.style.height = logicalHeight + 'px';
    
    // Scale the drawing context so everything draws at the correct size
    this.ctx.scale(devicePixelRatio, devicePixelRatio);
    
    // Enable better image smoothing
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
    
    console.log(`Canvas setup: ${logicalWidth}×${logicalHeight} logical, ${this.canvas.width}×${this.canvas.height} actual (${devicePixelRatio}x DPI)`);
  }

  private async loadTileImages(): Promise<void> {
    const tileFilenames = {
      [Tile.Red]: 'tile-red.svg',
      [Tile.Blue]: 'tile-blue.svg', 
      [Tile.Yellow]: 'tile-yellow.svg',
      [Tile.Black]: 'tile-black.svg',
      [Tile.White]: 'tile-turquoise.svg', // Using turquoise for white/light blue tiles
      [Tile.FirstPlayer]: 'tile-yellow.svg' // Using yellow as base for first player token
    };

    const loadPromises: Promise<void>[] = [];

    for (const [tile, filename] of Object.entries(tileFilenames)) {
      const promise = new Promise<void>((resolve) => {
        const img = new Image();
        
        img.onload = () => {
          this.tileImages.set(tile as Tile, img);
          resolve();
        };
        
        img.onerror = () => {
          console.warn(`Failed to load tile image: ${filename}`);
          resolve(); // Continue even if image fails to load
        };
        
        img.src = `/imgs/${filename}`;
      });
      
      loadPromises.push(promise);
    }

    try {
      await Promise.all(loadPromises);
      this.imagesLoaded = true;
      console.log('Tile images loaded successfully');
    } catch (error) {
      console.error('Error loading tile images:', error);
      this.imagesLoaded = false;
    }
  }

  // Layout utility methods - single source of truth for all positions
  private getFactoryBounds(factoryIndex: number) {
    const row = Math.floor(factoryIndex / this.layout.factories.perRow);
    const col = factoryIndex % this.layout.factories.perRow;
    const x = this.layout.factories.startX + col * (this.layout.factories.size + this.layout.factories.spacing);
    const y = this.layout.factories.startY + row * (this.layout.factories.size + this.layout.factories.spacing);
    
    return {
      x,
      y,
      width: this.layout.factories.size,
      height: this.layout.factories.size
    };
  }

  private getCenterBounds() {
    return {
      x: this.layout.center.x,
      y: this.layout.center.y,
      width: this.layout.center.width,
      height: this.layout.center.height
    };
  }

  private getPlayerBoardBounds(playerIndex: number) {
    const x = this.layout.playerBoards.startX + (playerIndex % 2) * (this.layout.playerBoards.width + this.layout.playerBoards.spacingX);
    const y = this.layout.playerBoards.startY + Math.floor(playerIndex / 2) * (this.layout.playerBoards.height + this.layout.playerBoards.spacingY);
    
    return {
      x,
      y,
      width: this.layout.playerBoards.width,
      height: this.layout.playerBoards.height
    };
  }

  private getPatternLineBounds(playerIndex: number, lineIndex: number) {
    const boardBounds = this.getPlayerBoardBounds(playerIndex);
    const maxTiles = lineIndex + 1;
    const tileSize = this.layout.playerBoards.patternLines.tileSize;
    
    return {
      x: boardBounds.x + 10,
      y: boardBounds.y + this.layout.playerBoards.patternLines.startY + lineIndex * (this.layout.playerBoards.patternLines.height + this.layout.playerBoards.patternLines.spacing),
      width: maxTiles * (tileSize + 2),
      height: this.layout.playerBoards.patternLines.height
    };
  }

  private getFloorBounds(playerIndex: number) {
    const boardBounds = this.getPlayerBoardBounds(playerIndex);
    
    return {
      x: boardBounds.x + 10,
      y: boardBounds.y + this.layout.playerBoards.floor.startY,
      width: this.layout.playerBoards.width - 20,
      height: this.layout.playerBoards.floor.height
    };
  }

  private getWallBounds(playerIndex: number) {
    const boardBounds = this.getPlayerBoardBounds(playerIndex);
    
    return {
      x: boardBounds.x + this.layout.playerBoards.wall.startX,
      y: boardBounds.y + this.layout.playerBoards.patternLines.startY,
      tileSize: this.layout.playerBoards.wall.tileSize,
      spacing: this.layout.playerBoards.wall.spacing
    };
  }

  render(): void {
    // Enable crisp rendering
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
    
    // Clear canvas with crisp edges
    this.ctx.clearRect(0, 0, this.layout.canvas.width, this.layout.canvas.height);
    
    // Draw Portuguese azulejo-inspired background
    this.drawBackground();

    // Draw game title
    this.drawTitle();

    // Draw game elements
    this.drawFactories();
    this.drawCenter();
    this.drawPlayerBoards();
    this.drawGameInfo();
    this.drawAvailableMoves();
  }

  private drawBackground(): void {
    // Material Design background
    this.ctx.fillStyle = '#f5f5f5'; // Material Design Grey 100
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Add subtle dot pattern
    this.ctx.globalAlpha = 0.05;
    for (let x = 0; x < this.canvas.width; x += 24) {
      for (let y = 0; y < this.canvas.height; y += 24) {
        this.ctx.fillStyle = '#9E9E9E';
        this.ctx.beginPath();
        this.ctx.arc(x, y, 1, 0, 2 * Math.PI);
        this.ctx.fill();
      }
    }
    this.ctx.globalAlpha = 1.0;
  }

  private drawTitle(): void {
    const centerX = this.layout.title.x;
    const titleY = this.layout.title.y;
    
    // Main title - Material Design H4
    this.ctx.fillStyle = '#212121'; // Material Design Grey 900
    this.ctx.font = '400 32px "Roboto", sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('Azul', centerX, titleY);
    
    // Subtitle - Material Design Body2
    this.ctx.fillStyle = '#757575'; // Material Design Grey 600
    this.ctx.font = '400 14px "Roboto", sans-serif';
    this.ctx.fillText('Strategic tile-laying board game', centerX, titleY + 24);
  }

  private drawFactories(): void {
    for (let i = 0; i < this.gameState.factories.length; i++) {
      const bounds = this.getFactoryBounds(i);
      this.drawFactory(bounds.x, bounds.y, bounds.width, this.gameState.factories[i], i);
    }
  }

  private getFactoryTilePositions(tiles: Tile[], factoryX: number, factoryY: number, factorySize: number): Array<{ tile: Tile; x: number; y: number; index: number }> {
    // Show each tile individually in a 2x2 grid
    const tileSize = (factorySize - 30) / 2;
    const tileSpacing = 10;
    const positions = [];
    
    for (let i = 0; i < Math.min(tiles.length, 4); i++) {
      const tileRow = Math.floor(i / 2);
      const tileCol = i % 2;
      const tileX = factoryX + 15 + tileCol * (tileSize + tileSpacing);
      const tileY = factoryY + 15 + tileRow * (tileSize + tileSpacing);
      
      positions.push({
        tile: tiles[i],
        x: tileX,
        y: tileY,
        index: i
      });
    }
    
    return positions;
  }

  private drawFactory(x: number, y: number, size: number, tiles: Tile[], factoryIndex: number): void {
    const isSelected = this.selectedFactory === factoryIndex;
    const isHovered = this.hoveredFactory === factoryIndex;
    const radius = 12;
    
    // Factory shadow
    this.ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
    this.ctx.shadowBlur = 8;
    this.ctx.shadowOffsetX = 3;
    this.ctx.shadowOffsetY = 3;
    
    // Factory background with gradient
    const gradient = this.ctx.createRadialGradient(x + size/2, y + size/2, 0, x + size/2, y + size/2, size/2);
    if (isSelected) {
      gradient.addColorStop(0, '#74b9ff');
      gradient.addColorStop(1, '#0984e3');
    } else if (isHovered) {
      gradient.addColorStop(0, '#b2bec3');
      gradient.addColorStop(1, '#636e72');
    } else {
      gradient.addColorStop(0, '#ddd');
      gradient.addColorStop(1, '#95a5a6');
    }
    
    this.ctx.fillStyle = gradient;
    this.drawRoundedRect(x, y, size, size, radius);
    this.ctx.fill();
    
    // Reset shadow
    this.ctx.shadowColor = 'transparent';
    this.ctx.shadowBlur = 0;
    this.ctx.shadowOffsetX = 0;
    this.ctx.shadowOffsetY = 0;
    
    // Factory border with depth
    this.ctx.strokeStyle = isSelected ? '#0984e3' : 'rgba(0, 0, 0, 0.3)';
    this.ctx.lineWidth = isSelected ? 3 : 2;
    this.drawRoundedRect(x, y, size, size, radius);
    this.ctx.stroke();
    
    // Inner highlight
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    this.ctx.lineWidth = 1;
    this.drawRoundedRect(x + 2, y + 2, size - 4, size - 4, radius - 2);
    this.ctx.stroke();

    // Get tile positions using the same logic as hit detection
    const tilePositions = this.getFactoryTilePositions(tiles, x, y, size);
    const tileSize = (size - 30) / 2;
    
    // Draw tiles
    for (const { tile, x: tileX, y: tileY } of tilePositions) {
      // Highlight if this tile type is selected or hovered
      const isTileSelected = this.selectedTile === tile && this.selectedFactory === factoryIndex;
      const isTileHovered = this.hoveredTile === tile && this.hoveredFactory === factoryIndex;
      
      if (isTileSelected || isTileHovered) {
        this.ctx.fillStyle = isTileSelected ? '#f1c40f' : '#e8f6f3';
        this.ctx.fillRect(tileX - 3, tileY - 3, tileSize + 6, tileSize + 6);
      }
      
      this.drawTile(tileX, tileY, tileSize, tile);
    }


  }

  private drawCenter(): void {
    const bounds = this.getCenterBounds();
    const radius = 15;

    const isSelected = this.selectedFactory === -1;
    const isHovered = this.hoveredFactory === -1;
    
    // Center shadow
    this.ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
    this.ctx.shadowBlur = 8;
    this.ctx.shadowOffsetX = 3;
    this.ctx.shadowOffsetY = 3;
    
    // Center background with elegant gradient
    const gradient = this.ctx.createLinearGradient(bounds.x, bounds.y, bounds.x, bounds.y + bounds.height);
    if (isSelected) {
      gradient.addColorStop(0, '#74b9ff');
      gradient.addColorStop(1, '#0984e3');
    } else if (isHovered) {
      gradient.addColorStop(0, '#dfe6e9');
      gradient.addColorStop(1, '#b2bec3');
    } else {
      gradient.addColorStop(0, '#f8f9fa');
      gradient.addColorStop(1, '#e9ecef');
    }
    
    this.ctx.fillStyle = gradient;
    this.drawRoundedRect(bounds.x, bounds.y, bounds.width, bounds.height, radius);
    this.ctx.fill();
    
    // Reset shadow
    this.ctx.shadowColor = 'transparent';
    this.ctx.shadowBlur = 0;
    this.ctx.shadowOffsetX = 0;
    this.ctx.shadowOffsetY = 0;
    
    // Center border
    this.ctx.strokeStyle = isSelected ? '#0984e3' : 'rgba(0, 0, 0, 0.2)';
    this.ctx.lineWidth = isSelected ? 3 : 2;
    this.drawRoundedRect(bounds.x, bounds.y, bounds.width, bounds.height, radius);
    this.ctx.stroke();
    
    // Inner highlight
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    this.ctx.lineWidth = 1;
    this.drawRoundedRect(bounds.x + 2, bounds.y + 2, bounds.width - 4, bounds.height - 4, radius - 2);
    this.ctx.stroke();

    // Get tile positions using the same logic as hit detection
    const tilePositions = this.getCenterTilePositions();
    
    // Draw regular tiles
    for (const { tile, x, y } of tilePositions) {
      // Highlight if this tile type is selected or hovered
      const isTileSelected = this.selectedTile === tile && this.selectedFactory === -1;
      const isTileHovered = this.hoveredTile === tile && this.hoveredFactory === -1;
      
      if (isTileSelected || isTileHovered) {
        this.ctx.fillStyle = isTileSelected ? '#f1c40f' : '#e8f6f3';
        this.ctx.fillRect(x - 3, y - 3, 30 + 6, 30 + 6);
      }
      
      this.drawTile(x, y, 30, tile);
    }
    
    // Draw FirstPlayer token separately if present
    if (this.gameState.center.includes(Tile.FirstPlayer)) {
      const tokenX = bounds.x + bounds.width - 40;
      const tokenY = bounds.y + 10;
      
      // Add a special highlight for first player token
      this.ctx.fillStyle = 'rgba(156, 39, 176, 0.2)'; // Purple highlight
      this.ctx.fillRect(tokenX - 5, tokenY - 5, 40, 40);
      
      this.drawTile(tokenX, tokenY, 30, Tile.FirstPlayer);
    }

    // Center label with elegant styling
    this.ctx.fillStyle = '#212121'; // Material Design Grey 900
    this.ctx.font = '500 16px "Roboto", sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.strokeStyle = '#ffffff';
    this.ctx.lineWidth = 3;
    this.ctx.strokeText('Table Center', bounds.x, bounds.y - 8);
    this.ctx.fillText('Table Center', bounds.x, bounds.y - 8);
    
    // Add helpful subtitle if there are tiles
    const regularTiles = this.gameState.center.filter(t => t !== Tile.FirstPlayer);
    if (regularTiles.length > 0) {
      this.ctx.fillStyle = '#757575'; // Material Design Grey 600
      this.ctx.font = '400 12px "Roboto", sans-serif';
      this.ctx.fillText('(organized by color)', bounds.x, bounds.y + 8);
    }
  }

  private drawPlayerBoards(): void {
    const numPlayers = this.gameState.playerBoards.length;
    
    for (let i = 0; i < numPlayers; i++) {
      const bounds = this.getPlayerBoardBounds(i);
      this.drawPlayerBoard(bounds.x, bounds.y, bounds.width, bounds.height, this.gameState.playerBoards[i], i);
    }
  }

  private drawPlayerBoard(x: number, y: number, width: number, height: number, board: PlayerBoard, playerIndex: number): void {
    // Board background
    this.ctx.fillStyle = playerIndex === this.gameState.currentPlayer ? '#e8f8f5' : '#f8f9fa';
    this.ctx.fillRect(x, y, width, height);
    
    // Board border
    this.ctx.strokeStyle = playerIndex === this.gameState.currentPlayer ? '#27ae60' : '#bdc3c7';
    this.ctx.lineWidth = playerIndex === this.gameState.currentPlayer ? 3 : 2;
    this.ctx.strokeRect(x, y, width, height);

    // Player label and score
    this.ctx.fillStyle = '#212121'; // Material Design Grey 900
    this.ctx.font = '500 16px "Roboto", sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(`Player ${playerIndex + 1}`, x + 10, y + 25);
    this.ctx.fillText(`Score: ${board.score}`, x + 150, y + 25);

    // Draw pattern lines (left side)
    for (let i = 0; i < 5; i++) {
      const lineBounds = this.getPatternLineBounds(playerIndex, i);
      this.drawPatternLine(lineBounds.x, lineBounds.y, i, board.lines[i], playerIndex);
    }

    // Draw wall (right side)
    const wallBounds = this.getWallBounds(playerIndex);
    this.drawWall(wallBounds.x, wallBounds.y, board.wall);

    // Draw floor line
    const floorBounds = this.getFloorBounds(playerIndex);
    this.drawFloor(floorBounds.x, floorBounds.y, floorBounds.width, board.floor, playerIndex);
  }

  private drawPatternLine(x: number, y: number, lineIndex: number, tiles: Tile[], playerIndex: number): void {
    const tileSize = this.layout.playerBoards.patternLines.tileSize;
    const maxTiles = lineIndex + 1;
    
    // Line background
    const isHovered = this.hoveredLine === lineIndex && this.gameState.currentPlayer === playerIndex;
    this.ctx.fillStyle = isHovered ? '#d5dbdb' : '#ecf0f1';
    this.ctx.fillRect(x, y, maxTiles * (tileSize + 2), tileSize);
    
    // Line border
    this.ctx.strokeStyle = '#bdc3c7';
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(x, y, maxTiles * (tileSize + 2), tileSize);

    // Draw tiles
    for (let i = 0; i < tiles.length; i++) {
      const tileX = x + 1 + i * (tileSize + 2);
      const tileY = y + 1;
      this.drawTile(tileX, tileY, tileSize - 2, tiles[i]);
    }

    // Draw empty slots
    for (let i = tiles.length; i < maxTiles; i++) {
      const tileX = x + 1 + i * (tileSize + 2);
      const tileY = y + 1;
      this.ctx.fillStyle = '#ffffff';
      this.ctx.fillRect(tileX, tileY, tileSize - 2, tileSize - 2);
      this.ctx.strokeStyle = '#bdc3c7';
      this.ctx.strokeRect(tileX, tileY, tileSize - 2, tileSize - 2);
    }
  }

  private drawWall(x: number, y: number, wall: Tile[][]): void {
    const tileSize = this.layout.playerBoards.wall.tileSize;
    const spacing = this.layout.playerBoards.wall.spacing;
    const wallPattern = [
      [Tile.Blue, Tile.Yellow, Tile.Red, Tile.Black, Tile.White],
      [Tile.White, Tile.Blue, Tile.Yellow, Tile.Red, Tile.Black],
      [Tile.Black, Tile.White, Tile.Blue, Tile.Yellow, Tile.Red],
      [Tile.Red, Tile.Black, Tile.White, Tile.Blue, Tile.Yellow],
      [Tile.Yellow, Tile.Red, Tile.Black, Tile.White, Tile.Blue]
    ];

    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 5; col++) {
        const tileX = x + col * (tileSize + spacing);
        const tileY = y + row * (tileSize + this.layout.playerBoards.patternLines.spacing + 2);
        const expectedTile = wallPattern[row][col];
        const hasTile = wall[row].includes(expectedTile);

        if (hasTile) {
          this.drawTile(tileX, tileY, tileSize, expectedTile);
        } else {
          // Draw placeholder using SVG with transparency
          this.drawPlaceholderTile(tileX, tileY, tileSize, expectedTile);
        }
      }
    }
  }

  private drawFloor(x: number, y: number, width: number, floor: Tile[], playerIndex: number): void {
    const floorHeight = 30;
    
    // Floor background
    const isHovered = this.hoveredLine === -1 && this.gameState.currentPlayer === playerIndex;
    this.ctx.fillStyle = isHovered ? '#fadbd8' : '#f8d7da';
    this.ctx.fillRect(x, y, width, floorHeight);
    
    // Floor border
    this.ctx.strokeStyle = '#e74c3c';
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(x, y, width, floorHeight);

    // Floor label
    this.ctx.fillStyle = '#e74c3c';
    this.ctx.font = '400 12px "Roboto", sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText('Floor', x + 5, y + 15);

    // Draw floor tiles
    const tileSize = 20;
    for (let i = 0; i < Math.min(floor.length, 7); i++) {
      const tileX = x + 50 + i * (tileSize + 2);
      const tileY = y + 5;
      this.drawTile(tileX, tileY, tileSize, floor[i]);
    }

    // Show penalty points
    const penalties = [-1, -1, -2, -2, -2, -3, -3];
    for (let i = 0; i < 7; i++) {
      const penaltyX = x + 50 + i * (tileSize + 2);
      const penaltyY = y + floorHeight - 8;
      this.ctx.fillStyle = '#e74c3c';
      this.ctx.font = '400 10px "Roboto", sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.fillText(`${penalties[i]}`, penaltyX + tileSize / 2, penaltyY);
    }
  }

  private drawTile(x: number, y: number, size: number, tile: Tile): void {
    const radius = Math.min(size * 0.15, 8);
    
    // Drop shadow for Material Design elevation
    this.ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
    this.ctx.shadowBlur = 4;
    this.ctx.shadowOffsetX = 2;
    this.ctx.shadowOffsetY = 2;
    
    // Use SVG image if loaded, otherwise fallback to solid color
    if (this.imagesLoaded && this.tileImages.has(tile)) {
      const img = this.tileImages.get(tile)!;
      
      // Create a clipping region with rounded corners
      this.ctx.save();
      this.drawRoundedRect(x, y, size, size, radius);
      this.ctx.clip();
      
      // Draw the SVG image
      this.ctx.drawImage(img, x, y, size, size);
      
      this.ctx.restore();
      
      // Add border for definition
      this.ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
      this.ctx.lineWidth = 1;
      this.drawRoundedRect(x, y, size, size, radius);
      this.ctx.stroke();
      
    } else {
      // Fallback to solid color if images aren't loaded
      this.ctx.fillStyle = this.tileColors[tile];
      this.drawRoundedRect(x, y, size, size, radius);
      this.ctx.fill();
      
      // Simple border for definition
      this.ctx.strokeStyle = tile === Tile.White ? '#e0e0e0' : 'rgba(0, 0, 0, 0.1)';
      this.ctx.lineWidth = 1;
      this.drawRoundedRect(x, y, size, size, radius);
      this.ctx.stroke();
    }
    
    // Reset shadow
    this.ctx.shadowColor = 'transparent';
    this.ctx.shadowBlur = 0;
    this.ctx.shadowOffsetX = 0;
    this.ctx.shadowOffsetY = 0;
    
    // Add small "1" text for FirstPlayer tile only
    if (tile === Tile.FirstPlayer) {
      // Add a semi-transparent dark circle behind the "1"
      this.ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      this.ctx.beginPath();
      this.ctx.arc(x + size / 2, y + size / 2, size * 0.3, 0, 2 * Math.PI);
      this.ctx.fill();
      
      // Add the "1" text
      this.ctx.fillStyle = '#ffffff';
      this.ctx.font = `700 ${Math.round(size * 0.4)}px "Roboto", sans-serif`;
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText('1', x + size / 2, y + size / 2);
    }
  }

  private drawPlaceholderTile(x: number, y: number, size: number, tile: Tile): void {
    const radius = Math.min(size * 0.15, 8);
    
    // Use SVG image with transparency if loaded, otherwise fallback to solid color with transparency
    if (this.imagesLoaded && this.tileImages.has(tile)) {
      const img = this.tileImages.get(tile)!;
      
      // Create a clipping region with rounded corners
      this.ctx.save();
      this.drawRoundedRect(x, y, size, size, radius);
      this.ctx.clip();
      
      // Draw the SVG image with transparency
      this.ctx.globalAlpha = 0.25; // 25% opacity for placeholder
      this.ctx.drawImage(img, x, y, size, size);
      this.ctx.globalAlpha = 1.0; // Reset opacity
      
      this.ctx.restore();
      
      // Add border for definition
      this.ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
      this.ctx.lineWidth = 1;
      this.drawRoundedRect(x, y, size, size, radius);
      this.ctx.stroke();
      
    } else {
      // Fallback to solid color with transparency if images aren't loaded
      this.ctx.fillStyle = this.tileColors[tile] + '40'; // Add transparency
      this.drawRoundedRect(x, y, size, size, radius);
      this.ctx.fill();
      
      // Border for definition
      this.ctx.strokeStyle = '#bdc3c7';
      this.ctx.lineWidth = 1;
      this.drawRoundedRect(x, y, size, size, radius);
      this.ctx.stroke();
    }
  }

  private drawRoundedRect(x: number, y: number, width: number, height: number, radius: number): void {
    this.ctx.beginPath();
    this.ctx.moveTo(x + radius, y);
    this.ctx.lineTo(x + width - radius, y);
    this.ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    this.ctx.lineTo(x + width, y + height - radius);
    this.ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    this.ctx.lineTo(x + radius, y + height);
    this.ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    this.ctx.lineTo(x, y + radius);
    this.ctx.quadraticCurveTo(x, y, x + radius, y);
    this.ctx.closePath();
  }

  private drawGameInfo(): void {
    const infoX = this.layout.gameInfo.x;
    const infoY = this.layout.gameInfo.y;

    this.ctx.fillStyle = '#212121'; // Material Design Grey 900
    this.ctx.font = '500 18px "Roboto", sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(`Round ${this.gameState.round}`, infoX, infoY);
    
    if (!this.gameState.gameOver) {
      this.ctx.fillText(`Current Player: ${this.gameState.currentPlayer + 1}`, infoX, infoY + 25);
      this.ctx.fillText(`Available Moves: ${this.gameState.availableMoves.length}`, infoX, infoY + 50);
    } else {
      const result = this.gameState.getResult();
      if (result.winner !== -1) {
        this.ctx.fillText(`Winner: Player ${result.winner + 1}!`, infoX, infoY + 25);
      } else {
        this.ctx.fillText('Game ended in a tie!', infoX, infoY + 25);
      }
    }

    // Selection info
    if (this.selectedFactory !== -2 && this.selectedTile) {
      const factoryText = this.selectedFactory === -1 ? 'Center' : `Factory ${this.selectedFactory + 1}`;
      this.ctx.fillStyle = '#212121'; // Material Design Grey 900
      this.ctx.font = '500 16px "Roboto", sans-serif';
      
      // Draw a small tile indicator
      const tileX = infoX;
      const tileY = infoY + 70;
      this.drawTile(tileX, tileY, 20, this.selectedTile);
      
      this.ctx.fillText(`Selected: ${this.selectedTile.charAt(0).toUpperCase() + this.selectedTile.slice(1)} tiles from ${factoryText}`, infoX + 30, infoY + 85);
      this.ctx.font = '400 14px "Roboto", sans-serif';
      this.ctx.fillStyle = '#757575'; // Material Design Grey 600
      this.ctx.fillText('Click a pattern line or floor to place tiles', infoX, infoY + 105);
    } else if (this.hoveredTile && this.hoveredFactory !== -2) {
      const factoryText = this.hoveredFactory === -1 ? 'Center' : `Factory ${this.hoveredFactory + 1}`;
      this.ctx.fillStyle = '#757575'; // Material Design Grey 600
      this.ctx.font = '400 14px "Roboto", sans-serif';
      this.ctx.fillText(`Hover: ${this.hoveredTile.charAt(0).toUpperCase() + this.hoveredTile.slice(1)} tiles in ${factoryText}`, infoX, infoY + 75);
      this.ctx.fillText('Click to select these tiles', infoX, infoY + 95);
    }
  }

  private drawAvailableMoves(): void {
    if (this.selectedFactory === -2 || !this.selectedTile) return;

    // Highlight valid destinations
    const currentBoard = this.gameState.playerBoards[this.gameState.currentPlayer];
    
    for (let i = 0; i < 5; i++) {
      if (currentBoard.canPlaceTile(this.selectedTile, i)) {
        // Highlight pattern line
        const lineBounds = this.getPatternLineBounds(this.gameState.currentPlayer, i);
        
        this.ctx.strokeStyle = '#27ae60';
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(lineBounds.x - 2, lineBounds.y - 2, lineBounds.width + 4, lineBounds.height + 4);
      }
    }

    // Always highlight floor as valid
    const floorBounds = this.getFloorBounds(this.gameState.currentPlayer);
    
    this.ctx.strokeStyle = '#e74c3c';
    this.ctx.lineWidth = 3;
    this.ctx.strokeRect(floorBounds.x - 2, floorBounds.y - 2, floorBounds.width + 4, floorBounds.height + 4);
  }

  private handleClick(event: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Check for specific tile clicks within factories/center
    const tileClick = this.getTileAt(x, y);
    if (tileClick.factory !== -2 && tileClick.tile !== null) {
      this.selectedFactory = tileClick.factory;
      this.selectedTile = tileClick.tile;
      return;
    }

    // Check pattern line clicks
    if (this.selectedFactory !== -2 && this.selectedTile) {
      const lineClick = this.getPatternLineAt(x, y);
      if (lineClick.player === this.gameState.currentPlayer && lineClick.line !== -2) {
        const move: Move = {
          factoryIndex: this.selectedFactory,
          tile: this.selectedTile,
          lineIndex: lineClick.line
        };

        this.executeMove(move);
      }
    }
  }

  private handleMouseMove(event: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Update hovered line
    const lineClick = this.getPatternLineAt(x, y);
    this.hoveredLine = lineClick.player === this.gameState.currentPlayer ? lineClick.line : -2;
    
    // Update hovered tile
    const tileClick = this.getTileAt(x, y);
    this.hoveredFactory = tileClick.factory;
    this.hoveredTile = tileClick.tile;
    
    // Update cursor
    this.canvas.style.cursor = this.hoveredLine !== -2 || tileClick.factory !== -2 ? 'pointer' : 'default';
  }

  private getCenterTilePositions(): Array<{ tile: Tile; x: number; y: number; index: number }> {
    const bounds = this.getCenterBounds();
    
    // Group tiles by color (excluding FirstPlayer token)
    const regularTiles = this.gameState.center.filter(t => t !== Tile.FirstPlayer);
    const tilesByColor = new Map<Tile, Tile[]>();
    
    // Group tiles by their color
    regularTiles.forEach(tile => {
      if (!tilesByColor.has(tile)) {
        tilesByColor.set(tile, []);
      }
      tilesByColor.get(tile)!.push(tile);
    });
    
    const tileSize = 30;
    const spacing = 5;
    const rowSpacing = 8;
    const positions = [];
    
    let currentY = bounds.y + 10;
    let globalIndex = 0;
    
    // Color order for consistent display
    const colorOrder = [Tile.Red, Tile.Blue, Tile.Yellow, Tile.Black, Tile.White];
    
    for (const color of colorOrder) {
      if (tilesByColor.has(color)) {
        const tilesOfColor = tilesByColor.get(color)!;
        let currentX = bounds.x + 10;
        
        // Place all tiles of this color in the current row
        for (let i = 0; i < tilesOfColor.length; i++) {
          positions.push({
            tile: color,
            x: currentX,
            y: currentY,
            index: globalIndex++
          });
          
          currentX += tileSize + spacing;
        }
        
        // Move to next row
        currentY += tileSize + rowSpacing;
      }
    }
    
    return positions;
  }

  private getTileAt(x: number, y: number): { factory: number; tile: Tile | null } {
    // Check center tiles using the same positioning logic
    const centerBounds = this.getCenterBounds();
    
    if (x >= centerBounds.x && x <= centerBounds.x + centerBounds.width && 
        y >= centerBounds.y && y <= centerBounds.y + centerBounds.height) {
      const tilePositions = this.getCenterTilePositions();
      
      for (const { tile, x: tileX, y: tileY } of tilePositions) {
        if (x >= tileX && x <= tileX + 30 && 
            y >= tileY && y <= tileY + 30) {
          return { factory: -1, tile };
        }
      }
    }

    // Check factory tiles
    for (let i = 0; i < this.gameState.factories.length; i++) {
      const factoryBounds = this.getFactoryBounds(i);

      if (x >= factoryBounds.x && x <= factoryBounds.x + factoryBounds.width && 
          y >= factoryBounds.y && y <= factoryBounds.y + factoryBounds.height) {
        
        // Use the same positioning logic as drawing
        const tiles = this.gameState.factories[i];
        const tilePositions = this.getFactoryTilePositions(tiles, factoryBounds.x, factoryBounds.y, factoryBounds.width);
        const tileSize = (factoryBounds.width - 30) / 2;
        
        for (const { tile, x: tileX, y: tileY } of tilePositions) {
          if (x >= tileX && x <= tileX + tileSize && 
              y >= tileY && y <= tileY + tileSize) {
            return { factory: i, tile };
          }
        }
        
        return { factory: i, tile: null };
      }
    }

    return { factory: -2, tile: null };
  }



  private getPatternLineAt(x: number, y: number): { player: number; line: number } {
    const numPlayers = this.gameState.playerBoards.length;
    
    for (let i = 0; i < numPlayers; i++) {
      const boardBounds = this.getPlayerBoardBounds(i);
      
      if (x >= boardBounds.x && x <= boardBounds.x + boardBounds.width && 
          y >= boardBounds.y && y <= boardBounds.y + boardBounds.height) {
        // Check pattern lines
        for (let j = 0; j < 5; j++) {
          const lineBounds = this.getPatternLineBounds(i, j);
          if (x >= lineBounds.x && x <= lineBounds.x + lineBounds.width &&
              y >= lineBounds.y && y <= lineBounds.y + lineBounds.height) {
            return { player: i, line: j };
          }
        }

        // Check floor
        const floorBounds = this.getFloorBounds(i);
        if (x >= floorBounds.x && x <= floorBounds.x + floorBounds.width &&
            y >= floorBounds.y && y <= floorBounds.y + floorBounds.height) {
          return { player: i, line: -1 };
        }
      }
    }

    return { player: -1, line: -2 };
  }

  private executeMove(move: Move): void {
    if (this.gameState.isValidMove(move)) {
      this.gameState.playMove(move);
      this.selectedFactory = -2;
      this.selectedTile = null;
      this.hoveredLine = -2;
      this.hoveredFactory = -2;
      this.hoveredTile = null;
    }
  }

  // Public method to execute a move (for AI)
  public playMove(move: Move): void {
    this.executeMove(move);
  }

  // Update game state
  public updateGameState(gameState: GameState): void {
    this.gameState = gameState;
    this.selectedFactory = -2;
    this.selectedTile = null;
    this.hoveredLine = -2;
    this.hoveredFactory = -2;
    this.hoveredTile = null;
  }
} 