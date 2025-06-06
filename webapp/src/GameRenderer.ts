import { BaseGameState } from './GameState.js';
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
  private gameState: BaseGameState;
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
    canvas: { width: 1400, height: 700 },
    title: { x: 700, y: 25 },
    factories: {
      startX: 50,
      startY: 80,
      size: 120,
      spacing: 20,
      perRow: 0
    },
    center: {
      x: 100,
      y: 420,
      width: 320,
      height: 120
    },
    playerBoards: {
      startX: 500,
      startY: 120,
      width: 350,
      height: 320,
      spacingX: 50,
      spacingY: 50,
      patternLines: {
        startY: 40,
        height: 30,
        spacing: 5,
        tileSize: 25
      },
      wall: {
        startX: 170,
        tileSize: 25,
        spacing: 2
      },
      floor: {
        startY: 255,
        height: 30
      }
    },
    gameInfo: { x: 1150, y: 570 }
  };

  // Traditional Azul/Portuguese azulejo inspired tile colors
  private tileColors = {
    [Tile.Red]: '#c0392b',     // Deep red like Portuguese terra cotta
    [Tile.Blue]: '#2c3e50',    // Deep navy blue like traditional azulejo
    [Tile.Yellow]: '#f39c12',  // Rich golden yellow
    [Tile.Black]: '#1a1a1a',   // Deep black
    [Tile.White]: '#ecf0f1',   // Creamy white like ceramic
    [Tile.FirstPlayer]: '#ffffff' // Pure white for first player token
  };

  // Standard tile size used across all game areas for consistency
  private readonly STANDARD_TILE_SIZE = 25;

  constructor(canvas: HTMLCanvasElement, gameState: BaseGameState) {
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
      [Tile.FirstPlayer]: 'tile-overlay-dark.svg' // Using dark overlay for first player token
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
      height: this.calculateCenterHeight()
    };
  }

  private calculateCenterHeight(): number {
    // Calculate how many rows of tiles we need
    const regularTiles = this.gameState.center.filter(t => t !== Tile.FirstPlayer);
    const tilesByColor = new Map<Tile, Tile[]>();

    // Group tiles by their color
    regularTiles.forEach(tile => {
      if (!tilesByColor.has(tile)) {
        tilesByColor.set(tile, []);
      }
      tilesByColor.get(tile)!.push(tile);
    });

    // Count unique colors that have tiles
    const colorOrder = [Tile.Red, Tile.Blue, Tile.Yellow, Tile.Black, Tile.White];
    const numRows = colorOrder.filter(color => tilesByColor.has(color)).length;

    if (numRows === 0) {
      // If no regular tiles, use minimum height
      return this.layout.center.height;
    }

    // Calculate required height using standard tile size for consistency
    const tileSize = this.STANDARD_TILE_SIZE;
    const rowSpacing = 8;
    const topPadding = 15; // Increased padding to prevent overlap with decorative border
    const bottomPadding = 15; // Increased padding to prevent overlap with decorative border

    const requiredHeight = topPadding + (numRows - 1) * (tileSize + rowSpacing) + tileSize + bottomPadding;

    // Use at least the original minimum height
    return Math.max(requiredHeight, this.layout.center.height);
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
      x: boardBounds.x + 20,
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
    // Portuguese azulejo inspired background
    const gradient = this.ctx.createLinearGradient(0, 0, this.canvas.width, this.canvas.height);
    gradient.addColorStop(0, '#f8f9fa'); // Creamy white
    gradient.addColorStop(0.5, '#e9ecef'); // Light grey
    gradient.addColorStop(1, '#dee2e6'); // Slightly darker grey

    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Add traditional azulejo pattern
    this.drawAzulejoPattern();
  }

  private drawAzulejoPattern(): void {
    // Draw subtle decorative azulejo-inspired pattern
    this.ctx.globalAlpha = 0.08;
    this.ctx.strokeStyle = '#2c3e50'; // Traditional azulejo blue
    this.ctx.lineWidth = 1;

    const patternSize = 40;

    for (let x = 0; x < this.canvas.width; x += patternSize) {
      for (let y = 0; y < this.canvas.height; y += patternSize) {
        // Draw decorative cross pattern
        this.ctx.beginPath();
        // Vertical line
        this.ctx.moveTo(x + patternSize / 2, y + 10);
        this.ctx.lineTo(x + patternSize / 2, y + patternSize - 10);
        // Horizontal line
        this.ctx.moveTo(x + 10, y + patternSize / 2);
        this.ctx.lineTo(x + patternSize - 10, y + patternSize / 2);
        // Small decorative corners
        this.ctx.moveTo(x + 15, y + 15);
        this.ctx.lineTo(x + 25, y + 15);
        this.ctx.lineTo(x + 25, y + 25);
        this.ctx.moveTo(x + patternSize - 15, y + 15);
        this.ctx.lineTo(x + patternSize - 25, y + 15);
        this.ctx.lineTo(x + patternSize - 25, y + 25);
        this.ctx.moveTo(x + 15, y + patternSize - 15);
        this.ctx.lineTo(x + 25, y + patternSize - 15);
        this.ctx.lineTo(x + 25, y + patternSize - 25);
        this.ctx.moveTo(x + patternSize - 15, y + patternSize - 15);
        this.ctx.lineTo(x + patternSize - 25, y + patternSize - 15);
        this.ctx.lineTo(x + patternSize - 25, y + patternSize - 25);
        this.ctx.stroke();
      }
    }

    this.ctx.globalAlpha = 1.0;
  }

  private drawCeramicCoaster(x: number, y: number, size: number, isSelected: boolean, isHovered: boolean): void {
    // Draw ceramic coaster-style factory display inspired by Portuguese pottery

    // Coaster shadow
    this.ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
    this.ctx.shadowBlur = 6;
    this.ctx.shadowOffsetX = 2;
    this.ctx.shadowOffsetY = 2;

    // Make the coaster larger to accommodate tiles - use 85% of the size
    const coasterRadius = (size * 0.85) / 2;

    // Main ceramic body
    this.ctx.fillStyle = '#f8f9fa'; // Ceramic white
    this.ctx.beginPath();
    this.ctx.arc(x + size / 2, y + size / 2, coasterRadius, 0, 2 * Math.PI);
    this.ctx.fill();

    // Reset shadow
    this.ctx.shadowColor = 'transparent';
    this.ctx.shadowBlur = 0;
    this.ctx.shadowOffsetX = 0;
    this.ctx.shadowOffsetY = 0;

    // Decorative rim
    this.ctx.strokeStyle = '#2c3e50'; // Traditional blue
    this.ctx.lineWidth = 3;
    this.ctx.beginPath();
    this.ctx.arc(x + size / 2, y + size / 2, coasterRadius, 0, 2 * Math.PI);
    this.ctx.stroke();

    // Inner decorative ring
    this.ctx.strokeStyle = '#d4af37'; // Gold accent
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.arc(x + size / 2, y + size / 2, coasterRadius - 4, 0, 2 * Math.PI);
    this.ctx.stroke();

    // Selection/hover state with enhanced visual feedback
    if (isSelected || isHovered) {
      if (isHovered && !isSelected) {
        // Enhanced hover effect with subtle glow
        this.ctx.shadowColor = 'rgba(212, 175, 55, 0.4)'; // Golden glow
        this.ctx.shadowBlur = 8;
        this.ctx.strokeStyle = '#d4af37'; // Gold for hover
        this.ctx.lineWidth = 3;
      } else {
        this.ctx.strokeStyle = isSelected ? '#d4af37' : '#85929e';
        this.ctx.lineWidth = isSelected ? 4 : 2;
      }
      this.ctx.beginPath();
      this.ctx.arc(x + size / 2, y + size / 2, coasterRadius + 2, 0, 2 * Math.PI);
      this.ctx.stroke();

      // Reset shadow
      this.ctx.shadowColor = 'transparent';
      this.ctx.shadowBlur = 0;
    }

    // Traditional azulejo-style decorative pattern in the center (smaller to not interfere with tiles)
    const centerX = x + size / 2;
    const centerY = y + size / 2;
    const patternRadius = size / 8; // Smaller pattern

    this.ctx.strokeStyle = '#2c3e50';
    this.ctx.lineWidth = 1;
    this.ctx.globalAlpha = 0.3; // Make pattern more subtle
    this.ctx.beginPath();
    // Draw small cross pattern
    this.ctx.moveTo(centerX - patternRadius, centerY);
    this.ctx.lineTo(centerX + patternRadius, centerY);
    this.ctx.moveTo(centerX, centerY - patternRadius);
    this.ctx.lineTo(centerX, centerY + patternRadius);
    // Small decorative corners
    for (let i = 0; i < 4; i++) {
      const angle = (i * Math.PI) / 2;
      const cornerX = centerX + Math.cos(angle) * patternRadius * 0.7;
      const cornerY = centerY + Math.sin(angle) * patternRadius * 0.7;
      this.ctx.moveTo(cornerX - 2, cornerY);
      this.ctx.lineTo(cornerX + 2, cornerY);
      this.ctx.moveTo(cornerX, cornerY - 2);
      this.ctx.lineTo(cornerX, cornerY + 2);
    }
    this.ctx.stroke();
    this.ctx.globalAlpha = 1.0; // Reset opacity
  }

  private drawOrnateCenterTable(bounds: any, isSelected: boolean, isHovered: boolean): void {
    // Draw ornate center table inspired by Portuguese azulejo patterns

    // Table shadow
    this.ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
    this.ctx.shadowBlur = 10;
    this.ctx.shadowOffsetX = 4;
    this.ctx.shadowOffsetY = 4;

    // Main table surface with ceramic-like gradient
    const gradient = this.ctx.createLinearGradient(bounds.x, bounds.y, bounds.x, bounds.y + bounds.height);
    if (isSelected) {
      gradient.addColorStop(0, '#ecf0f1');
      gradient.addColorStop(0.5, '#d5dbdb');
      gradient.addColorStop(1, '#bdc3c7');
    } else if (isHovered) {
      gradient.addColorStop(0, '#f8f9fa');
      gradient.addColorStop(0.5, '#e9ecef');
      gradient.addColorStop(1, '#dee2e6');
    } else {
      gradient.addColorStop(0, '#ffffff');
      gradient.addColorStop(0.5, '#f8f9fa');
      gradient.addColorStop(1, '#e9ecef');
    }

    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);

    // Reset shadow
    this.ctx.shadowColor = 'transparent';
    this.ctx.shadowBlur = 0;
    this.ctx.shadowOffsetX = 0;
    this.ctx.shadowOffsetY = 0;

    // Ornate border with Portuguese tile pattern
    this.ctx.strokeStyle = '#2c3e50';
    this.ctx.lineWidth = 3;
    this.ctx.strokeRect(bounds.x, bounds.y, bounds.width, bounds.height);

    // Inner decorative border
    this.ctx.strokeStyle = '#d4af37'; // Gold accent
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(bounds.x + 4, bounds.y + 4, bounds.width - 8, bounds.height - 8);

    // Traditional azulejo corner decorations
    const cornerSize = 20;
    const corners = [
      { x: bounds.x + 8, y: bounds.y + 8 }, // Top-left
      { x: bounds.x + bounds.width - cornerSize - 8, y: bounds.y + 8 }, // Top-right
      { x: bounds.x + 8, y: bounds.y + bounds.height - cornerSize - 8 }, // Bottom-left
      { x: bounds.x + bounds.width - cornerSize - 8, y: bounds.y + bounds.height - cornerSize - 8 } // Bottom-right
    ];

    this.ctx.strokeStyle = '#2c3e50';
    this.ctx.lineWidth = 1;

    corners.forEach(corner => {
      // Draw traditional azulejo corner pattern
      this.ctx.beginPath();
      // Diagonal lines
      this.ctx.moveTo(corner.x, corner.y + cornerSize / 2);
      this.ctx.lineTo(corner.x + cornerSize / 2, corner.y);
      this.ctx.moveTo(corner.x + cornerSize / 2, corner.y + cornerSize);
      this.ctx.lineTo(corner.x + cornerSize, corner.y + cornerSize / 2);
      // Small decorative elements
      this.ctx.moveTo(corner.x + 5, corner.y + 5);
      this.ctx.lineTo(corner.x + 10, corner.y + 5);
      this.ctx.moveTo(corner.x + 5, corner.y + 5);
      this.ctx.lineTo(corner.x + 5, corner.y + 10);
      this.ctx.stroke();
    });

    // Selection/hover highlight with enhanced visual feedback
    if (isSelected || isHovered) {
      if (isHovered && !isSelected) {
        // Enhanced hover effect with subtle glow
        this.ctx.shadowColor = 'rgba(212, 175, 55, 0.3)'; // Golden glow
        this.ctx.shadowBlur = 6;
        this.ctx.strokeStyle = '#d4af37'; // Gold for hover
        this.ctx.lineWidth = 3;
      } else {
        this.ctx.strokeStyle = isSelected ? '#d4af37' : '#85929e';
        this.ctx.lineWidth = isSelected ? 4 : 2;
      }
      this.ctx.strokeRect(bounds.x - 2, bounds.y - 2, bounds.width + 4, bounds.height + 4);

      // Reset shadow
      this.ctx.shadowColor = 'transparent';
      this.ctx.shadowBlur = 0;
    }
  }

  private drawAzulejoPlayerBoard(x: number, y: number, width: number, height: number, isCurrentPlayer: boolean): void {
    // Draw Portuguese azulejo-inspired player board

    // Board shadow
    this.ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
    this.ctx.shadowBlur = 6;
    this.ctx.shadowOffsetX = 3;
    this.ctx.shadowOffsetY = 3;

    // Main board background with ceramic gradient
    const gradient = this.ctx.createLinearGradient(x, y, x, y + height);
    if (isCurrentPlayer) {
      gradient.addColorStop(0, '#f8f9fa');
      gradient.addColorStop(0.5, '#e8f6f3');
      gradient.addColorStop(1, '#d5e8df');
    } else {
      gradient.addColorStop(0, '#ffffff');
      gradient.addColorStop(0.5, '#f8f9fa');
      gradient.addColorStop(1, '#ecf0f1');
    }

    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(x, y, width, height);

    // Reset shadow
    this.ctx.shadowColor = 'transparent';
    this.ctx.shadowBlur = 0;
    this.ctx.shadowOffsetX = 0;
    this.ctx.shadowOffsetY = 0;

    // Traditional blue border
    this.ctx.strokeStyle = '#2c3e50';
    this.ctx.lineWidth = isCurrentPlayer ? 4 : 3;
    this.ctx.strokeRect(x, y, width, height);

    // Gold accent border for current player
    if (isCurrentPlayer) {
      this.ctx.strokeStyle = '#d4af37';
      this.ctx.lineWidth = 2;
      this.ctx.strokeRect(x + 3, y + 3, width - 6, height - 6);
    }

    // Inner decorative border
    this.ctx.strokeStyle = '#85929e';
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(x + 6, y + 6, width - 12, height - 12);

    // Small corner decorations
    const cornerSize = 12;
    const corners = [
      { x: x + 10, y: y + 10 },
      { x: x + width - cornerSize - 10, y: y + 10 },
      { x: x + 10, y: y + height - cornerSize - 10 },
      { x: x + width - cornerSize - 10, y: y + height - cornerSize - 10 }
    ];

    this.ctx.strokeStyle = '#2c3e50';
    this.ctx.lineWidth = 1;

    corners.forEach(corner => {
      this.ctx.beginPath();
      // Small decorative cross
      this.ctx.moveTo(corner.x + cornerSize / 2 - 3, corner.y + cornerSize / 2);
      this.ctx.lineTo(corner.x + cornerSize / 2 + 3, corner.y + cornerSize / 2);
      this.ctx.moveTo(corner.x + cornerSize / 2, corner.y + cornerSize / 2 - 3);
      this.ctx.lineTo(corner.x + cornerSize / 2, corner.y + cornerSize / 2 + 3);
      this.ctx.stroke();
    });
  }

  private drawTitle(): void {
    const centerX = this.layout.title.x;
    const titleY = this.layout.title.y;

    // Decorative border around title
    this.ctx.strokeStyle = '#2c3e50';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(centerX - 120, titleY + 10);
    this.ctx.lineTo(centerX - 60, titleY + 10);
    this.ctx.moveTo(centerX + 60, titleY + 10);
    this.ctx.lineTo(centerX + 120, titleY + 10);
    this.ctx.stroke();

    // Main title with Portuguese influence
    this.ctx.fillStyle = '#2c3e50'; // Traditional azulejo blue
    this.ctx.font = 'bold 36px "Georgia", serif'; // More traditional serif font
    this.ctx.textAlign = 'center';
    this.ctx.fillText('AZUL', centerX, titleY);

    // Add gold accent to title
    this.ctx.strokeStyle = '#d4af37'; // Gold
    this.ctx.lineWidth = 1;
    this.ctx.strokeText('AZUL', centerX, titleY);

    // Subtitle with Portuguese inspiration
    this.ctx.fillStyle = '#5d6d7e';
    this.ctx.font = 'italic 14px "Georgia", serif';
    this.ctx.fillText('Strategic tile-laying board game', centerX, titleY + 24);
  }

  private drawFactories(): void {
    for (let i = 0; i < this.gameState.factories.length; i++) {
      const bounds = this.getFactoryBounds(i);
      this.drawFactory(bounds.x, bounds.y, bounds.width, this.gameState.factories[i], i);
    }
  }

  private getFactoryTilePositions(tiles: Tile[], factoryX: number, factoryY: number, factorySize: number): Array<{ tile: Tile; x: number; y: number; index: number }> {
    // Show each tile individually in a 2x2 grid using standard tile size
    const tileSize = this.STANDARD_TILE_SIZE;
    const tileSpacing = 6; // Spacing between tiles
    const positions = [];

    // Calculate the grid dimensions to center it
    const gridWidth = 2 * tileSize + tileSpacing;
    const gridHeight = 2 * tileSize + tileSpacing;
    const startX = factoryX + (factorySize - gridWidth) / 2;
    const startY = factoryY + (factorySize - gridHeight) / 2;

    for (let i = 0; i < Math.min(tiles.length, 4); i++) {
      const tileRow = Math.floor(i / 2);
      const tileCol = i % 2;
      const tileX = startX + tileCol * (tileSize + tileSpacing);
      const tileY = startY + tileRow * (tileSize + tileSpacing);

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

    // Draw ceramic coaster-style factory display
    this.drawCeramicCoaster(x, y, size, isSelected, isHovered);

    // Get tile positions using the same logic as hit detection
    const tilePositions = this.getFactoryTilePositions(tiles, x, y, size);
    const tileSize = this.STANDARD_TILE_SIZE;

    // Draw tiles
    for (const { tile, x: tileX, y: tileY } of tilePositions) {
      // Enhanced highlight for selected/hovered tiles
      const isTileSelected = this.selectedTile === tile && this.selectedFactory === factoryIndex;
      const isTileHovered = this.hoveredTile === tile && this.hoveredFactory === factoryIndex;

      if (isTileSelected || isTileHovered) {
        if (isTileHovered && !isTileSelected) {
          // Enhanced hover with subtle glow
          this.ctx.shadowColor = 'rgba(212, 175, 55, 0.6)';
          this.ctx.shadowBlur = 8;
          this.ctx.fillStyle = '#fff8dc'; // Warm cream highlight
        } else {
          this.ctx.fillStyle = '#f1c40f'; // Bright yellow for selection
        }
        this.ctx.fillRect(tileX - 4, tileY - 4, tileSize + 8, tileSize + 8);

        // Reset shadow
        this.ctx.shadowColor = 'transparent';
        this.ctx.shadowBlur = 0;
      }

      this.drawTile(tileX, tileY, tileSize, tile);
    }
  }

  private drawCenter(): void {
    const bounds = this.getCenterBounds();
    const isSelected = this.selectedFactory === -1;
    const isHovered = this.hoveredFactory === -1;

    // Draw ornate center table inspired by Portuguese tile work
    this.drawOrnateCenterTable(bounds, isSelected, isHovered);

    // Get tile positions using the same logic as hit detection
    const tilePositions = this.getCenterTilePositions();

    // Draw regular tiles
    for (const { tile, x, y } of tilePositions) {
      // Enhanced highlight for selected/hovered tiles
      const isTileSelected = this.selectedTile === tile && this.selectedFactory === -1;
      const isTileHovered = this.hoveredTile === tile && this.hoveredFactory === -1;

      if (isTileSelected || isTileHovered) {
        if (isTileHovered && !isTileSelected) {
          // Enhanced hover with subtle glow
          this.ctx.shadowColor = 'rgba(212, 175, 55, 0.6)';
          this.ctx.shadowBlur = 8;
          this.ctx.fillStyle = '#fff8dc'; // Warm cream highlight
        } else {
          this.ctx.fillStyle = '#f1c40f'; // Bright yellow for selection
        }
        this.ctx.fillRect(x - 4, y - 4, this.STANDARD_TILE_SIZE + 8, this.STANDARD_TILE_SIZE + 8);

        // Reset shadow
        this.ctx.shadowColor = 'transparent';
        this.ctx.shadowBlur = 0;
      }

      this.drawTile(x, y, this.STANDARD_TILE_SIZE, tile);
    }

    // Draw FirstPlayer token separately if present
    if (this.gameState.center.includes(Tile.FirstPlayer)) {
      const tokenX = bounds.x + bounds.width - this.STANDARD_TILE_SIZE - 15; // Match increased padding
      const tokenY = bounds.y + 15; // Match increased padding

      this.drawTile(tokenX, tokenY, this.STANDARD_TILE_SIZE, Tile.FirstPlayer);
    }

    // Center label with azulejo styling
    this.ctx.fillStyle = '#2c3e50';
    this.ctx.font = 'bold 16px "Georgia", serif';
    this.ctx.textAlign = 'left';
    this.ctx.strokeStyle = '#ffffff';
    this.ctx.lineWidth = 3;
    this.ctx.strokeText('Table Center', bounds.x, bounds.y - 8);
    this.ctx.fillText('Table Center', bounds.x, bounds.y - 8);

    // Add helpful subtitle if there are tiles
    const regularTiles = this.gameState.center.filter(t => t !== Tile.FirstPlayer);
    if (regularTiles.length > 0) {
      this.ctx.fillStyle = '#5d6d7e';
      this.ctx.font = 'italic 12px "Georgia", serif';
      this.ctx.fillText('(organized by color)', bounds.x, bounds.y + 8);
    }
  }

  private drawPlayerBoards(): void {
    const numPlayers = this.gameState.playerBoards.length;

    for (let i = 0; i < numPlayers; i++) {
      const bounds = this.getPlayerBoardBounds(i);
      // Draw score above the player board, centered
      const scoreText = `Score: ${this.gameState.playerBoards[i].score}`;
      this.ctx.fillStyle = '#333';
      this.ctx.font = 'bold 18px "Arial", sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.fillText(scoreText, bounds.x + bounds.width / 2, bounds.y - 15);
      // Draw the player board itself
      this.drawPlayerBoard(bounds.x, bounds.y, bounds.width, bounds.height, this.gameState.playerBoards[i], i);
    }
  }

  private drawPlayerBoard(x: number, y: number, width: number, height: number, board: PlayerBoard, playerIndex: number): void {
    const isCurrentPlayer = playerIndex === this.gameState.currentPlayer;

    // Background with azulejo pattern
    this.drawAzulejoPlayerBoard(x, y, width, height, isCurrentPlayer);

    // Player name and score
    this.ctx.fillStyle = isCurrentPlayer ? '#2c3e50' : '#5d6d7e';
    this.ctx.font = isCurrentPlayer ? 'bold 18px "Georgia", serif' : 'bold 16px "Georgia", serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(`Player ${playerIndex + 1}`, x + 10, y + 25);

    this.ctx.font = 'bold 16px "Georgia", serif';
    this.ctx.fillStyle = isCurrentPlayer ? '#d4af37' : '#5d6d7e';
    this.ctx.fillText(`Score: ${board.score}`, x + 10, y + height - 10);

    // Draw pattern lines with scoring indicators
    const patternY = y + this.layout.playerBoards.patternLines.startY;
    for (let i = 0; i < 5; i++) {
      const lineY = patternY + i * (this.layout.playerBoards.patternLines.height + this.layout.playerBoards.patternLines.spacing);

      // Check if line is complete and ready to score
      const isComplete = board.lines[i].length === i + 1;
      if (isComplete && board.lines[i].length > 0) {
        // Draw scoring indicator
        this.ctx.fillStyle = 'rgba(56, 142, 60, 0.2)';
        this.ctx.fillRect(x + 5, lineY - 2, 160, this.layout.playerBoards.patternLines.height + 4);

        // Add "READY!" text
        this.ctx.fillStyle = '#388e3c';
        this.ctx.font = 'bold 10px "Arial", sans-serif';
        this.ctx.textAlign = 'right';
        this.ctx.fillText('READY!', x + 160, lineY + this.layout.playerBoards.patternLines.height - 5);
        this.ctx.textAlign = 'left';
      }

      this.drawPatternLine(x + 10, lineY, i, board.lines[i], playerIndex);
    }

    // Draw wall with completion indicators
    const wallBounds = this.getWallBounds(playerIndex);
    this.drawWall(wallBounds.x, wallBounds.y, board.wall);

    // Add visual indicators for completed rows/columns/colors
    this.drawCompletionIndicators(wallBounds, board);

    // Draw floor with penalty indicator
    const floorY = y + this.layout.playerBoards.floor.startY;
    this.drawFloor(x + 10, floorY, width - 20, board.floor, playerIndex);

    // Show floor penalty if there are tiles
    if (board.floor.length > 0) {
      const penalties = [-1, -1, -2, -2, -2, -3, -3];
      let totalPenalty = 0;
      for (let i = 0; i < Math.min(board.floor.length, penalties.length); i++) {
        totalPenalty += penalties[i];
      }

      if (totalPenalty < 0) {
        this.ctx.fillStyle = '#d32f2f';
        this.ctx.font = 'bold 12px "Arial", sans-serif';
        this.ctx.textAlign = 'right';
        this.ctx.fillText(`${totalPenalty} pts`, x + width - 10, floorY + this.layout.playerBoards.floor.height - 5);
        this.ctx.textAlign = 'left';
      }
    }
  }

  private drawCompletionIndicators(wallBounds: any, board: PlayerBoard): void {
    // Check for completed rows
    for (let row = 0; row < 5; row++) {
      if (board.wall[row].filter(tile => tile !== null).length === 5) {
        // Draw row completion indicator
        this.ctx.strokeStyle = '#388e3c';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(wallBounds.x - 5, wallBounds.y + row * (wallBounds.tileSize + wallBounds.spacing) + wallBounds.tileSize / 2);
        this.ctx.lineTo(wallBounds.x + 5 * (wallBounds.tileSize + wallBounds.spacing), wallBounds.y + row * (wallBounds.tileSize + wallBounds.spacing) + wallBounds.tileSize / 2);
        this.ctx.stroke();

        // Add "+2" indicator
        this.ctx.fillStyle = '#388e3c';
        this.ctx.font = 'bold 10px "Arial", sans-serif';
        this.ctx.textAlign = 'right';
        this.ctx.fillText('+2', wallBounds.x + 5 * (wallBounds.tileSize + wallBounds.spacing) + 15, wallBounds.y + row * (wallBounds.tileSize + wallBounds.spacing) + wallBounds.tileSize / 2 + 3);
        this.ctx.textAlign = 'left';
      }
    }

    // Check for completed columns
    for (let col = 0; col < 5; col++) {
      let columnComplete = true;
      for (let row = 0; row < 5; row++) {
        if (board.wall[row][col] === null) {
          columnComplete = false;
          break;
        }
      }

      if (columnComplete) {
        // Draw column completion indicator
        this.ctx.strokeStyle = '#1976d2';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(wallBounds.x + col * (wallBounds.tileSize + wallBounds.spacing) + wallBounds.tileSize / 2, wallBounds.y - 5);
        this.ctx.lineTo(wallBounds.x + col * (wallBounds.tileSize + wallBounds.spacing) + wallBounds.tileSize / 2, wallBounds.y + 5 * (wallBounds.tileSize + wallBounds.spacing));
        this.ctx.stroke();

        // Add "+7" indicator
        this.ctx.fillStyle = '#1976d2';
        this.ctx.font = 'bold 10px "Arial", sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('+7', wallBounds.x + col * (wallBounds.tileSize + wallBounds.spacing) + wallBounds.tileSize / 2, wallBounds.y - 10);
        this.ctx.textAlign = 'left';
      }
    }
  }

  private drawPatternLine(x: number, y: number, lineIndex: number, tiles: Tile[], playerIndex: number): void {
    const tileSize = this.STANDARD_TILE_SIZE;
    const maxTiles = lineIndex + 1;

    // Line background with enhanced hover effect
    const isHovered = this.hoveredLine === lineIndex && this.gameState.currentPlayer === playerIndex;

    if (isHovered) {
      // Enhanced hover effect with golden glow
      this.ctx.shadowColor = 'rgba(212, 175, 55, 0.4)';
      this.ctx.shadowBlur = 6;
      this.ctx.fillStyle = '#f4f1e8'; // Warm cream color for hover
    } else {
      this.ctx.fillStyle = '#ecf0f1';
    }
    this.ctx.fillRect(x, y, maxTiles * (tileSize + 2), tileSize);

    // Reset shadow
    this.ctx.shadowColor = 'transparent';
    this.ctx.shadowBlur = 0;

    // Line border with enhanced hover styling
    this.ctx.strokeStyle = isHovered ? '#d4af37' : '#bdc3c7';
    this.ctx.lineWidth = isHovered ? 2 : 1;
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

  private drawWall(x: number, y: number, wall: Array<Array<Tile | null>>): void {
    const tileSize = this.STANDARD_TILE_SIZE;
    const spacing = this.layout.playerBoards.wall.spacing;
    const wallPattern = PlayerBoard.WALL_PATTERN;

    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 5; col++) {
        const tileX = x + col * (tileSize + spacing);
        const tileY = y + row * (this.layout.playerBoards.patternLines.height + this.layout.playerBoards.patternLines.spacing);
        const expectedTile = wallPattern[row][col];
        const actualTileOnWall = wall[row][col];

        if (actualTileOnWall !== null && actualTileOnWall === expectedTile) {
          this.drawTile(tileX, tileY, tileSize, actualTileOnWall);
        } else {
          this.drawPlaceholderTile(tileX, tileY, tileSize, expectedTile);
        }
      }
    }
  }

  private drawFloor(x: number, y: number, width: number, floor: Tile[], playerIndex: number): void {
    const floorHeight = 30;

    // Floor background with enhanced azulejo styling and hover effect
    const isHovered = this.hoveredLine === -1 && this.gameState.currentPlayer === playerIndex;

    if (isHovered) {
      // Enhanced hover effect with warm glow
      this.ctx.shadowColor = 'rgba(192, 57, 43, 0.4)'; // Red glow
      this.ctx.shadowBlur = 6;
    }

    const gradient = this.ctx.createLinearGradient(x, y, x + width, y);
    if (isHovered) {
      gradient.addColorStop(0, '#f8cecc'); // Brighter warm colors for hover
      gradient.addColorStop(1, '#f1948a');
    } else {
      gradient.addColorStop(0, '#f8d7da');
      gradient.addColorStop(1, '#fadbd8');
    }

    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(x, y, width, floorHeight);

    // Reset shadow
    this.ctx.shadowColor = 'transparent';
    this.ctx.shadowBlur = 0;

    // Enhanced ornate floor border
    this.ctx.strokeStyle = isHovered ? '#e74c3c' : '#c0392b'; // Brighter red for hover
    this.ctx.lineWidth = isHovered ? 3 : 2;
    this.ctx.strokeRect(x, y, width, floorHeight);

    // Inner decorative line
    this.ctx.strokeStyle = '#d4af37'; // Gold accent
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(x + 2, y + 2, width - 4, floorHeight - 4);

    // Floor label with azulejo styling
    this.ctx.fillStyle = '#c0392b';
    this.ctx.font = 'bold 12px "Georgia", serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText('Floor', x + 5, y + 18);

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
      // Add the "1" text with white color and text outline for better visibility
      this.ctx.fillStyle = '#ffffff';
      this.ctx.strokeStyle = '#000000';
      this.ctx.lineWidth = 2;
      this.ctx.font = `700 ${Math.round(size * 0.5)}px "Roboto", sans-serif`;
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';

      // Draw text outline first
      this.ctx.strokeText('1', x + size / 2, y + size / 2);
      // Then draw filled text
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

    // Game info with azulejo styling
    this.ctx.fillStyle = '#2c3e50';
    this.ctx.font = 'bold 18px "Georgia", serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(`Round ${this.gameState.round}`, infoX, infoY);

    if (!this.gameState.gameOver) {
      this.ctx.fillText(`Current Player: ${this.gameState.currentPlayer + 1}`, infoX, infoY + 25);
      this.ctx.fillText(`Available Moves: ${this.gameState.availableMoves.length}`, infoX, infoY + 50);

      // Show scoring hint for current player
      if (this.gameState.currentPlayer !== undefined) {
        const currentBoard = this.gameState.playerBoards[this.gameState.currentPlayer];
        let completedLines = 0;
        for (let i = 0; i < 5; i++) {
          if (currentBoard.lines[i].length === i + 1) {
            completedLines++;
          }
        }

        if (completedLines > 0) {
          this.ctx.fillStyle = '#388e3c';
          this.ctx.font = 'italic 14px "Georgia", serif';
          this.ctx.fillText(`${completedLines} line${completedLines > 1 ? 's' : ''} ready to score!`, infoX, infoY + 75);
        }
      }
    } else {
      const result = this.gameState.getResult();
      if (result.winner !== -1) {
        this.ctx.fillStyle = '#388e3c';
        this.ctx.font = 'bold 20px "Georgia", serif';
        this.ctx.fillText(`Winner: Player ${result.winner + 1}!`, infoX, infoY + 25);
      } else {
        this.ctx.fillStyle = '#f57c00';
        this.ctx.font = 'bold 20px "Georgia", serif';
        this.ctx.fillText('Game ended in a tie!', infoX, infoY + 25);
      }

      // Show final bonuses summary
      this.ctx.fillStyle = '#2c3e50';
      this.ctx.font = 'bold 16px "Georgia", serif';
      this.ctx.fillText('Final Bonuses:', infoX, infoY + 55);

      let yOffset = 75;
      for (let i = 0; i < this.gameState.playerBoards.length; i++) {
        const board = this.gameState.playerBoards[i];
        const finalResult = board.getFinalScoreCalculation();

        this.ctx.fillStyle = '#5d6d7e';
        this.ctx.font = '14px "Georgia", serif';
        this.ctx.fillText(`Player ${i + 1}: +${finalResult.bonus}`, infoX, infoY + yOffset);

        if (finalResult.bonus > 0) {
          let bonusDetails = '';
          if (finalResult.details.completedRows > 0) bonusDetails += `${finalResult.details.completedRows}R `;
          if (finalResult.details.completedColumns > 0) bonusDetails += `${finalResult.details.completedColumns}C `;
          if (finalResult.details.completedColors > 0) bonusDetails += `${finalResult.details.completedColors}Color `;

          this.ctx.fillStyle = '#388e3c';
          this.ctx.font = 'italic 12px "Georgia", serif';
          this.ctx.fillText(`(${bonusDetails.trim()})`, infoX + 80, infoY + yOffset);
        }

        yOffset += 20;
      }
    }

    // Selection info
    if (this.selectedFactory !== -2 && this.selectedTile) {
      const factoryText = this.selectedFactory === -1 ? 'Center' : `Factory ${this.selectedFactory + 1}`;
      this.ctx.fillStyle = '#2c3e50';
      this.ctx.font = 'bold 16px "Georgia", serif';

      // Draw a small tile indicator
      const tileX = infoX;
      const tileY = infoY + 120;
      this.drawTile(tileX, tileY, this.STANDARD_TILE_SIZE, this.selectedTile);

      this.ctx.fillText(`Selected: ${this.selectedTile.charAt(0).toUpperCase() + this.selectedTile.slice(1)} tiles from ${factoryText}`, infoX + 30, infoY + 135);
      this.ctx.font = 'italic 14px "Georgia", serif';
      this.ctx.fillStyle = '#5d6d7e';
      this.ctx.fillText('Click a pattern line or floor to place tiles', infoX, infoY + 155);
    } else if (this.hoveredTile && this.hoveredFactory !== -2) {
      const factoryText = this.hoveredFactory === -1 ? 'Center' : `Factory ${this.hoveredFactory + 1}`;
      this.ctx.fillStyle = '#5d6d7e';
      this.ctx.font = 'italic 14px "Georgia", serif';
      this.ctx.fillText(`Hover: ${this.hoveredTile.charAt(0).toUpperCase() + this.hoveredTile.slice(1)} tiles in ${factoryText}`, infoX, infoY + 125);
      this.ctx.fillText('Click to select these tiles', infoX, infoY + 145);
    }
  }

  private drawAvailableMoves(): void {
    if (this.selectedFactory === -2 || !this.selectedTile) return;

    // Highlight valid destinations with azulejo styling
    const currentBoard = this.gameState.playerBoards[this.gameState.currentPlayer];

    for (let i = 0; i < 5; i++) {
      if (currentBoard.canPlaceTile(this.selectedTile, i)) {
        // Highlight pattern line with gold
        const lineBounds = this.getPatternLineBounds(this.gameState.currentPlayer, i);

        this.ctx.strokeStyle = '#d4af37'; // Gold
        this.ctx.lineWidth = 4;
        this.ctx.strokeRect(lineBounds.x - 3, lineBounds.y - 3, lineBounds.width + 6, lineBounds.height + 6);

        // Inner highlight
        this.ctx.strokeStyle = '#f1c40f'; // Bright yellow
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(lineBounds.x - 1, lineBounds.y - 1, lineBounds.width + 2, lineBounds.height + 2);

        // Also highlight the corresponding wall position
        const wallBounds = this.getWallBounds(this.gameState.currentPlayer);
        const wallPattern = PlayerBoard.WALL_PATTERN;

        // Find the column where this tile would go in the wall
        let wallCol = -1;
        for (let col = 0; col < 5; col++) {
          if (wallPattern[i][col] === this.selectedTile) {
            wallCol = col;
            break;
          }
        }

        if (wallCol !== -1) {
          const wallTileX = wallBounds.x + wallCol * (wallBounds.tileSize + wallBounds.spacing);
          const wallTileY = wallBounds.y + i * (this.layout.playerBoards.patternLines.height + this.layout.playerBoards.patternLines.spacing);

          // Highlight the wall position with a different color to distinguish it
          this.ctx.strokeStyle = '#27ae60'; // Green for wall position
          this.ctx.lineWidth = 3;
          this.ctx.strokeRect(wallTileX - 2, wallTileY - 2, wallBounds.tileSize + 4, wallBounds.tileSize + 4);

          // Inner highlight
          this.ctx.strokeStyle = '#2ecc71'; // Bright green
          this.ctx.lineWidth = 1;
          this.ctx.strokeRect(wallTileX, wallTileY, wallBounds.tileSize, wallBounds.tileSize);
        }
      }
    }

    // Always highlight floor as valid with Portuguese red
    const floorBounds = this.getFloorBounds(this.gameState.currentPlayer);

    this.ctx.strokeStyle = '#c0392b'; // Deep red
    this.ctx.lineWidth = 4;
    this.ctx.strokeRect(floorBounds.x - 3, floorBounds.y - 3, floorBounds.width + 6, floorBounds.height + 6);

    // Inner highlight
    this.ctx.strokeStyle = '#e74c3c'; // Bright red
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(floorBounds.x - 1, floorBounds.y - 1, floorBounds.width + 2, floorBounds.height + 2);
  }

  private handleClick(event: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    // Convert mouse coordinates to canvas logical coordinates
    // The canvas has logical size set via CSS, so we need to use that ratio
    const scaleX = this.layout.canvas.width / rect.width;
    const scaleY = this.layout.canvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    // Debug logging to verify coordinate transformation
    console.log(`Click: client(${event.clientX - rect.left}, ${event.clientY - rect.top}) -> canvas(${x.toFixed(1)}, ${y.toFixed(1)}) scale(${scaleX.toFixed(2)}, ${scaleY.toFixed(2)})`);

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
    // Convert mouse coordinates to canvas logical coordinates
    // The canvas has logical size set via CSS, so we need to use that ratio
    const scaleX = this.layout.canvas.width / rect.width;
    const scaleY = this.layout.canvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

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

    const regularTiles = this.gameState.center.filter(t => t !== Tile.FirstPlayer);
    const tilesByColor = new Map<Tile, Tile[]>();

    regularTiles.forEach(tile => {
      if (!tilesByColor.has(tile)) {
        tilesByColor.set(tile, []);
      }
      tilesByColor.get(tile)!.push(tile);
    });

    const tileSize = this.STANDARD_TILE_SIZE;
    const spacing = 5;
    const rowSpacing = 8;
    const positions: Array<{ tile: Tile; x: number; y: number; index: number }> = [];

    let currentY = bounds.y + 15;
    let globalIndex = 0;

    const colorOrder = [Tile.Red, Tile.Blue, Tile.Yellow, Tile.Black, Tile.White];

    for (const color of colorOrder) {
      if (tilesByColor.has(color)) {
        const tilesOfColor = tilesByColor.get(color)!;
        let currentX = bounds.x + 15;

        for (let i = 0; i < tilesOfColor.length; i++) {
          positions.push({
            tile: color,
            x: currentX,
            y: currentY,
            index: globalIndex++
          });
          currentX += tileSize + spacing;
        }
        currentY += tileSize + rowSpacing;
      }
    }
    return positions;
  }

  private getTileAt(x: number, y: number): { factory: number; tile: Tile | null } {
    // Check center tiles using the same positioning logic - check tile positions directly
    const tilePositions = this.getCenterTilePositions();

    for (const { tile, x: tileX, y: tileY } of tilePositions) {
      if (x >= tileX && x <= tileX + this.STANDARD_TILE_SIZE &&
        y >= tileY && y <= tileY + this.STANDARD_TILE_SIZE) {
        return { factory: -1, tile };
      }
    }

    // Also check FirstPlayer token if present (drawn separately)
    if (this.gameState.center.includes(Tile.FirstPlayer)) {
      const centerBounds = this.getCenterBounds();
      const tokenX = centerBounds.x + centerBounds.width - this.STANDARD_TILE_SIZE - 15; // Match increased padding
      const tokenY = centerBounds.y + 15; // Match increased padding

      if (x >= tokenX && x <= tokenX + this.STANDARD_TILE_SIZE &&
        y >= tokenY && y <= tokenY + this.STANDARD_TILE_SIZE) {
        return { factory: -1, tile: Tile.FirstPlayer };
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
        const tileSize = this.STANDARD_TILE_SIZE; // Use same size as drawing logic

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
  public updateGameState(gameState: BaseGameState): void {
    this.gameState = gameState;
    this.selectedFactory = -2;
    this.selectedTile = null;
    this.hoveredLine = -2;
    this.hoveredFactory = -2;
    this.hoveredTile = null;
  }
}
