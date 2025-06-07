import { render, h } from 'preact';
import { BaseGameState } from './GameState';
import { Tile, Move } from './types';
import { Game } from './components/Game';
import { ApiAI } from './ApiAI';
import {
  GameState,
  TileColor,
  Factory,
  CenterTile,
  Player,
  PatternLine,
  WallSlot,
} from './types';

export class GameRenderer {
  private container: HTMLElement;
  private gameState: BaseGameState;
  private selectedFactory: number = -2; // -2 = none, -1 = center, 0+ = factory
  private selectedTile: Tile | null = null;
  private ai: ApiAI | null = null;
  private aiEnabled: boolean = false;

  constructor(container: HTMLElement, gameState: BaseGameState) {
    this.container = container;
    this.gameState = gameState;

    // Set up the container
    this.setupContainer();

    // Set up event listeners for game interactions
    this.setupEventListeners();

    // Initial render
    this.render();
  }

  private setupContainer(): void {
    this.container.className = 'azul-game-container';
    this.container.innerHTML = ''; // Clear any existing content
  }

  private setupEventListeners(): void {
    // Listen for factory selections
    this.container.addEventListener('factorySelected', (event: Event) => {
      const customEvent = event as CustomEvent;
      const { factoryIndex, color } = customEvent.detail;
      this.handleFactoryClick(factoryIndex, color);
    });

    // Listen for center selections
    this.container.addEventListener('centerSelected', (event: Event) => {
      const customEvent = event as CustomEvent;
      const { groupIndex, color } = customEvent.detail;
      this.handleCenterClick(groupIndex, color);
    });

    // Listen for pattern line selections
    this.container.addEventListener('patternLineSelected', (event: Event) => {
      const customEvent = event as CustomEvent;
      const { playerIndex, lineIndex, color } = customEvent.detail;
      this.handlePatternLineClick(playerIndex, lineIndex, color);
    });

    // Listen for floor selections
    this.container.addEventListener('floorSelected', (event: Event) => {
      const customEvent = event as CustomEvent;
      const { playerIndex, color } = customEvent.detail;
      this.handleFloorClick(playerIndex, color);
    });
  }

  private handleFactoryClick(factoryIndex: number, color: TileColor): void {
    // Find the tile in the factory
    const factory = this.gameState.factories[factoryIndex];
    const tile = factory.find(t => this.mapTileColor(t) === color);

    if (tile) {
      this.selectedFactory = factoryIndex;
      this.selectedTile = tile;
    }
  }

  private handleCenterClick(_groupIndex: number, color: TileColor): void {
    // Find the tile in center
    const centerTiles = this.gameState.center;
    const tile = centerTiles.find((t: any) => this.mapTileColor(t) === color);

    if (tile) {
      this.selectedFactory = -1; // Center
      this.selectedTile = tile;
    }
  }

  private handlePatternLineClick(
    playerIndex: number,
    lineIndex: number,
    _color: TileColor
  ): void {
    if (
      this.selectedTile &&
      this.isValidPatternLineDrop(playerIndex, lineIndex)
    ) {
      const move: Move = {
        factoryIndex: this.selectedFactory,
        tile: this.selectedTile,
        lineIndex: lineIndex,
      };
      this.playMove(move);
    }
  }

  private handleFloorClick(playerIndex: number, _color: TileColor): void {
    if (this.selectedTile && this.isValidFloorDrop(playerIndex)) {
      const move: Move = {
        factoryIndex: this.selectedFactory,
        tile: this.selectedTile,
        lineIndex: -1,
      };
      this.playMove(move);
    }
  }

  private mapTileColor(tile: Tile): TileColor {
    switch (tile) {
      case Tile.Red:
        return 'red';
      case Tile.Blue:
        return 'blue';
      case Tile.Yellow:
        return 'yellow';
      case Tile.Black:
        return 'black';
      case Tile.White:
        return 'white';
      case Tile.FirstPlayer:
        return 'first-player';
      default:
        return 'white';
    }
  }

  private convertGameState(): GameState {
    // Convert factories
    const factories: Factory[] = this.gameState.factories.map(factory => ({
      tiles: factory.map(tile => ({
        color: this.mapTileColor(tile),
        id: `${tile}-${Math.random()}`,
      })),
      isEmpty: factory.length === 0,
    }));

    // Convert center tiles
    const centerTileGroups = new Map<TileColor, number>();
    this.gameState.center.forEach((tile: any) => {
      const color = this.mapTileColor(tile);
      centerTileGroups.set(color, (centerTileGroups.get(color) || 0) + 1);
    });

    const centerTiles: CenterTile[] = Array.from(
      centerTileGroups.entries()
    ).map(([color, count]) => ({
      color,
      count,
    }));

    // Convert players
    const players: Player[] = this.gameState.playerBoards.map(
      (board, index) => {
        // Convert pattern lines
        const patternLines: PatternLine[] = board.lines.map(
          (line: any, lineIndex: any) => ({
            color: line.length > 0 ? this.mapTileColor(line[0]) : null,
            tiles: line.map((tile: any) => ({
              color: this.mapTileColor(tile),
              id: `${tile}-${Math.random()}`,
            })),
            capacity: lineIndex + 1,
            isComplete: line.length === lineIndex + 1,
          })
        );

        // Convert wall
        const wall: WallSlot[][] = board.wall.map(row =>
          row.map(slot => ({
            color: slot ? this.mapTileColor(slot) : 'white',
            isFilled: slot !== null,
            isScoring: false,
          }))
        );

        // Convert floor tiles
        const floorTiles = board.floor.map((tile: any) => ({
          color: this.mapTileColor(tile),
          id: `${tile}-${Math.random()}`,
        }));

        return {
          name: `Player ${index + 1}`,
          score: board.score,
          patternLines,
          wall,
          floorTiles,
          isReadyToScore: false, // Simplified for now
        };
      }
    );

    return {
      factories,
      centerTiles,
      players,
      currentPlayerIndex: this.gameState.currentPlayer,
      round: this.gameState.round,
      gamePhase: this.gameState.gameOver ? 'finished' : 'playing',
    };
  }

  render(): void {
    try {
      const gameState = this.convertGameState();

      // Render the Preact component with gameState as prop
      render(
        h(Game, { gameContainer: this.container, gameState }),
        this.container
      );

      // Also dispatch the event for backwards compatibility
      this.container.dispatchEvent(
        new CustomEvent('gameStateUpdate', {
          detail: gameState,
        })
      );
    } catch (error) {
      console.error('Error in GameRenderer.render():', error);
    }
  }

  private isValidPatternLineDrop(
    playerIndex: number,
    lineIndex: number
  ): boolean {
    if (!this.selectedTile) return false;

    const board = this.gameState.playerBoards[playerIndex];
    const line = board.lines[lineIndex];

    // Check if line is already complete
    if (line.length === lineIndex + 1) return false;

    // Check if line is empty or has same color
    if (line.length === 0) return true;
    return line[0] === this.selectedTile;
  }

  private isValidFloorDrop(_playerIndex: number): boolean {
    return this.selectedTile !== null;
  }

  public setAIEnabled(enabled: boolean) {
    this.aiEnabled = enabled;
    if (enabled && !this.ai) {
      this.ai = new ApiAI(1); // AI plays as player 1
    }
    this.checkForAIMove();
  }

  private async checkForAIMove() {
    if (!this.aiEnabled || !this.ai || this.gameState.gameOver) {
      return;
    }

    // If it's AI's turn
    if (this.gameState.currentPlayer === 1) {
      try {
        const result = await this.ai.getBestMove(this.gameState);
        this.playMove(result.move);
      } catch (error) {
        console.error('AI move error:', error);
      }
    }
  }

  public playMove(move: Move): void {
    // Clear selection
    this.selectedFactory = -2;
    this.selectedTile = null;

    // Apply the move to the game state
    this.gameState.playMove(move);

    // Re-render
    this.render();

    // Check for AI move after a short delay
    setTimeout(() => this.checkForAIMove(), 500);
  }

  public updateGameState(gameState: BaseGameState): void {
    this.gameState = gameState;
    this.render();
    this.checkForAIMove();
  }
}
