export type TileColor =
  | "red"
  | "blue"
  | "yellow"
  | "black"
  | "white"
  | "first-player";

// Original Tile enum for compatibility with existing game logic
export enum Tile {
  Red = "red",
  Blue = "blue",
  Yellow = "yellow",
  Black = "black",
  White = "white",
  FirstPlayer = "firstPlayer",
}

// New Tile interface for Preact components
export interface TileData {
  color: TileColor;
  id: string;
}

export interface Factory {
  tiles: TileData[];
  isEmpty: boolean;
}

export interface CenterTile {
  color: TileColor;
  count: number;
}

export interface PatternLine {
  color: TileColor | null;
  tiles: TileData[];
  capacity: number;
  isComplete: boolean;
}

export interface WallSlot {
  color: TileColor;
  isFilled: boolean;
  isScoring?: boolean;
}

export interface Player {
  name: string;
  score: number;
  patternLines: PatternLine[];
  wall: WallSlot[][];
  floorTiles: TileData[];
  isReadyToScore: boolean;
}

export interface GameState {
  factories: Factory[];
  centerTiles: CenterTile[];
  players: Player[];
  currentPlayerIndex: number;
  round: number;
  gamePhase: "playing" | "scoring" | "finished";
}

export interface GameEvents {
  factorySelected: { factoryIndex: number; color: TileColor };
  centerSelected: { groupIndex: number; color: TileColor };
  patternLineSelected: {
    playerIndex: number;
    lineIndex: number;
    color: TileColor;
  };
  floorSelected: { playerIndex: number; color: TileColor };
  gameStateUpdate: GameState;
}

export interface Move {
  factoryIndex: number; // -1 for center
  tile: Tile;
  lineIndex: number; // 0-4 for pattern lines, -1 for floor
}

export interface GameResult {
  winner: number;
  scores: number[];
  gameOver: boolean;
}

export enum GamePhase {
  TileSelection,
  WallTiling,
  GameEnd,
}

export interface SearchResult {
  move: Move;
  value: number;
  depth: number;
  nodesEvaluated: number;
}

export interface TilePlacementDetail {
  tile: Tile;
  row: number;
  col: number;
  score: number;
  adjacentTiles: {
    horizontal: number;
    vertical: number;
  };
}

export interface FloorPenaltyDetail {
  tile: Tile;
  position: number;
  penalty: number;
}

export interface ScoreDetails {
  tilesPlaced: TilePlacementDetail[];
  floorPenalties: FloorPenaltyDetail[];
  totalTileScore: number;
  totalFloorPenalty: number;
  previousScore: number;
  newScore: number;
}

export interface FinalScoreDetails {
  completedRows: number;
  completedColumns: number;
  completedColors: number;
  rowBonus: number;
  columnBonus: number;
  colorBonus: number;
  previousScore: number;
}
