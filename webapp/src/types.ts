export enum Tile {
  Red = 'red',
  Blue = 'blue',
  Yellow = 'yellow',
  Black = 'black',
  White = 'white',
  FirstPlayer = 'firstPlayer'
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
  GameEnd
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
