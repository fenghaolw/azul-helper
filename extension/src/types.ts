export interface PlayerBoard {
  id: string;
  lines: string[][];
  wall: string[][];
  floor: string[];
  score: number;
}

export interface GameStateData {
  factories: string[][];
  center: string[];
  playerBoards: PlayerBoard[];
  currentPlayerBoard: PlayerBoard;
  opponentBoard: PlayerBoard;
  currentPlayer: number;
  round: number;
}

export interface Move {
  factoryIndex: number;
  tile: string;
  lineIndex: number;
}

export interface AnalysisStats {
  nodes: number;
  depth: number;
  time: number;
  score: number;
}

export interface AnalysisResponse {
  move: Move;
  stats: AnalysisStats;
  error?: string;
}

export interface TileSVGs {
  [key: string]: string;
}

export interface DifficultySettings {
  [key: number]: {
    time: number;
    label: string;
  };
}
