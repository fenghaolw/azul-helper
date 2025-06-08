import { useState, useEffect, useRef, useContext } from "preact/hooks";
import { WebAppGameState } from "../GameState";
import { GameRenderer } from "../GameRenderer";
import { Tile } from "../types";
import { PlayerBoard } from "../PlayerBoard";
import { route } from "preact-router";
import { ReplayDataContext } from "../main";
import { Sidebar } from "./Sidebar";

interface ReplayData {
  agent1_name: string;
  agent2_name: string;
  moves: Move[];
  initial_state: any;
  final_state: any;
  final_scores: number[];
}

interface Move {
  player: number;
  agent: string;
  action: string;
  state_after: any;
}

interface GameReplayProps {
  replayData: ReplayData;
}

// Helper function to parse JSON game state into internal game state
function parseGameState(stateJson: any, replayData: ReplayData) {
  console.log("Parsing game state:", stateJson);
  const state = new WebAppGameState();
  state.round = stateJson.roundNumber || 1;
  state.currentPlayer = stateJson.currentPlayer || 0;
  state.factories = [];
  state.center = [];
  state.playerBoards = [
    new PlayerBoard(replayData.agent1_name),
    new PlayerBoard(replayData.agent2_name),
  ];

  // Helper function to convert color code to tile
  const colorToTile = (color: string): Tile | null => {
    switch (color.toUpperCase()) {
      case "R":
        return Tile.Red;
      case "B":
        return Tile.Blue;
      case "Y":
        return Tile.Yellow;
      case "K":
        return Tile.Black;
      case "W":
        return Tile.White;
      case "F":
        return Tile.FirstPlayer;
      default:
        return null;
    }
  };

  // Parse factories
  if (stateJson.factories) {
    console.log("Parsing factories:", stateJson.factories);
    state.factories = stateJson.factories.map((factory: any) => {
      const tiles: Tile[] = [];
      for (const [color, count] of Object.entries(factory)) {
        const tileCount = count as number;
        if (tileCount > 0) {
          const tile = colorToTile(color);
          if (tile) {
            for (let i = 0; i < tileCount; i++) {
              tiles.push(tile);
            }
          }
        }
      }
      console.log("Parsed factory tiles:", tiles);
      return tiles;
    });
  }

  // Parse center
  if (stateJson.center) {
    console.log("Parsing center:", stateJson.center);
    for (const [color, count] of Object.entries(stateJson.center)) {
      const tileCount = count as number;
      if (tileCount > 0) {
        const tile = colorToTile(color);
        if (tile) {
          for (let i = 0; i < tileCount; i++) {
            state.center.push(tile);
          }
        }
      }
    }
    console.log("Parsed center tiles:", state.center);
  }

  // Parse player boards
  if (stateJson.playerBoards) {
    console.log("Parsing player boards:", stateJson.playerBoards);
    stateJson.playerBoards.forEach((board: any, playerIndex: number) => {
      const playerBoard = state.playerBoards[playerIndex];
      playerBoard.score = board.score || 0;

      // Parse pattern lines
      if (board.patternLines) {
        console.log(
          `Parsing pattern lines for player ${playerIndex}:`,
          board.patternLines,
        );
        board.patternLines.forEach((line: any, lineIndex: number) => {
          if (line.color && line.count > 0) {
            const tile = colorToTile(line.color);
            if (tile) {
              playerBoard.lines[lineIndex] = Array(line.count).fill(tile);
            }
          }
        });
        console.log(
          `Parsed pattern lines for player ${playerIndex}:`,
          playerBoard.lines,
        );
      }

      // Parse wall
      if (board.wall) {
        console.log(`Parsing wall for player ${playerIndex}:`, board.wall);
        board.wall.forEach((row: any[], rowIndex: number) => {
          row.forEach((cell: number, colIndex: number) => {
            if (cell === 1) {
              // Determine tile color based on wall position
              const colors = [
                Tile.Red,
                Tile.Blue,
                Tile.Yellow,
                Tile.Black,
                Tile.White,
              ];
              const colorIndex = (rowIndex + colIndex) % 5;
              playerBoard.wall[rowIndex][colIndex] = colors[colorIndex];
            }
          });
        });
        console.log(`Parsed wall for player ${playerIndex}:`, playerBoard.wall);
      }

      // Parse floor
      if (board.floor) {
        console.log(`Parsing floor for player ${playerIndex}:`, board.floor);
        playerBoard.floor = [];
        board.floor.forEach((tile: string) => {
          const tileType = colorToTile(tile);
          if (tileType) {
            playerBoard.floor.push(tileType);
          }
        });
        console.log(
          `Parsed floor for player ${playerIndex}:`,
          playerBoard.floor,
        );
      }
    });
  }

  console.log("Final parsed state:", state);
  return state;
}

export const GameReplay = ({ replayData }: GameReplayProps) => {
  const { setReplayData } = useContext(ReplayDataContext);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(-1);
  const [gameState, setGameState] = useState<WebAppGameState | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed] = useState(1.0);
  const rendererRef = useRef<GameRenderer | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const currentMoveRef = useRef<HTMLDivElement>(null);

  // Initialize game state from replay data
  useEffect(() => {
    if (replayData && replayData.initial_state) {
      console.log("Initial replay data:", replayData);
      const initialState = parseGameState(replayData.initial_state, replayData);
      setGameState(initialState);
    }
  }, [replayData]);

  // Initialize renderer when container is ready
  useEffect(() => {
    if (containerRef.current && !rendererRef.current) {
      console.log("Initializing renderer with state:", gameState);
      const newRenderer = new GameRenderer(
        containerRef.current,
        gameState || new WebAppGameState(),
      );
      rendererRef.current = newRenderer;
    }
  }, [containerRef.current]);

  // Update renderer when game state changes
  useEffect(() => {
    if (rendererRef.current && gameState) {
      console.log("Updating renderer with new state:", gameState);
      rendererRef.current.updateGameState(gameState);
    }
  }, [gameState]);

  // Update game state when move index changes
  useEffect(() => {
    if (currentMoveIndex >= 0 && currentMoveIndex < replayData.moves.length) {
      const move = replayData.moves[currentMoveIndex];
      const newState = parseGameState(move.state_after, replayData);
      setGameState(newState);
    } else if (currentMoveIndex === replayData.moves.length) {
      // Use final state when we reach the end
      const finalState = parseGameState(replayData.final_state, replayData);
      setGameState(finalState);
    }
  }, [currentMoveIndex, replayData.moves, replayData.final_state]);

  // Handle playback
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isPlaying && currentMoveIndex < replayData.moves.length) {
      timer = setTimeout(() => {
        setCurrentMoveIndex(currentMoveIndex + 1);
      }, 1000 / playbackSpeed);
    } else if (currentMoveIndex >= replayData.moves.length) {
      setIsPlaying(false);
    }
    return () => clearTimeout(timer);
  }, [isPlaying, currentMoveIndex, replayData.moves.length, playbackSpeed]);

  // Scroll current move into view when it changes
  useEffect(() => {
    if (currentMoveRef.current) {
      currentMoveRef.current.scrollIntoView({
        behavior: "smooth",
        block: "center",
        inline: "nearest",
      });
    }
  }, [currentMoveIndex]);

  const handleStepForward = () => {
    if (currentMoveIndex < replayData.moves.length - 1) {
      setCurrentMoveIndex(currentMoveIndex + 1);
    }
  };

  const handleStepBackward = () => {
    if (currentMoveIndex > -1) {
      setCurrentMoveIndex(currentMoveIndex - 1);
    }
  };

  const handleReset = () => {
    setCurrentMoveIndex(-1);
    setIsPlaying(false);
    if (replayData && replayData.initial_state) {
      const initialState = parseGameState(replayData.initial_state, replayData);
      setGameState(initialState);
    }
  };

  const handleBackToGame = () => {
    setReplayData(null);
    route("/");
  };

  const handleMoveClick = (index: number) => {
    setCurrentMoveIndex(index);
  };

  return (
    <div className="azul-app">
      <Sidebar
        title="Game Replay"
        subtitle={`${replayData.agent1_name} vs ${replayData.agent2_name}`}
      >
        <div className="controls">
          <div className="controls__buttons">
            <button
              className="button button--primary"
              onClick={handleReset}
              disabled={currentMoveIndex === -1}
            >
              Reset
            </button>
            <button
              className="button button--primary"
              onClick={handleStepBackward}
              disabled={currentMoveIndex === -1}
            >
              Step Backward
            </button>
            <button
              className="button button--primary"
              onClick={handleStepForward}
              disabled={currentMoveIndex === replayData.moves.length}
            >
              Step Forward
            </button>
            <button
              className="button button--primary"
              onClick={handleBackToGame}
            >
              Back to Game
            </button>
          </div>

          <div className="moves">
            <div className="section-header">
              <div className="section-header__accent"></div>
              <h3 className="section-header__title">Move History</h3>
            </div>
            <div className="moves-list">
              {replayData.moves.map((move: any, index: number) => (
                <div
                  key={index}
                  ref={index === currentMoveIndex ? currentMoveRef : null}
                  className={`move-item ${index === currentMoveIndex ? "move-item--current" : ""}`}
                  onClick={() => handleMoveClick(index)}
                >
                  <div className="move-item__content">
                    <span className="move-item__player">
                      {move.player === 0
                        ? replayData.agent1_name
                        : replayData.agent2_name}
                    </span>
                    <span className="move-item__separator">•</span>
                    <span className="move-item__action">{move.action}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Sidebar>

      <div className="azul-app__game">
        <div ref={containerRef} className="game-replay__board" />
      </div>
    </div>
  );
};
