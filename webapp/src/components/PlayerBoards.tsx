import {Player, TileColor} from '../types';
import {Tile} from './Tile';

// Wall pattern for Azul (each row is shifted by one position)
const WALL_PATTERN: TileColor[][] = [
  ['blue', 'yellow', 'red', 'black', 'white'],
  ['white', 'blue', 'yellow', 'red', 'black'],
  ['black', 'white', 'blue', 'yellow', 'red'],
  ['red', 'black', 'white', 'blue', 'yellow'],
  ['yellow', 'red', 'black', 'white', 'blue'],
];

interface PlayerBoardsProps {
  players: Player[];
  currentPlayerIndex: number;
  selectedColor: TileColor | null;
  onPatternLineClick: (playerIndex: number, lineIndex: number) => void;
  onFloorClick: (playerIndex: number) => void;
}

export function PlayerBoards({
  players,
  currentPlayerIndex,
  selectedColor,
  onPatternLineClick,
  onFloorClick,
}: PlayerBoardsProps) {
  const canPlaceOnLine = (
    player: Player,
    lineIndex: number,
    color: TileColor,
    playerIndex: number
  ) => {
    const line = player.patternLines[lineIndex];
    return (
      playerIndex === currentPlayerIndex && // Only current player can place tiles
      !line.isComplete &&
      (line.color === null || line.color === color) &&
      selectedColor === color
    );
  };

  const handlePatternLineClick = (playerIndex: number, lineIndex: number) => {
    // Only allow clicks on current player's board
    if (playerIndex === currentPlayerIndex) {
      onPatternLineClick(playerIndex, lineIndex);
    }
  };

  const handleFloorClick = (playerIndex: number) => {
    // Only allow clicks on current player's board
    if (playerIndex === currentPlayerIndex) {
      onFloorClick(playerIndex);
    }
  };

  return (
    <div className="player-boards">
      {players.map((player, playerIndex) => (
        <div
          key={playerIndex}
          className={`player-board ${
            playerIndex === currentPlayerIndex
              ? 'player-board--current'
              : 'player-board--inactive'
          } ${player.isReadyToScore ? 'player-board--ready-to-score' : ''}`}
        >
          <div className="player-board__header">{player.name}</div>

          <div className="player-board__content">
            <div className="player-board__left">
              <div className="player-board__pattern-lines">
                {player.patternLines.map((line, lineIndex) => (
                  <div
                    key={lineIndex}
                    className={`pattern-line ${
                      line.isComplete ? 'pattern-line--complete' : ''
                    } ${
                      canPlaceOnLine(
                        player,
                        lineIndex,
                        selectedColor!,
                        playerIndex
                      )
                        ? 'pattern-line--valid-drop'
                        : ''
                    }`}
                    onClick={() =>
                      handlePatternLineClick(playerIndex, lineIndex)
                    }
                  >
                    <div className="pattern-line__slots">
                      {Array.from({length: lineIndex + 1}).map(
                        (_, slotIndex) => (
                          <div
                            key={slotIndex}
                            className={`pattern-line__slot ${
                              slotIndex < line.tiles.length
                                ? 'pattern-line__slot--filled'
                                : ''
                            } ${
                              canPlaceOnLine(
                                player,
                                lineIndex,
                                selectedColor!,
                                playerIndex
                              )
                                ? 'pattern-line__slot--valid-drop'
                                : ''
                            }`}
                          >
                            {slotIndex < line.tiles.length && (
                              <Tile color={line.tiles[slotIndex].color} />
                            )}
                          </div>
                        )
                      )}
                    </div>
                    <span className="pattern-line__arrow">→</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="player-board__right">
              <div className="player-board__wall">
                {WALL_PATTERN.map((row, rowIndex) =>
                  row.map((expectedColor, colIndex) => {
                    const slot = player.wall[rowIndex][colIndex];
                    return (
                      <div
                        key={`${rowIndex}-${colIndex}`}
                        className={`wall-slot ${
                          slot.isFilled
                            ? 'wall-slot--filled'
                            : 'wall-slot--empty'
                        } ${slot.isScoring ? 'wall-slot--scoring' : ''}`}
                      >
                        <Tile
                          color={expectedColor}
                          className={slot.isFilled ? '' : 'tile--faded'}
                        />
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          <div className="player-board__floor">
            <div className="player-board__floor-title">Floor</div>
            <div
              className="player-board__floor-content"
              onClick={() => handleFloorClick(playerIndex)}
            >
              <div className="floor-penalties">
                {[-1, -1, -2, -2, -2, -3, -3].map((penalty, index) => (
                  <div key={index} className="floor-penalty-slot">
                    <div className="floor-penalty-value">{penalty}</div>
                    <div className="floor-tile-slot">
                      {index < player.floorTiles.length && (
                        <Tile color={player.floorTiles[index].color} />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="player-board__score">Score: {player.score}</div>
        </div>
      ))}
    </div>
  );
}
