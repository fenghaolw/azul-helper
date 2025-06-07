import { Player, TileColor } from '../types';
import { Tile } from './Tile';

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
    onFloorClick
}: PlayerBoardsProps) {
    const canPlaceOnLine = (player: Player, lineIndex: number, color: TileColor) => {
        const line = player.patternLines[lineIndex];
        return !line.isComplete &&
            (line.color === null || line.color === color) &&
            selectedColor === color;
    };

    return (
        <div className="player-boards">
            {players.map((player, playerIndex) => (
                <div
                    key={playerIndex}
                    className={`player-board ${playerIndex === currentPlayerIndex ? 'player-board--current' : ''
                        } ${player.isReadyToScore ? 'player-board--ready-to-score' : ''
                        }`}
                >
                    <div className="player-board__header">
                        {player.name}
                    </div>

                    <div className="player-board__content">
                        <div className="player-board__left">
                            <div className="player-board__pattern-lines">
                                {player.patternLines.map((line, lineIndex) => (
                                    <div
                                        key={lineIndex}
                                        className={`pattern-line ${line.isComplete ? 'pattern-line--complete' : ''
                                            } ${canPlaceOnLine(player, lineIndex, selectedColor!) ? 'pattern-line--valid-drop' : ''
                                            }`}
                                        onClick={() => onPatternLineClick(playerIndex, lineIndex)}
                                    >
                                        <div className="pattern-line__slots">
                                            {Array.from({ length: lineIndex + 1 }).map((_, slotIndex) => (
                                                <div
                                                    key={slotIndex}
                                                    className={`pattern-line__slot ${slotIndex < line.tiles.length ? 'pattern-line__slot--filled' : ''
                                                        } ${canPlaceOnLine(player, lineIndex, selectedColor!) ? 'pattern-line__slot--valid-drop' : ''
                                                        }`}
                                                >
                                                    {slotIndex < line.tiles.length && (
                                                        <Tile color={line.tiles[slotIndex].color} />
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                        <span className="pattern-line__arrow">→</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="player-board__right">
                            <div className="player-board__wall">
                                {player.wall.map((row, rowIndex) =>
                                    row.map((slot, colIndex) => (
                                        <div
                                            key={`${rowIndex}-${colIndex}`}
                                            className={`wall-slot ${slot.isFilled ? 'wall-slot--filled' : ''
                                                } ${slot.isScoring ? 'wall-slot--scoring' : ''
                                                }`}
                                        >
                                            {slot.isFilled && (
                                                <Tile color={slot.color} />
                                            )}
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="player-board__floor">
                        <div className="player-board__floor-title">Chão (-1, -1, -2, -2, -2, -3, -3)</div>
                        <div
                            className="player-board__floor-content"
                            onClick={() => onFloorClick(playerIndex)}
                        >
                            {player.floorTiles.map((tile, index) => (
                                <Tile
                                    key={`${tile.id}-${index}`}
                                    color={tile.color}
                                />
                            ))}
                        </div>
                    </div>

                    <div className="player-board__score">
                        Pontuação: {player.score}
                    </div>
                </div>
            ))}
        </div>
    );
}
