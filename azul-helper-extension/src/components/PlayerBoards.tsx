import { gameState } from './App';
import TileIcon from './TileIcon';
import { PlayerBoard } from '../types';
import { TILE_ICON_SIZE } from '../constants';

function PlayerBoardComponent({
  board,
  isCurrentPlayer,
}: {
  board: PlayerBoard;
  isCurrentPlayer: boolean;
}) {
  const playerName = isCurrentPlayer ? 'Current Player' : 'Opponent';

  return (
    <div
      className={`
      md-card p-3 sm:p-4 lg:p-5 mb-2 text-sm
      ${isCurrentPlayer ? 'ring-2 ring-blue-500 ring-opacity-50 bg-blue-50' : ''}
    `}
    >
      <div className="mb-2">
        <span className="font-semibold text-sm text-gray-900">{playerName}</span>
      </div>

      {/* Pattern Lines & Wall - Side by Side with responsive gap */}
      <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:gap-6 mb-3">
        {/* Pattern Lines */}
        <div>
          <div className="font-medium text-gray-700 text-xs uppercase tracking-wide mb-2">
            Pattern Lines
          </div>
          <div className="space-y-1">
            {Array.from({ length: 5 }, (_, i) => {
              const line = board.lines[i] || [];
              return (
                <div key={i} className="flex items-center gap-1 min-h-4">
                  <span className="font-medium text-gray-700 text-xs w-3 text-center">{i + 1}</span>
                  <div className="flex gap-0.5 flex-1">
                    {line.length === 0 ? (
                      <span className="empty-indicator text-xs">—</span>
                    ) : (
                      line.map(
                        (tile, j) =>
                          tile &&
                          tile.trim() !== '' && (
                            <TileIcon key={j} tile={tile} size={TILE_ICON_SIZE} />
                          )
                      )
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Wall */}
        <div>
          <div className="font-medium text-gray-700 text-xs uppercase tracking-wide mb-2">Wall</div>
          <div className="grid grid-cols-5 gap-px">
            {Array.from({ length: 25 }, (_, index) => {
              const row = Math.floor(index / 5);
              const col = index % 5;
              const wallTile = board.wall[row] && board.wall[row][col];
              const hasTile = wallTile && wallTile.trim() !== '';

              return (
                <div key={index} className={`wall-spot w-6 h-6 ${hasTile ? 'filled' : ''}`}>
                  {hasTile && <TileIcon tile={wallTile} size={TILE_ICON_SIZE} />}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Floor Line */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-gray-700 text-xs uppercase tracking-wide">Floor:</span>
          <div className="flex gap-0.5 items-center">
            {!board.floor || board.floor.length === 0 ? (
              <span className="empty-indicator text-xs">—</span>
            ) : (
              board.floor.map(
                (tile, i) =>
                  tile &&
                  tile.trim() !== '' && <TileIcon key={i} tile={tile} size={TILE_ICON_SIZE} />
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function PlayerBoards() {
  if (!gameState.value) {
    return null;
  }

  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <div className="w-1 h-6 bg-purple-500 rounded-full"></div>
        <h3 className="font-semibold text-base text-gray-900">Player Boards</h3>
      </div>
      {gameState.value.playerBoards.map((board, index) => (
        <PlayerBoardComponent
          key={board.id}
          board={board}
          isCurrentPlayer={index === gameState.value!.currentPlayer}
        />
      ))}
    </div>
  );
}
