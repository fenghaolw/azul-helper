import { gameState } from './App';
import TileIcon from './TileIcon';
import { TILE_ICON_SIZE } from '../constants';

export default function GameState() {
  if (!gameState.value) {
    return null;
  }

  return (
    <div className="md-card p-3 sm:p-4 lg:p-5">
      <div className="flex items-center gap-2 mb-3 sm:mb-4">
        <div className="w-1 h-6 bg-green-500 rounded-full"></div>
        <h3 className="font-semibold text-base text-gray-900">Factories & Center</h3>
      </div>

      {/* Factories */}
      {gameState.value.factories.map((factory, i) => (
        <div key={i} className="flex items-center gap-2 sm:gap-3 lg:gap-4 mb-2 sm:mb-3 text-sm">
          <span className="font-medium text-gray-700 min-w-[60px] sm:min-w-[70px]">
            Factory {i + 1}:
          </span>
          <div className="flex gap-0.5">
            {factory.length === 0 ? (
              <span className="empty-indicator">(empty)</span>
            ) : (
              factory.map(
                (tile, j) =>
                  tile &&
                  tile.trim() !== '' && <TileIcon key={j} tile={tile} size={TILE_ICON_SIZE} />
              )
            )}
          </div>
        </div>
      ))}

      {/* Center */}
      <div className="flex items-center gap-2 sm:gap-3 lg:gap-4 text-sm">
        <span className="font-medium text-gray-700 min-w-[60px] sm:min-w-[70px]">Center:</span>
        <div className="flex gap-0.5">
          {!gameState.value.center || gameState.value.center.length === 0 ? (
            <span className="empty-indicator">(empty)</span>
          ) : (
            gameState.value.center.map(
              (tile, i) =>
                tile && tile.trim() !== '' && <TileIcon key={i} tile={tile} size={TILE_ICON_SIZE} />
            )
          )}
        </div>
      </div>
    </div>
  );
}
