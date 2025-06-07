import {CenterTile, TileColor} from '../types';
import {Tile} from './Tile';

interface CenterAreaProps {
  centerTiles: CenterTile[];
  selectedGroup: number | null;
  selectedColor: TileColor | null;
  onCenterClick: (groupIndex: number, color: TileColor) => void;
}

// Generate random but consistent positions for center tiles
const getCenterTilePosition = (
  groupIndex: number,
  tileIndex: number,
  totalTiles: number
) => {
  const seed = groupIndex * 100 + tileIndex;

  // Create better spread arrangement that scales with tile count
  const angle = (seed * 67.5) % 360;
  const maxRadius = Math.min(60, Math.max(30, totalTiles * 4)); // Scale radius based on tile count
  const radius = 10 + ((seed * 0.15) % maxRadius);
  const x = Math.cos((angle * Math.PI) / 180) * radius;
  const y = Math.sin((angle * Math.PI) / 180) * radius;
  const rotation = ((seed * 31) % 40) - 20; // Limited rotation between -20 and +20 degrees

  return {
    transform: `translate(${x}px, ${y}px) rotate(${rotation}deg)`,
    zIndex: tileIndex,
    '--hover-transform': `translate(${x}px, ${y}px) rotate(${rotation}deg) scale(1.15)`,
  };
};

export function CenterArea({
  centerTiles,
  selectedGroup,
  selectedColor,
  onCenterClick,
}: CenterAreaProps) {
  const handleGroupClick = (groupIndex: number, color: TileColor) => {
    onCenterClick(groupIndex, color);
  };

  // Convert center tiles to individual tiles for scattered display
  const allCenterTiles = centerTiles.flatMap((group, groupIndex) =>
    Array.from({length: group.count}, (_, tileIndex) => ({
      color: group.color,
      groupIndex,
      tileIndex,
      id: `${group.color}-${groupIndex}-${tileIndex}`,
    }))
  );

  return (
    <div className="center-area">
      <div className="center-area__content">
        {allCenterTiles.length === 0 ? (
          <div className="center-area__empty"></div>
        ) : (
          <div className="center-area__tiles">
            {allCenterTiles.map(tile => (
              <Tile
                key={tile.id}
                color={tile.color}
                isSelected={
                  selectedGroup === tile.groupIndex &&
                  selectedColor === tile.color
                }
                onClick={
                  tile.color !== 'first-player'
                    ? () => handleGroupClick(tile.groupIndex, tile.color)
                    : undefined
                }
                className="center-area__tile"
                style={getCenterTilePosition(
                  tile.groupIndex,
                  tile.tileIndex,
                  allCenterTiles.length
                )}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
