import { Factory, TileColor } from "../types";
import { Tile } from "./Tile";

interface FactoriesProps {
  factories: Factory[];
  selectedFactory: number | null;
  selectedColor: TileColor | null;
  onFactoryClick: (factoryIndex: number, color: TileColor) => void;
}

// Generate consistent but random-looking positions for tiles
const getTilePosition = (factoryIndex: number, tileIndex: number) => {
  const seed = factoryIndex * 1000 + tileIndex;
  const angle = (seed * 137.5) % 360; // Golden angle for good distribution
  const radius = 8 + ((seed * 0.05) % 15); // Smaller radius between 8-23px from center
  const x = Math.cos((angle * Math.PI) / 180) * radius;
  const y = Math.sin((angle * Math.PI) / 180) * radius;
  const rotation = ((seed * 23) % 60) - 30; // Limited rotation between -30 and +30 degrees

  return {
    transform: `translate(${x}px, ${y}px) rotate(${rotation}deg)`,
    zIndex: tileIndex,
    "--hover-transform": `translate(${x}px, ${y}px) rotate(${rotation}deg) scale(1.1)`,
  };
};

export function Factories({
  factories,
  selectedFactory,
  selectedColor,
  onFactoryClick,
}: FactoriesProps) {
  const handleFactoryClick = (factoryIndex: number, color: TileColor) => {
    onFactoryClick(factoryIndex, color);
  };

  return (
    <div className="factories">
      <h2 className="factories__title">Factories</h2>
      {factories.map((factory, index) => (
        <div key={index} className="factories__factory">
          <div
            className={`factory ${factory.isEmpty ? "factory--empty" : ""} ${
              selectedFactory === index ? "factory--selected" : ""
            }`}
            onClick={() =>
              factory.tiles.length > 0 &&
              handleFactoryClick(index, factory.tiles[0].color)
            }
          >
            <div className="factory__tiles">
              {factory.tiles.map((tile, tileIndex) => (
                <Tile
                  key={`${tile.id}-${tileIndex}`}
                  color={tile.color}
                  isSelected={
                    selectedFactory === index && selectedColor === tile.color
                  }
                  onClick={(e) => {
                    e?.stopPropagation();
                    handleFactoryClick(index, tile.color);
                  }}
                  className="factory__tile"
                  style={getTilePosition(index, tileIndex)}
                />
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
