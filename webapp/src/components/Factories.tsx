import { Factory, TileColor } from "../types";
import { Tile } from "./Tile";

interface FactoriesProps {
  factories: Factory[];
  selectedFactory: number | null;
  selectedColor: TileColor | null;
  onFactoryClick: (factoryIndex: number, color: TileColor) => void;
}

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
                  className={`factory__tile factory__tile--${index}-${tileIndex}`}
                />
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
