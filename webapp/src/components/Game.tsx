import { useState, useEffect } from "preact/hooks";
import { GameState, TileColor } from "../types";

import { CenterArea } from "./CenterArea";
import { PlayerBoards } from "./PlayerBoards";

interface GameProps {
  gameContainer: HTMLElement;
  gameState?: GameState | null;
  aiEnabled: boolean;
}

export function Game({
  gameContainer,
  gameState: initialGameState,
  aiEnabled,
}: GameProps) {
  const [gameState, setGameState] = useState<GameState | null>(
    initialGameState || null,
  );
  const [selectedFactory, setSelectedFactory] = useState<number | null>(null);
  const [selectedCenterGroup, setSelectedCenterGroup] = useState<number | null>(
    null,
  );
  const [selectedColor, setSelectedColor] = useState<TileColor | null>(null);

  // Generate consistent but random-looking positions for tiles
  const getTilePosition = (factoryIndex: number, tileIndex: number) => {
    const seed = factoryIndex * 1000 + tileIndex;
    const angle = (seed * 137.5) % 360; // Golden angle for good distribution
    const radius = 15 + ((seed * 0.08) % 30); // Larger radius between 15-45px from center
    const x = Math.cos((angle * Math.PI) / 180) * radius;
    const y = Math.sin((angle * Math.PI) / 180) * radius;
    const rotation = ((seed * 23) % 60) - 30; // Limited rotation between -30 and +30 degrees

    return {
      transform: `translate(${x}px, ${y}px) rotate(${rotation}deg)`,
      zIndex: tileIndex,
      "--hover-transform": `translate(${x}px, ${y}px) rotate(${rotation}deg) scale(1.1)`,
    };
  };

  // Get correct tile image path (same logic as Tile component)
  const getTileImagePath = (color: TileColor): string => {
    const basePath = "/imgs/";
    switch (color) {
      case "red":
        return `${basePath}tile-red.svg`;
      case "blue":
        return `${basePath}tile-blue.svg`;
      case "yellow":
        return `${basePath}tile-yellow.svg`;
      case "black":
        return `${basePath}tile-black.svg`;
      case "white":
        return `${basePath}tile-turquoise.svg`; // Using turquoise for white
      case "first-player":
        return `${basePath}tile-overlay-dark.svg`;
      default:
        return `${basePath}tile-turquoise.svg`;
    }
  };

  // Update state when prop changes
  useEffect(() => {
    if (initialGameState) {
      setGameState(initialGameState);
    }
  }, [initialGameState]);

  // Listen for game state updates via events (backup method)
  useEffect(() => {
    const handleGameStateUpdate = (event: CustomEvent<GameState>) => {
      setGameState(event.detail);
    };

    gameContainer.addEventListener(
      "gameStateUpdate",
      handleGameStateUpdate as EventListener,
    );

    return () => {
      gameContainer.removeEventListener(
        "gameStateUpdate",
        handleGameStateUpdate as EventListener,
      );
    };
  }, [gameContainer]);

  const handleFactoryClick = (factoryIndex: number, color: TileColor) => {
    console.log(`Factory ${factoryIndex} clicked, color: ${color}`);
    setSelectedFactory(factoryIndex);
    setSelectedCenterGroup(null);
    setSelectedColor(color);

    // Dispatch event for game logic
    gameContainer.dispatchEvent(
      new CustomEvent("factorySelected", {
        detail: { factoryIndex, color },
      }),
    );
  };

  const handleCenterClick = (groupIndex: number, color: TileColor) => {
    console.log(`Center group ${groupIndex} clicked, color: ${color}`);
    setSelectedCenterGroup(groupIndex);
    setSelectedFactory(null);
    setSelectedColor(color);

    // Dispatch event for game logic
    gameContainer.dispatchEvent(
      new CustomEvent("centerSelected", {
        detail: { groupIndex, color },
      }),
    );
  };

  const handlePatternLineClick = (playerIndex: number, lineIndex: number) => {
    console.log(`Pattern line ${lineIndex} clicked for player ${playerIndex}`);
    if (selectedColor) {
      // Dispatch event for game logic
      gameContainer.dispatchEvent(
        new CustomEvent("patternLineSelected", {
          detail: { playerIndex, lineIndex, color: selectedColor },
        }),
      );

      // Clear selection after successful move
      setSelectedFactory(null);
      setSelectedCenterGroup(null);
      setSelectedColor(null);
    }
  };

  const handleFloorClick = (playerIndex: number) => {
    console.log(`Floor clicked for player ${playerIndex}`);
    if (selectedColor) {
      // Dispatch event for game logic
      gameContainer.dispatchEvent(
        new CustomEvent("floorSelected", {
          detail: { playerIndex, color: selectedColor },
        }),
      );

      // Clear selection after successful move
      setSelectedFactory(null);
      setSelectedCenterGroup(null);
      setSelectedColor(null);
    }
  };

  if (!gameState) {
    return null;
  }

  return (
    <div className="game-container">
      <div className="simple-round-info">Round {gameState.round}</div>

      <div className="game-board">
        <div className="game-board__factory-area">
          <div className="factory-circle">
            <div className="factory-circle__center">
              <CenterArea
                centerTiles={gameState.centerTiles}
                selectedGroup={selectedCenterGroup}
                selectedColor={selectedColor}
                onCenterClick={handleCenterClick}
              />
            </div>

            <div className="factory-circle__factories">
              {gameState.factories.map((factory, index) => (
                <div
                  key={index}
                  className={`factory-position factory-position--${index}`}
                >
                  <div
                    className={`factory ${factory.isEmpty ? "factory--empty" : ""} ${selectedFactory === index ? "factory--selected" : ""}`}
                    onClick={() =>
                      factory.tiles.length > 0 &&
                      handleFactoryClick(index, factory.tiles[0].color)
                    }
                  >
                    <div className="factory__tiles">
                      {factory.tiles.map((tile, tileIndex) => (
                        <div
                          key={`${tile.id}-${tileIndex}`}
                          className={`tile tile--${tile.color} factory__tile ${selectedFactory === index &&
                              selectedColor === tile.color
                              ? "tile--selected"
                              : ""
                            }`}
                          style={{
                            backgroundImage: `url("${getTileImagePath(tile.color)}")`,
                            backgroundSize: "contain",
                            backgroundRepeat: "no-repeat",
                            backgroundPosition: "center",
                            position: "absolute",
                            ...getTilePosition(index, tileIndex),
                          }}
                          onClick={(e) => {
                            e?.stopPropagation();
                            handleFactoryClick(index, tile.color);
                          }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="game-board__players">
          <PlayerBoards
            players={gameState.players}
            currentPlayerIndex={gameState.currentPlayerIndex}
            selectedColor={selectedColor}
            onPatternLineClick={handlePatternLineClick}
            onFloorClick={handleFloorClick}
            aiEnabled={aiEnabled}
          />
        </div>
      </div>
    </div>
  );
}
