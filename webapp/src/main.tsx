import "./styles/main.scss";
import { render } from "preact";
import { useState, useEffect } from "preact/hooks";
import { AISettings } from "./components/AISettings";
import { WebAppGameState } from "./GameState";
import { GameReplay } from "./components/GameReplay";

// Import GameRenderer dynamically to avoid conflicts
let GameRenderer: any = null;

function App() {
  const [aiEnabled, setAiEnabled] = useState(true);
  const [gameState, setGameState] = useState<WebAppGameState | null>(null);
  const [gameRenderer, setGameRenderer] = useState<any>(null);
  const [rendererLoaded, setRendererLoaded] = useState(false);
  const [replayData, setReplayData] = useState<any>(null);
  const [view, setView] = useState<"game" | "replay">("game");

  // Load GameRenderer dynamically
  useEffect(() => {
    import("./GameRenderer")
      .then((module) => {
        GameRenderer = module.GameRenderer;
        setRendererLoaded(true);
        console.log("✅ GameRenderer loaded");
      })
      .catch((error) => {
        console.error("❌ Failed to load GameRenderer:", error);
      });
  }, []);

  // Initialize game state
  useEffect(() => {
    if (view === "game") {
      console.log("Initializing game state...");
      try {
        const initialGameState = new WebAppGameState(2);
        initialGameState.newGame();
        setGameState(initialGameState);
        console.log("✅ Game state initialized successfully");
      } catch (error) {
        console.error("❌ Error initializing game state:", error);
      }
    }
  }, [view]);

  // Create renderer when both GameRenderer class and game state are ready
  useEffect(() => {
    if (
      view === "game" &&
      rendererLoaded &&
      gameState &&
      GameRenderer &&
      !gameRenderer
    ) {
      const gameContainer = document.getElementById("game-area");
      if (gameContainer) {
        console.log("Creating GameRenderer...");
        try {
          // Clear the container first
          gameContainer.innerHTML = "";
          const renderer = new GameRenderer(gameContainer, gameState);
          renderer.setAIEnabled(aiEnabled);
          setGameRenderer(renderer);
          console.log("✅ GameRenderer created successfully");
        } catch (error) {
          console.error("❌ Error creating GameRenderer:", error);
        }
      }
    }
  }, [view, rendererLoaded, gameState, gameRenderer]);

  // Update AI enabled state in GameRenderer
  useEffect(() => {
    if (gameRenderer) {
      gameRenderer.setAIEnabled(aiEnabled);
    }
  }, [aiEnabled, gameRenderer]);

  const handleNewGame = () => {
    console.log("Creating new game...");
    try {
      const newGameState = new WebAppGameState(2);
      newGameState.newGame();
      setGameState(newGameState);

      if (gameRenderer) {
        gameRenderer.updateGameState(newGameState);
      }
      console.log("✅ New game created successfully");
    } catch (error) {
      console.error("❌ Error creating new game:", error);
    }
  };

  const handleToggleAI = () => {
    setAiEnabled(!aiEnabled);
  };

  const handleFileUpload = (event: Event) => {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target?.result as string);
          setReplayData(data);
          setView("replay");
        } catch (error) {
          console.error("❌ Error parsing replay file:", error);
          alert("Invalid replay file format");
        }
      };
      reader.readAsText(file);
    }
  };

  const handleBackToGame = () => {
    setView("game");
    setReplayData(null);
  };

  return (
    <div className="azul-app">
      <div className="azul-app__sidebar">
        {view === "game" ? (
          <AISettings
            aiEnabled={aiEnabled}
            onToggleAI={handleToggleAI}
            onNewGame={handleNewGame}
            round={gameState?.round || 1}
            onSwitchToReplay={() => {
              const input = document.createElement("input");
              input.type = "file";
              input.accept = ".json";
              input.onchange = (e) => {
                const file = (e.target as HTMLInputElement).files?.[0];
                if (file) {
                  const reader = new FileReader();
                  reader.onload = (e) => {
                    try {
                      const data = JSON.parse(e.target?.result as string);
                      setReplayData(data);
                      setView("replay");
                    } catch (error) {
                      console.error("Error loading replay:", error);
                      alert("Error loading replay file");
                    }
                  };
                  reader.readAsText(file);
                }
              };
              input.click();
            }}
          />
        ) : (
          <div className="replay-controls">
            <button onClick={handleBackToGame}>Back to Game</button>
            <input
              type="file"
              accept=".json"
              onChange={handleFileUpload}
              style={{ display: "none" }}
              id="replay-file-input"
            />
            <button
              onClick={() =>
                document.getElementById("replay-file-input")?.click()
              }
            >
              Load Replay
            </button>
          </div>
        )}
      </div>

      <div className="azul-app__game" id="gameContainer">
        {view === "game" ? (
          <div
            id="game-area"
            style={{
              width: "100%",
              height: "100%",
              minHeight: "600px",
            }}
          >
            {!rendererLoaded && (
              <div style={{ padding: "20px" }}>Loading game renderer...</div>
            )}
            {rendererLoaded && !gameState && (
              <div style={{ padding: "20px" }}>Loading game state...</div>
            )}
            {rendererLoaded && gameState && !gameRenderer && (
              <div style={{ padding: "20px" }}>Creating game...</div>
            )}
          </div>
        ) : (
          replayData && <GameReplay replayData={replayData} />
        )}
      </div>

      <style jsx>{`
        .replay-controls {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          padding: 1rem;
        }

        .replay-controls button {
          padding: 0.5rem 1rem;
          border: 1px solid #ccc;
          border-radius: 4px;
          background: #fff;
          cursor: pointer;
        }

        .replay-controls button:hover {
          background: #f0f0f0;
        }
      `}</style>
    </div>
  );
}

// Initialize the app when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  console.log("Starting Azul app...");
  const container = document.getElementById("app");
  if (container) {
    render(<App />, container);
    console.log("Azul app rendered successfully");
  }
});
