import { useState, useEffect } from "preact/hooks";
import { route } from "preact-router";
import { AISettings } from "./AISettings";
import { WebAppGameState } from "../GameState";
import { useContext } from "preact/hooks";
import { ReplayDataContext } from "../main";
import { ActionLog } from "./ActionLog";
import { Sidebar } from "./Sidebar";

// Import GameRenderer dynamically to avoid conflicts
let GameRenderer: any = null;

interface GameViewProps {
  aiEnabled: boolean;
  onToggleAI: () => void;
}

export function GameView({ aiEnabled, onToggleAI }: GameViewProps) {
  const [gameState, setGameState] = useState<WebAppGameState | null>(null);
  const [gameRenderer, setGameRenderer] = useState<any>(null);
  const [rendererLoaded, setRendererLoaded] = useState(false);
  const { setReplayData } = useContext(ReplayDataContext);

  // Load GameRenderer dynamically
  useEffect(() => {
    import("../GameRenderer")
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
    console.log("Initializing game state...");
    try {
      const initialGameState = new WebAppGameState(2);
      initialGameState.newGame();
      setGameState(initialGameState);
      console.log("✅ Game state initialized successfully");
    } catch (error) {
      console.error("❌ Error initializing game state:", error);
    }
  }, []);

  // Create renderer when both GameRenderer class and game state are ready
  useEffect(() => {
    if (rendererLoaded && gameState && GameRenderer && !gameRenderer) {
      const gameContainer = document.getElementById("game-area");
      if (gameContainer) {
        console.log("Creating GameRenderer...");
        try {
          // Clear the container first
          gameContainer.innerHTML = "";
          const renderer = new GameRenderer(gameContainer, gameState, aiEnabled);
          setGameRenderer(renderer);
          console.log("✅ GameRenderer created successfully");
        } catch (error) {
          console.error("❌ Error creating GameRenderer:", error);
        }
      }
    }
  }, [rendererLoaded, gameState, gameRenderer, aiEnabled]);

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

  const handleSwitchToReplay = () => {
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
            route("/replay");
          } catch (error) {
            console.error("Error loading replay:", error);
            alert("Error loading replay file");
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  return (
    <>
      <div className="azul-app__sidebar">
        <AISettings
          aiEnabled={aiEnabled}
          onToggleAI={onToggleAI}
          onNewGame={handleNewGame}
          round={gameState?.round || 1}
          onSwitchToReplay={handleSwitchToReplay}
        />
        {gameState && (
          <Sidebar title="Action Log" subtitle="Game History">
            <ActionLog gameState={gameState} />
          </Sidebar>
        )}
      </div>

      <div className="azul-app__game" id="gameContainer">
        <div id="game-area" className="game-area">
          {!rendererLoaded && (
            <div className="game-area__loading">Loading game renderer...</div>
          )}
          {rendererLoaded && !gameState && (
            <div className="game-area__loading">Loading game state...</div>
          )}
          {rendererLoaded && gameState && !gameRenderer && (
            <div className="game-area__loading">Creating game...</div>
          )}
        </div>
      </div>
    </>
  );
}
