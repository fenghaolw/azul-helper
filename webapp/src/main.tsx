import './styles/main.scss';
import {render} from 'preact';
import {useState, useEffect} from 'preact/hooks';
import {AISettings} from './components/AISettings';
import {WebAppGameState} from './GameState';

// Import GameRenderer dynamically to avoid conflicts
let GameRenderer: any = null;

function App() {
  const [aiEnabled, setAiEnabled] = useState(true);
  const [gameState, setGameState] = useState<WebAppGameState | null>(null);
  const [gameRenderer, setGameRenderer] = useState<any>(null);
  const [rendererLoaded, setRendererLoaded] = useState(false);

  // Load GameRenderer dynamically
  useEffect(() => {
    import('./GameRenderer').then((module) => {
      GameRenderer = module.GameRenderer;
      setRendererLoaded(true);
      console.log('✅ GameRenderer loaded');
    }).catch((error) => {
      console.error('❌ Failed to load GameRenderer:', error);
    });
  }, []);

  // Initialize game state
  useEffect(() => {
    console.log('Initializing game state...');
    try {
      const initialGameState = new WebAppGameState(2);
      initialGameState.newGame();
      setGameState(initialGameState);
      console.log('✅ Game state initialized successfully');
    } catch (error) {
      console.error('❌ Error initializing game state:', error);
    }
  }, []);

  // Create renderer when both GameRenderer class and game state are ready
  useEffect(() => {
    if (rendererLoaded && gameState && GameRenderer && !gameRenderer) {
      const gameContainer = document.getElementById('game-area');
      if (gameContainer) {
        console.log('Creating GameRenderer...');
        try {
          // Clear the container first
          gameContainer.innerHTML = '';
          const renderer = new GameRenderer(gameContainer, gameState);
          setGameRenderer(renderer);
          console.log('✅ GameRenderer created successfully');
        } catch (error) {
          console.error('❌ Error creating GameRenderer:', error);
        }
      }
    }
  }, [rendererLoaded, gameState, gameRenderer]);

  const handleNewGame = () => {
    console.log('Creating new game...');
    try {
      const newGameState = new WebAppGameState(2);
      newGameState.newGame();
      setGameState(newGameState);

      if (gameRenderer) {
        gameRenderer.updateGameState(newGameState);
      }
      console.log('✅ New game created successfully');
    } catch (error) {
      console.error('❌ Error creating new game:', error);
    }
  };

  const handleToggleAI = () => {
    setAiEnabled(!aiEnabled);
  };

  return (
    <div className="azul-app">
      <div className="azul-app__sidebar">
        <AISettings
          aiEnabled={aiEnabled}
          onToggleAI={handleToggleAI}
          onNewGame={handleNewGame}
          round={gameState?.round || 1}
        />
      </div>

      <div className="azul-app__game" id="gameContainer">
        {/* Separate container for GameRenderer - NOT managed by Preact */}
        <div 
          id="game-area" 
          style={{
            width: '100%',
            height: '100%',
            minHeight: '600px'
          }}
        >
          {!rendererLoaded && <div style={{padding: '20px'}}>Loading game renderer...</div>}
          {rendererLoaded && !gameState && <div style={{padding: '20px'}}>Loading game state...</div>}
          {rendererLoaded && gameState && !gameRenderer && <div style={{padding: '20px'}}>Creating game...</div>}
        </div>
      </div>
    </div>
  );
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  console.log('Starting Azul app...');
  const container = document.getElementById('app');
  if (container) {
    render(<App />, container);
    console.log('Azul app rendered successfully');
  }
});
