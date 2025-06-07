import './styles/main.scss';
import { render } from 'preact';
import { h } from 'preact';
import { Game } from './components/Game';
import { WebAppGameState } from './GameState.js';
import { GameRenderer } from './GameRenderer.js';

class AzulApp {
    private gameState: WebAppGameState;
    private renderer: GameRenderer;
    private gameContainer: HTMLElement;

    constructor() {
        console.log('AzulApp constructor starting');
        this.gameContainer = this.setupUI();
        console.log('Game container created:', this.gameContainer);
        this.gameState = new WebAppGameState(2);
        console.log('Game state created');
        this.gameState.newGame();
        console.log('New game started');
        this.renderer = new GameRenderer(this.gameContainer, this.gameState);
        console.log('GameRenderer created');
        this.startGameLoop();
        console.log('Game loop started');
    }

    private setupUI(): HTMLElement {
        // Create main container
        const container = document.createElement('div');
        container.className = 'azul-app-container';
        container.style.cssText = `
      min-height: 100vh;
      font-family: 'Roboto', sans-serif;
      background: #f8f9fa;
    `;

        // Create game container for Preact components
        const gameContainer = document.createElement('div');
        gameContainer.id = 'gameContainer';
        gameContainer.style.cssText = `
      width: 100%;
      height: 100%;
    `;

        container.appendChild(gameContainer);
        document.body.appendChild(container);

        // Render Preact components
        render(h(Game, { gameContainer }), gameContainer);

        return gameContainer;
    }

    private startGameLoop(): void {
        const loop = () => {
            this.update();
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }

    private update(): void {
        // Trigger re-render with current game state
        this.renderer.render();
    }
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new AzulApp();
});
