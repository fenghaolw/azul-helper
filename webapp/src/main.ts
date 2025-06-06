import { WebAppGameState } from './GameState.js';
import { GameRenderer } from './GameRenderer.js';
import { ApiAI } from './ApiAI.js';

class AzulApp {
  private gameState: WebAppGameState;
  private renderer: GameRenderer;
  private ai: ApiAI | null = null;
  private aiEnabled: boolean = true;
  private canvas: HTMLCanvasElement;
  private isAIThinking: boolean = false;

  // UI Elements
  private newGameBtn!: HTMLButtonElement;
  private aiToggleBtn!: HTMLButtonElement;
  private gameInfo!: HTMLDivElement;
  private aiStats!: HTMLDivElement;

  constructor() {
    this.canvas = this.setupUI();
    this.gameState = new WebAppGameState(2);
    this.gameState.newGame();
    this.renderer = new GameRenderer(this.canvas, this.gameState);

    // Initialize AI by default
    this.initializeAI();
    this.startGameLoop();
  }

  private initializeAI(): void {
    if (this.aiEnabled) {
      // AI difficulty is controlled by server startup flags, not frontend
      this.ai = new ApiAI(1);
    } else {
      this.ai = null;
    }
  }

  private setupUI(): HTMLCanvasElement {
    // Create main container
    const container = document.createElement('div');
    container.style.cssText = `
      display: flex;
      height: 100vh;
      font-family: 'Roboto', sans-serif;
      background: #f5f5f5;
    `;

    // Create game canvas
    const canvas = document.createElement('canvas');
    canvas.style.cssText = `
      flex: 1;
      background: white;
      border-right: 1px solid #e0e0e0;
    `;

    // Create sidebar
    const sidebar = document.createElement('div');
    sidebar.style.cssText = `
      width: 350px;
      background: white;
      box-shadow: -2px 0 8px rgba(0,0,0,0.1);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    `;

    const colors = {
      primary: '#1976d2',
      surface: '#ffffff',
      onSurface: '#1c1b1f',
      purple: '#7b1fa2'
    };

    // Header
    const header = document.createElement('div');
    header.style.cssText = `
      padding: 24px 20px;
      background: ${colors.primary};
      color: white;
    `;

    const title = document.createElement('h1');
    title.textContent = 'Azul Game';
    title.style.cssText = `
      margin: 0;
      font-size: 28px;
      font-weight: 400;
    `;

    const subtitle = document.createElement('p');
    subtitle.textContent = 'Strategic tile-laying board game';
    subtitle.style.cssText = `
      margin: 8px 0 0 0;
      opacity: 0.9;
      font-size: 14px;
    `;

    header.appendChild(title);
    header.appendChild(subtitle);

    // Controls section
    const controlsSection = document.createElement('div');
    controlsSection.style.cssText = `
      padding: 20px;
      border-bottom: 1px solid rgba(0,0,0,0.12);
    `;

    const controlsTitle = document.createElement('h2');
    controlsTitle.textContent = 'Game Controls';
    controlsTitle.style.cssText = `
      margin: 0 0 16px 0;
      font-size: 20px;
      font-weight: 500;
      color: ${colors.onSurface};
    `;

    // New Game button
    this.newGameBtn = this.createMaterialButton('New Game', colors.primary);
    this.newGameBtn.addEventListener('click', () => this.newGame());

    // AI Toggle button
    this.aiToggleBtn = this.createMaterialButton(
      this.aiEnabled ? 'Disable AI' : 'Enable AI',
      this.aiEnabled ? '#d32f2f' : '#388e3c'
    );
    this.aiToggleBtn.addEventListener('click', () => this.toggleAI());

    controlsSection.appendChild(controlsTitle);
    controlsSection.appendChild(this.newGameBtn);
    controlsSection.appendChild(this.aiToggleBtn);

    // Game info section
    const infoSection = document.createElement('div');
    infoSection.style.cssText = `
      padding: 20px;
      border-bottom: 1px solid rgba(0,0,0,0.12);
    `;

    const infoTitle = document.createElement('h2');
    infoTitle.textContent = 'Game Information';
    infoTitle.style.cssText = `
      margin: 0 0 16px 0;
      font-size: 20px;
      font-weight: 500;
      color: ${colors.onSurface};
    `;

    this.gameInfo = document.createElement('div');
    this.gameInfo.style.cssText = `
      background: rgba(25, 118, 210, 0.04);
      border-radius: 4px;
      padding: 16px;
      border-left: 4px solid ${colors.primary};
      font-size: 14px;
      line-height: 1.5;
      color: ${colors.onSurface};
    `;

    infoSection.appendChild(infoTitle);
    infoSection.appendChild(this.gameInfo);

    // AI stats section
    const statsSection = document.createElement('div');
    statsSection.style.cssText = `
      padding: 20px;
      border-bottom: 1px solid rgba(0,0,0,0.12);
    `;

    const statsTitle = document.createElement('h2');
    statsTitle.textContent = 'AI Status';
    statsTitle.style.cssText = `
      margin: 0 0 16px 0;
      font-size: 20px;
      font-weight: 500;
      color: ${colors.onSurface};
    `;

    this.aiStats = document.createElement('div');
    this.aiStats.style.cssText = `
      background: rgba(123, 31, 162, 0.04);
      border-radius: 4px;
      padding: 16px;
      border-left: 4px solid ${colors.purple};
      font-size: 14px;
      line-height: 1.5;
      color: ${colors.onSurface};
    `;

    // Debug: Set initial content to verify the element is working
    this.aiStats.innerHTML = '<div style="color: #999;">Loading AI statistics...</div>';
    console.log('AI stats element created:', this.aiStats);

    statsSection.appendChild(statsTitle);
    statsSection.appendChild(this.aiStats);

    // Assemble UI
    sidebar.appendChild(header);
    sidebar.appendChild(controlsSection);
    sidebar.appendChild(infoSection);
    sidebar.appendChild(statsSection);

    container.appendChild(canvas);
    container.appendChild(sidebar);

    document.body.appendChild(container);
    return canvas;
  }

  private createMaterialButton(text: string, color: string): HTMLButtonElement {
    const button = document.createElement('button');
    button.textContent = text;
    button.style.cssText = `
      width: 100%;
      padding: 12px 24px;
      margin: 8px 0;
      background: ${color};
      color: white;
      border: none;
      border-radius: 4px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    `;

    button.addEventListener('mouseenter', () => {
      button.style.filter = 'brightness(1.1)';
      button.style.transform = 'translateY(-1px)';
    });

    button.addEventListener('mouseleave', () => {
      button.style.filter = 'brightness(1)';
      button.style.transform = 'translateY(0)';
    });

    return button;
  }

  private toggleAI(): void {
    this.aiEnabled = !this.aiEnabled;
    this.aiToggleBtn.textContent = this.aiEnabled ? 'Disable AI' : 'Enable AI';
    this.aiToggleBtn.style.background = this.aiEnabled ? '#d32f2f' : '#388e3c';
    this.initializeAI();
    this.newGame(); // Start fresh game
  }

  private startGameLoop(): void {
    const loop = () => {
      this.update();
      this.renderer.render();
      requestAnimationFrame(loop);
    };
    loop();
  }

  private update(): void {
    this.updateGameInfo();
    this.updateAIStats();

    // Handle AI turn
    if (this.ai && this.aiEnabled && this.gameState.currentPlayer === 1 && !this.gameState.gameOver && !this.isAIThinking) {
      this.handleAITurn();
    }
  }

  private async handleAITurn(): Promise<void> {
    this.isAIThinking = true;

    try {
      const result = await this.ai!.getBestMove(this.gameState);

      // Add a small delay to make AI moves visible
      setTimeout(() => {
        this.renderer.playMove(result.move);
        this.isAIThinking = false;
      }, 500);

    } catch (error) {
      console.error('AI Error:', error);
      this.isAIThinking = false;
    }
  }

  private updateGameInfo(): void {
    const result = this.gameState.getResult();

    let html = `
      <div style="margin-bottom: 10px;">
        <strong>Round:</strong> ${this.gameState.round}
      </div>
    `;

    if (this.gameState.gameOver) {
      if (result.winner !== -1) {
        const winnerName = result.winner === 0 ? 'Human' : 'AI';
        html += `
          <div style="margin-bottom: 10px; color: #388e3c; font-weight: bold;">
            🎉 ${winnerName} wins!
          </div>
        `;
      } else {
        html += `
          <div style="margin-bottom: 10px; color: #f57c00; font-weight: bold;">
            🤝 It's a tie!
          </div>
        `;
      }
    } else {
      const currentPlayerName = this.gameState.currentPlayer === 0 ? 'Human' : 'AI';
      html += `
        <div style="margin-bottom: 10px;">
          <strong>Current Turn:</strong> ${currentPlayerName}
        </div>
        <div style="margin-bottom: 10px;">
          <strong>Available Moves:</strong> ${this.gameState.availableMoves.length}
        </div>
      `;

      if (this.isAIThinking) {
        html += `
          <div style="margin-bottom: 10px; color: #7b1fa2; font-style: italic;">
            🤖 AI is thinking...
          </div>
        `;
      }
    }

    // Show scores
    html += '<div style="margin-top: 15px;"><strong>Scores:</strong></div>';
    result.scores.forEach((score, index) => {
      const playerName = index === 0 ? 'Human' : 'AI';
      const isCurrentPlayer = index === this.gameState.currentPlayer && !this.gameState.gameOver;
      html += `
        <div style="margin-left: 10px; ${isCurrentPlayer ? 'font-weight: bold; color: #388e3c;' : ''}">
          ${playerName}: ${score} points
        </div>
      `;
    });

    this.gameInfo.innerHTML = html;
  }

  private updateAIStats(): void {
    // Debug logging
    console.log('Updating AI stats - aiEnabled:', this.aiEnabled, 'ai object:', !!this.ai);

    if (!this.aiEnabled || !this.ai) {
      this.aiStats.innerHTML = '<div style="color: #757575; font-style: italic;">AI is disabled - Human vs Human mode</div>';
      return;
    }

    const stats = this.ai.getStats();
    console.log('AI stats:', stats);
    console.log('AI server connected:', this.ai.isServerConnected());

    let html = `
      <div style="margin-bottom: 10px;">
        <strong>AI Agent:</strong> 🤖 ${this.ai.getAgentDisplayName()}
      </div>
    `;

    console.log('Agent display name:', this.ai.getAgentDisplayName());

    // Show connection status first
    if (this.ai.isServerConnected()) {
      html += `
        <div style="margin-bottom: 10px; color: #388e3c;">
          <strong>Status:</strong> 🚀 Connected
        </div>
      `;

      const currentApiUrl = (this.ai as any).apiBaseUrl;
      if (currentApiUrl) {
        const port = currentApiUrl.split(':').pop();
        html += `
          <div style="margin-bottom: 10px; font-size: 12px; color: #666;">
            <strong>Server:</strong> localhost:${port}
          </div>
        `;
      }
    } else {
      html += `
        <div style="margin-bottom: 10px; color: #d32f2f;">
          <strong>Status:</strong> ❌ Disconnected
        </div>
        <div style="margin-bottom: 10px; font-size: 12px; color: #666;">
          💡 Try: python start.py --server-only
        </div>
      `;
    }

    // Performance stats section
    if (stats.totalMoves && stats.totalMoves > 0) {
      html += `
        <div style="margin: 15px 0 10px 0; font-weight: bold; color: #1976d2; border-bottom: 1px solid #e0e0e0; padding-bottom: 5px;">
          📊 Performance Statistics
        </div>
      `;

      // Last search details
      html += `
        <div style="margin-bottom: 8px;">
          <strong>Last Search:</strong> ${stats.nodesEvaluated?.toLocaleString() || 'N/A'} nodes
        </div>
      `;

      if (stats.searchTime) {
        html += `
          <div style="margin-bottom: 8px;">
            <strong>Last Time:</strong> ${(stats.searchTime * 1000).toFixed(1)}ms
          </div>
        `;
      }

      // Average performance
      if (stats.averageSearchTime) {
        html += `
          <div style="margin-bottom: 8px;">
            <strong>Avg Time:</strong> ${(stats.averageSearchTime * 1000).toFixed(1)}ms
          </div>
        `;
      }

      // Total moves made
      html += `
        <div style="margin-bottom: 8px;">
          <strong>Moves Made:</strong> ${stats.totalMoves}
        </div>
      `;

      // Last move timing
      if (stats.lastMoveTime) {
        const timeSince = Date.now() - stats.lastMoveTime.getTime();
        const secondsAgo = Math.floor(timeSince / 1000);
        let timeText = '';
        if (secondsAgo < 60) {
          timeText = `${secondsAgo}s ago`;
        } else {
          const minutesAgo = Math.floor(secondsAgo / 60);
          timeText = `${minutesAgo}m ago`;
        }
        html += `
          <div style="margin-bottom: 8px; font-size: 12px; color: #666;">
            <strong>Last Move:</strong> ${timeText}
          </div>
        `;
      }

      // Performance indicator
      if (stats.averageSearchTime) {
        let performanceIcon = '🐌';
        let performanceText = 'Slow';
        let performanceColor = '#ff9800';

        if (stats.averageSearchTime < 0.1) {
          performanceIcon = '⚡';
          performanceText = 'Lightning Fast';
          performanceColor = '#4caf50';
        } else if (stats.averageSearchTime < 0.5) {
          performanceIcon = '🚀';
          performanceText = 'Fast';
          performanceColor = '#2196f3';
        } else if (stats.averageSearchTime < 2.0) {
          performanceIcon = '🏃';
          performanceText = 'Normal';
          performanceColor = '#ff9800';
        }

        html += `
          <div style="margin-bottom: 8px; font-size: 12px; color: ${performanceColor};">
            ${performanceIcon} <strong>Speed:</strong> ${performanceText}
          </div>
        `;
      }
    } else if (this.ai.isServerConnected()) {
      html += `
        <div style="margin: 15px 0 10px 0; font-style: italic; color: #666;">
          💭 Waiting for first move to show performance stats...
        </div>
      `;
    }

    // Configuration info
    if (this.ai.isServerConnected()) {
      html += `
        <div style="margin: 15px 0 10px 0; font-weight: bold; color: #7b1fa2; border-bottom: 1px solid #e0e0e0; padding-bottom: 5px;">
          ⚙️ Configuration
        </div>
        <div style="margin-bottom: 8px; font-size: 12px; color: #666;">
          💡 AI difficulty controlled by server configuration
        </div>
        <div style="margin-bottom: 8px; font-size: 12px; color: #666;">
          🔧 Restart server with different flags to change settings
        </div>
      `;
    }

    console.log('Generated AI stats HTML:', html);
    this.aiStats.innerHTML = html;
  }

  private newGame(): void {
    console.log('Starting new game...');
    this.gameState = new WebAppGameState(2);
    this.gameState.newGame();
    this.renderer.updateGameState(this.gameState);

    // Reset AI stats
    if (this.ai) {
      this.ai.resetStats();
    }
  }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
  new AzulApp();
});
