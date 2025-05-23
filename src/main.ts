import { GameState } from './GameState.js';
import { GameRenderer } from './GameRenderer.js';
import { AzulAI } from './AI.js';

class AzulApp {
  private gameState: GameState;
  private renderer: GameRenderer;
  private ai: AzulAI | null = null;
  private canvas: HTMLCanvasElement;
  private isAIThinking: boolean = false;

  // UI Elements
  private newGameBtn!: HTMLButtonElement;
  private aiToggleBtn!: HTMLButtonElement;
  private aiDifficultySelect!: HTMLSelectElement;
  private gameInfo!: HTMLDivElement;
  private aiStats!: HTMLDivElement;

  constructor() {
    this.setupUI();
    this.gameState = new GameState(2);
    this.canvas = document.getElementById('gameCanvas') as HTMLCanvasElement;
    this.renderer = new GameRenderer(this.canvas, this.gameState);
    
    // Enable AI by default at Expert difficulty
    this.enableAIByDefault();
    
    this.startGameLoop();
  }

  private enableAIByDefault(): void {
    // Enable AI with Expert difficulty (5000ms)
    const thinkingTime = parseInt(this.aiDifficultySelect.value);
    this.ai = new AzulAI(1, thinkingTime);
    this.aiToggleBtn.textContent = 'Disable AI Opponent';
    this.aiToggleBtn.style.background = '#d32f2f'; // Material Design error color
  }

  private setupUI(): void {
    // Add Material Design fonts
    const materialFonts = document.createElement('link');
    materialFonts.href = 'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Material+Symbols+Outlined&display=swap';
    materialFonts.rel = 'stylesheet';
    document.head.appendChild(materialFonts);

    // Material Design color palette
    const colors = {
      primary: '#1976d2',
      primaryDark: '#1565c0',
      primaryLight: '#42a5f5',
      secondary: '#0d47a1',
      surface: '#ffffff',
      background: '#f5f5f5',
      onSurface: '#212121',
      onSurfaceVariant: '#757575',
      error: '#d32f2f',
      success: '#388e3c',
      warning: '#f57c00',
      purple: '#7b1fa2'
    };

    // Create main container with Material Design background
    const appContainer = document.createElement('div');
    appContainer.style.cssText = `
      font-family: 'Roboto', 'Arial', sans-serif;
      background: ${colors.background};
      min-height: 100vh;
      margin: 0;
      padding: 24px;
      box-sizing: border-box;
    `;



    // Create main content container
    const contentContainer = document.createElement('div');
    contentContainer.style.cssText = `
      display: flex;
      gap: 24px;
      align-items: flex-start;
      flex-wrap: wrap;
      justify-content: center;
    `;

    // Create Material Design card for canvas
    const gameCard = document.createElement('div');
    gameCard.style.cssText = `
      background: ${colors.surface};
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
      padding: 16px;
      transition: box-shadow 0.3s ease;
    `;

    const canvas = document.createElement('canvas');
    canvas.id = 'gameCanvas';
    canvas.style.cssText = `
      border-radius: 4px;
      display: block;
    `;

    gameCard.appendChild(canvas);

    // Create Material Design card for controls
    const controlCard = document.createElement('div');
    controlCard.style.cssText = `
      background: ${colors.surface};
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
      min-width: 320px;
      max-width: 400px;
      overflow: hidden;
    `;

    // Game controls section
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
      letter-spacing: 0.15px;
    `;

    // Material Design buttons
    this.newGameBtn = this.createMaterialButton('New Game', colors.success, 'contained');
    this.newGameBtn.style.marginBottom = '12px';
    this.newGameBtn.addEventListener('click', this.newGame.bind(this));

    this.aiToggleBtn = this.createMaterialButton('Enable AI Opponent', colors.primary, 'contained');
    this.aiToggleBtn.style.marginBottom = '16px';
    this.aiToggleBtn.addEventListener('click', this.toggleAI.bind(this));

    // Material Design select field
    const difficultyFieldContainer = document.createElement('div');
    difficultyFieldContainer.style.cssText = `
      margin-bottom: 16px;
      position: relative;
    `;

    const difficultyLabel = document.createElement('label');
    difficultyLabel.textContent = 'AI Difficulty';
    difficultyLabel.style.cssText = `
      display: block;
      margin-bottom: 8px;
      color: ${colors.onSurfaceVariant};
      font-size: 14px;
      font-weight: 500;
      letter-spacing: 0.25px;
    `;

    this.aiDifficultySelect = document.createElement('select');
    this.aiDifficultySelect.style.cssText = `
      width: 100%;
      padding: 14px 16px;
      border: 1px solid rgba(0,0,0,0.23);
      border-radius: 4px;
      font-size: 16px;
      font-family: 'Roboto', sans-serif;
      background: ${colors.surface};
      color: ${colors.onSurface};
      transition: border-color 0.2s ease;
      outline: none;
    `;
    
    // Add focus styles for select
    this.aiDifficultySelect.addEventListener('focus', () => {
      this.aiDifficultySelect.style.borderColor = colors.primary;
      this.aiDifficultySelect.style.borderWidth = '2px';
      this.aiDifficultySelect.style.padding = '13px 15px';
    });
    
    this.aiDifficultySelect.addEventListener('blur', () => {
      this.aiDifficultySelect.style.borderColor = 'rgba(0,0,0,0.23)';
      this.aiDifficultySelect.style.borderWidth = '1px';
      this.aiDifficultySelect.style.padding = '14px 16px';
    });
    
    const difficulties = [
      { value: '500', text: 'Easy (0.5s)' },
      { value: '1000', text: 'Medium (1s)' },
      { value: '2000', text: 'Hard (2s)' },
      { value: '5000', text: 'Expert (5s)' }
    ];

    difficulties.forEach(diff => {
      const option = document.createElement('option');
      option.value = diff.value;
      option.textContent = diff.text;
      if (diff.value === '5000') option.selected = true;
      this.aiDifficultySelect.appendChild(option);
    });

    difficultyFieldContainer.appendChild(difficultyLabel);
    difficultyFieldContainer.appendChild(this.aiDifficultySelect);

    // Debug button with outlined style
    const debugBtn = this.createMaterialButton('Show Debug Info', colors.purple, 'outlined');
    debugBtn.addEventListener('click', () => {
      console.log('=== Current Game State Debug Info ===');
      console.log('Game State:', this.gameState);
      console.log('Player Boards:');
      this.gameState.playerBoards.forEach((board, index) => {
        console.log(`Player ${index + 1}:`, {
          score: board.score,
          wall: board.wall,
          lines: board.lines,
          floor: board.floor
        });
      });
      console.log('Available moves:', this.gameState.availableMoves);
      alert('Debug information logged to console (F12 to view)');
    });

    controlsSection.appendChild(controlsTitle);
    controlsSection.appendChild(this.newGameBtn);
    controlsSection.appendChild(this.aiToggleBtn);
    controlsSection.appendChild(difficultyFieldContainer);
    controlsSection.appendChild(debugBtn);

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
      letter-spacing: 0.15px;
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

    // AI stats section (collapsible)
    const statsSection = document.createElement('div');
    statsSection.style.cssText = `
      border-bottom: 1px solid rgba(0,0,0,0.12);
    `;
    
    const statsHeader = document.createElement('div');
    statsHeader.style.cssText = `
      padding: 20px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: background-color 0.2s ease;
    `;
    
    const statsTitle = document.createElement('h2');
    statsTitle.textContent = 'AI Statistics';
    statsTitle.style.cssText = `
      margin: 0;
      font-size: 20px;
      font-weight: 500;
      color: ${colors.onSurface};
      letter-spacing: 0.15px;
    `;

    const statsExpandIcon = document.createElement('span');
    statsExpandIcon.textContent = '▼';
    statsExpandIcon.style.cssText = `
      font-size: 14px;
      color: ${colors.onSurfaceVariant};
      transition: transform 0.2s ease;
      user-select: none;
    `;

    statsHeader.appendChild(statsTitle);
    statsHeader.appendChild(statsExpandIcon);

    const statsContent = document.createElement('div');
    statsContent.style.cssText = `
      padding: 0 20px 20px 20px;
      transition: all 0.3s ease;
      overflow: hidden;
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

    statsContent.appendChild(this.aiStats);

    // Add hover effect and click handler
    let statsExpanded = false;
    
    statsHeader.addEventListener('mouseenter', () => {
      statsHeader.style.backgroundColor = 'rgba(0,0,0,0.04)';
    });
    
    statsHeader.addEventListener('mouseleave', () => {
      statsHeader.style.backgroundColor = 'transparent';
    });
    
    statsHeader.addEventListener('click', () => {
      statsExpanded = !statsExpanded;
      if (statsExpanded) {
        statsContent.style.maxHeight = statsContent.scrollHeight + 'px';
        statsContent.style.opacity = '1';
        statsExpandIcon.style.transform = 'rotate(180deg)';
        statsExpandIcon.textContent = '▲';
      } else {
        statsContent.style.maxHeight = '0';
        statsContent.style.opacity = '0';
        statsExpandIcon.style.transform = 'rotate(0deg)';
        statsExpandIcon.textContent = '▼';
      }
    });

    // Start collapsed
    statsContent.style.maxHeight = '0';
    statsContent.style.opacity = '0';

    statsSection.appendChild(statsHeader);
    statsSection.appendChild(statsContent);

    // Instructions section (collapsible)
    const instructionsSection = document.createElement('div');
    
    const instructionsHeader = document.createElement('div');
    instructionsHeader.style.cssText = `
      padding: 20px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: background-color 0.2s ease;
    `;
    
    const instructionsTitle = document.createElement('h2');
    instructionsTitle.textContent = 'How to Play';
    instructionsTitle.style.cssText = `
      margin: 0;
      font-size: 20px;
      font-weight: 500;
      color: ${colors.onSurface};
      letter-spacing: 0.15px;
    `;

    const instructionsExpandIcon = document.createElement('span');
    instructionsExpandIcon.textContent = '▼';
    instructionsExpandIcon.style.cssText = `
      font-size: 14px;
      color: ${colors.onSurfaceVariant};
      transition: transform 0.2s ease;
      user-select: none;
    `;

    instructionsHeader.appendChild(instructionsTitle);
    instructionsHeader.appendChild(instructionsExpandIcon);

    const instructionsContent = document.createElement('div');
    instructionsContent.style.cssText = `
      padding: 0 20px 20px 20px;
      transition: all 0.3s ease;
      overflow: hidden;
    `;
    
    const instructionsList = document.createElement('ul');
    instructionsList.style.cssText = `
      margin: 0;
      padding-left: 20px;
      color: ${colors.onSurfaceVariant};
      font-size: 14px;
      line-height: 1.5;
    `;
    
    const instructionsText = [
      'Click on a factory or the center to select tiles',
      'Click on a pattern line (1-5) to place tiles',
      'Complete pattern lines to score points',
      'Avoid placing tiles on the floor (penalty points)',
      'Game ends when a player completes a horizontal row'
    ];

    instructionsText.forEach(text => {
      const li = document.createElement('li');
      li.textContent = text;
      li.style.marginBottom = '8px';
      instructionsList.appendChild(li);
    });

    instructionsContent.appendChild(instructionsList);

    // Add hover effect and click handler
    let instructionsExpanded = false;
    
    instructionsHeader.addEventListener('mouseenter', () => {
      instructionsHeader.style.backgroundColor = 'rgba(0,0,0,0.04)';
    });
    
    instructionsHeader.addEventListener('mouseleave', () => {
      instructionsHeader.style.backgroundColor = 'transparent';
    });
    
    instructionsHeader.addEventListener('click', () => {
      instructionsExpanded = !instructionsExpanded;
      if (instructionsExpanded) {
        instructionsContent.style.maxHeight = instructionsContent.scrollHeight + 'px';
        instructionsContent.style.opacity = '1';
        instructionsExpandIcon.style.transform = 'rotate(180deg)';
        instructionsExpandIcon.textContent = '▲';
      } else {
        instructionsContent.style.maxHeight = '0';
        instructionsContent.style.opacity = '0';
        instructionsExpandIcon.style.transform = 'rotate(0deg)';
        instructionsExpandIcon.textContent = '▼';
      }
    });

    // Start collapsed
    instructionsContent.style.maxHeight = '0';
    instructionsContent.style.opacity = '0';

    instructionsSection.appendChild(instructionsHeader);
    instructionsSection.appendChild(instructionsContent);

    // Assemble control card
    controlCard.appendChild(controlsSection);
    controlCard.appendChild(infoSection);
    controlCard.appendChild(statsSection);
    controlCard.appendChild(instructionsSection);

    // Assemble content
    contentContainer.appendChild(gameCard);
    contentContainer.appendChild(controlCard);

    // Assemble app
    appContainer.appendChild(contentContainer);

    document.body.appendChild(appContainer);

    // Set body styles with Material Design background
    document.body.style.cssText = `
      margin: 0;
      padding: 0;
      font-family: 'Roboto', 'Arial', sans-serif;
      background: ${colors.background};
    `;
  }

  // Helper method to create Material Design buttons
  private createMaterialButton(text: string, color: string, variant: 'contained' | 'outlined' = 'contained'): HTMLButtonElement {
    const button = document.createElement('button');
    button.textContent = text;
    
    const baseStyles = `
      font-family: 'Roboto', sans-serif;
      font-size: 14px;
      font-weight: 500;
      letter-spacing: 0.75px;
      text-transform: uppercase;
      border-radius: 4px;
      padding: 10px 24px;
      min-width: 64px;
      cursor: pointer;
      transition: all 0.2s ease;
      border: none;
      outline: none;
      width: 100%;
      position: relative;
      overflow: hidden;
    `;

    if (variant === 'contained') {
      button.style.cssText = baseStyles + `
        background: ${color};
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12), 0 2px 4px rgba(0,0,0,0.24);
      `;
      
      button.addEventListener('mouseenter', () => {
        button.style.boxShadow = '0 4px 8px rgba(0,0,0,0.16), 0 4px 8px rgba(0,0,0,0.32)';
        button.style.transform = 'translateY(-1px)';
      });
      
      button.addEventListener('mouseleave', () => {
        button.style.boxShadow = '0 2px 4px rgba(0,0,0,0.12), 0 2px 4px rgba(0,0,0,0.24)';
        button.style.transform = 'translateY(0)';
      });
    } else {
      button.style.cssText = baseStyles + `
        background: transparent;
        color: ${color};
        border: 1px solid ${color};
        box-shadow: none;
      `;
      
      button.addEventListener('mouseenter', () => {
        button.style.background = `${color}08`;
      });
      
      button.addEventListener('mouseleave', () => {
        button.style.background = 'transparent';
      });
    }

    // Add ripple effect
    button.addEventListener('click', (e) => {
      const ripple = document.createElement('span');
      const rect = button.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      const x = e.clientX - rect.left - size / 2;
      const y = e.clientY - rect.top - size / 2;
      
      ripple.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        left: ${x}px;
        top: ${y}px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        transform: scale(0);
        animation: ripple 0.6s linear;
        pointer-events: none;
      `;
      
      // Add animation keyframes if not already added
      if (!document.head.querySelector('#ripple-animation')) {
        const style = document.createElement('style');
        style.id = 'ripple-animation';
        style.textContent = `
          @keyframes ripple {
            to {
              transform: scale(4);
              opacity: 0;
            }
          }
        `;
        document.head.appendChild(style);
      }
      
      button.appendChild(ripple);
      setTimeout(() => ripple.remove(), 600);
    });

    return button;
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
    if (this.ai && this.gameState.currentPlayer === 1 && !this.gameState.gameOver && !this.isAIThinking) {
      this.handleAITurn();
    }
  }

  private async handleAITurn(): Promise<void> {
    this.isAIThinking = true;
    
    // Add a small delay for better UX
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
      const result = this.ai!.getBestMove(this.gameState);
      await new Promise(resolve => setTimeout(resolve, 200)); // Show AI thinking
      
      this.renderer.playMove(result.move);
    } catch (error) {
      console.error('AI error:', error);
      // Fallback to simple move
      const simpleMove = this.ai!.getSimpleMove(this.gameState);
      this.renderer.playMove(simpleMove);
    }
    
    this.isAIThinking = false;
  }

  private updateGameInfo(): void {
    const result = this.gameState.getResult();
    
    let html = `
      <div style="margin-bottom: 10px;">
        <strong>Round:</strong> ${this.gameState.round}
      </div>
      <div style="margin-bottom: 10px;">
        <strong>First Player:</strong> Player ${this.gameState.firstPlayerIndex + 1}
      </div>
    `;

    if (this.gameState.gameOver) {
      if (result.winner !== -1) {
        const winnerName = result.winner === 0 ? 'Human' : (this.ai ? 'AI' : 'Player 2');
        html += `
          <div style="margin-bottom: 10px; color: #388e3c; font-weight: bold;">
            🏆 Winner: ${winnerName} (Player ${result.winner + 1})
          </div>
        `;
      } else {
        html += `
          <div style="margin-bottom: 10px; color: #f57c00; font-weight: bold;">
            🤝 Game ended in a tie!
          </div>
        `;
      }
    } else {
      const currentPlayerName = this.gameState.currentPlayer === 0 ? 'Human' : 
                                (this.ai ? 'AI' : 'Player 2');
      html += `
        <div style="margin-bottom: 10px;">
          <strong>Current Turn:</strong> ${currentPlayerName} (Player ${this.gameState.currentPlayer + 1})
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
      const playerName = index === 0 ? 'Human' : (this.ai ? 'AI' : `Player ${index + 1}`);
      const isCurrentPlayer = index === this.gameState.currentPlayer && !this.gameState.gameOver;
      html += `
        <div style="margin-left: 10px; ${isCurrentPlayer ? 'font-weight: bold; color: #388e3c;' : ''}">
          ${playerName}: ${score} points
        </div>
      `;
    });

    // Show last round scoring details if available
    const lastRoundDetails = (this.gameState as any).lastRoundScoringDetails;
    if (lastRoundDetails && lastRoundDetails.length > 0) {
      html += `
        <div style="margin-top: 15px;">
          <strong>Last Round Scoring:</strong>
        </div>
      `;
      
      for (const playerResult of lastRoundDetails) {
        const playerName = playerResult.player === 0 ? 'Human' : (this.ai ? 'AI' : `Player ${playerResult.player + 1}`);
        html += `
          <div style="margin-left: 10px; margin-top: 8px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px;">
            <strong>${playerName}:</strong><br>
        `;
        
        if (playerResult.details.tilesPlaced.length > 0) {
          html += `<span style="color: #388e3c;">+${playerResult.details.totalTileScore} tiles</span><br>`;
        }
        
        if (playerResult.details.totalFloorPenalty < 0) {
          html += `<span style="color: #d32f2f;">${playerResult.details.totalFloorPenalty} floor</span><br>`;
        }
        
        const changeColor = playerResult.scoreGained >= 0 ? '#388e3c' : '#d32f2f';
        html += `<span style="color: ${changeColor}; font-weight: bold;">${playerResult.scoreGained >= 0 ? '+' : ''}${playerResult.scoreGained} total</span>`;
        html += '</div>';
      }
    }

    this.gameInfo.innerHTML = html;
  }

  private updateAIStats(): void {
    if (!this.ai) {
      this.aiStats.innerHTML = '<div style="color: #757575; font-style: italic;">AI is disabled</div>';
      return;
    }

    const stats = this.ai.getStats();
    const difficultyText = this.aiDifficultySelect.options[this.aiDifficultySelect.selectedIndex].text;

    const html = `
      <div style="margin-bottom: 10px;">
        <strong>Difficulty:</strong> ${difficultyText}
      </div>
      <div style="margin-bottom: 10px;">
        <strong>Last Search:</strong> ${stats.nodesEvaluated.toLocaleString()} nodes
      </div>
      <div style="margin-bottom: 10px;">
        <strong>Algorithm:</strong> Minimax + Alpha-Beta Pruning
      </div>
      <div style="margin-bottom: 10px;">
        <strong>Features:</strong> Iterative Deepening, Move Ordering
      </div>
    `;

    this.aiStats.innerHTML = html;
  }

  private newGame(): void {
    this.gameState = new GameState(2);
    this.renderer.updateGameState(this.gameState);
    
    if (this.ai) {
      this.ai.resetStats();
    }
  }

  private toggleAI(): void {
    if (this.ai) {
      this.ai = null;
      this.aiToggleBtn.textContent = 'Enable AI Opponent';
      this.aiToggleBtn.style.background = '#1976d2'; // Material Design primary
    } else {
      const thinkingTime = parseInt(this.aiDifficultySelect.value);
      this.ai = new AzulAI(1, thinkingTime);
      this.aiToggleBtn.textContent = 'Disable AI Opponent';
      this.aiToggleBtn.style.background = '#d32f2f'; // Material Design error
    }
    
    this.newGame(); // Start fresh game when toggling AI
  }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
  new AzulApp();
}); 