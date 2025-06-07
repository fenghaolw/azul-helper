import { WebAppGameState } from './GameState.js';
import { GameRenderer } from './GameRenderer.js';
import { ApiAI } from './ApiAI.js';
import { GamePhase } from './types.js';

class AzulApp {
  private gameState: WebAppGameState;
  private renderer: GameRenderer;
  private ai: ApiAI | null = null;
  private aiEnabled: boolean = true;
  private gameContainer: HTMLElement;
  private isAIThinking: boolean = false;

  // UI Elements
  private newGameBtn!: HTMLButtonElement;
  private aiToggleBtn!: HTMLButtonElement;
  private gameInfo!: HTMLDivElement;
  private aiStats!: HTMLDivElement;
  private scoringExplanation!: HTMLDivElement;
  private lastRoundScoring!: HTMLDivElement;

  constructor() {
    this.gameContainer = this.setupUI();
    this.gameState = new WebAppGameState(2);
    this.gameState.newGame();
    this.renderer = new GameRenderer(this.gameContainer, this.gameState);

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

  private setupUI(): HTMLElement {
    // Create main container
    const container = document.createElement('div');
    container.style.cssText = `
      display: flex;
      height: 100vh;
      font-family: 'Roboto', sans-serif;
      background: #f5f5f5;
    `;

    // Create game container (replaces canvas)
    const gameContainer = document.createElement('div');
    gameContainer.style.cssText = `
      flex: 1;
      background: white;
      border-right: 1px solid #e0e0e0;
      overflow: auto;
    `;

    // Create sidebar with scrolling
    const sidebar = document.createElement('div');
    sidebar.style.cssText = `
      width: 380px;
      background: white;
      box-shadow: -2px 0 8px rgba(0,0,0,0.1);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      max-height: 100vh;
    `;

    const colors = {
      primary: '#1976d2',
      surface: '#ffffff',
      onSurface: '#1c1b1f',
      purple: '#7b1fa2',
      green: '#388e3c',
      orange: '#f57c00',
      red: '#d32f2f'
    };

    // Header
    const header = document.createElement('div');
    header.style.cssText = `
      padding: 24px 20px;
      background: ${colors.primary};
      color: white;
      flex-shrink: 0;
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
      flex-shrink: 0;
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
      flex-shrink: 0;
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

    // Last Round Scoring section
    const lastRoundSection = document.createElement('div');
    lastRoundSection.style.cssText = `
      padding: 20px;
      border-bottom: 1px solid rgba(0,0,0,0.12);
      flex-shrink: 0;
    `;

    const lastRoundTitle = document.createElement('h2');
    lastRoundTitle.textContent = 'Last Round Scoring';
    lastRoundTitle.style.cssText = `
      margin: 0 0 16px 0;
      font-size: 20px;
      font-weight: 500;
      color: ${colors.onSurface};
    `;

    this.lastRoundScoring = document.createElement('div');
    this.lastRoundScoring.style.cssText = `
      background: rgba(56, 142, 60, 0.04);
      border-radius: 4px;
      padding: 16px;
      border-left: 4px solid ${colors.green};
      font-size: 14px;
      line-height: 1.5;
      color: ${colors.onSurface};
    `;

    this.lastRoundScoring.innerHTML = '<div style="color: #999; font-style: italic;">No scoring yet this game.</div>';

    lastRoundSection.appendChild(lastRoundTitle);
    lastRoundSection.appendChild(this.lastRoundScoring);

    // Scoring explanation section
    const scoringSection = document.createElement('div');
    scoringSection.style.cssText = `
      padding: 20px;
      border-bottom: 1px solid rgba(0,0,0,0.12);
      flex-shrink: 0;
    `;

    const scoringTitle = document.createElement('h2');
    scoringTitle.textContent = 'Scoring Rules';
    scoringTitle.style.cssText = `
      margin: 0 0 16px 0;
      font-size: 20px;
      font-weight: 500;
      color: ${colors.onSurface};
    `;

    this.scoringExplanation = document.createElement('div');
    this.scoringExplanation.style.cssText = `
      background: rgba(245, 124, 0, 0.04);
      border-radius: 4px;
      padding: 16px;
      border-left: 4px solid ${colors.orange};
      font-size: 14px;
      line-height: 1.5;
      color: ${colors.onSurface};
    `;

    this.updateScoringExplanation();

    scoringSection.appendChild(scoringTitle);
    scoringSection.appendChild(this.scoringExplanation);

    // AI stats section
    const statsSection = document.createElement('div');
    statsSection.style.cssText = `
      padding: 20px;
      border-bottom: 1px solid rgba(0,0,0,0.12);
      flex-shrink: 0;
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
    sidebar.appendChild(lastRoundSection);
    sidebar.appendChild(scoringSection);
    sidebar.appendChild(statsSection);

    container.appendChild(gameContainer);
    container.appendChild(sidebar);
    document.body.appendChild(container);

    return gameContainer;
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

    // Check if we just finished a round (round scoring occurred)
    if (this.gameState.phase === GamePhase.TileSelection && this.gameState.round > 1) {
      // Try to capture scoring details from the game state
      // This is a simplified version - in a real implementation, we'd capture this during the actual scoring
      this.updateLastRoundScoring();
    }

    this.renderer.render();

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

      // Show final bonus scoring
      html += '<div style="margin-top: 15px;"><strong>Final Bonuses:</strong></div>';
      for (let i = 0; i < this.gameState.playerBoards.length; i++) {
        const board = this.gameState.playerBoards[i];
        const finalScoreResult = board.getFinalScoreCalculation();
        const playerName = i === 0 ? 'Human' : 'AI';

        if (finalScoreResult.bonus > 0) {
          html += `
            <div style="margin-left: 10px; margin-bottom: 8px;">
              <div style="font-weight: bold;">${playerName} Final Bonuses:</div>
              <div style="margin-left: 15px; font-size: 12px;">
                ${finalScoreResult.details.completedRows > 0 ? `• Rows: ${finalScoreResult.details.completedRows} × 2 = ${finalScoreResult.details.rowBonus} pts<br>` : ''}
                ${finalScoreResult.details.completedColumns > 0 ? `• Columns: ${finalScoreResult.details.completedColumns} × 7 = ${finalScoreResult.details.columnBonus} pts<br>` : ''}
                ${finalScoreResult.details.completedColors > 0 ? `• Colors: ${finalScoreResult.details.completedColors} × 10 = ${finalScoreResult.details.colorBonus} pts<br>` : ''}
                <strong>Total Bonus: +${finalScoreResult.bonus} points</strong>
              </div>
            </div>
          `;
        } else {
          html += `
            <div style="margin-left: 10px; color: #666;">
              ${playerName}: No bonuses earned
            </div>
          `;
        }
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

  private updateScoringExplanation(): void {
    const html = `
      <div style="margin-bottom: 12px;">
        <h3 style="margin: 0 0 8px 0; color: #f57c00; font-size: 16px;">📊 How Scoring Works</h3>
        <button id="toggleScoringDetails" style="background: none; border: 1px solid #f57c00; color: #f57c00; padding: 4px 8px; font-size: 11px; border-radius: 3px; cursor: pointer;">
          Show Examples
        </button>
      </div>

      <div style="margin-bottom: 10px;">
        <h4 style="margin: 0 0 5px 0; font-size: 14px; color: #333;">🎯 Round Scoring (Each Turn)</h4>
        <div style="font-size: 12px; margin-left: 10px; line-height: 1.4;">
          When you complete a pattern line and move a tile to the wall:
          <br>• <strong>Base:</strong> 1 point for the tile
          <br>• <strong>Horizontal line:</strong> +1 point for each connected tile in the same row
          <br>• <strong>Vertical line:</strong> +1 point for each connected tile in the same column

          <div id="scoringExamples" style="display: none; margin-top: 8px; padding: 6px; background: rgba(25, 118, 210, 0.1); border-radius: 3px; font-size: 11px;">
            <strong>Examples:</strong>
            <br>• Isolated tile: 1 point
            <br>• 3-tile horizontal line: 3 points
            <br>• 2-tile vertical line: 2 points
            <br>• Corner connection (3H + 2V): 5 points
            <br>• Cross pattern (3H + 3V): 5 points (tile counted once)
          </div>
        </div>
      </div>

      <div style="margin-bottom: 10px;">
        <h4 style="margin: 0 0 5px 0; font-size: 14px; color: #333;">💰 Floor Penalties</h4>
        <div style="font-size: 12px; margin-left: 10px; line-height: 1.4;">
          Floor tiles penalty: -1, -1, -2, -2, -2, -3, -3 points
          <br><span style="color: #d32f2f;">Avoid filling your floor line!</span>

          <div id="floorExamples" style="display: none; margin-top: 6px; padding: 6px; background: rgba(211, 47, 47, 0.1); border-radius: 3px; font-size: 11px; color: #d32f2f;">
            <strong>Floor Examples:</strong>
            <br>• 1 tile: -1 point
            <br>• 3 tiles: -4 points (-1, -1, -2)
            <br>• 7 tiles: -14 points (maximum penalty)
          </div>
        </div>
      </div>

      <div style="margin-bottom: 10px;">
        <h4 style="margin: 0 0 5px 0; font-size: 14px; color: #333;">🏆 End-Game Bonuses</h4>
        <div style="font-size: 12px; margin-left: 10px; line-height: 1.4;">
          <div style="margin-bottom: 4px;"><strong>Complete Rows:</strong> +2 points each</div>
          <div style="margin-bottom: 4px;"><strong>Complete Columns:</strong> +7 points each</div>
          <div style="margin-bottom: 4px;"><strong>Complete Colors:</strong> +10 points each</div>
          <div style="color: #388e3c; font-style: italic;">Focus on completing rows and colors for big bonuses!</div>

          <div id="bonusExamples" style="display: none; margin-top: 6px; padding: 6px; background: rgba(56, 142, 60, 0.1); border-radius: 3px; font-size: 11px; color: #388e3c;">
            <strong>Bonus Strategy:</strong>
            <br>• Top 3 rows are easiest to complete
            <br>• Columns give highest individual bonus
            <br>• Color sets give highest overall bonus
            <br>• Maximum possible bonus: 59 points!
          </div>
        </div>
      </div>

      <div style="background: rgba(25, 118, 210, 0.1); padding: 8px; border-radius: 4px; margin-top: 10px;">
        <div style="font-size: 12px; color: #1976d2; font-weight: bold;">💡 Pro Tip</div>
        <div style="font-size: 11px; color: #1976d2; margin-top: 2px;">
          Plan for adjacency! Placing tiles next to existing ones multiplies your points.
        </div>
      </div>
    `;

    this.scoringExplanation.innerHTML = html;

    // Add event listener for the toggle button
    const toggleButton = document.getElementById('toggleScoringDetails');
    if (toggleButton) {
      toggleButton.addEventListener('click', () => {
        const examples = ['scoringExamples', 'floorExamples', 'bonusExamples'];
        const currentlyVisible = document.getElementById('scoringExamples')?.style.display !== 'none';

        examples.forEach(id => {
          const element = document.getElementById(id);
          if (element) {
            element.style.display = currentlyVisible ? 'none' : 'block';
          }
        });

        toggleButton.textContent = currentlyVisible ? 'Show Examples' : 'Hide Examples';
      });
    }
  }

  private updateLastRoundScoring(scoringDetails?: { playerIndex: number; details: any }[]): void {
    if (!scoringDetails || scoringDetails.length === 0) {
      // Show general information about what happened last round if available
      if (this.gameState.round > 1) {
        let html = `<div style="margin-bottom: 10px;"><strong>Round ${this.gameState.round - 1} Completed</strong></div>`;

        // Show current scores as an indication of what happened
        const result = this.gameState.getResult();
        html += '<div style="font-size: 12px;">';
        result.scores.forEach((score, index) => {
          const playerName = index === 0 ? 'Human' : 'AI';
          html += `
            <div style="margin-bottom: 4px;">
              ${playerName}: ${score} points
            </div>
          `;
        });
        html += '</div>';

        html += `
          <div style="margin-top: 8px; padding: 6px; background: rgba(25, 118, 210, 0.1); border-radius: 3px; font-size: 11px; color: #1976d2;">
            💡 Each player moved completed pattern lines to their wall and applied floor penalties.
          </div>
        `;

        this.lastRoundScoring.innerHTML = html;
      } else {
        this.lastRoundScoring.innerHTML = '<div style="color: #999; font-style: italic;">No scoring yet this game.</div>';
      }
      return;
    }

    let html = `<div style="margin-bottom: 10px;"><strong>Round ${this.gameState.round - 1} Scoring:</strong></div>`;

    for (const playerScoring of scoringDetails) {
      const playerName = playerScoring.playerIndex === 0 ? 'Human' : 'AI';
      const details = playerScoring.details;

      html += `
        <div style="margin-bottom: 12px; padding: 8px; background: rgba(56, 142, 60, 0.1); border-radius: 4px;">
          <div style="font-weight: bold; margin-bottom: 6px;">${playerName}</div>
      `;

      if (details.tilesPlaced && details.tilesPlaced.length > 0) {
        html += '<div style="font-size: 12px; margin-bottom: 4px;"><strong>Tiles Placed:</strong></div>';
        for (const tilePlacement of details.tilesPlaced) {
          const tileEmoji = this.getTileEmoji(tilePlacement.tile);
          html += `
            <div style="font-size: 11px; margin-left: 10px; margin-bottom: 2px;">
              ${tileEmoji} Row ${tilePlacement.row + 1}: ${tilePlacement.score} points
              ${tilePlacement.adjacentTiles.horizontal > 1 || tilePlacement.adjacentTiles.vertical > 1 ?
              ` (${tilePlacement.adjacentTiles.horizontal}H + ${tilePlacement.adjacentTiles.vertical}V)` : ''}
            </div>
          `;
        }
      }

      if (details.totalFloorPenalty < 0) {
        html += `
          <div style="font-size: 12px; color: #d32f2f; margin-top: 4px;">
            Floor Penalty: ${details.totalFloorPenalty} points
          </div>
        `;
      }

      const netScore = details.totalTileScore + details.totalFloorPenalty;
      html += `
        <div style="font-size: 12px; font-weight: bold; margin-top: 4px; color: ${netScore >= 0 ? '#388e3c' : '#d32f2f'};">
          Round Total: ${netScore >= 0 ? '+' : ''}${netScore} points
        </div>
      `;

      html += '</div>';
    }

    this.lastRoundScoring.innerHTML = html;
  }

  private getTileEmoji(tile: string): string {
    switch (tile.toLowerCase()) {
      case 'red': return '🔴';
      case 'blue': return '🔵';
      case 'yellow': return '🟡';
      case 'black': return '⚫';
      case 'white': return '⚪';
      default: return '🟦';
    }
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
