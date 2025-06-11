import { useEffect } from 'preact/hooks';
import { signal } from '@preact/signals';
import AnalyzeButton from './AnalyzeButton';
import MoveSuggestion from './MoveSuggestion';
import GameState from './GameState';
import PlayerBoards from './PlayerBoards';
import Settings from './Settings';
import ErrorDisplay from './ErrorDisplay';
import { GameStateData, AnalysisResponse } from '../types';

// Global signals
export const gameState = signal<GameStateData | null>(null);
export const analysisResult = signal<AnalysisResponse | null>(null);
export const isAnalyzing = signal<boolean>(false);
export const error = signal<string>('');
export const difficulty = signal<number>(4);

export default function App() {
  useEffect(() => {
    // Auto-extract game state on load
    const timer = setTimeout(() => {
      extractAndDisplayGameState();
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // Listen for game state updates
    const messageListener = (message: { action: string }) => {
      if (message.action === 'gameStateUpdated') {
        extractAndDisplayGameState();
      }
    };

    chrome.runtime.onMessage.addListener(messageListener);
    return () => chrome.runtime.onMessage.removeListener(messageListener);
  }, []);

  const extractAndDisplayGameState = async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (!tab.id) {
        throw new Error('No active tab found');
      }

      const response = await chrome.tabs.sendMessage(tab.id, { action: 'getGameState' });

      if (!response || !response.gameState) {
        error.value = 'Could not extract game state from the page';
        return;
      }

      console.log('Received gameState:', JSON.parse(JSON.stringify(response.gameState)));
      gameState.value = response.gameState;
      error.value = '';
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'An unknown error occurred';
      gameState.value = null;
    }
  };

  const runAIAnalysis = async () => {
    if (!gameState.value) {
      error.value = 'No game state available. Please wait for game state to be extracted.';
      return;
    }

    try {
      isAnalyzing.value = true;
      error.value = '';

      const timeLimit = getDifficultyTimeLimit(difficulty.value);

      chrome.runtime.sendMessage(
        {
          action: 'analyzePosition',
          gameState: gameState.value,
          timeLimit,
        },
        (response: AnalysisResponse) => {
          isAnalyzing.value = false;
          analysisResult.value = response;

          if (response.error) {
            error.value = response.error;
          }
        }
      );
    } catch (err) {
      isAnalyzing.value = false;
      error.value = err instanceof Error ? err.message : 'An unknown error occurred';
    }
  };

  const getDifficultyTimeLimit = (level: number): number => {
    const settings: { [key: number]: number } = {
      1: 500, // Easy
      2: 1000, // Medium
      3: 2000, // Hard
      4: 5000, // Expert
    };
    return settings[level] || 1000;
  };

  return (
    <div className="p-3 sm:p-4 lg:p-6 min-w-[300px] w-full bg-gray-50 min-h-screen">
      {/* Always full width at top */}
      <div className="mb-3 sm:mb-4 lg:mb-5">
        <AnalyzeButton onAnalyze={runAIAnalysis} />
      </div>

      {/* Responsive grid layout */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3 sm:gap-4 lg:gap-5">
        {/* Column 1 */}
        <div className="flex flex-col gap-3 sm:gap-4 lg:gap-5">
          <MoveSuggestion />
          <div className="xl:hidden">
            <GameState />
          </div>
        </div>

        {/* Column 2 */}
        <div className="flex flex-col gap-3 sm:gap-4 lg:gap-5">
          <PlayerBoards />
          <div className="xl:hidden">
            <Settings />
          </div>
        </div>

        {/* Column 3 - Only visible on extra wide panels */}
        <div className="hidden xl:flex flex-col gap-3 sm:gap-4 lg:gap-5">
          <GameState />
          <Settings />
        </div>
      </div>

      {/* Error display spans full width at bottom */}
      <div className="mt-3 sm:mt-4 lg:mt-5">
        <ErrorDisplay />
      </div>
    </div>
  );
}
